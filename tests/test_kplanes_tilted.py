"""Fused TILTED K-Planes kernel vs the pure-torch reference: forward + all grads.

The reference is a standalone copy of quantem's
``interpolate_ms_features_tilted`` restricted to one multiscale level
(``core/ml/models/kplanes.py``), so parity here is parity with the consuming
code path.
"""

import pytest
import torch
import torch.nn.functional as F

from quantem.cuda import kplanes_tilted_fuse

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

# (B, T, C, H, W) — odd sizes, C not a multiple of 4, non-square planes
CONFIGS = [
    (37, 1, 1, 5, 5),
    (257, 3, 5, 7, 9),
    (4096, 4, 8, 200, 200),
    (2000, 8, 32, 50, 33),
]


def reference(pts: torch.Tensor, rotations: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """One level of quantem's interpolate_ms_features_tilted, verbatim semantics."""
    T = rotations.shape[0]
    B = pts.shape[0]
    C = plane.shape[1]
    rotated = torch.einsum("tij,bj->tbi", rotations, pts)
    idx = torch.tensor([[0, 1], [2, 0], [1, 2]], device=pts.device)
    coords = (
        rotated.unsqueeze(1).expand(T, 3, B, 3).gather(-1, idx.view(1, 3, 1, 2).expand(T, 3, B, 2))
    )
    coord_tensor = coords.reshape(3 * T, B, 1, 2)
    sampled = F.grid_sample(
        plane, coord_tensor, align_corners=True, mode="bilinear", padding_mode="border"
    )
    return sampled.squeeze(-1).view(T, 3, C, B).prod(dim=1).permute(2, 0, 1).reshape(B, T * C)


def _inputs(B, T, C, H, W, seed=0, requires_grad=False, pts_scale=1.1):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, lo, hi):
        return torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(
            lo, hi, generator=gen
        )

    # pts_scale > 1 exercises the border clamp on a fraction of the points
    pts = rand((B, 3), -pts_scale, pts_scale)
    rotations = rand((T, 3, 3), -1.0, 1.0)
    plane = rand((3 * T, C, H, W), 0.1, 0.5)
    if requires_grad:
        for t in (pts, rotations, plane):
            t.requires_grad_(True)
    return pts, rotations, plane


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS)
def test_forward_matches_reference(cfg):
    pts, rotations, plane = _inputs(*cfg)
    expected = reference(pts, rotations, plane)
    actual = kplanes_tilted_fuse(pts, rotations, plane)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-6)


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS)
def test_grads_match_fp64_reference(cfg):
    """fp64-anchored: the kernel's deviation from the fp64 reference gradients
    must be no worse than 2x torch's own fp32 deviation. Self-calibrating, so
    it stays meaningful where summation order (atomics vs torch reductions)
    makes direct fp32<->fp32 comparison brittle."""
    pts, rotations, plane = _inputs(*cfg)
    upstream = torch.randn_like(reference(pts, rotations, plane))

    def grads(fn, dtype):
        inp = tuple(t.detach().to(dtype).requires_grad_(True) for t in (pts, rotations, plane))
        out = fn(*inp)
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), inp)

    g64 = grads(reference, torch.float64)
    g_torch = grads(reference, torch.float32)
    g_kernel = grads(kplanes_tilted_fuse, torch.float32)

    for name, gt, gk, gd in zip(("pts", "rotations", "plane"), g_torch, g_kernel, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_kernel = (gk.double() - gd).abs().max().item()
        assert err_kernel <= max(2.0 * err_torch, 1e-6), (
            f"grad_{name}: kernel err {err_kernel:.3e} vs torch fp32 err {err_torch:.3e}"
        )


@requires_cuda
def test_points_exactly_on_border():
    """±1 coords sit exactly on the clip boundary, where torch zeroes the
    coordinate gradient — the case ray z-sampling (linspace(-1, 1)) hits."""
    T, C, H, W = 2, 3, 11, 11
    gen = torch.Generator(device="cuda").manual_seed(7)
    rotations = torch.stack(
        [torch.eye(3, device="cuda"), torch.eye(3, device="cuda").roll(1, 0)]
    ).float()
    plane = torch.empty((3 * T, C, H, W), device="cuda").uniform_(0.1, 0.5, generator=gen)
    base = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [2.0, 0.5, -3.0],
        ],
        device="cuda",
    )
    for tensors in (
        (base.clone().requires_grad_(True), rotations, plane),
        (base.clone().requires_grad_(True), rotations.clone().requires_grad_(True), plane),
    ):
        pts = tensors[0]
        expected = reference(*tensors)
        actual = kplanes_tilted_fuse(*tensors)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        gref = torch.autograd.grad(expected.sum(), pts, retain_graph=True)[0]
        gact = torch.autograd.grad(actual.sum(), pts, retain_graph=True)[0]
        torch.testing.assert_close(gact, gref, rtol=1e-4, atol=1e-6)


@requires_cuda
def test_upstream_grad_scaling():
    pts, rotations, plane = _inputs(64, 2, 4, 9, 9, requires_grad=True)
    kplanes_tilted_fuse(pts, rotations, plane).sum().backward()
    g1 = plane.grad.clone()
    plane.grad = None
    (kplanes_tilted_fuse(pts, rotations, plane) * 3.0).sum().backward()
    torch.testing.assert_close(plane.grad, g1 * 3.0, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_non_contiguous_inputs():
    pts, rotations, plane = _inputs(128, 2, 4, 9, 9)
    pts_nc = pts.t().contiguous().t()
    plane_nc = plane.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    assert not pts_nc.is_contiguous() and not plane_nc.is_contiguous()
    torch.testing.assert_close(
        kplanes_tilted_fuse(pts_nc, rotations, plane_nc),
        reference(pts, rotations, plane),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_cuda
def test_compile_fullgraph():
    pts, rotations, plane = _inputs(512, 2, 4, 17, 17, requires_grad=True)

    @torch.compile(fullgraph=True)
    def f(p, r, g):
        return kplanes_tilted_fuse(p, r, g).square().mean()

    loss = f(pts, rotations, plane)
    loss.backward()
    assert pts.grad is not None and rotations.grad is not None and plane.grad is not None


@requires_cuda
def test_input_validation():
    pts, rotations, plane = _inputs(16, 2, 4, 9, 9)
    with pytest.raises(ValueError, match=r"pts \[B, 3\]"):
        kplanes_tilted_fuse(pts[:, :2], rotations, plane)
    with pytest.raises(ValueError, match=r"rotations \[T, 3, 3\]"):
        kplanes_tilted_fuse(pts, rotations[:, :2, :], plane)
    with pytest.raises(ValueError, match=r"3\*T"):
        kplanes_tilted_fuse(pts, rotations, plane[:3])
    with pytest.raises(TypeError, match="fp32-only"):
        kplanes_tilted_fuse(pts.double(), rotations, plane)


def test_cpu_rejected():
    pts = torch.rand(8, 3)
    rotations = torch.rand(2, 3, 3)
    plane = torch.rand(6, 4, 9, 9)
    with pytest.raises(ValueError, match="CUDA"):
        kplanes_tilted_fuse(pts, rotations, plane)
