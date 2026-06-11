"""Fused NON-TILTED K-Planes kernel vs the pure-torch reference: forward + all grads.

The reference is a standalone copy of quantem's ``interpolate_ms_features``
restricted to one multiscale level (``core/ml/models/kplanes.py``), so parity
here is parity with the consuming code path.
"""

import pytest
import torch
import torch.nn.functional as F

from quantem.cuda.core.ml import kplanes_fuse

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

# (B, C, H, W) — odd sizes, C not a multiple of 4, non-square planes
CONFIGS = [
    (37, 1, 5, 5),
    (257, 5, 7, 9),
    (4096, 8, 200, 200),
    (2000, 33, 50, 33),
]


def reference(pts: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """One level of quantem's interpolate_ms_features, verbatim semantics."""
    C = plane.shape[1]
    mat_mode = [[0, 1], [0, 2], [1, 2]]
    coord_plane = torch.stack(
        [
            pts[:, mat_mode[0]],
            pts[:, mat_mode[1]],
            pts[:, mat_mode[2]],
        ]
    ).view(3, -1, 1, 2)
    feats = F.grid_sample(
        plane, coord_plane, align_corners=True, mode="bilinear", padding_mode="border"
    ).reshape(3, C, -1)
    return (feats[0] * feats[1] * feats[2]).T


def _inputs(B, C, H, W, seed=0, requires_grad=False, pts_scale=1.1):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def rand(shape, lo, hi):
        return torch.empty(shape, device="cuda", dtype=torch.float32).uniform_(
            lo, hi, generator=gen
        )

    # pts_scale > 1 exercises the border clamp on a fraction of the points
    pts = rand((B, 3), -pts_scale, pts_scale)
    plane = rand((3, C, H, W), 0.1, 0.5)
    if requires_grad:
        for t in (pts, plane):
            t.requires_grad_(True)
    return pts, plane


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS)
def test_forward_matches_reference(cfg):
    pts, plane = _inputs(*cfg)
    expected = reference(pts, plane)
    actual = kplanes_fuse(pts, plane)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-6)


@requires_cuda
@pytest.mark.parametrize("cfg", CONFIGS)
def test_grads_match_fp64_reference(cfg):
    """fp64-anchored: the kernel's deviation from the fp64 reference gradients
    must be no worse than 2x torch's own fp32 deviation. Self-calibrating, so
    it stays meaningful where summation order (atomics vs torch reductions)
    makes direct fp32<->fp32 comparison brittle."""
    pts, plane = _inputs(*cfg)
    upstream = torch.randn_like(reference(pts, plane))

    def grads(fn, dtype):
        inp = tuple(t.detach().to(dtype).requires_grad_(True) for t in (pts, plane))
        out = fn(*inp)
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), inp)

    g64 = grads(reference, torch.float64)
    g_torch = grads(reference, torch.float32)
    g_kernel = grads(kplanes_fuse, torch.float32)

    for name, gt, gk, gd in zip(("pts", "plane"), g_torch, g_kernel, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_kernel = (gk.double() - gd).abs().max().item()
        assert err_kernel <= max(2.0 * err_torch, 1e-6), (
            f"grad_{name}: kernel err {err_kernel:.3e} vs torch fp32 err {err_torch:.3e}"
        )


@requires_cuda
def test_points_exactly_on_border():
    """±1 coords sit exactly on the clip boundary, where torch zeroes the
    coordinate gradient — the case ray z-sampling (linspace(-1, 1)) hits."""
    C, H, W = 3, 11, 11
    gen = torch.Generator(device="cuda").manual_seed(7)
    plane = torch.empty((3, C, H, W), device="cuda").uniform_(0.1, 0.5, generator=gen)
    pts = torch.tensor(
        [
            [-1.0, -1.0, -1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0],
            [2.0, 0.5, -3.0],
        ],
        device="cuda",
        requires_grad=True,
    )
    expected = reference(pts, plane)
    actual = kplanes_fuse(pts, plane)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    gref = torch.autograd.grad(expected.sum(), pts, retain_graph=True)[0]
    gact = torch.autograd.grad(actual.sum(), pts, retain_graph=True)[0]
    torch.testing.assert_close(gact, gref, rtol=1e-4, atol=1e-6)


@requires_cuda
def test_upstream_grad_scaling():
    pts, plane = _inputs(64, 4, 9, 9, requires_grad=True)
    kplanes_fuse(pts, plane).sum().backward()
    g1 = plane.grad.clone()
    plane.grad = None
    (kplanes_fuse(pts, plane) * 3.0).sum().backward()
    torch.testing.assert_close(plane.grad, g1 * 3.0, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_non_contiguous_inputs():
    pts, plane = _inputs(128, 4, 9, 9)
    pts_nc = pts.t().contiguous().t()
    plane_nc = plane.permute(0, 1, 3, 2).contiguous().permute(0, 1, 3, 2)
    assert not pts_nc.is_contiguous() and not plane_nc.is_contiguous()
    torch.testing.assert_close(
        kplanes_fuse(pts_nc, plane_nc),
        reference(pts, plane),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_cuda
def test_compile_fullgraph():
    pts, plane = _inputs(512, 4, 17, 17, requires_grad=True)

    @torch.compile(fullgraph=True)
    def f(p, g):
        return kplanes_fuse(p, g).square().mean()

    loss = f(pts, plane)
    loss.backward()
    assert pts.grad is not None and plane.grad is not None


@requires_cuda
def test_input_validation():
    pts, plane = _inputs(16, 4, 9, 9)
    with pytest.raises(ValueError, match=r"pts \[B, 3\]"):
        kplanes_fuse(pts[:, :2], plane)
    with pytest.raises(ValueError, match=r"plane \[3, C, H, W\]"):
        kplanes_fuse(pts, plane[:2])
    with pytest.raises(TypeError, match="fp32-only"):
        kplanes_fuse(pts.double(), plane)


def test_cpu_rejected():
    pts = torch.rand(8, 3)
    plane = torch.rand(3, 4, 9, 9)
    with pytest.raises(ValueError, match="CUDA"):
        kplanes_fuse(pts, plane)
