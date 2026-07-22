"""Fused TILTED K-Planes kernel vs the pure-torch reference: forward + all grads.

The reference is a standalone copy of quantem's
``interpolate_ms_features_tilted`` restricted to one multiscale level
(``core/ml/models/kplanes.py``), so parity here is parity with the consuming
code path.
"""

import pytest
import torch
import torch.nn.functional as F

from quantem.cuda.core.ml import kplanes_tilted_fuse, kplanes_tilted_tv_fuse
from quantem.cuda.core.ml._ops import _channels_last, _restore_plane_layout

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


def test_channels_last_plane_and_gradient_views_avoid_copies():
    plane = torch.rand(6, 4, 9, 11).contiguous(memory_format=torch.channels_last)

    plane_cl = _channels_last(plane)
    assert plane_cl.is_contiguous()
    assert plane_cl.data_ptr() == plane.data_ptr()

    grad_plane_cl = torch.zeros_like(plane_cl)
    grad_plane = _restore_plane_layout(grad_plane_cl, plane)
    assert grad_plane.is_contiguous(memory_format=torch.channels_last)
    assert grad_plane.data_ptr() == grad_plane_cl.data_ptr()


def test_row_major_plane_and_gradient_keep_copy_fallback():
    plane = torch.rand(6, 4, 9, 11)

    plane_cl = _channels_last(plane)
    assert plane_cl.is_contiguous()
    assert plane_cl.data_ptr() != plane.data_ptr()

    grad_plane_cl = torch.zeros_like(plane_cl)
    grad_plane = _restore_plane_layout(grad_plane_cl, plane)
    assert grad_plane.is_contiguous()
    assert grad_plane.data_ptr() != grad_plane_cl.data_ptr()


# ── kplanes_tilted_tv_fuse ────────────────────────────────────────────────

# TV test configs: (B, T, C, H, W)
TV_CONFIGS = [
    (37, 1, 1, 5, 5),
    (257, 3, 5, 7, 9),
    (1024, 4, 8, 50, 50),
    (2000, 4, 32, 33, 33),
]


def reference_tv(
    pts: torch.Tensor,
    rotations: torch.Tensor,
    plane: torch.Tensor,
    h: float,
) -> torch.Tensor:
    """Pure-torch TV reference: evaluate kplanes reference at 4 tap locations.

    Returns [4, B, T*C] — tap outermost to match the kernel output layout.
    """
    taps = []
    for axis in range(4):
        if axis == 0:
            p = pts
        else:
            # x + h*e_{axis-1}
            delta = torch.zeros_like(pts)
            delta[:, axis - 1] = h
            p = pts + delta
        taps.append(reference(p, rotations, plane))
    return torch.stack(taps, dim=0)  # [4, B, T*C]


@requires_cuda
@pytest.mark.parametrize("cfg", TV_CONFIGS)
def test_tv_forward_tap0_matches_base(cfg):
    """tap 0 output must be bitwise-identical to kplanes_tilted_fuse."""
    B, T, C, H, W = cfg
    pts, rotations, plane = _inputs(B, T, C, H, W, seed=42)
    base = kplanes_tilted_fuse(pts, rotations, plane)
    tv_out = kplanes_tilted_tv_fuse(pts, rotations, plane, h=0.01)
    assert tv_out.shape == (4, B, T * C)
    torch.testing.assert_close(tv_out[0], base, rtol=0.0, atol=0.0)


@requires_cuda
@pytest.mark.parametrize("cfg", TV_CONFIGS)
def test_tv_forward_taps_match_reference(cfg):
    """All 4 taps must match the pure-torch reference to fp32 tolerance."""
    B, T, C, H, W = cfg
    pts, rotations, plane = _inputs(B, T, C, H, W, seed=7)
    h = 0.05
    expected = reference_tv(pts, rotations, plane, h=h)
    actual = kplanes_tilted_tv_fuse(pts, rotations, plane, h=h)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=5e-6)


@requires_cuda
@pytest.mark.parametrize("cfg", TV_CONFIGS)
def test_tv_grads_match_fp64_reference(cfg):
    """fp64-anchored gradient test mirroring test_grads_match_fp64_reference.

    Covers pts, rotations, and plane.  The fp64 reference is the pure-torch
    4-tap computation; kernel grad error must be ≤ max(K× torch fp32 error, 1e-6).
    K is 4 for rotations — their grads accumulate over all B points via
    atomics, so summation-order error runs up to ~3.5× torch's tree
    reductions on the large configs — and 2 elsewhere.
    """
    B, T, C, H, W = cfg
    pts, rotations, plane = _inputs(B, T, C, H, W, seed=99)
    h = 0.02
    gen = torch.Generator(device="cuda").manual_seed(7)
    upstream = torch.randn(4, B, T * C, device="cuda", generator=gen)

    def grads(fn, dtype):
        inp = tuple(t.detach().to(dtype).requires_grad_(True) for t in (pts, rotations, plane))

        def tv_ref(*args):
            return reference_tv(*args, h=float(h))

        def tv_kernel(*args):
            return fn(*args, h=float(h))

        impl = tv_ref if fn is reference else tv_kernel
        out = impl(*inp)
        return torch.autograd.grad((out * upstream.to(dtype)).sum(), inp)

    g64 = grads(reference, torch.float64)
    g_torch = grads(reference, torch.float32)
    g_kernel = grads(kplanes_tilted_tv_fuse, torch.float32)

    for name, gt, gk, gd in zip(("pts", "rotations", "plane"), g_torch, g_kernel, g64):
        err_torch = (gt.double() - gd).abs().max().item()
        err_kernel = (gk.double() - gd).abs().max().item()
        k = 4.0 if name == "rotations" else 2.0
        assert err_kernel <= max(k * err_torch, 1e-6), (
            f"tv grad_{name}: kernel err {err_kernel:.3e} vs torch fp32 err {err_torch:.3e}"
        )


@requires_cuda
def test_tv_border_points():
    """Points at ±1 and beyond, including taps crossing the border."""
    T, C, H, W = 2, 4, 11, 11
    gen = torch.Generator(device="cuda").manual_seed(13)
    rotations = torch.stack(
        [torch.eye(3, device="cuda"), torch.eye(3, device="cuda").roll(1, 0)]
    ).float()
    plane = torch.empty((3 * T, C, H, W), device="cuda").uniform_(0.1, 0.5, generator=gen)

    # Include points where taps cross the border even if base is interior
    base = torch.tensor(
        [
            [-1.0, -1.0, -1.0],  # base on corner
            [1.0, 1.0, 1.0],  # base on opposite corner
            [0.98, 0.0, 0.0],  # tap 1 (x+h) crosses border at h=0.05
            [0.0, 0.98, 0.0],  # tap 2 (y+h) crosses border
            [0.0, 0.0, 0.98],  # tap 3 (z+h) crosses border
            [2.0, -3.0, 0.5],  # well outside
        ],
        device="cuda",
    )
    h = 0.05
    pts = base.clone().requires_grad_(True)
    expected = reference_tv(pts, rotations, plane, h=h)
    actual = kplanes_tilted_tv_fuse(pts, rotations, plane, h=h)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-6)

    # Grad should be finite and match reference
    gref = torch.autograd.grad(expected.sum(), pts, retain_graph=True)[0]
    gact = torch.autograd.grad(actual.sum(), pts, retain_graph=True)[0]
    assert torch.isfinite(gact).all(), "tv grad_pts contains non-finite values"
    torch.testing.assert_close(gact, gref, rtol=1e-4, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("h_val", [2.0, 1e-4])
def test_tv_edge_case_h(h_val):
    """h larger than one cell and h tiny — no NaNs, grads finite."""
    B, T, C, H, W = 256, 2, 8, 17, 17
    pts, rotations, plane = _inputs(B, T, C, H, W, seed=55, requires_grad=True)
    out = kplanes_tilted_tv_fuse(pts, rotations, plane, h=h_val)
    assert torch.isfinite(out).all(), f"output has non-finite values for h={h_val}"
    out.sum().backward()
    for name, t in (("pts", pts), ("rotations", rotations), ("plane", plane)):
        assert torch.isfinite(t.grad).all(), f"grad_{name} has non-finite values for h={h_val}"


@requires_cuda
def test_tv_compile_fullgraph():
    """torch.compile(fullgraph=True) must work (fake tensor shapes correct)."""
    pts, rotations, plane = _inputs(512, 2, 4, 17, 17, requires_grad=True)

    @torch.compile(fullgraph=True)
    def f(p, r, g):
        return kplanes_tilted_tv_fuse(p, r, g, h=0.01).square().mean()

    loss = f(pts, rotations, plane)
    loss.backward()
    assert pts.grad is not None and rotations.grad is not None and plane.grad is not None


@requires_cuda
def test_tv_input_validation():
    pts, rotations, plane = _inputs(16, 2, 4, 9, 9)
    with pytest.raises(ValueError, match=r"pts \[B, 3\]"):
        kplanes_tilted_tv_fuse(pts[:, :2], rotations, plane, h=0.01)
    with pytest.raises(ValueError, match=r"rotations \[T, 3, 3\]"):
        kplanes_tilted_tv_fuse(pts, rotations[:, :2, :], plane, h=0.01)
    with pytest.raises(ValueError, match=r"3\*T"):
        kplanes_tilted_tv_fuse(pts, rotations, plane[:3], h=0.01)
    with pytest.raises(TypeError, match="fp32-only"):
        kplanes_tilted_tv_fuse(pts.double(), rotations, plane, h=0.01)


def test_tv_cpu_rejected():
    pts = torch.rand(8, 3)
    rotations = torch.rand(2, 3, 3)
    plane = torch.rand(6, 4, 9, 9)
    with pytest.raises(ValueError, match="CUDA"):
        kplanes_tilted_tv_fuse(pts, rotations, plane, h=0.01)
