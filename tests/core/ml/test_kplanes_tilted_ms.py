"""Three-level fused TILTED K-Planes parity, striding, and compile tests."""

import os
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from quantem.cuda.core.ml import kplanes_tilted_fuse, kplanes_tilted_fuse_ms
from quantem.cuda.core.ml._ops import _kplanes_tilted_fuse_ms_bwd

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

GATES = (0.25, 0.75, 1.0)
UNIT_GATES = (1.0, 1.0, 1.0)


def reference(pts: torch.Tensor, rotations: torch.Tensor, plane: torch.Tensor) -> torch.Tensor:
    """Pure-torch single-level reference used by the existing tilted tests."""
    t = rotations.shape[0]
    b = pts.shape[0]
    c = plane.shape[1]
    rotated = torch.einsum("tij,bj->tbi", rotations, pts)
    x, y, z = rotated.unbind(-1)
    coords = torch.stack(
        (
            torch.stack((x, y), dim=-1),
            torch.stack((z, x), dim=-1),
            torch.stack((y, z), dim=-1),
        ),
        dim=1,
    )
    sampled = F.grid_sample(
        plane,
        coords.reshape(3 * t, b, 1, 2),
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )
    return sampled.squeeze(-1).view(t, 3, c, b).prod(dim=1).permute(2, 0, 1).reshape(b, t * c)


def _inputs(requires_grad=False, seed=0, dtype=torch.float32):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    pts = torch.empty((37, 3), device="cuda", dtype=torch.float32).uniform_(
        -1.1, 1.1, generator=generator
    )
    rotations = torch.empty((3, 3, 3), device="cuda", dtype=torch.float32).uniform_(
        -1.0, 1.0, generator=generator
    )
    grids = tuple(
        torch.empty((9, channels, height, width), device="cuda", dtype=torch.float32)
        .uniform_(0.1, 0.5, generator=generator)
        .contiguous(memory_format=torch.channels_last)
        .to(dtype)
        for channels, height, width in ((3, 5, 7), (5, 11, 13), (7, 19, 23))
    )
    if requires_grad:
        pts.requires_grad_(True)
        rotations.requires_grad_(True)
        for grid in grids:
            grid.requires_grad_(True)
    return pts, rotations, grids


def _single_cat(pts, rotations, *grids):
    return torch.cat(
        [kplanes_tilted_fuse(pts, rotations, grid) * gate for grid, gate in zip(grids, GATES)],
        dim=-1,
    )


def _torch_cat(pts, rotations, *grids):
    return torch.cat(
        [reference(pts, rotations, grid) * gate for grid, gate in zip(grids, GATES)], dim=-1
    )


def _multiscale(pts, rotations, *grids):
    return kplanes_tilted_fuse_ms(pts, rotations, *grids, *GATES)


@requires_cuda
@pytest.mark.parametrize("explicit_gates", [True, False], ids=["explicit", "default"])
def test_ms_unit_gates_exactly_match_ungated_levels(explicit_gates):
    pts, rotations, grids = _inputs(seed=2)
    base_inputs = (pts, rotations, *grids)
    single_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in base_inputs)
    ms_inputs = tuple(tensor.detach().clone().requires_grad_(True) for tensor in base_inputs)

    pts_single, rotations_single, *grids_single = single_inputs
    expected = torch.cat(
        [kplanes_tilted_fuse(pts_single, rotations_single, grid) for grid in grids_single],
        dim=-1,
    )
    pts_ms, rotations_ms, *grids_ms = ms_inputs
    if explicit_gates:
        actual = kplanes_tilted_fuse_ms(pts_ms, rotations_ms, *grids_ms, *UNIT_GATES)
    else:
        actual = kplanes_tilted_fuse_ms(pts_ms, rotations_ms, *grids_ms)

    assert torch.equal(actual, expected)

    generator = torch.Generator(device="cuda").manual_seed(23)
    upstream = torch.empty_like(expected).uniform_(-1.0, 1.0, generator=generator)
    expected_grads = torch.autograd.grad(expected, single_inputs, upstream)
    actual_grads = torch.autograd.grad(actual, ms_inputs, upstream)
    # Fused level blocks interleave atomicAdd operations, so gradient summation order differs.
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-7)


@requires_cuda
def test_ms_forward_matches_concatenated_levels():
    pts, rotations, grids = _inputs(seed=3)
    expected = _single_cat(pts, rotations, *grids)
    actual = _multiscale(pts, rotations, *grids)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@requires_cuda
def test_ms_bf16_autocast_output_matches_fp32_output_cast(monkeypatch):
    pts, rotations, grids = _inputs(seed=4)

    monkeypatch.setenv("QUANTEM_KPLANES_MS_BF16_OUT", "0")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        expected_fp32 = _multiscale(pts, rotations, *grids)
    assert expected_fp32.dtype == torch.float32

    monkeypatch.setenv("QUANTEM_KPLANES_MS_BF16_OUT", "1")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual = _multiscale(pts, rotations, *grids)

    assert actual.dtype == torch.bfloat16
    # Both paths round the same fp32 epilogue value once to bf16: the new path
    # does so at the CUDA store, while the reference uses an explicit cast.
    assert torch.equal(actual, expected_fp32.to(torch.bfloat16))


@requires_cuda
def test_ms_non_bf16_autocast_keeps_fp32_output(monkeypatch):
    pts, rotations, grids = _inputs(seed=4)
    monkeypatch.setenv("QUANTEM_KPLANES_MS_BF16_OUT", "1")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        actual = _multiscale(pts, rotations, *grids)

    assert actual.dtype == torch.float32


@requires_cuda
def test_ms_autocast_off_is_byte_identical_with_kill_switch(monkeypatch):
    pts, rotations, grids = _inputs(seed=4)
    monkeypatch.setenv("QUANTEM_KPLANES_MS_BF16_OUT", "1")
    enabled = _multiscale(pts, rotations, *grids)
    monkeypatch.setenv("QUANTEM_KPLANES_MS_BF16_OUT", "0")
    disabled = _multiscale(pts, rotations, *grids)

    assert enabled.dtype == disabled.dtype == torch.float32
    assert torch.equal(enabled.view(torch.uint8), disabled.view(torch.uint8))


@requires_cuda
def test_ms_grads_match_fp64_anchored_single_levels():
    """Account for the atomic reduction's documented run-to-run envelope (ISS-4)."""
    pts, rotations, grids = _inputs(seed=5)
    upstream = torch.randn_like(_single_cat(pts, rotations, *grids))

    def grads(fn, dtype):
        inputs = tuple(
            tensor.detach().to(dtype).requires_grad_(True) for tensor in (pts, rotations, *grids)
        )
        output = fn(*inputs)
        return torch.autograd.grad((output * upstream.to(dtype)).sum(), inputs)

    grad64 = grads(_torch_cat, torch.float64)
    grad_single = grads(_single_cat, torch.float32)
    grad_ms_a = grads(_multiscale, torch.float32)
    grad_ms_b = grads(_multiscale, torch.float32)

    names = ("pts", "rotations", "plane0", "plane1", "plane2")
    for name, g64, gsingle, gms_a, gms_b in zip(names, grad64, grad_single, grad_ms_a, grad_ms_b):
        err_single = (gsingle.double() - g64).abs().max().item()
        err_ms = (gms_a.double() - g64).abs().max().item()
        run_spread = (gms_a - gms_b).abs().max().item()
        assert err_ms <= max(2.0 * err_single, 2.0 * run_spread, 1e-6), (
            f"grad_{name}: ms err {err_ms:.3e}, single err {err_single:.3e}, "
            f"run spread {run_spread:.3e}"
        )


@requires_cuda
def test_ms_backward_reads_strided_upstream_without_packing():
    pts, rotations, grids = _inputs(seed=7)
    widths = [rotations.shape[0] * grid.shape[1] for grid in grids]
    total_width = sum(widths)
    backing = torch.randn((pts.shape[0], total_width + 17), device="cuda")
    upstream = backing[:, 5 : 5 + total_width]
    assert upstream.stride() == (total_width + 17, 1)

    actual = _kplanes_tilted_fuse_ms_bwd(pts, rotations, *grids, upstream, *GATES)

    inputs = tuple(tensor.detach().requires_grad_(True) for tensor in (pts, rotations, *grids))
    expected = torch.autograd.grad((_single_cat(*inputs) * upstream).sum(), inputs)
    for expected_grad, actual_grad in zip(expected, actual):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=3e-6)


@requires_cuda
def test_ms_bf16_gout_matches_fp32_gout_of_same_values():
    pts, rotations, grids = _inputs(seed=8)
    width = rotations.shape[0] * sum(grid.shape[1] for grid in grids)
    generator = torch.Generator(device="cuda").manual_seed(29)
    upstream_bf16 = torch.empty(
        (pts.shape[0], width), device="cuda", dtype=torch.bfloat16
    ).uniform_(-1.0, 1.0, generator=generator)
    upstream_bf16[:, ::5] = 0
    upstream_bf16[:, 1::7] = torch.tensor(5e-4, device="cuda", dtype=torch.bfloat16)

    actual = _kplanes_tilted_fuse_ms_bwd(
        pts, rotations, *grids, upstream_bf16, *GATES
    )
    expected = _kplanes_tilted_fuse_ms_bwd(
        pts, rotations, *grids, upstream_bf16.float(), *GATES
    )

    for actual_grad, expected_grad in zip(actual, expected):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-6, atol=1e-7)


@requires_cuda
def test_ms_v5_bf16_gout_composes_with_threshold_ballot():
    env = os.environ.copy()
    env["QUANTEM_KPLANES_BWD_VARIANT"] = "5"
    env["QUANTEM_KPLANES_BWD_ZERO_TAU"] = "1e-3"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_ms_bf16_gout_matches_fp32_gout_of_same_values",
            "-q",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, result.stdout + result.stderr


@requires_cuda
@pytest.mark.skipif(
    os.environ.get("QUANTEM_KPLANES_BWD_VARIANT") != "5"
    or os.environ.get("QUANTEM_KPLANES_BWD_ZERO_TAU") != "1e-3",
    reason="requires the V5 tau subprocess",
)
def test_ms_bf16_grid_and_gout_autocast_v5_smoke():
    pts, rotations, grids = _inputs(seed=10, dtype=torch.bfloat16)
    grids_fp32 = tuple(grid.float() for grid in grids)
    expected_out = _multiscale(pts, rotations, *grids_fp32)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        actual_out = _multiscale(pts, rotations, *grids)

    assert actual_out.dtype == torch.bfloat16
    torch.testing.assert_close(
        actual_out, expected_out.to(torch.bfloat16), rtol=2e-4, atol=2e-6
    )

    generator = torch.Generator(device="cuda").manual_seed(31)
    upstream_bf16 = torch.empty_like(actual_out).uniform_(
        -1.0, 1.0, generator=generator
    )
    upstream_bf16[:, ::5] = 0
    upstream_bf16[:, 1::7] = torch.tensor(
        5e-4, device="cuda", dtype=torch.bfloat16
    )
    actual_grads = _kplanes_tilted_fuse_ms_bwd(
        pts, rotations, *grids, upstream_bf16, *GATES
    )
    expected_grads = _kplanes_tilted_fuse_ms_bwd(
        pts, rotations, *grids_fp32, upstream_bf16.float(), *GATES
    )
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=3e-6)


@requires_cuda
def test_ms_v5_bf16_grid_and_gout_autocast():
    env = os.environ.copy()
    env["QUANTEM_KPLANES_BWD_VARIANT"] = "5"
    env["QUANTEM_KPLANES_BWD_ZERO_TAU"] = "1e-3"
    env["QUANTEM_KPLANES_MS_BF16_OUT"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_ms_bf16_grid_and_gout_autocast_v5_smoke",
            "-q",
        ],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout, result.stdout + result.stderr


@requires_cuda
@pytest.mark.skipif(
    os.environ.get("QUANTEM_KPLANES_BWD_VARIANT") not in {"4", "5"},
    reason="bf16 plane storage is only enabled for backward variants 4 and 5",
)
def test_ms_bf16_variant_smoke():
    pts, rotations, grids = _inputs(seed=11, dtype=torch.bfloat16)
    expected = torch.cat(
        [kplanes_tilted_fuse(pts, rotations, grid) * gate for grid, gate in zip(grids, GATES)],
        dim=-1,
    )
    actual = _multiscale(pts, rotations, *grids)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)

    upstream = torch.randn_like(actual)
    actual_grads = _kplanes_tilted_fuse_ms_bwd(pts, rotations, *grids, upstream, *GATES)
    inputs = tuple(
        tensor.detach().float().requires_grad_(True) for tensor in (pts, rotations, *grids)
    )
    expected_grads = torch.autograd.grad((_torch_cat(*inputs) * upstream).sum(), inputs)
    for expected_grad, actual_grad in zip(expected_grads, actual_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-4, atol=3e-6)


@requires_cuda
@pytest.mark.parametrize("variant", [0, 3, 4, 5])
def test_ms_backward_variant_sweep(variant):
    env = os.environ.copy()
    env["QUANTEM_KPLANES_BWD_VARIANT"] = str(variant)
    env.pop("QUANTEM_KPLANES_BWD_ZERO_TAU", None)
    targets = ["test_ms_backward_reads_strided_upstream_without_packing"]
    if variant in {4, 5}:
        targets.append("test_ms_bf16_variant_smoke")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *(f"{__file__}::{target}" for target in targets), "-q"],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{len(targets)} passed" in result.stdout, result.stdout + result.stderr


@requires_cuda
def test_ms_compile_fullgraph():
    pts, rotations, grids = _inputs(requires_grad=True, seed=13)

    @torch.compile(fullgraph=True)
    def compiled(p, r, g0, g1, g2):
        return kplanes_tilted_fuse_ms(p, r, g0, g1, g2, *GATES).square().mean()

    loss = compiled(pts, rotations, *grids)
    loss.backward()
    for tensor in (pts, rotations, *grids):
        assert tensor.grad is not None


def test_ms_cpu_rejected():
    pts = torch.rand(8, 3)
    rotations = torch.rand(2, 3, 3)
    grids = [torch.rand(6, channels, size, size) for channels, size in ((2, 5), (3, 7), (4, 9))]
    with pytest.raises(ValueError, match="CUDA"):
        kplanes_tilted_fuse_ms(pts, rotations, *grids)
