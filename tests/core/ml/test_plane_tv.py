"""Plane-wise 2-D TV forward/backward parity for three multiscale grids."""

import pytest
import torch

from quantem.cuda.core.ml import plane_tv_loss
from quantem.cuda.core.ml._ops import _channels_last

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

LEVEL_SHAPES = ((2, 5, 7), (3, 9, 11), (4, 13, 17))


def plane_tv_reference(grids: tuple[torch.Tensor, ...], rotations: int) -> torch.Tensor:
    """The eager fp32 chain from ObjectTensorDecomp._get_plane_tv_loss."""
    levels = []
    for grid in grids:
        dh = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
        dw = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
        levels.append((dh + dw).view(rotations, 3).sum(dim=1).mean())
    return torch.stack(levels).sum()


def make_grids(rotations: int, *, seed: int, channels_last: bool = True):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    grids = []
    for channels, height, width in LEVEL_SHAPES:
        grid = torch.empty(
            (3 * rotations, channels, height, width), device="cuda", dtype=torch.float32
        ).uniform_(-0.5, 0.5, generator=generator)
        if channels_last:
            grid = grid.contiguous(memory_format=torch.channels_last)
        grids.append(grid)
    return tuple(grids)


@requires_cuda
@pytest.mark.parametrize("rotations", [1, 4], ids=["kplanes", "tilted_t4"])
def test_plane_tv_forward_matches_eager_fp32(rotations):
    grids = make_grids(rotations, seed=10 + rotations)
    expected = plane_tv_reference(grids, rotations)
    actual = plane_tv_loss(*grids)
    assert actual.shape == ()
    assert actual.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)


@requires_cuda
@pytest.mark.parametrize("rotations", [1, 3], ids=["kplanes", "tilted_t3"])
def test_plane_tv_backward_matches_eager_fp32(rotations):
    base = make_grids(rotations, seed=20 + rotations)
    fused_grids = tuple(grid.detach().clone().requires_grad_(True) for grid in base)
    eager_grids = tuple(grid.detach().clone().requires_grad_(True) for grid in base)
    upstream = torch.tensor(0.375, device="cuda")

    fused_grads = torch.autograd.grad(plane_tv_loss(*fused_grids), fused_grids, upstream)
    eager_grads = torch.autograd.grad(
        plane_tv_reference(eager_grids, rotations), eager_grids, upstream
    )
    for actual, expected in zip(fused_grads, eager_grads):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-7)


@requires_cuda
@pytest.mark.parametrize("direction", ["height", "width"])
def test_plane_tv_directional_terms_and_gradients(direction):
    rotations = 2
    base = []
    for channels, height, width in LEVEL_SHAPES:
        if direction == "height":
            values = torch.arange(height, device="cuda", dtype=torch.float32).view(1, 1, -1, 1)
        else:
            values = torch.arange(width, device="cuda", dtype=torch.float32).view(1, 1, 1, -1)
        base.append(
            values.expand(3 * rotations, channels, height, width)
            .contiguous(memory_format=torch.channels_last)
            .div(max(height, width))
        )
    fused_grids = tuple(grid.detach().clone().requires_grad_(True) for grid in base)
    eager_grids = tuple(grid.detach().clone().requires_grad_(True) for grid in base)

    actual = plane_tv_loss(*fused_grids)
    expected = plane_tv_reference(eager_grids, rotations)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    actual_grads = torch.autograd.grad(actual, fused_grids)
    expected_grads = torch.autograd.grad(expected, eager_grids)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=2e-7)


@requires_cuda
def test_plane_tv_channels_last_is_zero_copy_and_gradient_preserves_layout():
    grids = tuple(grid.requires_grad_(True) for grid in make_grids(2, seed=31))
    for grid in grids:
        assert grid.is_contiguous(memory_format=torch.channels_last)
        assert _channels_last(grid).data_ptr() == grid.data_ptr()

    plane_tv_loss(*grids).backward()
    for grid in grids:
        assert grid.grad is not None
        assert grid.grad.is_contiguous(memory_format=torch.channels_last)


@requires_cuda
def test_plane_tv_accepts_standard_nchw_and_restores_gradient_layout():
    grids = tuple(
        grid.detach().contiguous().requires_grad_(True)
        for grid in make_grids(2, seed=32, channels_last=False)
    )
    expected_grids = tuple(grid.detach().clone().requires_grad_(True) for grid in grids)
    actual = plane_tv_loss(*grids)
    expected = plane_tv_reference(expected_grids, 2)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)
    actual.backward()
    expected.backward()
    for grid, expected_grid in zip(grids, expected_grids):
        assert grid.grad is not None and grid.grad.is_contiguous()
        torch.testing.assert_close(grid.grad, expected_grid.grad, rtol=1e-5, atol=2e-7)


def test_plane_tv_rejects_cpu_tensors():
    grids = tuple(
        torch.rand((6, channels, height, width)) for channels, height, width in LEVEL_SHAPES
    )
    with pytest.raises(ValueError, match="CUDA"):
        plane_tv_loss(*grids)
