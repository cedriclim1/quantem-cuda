"""TV loss kernels vs pure-torch references: forward values and gradients."""

import pytest
import torch

from quantem.cuda import tv_loss_iso_3d, tv_loss_sq_3d

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

SHAPES = [(33, 57, 18), (3, 32, 32, 32), (2, 3, 16, 16, 16)]


def tv_iso_ref(volume: torch.Tensor, eps: float) -> torch.Tensor:
    """Corner-restricted isotropic TV, mean over all corners (incl. leading dims)."""
    v = volume.reshape(-1, *volume.shape[-3:])
    dd = (v[:, 1:, :, :] - v[:, :-1, :, :])[:, :, :-1, :-1]
    dh = (v[:, :, 1:, :] - v[:, :, :-1, :])[:, :-1, :, :-1]
    dw = (v[:, :, :, 1:] - v[:, :, :, :-1])[:, :-1, :-1, :]
    return (dd.pow(2) + dh.pow(2) + dw.pow(2) + eps).sqrt().mean()


def tv_sq_ref(volume: torch.Tensor) -> torch.Tensor:
    """quantem's tv_vol formulation: unrestricted squared forward differences."""
    tv_d = torch.pow(volume[..., 1:, :, :] - volume[..., :-1, :, :], 2).sum()
    tv_h = torch.pow(volume[..., :, 1:, :] - volume[..., :, :-1, :], 2).sum()
    tv_w = torch.pow(volume[..., :, :, 1:] - volume[..., :, :, :-1], 2).sum()
    return tv_d + tv_h + tv_w


def _rand_volume(shape, seed=0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.rand(shape, device="cuda", dtype=torch.float32, generator=gen)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_iso_forward(shape):
    vol = _rand_volume(shape)
    expected = tv_iso_ref(vol, eps=1e-8)
    actual = tv_loss_iso_3d(vol, eps=1e-8)
    assert actual.shape == ()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_iso_grad(shape):
    vol = _rand_volume(shape, seed=1)
    v_kernel = vol.clone().requires_grad_(True)
    v_ref = vol.clone().requires_grad_(True)
    tv_loss_iso_3d(v_kernel, eps=1e-6).backward()
    tv_iso_ref(v_ref, eps=1e-6).backward()
    torch.testing.assert_close(v_kernel.grad, v_ref.grad, rtol=1e-4, atol=1e-7)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_sq_forward(shape):
    vol = _rand_volume(shape, seed=2)
    expected = tv_sq_ref(vol)
    actual = tv_loss_sq_3d(vol)
    assert actual.shape == ()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@requires_cuda
@pytest.mark.parametrize("shape", SHAPES)
def test_sq_grad(shape):
    vol = _rand_volume(shape, seed=3)
    v_kernel = vol.clone().requires_grad_(True)
    v_ref = vol.clone().requires_grad_(True)
    tv_loss_sq_3d(v_kernel).backward()
    tv_sq_ref(v_ref).backward()
    torch.testing.assert_close(v_kernel.grad, v_ref.grad, rtol=1e-4, atol=1e-6)


@requires_cuda
def test_upstream_grad_scaling():
    # weight * loss must scale the volume gradient by weight (chain rule
    # through the custom op's grad_out handling).
    vol = _rand_volume((16, 16, 16), seed=4)
    v1 = vol.clone().requires_grad_(True)
    v2 = vol.clone().requires_grad_(True)
    tv_loss_sq_3d(v1).backward()
    (3.0 * tv_loss_sq_3d(v2)).backward()
    torch.testing.assert_close(v2.grad, 3.0 * v1.grad, rtol=1e-6, atol=1e-8)


@requires_cuda
def test_non_contiguous_input():
    vol = _rand_volume((24, 18, 30), seed=5).permute(2, 0, 1)
    assert not vol.is_contiguous()
    torch.testing.assert_close(tv_loss_sq_3d(vol), tv_sq_ref(vol), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        tv_loss_iso_3d(vol), tv_iso_ref(vol, eps=1e-8), rtol=1e-5, atol=1e-7
    )


@requires_cuda
def test_degenerate_dims():
    # A dim of size 1 leaves no corners for the isotropic variant (loss 0,
    # grad 0) while the squared variant still sums the remaining axes.
    vol = _rand_volume((1, 32, 32), seed=6).requires_grad_(True)
    loss = tv_loss_iso_3d(vol)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.all(vol.grad == 0)

    vol2 = _rand_volume((1, 32, 32), seed=6)
    torch.testing.assert_close(tv_loss_sq_3d(vol2), tv_sq_ref(vol2), rtol=1e-5, atol=1e-6)


@requires_cuda
def test_torch_compile():
    def loss_fn(v):
        return tv_loss_iso_3d(v, eps=1e-8) + 0.1 * tv_loss_sq_3d(v)

    compiled = torch.compile(loss_fn, fullgraph=True)
    vol = _rand_volume((32, 32, 32), seed=7).requires_grad_(True)
    eager = loss_fn(vol)
    traced = compiled(vol)
    torch.testing.assert_close(traced, eager, rtol=1e-6, atol=1e-8)


def test_rejects_cpu_tensor():
    vol = torch.rand(8, 8, 8)
    with pytest.raises(ValueError, match="CUDA"):
        tv_loss_iso_3d(vol)
    with pytest.raises(ValueError, match="CUDA"):
        tv_loss_sq_3d(vol)


def test_rejects_bad_dtype_and_ndim():
    vol64 = torch.rand(8, 8, 8, dtype=torch.float64)
    with pytest.raises(TypeError, match="fp32"):
        tv_loss_iso_3d(vol64)
    with pytest.raises(ValueError, match="ndim"):
        tv_loss_sq_3d(torch.rand(8, 8))
