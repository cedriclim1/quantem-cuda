"""Fused trunc-exp density-tail forward/backward and layout parity."""

import pytest
import torch

from quantem.cuda.core.ml import density_tail

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")


class _ReferenceTruncExp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, offset):
        ctx.save_for_backward(values)
        ctx.offset = offset
        return torch.exp(values - offset)

    @staticmethod
    def backward(ctx, grad_out):
        (values,) = ctx.saved_tensors
        shifted = values - ctx.offset
        return grad_out * torch.exp(shifted.clamp(max=15)), None


@requires_cuda
@pytest.mark.parametrize(
    ("dtype", "rtol", "atol"),
    [(torch.float32, 2e-6, 2e-7), (torch.bfloat16, 8e-3, 2e-3)],
)
def test_density_tail_forward_backward_matches_reference(dtype, rtol, atol):
    base = torch.linspace(-18, 18, 4099, device="cuda", dtype=dtype).reshape(-1, 1)
    actual_values = base.detach().clone().requires_grad_(True)
    expected_values = base.detach().clone().requires_grad_(True)
    upstream = torch.linspace(-0.75, 1.25, base.numel(), device="cuda", dtype=dtype).reshape_as(
        base
    )

    actual = density_tail(actual_values, 1.375)
    expected = _ReferenceTruncExp.apply(expected_values, 1.375)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)

    actual_grad = torch.autograd.grad(actual, actual_values, upstream)[0]
    expected_grad = torch.autograd.grad(expected, expected_values, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=rtol, atol=atol)


@requires_cuda
def test_density_tail_uses_input_and_upstream_strides_without_packing():
    values_backing = torch.randn((257, 7), device="cuda")
    values = values_backing[:, 1::3].requires_grad_(True)
    upstream_backing = torch.randn((257, 9), device="cuda")
    upstream = upstream_backing[:, 2:8:3]
    assert not values.is_contiguous()
    assert not upstream.is_contiguous()

    actual = density_tail(values, 0.625)
    expected_values = values.detach().clone().requires_grad_(True)
    expected = _ReferenceTruncExp.apply(expected_values, 0.625)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-7)

    actual_grad = torch.autograd.grad(actual, values, upstream)[0]
    expected_grad = torch.autograd.grad(expected, expected_values, upstream)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=2e-7)


@requires_cuda
def test_density_tail_compile_fullgraph():
    values = torch.randn((1025, 1), device="cuda", requires_grad=True)

    @torch.compile(fullgraph=True)
    def compiled(tensor):
        return density_tail(tensor, 1.25).square().mean()

    compiled(values).backward()
    assert values.grad is not None


def test_density_tail_rejects_cpu_tensor():
    with pytest.raises(ValueError, match="CUDA"):
        density_tail(torch.randn(8, 1), 1.0)
