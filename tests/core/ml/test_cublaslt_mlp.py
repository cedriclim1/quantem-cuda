"""Parity coverage for the cuBLASLt fused two-hidden-layer sigma head."""

import math

import pytest
import torch
from torch import nn

from quantem.cuda.core.ml import fused_hidden_mlp

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")

INPUT_WIDTH = 1152
HIDDEN_WIDTH = 384


def _make_layers(device):
    generator = torch.Generator(device=device).manual_seed(413)
    layers = (
        nn.Linear(INPUT_WIDTH, HIDDEN_WIDTH, device=device),
        nn.Linear(HIDDEN_WIDTH, HIDDEN_WIDTH, device=device),
        nn.Linear(HIDDEN_WIDTH, 1, device=device),
    )
    with torch.no_grad():
        for layer in layers:
            layer.weight.copy_(
                torch.randn(layer.weight.shape, generator=generator, device=device)
                / math.sqrt(layer.in_features)
            )
            layer.bias.copy_(
                torch.randn(layer.bias.shape, generator=generator, device=device) * 0.01
            )
    return layers


def _one_ulp_close(actual, expected):
    positive = torch.full_like(expected, float("inf"))
    negative = torch.full_like(expected, float("-inf"))
    return torch.all(
        (actual == expected)
        | (actual == torch.nextafter(expected, positive))
        | (actual == torch.nextafter(expected, negative))
    )


def _fp32_boundary_reference(x, layers, grad_out):
    """FP32 GEMMs with bf16 rounding only at the fused op's storage boundaries."""
    w1, w2, w3 = (layer.weight.bfloat16().float() for layer in layers)
    b1, b2, b3 = (layer.bias.bfloat16().float() for layer in layers)
    x32 = x.float()

    z1 = x32 @ w1.T + b1
    h1 = z1.relu().bfloat16().float()
    z2 = h1 @ w2.T + b2
    h2 = z2.relu().bfloat16().float()
    out = (h2 @ w3.T + b3).bfloat16()

    dy = grad_out.float()
    grad_w3 = dy.T @ h2
    grad_b3 = dy.sum(dim=0)
    dh2 = dy @ w3
    dz2_full = dh2 * (z2 > 0)
    dz2 = dz2_full.bfloat16().float()
    grad_b2 = dz2_full.sum(dim=0).bfloat16().float()
    grad_w2 = dz2.T @ h1
    dh1 = dz2 @ w2
    dz1_full = dh1 * (z1 > 0)
    dz1 = dz1_full.bfloat16().float()
    grad_b1 = dz1_full.sum(dim=0).bfloat16().float()
    grad_w1 = dz1.T @ x32
    grad_x = (dz1 @ w1).bfloat16()
    return out, (grad_x, grad_w1, grad_b1, grad_w2, grad_b2, grad_w3, grad_b3)


@requires_cuda
@pytest.mark.parametrize("rows", [1, 37, 2048, 367000])
def test_cublaslt_mlp_forward_and_gradients(rows):
    torch.manual_seed(917)
    device = torch.device("cuda")
    layers = _make_layers(device)
    x = (torch.randn((rows, INPUT_WIDTH), device=device) * 0.25).bfloat16().requires_grad_()
    grad_out = torch.randn((rows, 1), device=device).bfloat16()

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        eager = layers[2](torch.relu(layers[1](torch.relu(layers[0](x)))))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = fused_hidden_mlp(
            x,
            layers[0].weight,
            layers[0].bias,
            layers[1].weight,
            layers[1].bias,
            layers[2].weight,
            layers[2].bias,
        )
    assert actual.shape == (rows, 1)
    assert actual.dtype == torch.bfloat16
    assert _one_ulp_close(actual, eager), "forward differs from autocast eager by more than 1 ulp"

    actual.backward(grad_out)
    actual_grads = (
        x.grad,
        layers[0].weight.grad,
        layers[0].bias.grad,
        layers[1].weight.grad,
        layers[1].bias.grad,
        layers[2].weight.grad,
        layers[2].bias.grad,
    )
    reference_out, reference_grads = _fp32_boundary_reference(x.detach(), layers, grad_out)
    assert _one_ulp_close(actual.detach(), reference_out)
    for actual_grad, reference_grad in zip(actual_grads, reference_grads):
        torch.testing.assert_close(
            actual_grad.float(), reference_grad.float(), rtol=3e-3, atol=1e-4
        )


def test_fused_hidden_mlp_rejects_fp32_cpu_input():
    layers = _make_layers(torch.device("cpu"))
    with pytest.raises(ValueError, match="CUDA"):
        fused_hidden_mlp(
            torch.randn(4, INPUT_WIDTH),
            layers[0].weight,
            layers[0].bias,
            layers[1].weight,
            layers[1].bias,
            layers[2].weight,
            layers[2].bias,
        )
