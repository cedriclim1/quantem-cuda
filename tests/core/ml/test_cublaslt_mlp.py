"""Parity coverage for the cuBLASLt fused two-hidden-layer sigma head."""

import copy
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


@requires_cuda
@pytest.mark.parametrize("rows", [1, 37, 2048, 367000])
def test_cublaslt_mlp_forward_and_gradients(rows):
    torch.manual_seed(917)
    device = torch.device("cuda")
    layers = _make_layers(device)
    eager_layers = copy.deepcopy(layers)
    truth_layers = copy.deepcopy(layers)
    x = (torch.randn((rows, INPUT_WIDTH), device=device) * 0.25).bfloat16().requires_grad_()
    eager_x = x.detach().clone().requires_grad_()
    truth_x = x.detach().float().requires_grad_()
    grad_out = torch.randn((rows, 1), device=device).bfloat16()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager = eager_layers[2](torch.relu(eager_layers[1](torch.relu(eager_layers[0](eager_x)))))
    eager.backward(grad_out)
    eager_grads = (
        eager_x.grad,
        *(parameter.grad for layer in eager_layers for parameter in layer.parameters()),
    )

    fp32_truth = truth_layers[2](torch.relu(truth_layers[1](torch.relu(truth_layers[0](truth_x)))))
    fp32_truth.backward(grad_out.float())
    truth_grads = (
        truth_x.grad,
        *(parameter.grad for layer in truth_layers for parameter in layer.parameters()),
    )

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
    eager_error = (eager.float() - fp32_truth).abs()
    fused_error = (actual.float() - fp32_truth).abs()
    assert torch.mean(fused_error) <= torch.mean(eager_error) * 1.10 + 1e-7
    out_scale = fp32_truth.abs().max()
    bf16_eps_at_out_scale = out_scale * 2**-8
    # The per-element max over 1e2..1e5 bf16-rounded outputs is a noisy order
    # statistic: measured fused/eager maxima were 1.2e-3/9e-4 on O(0.1)
    # outputs, one bf16 quantum apart.
    assert torch.max(fused_error) <= (torch.max(eager_error) * 1.5 + 2 * bf16_eps_at_out_scale)

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
    # At M=37 the measured fused/eager mean-error ratios against fp32 truth
    # were 1.11 for dx and 1.34 for dW1; reduction order makes direct closeness
    # to a semi-analytic reference inappropriate for both bf16 paths.
    for actual_grad, eager_grad, truth_grad in zip(actual_grads, eager_grads, truth_grads):
        eager_error = (eager_grad.float() - truth_grad).abs()
        fused_error = (actual_grad.float() - truth_grad).abs()
        assert torch.mean(fused_error) <= torch.mean(eager_error) * 1.5 + 1e-7
        truth_scale = truth_grad.abs().max()
        bf16_eps_at_truth_scale = truth_scale * 2**-8
        assert torch.max(fused_error) <= (
            torch.max(eager_error) * 1.5 + 2 * bf16_eps_at_truth_scale
        )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_cublaslt_mlp_uses_a_handle_per_device():
    for device_index in (0, 1):
        device = torch.device("cuda", device_index)
        layers = _make_layers(device)
        x = torch.randn((1, INPUT_WIDTH), device=device, dtype=torch.bfloat16)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = fused_hidden_mlp(
                x,
                layers[0].weight,
                layers[0].bias,
                layers[1].weight,
                layers[1].bias,
                layers[2].weight,
                layers[2].bias,
            )
        assert output.device == device


def test_backward_fallback_memoizes_shapes_and_warns_once_per_reason(monkeypatch):
    from quantem.cuda.core.ml import _ops

    monkeypatch.setattr(_ops, "_unsupported_mlp_shapes", {})
    monkeypatch.setattr(_ops, "_warned_mlp_fallback_reasons", set())
    key1 = (0, 37, INPUT_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH, 1)
    key2 = (0, 2048, INPUT_WIDTH, HIDDEN_WIDTH, HIDDEN_WIDTH, 1)
    reason = "RuntimeError: no compatible DRELU_BGRAD heuristic"

    with pytest.warns(RuntimeWarning, match="future calls will use the eager path") as records:
        _ops._memoize_unsupported_mlp_shape(key1, reason)
        _ops._memoize_unsupported_mlp_shape(key2, reason)

    assert len(records) == 1
    assert _ops._unsupported_mlp_reason(key1) == reason
    assert _ops._unsupported_mlp_reason(key2) == reason


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
