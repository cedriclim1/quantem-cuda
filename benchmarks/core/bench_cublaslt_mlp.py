"""Micro A/B for the production sigma-head shape."""

import argparse
import time

import torch
from torch import nn

from quantem.cuda.core.ml import fused_hidden_mlp


def _time_step(fn, iterations):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=367000)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    layers = (
        nn.Linear(1152, 384, device=device),
        nn.Linear(384, 384, device=device),
        nn.Linear(384, 1, device=device),
    )
    x = torch.randn((args.rows, 1152), device=device, dtype=torch.bfloat16, requires_grad=True)
    upstream = torch.randn((args.rows, 1), device=device, dtype=torch.bfloat16)

    def clear_grads():
        x.grad = None
        for layer in layers:
            layer.weight.grad = None
            layer.bias.grad = None

    def eager_step():
        clear_grads()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = layers[2](torch.relu(layers[1](torch.relu(layers[0](x)))))
        out.backward(upstream)

    def fused_step():
        clear_grads()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = fused_hidden_mlp(
                x,
                layers[0].weight,
                layers[0].bias,
                layers[1].weight,
                layers[1].bias,
                layers[2].weight,
                layers[2].bias,
            )
        out.backward(upstream)

    eager_ms = _time_step(eager_step, args.iterations)
    fused_ms = _time_step(fused_step, args.iterations)
    step_flops = 6 * args.rows * (1152 * 384 + 384 * 384 + 384)
    eager_tflops = step_flops / eager_ms / 1e9
    fused_tflops = step_flops / fused_ms / 1e9
    print(
        f"rows={args.rows} eager={eager_ms:.3f} ms ({eager_tflops:.2f} TFLOP/s) "
        f"fused={fused_ms:.3f} ms ({fused_tflops:.2f} TFLOP/s)"
    )
    print(f"delta={eager_ms - fused_ms:.3f} ms speedup={eager_ms / fused_ms:.3f}x")


if __name__ == "__main__":
    main()
