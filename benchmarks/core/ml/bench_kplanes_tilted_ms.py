"""Three-level K-Planes fusion vs per-level launches + gate mul + cat.

The default shape mirrors the profiled static fp32 tomography step:
B=2048, T=8, C=48, and plane sizes 50²/100²/200².
"""

import torch

from quantem.cuda.core.ml import kplanes_tilted_fuse, kplanes_tilted_fuse_ms

B = 2048
T = 8
C = 48
SIZES = (50, 100, 200)
GATES = (0.25, 0.75, 1.0)
WARMUP = 10
ITERS = 50


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / ITERS


def main():
    assert torch.cuda.is_available(), "CUDA required for benchmark"
    device = torch.device("cuda")
    pts = (torch.rand(B, 3, device=device) * 2 - 1).requires_grad_(True)
    rotations = torch.randn(T, 3, 3, device=device).requires_grad_(True)
    grids = tuple(
        torch.empty(3 * T, C, size, size, device=device)
        .uniform_(0.1, 0.5)
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_(True)
        for size in SIZES
    )
    upstream = torch.randn(B, T * C * 3, device=device)

    def baseline():
        return torch.cat(
            [kplanes_tilted_fuse(pts, rotations, grid) * gate for grid, gate in zip(grids, GATES)],
            dim=-1,
        )

    def multiscale():
        return kplanes_tilted_fuse_ms(pts, rotations, *grids, *GATES)

    tensors = (pts, rotations, *grids)

    def fwd_bwd(fn):
        def run():
            torch.autograd.backward(fn(), upstream)
            for tensor in tensors:
                tensor.grad = None

        return run

    print(f"device: {torch.cuda.get_device_name(device)}")
    print(f"B={B}, T={T}, C={C}, sizes={SIZES}, warmup={WARMUP}, iters={ITERS}\n")
    print(f"{'op':<9s} {'per-level+cat':>15s} {'multiscale':>12s} {'speedup':>9s}")
    for label, base_fn, fused_fn in (
        ("fwd", baseline, multiscale),
        ("fwd+bwd", fwd_bwd(baseline), fwd_bwd(multiscale)),
    ):
        base_ms = time_ms(base_fn)
        fused_ms = time_ms(fused_fn)
        print(f"{label:<9s} {base_ms:>12.3f} ms {fused_ms:>9.3f} ms {base_ms / fused_ms:>8.2f}x")


if __name__ == "__main__":
    main()
