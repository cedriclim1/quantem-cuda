"""Three-level fused plane TV versus the eager production loss chain."""

import torch

from quantem.cuda.core.ml import plane_tv_loss

T = 8
C = 48
SIZES = (50, 100, 200)
WARMUP = 10
ITERS = 50


def eager(grids):
    levels = []
    for grid in grids:
        dh = (grid[:, :, 1:, :] - grid[:, :, :-1, :]).pow(2).mean(dim=(1, 2, 3))
        dw = (grid[:, :, :, 1:] - grid[:, :, :, :-1]).pow(2).mean(dim=(1, 2, 3))
        levels.append((dh + dw).view(T, 3).sum(dim=1).mean())
    return torch.stack(levels).sum()


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
    grids = tuple(
        torch.rand((3 * T, C, size, size), device="cuda")
        .contiguous(memory_format=torch.channels_last)
        .requires_grad_(True)
        for size in SIZES
    )
    grid_gb = sum(grid.numel() * grid.element_size() for grid in grids) / 1e9

    def fused():
        return plane_tv_loss(*grids)

    def fwd_bwd(fn):
        def run():
            fn().backward()
            for grid in grids:
                grid.grad = None

        return run

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"T={T}, C={C}, sizes={SIZES}, grid={grid_gb:.3f} GB")
    print(f"{'pass':<9} {'eager ms':>10} {'fused ms':>10} {'speedup':>9} {'fused GB/s':>12}")
    for label, eager_fn, fused_fn, byte_multiplier in (
        ("fwd", lambda: eager(grids), fused, 1),
        ("fwd+bwd", fwd_bwd(lambda: eager(grids)), fwd_bwd(fused), 2),
    ):
        eager_ms = time_ms(eager_fn)
        fused_ms = time_ms(fused_fn)
        effective_gbps = byte_multiplier * grid_gb / (fused_ms / 1000.0)
        print(
            f"{label:<9} {eager_ms:>8.3f} ms {fused_ms:>8.3f} ms "
            f"{eager_ms / fused_ms:>8.2f}x {effective_gbps:>11.1f}"
        )


if __name__ == "__main__":
    main()
