"""Benchmark the fused TV kernels against pure-torch implementations.

Run on a CUDA machine:

    uv run python benchmarks/bench_tv_loss.py
"""

import torch

from quantem.cuda.core import tv_loss_iso_3d, tv_loss_sq_3d

SHAPES = [(256, 256, 256), (4, 256, 256, 256), (512, 512, 512)]
N_WARMUP = 5
N_ITER = 20


def tv_iso_torch(volume: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    v = volume.reshape(-1, *volume.shape[-3:])
    dd = (v[:, 1:, :, :] - v[:, :-1, :, :])[:, :, :-1, :-1]
    dh = (v[:, :, 1:, :] - v[:, :, :-1, :])[:, :-1, :, :-1]
    dw = (v[:, :, :, 1:] - v[:, :, :, :-1])[:, :-1, :-1, :]
    return (dd.pow(2) + dh.pow(2) + dw.pow(2) + eps).sqrt().mean()


def tv_sq_torch(volume: torch.Tensor) -> torch.Tensor:
    tv_d = torch.pow(volume[..., 1:, :, :] - volume[..., :-1, :, :], 2).sum()
    tv_h = torch.pow(volume[..., :, 1:, :] - volume[..., :, :-1, :], 2).sum()
    tv_w = torch.pow(volume[..., :, :, 1:] - volume[..., :, :, :-1], 2).sum()
    return tv_d + tv_h + tv_w


def time_ms(fn, vol, backward: bool) -> float:
    for _ in range(N_WARMUP):
        v = vol.detach().requires_grad_(backward)
        loss = fn(v)
        if backward:
            loss.backward()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N_ITER):
        v = vol.detach().requires_grad_(backward)
        loss = fn(v)
        if backward:
            loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N_ITER


def main() -> None:
    assert torch.cuda.is_available(), "benchmark requires a CUDA device"
    print(f"device: {torch.cuda.get_device_name()}")
    header = (
        f"{'shape':>22} {'variant':>9} {'pass':>8} {'torch ms':>9} {'cuda ms':>8} {'speedup':>8}"
    )
    print(header)
    print("-" * len(header))
    for shape in SHAPES:
        vol = torch.rand(shape, device="cuda", dtype=torch.float32)
        for name, ref_fn, cuda_fn in [
            ("iso", tv_iso_torch, tv_loss_iso_3d),
            ("squared", tv_sq_torch, tv_loss_sq_3d),
        ]:
            for backward in (False, True):
                t_ref = time_ms(ref_fn, vol, backward)
                t_cuda = time_ms(cuda_fn, vol, backward)
                tag = "fwd+bwd" if backward else "fwd"
                print(
                    f"{str(shape):>22} {name:>9} {tag:>8} "
                    f"{t_ref:>9.3f} {t_cuda:>8.3f} {t_ref / t_cuda:>7.2f}x"
                )


if __name__ == "__main__":
    main()
