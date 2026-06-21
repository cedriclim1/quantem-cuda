"""Benchmark: kplanes_tilted_tv_fuse (N base points, 4 taps fused)
vs kplanes_tilted_fuse on 4*N concatenated points (4 separate evaluations).

Configs match the spec: N=10_000 and N=50_000, T=4, C=32, planes 200², h=0.01.
Timing: 20 warmup + 100 iters per measurement with CUDA events.
"""

import torch

from quantem.cuda.core.ml import kplanes_tilted_fuse, kplanes_tilted_tv_fuse

WARMUP = 20
ITERS = 100
H = 200
T = 4
C = 32
H_STEP = 0.01

CONFIGS = [
    ("N=10_000", 10_000),
    ("N=50_000", 50_000),
]


def time_ms(fn, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def make_inputs(N, device):
    pts = (torch.rand(N, 3, device=device) * 2 - 1).requires_grad_(True)
    rotations = torch.randn(T, 3, 3, device=device).requires_grad_(True)
    plane = torch.empty(3 * T, C, H, H, device=device).uniform_(0.1, 0.5).requires_grad_(True)
    return pts, rotations, plane


def fwd_baseline(pts, rotations, plane):
    """4 separate kplanes_tilted_fuse calls on perturbed point sets."""
    outs = [kplanes_tilted_fuse(pts, rotations, plane)]
    for axis in range(3):
        delta = torch.zeros_like(pts)
        delta[:, axis] = H_STEP
        outs.append(kplanes_tilted_fuse(pts + delta, rotations, plane))
    return torch.stack(outs, dim=0)  # [4, N, T*C]


def fwd_bwd_baseline(pts, rotations, plane, upstream):
    out = fwd_baseline(pts, rotations, plane)
    (out * upstream).sum().backward()
    for t in (pts, rotations, plane):
        t.grad = None


def fwd_tv(pts, rotations, plane):
    return kplanes_tilted_tv_fuse(pts, rotations, plane, h=H_STEP)


def fwd_bwd_tv(pts, rotations, plane, upstream):
    out = fwd_tv(pts, rotations, plane)
    (out * upstream).sum().backward()
    for t in (pts, rotations, plane):
        t.grad = None


def main():
    assert torch.cuda.is_available(), "CUDA required for benchmark"
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(dev)}")
    print(f"T={T}, C={C}, planes {H}², h={H_STEP}, warmup={WARMUP}, iters={ITERS}\n")
    print(
        f"{'config':<14s} {'op':<9s} {'baseline (4x fuse)':>18s} {'tv_fuse':>10s} {'speedup':>9s}"
    )
    print("-" * 66)

    for label, N in CONFIGS:
        pts, rotations, plane = make_inputs(N, dev)
        upstream = torch.randn(4, N, T * C, device=dev)

        for op_label, base_fn, tv_fn in (
            ("fwd", fwd_baseline, fwd_tv),
            ("fwd+bwd", fwd_bwd_baseline, fwd_bwd_tv),
        ):
            if "bwd" in op_label:
                ms_base = time_ms(lambda: fwd_bwd_baseline(pts, rotations, plane, upstream))
                ms_tv = time_ms(lambda: fwd_bwd_tv(pts, rotations, plane, upstream))
            else:
                ms_base = time_ms(lambda: fwd_baseline(pts, rotations, plane))
                ms_tv = time_ms(lambda: fwd_tv(pts, rotations, plane))

            ratio = ms_base / ms_tv
            print(
                f"{label:<14s} {op_label:<9s} {ms_base:>15.3f} ms {ms_tv:>8.3f} ms {ratio:>8.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
