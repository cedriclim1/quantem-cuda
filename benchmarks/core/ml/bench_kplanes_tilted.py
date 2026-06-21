"""Fused TILTED K-Planes interpolation vs the pure-torch chain.

Configs mirror quantem tomography DIP/INR workloads: B = batch_rays x
samples_per_ray points per training batch.
"""

import torch
import torch.nn.functional as F

from quantem.cuda.core.ml import kplanes_tilted_fuse

# (label, B, T, C, H, W)
CONFIGS = [
    ("light  (8192x200 rays, T=4,  C=8,  200²)", 8192 * 200, 4, 8, 200, 200),
    ("heavy  (8192x200 rays, T=8,  C=32, 200²)", 8192 * 200, 8, 32, 200, 200),
    ("decode (5*N² pts,      T=4,  C=8,  200²)", 5 * 200 * 200, 4, 8, 200, 200),
]


def reference(pts, rotations, plane):
    T = rotations.shape[0]
    B = pts.shape[0]
    C = plane.shape[1]
    rotated = torch.einsum("tij,bj->tbi", rotations, pts)
    idx = torch.tensor([[0, 1], [2, 0], [1, 2]], device=pts.device)
    coords = (
        rotated.unsqueeze(1).expand(T, 3, B, 3).gather(-1, idx.view(1, 3, 1, 2).expand(T, 3, B, 2))
    )
    sampled = F.grid_sample(
        plane,
        coords.reshape(3 * T, B, 1, 2),
        align_corners=True,
        mode="bilinear",
        padding_mode="border",
    )
    return sampled.squeeze(-1).view(T, 3, C, B).prod(dim=1).permute(2, 0, 1).reshape(B, T * C)


def time_ms(fn, warmup=5, iters=20):
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


def ray_points(B, dev, samples_per_ray=200):
    """Points structured like the tomography sampler: consecutive entries are
    consecutive samples along a ray (create_batch_rays), i.e. spatially
    adjacent — the cache behavior the real workload has."""
    n_rays = B // samples_per_ray
    xy = torch.rand(n_rays, 1, 2, device=dev) * 2 - 1
    z = torch.linspace(-1, 1, samples_per_ray, device=dev).view(1, -1, 1)
    rays = torch.cat([xy.expand(-1, samples_per_ray, -1), z.expand(n_rays, -1, -1)], dim=-1)
    return rays.reshape(-1, 3)


def main():
    assert torch.cuda.is_available()
    dev = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name(dev)}\n")
    print(f"{'config':<44s} {'op':<9s} {'torch':>9s} {'fused':>9s} {'speedup':>8s}")

    for label, B, T, C, H, W in CONFIGS:
        if "rays" in label:
            pts = ray_points(B, dev).requires_grad_(True)
        else:
            pts = (torch.rand(B, 3, device=dev) * 2 - 1).requires_grad_(True)
        rotations = torch.randn(T, 3, 3, device=dev).requires_grad_(True)
        plane = torch.empty(3 * T, C, H, W, device=dev).uniform_(0.1, 0.5).requires_grad_(True)
        upstream = torch.randn(B, T * C, device=dev)

        def fwd(fn):
            return lambda: fn(pts, rotations, plane)

        def fwd_bwd(fn):
            def run():
                out = fn(pts, rotations, plane)
                (out * upstream).sum().backward()
                for t in (pts, rotations, plane):
                    t.grad = None

            return run

        for op, mk in (("fwd", fwd), ("fwd+bwd", fwd_bwd)):
            ms_ref = time_ms(mk(reference))
            ms_fused = time_ms(mk(kplanes_tilted_fuse))
            print(
                f"{label:<44s} {op:<9s} {ms_ref:8.2f}m {ms_fused:8.2f}m {ms_ref / ms_fused:7.1f}x"
            )


if __name__ == "__main__":
    main()
