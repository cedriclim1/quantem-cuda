"""Fused NON-TILTED K-Planes interpolation vs the pure-torch chain.

Configs mirror quantem tomography INR workloads: B = batch_rays x
samples_per_ray points per training batch, one call per multiscale level.
"""

import torch
import torch.nn.functional as F

from quantem.cuda.core.ml import kplanes_fuse

# (label, B, C, H, W)
CONFIGS = [
    ("light  (8192x200 rays, C=8,  200²)", 8192 * 200, 8, 200, 200),
    ("heavy  (8192x200 rays, C=32, 200²)", 8192 * 200, 32, 200, 200),
    ("decode (5*N² pts,      C=8,  200²)", 5 * 200 * 200, 8, 200, 200),
]


def reference(pts, plane):
    """One level of quantem's interpolate_ms_features, verbatim semantics."""
    C = plane.shape[1]
    mat_mode = [[0, 1], [0, 2], [1, 2]]
    coord_plane = torch.stack(
        [
            pts[:, mat_mode[0]],
            pts[:, mat_mode[1]],
            pts[:, mat_mode[2]],
        ]
    ).view(3, -1, 1, 2)
    feats = F.grid_sample(
        plane, coord_plane, align_corners=True, mode="bilinear", padding_mode="border"
    ).reshape(3, C, -1)
    return (feats[0] * feats[1] * feats[2]).T


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
    print(f"{'config':<40s} {'op':<9s} {'torch':>9s} {'fused':>9s} {'speedup':>8s}")

    for label, B, C, H, W in CONFIGS:
        if "rays" in label:
            pts = ray_points(B, dev).requires_grad_(True)
        else:
            pts = (torch.rand(B, 3, device=dev) * 2 - 1).requires_grad_(True)
        plane = torch.empty(3, C, H, W, device=dev).uniform_(0.1, 0.5).requires_grad_(True)
        upstream = torch.randn(B, C, device=dev)

        def fwd(fn):
            return lambda: fn(pts, plane)

        def fwd_bwd(fn):
            def run():
                out = fn(pts, plane)
                (out * upstream).sum().backward()
                for t in (pts, plane):
                    t.grad = None

            return run

        for op, mk in (("fwd", fwd), ("fwd+bwd", fwd_bwd)):
            ms_ref = time_ms(mk(reference))
            ms_fused = time_ms(mk(kplanes_fuse))
            print(
                f"{label:<40s} {op:<9s} {ms_ref:8.2f}m {ms_fused:8.2f}m {ms_ref / ms_fused:7.1f}x"
            )


if __name__ == "__main__":
    main()
