#pragma once
/* ── csrc/ops.h ──────────────────────────────────────────────────────────
 * Host-side launcher declarations for the CUDA kernels. All launchers are
 * asynchronous on `stream`; error checking is launch-time only (see
 * CUDA_CHECK_KERNEL in common.cuh).
 *
 * Volumes are contiguous fp32 [B, D, H, W]; B flattens any leading
 * batch/channel dims (B = 1 for a plain 3-D volume).
 */

#include <cuda_runtime.h>

/* Isotropic 3-D TV: acc += Σ_b Σ_corners sqrt(dd² + dh² + dw² + eps).
 * Corner set per batch: (D−1)·(H−1)·(W−1). acc is a single fp32 device
 * scalar the caller pre-zeroes; normalization happens in Python. */
void tv_loss_iso_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    int B, int D, int H, int W,
    float eps,
    cudaStream_t stream
);

/* Backward of the isotropic TV sum. g_scaled is a single fp32 device scalar
 * holding upstream_grad / N (caller pre-scales); grad_vol is written, not
 * accumulated. */
void tv_loss_iso_3d_grad_cuda(
    const float *d_vol,
    const float *d_g_scaled,
    float       *d_grad_vol,
    int B, int D, int H, int W,
    float eps,
    cudaStream_t stream
);

/* Squared-anisotropic 3-D TV: acc += Σ_b Σ_axes Σ (forward-difference)².
 * Each axis sums over its full complementary index range (matching
 * quantem's tv_vol formulation). acc is a single fp32 device scalar the
 * caller pre-zeroes. */
void tv_loss_sq_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    int B, int D, int H, int W,
    cudaStream_t stream
);

/* Backward of the squared-anisotropic TV sum. g is a single fp32 device
 * scalar holding the raw upstream grad (the factor 2 lives in the kernel);
 * grad_vol is written, not accumulated. */
void tv_loss_sq_3d_grad_cuda(
    const float *d_vol,
    const float *d_g,
    float       *d_grad_vol,
    int B, int D, int H, int W,
    cudaStream_t stream
);

/* Fused TILTED K-Planes feature interpolation (one multiscale level):
 * rotate each point by T matrices, bilinearly sample the 3 planes per
 * rotation (grid_sample align_corners=True / border semantics), Hadamard-
 * multiply, write features. pts [B,3]; R [T,3,3]; grid [3T,H,W,C]
 * CHANNELS-LAST with plane index t*3 + {XY, ZX, YZ}; out [B, T*C] fully
 * written. */
void kplanes_tilted_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    float       *d_out,
    long B, int T, int C, int H, int W,
    cudaStream_t stream
);

/* Backward of the fused interpolation. gout [B, T*C]; ggrid [3T,H,W,C]
 * (channels-last, like grid), gR [T,3,3] and gpts [B,3] are accumulated
 * into (caller pre-zeroes all three). Coordinate gradients are zeroed
 * where the border clip engaged, matching torch's grid_sampler. */
void kplanes_tilted_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    const float *d_gout,
    float       *d_ggrid,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C, int H, int W,
    cudaStream_t stream
);
