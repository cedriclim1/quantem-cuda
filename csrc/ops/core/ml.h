#pragma once
/* ── csrc/ops/core/ml.h ──────────────────────────────────────────────────
 * Host-side launcher declarations for the core.ml CUDA kernels (K-Planes /
 * tensor-decomposition models). All launchers are asynchronous on
 * `stream`; error checking is launch-time only (see CUDA_CHECK_KERNEL in
 * common.cuh).
 */

#include <cuda_runtime.h>

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
