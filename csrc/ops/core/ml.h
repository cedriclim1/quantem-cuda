#pragma once
/* ── csrc/ops/core/ml.h ──────────────────────────────────────────────────
 * Host-side launcher declarations for the core.ml CUDA kernels (K-Planes /
 * tensor-decomposition models). All launchers are asynchronous on
 * `stream`; error checking is launch-time only (see CUDA_CHECK_KERNEL in
 * common.cuh).
 */

#include <cuda_runtime.h>

/* Three-level plane-wise 2-D squared-TV loss. Grids are fp32 channels-last
 * [3*T,H,W,C]. Forward writes one scalar after applying each plane's distinct
 * H/W mean denominator and averaging over T. Backward fully writes all three
 * fp32 gradient grids. */
void plane_tv_loss_cuda(
    const float *d_grid0, const float *d_grid1, const float *d_grid2,
    float *d_partials, float *d_output,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations, int num_blocks,
    cudaStream_t stream
);

void plane_tv_loss_grad_cuda(
    const float *d_grid0, const float *d_grid1, const float *d_grid2,
    const float *d_grad_output,
    float *d_grad_grid0, float *d_grad_grid1, float *d_grad_grid2,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations, int num_blocks, bool accumulate,
    cudaStream_t stream
);

/* Fused trunc-exp density tail. Logical tensors have one or two dimensions;
 * explicit element strides support non-contiguous views. The forward computes
 * exp(values - offset); backward computes grad_output * exp(min(values -
 * offset, 15)). Input/output storage is fp32 or bf16 as selected by is_bf16. */
void density_tail_cuda(
    const void *d_values,
    void *d_output,
    long rows, long cols,
    long value_stride0, long value_stride1,
    long output_stride0, long output_stride1,
    float offset, bool is_bf16,
    cudaStream_t stream
);

void density_tail_grad_cuda(
    const void *d_values,
    const void *d_grad_output,
    void *d_grad_values,
    long rows, long cols,
    long value_stride0, long value_stride1,
    long grad_output_stride0, long grad_output_stride1,
    long grad_value_stride0, long grad_value_stride1,
    float offset, bool is_bf16,
    cudaStream_t stream
);

/* Fused TILTED K-Planes feature interpolation (one multiscale level):
 * rotate each point by T matrices, bilinearly sample the 3 planes per
 * rotation (grid_sample align_corners=True / border semantics), Hadamard-
 * multiply, write features. pts [B,3]; R [T,3,3]; grid [3T,H,W,C] fp32/bf16
 * CHANNELS-LAST with plane index t*3 + {XY, ZX, YZ}; out [B, T*C] fully
 * written. */
void kplanes_tilted_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const void  *d_grid,
    float       *d_out,
    long B, int T, int C, int H, int W,
    bool grid_is_bf16,
    cudaStream_t stream
);

/* Backward of the fused interpolation. gout [B, T*C]; grid may be fp32/bf16;
 * ggrid [3T,H,W,C] is always fp32
 * (channels-last, like grid), gR [T,3,3] and gpts [B,3] are accumulated
 * into (caller pre-zeroes all three). Coordinate gradients are zeroed
 * where the border clip engaged, matching torch's grid_sampler. */
void kplanes_tilted_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const void  *d_grid,
    const float *d_gout,
    float       *d_ggrid,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C, int H, int W,
    bool grid_is_bf16,
    cudaStream_t stream
);

/* Three-level multiscale interpolation. One 2-D launch partitions blocks by
 * level (grid.y) and writes directly into one fp32/bf16
 * [B, sum_l T*C_l] output. */
void kplanes_tilted_fuse_ms_cuda(
    const float *d_pts,
    const float *d_R,
    const void  *d_grid0,
    const void  *d_grid1,
    const void  *d_grid2,
    void        *d_out,
    long B, int T,
    int C0, int H0, int W0,
    int C1, int H1, int W1,
    int C2, int H2, int W2,
    float scale0, float scale1, float scale2,
    bool grid_is_bf16,
    bool output_is_bf16,
    cudaStream_t stream
);

/* Backward of the three-level op. gout is an fp32/bf16 inner-contiguous 2-D view;
 * gout_row_stride is its element stride between rows. The three level
 * offsets are derived from T*C_l. Each ggrid_l is dense fp32. */
void kplanes_tilted_fuse_ms_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const void  *d_grid0,
    const void  *d_grid1,
    const void  *d_grid2,
    const void  *d_gout,
    float       *d_ggrid0,
    float       *d_ggrid1,
    float       *d_ggrid2,
    float       *d_gR,
    float       *d_gpts,
    long B, int T,
    int C0, int H0, int W0,
    int C1, int H1, int W1,
    int C2, int H2, int W2,
    long gout_row_stride,
    float scale0, float scale1, float scale2,
    bool grid_is_bf16,
    bool gout_is_bf16,
    cudaStream_t stream
);

/* TV-specialized fused TILTED K-Planes forward.
 * Evaluates the base computation at 4 tap locations per point:
 *   tap 0: R·x          (base point)
 *   tap 1: R·x + h·R[:,0]  (x + h·ex after rotation)
 *   tap 2: R·x + h·R[:,1]  (x + h·ey after rotation)
 *   tap 3: R·x + h·R[:,2]  (x + h·ez after rotation)
 * pts [B,3]; R [T,3,3]; grid [3T,H,W,C] channels-last;
 * out [4, B, T*C] — tap dim is outermost so out[0] is bit-identical
 * in layout to kplanes_tilted_fuse output. */
void kplanes_tilted_tv_fuse_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    float       *d_out,
    long B, int T, int C, int H, int W,
    float h,
    cudaStream_t stream
);

/* Backward of the TV fused interpolation.
 * gout [4, B, T*C]; ggrid [3T,H,W,C], gR [T,3,3], gpts [B,3] accumulated
 * (caller pre-zeroes). h must equal the value used in the forward. */
void kplanes_tilted_tv_fuse_grad_cuda(
    const float *d_pts,
    const float *d_R,
    const float *d_grid,
    const float *d_gout,
    float       *d_ggrid,
    float       *d_gR,
    float       *d_gpts,
    long B, int T, int C, int H, int W,
    float h,
    cudaStream_t stream
);
