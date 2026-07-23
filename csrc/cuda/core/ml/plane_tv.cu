/* Fused plane-wise 2-D squared-TV loss for three multiscale K-Planes grids.
 *
 * Each logical NCHW grid is presented in channels-last physical order as
 * [3*T, H, W, C]. For every plane independently, the loss is
 *
 *   mean((x[..., 1:, :] - x[..., :-1, :])^2)
 * + mean((x[..., :, 1:] - x[..., :, :-1])^2),
 *
 * summed over the three planes and averaged over T rotations. The forward
 * uses a deterministic two-pass reduction: one fused transform/reduction
 * launch over all three levels, followed by one fixed-order scalar reduction.
 * The backward is voxel-driven and writes every grid gradient exactly once.
 */

#include "common.cuh"
#include "ops/core/ml.h"

#include <math_constants.h>

namespace {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value, unsigned mask, int width) {
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(mask, value, offset);
    }
    return value;
}

__device__ __forceinline__ float block_sum(float value) {
    __shared__ float warp_sums[kBlockSize / kWarpSize];
    constexpr int kNumWarps = kBlockSize / kWarpSize;

    int lane = threadIdx.x & (kWarpSize - 1);
    int warp = threadIdx.x / kWarpSize;
    value = warp_sum(value, 0xffffffffu, kWarpSize);
    if (lane == 0) warp_sums[warp] = value;
    __syncthreads();

    float total = 0.0f;
    if (warp == 0 && lane < kNumWarps) {
        total = warp_sums[lane];
        total = warp_sum(total, (1u << kNumWarps) - 1u, kNumWarps);
    }
    return total;
}

__device__ __forceinline__ void select_level(
    int level,
    const float *grid0, const float *grid1, const float *grid2,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    const float *&grid, int &P, int &C, int &H, int &W
) {
    if (level == 0) {
        grid = grid0; P = P0; C = C0; H = H0; W = W0;
    } else if (level == 1) {
        grid = grid1; P = P1; C = C1; H = H1; W = W1;
    } else {
        grid = grid2; P = P2; C = C2; H = H2; W = W2;
    }
}

__global__ void plane_tv_partials_kernel(
    const float *__restrict__ grid0,
    const float *__restrict__ grid1,
    const float *__restrict__ grid2,
    float *__restrict__ partials,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations
) {
    int level = blockIdx.y;
    const float *grid;
    int P, C, H, W;
    select_level(
        level, grid0, grid1, grid2,
        P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
        grid, P, C, H, W
    );

    size_t count = (size_t)P * H * W * C;
    size_t row_stride = (size_t)W * C;
    size_t step = (size_t)gridDim.x * blockDim.x;
    float inv_rotations = 1.0f / (float)rotations;
    float inv_h = H > 1
        ? inv_rotations / ((float)C * (H - 1) * W)
        : 0.0f;
    float inv_w = W > 1
        ? inv_rotations / ((float)C * H * (W - 1))
        : 0.0f;

    float local = 0.0f;
    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += step) {
        size_t logical = index / C;
        int w = (int)(logical % W);
        int h = (int)((logical / W) % H);
        float value = grid[index];
        if (h + 1 < H) {
            float diff = grid[index + row_stride] - value;
            local = fmaf(diff, diff * inv_h, local);
        }
        if (w + 1 < W) {
            float diff = grid[index + C] - value;
            local = fmaf(diff, diff * inv_w, local);
        }
    }

    float total = block_sum(local);
    if (threadIdx.x == 0) {
        partials[(size_t)level * gridDim.x + blockIdx.x] = total;
    }
}

__global__ void plane_tv_finalize_kernel(
    const float *__restrict__ partials,
    float *__restrict__ output,
    int partial_count,
    bool has_empty_difference_axis
) {
    float local = 0.0f;
    for (int index = threadIdx.x; index < partial_count; index += blockDim.x) {
        local += partials[index];
    }
    float total = block_sum(local);
    if (threadIdx.x == 0) {
        // torch.mean over an empty difference tensor returns NaN.
        output[0] = has_empty_difference_axis ? CUDART_NAN_F : total;
    }
}

template <bool Accumulate>
__global__ void plane_tv_grad_kernel(
    const float *__restrict__ grid0,
    const float *__restrict__ grid1,
    const float *__restrict__ grid2,
    const float *__restrict__ grad_output,
    float *__restrict__ grad_grid0,
    float *__restrict__ grad_grid1,
    float *__restrict__ grad_grid2,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations
) {
    int level = blockIdx.y;
    const float *grid;
    int P, C, H, W;
    select_level(
        level, grid0, grid1, grid2,
        P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
        grid, P, C, H, W
    );
    float *grad_grid = level == 0 ? grad_grid0 : (level == 1 ? grad_grid1 : grad_grid2);

    size_t count = (size_t)P * H * W * C;
    size_t row_stride = (size_t)W * C;
    size_t step = (size_t)gridDim.x * blockDim.x;
    float upstream = __ldg(grad_output);
    float rotation_scale = 2.0f * upstream / (float)rotations;
    float scale_h = H > 1 ? rotation_scale / ((float)C * (H - 1) * W) : 0.0f;
    float scale_w = W > 1 ? rotation_scale / ((float)C * H * (W - 1)) : 0.0f;

    for (size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += step) {
        size_t logical = index / C;
        int w = (int)(logical % W);
        int h = (int)((logical / W) % H);
        float value = grid[index];
        float grad = 0.0f;

        if (h > 0) grad += (value - grid[index - row_stride]) * scale_h;
        if (h + 1 < H) grad += (value - grid[index + row_stride]) * scale_h;
        if (w > 0) grad += (value - grid[index - C]) * scale_w;
        if (w + 1 < W) grad += (value - grid[index + C]) * scale_w;

        if constexpr (Accumulate) {
            grad_grid[index] += grad;
        } else {
            grad_grid[index] = grad;
        }
    }
}

}  // namespace

void plane_tv_loss_cuda(
    const float *d_grid0, const float *d_grid1, const float *d_grid2,
    float *d_partials, float *d_output,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations, int num_blocks,
    cudaStream_t stream
) {
    dim3 grid(num_blocks, 3, 1);
    plane_tv_partials_kernel<<<grid, kBlockSize, 0, stream>>>(
        d_grid0, d_grid1, d_grid2, d_partials,
        P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
        rotations
    );
    CUDA_CHECK_KERNEL();

    bool has_empty_axis = H0 < 2 || W0 < 2 || H1 < 2 || W1 < 2 || H2 < 2 || W2 < 2;
    plane_tv_finalize_kernel<<<1, kBlockSize, 0, stream>>>(
        d_partials, d_output, 3 * num_blocks, has_empty_axis
    );
    CUDA_CHECK_KERNEL();
}

void plane_tv_loss_grad_cuda(
    const float *d_grid0, const float *d_grid1, const float *d_grid2,
    const float *d_grad_output,
    float *d_grad_grid0, float *d_grad_grid1, float *d_grad_grid2,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations, int num_blocks, bool accumulate,
    cudaStream_t stream
) {
    dim3 grid(num_blocks, 3, 1);
    if (accumulate) {
        plane_tv_grad_kernel<true><<<grid, kBlockSize, 0, stream>>>(
            d_grid0, d_grid1, d_grid2, d_grad_output,
            d_grad_grid0, d_grad_grid1, d_grad_grid2,
            P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
            rotations
        );
    } else {
        plane_tv_grad_kernel<false><<<grid, kBlockSize, 0, stream>>>(
            d_grid0, d_grid1, d_grid2, d_grad_output,
            d_grad_grid0, d_grad_grid1, d_grad_grid2,
            P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
            rotations
        );
    }
    CUDA_CHECK_KERNEL();
}
