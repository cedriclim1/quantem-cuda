/* ── csrc/cuda/core/tv_loss.cu ────────────────────────────────────────────────
 * Fused 3-D Total-Variation losses + analytic gradients, batched over a
 * flattened leading dim B (channels and/or batch; B = 1 for a plain
 * volume). Two variants:
 *
 * Isotropic (corner-driven, half-pixel-shifted formulation):
 *
 *   sum(vol) = Σ_n Σ_{i,j,k} sqrt( dd² + dh² + dw² + eps )
 *
 *   dd(n,i,j,k) = vol[n, i+1, j,   k  ] − vol[n, i, j, k]
 *   dh(n,i,j,k) = vol[n, i,   j+1, k  ] − vol[n, i, j, k]
 *   dw(n,i,j,k) = vol[n, i,   j,   k+1] − vol[n, i, j, k]
 *
 *   with i ∈ [0, D−2], j ∈ [0, H−2], k ∈ [0, W−2]. The corner count per
 *   batch is (D−1)·(H−1)·(W−1); the 1/N normalization is applied by the
 *   Python caller.
 *
 * Squared-anisotropic (matches quantem's tv_vol formulation):
 *
 *   sum(vol) = Σ_n [ Σ_{i<D−1,j,k} dd² + Σ_{i,j<H−1,k} dh² + Σ_{i,j,k<W−1} dw² ]
 *
 *   i.e. each axis sums its forward differences over the full
 *   complementary index range — no corner restriction, no sqrt.
 *
 * Forward kernels reduce block-locally in shared memory → one atomicAdd
 * per block into a single fp32 accumulator. Backward kernels are
 * voxel-driven gathers that recompute differences from `vol` and write
 * (not accumulate) grad_vol; callers pre-zero acc and may rely on every
 * voxel being written by the gradient kernels.
 *
 * The z grid dimension linearizes (batch, depth) and is grid-strided, so
 * B·D may exceed the 65535 gridDim.z cap.
 */

#include "common.cuh"
#include "ops/core.h"

/* ── isotropic forward ────────────────────────────────────────────────── */

__global__ static void tv_loss_iso_3d_kernel(
    const float *__restrict__ vol,    /* [B, D, H, W] */
    float       *__restrict__ acc,    /* [1] — Σ s, atomicAdd target */
    int B, int D, int H, int W,
    float eps
) {
    /* Block layout: (blockDim.x, blockDim.y, 1) over (k_corner, j_corner);
     * blockIdx.z grid-strides over linearized (batch, i_corner). Threads
     * outside the corner grid contribute 0 so the block reduction is
     * straightforward. */
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    long long n_slices = (long long)B * (D - 1);
    size_t plane = (size_t)H * W;

    float local = 0.0f;
    if (k < W - 1 && j < H - 1) {
        for (long long z = blockIdx.z; z < n_slices; z += gridDim.z) {
            int n = (int)(z / (D - 1));
            int i = (int)(z % (D - 1));
            size_t base = ((size_t)n * D + i) * plane + (size_t)j * W + k;
            float v000 = vol[base];
            float vp_d = vol[base + plane];        /* vol[n, i+1, j,   k  ] */
            float vp_h = vol[base + (size_t)W];    /* vol[n, i,   j+1, k  ] */
            float vp_w = vol[base + 1];            /* vol[n, i,   j,   k+1] */

            float dd = vp_d - v000;
            float dh = vp_h - v000;
            float dw = vp_w - v000;
            local += sqrtf(dd * dd + dh * dh + dw * dw + eps);
        }
    }

    /* Block-level reduction in shared memory. blockDim.x·blockDim.y is a
     * power of 2 (we launch 32×4×1 = 128 threads; 32 is required for
     * warp-level coalescing of vol reads anyway). */
    extern __shared__ float smem[];
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    smem[tid] = local;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(acc, smem[0]);
}

void tv_loss_iso_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    int B, int D, int H, int W,
    float eps,
    cudaStream_t stream
) {
    if (D < 2 || H < 2 || W < 2) {
        /* No corners; caller's d_acc is already zero — leave it. */
        return;
    }

    long long n_slices = (long long)B * (D - 1);
    dim3 block(32, 4, 1);
    dim3 grid(((W - 1) + block.x - 1) / block.x,
              ((H - 1) + block.y - 1) / block.y,
              (unsigned)(n_slices < 65535 ? n_slices : 65535));

    size_t smem_bytes = block.x * block.y * sizeof(float);
    tv_loss_iso_3d_kernel<<<grid, block, smem_bytes, stream>>>(
        d_vol, d_acc, B, D, H, W, eps
    );
    CUDA_CHECK_KERNEL();
}

/* ── isotropic backward ───────────────────────────────────────────────── */

__device__ __forceinline__ float corner_inv_s(
    const float *__restrict__ vol_n,   /* batch-offset volume [D, H, W] */
    int i, int j, int k,
    int H, int W,
    float eps,
    float &dd, float &dh, float &dw
) {
    size_t base = ((size_t)i * H + j) * W + k;
    float v000 = vol_n[base];
    dd = vol_n[base + (size_t)H * W] - v000;
    dh = vol_n[base + (size_t)W]     - v000;
    dw = vol_n[base + 1]             - v000;
    return rsqrtf(dd * dd + dh * dh + dw * dw + eps);
}

__global__ static void tv_loss_iso_3d_grad_kernel(
    const float *__restrict__ vol,         /* [B, D, H, W] */
    const float *__restrict__ g_scaled,    /* [1] = grad_out · 1/N */
    float       *__restrict__ grad_vol,    /* [B, D, H, W] */
    int B, int D, int H, int W,
    float eps
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;   /* k axis (W) */
    int b = blockIdx.y * blockDim.y + threadIdx.y;   /* j axis (H) */
    if (b >= H || c >= W) return;

    float g = __ldg(g_scaled);
    long long n_slabs = (long long)B * D;
    size_t plane = (size_t)H * W;

    for (long long z = (long long)blockIdx.z * blockDim.z + threadIdx.z;
         z < n_slabs;
         z += (long long)gridDim.z * blockDim.z) {
        int n = (int)(z / D);
        int a = (int)(z % D);
        const float *vol_n = vol + (size_t)n * D * plane;

        float acc = 0.0f;
        float dd, dh, dw;

        /* Corner anchored at (a, b, c) — vol[a,b,c] is the "central" voxel.
         * Contribution: -(dd + dh + dw) / s. */
        if (a < D - 1 && b < H - 1 && c < W - 1) {
            float inv_s = corner_inv_s(vol_n, a, b, c, H, W, eps, dd, dh, dw);
            acc += -(dd + dh + dw) * inv_s;
        }
        /* Corner anchored at (a-1, b, c) — vol[a,b,c] is the +d neighbor.
         * Contribution: dd / s. */
        if (a >= 1 && b < H - 1 && c < W - 1) {
            float inv_s = corner_inv_s(vol_n, a - 1, b, c, H, W, eps, dd, dh, dw);
            acc += dd * inv_s;
        }
        /* Corner anchored at (a, b-1, c) — vol[a,b,c] is the +h neighbor.
         * Contribution: dh / s. */
        if (a < D - 1 && b >= 1 && c < W - 1) {
            float inv_s = corner_inv_s(vol_n, a, b - 1, c, H, W, eps, dd, dh, dw);
            acc += dh * inv_s;
        }
        /* Corner anchored at (a, b, c-1) — vol[a,b,c] is the +w neighbor.
         * Contribution: dw / s. */
        if (a < D - 1 && b < H - 1 && c >= 1) {
            float inv_s = corner_inv_s(vol_n, a, b, c - 1, H, W, eps, dd, dh, dw);
            acc += dw * inv_s;
        }

        grad_vol[((size_t)n * D + a) * plane + (size_t)b * W + c] = g * acc;
    }
}

void tv_loss_iso_3d_grad_cuda(
    const float *d_vol,
    const float *d_g_scaled,
    float       *d_grad_vol,
    int B, int D, int H, int W,
    float eps,
    cudaStream_t stream
) {
    if (D < 2 || H < 2 || W < 2) {
        /* No corners — grad is zero everywhere; caller pre-zeroed. */
        return;
    }

    long long n_slabs = (long long)B * D;
    dim3 block(32, 4, 2);
    long long grid_z = (n_slabs + block.z - 1) / block.z;
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              (unsigned)(grid_z < 65535 ? grid_z : 65535));

    tv_loss_iso_3d_grad_kernel<<<grid, block, 0, stream>>>(
        d_vol, d_g_scaled, d_grad_vol, B, D, H, W, eps
    );
    CUDA_CHECK_KERNEL();
}

/* ── squared-anisotropic forward ──────────────────────────────────────── */

__global__ static void tv_loss_sq_3d_kernel(
    const float *__restrict__ vol,    /* [B, D, H, W] */
    float       *__restrict__ acc,    /* [1] — Σ diff², atomicAdd target */
    int B, int D, int H, int W
) {
    /* Voxel-driven: each thread owns the forward differences anchored at
     * its voxel (one per axis where in range), so every term is counted
     * exactly once. */
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;

    long long n_slabs = (long long)B * D;
    size_t plane = (size_t)H * W;

    float local = 0.0f;
    if (b < H && c < W) {
        for (long long z = (long long)blockIdx.z * blockDim.z + threadIdx.z;
             z < n_slabs;
             z += (long long)gridDim.z * blockDim.z) {
            int n = (int)(z / D);
            int a = (int)(z % D);
            size_t base = ((size_t)n * D + a) * plane + (size_t)b * W + c;
            float v = vol[base];

            if (a < D - 1) {
                float dd = vol[base + plane] - v;
                local += dd * dd;
            }
            if (b < H - 1) {
                float dh = vol[base + (size_t)W] - v;
                local += dh * dh;
            }
            if (c < W - 1) {
                float dw = vol[base + 1] - v;
                local += dw * dw;
            }
        }
    }

    /* Block-level reduction (32·4·2 = 256 threads, power of 2). */
    extern __shared__ float smem[];
    int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y * blockDim.z;
    smem[tid] = local;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if (tid == 0) atomicAdd(acc, smem[0]);
}

void tv_loss_sq_3d_cuda(
    const float *d_vol,
    float       *d_acc,
    int B, int D, int H, int W,
    cudaStream_t stream
) {
    long long n_slabs = (long long)B * D;
    dim3 block(32, 4, 2);
    long long grid_z = (n_slabs + block.z - 1) / block.z;
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              (unsigned)(grid_z < 65535 ? grid_z : 65535));

    size_t smem_bytes = block.x * block.y * block.z * sizeof(float);
    tv_loss_sq_3d_kernel<<<grid, block, smem_bytes, stream>>>(
        d_vol, d_acc, B, D, H, W
    );
    CUDA_CHECK_KERNEL();
}

/* ── squared-anisotropic backward ─────────────────────────────────────── */

__global__ static void tv_loss_sq_3d_grad_kernel(
    const float *__restrict__ vol,         /* [B, D, H, W] */
    const float *__restrict__ g,           /* [1] — raw upstream grad */
    float       *__restrict__ grad_vol,    /* [B, D, H, W] */
    int B, int D, int H, int W
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int b = blockIdx.y * blockDim.y + threadIdx.y;
    if (b >= H || c >= W) return;

    float gv = __ldg(g);
    long long n_slabs = (long long)B * D;
    size_t plane = (size_t)H * W;

    for (long long z = (long long)blockIdx.z * blockDim.z + threadIdx.z;
         z < n_slabs;
         z += (long long)gridDim.z * blockDim.z) {
        int n = (int)(z / D);
        int a = (int)(z % D);
        size_t base = ((size_t)n * D + a) * plane + (size_t)b * W + c;
        float v = vol[base];

        /* ∂/∂v Σ diff² = 2·[ Σ_axes (v − prev) − (next − v) ], each term
         * present only where the corresponding difference exists. */
        float acc = 0.0f;
        if (a < D - 1) acc -= vol[base + plane] - v;
        if (a >= 1)    acc += v - vol[base - plane];
        if (b < H - 1) acc -= vol[base + (size_t)W] - v;
        if (b >= 1)    acc += v - vol[base - (size_t)W];
        if (c < W - 1) acc -= vol[base + 1] - v;
        if (c >= 1)    acc += v - vol[base - 1];

        grad_vol[base] = gv * 2.0f * acc;
    }
}

void tv_loss_sq_3d_grad_cuda(
    const float *d_vol,
    const float *d_g,
    float       *d_grad_vol,
    int B, int D, int H, int W,
    cudaStream_t stream
) {
    long long n_slabs = (long long)B * D;
    dim3 block(32, 4, 2);
    long long grid_z = (n_slabs + block.z - 1) / block.z;
    dim3 grid((W + block.x - 1) / block.x,
              (H + block.y - 1) / block.y,
              (unsigned)(grid_z < 65535 ? grid_z : 65535));

    tv_loss_sq_3d_grad_kernel<<<grid, block, 0, stream>>>(
        d_vol, d_g, d_grad_vol, B, D, H, W
    );
    CUDA_CHECK_KERNEL();
}
