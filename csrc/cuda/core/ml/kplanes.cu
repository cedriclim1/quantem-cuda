/* ── csrc/cuda/core/ml/kplanes.cu ───────────────────────────────────────────────────
 * Fused NON-TILTED K-Planes feature interpolation (one multiscale level).
 *
 * Replaces the torch chain in quantem's interpolate_ms_features
 *     stack(plane coords) → F.grid_sample → prod(3 planes) → transpose
 * with one kernel per direction. Same memory design as the tilted kernel
 * (kplanes_tilted.cu): CHANNELS-LAST grids, lane-per-channel parallelism,
 * warp-segment shuffle reduction for the coordinate gradients, branchless
 * border-clamped taps. See that file's header comment for the rationale —
 * this kernel is the tilted one with the rotation machinery removed and
 * quantem's non-tilted plane convention.
 *
 * Plane coordinate pairs differ from the tilted op. quantem's
 * interpolate_ms_features uses mat_mode = [[0,1],[0,2],[1,2]] with
 * grid_sample's (x → W axis, y → H axis) ordering:
 *   p=0: (gx, gy) = (p0, p1)
 *   p=1: (gx, gy) = (p0, p2)
 *   p=2: (gx, gy) = (p1, p2)
 * (The tilted op samples plane 1 at (z, x) — transposed — so the two ops
 * are NOT interchangeable via an identity rotation.)
 *
 * Semantics match F.grid_sample(align_corners=True, padding_mode="border")
 * exactly, gradients included.
 *
 * Layouts (all contiguous fp32):
 *   pts  [B, 3]        points in [-1, 1]³ (border-clamped outside)
 *   grid [3, H, W, C]  channels-last
 *   out  [B, C]        out[b, c] = Π_p sample(plane p, c)
 */

#include "common.cuh"
#include "ops/core/ml.h"

namespace {

struct Tap {
    int off[4];     // corner offsets into a (H, W, C) plane, +1 sides clamped
    float w[4];     // bilinear weights (nw, ne, sw, se); 0 where clamped
    float tx, ty;   // fractional offsets in [0, 1]
    float gxm, gym; // d(ix)/d(gx) including the border-clip mask
};

__device__ __forceinline__ Tap make_tap(float gx, float gy, int H, int W, int C) {
    Tap t;
    float ix = (gx + 1.f) * 0.5f * (float)(W - 1);
    float iy = (gy + 1.f) * 0.5f * (float)(H - 1);
    // torch clip_coordinates_set_grad: gradient is zero at and beyond the
    // border (in <= 0 or in >= size-1), so strict inequalities here.
    t.gxm = (ix > 0.f && ix < (float)(W - 1)) ? 0.5f * (float)(W - 1) : 0.f;
    t.gym = (iy > 0.f && iy < (float)(H - 1)) ? 0.5f * (float)(H - 1) : 0.f;
    ix = fminf(fmaxf(ix, 0.f), (float)(W - 1));
    iy = fminf(fmaxf(iy, 0.f), (float)(H - 1));
    const int x0 = (int)ix;
    const int y0 = (int)iy;
    t.tx = ix - (float)x0;
    t.ty = iy - (float)y0;
    t.w[0] = (1.f - t.tx) * (1.f - t.ty);
    t.w[1] = t.tx * (1.f - t.ty);
    t.w[2] = (1.f - t.tx) * t.ty;
    t.w[3] = t.tx * t.ty;
    const int x1 = min(x0 + 1, W - 1); // clamped ⟺ tx == 0 ⟺ w[1] = w[3] = 0
    const int y1 = min(y0 + 1, H - 1); // clamped ⟺ ty == 0 ⟺ w[2] = w[3] = 0
    t.off[0] = (y0 * W + x0) * C;
    t.off[1] = (y0 * W + x1) * C;
    t.off[2] = (y1 * W + x0) * C;
    t.off[3] = (y1 * W + x1) * C;
    return t;
}

__global__ void kplanes_fwd_kernel(
    const float *__restrict__ pts,
    const float *__restrict__ grid, // [3, H, W, C] channels-last
    float *__restrict__ out,
    long B, int C, int H, int W
) {
    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * (long)C) return;
    const long b = tid / C;
    const int c = (int)(tid - b * C);

    const float p0 = pts[3 * b], p1 = pts[3 * b + 1], p2 = pts[3 * b + 2];

    const long planeHWC = (long)H * W * C;
    const float gxs[3] = {p0, p0, p1};
    const float gys[3] = {p1, p2, p2};

    float prod = 1.f;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const Tap tp = make_tap(gxs[p], gys[p], H, W, C);
        const float *pc = grid + (long)p * planeHWC + c;
        prod *= tp.w[0] * pc[tp.off[0]] + tp.w[1] * pc[tp.off[1]] + tp.w[2] * pc[tp.off[2]]
              + tp.w[3] * pc[tp.off[3]];
    }
    out[tid] = prod; // out[b, c] — tid is exactly that flat index
}

__global__ void kplanes_bwd_kernel(
    const float *__restrict__ pts,
    const float *__restrict__ grid, // [3, H, W, C] channels-last
    const float *__restrict__ gout, // [B, C]
    float *__restrict__ ggrid,      // [3, H, W, C], pre-zeroed
    float *__restrict__ gpts,       // [B, 3], pre-zeroed
    long B, int C, int H, int W
) {
    // A segment = the lanes covering one point's channels (one 32-channel
    // chunk of them when C > 32). Segments never straddle warps; lanes past
    // the last whole segment idle but join the shuffles.
    const int lane = threadIdx.x & 31;
    const long warp_id = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int CW = C < 32 ? C : 32; // channels per segment
    long b;                         // point index
    int c, lis;                     // channel, lane-in-segment
    bool active;
    if (C <= 32) {
        const int spw = 32 / CW; // segments per warp
        const int seg = lane / CW;
        lis = lane - seg * CW;
        b = warp_id * spw + seg;
        c = lis;
        active = seg < spw && b < B;
    } else {
        const int wpg = (C + 31) >> 5; // warps per point
        b = warp_id / wpg;
        lis = lane;
        c = (int)(warp_id - b * wpg) * 32 + lane;
        active = b < B && c < C;
    }
    // Inactive (overhanging) threads still execute the addressing below.
    const long b_safe = b < B ? b : 0;

    const float p0 = pts[3 * b_safe], p1 = pts[3 * b_safe + 1], p2 = pts[3 * b_safe + 2];

    const long planeHWC = (long)H * W * C;
    const float gxs[3] = {p0, p0, p1};
    const float gys[3] = {p1, p2, p2};

    float dgx[3] = {0.f, 0.f, 0.f}; // dL/d(ix_p), pre-unnormalize
    float dgy[3] = {0.f, 0.f, 0.f};

    if (active) {
        Tap tap[3];
        float s[3], dsdix[3], dsdiy[3];
        const float go = gout[b * C + c];
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            tap[p] = make_tap(gxs[p], gys[p], H, W, C);
            const Tap tp = tap[p];
            const float *pc = grid + (long)p * planeHWC + c;
            const float v0 = pc[tp.off[0]], v1 = pc[tp.off[1]];
            const float v2 = pc[tp.off[2]], v3 = pc[tp.off[3]];
            s[p] = v0 * tp.w[0] + v1 * tp.w[1] + v2 * tp.w[2] + v3 * tp.w[3];
            // Clamped +1 corners read the 0-side value instead of torch's
            // guarded 0, but that only changes dsdix/dsdiy exactly where the
            // point sits on the border — where gxm/gym is 0 anyway.
            dsdix[p] = (1.f - tp.ty) * (v1 - v0) + tp.ty * (v3 - v2);
            dsdiy[p] = (1.f - tp.tx) * (v2 - v0) + tp.tx * (v3 - v1);
        }
#pragma unroll
        for (int p = 0; p < 3; ++p) {
            // d(prod)/d(s_p) = product of the other two plane samples
            const float other = s[(p + 1) % 3] * s[(p + 2) % 3] * go;
            const Tap tp = tap[p];
            float *gpc = ggrid + (long)p * planeHWC + c;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float val = other * tp.w[k];
                if (val != 0.f) atomicAdd(&gpc[tp.off[k]], val);
            }
            dgx[p] += other * dsdix[p];
            dgy[p] += other * dsdiy[p];
        }
    }

    // Segment-reduce the 6 coordinate-grad sums over channels.
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        for (int off = 1; off < CW; off <<= 1) {
            const float ox = __shfl_down_sync(0xffffffffu, dgx[p], off);
            const float oy = __shfl_down_sync(0xffffffffu, dgy[p], off);
            if (lis + off < CW) {
                dgx[p] += ox;
                dgy[p] += oy;
            }
        }
    }

    if (active && lis == 0) {
        // Map plane-coordinate grads back to the point. Pairs:
        // p=0 (p0,p1), p=1 (p0,p2), p=2 (p1,p2).
        // make_tap is deterministic, so recomputing the masks is exact.
        const Tap t0 = make_tap(gxs[0], gys[0], H, W, C);
        const Tap t1 = make_tap(gxs[1], gys[1], H, W, C);
        const Tap t2 = make_tap(gxs[2], gys[2], H, W, C);
        const float d0 = dgx[0] * t0.gxm + dgx[1] * t1.gxm;
        const float d1 = dgy[0] * t0.gym + dgx[2] * t2.gxm;
        const float d2 = dgy[1] * t1.gym + dgy[2] * t2.gym;
        // One segment per point for C <= 32, but C > 32 splits a point
        // across warps — accumulate.
        atomicAdd(&gpts[3 * b], d0);
        atomicAdd(&gpts[3 * b + 1], d1);
        atomicAdd(&gpts[3 * b + 2], d2);
    }
}

constexpr int kThreads = 256;

inline int n_blocks(long n) { return (int)((n + kThreads - 1) / kThreads); }

} // namespace

void kplanes_fuse_cuda(
    const float *d_pts,
    const float *d_grid,
    float *d_out,
    long B, int C, int H, int W,
    cudaStream_t stream
) {
    if (B == 0) return;
    kplanes_fwd_kernel<<<n_blocks(B * (long)C), kThreads, 0, stream>>>(
        d_pts, d_grid, d_out, B, C, H, W);
    CUDA_CHECK_KERNEL();
}

void kplanes_fuse_grad_cuda(
    const float *d_pts,
    const float *d_grid,
    const float *d_gout,
    float *d_ggrid,
    float *d_gpts,
    long B, int C, int H, int W,
    cudaStream_t stream
) {
    if (B == 0) return;
    long n_warps;
    if (C <= 32) {
        const int spw = 32 / (C < 32 ? C : 32);
        n_warps = (B + spw - 1) / spw;
    } else {
        n_warps = B * ((C + 31) >> 5);
    }
    kplanes_bwd_kernel<<<n_blocks(n_warps * 32), kThreads, 0, stream>>>(
        d_pts, d_grid, d_gout, d_ggrid, d_gpts, B, C, H, W);
    CUDA_CHECK_KERNEL();
}
