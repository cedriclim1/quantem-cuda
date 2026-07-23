#pragma once

#include <cuda_bf16.h>

namespace kplanes_tilted_detail {

template <bool MultiLevel>
__device__ __forceinline__ int level_index() {
    if constexpr (MultiLevel) return (int)blockIdx.y;
    return 0;
}

struct Tap {
    int off[4];
    float w[4];
    float tx, ty;
    float gxm, gym;
};

__device__ __forceinline__ Tap make_tap(float gx, float gy, int H, int W, int C) {
    Tap t;
    float ix = (gx + 1.f) * 0.5f * (float)(W - 1);
    float iy = (gy + 1.f) * 0.5f * (float)(H - 1);
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
    const int x1 = min(x0 + 1, W - 1);
    const int y1 = min(y0 + 1, H - 1);
    t.off[0] = (y0 * W + x0) * C;
    t.off[1] = (y0 * W + x1) * C;
    t.off[2] = (y1 * W + x0) * C;
    t.off[3] = (y1 * W + x1) * C;
    return t;
}

__device__ __forceinline__ void rotate(
    const float *Rt, float px, float py, float pz, float &rx, float &ry, float &rz
) {
    rx = Rt[0] * px + Rt[1] * py + Rt[2] * pz;
    ry = Rt[3] * px + Rt[4] * py + Rt[5] * pz;
    rz = Rt[6] * px + Rt[7] * py + Rt[8] * pz;
}

__device__ __forceinline__ float grid_value(const float *p, int off) { return p[off]; }

__device__ __forceinline__ float grid_value(const __nv_bfloat16 *p, int off) {
    return __bfloat162float(p[off]);
}

__device__ __forceinline__ float gout_value(const float *p, long off) { return p[off]; }

__device__ __forceinline__ float gout_value(const __nv_bfloat16 *p, long off) {
    return __bfloat162float(p[off]);
}

__device__ __forceinline__ void stage_rotations(
    const float *R, float *sR, float *sgR, int T
) {
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        sR[i] = R[i];
        if (sgR != nullptr) sgR[i] = 0.f;
    }
}

__device__ __forceinline__ void stage_rotation_columns(
    const float *R, float *sR, float *sRc, int T, float h
) {
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        sR[i] = R[i];
        const int local = i % 9;
        const int t = i / 9;
        const int row = local / 3;
        const int col = local % 3;
        sRc[t * 9 + col * 3 + row] = h * R[t * 9 + row * 3 + col];
    }
}

__device__ __forceinline__ void stage_rotation_columns(
    const float *R, float *sR, float *sRc, float *sgR, int T, float h
) {
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        sR[i] = R[i];
        const int local = i % 9;
        const int t = i / 9;
        const int row = local / 3;
        const int col = local % 3;
        sRc[t * 9 + col * 3 + row] = h * R[t * 9 + row * 3 + col];
        sgR[i] = 0.f;
    }
}

struct Segment {
    int lane;
    int width;
    long group;
    int channel;
    int lane_in_segment;
    bool active;
    unsigned int mask;
};

template <bool NeedMask>
__device__ __forceinline__ Segment make_segment(
    long block_index, long n_groups, int C
) {
    Segment s;
    s.lane = threadIdx.x & 31;
    const long warp_id = (block_index * blockDim.x + threadIdx.x) >> 5;
    s.width = C < 32 ? C : 32;
    if (C <= 32) {
        const int spw = 32 / s.width;
        const int seg = s.lane / s.width;
        s.lane_in_segment = s.lane - seg * s.width;
        s.group = warp_id * spw + seg;
        s.channel = s.lane_in_segment;
        s.active = seg < spw && s.group < n_groups;
        if constexpr (NeedMask) {
            if (seg < spw) {
                const unsigned int low_bits = 0xffffffffu >> (32 - s.width);
                s.mask = low_bits << (seg * s.width);
            } else {
                s.mask = 1u << s.lane;
            }
        }
    } else {
        const int wpg = (C + 31) >> 5;
        s.group = warp_id / wpg;
        s.lane_in_segment = s.lane;
        s.channel = (int)(warp_id - s.group * wpg) * 32 + s.lane;
        s.active = s.group < n_groups && s.channel < C;
        if constexpr (NeedMask) s.mask = 0xffffffffu;
    }
    return s;
}

template <typename GridT>
__device__ __forceinline__ float sample_forward(
    const GridT *grid, int t, int c, int C, int H, int W,
    float rx, float ry, float rz
) {
    const long planeHWC = (long)H * W * C;
    const float gxs[3] = {rx, rz, ry};
    const float gys[3] = {ry, rx, rz};
    float prod = 1.f;
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const Tap tp = make_tap(gxs[p], gys[p], H, W, C);
        const GridT *pc = grid + (long)(t * 3 + p) * planeHWC + c;
        prod *= tp.w[0] * grid_value(pc, tp.off[0])
              + tp.w[1] * grid_value(pc, tp.off[1])
              + tp.w[2] * grid_value(pc, tp.off[2])
              + tp.w[3] * grid_value(pc, tp.off[3]);
    }
    return prod;
}

template <typename GridT>
__device__ __forceinline__ void sample_backward(
    const GridT *grid, float *ggrid, int t, int c, int C, int H, int W,
    float rx, float ry, float rz, float go, float (&dgx)[3], float (&dgy)[3]
) {
    const long planeHWC = (long)H * W * C;
    const float gxs[3] = {rx, rz, ry};
    const float gys[3] = {ry, rx, rz};
    Tap tap[3];
    float s[3], dsdix[3], dsdiy[3];
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        tap[p] = make_tap(gxs[p], gys[p], H, W, C);
        const Tap tp = tap[p];
        const GridT *pc = grid + (long)(t * 3 + p) * planeHWC + c;
        const float v0 = grid_value(pc, tp.off[0]);
        const float v1 = grid_value(pc, tp.off[1]);
        const float v2 = grid_value(pc, tp.off[2]);
        const float v3 = grid_value(pc, tp.off[3]);
        s[p] = v0 * tp.w[0] + v1 * tp.w[1] + v2 * tp.w[2] + v3 * tp.w[3];
        dsdix[p] = (1.f - tp.ty) * (v1 - v0) + tp.ty * (v3 - v2);
        dsdiy[p] = (1.f - tp.tx) * (v2 - v0) + tp.tx * (v3 - v1);
    }
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        const float other = s[(p + 1) % 3] * s[(p + 2) % 3] * go;
        const Tap tp = tap[p];
        float *gpc = ggrid + (long)(t * 3 + p) * planeHWC + c;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float val = other * tp.w[k];
            if (val != 0.f) atomicAdd(&gpc[tp.off[k]], val);
        }
        dgx[p] += other * dsdix[p];
        dgy[p] += other * dsdiy[p];
    }
}

__device__ __forceinline__ void reduce_segment(
    float (&dgx)[3], float (&dgy)[3], int width, int lane_in_segment
) {
#pragma unroll
    for (int p = 0; p < 3; ++p) {
        for (int off = 1; off < width; off <<= 1) {
            const float ox = __shfl_down_sync(0xffffffffu, dgx[p], off);
            const float oy = __shfl_down_sync(0xffffffffu, dgy[p], off);
            if (lane_in_segment + off < width) {
                dgx[p] += ox;
                dgy[p] += oy;
            }
        }
    }
}

__device__ __forceinline__ void rotated_gradient(
    float rx, float ry, float rz, int C, int H, int W,
    const float (&dgx)[3], const float (&dgy)[3],
    float &drx, float &dry, float &drz
) {
    const Tap t0 = make_tap(rx, ry, H, W, C);
    const Tap t1 = make_tap(rz, rx, H, W, C);
    const Tap t2 = make_tap(ry, rz, H, W, C);
    drx = dgx[0] * t0.gxm + dgy[1] * t1.gym;
    dry = dgy[0] * t0.gym + dgx[2] * t2.gxm;
    drz = dgx[1] * t1.gxm + dgy[2] * t2.gym;
}

__device__ __forceinline__ void accumulate_rotation_grads(
    float *sg, float px, float py, float pz, float drx, float dry, float drz
) {
    atomicAdd(&sg[0], drx * px);
    atomicAdd(&sg[1], drx * py);
    atomicAdd(&sg[2], drx * pz);
    atomicAdd(&sg[3], dry * px);
    atomicAdd(&sg[4], dry * py);
    atomicAdd(&sg[5], dry * pz);
    atomicAdd(&sg[6], drz * px);
    atomicAdd(&sg[7], drz * py);
    atomicAdd(&sg[8], drz * pz);
}

__device__ __forceinline__ void accumulate_point_grads(
    const float *Rt, float *gpts, long b, float drx, float dry, float drz
) {
    atomicAdd(&gpts[3 * b], Rt[0] * drx + Rt[3] * dry + Rt[6] * drz);
    atomicAdd(&gpts[3 * b + 1], Rt[1] * drx + Rt[4] * dry + Rt[7] * drz);
    atomicAdd(&gpts[3 * b + 2], Rt[2] * drx + Rt[5] * dry + Rt[8] * drz);
}

__device__ __forceinline__ void flush_rotation_grads(float *sgR, float *gR, int T) {
    for (int i = threadIdx.x; i < T * 9; i += blockDim.x) {
        if (sgR[i] != 0.f) atomicAdd(&gR[i], sgR[i]);
    }
}

} // namespace kplanes_tilted_detail
