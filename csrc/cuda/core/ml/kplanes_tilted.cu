/* Fused TILTED K-Planes interpolation.
 *
 * Grids are channels-last [3T,H,W,C]. A thread owns one channel of a
 * (point, rotation) group; backward coordinate gradients are reduced over
 * warp-local channel segments. Single-level and three-level launches share
 * the same templated entry points, while compile-time policies preserve the
 * baseline, exact-zero ballot, and threshold-ballot backward variants.
 */

#include "common.cuh"
#include "kplanes_tilted_common.cuh"
#include "ops/core/ml.h"

#include <cstdlib>
#include <stdexcept>

namespace {

using namespace kplanes_tilted_detail;

struct None {
    static constexpr bool enabled = false;
    static constexpr bool magnitude = false;
};

// Magnitude=false is V3/V4's `go != 0`; Magnitude=true is V5's
// `fabsf(go) > tau`. Keeping these separate preserves NaN behavior exactly.
template <bool Magnitude>
struct ThresholdBallot {
    static constexpr bool enabled = true;
    static constexpr bool magnitude = Magnitude;

    __device__ __forceinline__ static bool keep(
        unsigned int mask, float go, float tau
    ) {
        if constexpr (Magnitude) {
            return __ballot_sync(mask, fabsf(go) > tau) != 0u;
        } else {
            return __ballot_sync(mask, go != 0.f) != 0u;
        }
    }
};

template <typename GridT, typename OutT, bool MultiLevel>
__global__ void kplanes_tilted_fwd_core(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const GridT *__restrict__ grid0,
    const GridT *__restrict__ grid1,
    const GridT *__restrict__ grid2,
    OutT *__restrict__ out,
    long B, int T,
    int C0, int H0, int W0,
    int C1, int H1, int W1,
    int C2, int H2, int W2,
    float scale0, float scale1, float scale2
) {
    const int level = level_index<MultiLevel>();
    const GridT *grid = grid0;
    int C = C0, H = H0, W = W0;
    long out_col_offset = 0;
    float scale = 1.f;
    if constexpr (MultiLevel) {
        scale = scale0;
        if (level == 1) {
            grid = grid1;
            C = C1;
            H = H1;
            W = W1;
            out_col_offset = (long)T * C0;
            scale = scale1;
        } else if (level == 2) {
            grid = grid2;
            C = C2;
            H = H2;
            W = W2;
            out_col_offset = (long)T * (C0 + C1);
            scale = scale2;
        }
    }

    const long level_elements = B * (long)(T * C);
    if ((long)blockIdx.x * blockDim.x >= level_elements) return;

    extern __shared__ float sR[];
    stage_rotations(R, sR, nullptr, T);
    __syncthreads();

    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= level_elements) return;
    const long b = tid / (T * C);
    const int rem = (int)(tid - b * (T * C));
    const int t = rem / C;
    const int c = rem - t * C;
    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
    float rx, ry, rz;
    rotate(sR + t * 9, px, py, pz, rx, ry, rz);
    const long out_row_stride = MultiLevel
        ? (long)T * (C0 + C1 + C2)
        : (long)T * C0;
    out[b * out_row_stride + out_col_offset + rem] =
        sample_forward(grid, t, c, C, H, W, rx, ry, rz) * scale;
}

template <typename GridT, typename GoutT, class EarlyOut, bool MultiLevel>
__global__ void kplanes_tilted_bwd_core(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const GridT *__restrict__ grid0,
    const GridT *__restrict__ grid1,
    const GridT *__restrict__ grid2,
    const GoutT *__restrict__ gout,
    float *__restrict__ ggrid0,
    float *__restrict__ ggrid1,
    float *__restrict__ ggrid2,
    float *__restrict__ gR,
    float *__restrict__ gpts,
    long B, int T,
    int C0, int H0, int W0,
    int C1, int H1, int W1,
    int C2, int H2, int W2,
    long gout_row_stride,
    float scale0, float scale1, float scale2,
    float zero_tau
) {
    const int level = level_index<MultiLevel>();
    const GridT *grid = grid0;
    float *ggrid = ggrid0;
    int C = C0, H = H0, W = W0;
    long gout_col_offset = 0;
    float scale = 1.f;
    if constexpr (MultiLevel) {
        scale = scale0;
        if (level == 1) {
            grid = grid1;
            ggrid = ggrid1;
            C = C1;
            H = H1;
            W = W1;
            gout_col_offset = (long)T * C0;
            scale = scale1;
        } else if (level == 2) {
            grid = grid2;
            ggrid = ggrid2;
            C = C2;
            H = H2;
            W = W2;
            gout_col_offset = (long)T * (C0 + C1);
            scale = scale2;
        }
    }

    const long row_stride = MultiLevel ? gout_row_stride : (long)T * C0;

    extern __shared__ float smem[];
    float *sR = smem;
    float *sgR = smem + T * 9;
    stage_rotations(R, sR, sgR, T);
    __syncthreads();

    const long n_groups = B * (long)T;
    const Segment segment = make_segment<EarlyOut::enabled>(blockIdx.x, n_groups, C);
    const long g_safe = segment.group < n_groups ? segment.group : 0;
    const long b = g_safe / T;
    const int t = (int)(g_safe - b * T);
    float dgx[3] = {0.f, 0.f, 0.f};
    float dgy[3] = {0.f, 0.f, 0.f};

    if constexpr (!EarlyOut::magnitude) {
        // Baseline and V3/V4 keep the historical live ranges and ordering.
        const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
        const float *Rt = sR + t * 9;
        float rx, ry, rz;
        rotate(Rt, px, py, pz, rx, ry, rz);
        bool keep = true;
        float go = 0.f;
        if constexpr (EarlyOut::enabled) {
            go = segment.active
                ? gout_value(
                      gout, b * row_stride + gout_col_offset + t * C + segment.channel) * scale
                : 0.f;
            keep = EarlyOut::keep(segment.mask, go, zero_tau);
        }
        if (segment.active && keep) {
            if constexpr (!EarlyOut::enabled) {
                go = gout_value(
                         gout, b * row_stride + gout_col_offset + t * C + segment.channel) * scale;
            }
            sample_backward(
                grid, ggrid, t, segment.channel, C, H, W, rx, ry, rz, go, dgx, dgy);
        }
        reduce_segment(dgx, dgy, segment.width, segment.lane_in_segment);
        if (segment.active && keep && segment.lane_in_segment == 0) {
            float drx, dry, drz;
            rotated_gradient(rx, ry, rz, C, H, W, dgx, dgy, drx, dry, drz);
            accumulate_rotation_grads(sgR + t * 9, px, py, pz, drx, dry, drz);
            accumulate_point_grads(Rt, gpts, b, drx, dry, drz);
        }
    } else {
        // V5 scopes dense values tightly, then recomputes leader-only values.
        const float go = segment.active
            ? gout_value(
                  gout, b * row_stride + gout_col_offset + t * C + segment.channel) * scale
            : 0.f;
        if (!EarlyOut::keep(segment.mask, go, zero_tau)) goto threshold_segment_reduction;
        if (segment.active) {
            const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
            float rx, ry, rz;
            rotate(sR + t * 9, px, py, pz, rx, ry, rz);
            sample_backward(
                grid, ggrid, t, segment.channel, C, H, W, rx, ry, rz, go, dgx, dgy);
        }
threshold_segment_reduction:
        reduce_segment(dgx, dgy, segment.width, segment.lane_in_segment);
        if (!EarlyOut::keep(segment.mask, go, zero_tau)) goto threshold_done;
        if (segment.active && segment.lane_in_segment == 0) {
            const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
            const float *Rt = sR + t * 9;
            float rx, ry, rz;
            rotate(Rt, px, py, pz, rx, ry, rz);
            float drx, dry, drz;
            rotated_gradient(rx, ry, rz, C, H, W, dgx, dgy, drx, dry, drz);
            accumulate_rotation_grads(sgR + t * 9, px, py, pz, drx, dry, drz);
            accumulate_point_grads(Rt, gpts, b, drx, dry, drz);
        }
threshold_done:;
    }

    __syncthreads();
    flush_rotation_grads(sgR, gR, T);
}

#if __CUDA_ARCH__ == 750
#define KPLANES_TV_FWD_REGS 47
#elif __CUDA_ARCH__ == 860 || __CUDA_ARCH__ == 1200
#define KPLANES_TV_FWD_REGS 40
#else
#define KPLANES_TV_FWD_REGS 32
#endif
template <typename GridT>
__global__ __maxnreg__(KPLANES_TV_FWD_REGS) void kplanes_tilted_tv_fwd_core(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const GridT *__restrict__ grid,
    float *__restrict__ out,
    long B, int T, int C, int H, int W,
    float h
) {
    extern __shared__ float smem[];
    float *sR = smem;
    float *sRc = smem + T * 9;
    stage_rotation_columns(R, sR, sRc, T, h);
    __syncthreads();

    const long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    const long total = 4L * B * (long)(T * C);
    if (tid >= total) return;
    const int tap = (int)(tid / (B * (long)(T * C)));
    const long rem0 = tid - (long)tap * B * (long)(T * C);
    const long b = rem0 / (T * C);
    const int rem1 = (int)(rem0 - b * (T * C));
    const int t = rem1 / C;
    const int c = rem1 - t * C;
    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
    float rx, ry, rz;
    rotate(sR + t * 9, px, py, pz, rx, ry, rz);
    if (tap >= 1) {
        const float *rc = sRc + t * 9 + (tap - 1) * 3;
        rx += rc[0];
        ry += rc[1];
        rz += rc[2];
    }
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
    out[((long)tap * B + b) * (T * C) + t * C + c] = prod;
}
#undef KPLANES_TV_FWD_REGS

#if __CUDA_ARCH__ == 1200
#define KPLANES_TV_BWD_LIMIT __maxnreg__(78)
#else
#define KPLANES_TV_BWD_LIMIT
#endif
template <typename GridT>
__global__ KPLANES_TV_BWD_LIMIT void kplanes_tilted_tv_bwd_core(
    const float *__restrict__ pts,
    const float *__restrict__ R,
    const GridT *__restrict__ grid,
    const float *__restrict__ gout,
    float *__restrict__ ggrid,
    float *__restrict__ gR,
    float *__restrict__ gpts,
    long B, int T, int C, int H, int W,
    float h
) {
    extern __shared__ float smem[];
    float *sR = smem;
    float *sRc = smem + T * 9;
    float *sgR = smem + T * 18;
    stage_rotation_columns(R, sR, sRc, sgR, T, h);
    __syncthreads();

    const long n_groups = 4L * B * (long)T;
    const Segment segment = make_segment<false>(blockIdx.x, n_groups, C);
    const long g_safe = segment.group < n_groups ? segment.group : 0;
    const int tap = (int)(g_safe / (B * (long)T));
    const long bt = g_safe - (long)tap * B * (long)T;
    const long b = bt / T;
    const int t = (int)(bt - b * T);
    const float px = pts[3 * b], py = pts[3 * b + 1], pz = pts[3 * b + 2];
    const float *Rt = sR + t * 9;
    float rx, ry, rz;
    rotate(Rt, px, py, pz, rx, ry, rz);
    if (tap >= 1) {
        const float *rc = sRc + t * 9 + (tap - 1) * 3;
        rx += rc[0];
        ry += rc[1];
        rz += rc[2];
    }
    float dgx[3] = {0.f, 0.f, 0.f};
    float dgy[3] = {0.f, 0.f, 0.f};
    if (segment.active) {
        const float go = gout[((long)tap * B + b) * (T * C) + t * C + segment.channel];
        sample_backward(
            grid, ggrid, t, segment.channel, C, H, W, rx, ry, rz, go, dgx, dgy);
    }
    reduce_segment(dgx, dgy, segment.width, segment.lane_in_segment);

    if (segment.active && segment.lane_in_segment == 0) {
        float drx, dry, drz;
        rotated_gradient(rx, ry, rz, C, H, W, dgx, dgy, drx, dry, drz);
        float *sg = sgR + t * 9;
        accumulate_rotation_grads(sg, px, py, pz, drx, dry, drz);
        if (tap >= 1) {
            const int axis = tap - 1;
            atomicAdd(&sg[axis], h * drx);
            atomicAdd(&sg[3 + axis], h * dry);
            atomicAdd(&sg[6 + axis], h * drz);
        }
        accumulate_point_grads(Rt, gpts, b, drx, dry, drz);
    }

    __syncthreads();
    flush_rotation_grads(sgR, gR, T);
}
#undef KPLANES_TV_BWD_LIMIT

constexpr int kThreads = 256;

inline int n_blocks(long n) { return (int)((n + kThreads - 1) / kThreads); }

inline long n_warps(long groups, int C) {
    if (C <= 32) {
        const int spw = 32 / (C < 32 ? C : 32);
        return (groups + spw - 1) / spw;
    }
    return groups * ((C + 31) >> 5);
}

struct KplanesTiltedBwdConfig {
    int variant;
    float zero_tau;
};

inline const KplanesTiltedBwdConfig &kplanes_tilted_bwd_config() {
    static const KplanesTiltedBwdConfig config = [] {
        // Production default: V5's threshold ballot with the parity-validated
        // tau. Explicit environment values, including variant 0 and tau 0,
        // restore the requested baseline/exact behavior.
        KplanesTiltedBwdConfig result{5, 6e-8f};
        const char *value = std::getenv("QUANTEM_KPLANES_BWD_VARIANT");
        if (value != nullptr && value[0] != '\0' && value[1] == '\0') {
            if (value[0] == '0') result.variant = 0;
            if (value[0] == '3') result.variant = 3;
            if (value[0] == '4') result.variant = 4;
            if (value[0] == '5') result.variant = 5;
        }
        const char *tau_value = std::getenv("QUANTEM_KPLANES_BWD_ZERO_TAU");
        if (tau_value != nullptr) {
            char *end = nullptr;
            const float parsed = std::strtof(tau_value, &end);
            if (end != tau_value && end[0] == '\0' && parsed >= 0.f) result.zero_tau = parsed;
        }
        return result;
    }();
    return config;
}

template <typename GridT, typename OutT, bool MultiLevel>
inline void launch_fwd(
    dim3 blocks, size_t shmem, cudaStream_t stream,
    const float *pts, const float *R,
    const void *grid0, const void *grid1, const void *grid2,
    void *out, long B, int T,
    int C0, int H0, int W0, int C1, int H1, int W1, int C2, int H2, int W2,
    float scale0, float scale1, float scale2
) {
    kplanes_tilted_fwd_core<GridT, OutT, MultiLevel><<<blocks, kThreads, shmem, stream>>>(
        pts, R, static_cast<const GridT *>(grid0), static_cast<const GridT *>(grid1),
        static_cast<const GridT *>(grid2), static_cast<OutT *>(out), B, T,
        C0, H0, W0, C1, H1, W1, C2, H2, W2, scale0, scale1, scale2);
}

template <typename GridT, typename GoutT, class EarlyOut, bool MultiLevel>
inline void launch_bwd(
    dim3 blocks, size_t shmem, cudaStream_t stream,
    const float *pts, const float *R,
    const void *grid0, const void *grid1, const void *grid2, const void *gout,
    float *ggrid0, float *ggrid1, float *ggrid2, float *gR, float *gpts,
    long B, int T,
    int C0, int H0, int W0, int C1, int H1, int W1, int C2, int H2, int W2,
    long gout_row_stride, float scale0, float scale1, float scale2, float zero_tau
) {
    kplanes_tilted_bwd_core<GridT, GoutT, EarlyOut, MultiLevel><<<
        blocks, kThreads, shmem, stream>>>(
        pts, R, static_cast<const GridT *>(grid0), static_cast<const GridT *>(grid1),
        static_cast<const GridT *>(grid2), static_cast<const GoutT *>(gout),
        ggrid0, ggrid1, ggrid2, gR, gpts, B, T,
        C0, H0, W0, C1, H1, W1, C2, H2, W2, gout_row_stride,
        scale0, scale1, scale2, zero_tau);
}

template <bool MultiLevel, typename GoutT>
struct BwdLaunch {
    dim3 blocks;
    size_t shmem;
    cudaStream_t stream;
    const float *pts;
    const float *R;
    const void *grid0;
    const void *grid1;
    const void *grid2;
    const void *gout;
    float *ggrid0;
    float *ggrid1;
    float *ggrid2;
    float *gR;
    float *gpts;
    long B;
    int T;
    int C0, H0, W0, C1, H1, W1, C2, H2, W2;
    long gout_row_stride;
    float scale0, scale1, scale2, zero_tau;

    template <typename GridT, class EarlyOut>
    void operator()() const {
        launch_bwd<GridT, GoutT, EarlyOut, MultiLevel>(
            blocks, shmem, stream, pts, R, grid0, grid1, grid2, gout,
            ggrid0, ggrid1, ggrid2, gR, gpts, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, gout_row_stride,
            scale0, scale1, scale2, zero_tau);
    }
};

inline void validate_bwd_storage(bool grid_is_bf16, int variant) {
    if (grid_is_bf16 && variant != 4 && variant != 5) {
        throw std::invalid_argument(
            "bf16 kplanes grid requires QUANTEM_KPLANES_BWD_VARIANT=4 or 5 "
            "at process startup");
    }
}

template <typename Launch>
inline void dispatch_bwd(bool grid_is_bf16, int variant, Launch &&launch) {
    switch (variant) {
    case 3:
        launch.template operator()<float, ThresholdBallot<false>>();
        break;
    case 4:
        if (grid_is_bf16) {
            launch.template operator()<__nv_bfloat16, ThresholdBallot<false>>();
        } else {
            launch.template operator()<float, ThresholdBallot<false>>();
        }
        break;
    case 5:
        if (grid_is_bf16) {
            launch.template operator()<__nv_bfloat16, ThresholdBallot<true>>();
        } else {
            launch.template operator()<float, ThresholdBallot<true>>();
        }
        break;
    default:
        launch.template operator()<float, None>();
        break;
    }
}

} // namespace

void kplanes_tilted_fuse_cuda(
    const float *d_pts, const float *d_R, const void *d_grid, float *d_out,
    long B, int T, int C, int H, int W, bool grid_is_bf16, cudaStream_t stream
) {
    if (B == 0) return;
    const dim3 blocks(n_blocks(B * (long)(T * C)), 1);
    const size_t shmem = (size_t)T * 9 * sizeof(float);
    if (grid_is_bf16) {
        launch_fwd<__nv_bfloat16, float, false>(
            blocks, shmem, stream, d_pts, d_R, d_grid, d_grid, d_grid, d_out, B, T,
            C, H, W, 0, 0, 0, 0, 0, 0, 1.f, 1.f, 1.f);
    } else {
        launch_fwd<float, float, false>(
            blocks, shmem, stream, d_pts, d_R, d_grid, d_grid, d_grid, d_out, B, T,
            C, H, W, 0, 0, 0, 0, 0, 0, 1.f, 1.f, 1.f);
    }
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_fuse_grad_cuda(
    const float *d_pts, const float *d_R, const void *d_grid, const float *d_gout,
    float *d_ggrid, float *d_gR, float *d_gpts,
    long B, int T, int C, int H, int W, bool grid_is_bf16, cudaStream_t stream
) {
    if (B == 0) return;
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    const KplanesTiltedBwdConfig &config = kplanes_tilted_bwd_config();
    validate_bwd_storage(grid_is_bf16, config.variant);
    const BwdLaunch<false, float> launch{
        dim3(n_blocks(n_warps(B * (long)T, C) * 32), 1), shmem, stream,
        d_pts, d_R, d_grid, d_grid, d_grid, d_gout,
        d_ggrid, d_ggrid, d_ggrid, d_gR, d_gpts, B, T,
        C, H, W, 0, 0, 0, 0, 0, 0, T * (long)C,
        1.f, 1.f, 1.f, config.zero_tau};
    dispatch_bwd(grid_is_bf16, config.variant, launch);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_fuse_ms_cuda(
    const float *d_pts, const float *d_R,
    const void *d_grid0, const void *d_grid1, const void *d_grid2, void *d_out,
    long B, int T,
    int C0, int H0, int W0, int C1, int H1, int W1, int C2, int H2, int W2,
    float scale0, float scale1, float scale2,
    bool grid_is_bf16, bool output_is_bf16, cudaStream_t stream
) {
    if (B == 0) return;
    int blocks = n_blocks(B * (long)(T * C0));
    blocks = max(blocks, n_blocks(B * (long)(T * C1)));
    blocks = max(blocks, n_blocks(B * (long)(T * C2)));
    const dim3 grid_dim(blocks, 3);
    const size_t shmem = (size_t)T * 9 * sizeof(float);
    if (grid_is_bf16 && output_is_bf16) {
        launch_fwd<__nv_bfloat16, __nv_bfloat16, true>(
            grid_dim, shmem, stream, d_pts, d_R, d_grid0, d_grid1, d_grid2, d_out, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, scale0, scale1, scale2);
    } else if (grid_is_bf16) {
        launch_fwd<__nv_bfloat16, float, true>(
            grid_dim, shmem, stream, d_pts, d_R, d_grid0, d_grid1, d_grid2, d_out, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, scale0, scale1, scale2);
    } else if (output_is_bf16) {
        launch_fwd<float, __nv_bfloat16, true>(
            grid_dim, shmem, stream, d_pts, d_R, d_grid0, d_grid1, d_grid2, d_out, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, scale0, scale1, scale2);
    } else {
        launch_fwd<float, float, true>(
            grid_dim, shmem, stream, d_pts, d_R, d_grid0, d_grid1, d_grid2, d_out, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, scale0, scale1, scale2);
    }
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_fuse_ms_grad_cuda(
    const float *d_pts, const float *d_R,
    const void *d_grid0, const void *d_grid1, const void *d_grid2, const void *d_gout,
    float *d_ggrid0, float *d_ggrid1, float *d_ggrid2, float *d_gR, float *d_gpts,
    long B, int T,
    int C0, int H0, int W0, int C1, int H1, int W1, int C2, int H2, int W2,
    long gout_row_stride, float scale0, float scale1, float scale2,
    bool grid_is_bf16, bool gout_is_bf16, cudaStream_t stream
) {
    if (B == 0) return;
    int blocks = n_blocks(n_warps(B * (long)T, C0) * 32);
    blocks = max(blocks, n_blocks(n_warps(B * (long)T, C1) * 32));
    blocks = max(blocks, n_blocks(n_warps(B * (long)T, C2) * 32));
    const dim3 grid_dim(blocks, 3);
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    const KplanesTiltedBwdConfig &config = kplanes_tilted_bwd_config();
    validate_bwd_storage(grid_is_bf16, config.variant);
    if (gout_is_bf16) {
        const BwdLaunch<true, __nv_bfloat16> launch{
            grid_dim, shmem, stream, d_pts, d_R, d_grid0, d_grid1, d_grid2, d_gout,
            d_ggrid0, d_ggrid1, d_ggrid2, d_gR, d_gpts, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, gout_row_stride,
            scale0, scale1, scale2, config.zero_tau};
        dispatch_bwd(grid_is_bf16, config.variant, launch);
    } else {
        const BwdLaunch<true, float> launch{
            grid_dim, shmem, stream, d_pts, d_R, d_grid0, d_grid1, d_grid2, d_gout,
            d_ggrid0, d_ggrid1, d_ggrid2, d_gR, d_gpts, B, T,
            C0, H0, W0, C1, H1, W1, C2, H2, W2, gout_row_stride,
            scale0, scale1, scale2, config.zero_tau};
        dispatch_bwd(grid_is_bf16, config.variant, launch);
    }
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_tv_fuse_cuda(
    const float *d_pts, const float *d_R, const float *d_grid, float *d_out,
    long B, int T, int C, int H, int W, float h, cudaStream_t stream
) {
    if (B == 0) return;
    const size_t shmem = (size_t)T * 18 * sizeof(float);
    kplanes_tilted_tv_fwd_core<float><<<
        n_blocks(4L * B * (long)(T * C)), kThreads, shmem, stream>>>(
        d_pts, d_R, d_grid, d_out, B, T, C, H, W, h);
    CUDA_CHECK_KERNEL();
}

void kplanes_tilted_tv_fuse_grad_cuda(
    const float *d_pts, const float *d_R, const float *d_grid, const float *d_gout,
    float *d_ggrid, float *d_gR, float *d_gpts,
    long B, int T, int C, int H, int W, float h, cudaStream_t stream
) {
    if (B == 0) return;
    const size_t shmem = (size_t)T * 27 * sizeof(float);
    const long warps = n_warps(4L * B * (long)T, C);
    kplanes_tilted_tv_bwd_core<float><<<
        n_blocks(warps * 32), kThreads, shmem, stream>>>(
        d_pts, d_R, d_grid, d_gout, d_ggrid, d_gR, d_gpts, B, T, C, H, W, h);
    CUDA_CHECK_KERNEL();
}
