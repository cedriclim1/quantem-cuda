/* ── csrc/bindings/core/ml.cpp ───────────────────────────────────────────
 * Raw-pointer bindings for the quantem.cuda.core.ml kernels (K-Planes /
 * tensor-decomposition models). Registered into the single `_core`
 * extension by csrc/bindings.cpp; see csrc/bindings/registry.h for the
 * conventions.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include "bindings/registry.h"
#include "ops/core/ml.h"

namespace py = pybind11;

namespace quantem_cuda {

static void py_plane_tv_loss_cuda(
    long grid0_ptr, long grid1_ptr, long grid2_ptr,
    long partials_ptr, long output_ptr,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations, int num_blocks,
    long stream_ptr
) {
    plane_tv_loss_cuda(
        reinterpret_cast<const float *>(grid0_ptr),
        reinterpret_cast<const float *>(grid1_ptr),
        reinterpret_cast<const float *>(grid2_ptr),
        reinterpret_cast<float *>(partials_ptr),
        reinterpret_cast<float *>(output_ptr),
        P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
        rotations, num_blocks, to_stream(stream_ptr)
    );
}

static void py_plane_tv_loss_grad_cuda(
    long grid0_ptr, long grid1_ptr, long grid2_ptr,
    long grad_output_ptr,
    long grad_grid0_ptr, long grad_grid1_ptr, long grad_grid2_ptr,
    int P0, int C0, int H0, int W0,
    int P1, int C1, int H1, int W1,
    int P2, int C2, int H2, int W2,
    int rotations, int num_blocks,
    long stream_ptr
) {
    plane_tv_loss_grad_cuda(
        reinterpret_cast<const float *>(grid0_ptr),
        reinterpret_cast<const float *>(grid1_ptr),
        reinterpret_cast<const float *>(grid2_ptr),
        reinterpret_cast<const float *>(grad_output_ptr),
        reinterpret_cast<float *>(grad_grid0_ptr),
        reinterpret_cast<float *>(grad_grid1_ptr),
        reinterpret_cast<float *>(grad_grid2_ptr),
        P0, C0, H0, W0, P1, C1, H1, W1, P2, C2, H2, W2,
        rotations, num_blocks, to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long out_ptr,
    long B, int T, int C, int H, int W,
    bool grid_is_bf16,
    long stream_ptr
) {
    kplanes_tilted_fuse_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const void *>(grid_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H, W,
        grid_is_bf16,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse_grad_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long gout_ptr,
    long ggrid_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C, int H, int W,
    bool grid_is_bf16,
    long stream_ptr
) {
    kplanes_tilted_fuse_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const void *>(grid_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(ggrid_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H, W,
        grid_is_bf16,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse_ms_cuda(
    long pts_ptr, long r_ptr,
    long grid0_ptr, long grid1_ptr, long grid2_ptr,
    long out_ptr,
    long B, int T,
    int C0, int H0, int W0,
    int C1, int H1, int W1,
    int C2, int H2, int W2,
    float scale0, float scale1, float scale2,
    bool grid_is_bf16,
    long stream_ptr
) {
    kplanes_tilted_fuse_ms_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const void *>(grid0_ptr),
        reinterpret_cast<const void *>(grid1_ptr),
        reinterpret_cast<const void *>(grid2_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T,
        C0, H0, W0, C1, H1, W1, C2, H2, W2,
        scale0, scale1, scale2,
        grid_is_bf16,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse_ms_grad_cuda(
    long pts_ptr, long r_ptr,
    long grid0_ptr, long grid1_ptr, long grid2_ptr,
    long gout_ptr,
    long ggrid0_ptr, long ggrid1_ptr, long ggrid2_ptr,
    long gr_ptr, long gpts_ptr,
    long B, int T,
    int C0, int H0, int W0,
    int C1, int H1, int W1,
    int C2, int H2, int W2,
    long gout_row_stride,
    float scale0, float scale1, float scale2,
    bool grid_is_bf16,
    long stream_ptr
) {
    kplanes_tilted_fuse_ms_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const void *>(grid0_ptr),
        reinterpret_cast<const void *>(grid1_ptr),
        reinterpret_cast<const void *>(grid2_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(ggrid0_ptr),
        reinterpret_cast<float *>(ggrid1_ptr),
        reinterpret_cast<float *>(ggrid2_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T,
        C0, H0, W0, C1, H1, W1, C2, H2, W2,
        gout_row_stride,
        scale0, scale1, scale2,
        grid_is_bf16,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_tv_fuse_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long out_ptr,
    long B, int T, int C, int H, int W,
    float h,
    long stream_ptr
) {
    kplanes_tilted_tv_fuse_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H, W, h,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_tv_fuse_grad_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long gout_ptr,
    long ggrid_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C, int H, int W,
    float h,
    long stream_ptr
) {
    kplanes_tilted_tv_fuse_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(ggrid_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H, W, h,
        to_stream(stream_ptr)
    );
}

void register_core_ml_ops(py::module_ &m) {
    m.def("plane_tv_loss_cuda", &py_plane_tv_loss_cuda,
          "Three-level plane-wise 2-D squared-TV forward. Grids are fp32 "
          "[3*T,H,W,C] channels-last; output is one fp32 scalar.",
          py::arg("grid0_ptr"), py::arg("grid1_ptr"), py::arg("grid2_ptr"),
          py::arg("partials_ptr"), py::arg("output_ptr"),
          py::arg("P0"), py::arg("C0"), py::arg("H0"), py::arg("W0"),
          py::arg("P1"), py::arg("C1"), py::arg("H1"), py::arg("W1"),
          py::arg("P2"), py::arg("C2"), py::arg("H2"), py::arg("W2"),
          py::arg("rotations"), py::arg("num_blocks"), py::arg("stream_ptr"));

    m.def("plane_tv_loss_grad_cuda", &py_plane_tv_loss_grad_cuda,
          "Analytic backward of plane_tv_loss; fully writes three fp32 gradients.",
          py::arg("grid0_ptr"), py::arg("grid1_ptr"), py::arg("grid2_ptr"),
          py::arg("grad_output_ptr"),
          py::arg("grad_grid0_ptr"), py::arg("grad_grid1_ptr"),
          py::arg("grad_grid2_ptr"),
          py::arg("P0"), py::arg("C0"), py::arg("H0"), py::arg("W0"),
          py::arg("P1"), py::arg("C1"), py::arg("H1"), py::arg("W1"),
          py::arg("P2"), py::arg("C2"), py::arg("H2"), py::arg("W2"),
          py::arg("rotations"), py::arg("num_blocks"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse_cuda", &py_kplanes_tilted_fuse_cuda,
          "Fused TILTED K-Planes interpolation (one level). pts_ptr: fp32 "
          "[B,3]; r_ptr: fp32 [T,3,3]; grid_ptr: fp32 or bf16 "
          "[3T,C,H,W]; out_ptr: fp32 [B,T*C], fully written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("grid_is_bf16"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse_grad_cuda", &py_kplanes_tilted_fuse_grad_cuda,
          "Backward of kplanes_tilted_fuse. gout_ptr: fp32 [B,T*C]. "
          "grid_ptr may be fp32 or bf16; ggrid/gr remain fp32 and are "
          "accumulated into (pre-zero them); gpts is fully written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("gout_ptr"), py::arg("ggrid_ptr"), py::arg("gr_ptr"),
          py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("grid_is_bf16"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse_ms_cuda", &py_kplanes_tilted_fuse_ms_cuda,
          "Three-level TILTED K-Planes forward. One launch writes fp32 "
          "[B, T*(C0+C1+C2)] without concatenation.",
          py::arg("pts_ptr"), py::arg("r_ptr"),
          py::arg("grid0_ptr"), py::arg("grid1_ptr"), py::arg("grid2_ptr"),
          py::arg("out_ptr"), py::arg("B"), py::arg("T"),
          py::arg("C0"), py::arg("H0"), py::arg("W0"),
          py::arg("C1"), py::arg("H1"), py::arg("W1"),
          py::arg("C2"), py::arg("H2"), py::arg("W2"),
          py::arg("scale0"), py::arg("scale1"), py::arg("scale2"),
          py::arg("grid_is_bf16"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse_ms_grad_cuda", &py_kplanes_tilted_fuse_ms_grad_cuda,
          "Backward of the three-level op. Reads an inner-contiguous gout "
          "using its row stride and level column offsets.",
          py::arg("pts_ptr"), py::arg("r_ptr"),
          py::arg("grid0_ptr"), py::arg("grid1_ptr"), py::arg("grid2_ptr"),
          py::arg("gout_ptr"),
          py::arg("ggrid0_ptr"), py::arg("ggrid1_ptr"), py::arg("ggrid2_ptr"),
          py::arg("gr_ptr"), py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"),
          py::arg("C0"), py::arg("H0"), py::arg("W0"),
          py::arg("C1"), py::arg("H1"), py::arg("W1"),
          py::arg("C2"), py::arg("H2"), py::arg("W2"),
          py::arg("gout_row_stride"),
          py::arg("scale0"), py::arg("scale1"), py::arg("scale2"),
          py::arg("grid_is_bf16"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_tv_fuse_cuda", &py_kplanes_tilted_tv_fuse_cuda,
          "TV-specialized fused TILTED K-Planes forward. pts_ptr: fp32 [B,3]; "
          "r_ptr: fp32 [T,3,3]; grid_ptr: fp32 [3T,H,W,C] channels-last; "
          "out_ptr: fp32 [4,B,T*C], fully written. "
          "h: finite-difference step in world [-1,1] units.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("h"), py::arg("stream_ptr"));

    m.def("kplanes_tilted_tv_fuse_grad_cuda", &py_kplanes_tilted_tv_fuse_grad_cuda,
          "Backward of kplanes_tilted_tv_fuse. gout_ptr: fp32 [4,B,T*C]. "
          "ggrid/gr are accumulated into (pre-zero them); gpts is fully "
          "written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("gout_ptr"), py::arg("ggrid_ptr"), py::arg("gr_ptr"),
          py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("h"), py::arg("stream_ptr"));
}

}  // namespace quantem_cuda
