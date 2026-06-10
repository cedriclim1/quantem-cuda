/* ── csrc/bindings/tomography.cpp ────────────────────────────────────────
 * Raw-pointer bindings for the quantem.cuda.tomography kernels (K-Planes /
 * INR object models). Registered into the single `_core` extension by
 * csrc/bindings.cpp; see csrc/bindings/registry.h for the conventions.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include "bindings/registry.h"
#include "ops/tomography.h"

namespace py = pybind11;

namespace quantem_cuda {

static void py_kplanes_tilted_fuse_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long out_ptr,
    long B, int T, int C, int H, int W,
    long stream_ptr
) {
    kplanes_tilted_fuse_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<float *>(out_ptr),
        B, T, C, H, W,
        to_stream(stream_ptr)
    );
}

static void py_kplanes_tilted_fuse_grad_cuda(
    long pts_ptr, long r_ptr, long grid_ptr, long gout_ptr,
    long ggrid_ptr, long gr_ptr, long gpts_ptr,
    long B, int T, int C, int H, int W,
    long stream_ptr
) {
    kplanes_tilted_fuse_grad_cuda(
        reinterpret_cast<const float *>(pts_ptr),
        reinterpret_cast<const float *>(r_ptr),
        reinterpret_cast<const float *>(grid_ptr),
        reinterpret_cast<const float *>(gout_ptr),
        reinterpret_cast<float *>(ggrid_ptr),
        reinterpret_cast<float *>(gr_ptr),
        reinterpret_cast<float *>(gpts_ptr),
        B, T, C, H, W,
        to_stream(stream_ptr)
    );
}

void register_tomography_ops(py::module_ &m) {
    m.def("kplanes_tilted_fuse_cuda", &py_kplanes_tilted_fuse_cuda,
          "Fused TILTED K-Planes interpolation (one level). pts_ptr: fp32 "
          "[B,3]; r_ptr: fp32 [T,3,3]; grid_ptr: fp32 [3T,C,H,W]; out_ptr: "
          "fp32 [B,T*C], fully written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("out_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));

    m.def("kplanes_tilted_fuse_grad_cuda", &py_kplanes_tilted_fuse_grad_cuda,
          "Backward of kplanes_tilted_fuse. gout_ptr: fp32 [B,T*C]. "
          "ggrid/gr are accumulated into (pre-zero them); gpts is fully "
          "written.",
          py::arg("pts_ptr"), py::arg("r_ptr"), py::arg("grid_ptr"),
          py::arg("gout_ptr"), py::arg("ggrid_ptr"), py::arg("gr_ptr"),
          py::arg("gpts_ptr"),
          py::arg("B"), py::arg("T"), py::arg("C"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));
}

}  // namespace quantem_cuda
