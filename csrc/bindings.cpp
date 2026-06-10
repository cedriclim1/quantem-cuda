/* ── csrc/bindings.cpp ───────────────────────────────────────────────────
 * pybind11 module `quantem.cuda._core`.
 *
 * The binding layer is deliberately torch-free: tensors cross the boundary
 * as raw device pointers (`tensor.data_ptr()`) plus shape ints and the
 * caller's CUDA stream handle. Keeping libtorch out of the link line is
 * what lets one compiled wheel work across PyTorch versions — the only
 * shared dependency is libcudart, which resolves to the copy PyTorch has
 * already loaded at import time. All torch-facing niceties (autograd,
 * torch.compile registration, validation) live in quantem/cuda/_ops.py.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include "ops.h"

namespace py = pybind11;

static cudaStream_t to_stream(long stream_ptr) {
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

static void py_tv_loss_iso_3d_cuda(
    long vol_ptr, long acc_ptr,
    int B, int D, int H, int W,
    float eps,
    long stream_ptr
) {
    tv_loss_iso_3d_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(acc_ptr),
        B, D, H, W, eps,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_iso_3d_grad_cuda(
    long vol_ptr, long g_scaled_ptr, long grad_vol_ptr,
    int B, int D, int H, int W,
    float eps,
    long stream_ptr
) {
    tv_loss_iso_3d_grad_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(g_scaled_ptr),
        reinterpret_cast<float *>(grad_vol_ptr),
        B, D, H, W, eps,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_sq_3d_cuda(
    long vol_ptr, long acc_ptr,
    int B, int D, int H, int W,
    long stream_ptr
) {
    tv_loss_sq_3d_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(acc_ptr),
        B, D, H, W,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_sq_3d_grad_cuda(
    long vol_ptr, long g_ptr, long grad_vol_ptr,
    int B, int D, int H, int W,
    long stream_ptr
) {
    tv_loss_sq_3d_grad_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(g_ptr),
        reinterpret_cast<float *>(grad_vol_ptr),
        B, D, H, W,
        to_stream(stream_ptr)
    );
}

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

PYBIND11_MODULE(_core, m) {
    m.doc() = "quantem-cuda compiled kernels (raw-pointer API; use quantem.cuda instead)";
    m.attr("__cudart_version__") = CUDART_VERSION;

    m.def("tv_loss_iso_3d_cuda", &py_tv_loss_iso_3d_cuda,
          "Isotropic 3-D TV forward. vol_ptr: fp32 [B,D,H,W]; acc_ptr: "
          "pre-zeroed single-element fp32 accumulator (unnormalized corner "
          "sum on return).",
          py::arg("vol_ptr"), py::arg("acc_ptr"),
          py::arg("B"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("eps"), py::arg("stream_ptr"));

    m.def("tv_loss_iso_3d_grad_cuda", &py_tv_loss_iso_3d_grad_cuda,
          "Backward of tv_loss_iso_3d. g_scaled_ptr: single-element fp32 "
          "device scalar = upstream_grad / N. grad_vol is fully written.",
          py::arg("vol_ptr"), py::arg("g_scaled_ptr"), py::arg("grad_vol_ptr"),
          py::arg("B"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("eps"), py::arg("stream_ptr"));

    m.def("tv_loss_sq_3d_cuda", &py_tv_loss_sq_3d_cuda,
          "Squared-anisotropic 3-D TV forward (quantem tv_vol parity). "
          "acc_ptr: pre-zeroed single-element fp32 accumulator "
          "(unnormalized Σ diff² on return).",
          py::arg("vol_ptr"), py::arg("acc_ptr"),
          py::arg("B"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));

    m.def("tv_loss_sq_3d_grad_cuda", &py_tv_loss_sq_3d_grad_cuda,
          "Backward of tv_loss_sq_3d. g_ptr: single-element fp32 device "
          "scalar = raw upstream grad (factor 2 applied in-kernel). "
          "grad_vol is fully written.",
          py::arg("vol_ptr"), py::arg("g_ptr"), py::arg("grad_vol_ptr"),
          py::arg("B"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));

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
