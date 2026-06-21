/* ── csrc/bindings/core.cpp ──────────────────────────────────────────────
 * Raw-pointer bindings for the quantem.cuda.core kernels (shared TV
 * regularizers). Registered into the single `_core` extension by
 * csrc/bindings.cpp; see csrc/bindings/registry.h for the conventions.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include "bindings/registry.h"
#include "ops/core.h"

namespace py = pybind11;

namespace quantem_cuda {

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

static void py_tv_loss_l1_3d_cuda(
    long vol_ptr, long acc_ptr,
    int B, int D, int H, int W,
    long stream_ptr
) {
    tv_loss_l1_3d_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<float *>(acc_ptr),
        B, D, H, W,
        to_stream(stream_ptr)
    );
}

static void py_tv_loss_l1_3d_grad_cuda(
    long vol_ptr, long g_ptr, long grad_vol_ptr,
    int B, int D, int H, int W,
    long stream_ptr
) {
    tv_loss_l1_3d_grad_cuda(
        reinterpret_cast<const float *>(vol_ptr),
        reinterpret_cast<const float *>(g_ptr),
        reinterpret_cast<float *>(grad_vol_ptr),
        B, D, H, W,
        to_stream(stream_ptr)
    );
}

void register_core_ops(py::module_ &m) {
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

    m.def("tv_loss_l1_3d_cuda", &py_tv_loss_l1_3d_cuda,
          "L1-anisotropic 3-D TV forward. vol_ptr: fp32 [B,D,H,W]; acc_ptr: "
          "pre-zeroed 3-element fp32 accumulator (per-axis unnormalized "
          "Σ|diff| on return, order d/h/w).",
          py::arg("vol_ptr"), py::arg("acc_ptr"),
          py::arg("B"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));

    m.def("tv_loss_l1_3d_grad_cuda", &py_tv_loss_l1_3d_grad_cuda,
          "Backward of tv_loss_l1_3d. g_ptr: 3-element fp32 device array of "
          "per-axis upstream grads (order d/h/w). grad_vol is fully "
          "written; sign(0) = 0.",
          py::arg("vol_ptr"), py::arg("g_ptr"), py::arg("grad_vol_ptr"),
          py::arg("B"), py::arg("D"), py::arg("H"), py::arg("W"),
          py::arg("stream_ptr"));
}

}  // namespace quantem_cuda
