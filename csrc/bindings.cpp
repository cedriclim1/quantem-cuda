/* ── csrc/bindings.cpp ───────────────────────────────────────────────────
 * pybind11 module entry for `quantem.cuda._core`: one compiled extension
 * shared by all quantem.cuda submodules. Each submodule contributes its
 * raw-pointer bindings via a registrar in csrc/bindings/<module>.cpp,
 * declared in csrc/bindings/registry.h.
 *
 * The binding layer is deliberately torch-free: tensors cross the boundary
 * as raw device pointers (`tensor.data_ptr()`) plus shape ints and the
 * caller's CUDA stream handle. Keeping libtorch out of the link line is
 * what lets one compiled wheel work across PyTorch versions — the only
 * shared dependency is libcudart, which resolves to the copy PyTorch has
 * already loaded at import time. All torch-facing niceties (autograd,
 * torch.compile registration, validation) live in quantem/cuda/<module>/_ops.py.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

#include "bindings/registry.h"

PYBIND11_MODULE(_core, m) {
    m.doc() = "quantem-cuda compiled kernels (raw-pointer API; use quantem.cuda.* instead)";
    m.attr("__cudart_version__") = CUDART_VERSION;

    quantem_cuda::register_core_ops(m);
    quantem_cuda::register_core_ml_ops(m);
}
