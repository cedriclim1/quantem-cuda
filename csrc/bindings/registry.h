#pragma once
/* ── csrc/bindings/registry.h ────────────────────────────────────────────
 * Registrar declarations for the per-submodule binding files
 * (csrc/bindings/<module>.cpp), assembled into the single `_core`
 * extension by csrc/bindings.cpp. One registrar per quantem.cuda
 * submodule; add a declaration here when a new submodule lands.
 */

#include <pybind11/pybind11.h>

#include <cuda_runtime.h>

namespace quantem_cuda {

/* Stream handles cross the binding layer as the integer value of
 * torch.cuda.current_stream().cuda_stream. */
inline cudaStream_t to_stream(long stream_ptr) {
    return reinterpret_cast<cudaStream_t>(stream_ptr);
}

/* quantem.cuda.core — shared kernels (TV regularizers). */
void register_core_ops(pybind11::module_ &m);

/* quantem.cuda.core.ml — K-Planes / tensor-decomposition model kernels. */
void register_core_ml_ops(pybind11::module_ &m);

}  // namespace quantem_cuda
