#pragma once
/* ── csrc/common.cuh ─────────────────────────────────────────────────────
 * Shared CUDA helpers included by all .cu translation units.
 * Not compiled on its own — included via #include "common.cuh".
 */

#include <cuda_runtime.h>
#include <math.h>
#include <stdexcept>
#include <string>

/* ── runtime API error check ──────────────────────────────────────────── */
#define QUANTEM_CUDA_CHECK(call) do {                                         \
    cudaError_t _err = (call);                                                \
    if (_err != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA error at " __FILE__ ":")   \
            + std::to_string(__LINE__) + ": " + cudaGetErrorString(_err));    \
    }                                                                          \
} while (0)

/* ── kernel launch error check ────────────────────────────────────────────
 * Called immediately after a <<< >>> launch to surface launch-time errors
 * (invalid configuration, too much shared memory, etc.) without a
 * host-device sync. Stream-ordered errors are surfaced later when PyTorch
 * syncs on read. */
#define CUDA_CHECK_KERNEL() do {                                              \
    cudaError_t _err = cudaPeekAtLastError();                                 \
    if (_err != cudaSuccess) {                                                \
        throw std::runtime_error(std::string("CUDA kernel launch error at "   \
            __FILE__ ":") + std::to_string(__LINE__) + ": "                   \
            + cudaGetErrorString(_err));                                       \
    }                                                                          \
} while (0)
