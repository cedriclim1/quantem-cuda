/* Fused trunc-exp density activation for fp32/bf16 tensors.
 *
 * Forward:  output = exp(values - offset)
 * Backward: grad_values = grad_output * exp(min(values - offset, 15))
 *
 * The production tensor is [samples, 1], but explicit two-dimensional
 * element strides also cover sliced/non-contiguous audit inputs without a
 * packing allocation. Outputs are dense and each logical element is written
 * exactly once.
 */

#include "common.cuh"
#include "ops/core/ml.h"

#include <cuda_bf16.h>

#include <algorithm>

namespace {

constexpr int kBlockSize = 256;

template <typename T>
__device__ __forceinline__ float load_value(const T *ptr, size_t offset) {
    return static_cast<float>(ptr[offset]);
}

template <>
__device__ __forceinline__ float load_value<__nv_bfloat16>(
    const __nv_bfloat16 *ptr, size_t offset
) {
    return __bfloat162float(ptr[offset]);
}

template <typename T>
__device__ __forceinline__ void store_value(T *ptr, size_t offset, float value) {
    ptr[offset] = static_cast<T>(value);
}

template <>
__device__ __forceinline__ void store_value<__nv_bfloat16>(
    __nv_bfloat16 *ptr, size_t offset, float value
) {
    ptr[offset] = __float2bfloat16_rn(value);
}

template <typename T>
__device__ __forceinline__ float subtract_offset(float value, float offset) {
    return value - offset;
}

template <>
__device__ __forceinline__ float subtract_offset<__nv_bfloat16>(float value, float offset) {
    // Match torch's bf16 pointwise subtraction cast before the following exp.
    return __bfloat162float(__float2bfloat16_rn(value - offset));
}

template <typename T>
__global__ void density_tail_kernel(
    const T *__restrict__ values,
    T *__restrict__ output,
    size_t rows, size_t cols,
    size_t value_stride0, size_t value_stride1,
    size_t output_stride0, size_t output_stride1,
    float offset
) {
    size_t count = rows * cols;
    size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count;
         linear += step) {
        size_t row = linear / cols;
        size_t col = linear - row * cols;
        size_t input_index = row * value_stride0 + col * value_stride1;
        size_t output_index = row * output_stride0 + col * output_stride1;
        float shifted = subtract_offset<T>(load_value(values, input_index), offset);
        store_value(output, output_index, expf(shifted));
    }
}

template <typename T>
__global__ void density_tail_grad_kernel(
    const T *__restrict__ values,
    const T *__restrict__ grad_output,
    T *__restrict__ grad_values,
    size_t rows, size_t cols,
    size_t value_stride0, size_t value_stride1,
    size_t grad_output_stride0, size_t grad_output_stride1,
    size_t grad_value_stride0, size_t grad_value_stride1,
    float offset
) {
    size_t count = rows * cols;
    size_t step = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t linear = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linear < count;
         linear += step) {
        size_t row = linear / cols;
        size_t col = linear - row * cols;
        size_t input_index = row * value_stride0 + col * value_stride1;
        size_t gout_index = row * grad_output_stride0 + col * grad_output_stride1;
        size_t grad_index = row * grad_value_stride0 + col * grad_value_stride1;
        float shifted = subtract_offset<T>(load_value(values, input_index), offset);
        float exponent = expf(fminf(shifted, 15.0f));
        float upstream = load_value(grad_output, gout_index);
        store_value(grad_values, grad_index, upstream * exponent);
    }
}

int num_blocks(long rows, long cols) {
    size_t count = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    size_t blocks = (count + kBlockSize - 1) / kBlockSize;
    return static_cast<int>(std::min<size_t>(blocks, 65535));
}

}  // namespace

void density_tail_cuda(
    const void *d_values,
    void *d_output,
    long rows, long cols,
    long value_stride0, long value_stride1,
    long output_stride0, long output_stride1,
    float offset, bool is_bf16,
    cudaStream_t stream
) {
    int blocks = num_blocks(rows, cols);
    if (is_bf16) {
        density_tail_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(d_values),
            reinterpret_cast<__nv_bfloat16 *>(d_output),
            rows, cols,
            value_stride0, value_stride1,
            output_stride0, output_stride1,
            offset
        );
    } else {
        density_tail_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float *>(d_values),
            reinterpret_cast<float *>(d_output),
            rows, cols,
            value_stride0, value_stride1,
            output_stride0, output_stride1,
            offset
        );
    }
    CUDA_CHECK_KERNEL();
}

void density_tail_grad_cuda(
    const void *d_values,
    const void *d_grad_output,
    void *d_grad_values,
    long rows, long cols,
    long value_stride0, long value_stride1,
    long grad_output_stride0, long grad_output_stride1,
    long grad_value_stride0, long grad_value_stride1,
    float offset, bool is_bf16,
    cudaStream_t stream
) {
    int blocks = num_blocks(rows, cols);
    if (is_bf16) {
        density_tail_grad_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16 *>(d_values),
            reinterpret_cast<const __nv_bfloat16 *>(d_grad_output),
            reinterpret_cast<__nv_bfloat16 *>(d_grad_values),
            rows, cols,
            value_stride0, value_stride1,
            grad_output_stride0, grad_output_stride1,
            grad_value_stride0, grad_value_stride1,
            offset
        );
    } else {
        density_tail_grad_kernel<<<blocks, kBlockSize, 0, stream>>>(
            reinterpret_cast<const float *>(d_values),
            reinterpret_cast<const float *>(d_grad_output),
            reinterpret_cast<float *>(d_grad_values),
            rows, cols,
            value_stride0, value_stride1,
            grad_output_stride0, grad_output_stride1,
            grad_value_stride0, grad_value_stride1,
            offset
        );
    }
    CUDA_CHECK_KERNEL();
}
