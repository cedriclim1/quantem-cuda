#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "ops/core/ml.h"

namespace {

constexpr cudaDataType_t kBf16 = CUDA_R_16BF;
constexpr cudaDataType_t kFp32 = CUDA_R_32F;
constexpr cublasComputeType_t kCompute = CUBLAS_COMPUTE_32F;
constexpr int64_t kMBucket = 256;

std::string status_string(cublasStatus_t status) {
    // cublasGetStatusString lives in libcublas rather than libcublasLt.  Keep
    // this extension linked only to the Lt API and report its stable enum.
    return std::to_string(static_cast<int>(status));
}

void check(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuBLASLt ") + what + ": " + status_string(status));
    }
}

struct MatmulDesc {
    cublasLtMatmulDesc_t value = nullptr;
    MatmulDesc() { check(cublasLtMatmulDescCreate(&value, kCompute, kFp32), "create matmul descriptor"); }
    ~MatmulDesc() { if (value != nullptr) cublasLtMatmulDescDestroy(value); }
    MatmulDesc(const MatmulDesc &) = delete;
    MatmulDesc &operator=(const MatmulDesc &) = delete;
};

struct MatrixLayout {
    cublasLtMatrixLayout_t value = nullptr;
    MatrixLayout(cudaDataType_t type, uint64_t rows, uint64_t cols, int64_t ld) {
        check(cublasLtMatrixLayoutCreate(&value, type, rows, cols, ld), "create matrix layout");
    }
    ~MatrixLayout() { if (value != nullptr) cublasLtMatrixLayoutDestroy(value); }
    MatrixLayout(const MatrixLayout &) = delete;
    MatrixLayout &operator=(const MatrixLayout &) = delete;
};

struct Preference {
    cublasLtMatmulPreference_t value = nullptr;
    Preference() { check(cublasLtMatmulPreferenceCreate(&value), "create preference"); }
    ~Preference() { if (value != nullptr) cublasLtMatmulPreferenceDestroy(value); }
    Preference(const Preference &) = delete;
    Preference &operator=(const Preference &) = delete;
};

template <typename T>
void set_attr(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const T &value) {
    check(cublasLtMatmulDescSetAttribute(desc, attr, &value, sizeof(value)), "set matmul attribute");
}

cublasLtHandle_t handle() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        throw std::runtime_error("cuBLASLt could not query the current CUDA device");
    }

    // A cuBLASLt handle becomes associated with the current device on first
    // use. Keep one process-lifetime handle per device and intentionally leak
    // them to avoid CUDA-runtime teardown ordering hazards.
    static auto *handles = new std::unordered_map<int, cublasLtHandle_t>();
    static auto *mutex = new std::mutex();
    std::lock_guard<std::mutex> lock(*mutex);
    auto it = handles->find(device);
    if (it == handles->end()) {
        cublasLtHandle_t created = nullptr;
        check(cublasLtCreate(&created), "create handle");
        it = handles->emplace(device, created).first;
    }
    return it->second;
}

enum class MatmulKind : int {
    kForwardRelu = 0,
    kForwardBias = 1,
    kDgradRelu = 2,
    kDgrad = 3,
    kWgrad = 4,
};

struct AlgoKey {
    int device;
    MatmulKind kind;
    int64_t m_bucket;
    int64_t n;
    int64_t k;
    cudaDataType_t d_type;

    bool operator==(const AlgoKey &other) const {
        return device == other.device && kind == other.kind && m_bucket == other.m_bucket &&
               n == other.n && k == other.k && d_type == other.d_type;
    }
};

struct AlgoKeyHash {
    size_t operator()(const AlgoKey &key) const {
        size_t value = static_cast<size_t>(key.device + 1);
        value = value * 1315423911u + static_cast<size_t>(key.kind);
        value = value * 1315423911u + static_cast<size_t>(key.m_bucket);
        value = value * 1315423911u + static_cast<size_t>(key.n);
        value = value * 1315423911u + static_cast<size_t>(key.k);
        value = value * 1315423911u + static_cast<size_t>(key.d_type);
        return value;
    }
};

struct CachedAlgo {
    cublasLtMatmulAlgo_t algo;
};

std::unordered_map<AlgoKey, CachedAlgo, AlgoKeyHash> &algo_cache() {
    static std::unordered_map<AlgoKey, CachedAlgo, AlgoKeyHash> cache;
    return cache;
}

std::mutex &algo_cache_mutex() {
    static std::mutex mutex;
    return mutex;
}

AlgoKey make_key(MatmulKind kind, int64_t m, int64_t n, int64_t k, cudaDataType_t d_type) {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        throw std::runtime_error("cuBLASLt could not query the current CUDA device");
    }
    return {device, kind, ((m + kMBucket - 1) / kMBucket) * kMBucket, n, k, d_type};
}

cublasLtMatmulAlgo_t select_algo(
    const AlgoKey &key,
    cublasLtMatmulDesc_t op,
    cublasLtMatrixLayout_t a,
    cublasLtMatrixLayout_t b,
    cublasLtMatrixLayout_t c,
    cublasLtMatrixLayout_t d,
    size_t workspace_bytes
) {
    {
        std::lock_guard<std::mutex> lock(algo_cache_mutex());
        auto it = algo_cache().find(key);
        if (it != algo_cache().end()) {
            cublasLtMatmulHeuristicResult_t checked{};
            if (cublasLtMatmulAlgoCheck(handle(), op, a, b, c, d, &it->second.algo, &checked) ==
                    CUBLAS_STATUS_SUCCESS &&
                checked.state == CUBLAS_STATUS_SUCCESS && checked.workspaceSize <= workspace_bytes) {
                return it->second.algo;
            }
            algo_cache().erase(it);
        }
    }

    Preference preference;
    check(
        cublasLtMatmulPreferenceSetAttribute(
            preference.value,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_bytes,
            sizeof(workspace_bytes)
        ),
        "set workspace preference"
    );
    cublasLtMatmulHeuristicResult_t results[8]{};
    int count = 0;
    check(
        cublasLtMatmulAlgoGetHeuristic(
            handle(), op, a, b, c, d, preference.value, 8, results, &count
        ),
        "get algorithm heuristic"
    );
    auto found = std::find_if(results, results + count, [workspace_bytes](const auto &result) {
        return result.state == CUBLAS_STATUS_SUCCESS && result.workspaceSize <= workspace_bytes;
    });
    if (found == results + count) {
        throw std::runtime_error("cuBLASLt found no supported algorithm for fused MLP shape");
    }
    {
        std::lock_guard<std::mutex> lock(algo_cache_mutex());
        algo_cache()[key] = CachedAlgo{found->algo};
    }
    return found->algo;
}

void launch_matmul(
    MatmulKind kind,
    int64_t cache_m,
    int64_t cache_n,
    int64_t cache_k,
    MatmulDesc &op,
    MatrixLayout &a_layout,
    MatrixLayout &b_layout,
    MatrixLayout &d_layout,
    const void *a,
    const void *b,
    void *d,
    void *workspace,
    size_t workspace_bytes,
    cudaStream_t stream,
    cudaDataType_t d_type
) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const AlgoKey key = make_key(kind, cache_m, cache_n, cache_k, d_type);
    const cublasLtMatmulAlgo_t algo = select_algo(
        key, op.value, a_layout.value, b_layout.value, d_layout.value, d_layout.value,
        workspace_bytes
    );
    check(
        cublasLtMatmul(
            handle(), op.value, &alpha,
            a, a_layout.value, b, b_layout.value,
            &beta, d, d_layout.value, d, d_layout.value,
            &algo, workspace, workspace_bytes, stream
        ),
        "launch matmul"
    );
}

void forward_linear(
    const void *x, const void *weight, const void *bias,
    void *output, void *aux,
    int64_t m, int64_t n, int64_t k, int64_t aux_ld_bits,
    void *workspace, size_t workspace_bytes, cudaStream_t stream,
    bool relu
) {
    MatmulDesc op;
    const cublasOperation_t trans_a = CUBLAS_OP_T;
    const cublasOperation_t trans_b = CUBLAS_OP_N;
    set_attr(op.value, CUBLASLT_MATMUL_DESC_TRANSA, trans_a);
    set_attr(op.value, CUBLASLT_MATMUL_DESC_TRANSB, trans_b);
    const cublasLtEpilogue_t epilogue =
        relu ? CUBLASLT_EPILOGUE_RELU_AUX_BIAS : CUBLASLT_EPILOGUE_BIAS;
    set_attr(op.value, CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
    set_attr(op.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias);
    if (relu) {
        set_attr(op.value, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, aux);
        set_attr(op.value, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, aux_ld_bits);
    }

    // A is row-major weight [N,K], viewed as column-major [K,N].
    // B is row-major input [M,K], viewed as column-major [K,M].
    // D is column-major [N,M], the same bytes as row-major [M,N].
    MatrixLayout a_layout(kBf16, k, n, k);
    MatrixLayout b_layout(kBf16, k, m, k);
    MatrixLayout d_layout(kBf16, n, m, n);
    launch_matmul(
        relu ? MatmulKind::kForwardRelu : MatmulKind::kForwardBias,
        m, n, k, op, a_layout, b_layout, d_layout,
        weight, x, output, workspace, workspace_bytes, stream, kBf16
    );
}

void dgrad(
    const void *grad, const void *weight, void *output,
    const void *aux, void *bias_grad,
    int64_t m, int64_t n, int64_t k, int64_t aux_ld_bits,
    void *workspace, size_t workspace_bytes, cudaStream_t stream,
    bool drelu
) {
    MatmulDesc op;
    const cublasOperation_t no_trans = CUBLAS_OP_N;
    set_attr(op.value, CUBLASLT_MATMUL_DESC_TRANSA, no_trans);
    set_attr(op.value, CUBLASLT_MATMUL_DESC_TRANSB, no_trans);
    if (drelu) {
        const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_DRELU_BGRAD;
        set_attr(op.value, CUBLASLT_MATMUL_DESC_EPILOGUE, epilogue);
        set_attr(op.value, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, aux);
        set_attr(op.value, CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, aux_ld_bits);
        set_attr(op.value, CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias_grad);
    }

    // weight [N,K] -> col [K,N], grad [M,N] -> col [N,M].
    MatrixLayout a_layout(kBf16, k, n, k);
    MatrixLayout b_layout(kBf16, n, m, n);
    MatrixLayout d_layout(kBf16, k, m, k);
    launch_matmul(
        drelu ? MatmulKind::kDgradRelu : MatmulKind::kDgrad,
        m, k, n, op, a_layout, b_layout, d_layout,
        weight, grad, output, workspace, workspace_bytes, stream, kBf16
    );
}

void wgrad(
    const void *x, const void *grad, float *grad_weight,
    int64_t m, int64_t n, int64_t k,
    void *workspace, size_t workspace_bytes, cudaStream_t stream
) {
    MatmulDesc op;
    const cublasOperation_t trans_a = CUBLAS_OP_N;
    const cublasOperation_t trans_b = CUBLAS_OP_T;
    set_attr(op.value, CUBLASLT_MATMUL_DESC_TRANSA, trans_a);
    set_attr(op.value, CUBLASLT_MATMUL_DESC_TRANSB, trans_b);

    // X^T [K,M] @ grad [M,N] -> [K,N], underlying row-major dW [N,K].
    MatrixLayout a_layout(kBf16, k, m, k);
    MatrixLayout b_layout(kBf16, n, m, n);
    MatrixLayout d_layout(kFp32, k, n, k);
    launch_matmul(
        MatmulKind::kWgrad, m, n, k, op, a_layout, b_layout, d_layout,
        x, grad, grad_weight, workspace, workspace_bytes, stream, kFp32
    );
}

}  // namespace

void cublaslt_mlp_forward_cuda(
    const void *x,
    const void *w1, const void *b1,
    const void *w2, const void *b2,
    const void *w3, const void *b3,
    void *h1, void *h2, void *out,
    void *aux1, void *aux2,
    long M, int K, int H1, int H2, int O,
    long aux1_ld_bits, long aux2_ld_bits,
    void *workspace, size_t workspace_bytes,
    cudaStream_t stream
) {
    forward_linear(x, w1, b1, h1, aux1, M, H1, K, aux1_ld_bits,
                   workspace, workspace_bytes, stream, true);
    forward_linear(h1, w2, b2, h2, aux2, M, H2, H1, aux2_ld_bits,
                   workspace, workspace_bytes, stream, true);
    forward_linear(h2, w3, b3, out, nullptr, M, O, H2, 0,
                   workspace, workspace_bytes, stream, false);
}

void cublaslt_mlp_backward_cuda(
    const void *x,
    const void *w1, const void *w2, const void *w3,
    const void *h1, const void *h2,
    const void *aux1, const void *aux2,
    const void *grad_out,
    void *grad_x,
    float *grad_w1, float *grad_w2, float *grad_w3,
    void *grad_b1, void *grad_b2,
    void *dz1, void *dz2,
    long M, int K, int H1, int H2, int O,
    long aux1_ld_bits, long aux2_ld_bits,
    void *workspace, size_t workspace_bytes,
    cudaStream_t stream
) {
    // Cross-layer fusion is essential here: the output of each downstream
    // dgrad has exactly the shape of the preceding hidden activation mask.
    dgrad(grad_out, w3, dz2, aux2, grad_b2, M, O, H2, aux2_ld_bits,
          workspace, workspace_bytes, stream, true);
    wgrad(h2, grad_out, grad_w3, M, O, H2, workspace, workspace_bytes, stream);

    dgrad(dz2, w2, dz1, aux1, grad_b1, M, H2, H1, aux1_ld_bits,
          workspace, workspace_bytes, stream, true);
    wgrad(h1, dz2, grad_w2, M, H2, H1, workspace, workspace_bytes, stream);

    dgrad(dz1, w1, grad_x, nullptr, nullptr, M, H1, K, 0,
          workspace, workspace_bytes, stream, false);
    wgrad(x, dz1, grad_w1, M, H1, K, workspace, workspace_bytes, stream);
}
