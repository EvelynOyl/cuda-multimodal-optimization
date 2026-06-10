/**
 * softmax_cuda.cpp — PyTorch C++ extension binding for CUDA Softmax operators.
 *
 * Provides:
 *   - online_safe_softmax(input, scale)     → numerically stable softmax
 *   - causal_softmax(scores, scale)         → softmax with causal mask
 *   - softmax_backward(output, grad_output) → gradient of softmax
 *
 * Build:  python csrc/setup.py build_ext --inplace
 * Usage:  import softmax_cuda; softmax_cuda.online_safe_softmax(x, scale)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// ── Forward declarations ──────────────────────────────────────────────────
template <typename scalar_t>
void online_safe_softmax_kernel(
    const scalar_t* input, scalar_t* output,
    const int rows, const int cols, const float scale);

template <typename scalar_t>
void causal_softmax_kernel(
    const scalar_t* attn_scores, scalar_t* attn_probs,
    const int B, const int H, const int N, const float scale);

template <typename scalar_t>
void softmax_backward_kernel(
    const scalar_t* output, const scalar_t* grad_output, scalar_t* grad_input,
    const int rows, const int cols);

// ── Launch config ─────────────────────────────────────────────────────────
#define SOFTMAX_BLOCK_SIZE 256

// ═══════════════════════════════════════════════════════════════════════════
// online_safe_softmax:  σ(x) along the last dimension
// ═══════════════════════════════════════════════════════════════════════════

torch::Tensor online_safe_softmax_cuda(
    const torch::Tensor& input,   // [..., D]  — softmax over last dim
    const float scale = 1.0f)
{
    TORCH_CHECK(input.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.dim() >= 1, "Input must have at least 1 dimension");

    const at::cuda::OptionalCUDAGuard guard(input.device());

    const int cols = input.size(-1);
    const int rows = input.numel() / cols;

    auto input_contig = input.contiguous();
    auto output = torch::empty_like(input_contig);

    const int blocks = rows;
    const int threads = SOFTMAX_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "online_safe_softmax_cuda", ([&] {
        online_safe_softmax_kernel<scalar_t><<<blocks, threads>>>(
            input_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            rows, cols, scale);
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "online_safe_softmax kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return output;
}

// ═══════════════════════════════════════════════════════════════════════════
// causal_softmax: softmax with causal (lower-triangular) mask
// ═══════════════════════════════════════════════════════════════════════════

torch::Tensor causal_softmax_cuda(
    const torch::Tensor& attn_scores,  // [B, H, N, N]
    const float scale = 1.0f)
{
    TORCH_CHECK(attn_scores.device().is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(attn_scores.dim() == 4, "Input must be 4D [B, H, N, N]");

    const at::cuda::OptionalCUDAGuard guard(attn_scores.device());

    const int B = attn_scores.size(0);
    const int H = attn_scores.size(1);
    const int N = attn_scores.size(2);

    TORCH_CHECK(attn_scores.size(3) == N, "Last two dims must be square [N, N]");

    auto scores_contig = attn_scores.contiguous();
    auto probs = torch::empty_like(scores_contig);

    const int total_rows = B * H * N;
    const int blocks = total_rows;
    const int threads = SOFTMAX_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(attn_scores.scalar_type(), "causal_softmax_cuda", ([&] {
        causal_softmax_kernel<scalar_t><<<blocks, threads>>>(
            scores_contig.data_ptr<scalar_t>(),
            probs.data_ptr<scalar_t>(),
            B, H, N, scale);
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "causal_softmax kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return probs;
}

// ═══════════════════════════════════════════════════════════════════════════
// softmax_backward: dX = P * (dY - Σ(P * dY))
// ═══════════════════════════════════════════════════════════════════════════

torch::Tensor softmax_backward_cuda(
    const torch::Tensor& output,        // P = softmax(X)
    const torch::Tensor& grad_output)   // dY
{
    TORCH_CHECK(output.device().is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(grad_output.device().is_cuda(), "GradOutput must be a CUDA tensor");
    TORCH_CHECK(output.sizes() == grad_output.sizes(), "Shape mismatch");

    const at::cuda::OptionalCUDAGuard guard(output.device());

    const int cols = output.size(-1);
    const int rows = output.numel() / cols;

    auto out_contig  = output.contiguous();
    auto grad_contig = grad_output.contiguous();
    auto grad_input  = torch::empty_like(out_contig);

    const int blocks  = rows;
    const int threads = SOFTMAX_BLOCK_SIZE;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "softmax_backward_cuda", ([&] {
        softmax_backward_kernel<scalar_t><<<blocks, threads>>>(
            out_contig.data_ptr<scalar_t>(),
            grad_contig.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            rows, cols);
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "softmax_backward kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return grad_input;
}

// ═══════════════════════════════════════════════════════════════════════════
// PyTorch module registration
// ═══════════════════════════════════════════════════════════════════════════

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("online_safe_softmax", &online_safe_softmax_cuda,
          "Online safe softmax: σ(x) = exp(x-max) / Σexp(x-max)",
          py::arg("input"), py::arg("scale") = 1.0f);
    m.def("causal_softmax", &causal_softmax_cuda,
          "Causal-masked softmax for self-attention",
          py::arg("attn_scores"), py::arg("scale") = 1.0f);
    m.def("softmax_backward", &softmax_backward_cuda,
          "Backward pass for softmax: dX = P*(dY - ΣP·dY)");
    m.doc() = "Optimized CUDA softmax operators with online safe algorithm";
}
