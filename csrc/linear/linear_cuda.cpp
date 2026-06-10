/**
 * linear_cuda.cpp — PyTorch C++ extension binding for CUDA Linear operators.
 *
 * Provides:
 *   - tiled_gemm(A, B)       → C = A @ B   (forward pass)
 *   - tiled_gemm_bt(A, B)    → C = A @ B^T  (gradient passthrough)
 *   - linear_bias_gelu(A, W, b) → GELU(A @ W + b)  (fused FFN activation)
 *
 * Build:  python csrc/setup.py build_ext --inplace
 * Usage:  import linear_cuda; linear_cuda.tiled_gemm(a, b)
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

// ── Forward declarations from linear_cuda_kernel.cu ───────────────────────
template <typename scalar_t>
void tiled_gemm_kernel(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    const int M, const int N, const int K);

template <typename scalar_t>
void tiled_gemm_bt_kernel(
    const scalar_t* A, const scalar_t* B, scalar_t* C,
    const int M, const int N, const int K);

template <typename scalar_t>
void linear_bias_gelu_kernel(
    const scalar_t* input, const scalar_t* weight, const scalar_t* bias,
    scalar_t* output, const int M, const int N, const int K);

// ── Launch configuration ──────────────────────────────────────────────────
#define TILE_M  128
#define TILE_N  128
#define THREADS_PER_BLOCK 256

// ═══════════════════════════════════════════════════════════════════════════
// tiled_gemm: C = A @ B  (all row-major, B is [K, N])
// ═══════════════════════════════════════════════════════════════════════════
torch::Tensor tiled_gemm_cuda(
    const torch::Tensor& A,    // [M, K] or [B, M, K]
    const torch::Tensor& B)    // [K, N] or [B, K, N]
{
    // ── Input validation ──────────────────────────────────────────────────
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.device().is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(A.dim() >= 2, "A must have at least 2 dimensions");
    TORCH_CHECK(B.dim() >= 2, "B must have at least 2 dimensions");

    const at::cuda::OptionalCUDAGuard guard(A.device());

    // Handle batched input
    const bool batched = (A.dim() == 3);
    const int B_dim = batched ? A.size(0) : 1;
    const int M = batched ? A.size(1) : A.size(0);
    const int K = batched ? A.size(2) : A.size(1);
    const int N = batched ? B.size(2) : B.size(1);

    TORCH_CHECK(batched ? B.size(0) == B_dim : true, "Batch dimensions must match");
    TORCH_CHECK(batched ? B.size(1) == K : B.size(0) == K, "Inner dimension mismatch: A[-1] == K and B[-2] == K required");

    // Ensure contiguous
    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();

    auto C = torch::empty(
        batched ? torch::IntArrayRef{B_dim, M, N} : torch::IntArrayRef{M, N},
        A.options());

    // ── Launch grid ───────────────────────────────────────────────────────
    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(
        (N + TILE_N - 1) / (TILE_N / 8 * 8),   // blocks in N direction
        (M + TILE_M - 1) / TILE_M               // blocks in M direction
    );

    // Adjust for batched: add batch dimension as z
    const dim3 grid_3d(grid.x, grid.y, B_dim);

    // ── Dispatch by dtype ─────────────────────────────────────────────────
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tiled_gemm_cuda", ([&] {
        // For batched, we need to loop over the batch dimension
        // (simplified: process each batch item sequentially)
        for (int b = 0; b < B_dim; ++b) {
            auto A_b = batched ? A_contig[b] : A_contig;
            auto B_b = batched ? B_contig[b] : B_contig;
            auto C_b = batched ? C[b] : C;

            tiled_gemm_kernel<scalar_t><<<grid, block>>>(
                A_b.data_ptr<scalar_t>(),
                B_b.data_ptr<scalar_t>(),
                C_b.data_ptr<scalar_t>(),
                M, N, K);
        }
    }));

    // Check for kernel launch errors
    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "tiled_gemm kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return C;
}

// ═══════════════════════════════════════════════════════════════════════════
// tiled_gemm_bt: C = A @ B^T   (A: [M,K], B: [N,K], C: [M,N])
// ═══════════════════════════════════════════════════════════════════════════
torch::Tensor tiled_gemm_bt_cuda(
    const torch::Tensor& A,    // [M, K]
    const torch::Tensor& B)    // [N, K] — will be implicitly transposed
{
    TORCH_CHECK(A.device().is_cuda() && B.device().is_cuda(), "Both inputs must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Both inputs must be 2D");

    const at::cuda::OptionalCUDAGuard guard(A.device());

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(0);
    TORCH_CHECK(B.size(1) == K, "Inner dimension mismatch");

    auto A_contig = A.contiguous();
    auto B_contig = B.contiguous();
    auto C = torch::empty({M, N}, A.options());

    const dim3 block(THREADS_PER_BLOCK);
    const dim3 grid(
        (N + TILE_N - 1) / TILE_N,
        (M + TILE_M - 1) / TILE_M);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(A.scalar_type(), "tiled_gemm_bt_cuda", ([&] {
        tiled_gemm_bt_kernel<scalar_t><<<grid, block>>>(
            A_contig.data_ptr<scalar_t>(),
            B_contig.data_ptr<scalar_t>(),
            C.data_ptr<scalar_t>(),
            M, N, K);
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "tiled_gemm_bt kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return C;
}

// ═══════════════════════════════════════════════════════════════════════════
// linear_bias_gelu: GELU(A @ W + b)  — fused FFN activation
// ═══════════════════════════════════════════════════════════════════════════
torch::Tensor linear_bias_gelu_cuda(
    const torch::Tensor& input,    // [M, K]
    const torch::Tensor& weight,   // [K, N]
    const torch::Tensor& bias)     // [N]
{
    TORCH_CHECK(input.device().is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.dim() == 2, "Input must be 2D");
    TORCH_CHECK(weight.dim() == 2, "Weight must be 2D");
    TORCH_CHECK(bias.dim() == 1, "Bias must be 1D");

    const at::cuda::OptionalCUDAGuard guard(input.device());

    const int M = input.size(0);
    const int K = input.size(1);
    const int N = weight.size(1);

    TORCH_CHECK(weight.size(0) == K, "Weight inner dimension mismatch");
    TORCH_CHECK(bias.size(0) == N, "Bias size must match output features");

    auto input_contig  = input.contiguous();
    auto weight_contig = weight.contiguous();
    auto bias_contig   = bias.contiguous();
    auto output = torch::empty({M, N}, input.options());

    const int total_elements = M * N;
    const int threads = 256;
    const int blocks = (total_elements + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "linear_bias_gelu_cuda", ([&] {
        linear_bias_gelu_kernel<scalar_t><<<blocks, threads>>>(
            input_contig.data_ptr<scalar_t>(),
            weight_contig.data_ptr<scalar_t>(),
            bias_contig.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            M, N, K);
    }));

    TORCH_CHECK(cudaGetLastError() == cudaSuccess,
        "linear_bias_gelu kernel launch failed: ", cudaGetErrorString(cudaGetLastError()));

    return output;
}

// ═══════════════════════════════════════════════════════════════════════════
// PyTorch module registration
// ═══════════════════════════════════════════════════════════════════════════
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("tiled_gemm",        &tiled_gemm_cuda,        "Tiled GEMM: C = A @ B");
    m.def("tiled_gemm_bt",     &tiled_gemm_bt_cuda,     "Tiled GEMM with B transposed: C = A @ B^T");
    m.def("linear_bias_gelu",  &linear_bias_gelu_cuda,  "Fused Linear + Bias + GELU");
    m.doc() = "Optimized CUDA linear operators for multimodal transformer inference";
}
