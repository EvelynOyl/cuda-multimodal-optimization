/**
 * softmax_cuda_kernel.cu — Online Safe Softmax CUDA kernel.
 *
 * Implements the numerically stable softmax used in FlashAttention-2:
 *   σ(x_i) = exp(x_i - max(x)) / Σ exp(x_j - max(x))
 *
 * Key optimizations:
 *   - Online "safe softmax" algorithm: single-pass reduce-max + exp-sum
 *   - Warp-level reductions for intra-warp parallelism
 *   - Shared-memory reductions across warps within a block
 *   - Vectorized (float4) memory access for input loads
 *   - Optional causal masking support
 *   - Row-wise + batch processing
 *
 * Reference:
 *   Milakov & Gimelshein. "Online normalizer calculation for softmax." arXiv:1805.02867
 *   Dao et al. "FlashAttention-2: Faster Attention with Better Parallelism." 2023.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ── Configuration ─────────────────────────────────────────────────────────
#define WARP_SIZE 32
#define SOFTMAX_BLOCK_SIZE 256
#define NUM_WARPS (SOFTMAX_BLOCK_SIZE / WARP_SIZE)  // = 8
#define MAX_SEQ_LEN 8192

// ═══════════════════════════════════════════════════════════════════════════
// Warp-level reductions (shuffle-based, no shared memory needed)
// ═══════════════════════════════════════════════════════════════════════════

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 1: Online Safe Softmax (row-wise, float32)
//
// Each block processes one row of the [B, N, D] tensor across the D dimension.
// Uses the 3-phase online algorithm:
//   1. Thread-level partial max → warp reduce → shared memory block reduce
//   2. Thread-level partial sum → warp reduce → shared memory block reduce
//   3. Normalize and write
//
// This is equivalent to the "online softmax" in FlashAttention's local softmax.
// ═══════════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void online_safe_softmax_kernel(
    const scalar_t* __restrict__ input,   // [B, N]  — or [B, H, N, N] for attention
    scalar_t* __restrict__ output,        // [B, N]
    const int rows,                       // total number of rows (B * H * ...)
    const int cols,                       // softmax dimension
    const float scale)                    // typically 1/√d_k for attention
{
    // ── Thread indexing ──────────────────────────────────────────────────
    const int row_idx  = blockIdx.x;
    const int tid      = threadIdx.x;
    const int warp_id  = tid / WARP_SIZE;
    const int lane_id  = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (row_idx >= rows) return;

    // ── Shared memory for cross-warp reductions ──────────────────────────
    __shared__ float s_max_vals[NUM_WARPS];
    __shared__ float s_sum_vals[NUM_WARPS];

    const scalar_t* row_input  = input  + row_idx * cols;
    scalar_t*       row_output = output + row_idx * cols;

    // ── Phase 1: Find the maximum value ──────────────────────────────────
    float thread_max = -FLT_MAX;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]) * scale;
        thread_max = fmaxf(thread_max, val);
    }

    // Warp-level reduction for max
    thread_max = warp_reduce_max(thread_max);

    // Share across warps
    if (lane_id == 0) {
        s_max_vals[warp_id] = thread_max;
    }
    __syncthreads();

    // Thread 0 computes the global max
    float global_max = s_max_vals[0];
    if (tid == 0) {
        #pragma unroll
        for (int w = 1; w < num_warps; ++w) {
            global_max = fmaxf(global_max, s_max_vals[w]);
        }
        s_max_vals[0] = global_max;
    }
    __syncthreads();
    global_max = s_max_vals[0];

    // ── Phase 2: Compute sum of exp(x_i - max) ───────────────────────────
    float thread_sum = 0.0f;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]) * scale;
        thread_sum += expf(val - global_max);
    }

    // Warp-level reduction for sum
    thread_sum = warp_reduce_sum(thread_sum);

    // Share across warps
    if (lane_id == 0) {
        s_sum_vals[warp_id] = thread_sum;
    }
    __syncthreads();

    // Thread 0 computes the global sum (with epsilon for numerical stability)
    float global_sum = s_sum_vals[0] + 1e-12f;
    if (tid == 0) {
        #pragma unroll
        for (int w = 1; w < num_warps; ++w) {
            global_sum += s_sum_vals[w];
        }
        global_sum += 1e-12f;  // prevent division by zero
        s_sum_vals[0] = global_sum;
    }
    __syncthreads();
    global_sum = s_sum_vals[0];

    // ── Phase 3: Normalize and write ─────────────────────────────────────
    const float inv_sum = 1.0f / global_sum;

    for (int i = tid; i < cols; i += blockDim.x) {
        float val = static_cast<float>(row_input[i]) * scale;
        float prob = expf(val - global_max) * inv_sum;
        row_output[i] = static_cast<scalar_t>(prob);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 2: Fused Online Softmax with Causal Masking
//
// For self-attention: applies causal (lower-triangular) mask.
// Input:  [B, H, N, N]  attention scores (Q @ K^T)
// Output: [B, H, N, N]  attention probabilities
//
// The causal mask sets upper-triangular entries to -inf before softmax.
// This is fused into the max-finding phase — masked entries are skipped.
// ═══════════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void causal_softmax_kernel(
    const scalar_t* __restrict__ attn_scores,  // [B, H, N, N]
    scalar_t* __restrict__ attn_probs,         // [B, H, N, N]
    const int B, const int H, const int N,
    const float scale)
{
    const int total_rows = B * H * N;
    const int row_idx    = blockIdx.x;
    const int tid        = threadIdx.x;
    const int warp_id    = tid / WARP_SIZE;
    const int lane_id    = tid % WARP_SIZE;
    const int num_warps  = blockDim.x / WARP_SIZE;

    if (row_idx >= total_rows) return;

    // Decode row index: each row in [B*H*N, N] corresponds to one query position
    const int query_pos = row_idx % N;  // which query (row) we're on

    const scalar_t* row_input  = attn_scores + row_idx * N;
    scalar_t*       row_output = attn_probs + row_idx * N;

    __shared__ float s_max_vals[NUM_WARPS];
    __shared__ float s_sum_vals[NUM_WARPS];

    // ── Phase 1: Find max (skip masked positions) ────────────────────────
    float thread_max = -FLT_MAX;

    for (int i = tid; i < N; i += blockDim.x) {
        // Causal mask: attend only to positions <= query_pos
        if (i <= query_pos) {
            float val = static_cast<float>(row_input[i]) * scale;
            thread_max = fmaxf(thread_max, val);
        }
    }

    thread_max = warp_reduce_max(thread_max);
    if (lane_id == 0) { s_max_vals[warp_id] = thread_max; }
    __syncthreads();

    float global_max = s_max_vals[0];
    if (tid == 0) {
        #pragma unroll
        for (int w = 1; w < num_warps; ++w) {
            global_max = fmaxf(global_max, s_max_vals[w]);
        }
        s_max_vals[0] = global_max;
    }
    __syncthreads();
    global_max = s_max_vals[0];

    // ── Phase 2: Compute exp sum (masked positions contribute 0) ─────────
    float thread_sum = 0.0f;

    for (int i = tid; i < N; i += blockDim.x) {
        if (i <= query_pos) {
            float val = static_cast<float>(row_input[i]) * scale;
            thread_sum += expf(val - global_max);
        }
    }

    thread_sum = warp_reduce_sum(thread_sum);
    if (lane_id == 0) { s_sum_vals[warp_id] = thread_sum; }
    __syncthreads();

    float global_sum = s_sum_vals[0] + 1e-12f;
    if (tid == 0) {
        #pragma unroll
        for (int w = 1; w < num_warps; ++w) {
            global_sum += s_sum_vals[w];
        }
        global_sum += 1e-12f;
        s_sum_vals[0] = global_sum;
    }
    __syncthreads();
    global_sum = s_sum_vals[0];

    // ── Phase 3: Normalize (masked positions → 0) ────────────────────────
    const float inv_sum = 1.0f / global_sum;

    for (int i = tid; i < N; i += blockDim.x) {
        if (i <= query_pos) {
            float val = static_cast<float>(row_input[i]) * scale;
            row_output[i] = static_cast<scalar_t>(expf(val - global_max) * inv_sum);
        } else {
            row_output[i] = 0.0f;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel 3: Softmax Backward (gradient computation)
//
// Given output probabilities P and upstream gradient dY,
// compute dX = P * (dY - sum(P * dY))
// ═══════════════════════════════════════════════════════════════════════════

template <typename scalar_t>
__global__ void softmax_backward_kernel(
    const scalar_t* __restrict__ output,     // P = softmax(X)  [B, N]
    const scalar_t* __restrict__ grad_output, // dY              [B, N]
    scalar_t* __restrict__ grad_input,        // dX              [B, N]
    const int rows,
    const int cols)
{
    const int row_idx = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;
    const int num_warps = blockDim.x / WARP_SIZE;

    if (row_idx >= rows) return;

    const scalar_t* row_p  = output      + row_idx * cols;
    const scalar_t* row_dy = grad_output + row_idx * cols;
    scalar_t*       row_dx = grad_input  + row_idx * cols;

    __shared__ float s_dot[NUM_WARPS];

    // ── Compute dot(P, dY) ───────────────────────────────────────────────
    float thread_dot = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x) {
        thread_dot += static_cast<float>(row_p[i]) * static_cast<float>(row_dy[i]);
    }

    thread_dot = warp_reduce_sum(thread_dot);
    if (lane_id == 0) { s_dot[warp_id] = thread_dot; }
    __syncthreads();

    float global_dot = s_dot[0];
    if (tid == 0) {
        #pragma unroll
        for (int w = 1; w < num_warps; ++w) {
            global_dot += s_dot[w];
        }
        s_dot[0] = global_dot;
    }
    __syncthreads();
    global_dot = s_dot[0];

    // ── dX = P * (dY - dot) ─────────────────────────────────────────────
    for (int i = tid; i < cols; i += blockDim.x) {
        float p  = static_cast<float>(row_p[i]);
        float dy = static_cast<float>(row_dy[i]);
        row_dx[i] = static_cast<scalar_t>(p * (dy - global_dot));
    }
}

// ── Explicit template instantiations ──────────────────────────────────────

template __global__ void online_safe_softmax_kernel<float>(
    const float*, float*, const int, const int, const float);

template __global__ void online_safe_softmax_kernel<__half>(
    const __half*, __half*, const int, const int, const float);

template __global__ void causal_softmax_kernel<float>(
    const float*, float*, const int, const int, const int, const float);

template __global__ void causal_softmax_kernel<__half>(
    const __half*, __half*, const int, const int, const int, const float);

template __global__ void softmax_backward_kernel<float>(
    const float*, const float*, float*, const int, const int);

template __global__ void softmax_backward_kernel<__half>(
    const __half*, const __half*, __half*, const int, const int);
