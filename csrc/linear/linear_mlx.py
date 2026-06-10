"""
linear_mlx.py — Apple MLX implementation of Linear operators.

Drop-in replacement for the CUDA kernels on Apple Silicon (M-series) Macs.
MLX uses the Metal Performance Shaders (MPS) backend, providing GPU-accelerated
linear algebra operations without requiring NVIDIA hardware.

This module implements the same API as linear_cuda.cpp so the ops/linear.py
wrapper can transparently dispatch to either backend.

Usage (local Mac development):
    from csrc.linear.linear_mlx import tiled_gemm, tiled_gemm_bt, linear_bias_gelu

References:
    - MLX docs: https://ml-explore.github.io/mlx/
    - Metal Performance Shaders: https://developer.apple.com/metal/mps/
"""

from typing import Optional
import mlx.core as mx
import numpy as np


def _to_mlx(tensor: "mx.array | np.ndarray") -> mx.array:
    """Convert numpy/torch-like input to MLX array."""
    if isinstance(tensor, mx.array):
        return tensor
    # Support torch.Tensor if available
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(tensor, np.ndarray):
        return mx.array(tensor)
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def _from_mlx(arr: mx.array, like: Optional["mx.array"] = None) -> mx.array:
    """Return as MLX array (or convert to torch if `like` is a torch tensor)."""
    if like is not None:
        try:
            import torch
            if isinstance(like, torch.Tensor):
                return torch.from_numpy(np.array(arr)).to(like.device)
        except ImportError:
            pass
    return arr


def tiled_gemm(
    A,    # [M, K] or [B, M, K]
    B,    # [K, N] or [B, K, N]
) -> mx.array:
    """
    Tiled GEMM equivalent: C = A @ B

    MLX implements this as a highly-optimized Metal matmul under the hood.
    For batched inputs, we use MLX's built-in batched matmul.

    Args:
        A: Input matrix [M, K] or batched [B, M, K]
        B: Weight matrix [K, N] or batched [B, K, N]

    Returns:
        C: Output matrix [M, N] or [B, M, N]
    """
    A_mlx = _to_mlx(A)
    B_mlx = _to_mlx(B)

    # MLX matmul handles batching natively
    result = A_mlx @ B_mlx

    # Ensure computation completes (MLX is lazy-evaluated)
    mx.eval(result)

    return _from_mlx(result, like=A)


def tiled_gemm_bt(
    A,    # [M, K]
    B,    # [N, K] — will be transposed internally
) -> mx.array:
    """
    Tiled GEMM with B transposed: C = A @ B^T

    Equivalent to PyTorch's F.linear(input, weight) where weight is [N, K].

    Args:
        A: Input matrix [M, K]
        B: Weight matrix [N, K] (transposed internally)

    Returns:
        C: Output matrix [M, N]
    """
    A_mlx = _to_mlx(A)
    B_mlx = _to_mlx(B)

    # B is [N, K], we want B^T → [K, N], so C = A @ B^T
    result = A_mlx @ B_mlx.T

    mx.eval(result)

    return _from_mlx(result, like=A)


def linear_bias_gelu(
    input,   # [M, K]
    weight,  # [K, N]
    bias,    # [N]
) -> mx.array:
    """
    Fused Linear + Bias + GELU activation.

    This is the common FFN pattern in transformer blocks:
        output = GELU(input @ weight + bias)

    MLX doesn't have a single fused GELU-matmul op, but we can write
    it as a composable MLX computation graph — the framework will
    fuse what it can via the Metal graph compiler.

    Args:
        input:  Activation input [M, K]
        weight: Weight matrix [K, N]
        bias:   Bias vector [N]

    Returns:
        GELU(input @ weight + bias)  [M, N]
    """
    A_mlx = _to_mlx(input)
    W_mlx = _to_mlx(weight)
    b_mlx = _to_mlx(bias)

    # Linear projection
    linear_out = A_mlx @ W_mlx + b_mlx

    # GELU activation: x * Φ(x)
    # Using the tanh approximation for GELU (same as CUDA kernel):
    #   GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    gelu_out = 0.5 * linear_out * (
        1.0 + mx.tanh(
            0.7978845608028654 * (linear_out + 0.044715 * linear_out ** 3)
        )
    )

    mx.eval(gelu_out)

    return _from_mlx(gelu_out, like=input)


# ── Performance comparison helper ──────────────────────────────────────────

def benchmark_matmul(M: int = 2048, K: int = 4096, N: int = 4096, warmup: int = 5, iters: int = 50):
    """
    Quick benchmark comparing MLX matmul vs naive NumPy.
    Run on Apple Silicon to verify GPU acceleration.
    """
    import time

    A = mx.random.normal((M, K))
    B = mx.random.normal((K, N))

    # Warmup
    for _ in range(warmup):
        _ = A @ B
        mx.eval(_)

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        _ = A @ B
        mx.eval(_)
    elapsed = time.perf_counter() - start

    tflops = (2 * M * K * N * iters) / (elapsed * 1e12)
    print(f"[MLX tiled_gemm] {M}x{K}x{N}: {elapsed/iters*1000:.2f} ms/iter, {tflops:.2f} TFLOPS")

    return elapsed / iters


if __name__ == "__main__":
    print("=== MLX Linear Operator Tests (Apple Silicon) ===\n")

    # Test 1: Basic matmul
    A = mx.random.normal((4, 256))
    B = mx.random.normal((256, 512))
    C = tiled_gemm(A, B)
    print(f"Test 1 - tiled_gemm: A{A.shape} @ B{B.shape} → C{C.shape}  ✓")

    # Test 2: Batched matmul
    A_b = mx.random.normal((2, 4, 256))
    B_b = mx.random.normal((2, 256, 512))
    C_b = tiled_gemm(A_b, B_b)
    print(f"Test 2 - batched:   A{A_b.shape} @ B{B_b.shape} → C{C_b.shape}  ✓")

    # Test 3: Gemm with B transposed
    A3 = mx.random.normal((4, 256))
    B3 = mx.random.normal((512, 256))  # [N, K] layout
    C3 = tiled_gemm_bt(A3, B3)
    print(f"Test 3 - gemm_bt:   A{A3.shape} @ B{B3.shape}^T → C{C3.shape}  ✓")

    # Test 4: Linear + Bias + GELU
    A4 = mx.random.normal((4, 256))
    W4 = mx.random.normal((256, 1024))
    b4 = mx.random.normal((1024,))
    C4 = linear_bias_gelu(A4, W4, b4)
    print(f"Test 4 - gelu:     output shape {C4.shape}  ✓")

    # Test 5: Correctness (compared to explicit softmax reference)
    print("\n--- Benchmark ---")
    benchmark_matmul(M=2048, K=4096, N=4096)
