"""
ops/linear.py — Linear operator wrappers with automatic backend dispatch.

Routes calls to CUDA extension, MLX, or PyTorch native based on runtime
hardware detection.
"""

import torch
from .utils import get_backend, Backend


def tiled_gemm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Tiled GEMM: C = A @ B

    Uses optimized CUDA kernel when available, otherwise falls back
    to PyTorch's built-in matmul (which uses cuBLAS on GPU).

    Args:
        A: [M, K] or [B, M, K]
        B: [K, N] or [B, K, N]

    Returns:
        C: [M, N] or [B, M, N]
    """
    backend = get_backend()

    if backend == Backend.CUDA:
        try:
            import linear_cuda
            return linear_cuda.tiled_gemm(A, B)
        except ImportError:
            pass  # fall through to native
    elif backend == Backend.MLX:
        from csrc.linear.linear_mlx import tiled_gemm as mlx_gemm
        return mlx_gemm(A, B)

    # CPU / native fallback
    return torch.matmul(A, B)


def tiled_gemm_bt(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Tiled GEMM with B transposed: C = A @ B^T

    Equivalent to torch.nn.functional.linear(A, B).

    Args:
        A: [M, K]
        B: [N, K] — weight matrix (transposed internally)

    Returns:
        C: [M, N]
    """
    backend = get_backend()

    if backend == Backend.CUDA:
        try:
            import linear_cuda
            return linear_cuda.tiled_gemm_bt(A, B)
        except ImportError:
            pass
    elif backend == Backend.MLX:
        from csrc.linear.linear_mlx import tiled_gemm_bt as mlx_gemm_bt
        return mlx_gemm_bt(A, B)

    return torch.nn.functional.linear(A, B)


def linear_bias_gelu(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """
    Fused Linear + Bias + GELU: GELU(input @ weight + bias)

    Common pattern in transformer FFN blocks. The CUDA kernel fuses
    all three operations into a single kernel launch, avoiding
    intermediate tensor allocations and memory round-trips.

    Args:
        input:  [M, K]
        weight: [K, N]
        bias:   [N]

    Returns:
        GELU(input @ weight + bias) [M, N]
    """
    backend = get_backend()

    if backend == Backend.CUDA:
        try:
            import linear_cuda
            return linear_cuda.linear_bias_gelu(input, weight, bias)
        except ImportError:
            pass
    elif backend == Backend.MLX:
        from csrc.linear.linear_mlx import linear_bias_gelu as mlx_gelu
        return mlx_gelu(input, weight, bias)

    # PyTorch native fallback (two separate ops)
    hidden = torch.nn.functional.linear(input, weight, bias)
    return torch.nn.functional.gelu(hidden, approximate="tanh")
