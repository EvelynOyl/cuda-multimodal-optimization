"""
ops/softmax.py — Softmax operator wrappers with automatic backend dispatch.

Routes calls to CUDA extension, MLX, or PyTorch native based on runtime
hardware detection.
"""

from typing import Optional
import torch
from .utils import get_backend, Backend


def online_safe_softmax(
    input: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Online safe softmax along last dimension.

    σ(x_i) = exp(x_i * scale - max) / Σ exp(x_j * scale - max)

    The CUDA kernel uses an online algorithm that computes max and sum
    in a single pass, avoiding numerical overflow.

    Args:
        input: [..., D] — softmax over last dim
        scale: Scaling factor (use 1/√d_k for attention)

    Returns:
        Softmax probabilities, same shape as input
    """
    backend = get_backend()

    if backend == Backend.CUDA:
        try:
            import softmax_cuda
            return softmax_cuda.online_safe_softmax(input, scale)
        except ImportError:
            pass
    elif backend == Backend.MLX:
        from csrc.softmax.softmax_mlx import online_safe_softmax as mlx_sm
        return mlx_sm(input, scale)

    # PyTorch native fallback (also numerically stable)
    return torch.softmax(input * scale, dim=-1)


def causal_softmax(
    attn_scores: torch.Tensor,
    scale: float = 1.0,
) -> torch.Tensor:
    """
    Causal-masked softmax for self-attention.

    Position i can only attend to positions ≤ i. Upper-triangular
    entries are set to -inf before softmax.

    Args:
        attn_scores: [B, H, N, N] attention scores (Q @ K^T)
        scale:       Scaling factor (1/√d_k)

    Returns:
        Attention probabilities [B, H, N, N] with causal mask applied
    """
    backend = get_backend()

    if backend == Backend.CUDA:
        try:
            import softmax_cuda
            return softmax_cuda.causal_softmax(attn_scores, scale)
        except ImportError:
            pass
    elif backend == Backend.MLX:
        from csrc.softmax.softmax_mlx import causal_softmax as mlx_causal
        return mlx_causal(attn_scores, scale)

    # PyTorch native fallback
    N = attn_scores.size(-1)
    mask = torch.tril(torch.ones(N, N, device=attn_scores.device)).bool()
    scaled = attn_scores * scale
    scaled = scaled.masked_fill(~mask, float("-inf"))
    return torch.softmax(scaled, dim=-1)


def softmax_backward(
    output: torch.Tensor,
    grad_output: torch.Tensor,
) -> torch.Tensor:
    """
    Backward pass for softmax.

    dX = P * (dY - Σ_j P_j * dY_j)   where P = softmax(X)

    Args:
        output:      Softmax probabilities P
        grad_output: Upstream gradient dY

    Returns:
        Gradient dX w.r.t. softmax input
    """
    backend = get_backend()

    if backend == Backend.CUDA:
        try:
            import softmax_cuda
            return softmax_cuda.softmax_backward(output, grad_output)
        except ImportError:
            pass
    elif backend == Backend.MLX:
        from csrc.softmax.softmax_mlx import softmax_backward as mlx_bwd
        return mlx_bwd(output, grad_output)

    # PyTorch: use autograd
    # Create a tensor that requires grad and do the backward
    # (This is less efficient but correct as fallback)
    P = output.detach().requires_grad_(True)
    # We can't easily run backward without autograd context,
    # so use the analytical formula
    dot = (P * grad_output).sum(dim=-1, keepdim=True)
    dX = P * (grad_output - dot)
    return dX
