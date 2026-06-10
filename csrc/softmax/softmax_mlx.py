"""
softmax_mlx.py — Apple MLX implementation of Softmax operators.

Drop-in replacement for the CUDA softmax kernels on Apple Silicon.
Implements the same online safe softmax algorithm in MLX primitives.

MLX provides a built-in `mx.softmax` which is already numerically stable
and GPU-accelerated via Metal. This module wraps it with the same API
as the CUDA extension for drop-in compatibility.

Usage (local Mac development):
    from csrc.softmax.softmax_mlx import online_safe_softmax, causal_softmax, softmax_backward
"""

from typing import Optional
import mlx.core as mx
import numpy as np


def _to_mlx(tensor: "mx.array | np.ndarray") -> mx.array:
    """Convert numpy/torch-like input to MLX array."""
    if isinstance(tensor, mx.array):
        return tensor
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
    except ImportError:
        pass
    if isinstance(tensor, np.ndarray):
        return mx.array(tensor)
    raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def _from_mlx(arr: mx.array, like=None):
    """Return MLX array, optionally converting back to torch."""
    mx.eval(arr)
    if like is not None:
        try:
            import torch
            if isinstance(like, torch.Tensor):
                return torch.from_numpy(np.array(arr)).to(like.device)
        except ImportError:
            pass
    return arr


def online_safe_softmax(
    input,      # [..., D]
    scale: float = 1.0,
) -> mx.array:
    """
    Online safe softmax along the last dimension.

    σ(x_i) = exp(x_i * scale - max) / Σ exp(x_j * scale - max)

    MLX's built-in softmax is already numerically stable (uses the
    same max-subtraction trick internally). We apply scale before
    softmax for compatibility with attention scaling.

    Args:
        input: Input tensor, softmax over last dim
        scale: Scaling factor (typically 1/√d_k for attention)

    Returns:
        Softmax probabilities, same shape as input
    """
    x = _to_mlx(input)

    # Apply scale then softmax — MLX handles numerical stability internally
    scaled = x * scale
    result = mx.softmax(scaled, axis=-1)

    return _from_mlx(result, like=input)


def causal_softmax(
    attn_scores,  # [B, H, N, N]
    scale: float = 1.0,
) -> mx.array:
    """
    Causal-masked softmax for self-attention.

    Applies a lower-triangular mask before softmax so that
    position i can only attend to positions ≤ i.

    Args:
        attn_scores: Attention scores [B, H, N, N]
        scale:       Scaling factor (1/√d_k)

    Returns:
        Causal attention probabilities [B, H, N, N]
    """
    scores = _to_mlx(attn_scores)
    B, H, N, _N2 = scores.shape
    assert N == _N2, f"Last two dims must be square, got ({N}, {_N2})"

    scaled = scores * scale

    # Create causal mask: lower triangular (1) + upper triangular (-inf)
    # MLX equivalent of torch.tril
    mask = mx.tril(mx.ones((N, N)))

    # Add mask: masked positions become -inf, unmasked unchanged
    # masked[i,j] = scaled[i,j] if mask[i,j]==1 else -inf
    neg_inf = mx.full((N, N), float("-inf"))
    masked = mx.where(mask, scaled, neg_inf)

    result = mx.softmax(masked, axis=-1)

    return _from_mlx(result, like=attn_scores)


def softmax_backward(
    output,       # P = softmax(X)  [..., D]
    grad_output,  # dY              [..., D]
) -> mx.array:
    """
    Backward pass for softmax.

    Given output probabilities P and upstream gradient dY:
        dX = P * (dY - Σ_j P_j * dY_j)

    This is the standard softmax gradient. For MLX we compute it
    explicitly since mlx.core doesn't have a dedicated softmax_backward.

    Args:
        output:      Softmax probabilities P
        grad_output: Upstream gradient dY

    Returns:
        Gradient w.r.t. input dX, same shape
    """
    P = _to_mlx(output)
    dY = _to_mlx(grad_output)

    # dX = P * (dY - sum(P * dY, axis=-1, keepdims=True))
    dot = (P * dY).sum(axis=-1, keepdims=True)
    dX = P * (dY - dot)

    return _from_mlx(dX, like=output)


# ── Utility: flash-attention-style scaled dot-product attention ────────────

def scaled_dot_product_attention(
    Q,         # [B, H, N, d_k]
    K,         # [B, H, N, d_k]
    V,         # [B, H, N, d_v]
    scale: Optional[float] = None,
    causal: bool = True,
) -> mx.array:
    """
    Scaled dot-product attention using MLX primitives.

    This is a reference implementation for testing against the
    CUDA flash-attention path. It computes:
        Attention(Q, K, V) = softmax(Q @ K^T / √d_k + mask) @ V

    Args:
        Q: Query tensor  [B, H, N, d_k]
        K: Key tensor    [B, H, N, d_k]
        V: Value tensor  [B, H, N, d_v]
        scale: Scale factor (default: 1/√d_k)
        causal: Whether to apply causal masking

    Returns:
        Attention output [B, H, N, d_v]
    """
    B, H, N, d_k = Q.shape
    d_v = V.shape[-1]

    if scale is None:
        scale = 1.0 / np.sqrt(d_k)

    # Q @ K^T → [B, H, N, N]
    scores = Q @ K.transpose(0, 1, 3, 2)  # MLX supports batched transpose

    if causal:
        probs = causal_softmax(scores, scale=scale)
    else:
        probs = online_safe_softmax(scores, scale=scale)

    # probs @ V → [B, H, N, d_v]
    out = probs @ V

    return _from_mlx(out, like=Q)


# ── Benchmark ──────────────────────────────────────────────────────────────

def benchmark_softmax(B: int = 4, H: int = 16, N: int = 2048, warmup: int = 5, iters: int = 50):
    """Benchmark softmax throughput on Apple Silicon."""
    import time

    x = mx.random.normal((B * H, N))

    # Warmup
    for _ in range(warmup):
        _ = mx.softmax(x, axis=-1)
        mx.eval(_)

    start = time.perf_counter()
    for _ in range(iters):
        _ = mx.softmax(x, axis=-1)
        mx.eval(_)
    elapsed = time.perf_counter() - start

    total_tokens = B * H * N * iters
    print(f"[MLX softmax] {B}x{H}x{N}: {elapsed/iters*1000:.3f} ms/iter, "
          f"{total_tokens/elapsed/1e6:.2f} M tokens/s")

    return elapsed / iters


if __name__ == "__main__":
    print("=== MLX Softmax Operator Tests (Apple Silicon) ===\n")

    # Test 1: Basic safe softmax
    x = mx.random.normal((16, 512))
    p = online_safe_softmax(x, scale=1.0)
    # Verify: probabilities sum to 1 along last dim
    sums = p.sum(axis=-1)
    assert mx.allclose(sums, mx.ones_like(sums), rtol=1e-4), f"Sums: {sums}"
    assert mx.all(p >= 0), "Negative probabilities!"
    print(f"Test 1 - safe_softmax: shape {p.shape}, sums ≈ 1  ✓")

    # Test 2: Scaled softmax (attention scale)
    d_k = 64
    x2 = mx.random.normal((4, 16, 256, d_k))
    p2 = online_safe_softmax(x2, scale=1.0 / np.sqrt(d_k))
    sums2 = p2.sum(axis=-1)
    assert mx.allclose(sums2, mx.ones_like(sums2), rtol=1e-4)
    print(f"Test 2 - scaled:      shape {p2.shape}, scale=1/√{d_k}  ✓")

    # Test 3: Causal softmax
    B, H, N = 2, 4, 64
    scores = mx.random.normal((B, H, N, N))
    probs = causal_softmax(scores, scale=1.0 / np.sqrt(64))
    # Check upper triangular is zero
    for i in range(N):
        for j in range(N):
            if j > i:
                assert mx.all(probs[:, :, i, j] == 0.0), f"Causal mask leak at ({i},{j})"
    sums3 = probs.sum(axis=-1)
    assert mx.allclose(sums3, mx.ones_like(sums3), rtol=1e-4)
    print(f"Test 3 - causal:      shape {probs.shape}, upper-tri → 0  ✓")

    # Test 4: Softmax backward
    p4 = mx.softmax(mx.random.normal((8, 256)), axis=-1)
    dy = mx.random.normal((8, 256))
    dx = softmax_backward(p4, dy)
    # Verify: gradient sums to 0 per row (since softmax output sums to 1)
    dx_sums = dx.sum(axis=-1)
    assert mx.allclose(dx_sums, mx.zeros_like(dx_sums), atol=1e-5), f"Gradient sums: {dx_sums}"
    print(f"Test 4 - backward:    shape {dx.shape}, ΣdX ≈ 0  ✓")

    # Test 5: Scaled dot-product attention
    Q = mx.random.normal((2, 8, 256, 64))
    K = mx.random.normal((2, 8, 256, 64))
    V = mx.random.normal((2, 8, 256, 64))
    attn_out = scaled_dot_product_attention(Q, K, V, causal=True)
    print(f"Test 5 - SDPA:        {Q.shape} → {attn_out.shape}  ✓")

    print("\n--- Benchmark ---")
    benchmark_softmax(B=4, H=16, N=2048)
