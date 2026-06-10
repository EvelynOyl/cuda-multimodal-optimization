"""
ops — Optimized CUDA/MLX operators for multimodal transformers.

Auto-detects the available backend:
  - CUDA: Uses compiled CUDA extensions (linear_cuda, softmax_cuda)
  - MLX:  Falls back to Apple MLX (Metal Performance Shaders) on Mac
  - CPU:  Falls back to PyTorch native ops as last resort
"""

from .utils import get_backend, Backend
from .linear import tiled_gemm, tiled_gemm_bt, linear_bias_gelu
from .softmax import online_safe_softmax, causal_softmax, softmax_backward

__all__ = [
    "get_backend",
    "Backend",
    "tiled_gemm",
    "tiled_gemm_bt",
    "linear_bias_gelu",
    "online_safe_softmax",
    "causal_softmax",
    "softmax_backward",
]
