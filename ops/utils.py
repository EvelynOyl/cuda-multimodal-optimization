"""
ops/utils.py — Backend detection and dispatch utilities.

Auto-detects available hardware backends:
  - CUDA:  NVIDIA GPU with compiled CUDA extensions
  - MLX:   Apple Silicon with MLX framework
  - CPU:   PyTorch native ops (fallback)
"""

import os
import sys
import enum
import functools
from typing import Callable, Any


class Backend(enum.Enum):
    CUDA = "cuda"
    MLX  = "mlx"
    CPU  = "cpu"


_backend_cache: Backend | None = None


def detect_backend() -> Backend:
    """
    Auto-detect the optimal backend.
    Priority: CUDA > MLX > CPU

    Returns:
        Backend enum value
    """
    # Check CUDA
    try:
        import torch
        if torch.cuda.is_available():
            # Verify CUDA extensions are importable
            try:
                import linear_cuda
                import softmax_cuda
                return Backend.CUDA
            except ImportError:
                # CUDA available but extensions not built — use torch native
                return Backend.CPU
    except ImportError:
        pass

    # Check MLX (Apple Silicon)
    try:
        import mlx.core
        # Verify Metal is available
        if mlx.core.metal.is_available():
            return Backend.MLX
    except (ImportError, AttributeError):
        pass

    return Backend.CPU


def get_backend() -> Backend:
    """
    Get current backend (cached after first detection).
    Override with environment variable OPS_BACKEND=cuda|mlx|cpu.
    """
    global _backend_cache

    # Honor explicit override
    override = os.environ.get("OPS_BACKEND", "").lower()
    if override in ("cuda", "mlx", "cpu"):
        return Backend(override)

    if _backend_cache is None:
        _backend_cache = detect_backend()

    return _backend_cache


def set_backend(backend: Backend):
    """Manually set the backend."""
    global _backend_cache
    _backend_cache = backend


def backend_dispatch(
    cuda_fn: Callable | None = None,
    mlx_fn: Callable | None = None,
    cpu_fn: Callable | None = None,
):
    """
    Decorator/factory that dispatches a function call to the correct
    backend implementation based on runtime detection.

    Usage:
        dispatcher = backend_dispatch(
            cuda_fn=_cuda_impl,
            mlx_fn=_mlx_impl,
            cpu_fn=_cpu_impl,
        )
        result = dispatcher(*args, **kwargs)
    """

    def dispatcher(*args, **kwargs) -> Any:
        backend = get_backend()

        if backend == Backend.CUDA and cuda_fn is not None:
            return cuda_fn(*args, **kwargs)
        elif backend == Backend.MLX and mlx_fn is not None:
            return mlx_fn(*args, **kwargs)
        elif cpu_fn is not None:
            return cpu_fn(*args, **kwargs)
        else:
            raise RuntimeError(
                f"No implementation available for backend={backend.value}. "
                f"Available: CUDA={cuda_fn is not None}, MLX={mlx_fn is not None}, "
                f"CPU={cpu_fn is not None}"
            )

    return dispatcher
