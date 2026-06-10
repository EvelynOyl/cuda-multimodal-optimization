"""
tests/test_linear.py — Tests for Linear operators.

Validates correctness of CUDA/MLX/PyTorch implementations
and measures performance.
"""

import sys
import time
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, "..")

from ops.utils import get_backend, Backend


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def small_inputs():
    """Small tensors for correctness tests."""
    torch.manual_seed(42)
    A = torch.randn(8, 64, dtype=torch.float32)
    B = torch.randn(64, 128, dtype=torch.float32)
    return A, B


@pytest.fixture
def batched_inputs():
    """Batched tensors."""
    torch.manual_seed(42)
    A = torch.randn(4, 16, 256, dtype=torch.float32)
    B = torch.randn(4, 256, 512, dtype=torch.float32)
    return A, B


# ── Correctness Tests ──────────────────────────────────────────────────────

class TestTiledGEMM:
    """Verify tiled_gemm matches torch.matmul."""

    def test_gemm_correctness_small(self, small_inputs):
        """Small matmul: our implementation should match PyTorch."""
        A, B = small_inputs
        backend = get_backend()

        if backend == Backend.CUDA:
            A, B = A.cuda(), B.cuda()

        from ops.linear import tiled_gemm
        C_custom = tiled_gemm(A, B)
        C_ref = torch.matmul(A, B)

        # FP32: expect close-to-exact match
        assert torch.allclose(C_custom, C_ref, rtol=1e-4, atol=1e-6), \
            f"Max diff: {(C_custom - C_ref).abs().max():.6f}"

    def test_gemm_correctness_batched(self, batched_inputs):
        """Batched matmul correctness."""
        A, B = batched_inputs
        backend = get_backend()

        if backend == Backend.CUDA:
            A, B = A.cuda(), B.cuda()

        from ops.linear import tiled_gemm
        C_custom = tiled_gemm(A, B)
        C_ref = torch.matmul(A, B)

        assert C_custom.shape == C_ref.shape
        assert torch.allclose(C_custom, C_ref, rtol=1e-4, atol=1e-5)

    def test_gemm_shape(self, small_inputs):
        """Output shapes are correct."""
        A, B = small_inputs
        from ops.linear import tiled_gemm

        # 2D: [M,K] @ [K,N] → [M,N]
        C = tiled_gemm(A, B)
        assert C.shape == (8, 128)

        # Batched: [B,M,K] @ [B,K,N] → [B,M,N]
        Ab = A.unsqueeze(0).expand(3, -1, -1)
        Bb = B.unsqueeze(0).expand(3, -1, -1)
        Cb = tiled_gemm(Ab, Bb)
        assert Cb.shape == (3, 8, 128)


class TestGEMMBT:
    """Verify tiled_gemm_bt (transposed B)."""

    def test_gemm_bt_correctness(self):
        torch.manual_seed(42)
        A = torch.randn(8, 64, dtype=torch.float32)
        B = torch.randn(128, 64, dtype=torch.float32)  # Weight matrix [N, K]

        from ops.linear import tiled_gemm_bt

        C_custom = tiled_gemm_bt(A, B)
        C_ref = torch.nn.functional.linear(A, B)  # [M, N]

        assert C_custom.shape == (8, 128)
        assert torch.allclose(C_custom, C_ref, rtol=1e-4, atol=1e-6)

    def test_gemm_bt_equivalent_to_gemm(self):
        """gemm_bt(A, B) should equal gemm(A, B^T)."""
        torch.manual_seed(42)
        A = torch.randn(8, 64, dtype=torch.float32)
        B = torch.randn(128, 64, dtype=torch.float32)  # [N, K]

        from ops.linear import tiled_gemm_bt, tiled_gemm

        C1 = tiled_gemm_bt(A, B)
        C2 = tiled_gemm(A, B.T)

        assert torch.allclose(C1, C2, rtol=1e-4, atol=1e-6)


class TestLinearBiasGELU:
    """Verify fused Linear+Bias+GELU."""

    def test_gelu_correctness(self):
        torch.manual_seed(42)
        A = torch.randn(4, 256, dtype=torch.float32)
        W = torch.randn(256, 1024, dtype=torch.float32)
        b = torch.randn(1024, dtype=torch.float32)

        from ops.linear import linear_bias_gelu

        C_custom = linear_bias_gelu(A, W, b)

        # Reference: two-step
        hidden = torch.nn.functional.linear(A, W, b)
        C_ref = torch.nn.functional.gelu(hidden, approximate="tanh")

        assert C_custom.shape == (4, 1024)
        assert torch.allclose(C_custom, C_ref, rtol=1e-4, atol=1e-5)

    def test_gelu_non_negative_for_large_positive(self):
        """GELU should approach identity for large positive inputs."""
        A = torch.full((2, 64), 10.0, dtype=torch.float32)
        W = torch.eye(64, dtype=torch.float32)
        b = torch.zeros(64)

        from ops.linear import linear_bias_gelu

        out = linear_bias_gelu(A, W, b)
        # GELU(10) ≈ 10 (close to identity for large x)
        assert torch.all(out > 9.5)


# ── Performance Benchmarks ─────────────────────────────────────────────────

def test_gemm_performance():
    """Benchmark tiled_gemm vs torch.matmul."""
    if get_backend() == Backend.CPU:
        pytest.skip("Skipping benchmark on CPU")

    torch.manual_seed(42)
    M, K, N = 2048, 4096, 4096
    A = torch.randn(M, K, dtype=torch.float32)
    B = torch.randn(K, N, dtype=torch.float32)

    if torch.cuda.is_available():
        A, B = A.cuda(), B.cuda()

    from ops.linear import tiled_gemm

    # Warmup
    for _ in range(5):
        _ = tiled_gemm(A, B)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    # Benchmark custom
    iters = 20
    start = time.perf_counter()
    for _ in range(iters):
        _ = tiled_gemm(A, B)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    custom_time = (time.perf_counter() - start) / iters

    # Benchmark torch
    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.matmul(A, B)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch_time = (time.perf_counter() - start) / iters

    tflops_custom = (2 * M * K * N) / (custom_time * 1e12)
    tflops_torch  = (2 * M * K * N) / (torch_time * 1e12)

    print(f"\n  Custom tiled_gemm: {custom_time*1000:.2f} ms ({tflops_custom:.2f} TFLOPS)")
    print(f"  Torch matmul:      {torch_time*1000:.2f} ms ({tflops_torch:.2f} TFLOPS)")

    # Custom should be within ~2× of cuBLAS (optimized baseline)
    assert tflops_custom > tflops_torch * 0.3, \
        f"Custom implementation significantly slower than torch ({tflops_custom:.2f} vs {tflops_torch:.2f} TFLOPS)"


# ── MLX Tests ──────────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not (hasattr(torch, '_C') or True),
    reason="MLX tests: run manually on Apple Silicon"
)
class TestMLXLinear:
    """MLX-specific tests (run on macOS with `pytest -m mlx`)."""

    def test_mlx_gemm_shapes(self):
        """MLX matmul produces correct shapes."""
        try:
            from csrc.linear.linear_mlx import tiled_gemm as mlx_gemm
            import mlx.core as mx

            A = mx.random.normal((8, 64))
            B = mx.random.normal((64, 128))
            C = mlx_gemm(A, B)

            assert C.shape == (8, 128)
        except ImportError:
            pytest.skip("MLX not available")

    def test_mlx_gemm_correctness(self):
        """MLX matmul matches PyTorch."""
        try:
            from csrc.linear.linear_mlx import tiled_gemm as mlx_gemm
            import mlx.core as mx

            np.random.seed(42)
            A_np = np.random.randn(8, 64).astype(np.float32)
            B_np = np.random.randn(64, 128).astype(np.float32)

            C_mlx = mlx_gemm(A_np, B_np)
            C_pt = torch.from_numpy(A_np) @ torch.from_numpy(B_np)

            assert np.allclose(np.array(C_mlx), C_pt.numpy(), rtol=1e-4, atol=1e-5)
        except ImportError:
            pytest.skip("MLX not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
