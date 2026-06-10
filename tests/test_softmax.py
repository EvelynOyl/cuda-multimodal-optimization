"""
tests/test_softmax.py — Tests for Softmax operators.

Validates numerical stability, causal masking, and gradient correctness.
"""

import sys
import time
import pytest
import torch
import numpy as np

sys.path.insert(0, "..")

from ops.utils import get_backend, Backend


# ── Correctness Tests ──────────────────────────────────────────────────────

class TestOnlineSafeSoftmax:
    """Verify online safe softmax correctness."""

    def test_probabilities_sum_to_one(self):
        """Each row should sum to 1."""
        torch.manual_seed(42)
        x = torch.randn(16, 512, dtype=torch.float32)

        from ops.softmax import online_safe_softmax

        p = online_safe_softmax(x, scale=1.0)
        row_sums = p.sum(dim=-1)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4)

    def test_all_non_negative(self):
        """Probabilities should be non-negative."""
        torch.manual_seed(42)
        x = torch.randn(16, 256, dtype=torch.float32)

        from ops.softmax import online_safe_softmax

        p = online_safe_softmax(x)
        assert torch.all(p >= 0)

    def test_numerical_stability_large_values(self):
        """Should handle large values without overflow."""
        # Values that would overflow naive exp(x)
        x = torch.tensor([[1000.0, 2000.0, 3000.0]], dtype=torch.float32)

        from ops.softmax import online_safe_softmax

        p = online_safe_softmax(x)
        row_sum = p.sum()

        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()
        assert torch.allclose(row_sum, torch.tensor(1.0), rtol=1e-4)

    def test_numerical_stability_negative_values(self):
        """Should handle very negative values."""
        x = torch.tensor([[-3000.0, -2000.0, -1000.0]], dtype=torch.float32)

        from ops.softmax import online_safe_softmax

        p = online_safe_softmax(x)
        row_sum = p.sum()

        assert not torch.isnan(p).any()
        assert not torch.isinf(p).any()
        assert torch.allclose(row_sum, torch.tensor(1.0), rtol=1e-4)

    def test_matches_pytorch_softmax(self):
        """Online safe softmax should match PyTorch softmax."""
        torch.manual_seed(42)
        x = torch.randn(32, 128, dtype=torch.float32)

        from ops.softmax import online_safe_softmax

        p_custom = online_safe_softmax(x, scale=1.0)
        p_ref = torch.softmax(x, dim=-1)

        assert torch.allclose(p_custom, p_ref, rtol=1e-5, atol=1e-6)

    def test_scale_effect(self):
        """Scale should be applied before softmax — equivalent to scaling input."""
        torch.manual_seed(42)
        x = torch.randn(16, 64, dtype=torch.float32)
        scale = 0.125  # 1/√64

        from ops.softmax import online_safe_softmax

        p1 = online_safe_softmax(x, scale=scale)
        p2 = torch.softmax(x * scale, dim=-1)

        assert torch.allclose(p1, p2, rtol=1e-5)

    def test_batched_input(self):
        """Should work with batched/multi-dimensional inputs."""
        torch.manual_seed(42)
        x = torch.randn(4, 8, 16, 64, dtype=torch.float32)

        from ops.softmax import online_safe_softmax

        p = online_safe_softmax(x)
        row_sums = p.sum(dim=-1)

        assert p.shape == x.shape
        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4)


class TestCausalSoftmax:
    """Verify causal softmax correctness."""

    def test_upper_triangular_is_zero(self):
        """Masked (upper-triangular) positions should be 0."""
        B, H, N = 2, 4, 32
        torch.manual_seed(42)
        scores = torch.randn(B, H, N, N, dtype=torch.float32)

        from ops.softmax import causal_softmax

        probs = causal_softmax(scores, scale=1.0)

        for i in range(N):
            for j in range(N):
                if j > i:
                    assert torch.all(probs[:, :, i, j] == 0.0), \
                        f"Position ({i},{j}) should be masked but got {probs[0,0,i,j]:.6f}"

    def test_rows_sum_to_one(self):
        """Each row should still sum to 1 (over unmasked positions)."""
        B, H, N = 2, 4, 32
        torch.manual_seed(42)
        scores = torch.randn(B, H, N, N, dtype=torch.float32)

        from ops.softmax import causal_softmax

        probs = causal_softmax(scores, scale=1.0)
        row_sums = probs.sum(dim=-1)

        assert torch.allclose(row_sums, torch.ones_like(row_sums), rtol=1e-4), \
            f"Max deviation from 1: {(row_sums - 1).abs().max():.6f}"

    def test_matches_pytorch_causal(self):
        """Should match PyTorch causal softmax."""
        B, H, N = 2, 4, 16
        torch.manual_seed(42)
        scores = torch.randn(B, H, N, N, dtype=torch.float32)

        from ops.softmax import causal_softmax

        probs_custom = causal_softmax(scores, scale=1.0)

        # PyTorch reference
        mask = torch.tril(torch.ones(N, N)).bool()
        masked = scores.masked_fill(~mask, float("-inf"))
        probs_ref = torch.softmax(masked, dim=-1)

        assert torch.allclose(probs_custom, probs_ref, rtol=1e-5)


class TestSoftmaxBackward:
    """Verify softmax backward gradient."""

    def test_gradient_sums_to_zero(self):
        """Since softmax output sums to 1, the gradient should sum to 0 per row."""
        torch.manual_seed(42)
        p = torch.softmax(torch.randn(8, 256, dtype=torch.float32), dim=-1)
        dy = torch.randn(8, 256, dtype=torch.float32)

        from ops.softmax import softmax_backward

        dx = softmax_backward(p, dy)
        dx_sums = dx.sum(dim=-1)

        assert torch.allclose(dx_sums, torch.zeros_like(dx_sums), atol=1e-5), \
            f"Gradient sums: {dx_sums}"

    def test_matches_pytorch_autograd(self):
        """Our analytical backward should match PyTorch autograd."""
        torch.manual_seed(42)
        x = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
        dy = torch.randn(8, 64, dtype=torch.float32)

        # PyTorch autograd
        p = torch.softmax(x, dim=-1)
        p.backward(dy, retain_graph=True)
        dx_ref = x.grad.clone()

        # Our analytical backward
        from ops.softmax import softmax_backward
        dx_custom = softmax_backward(p.detach(), dy)

        assert torch.allclose(dx_custom, dx_ref, rtol=1e-4, atol=1e-5)


# ── Performance ────────────────────────────────────────────────────────────

def test_softmax_performance():
    """Benchmark online safe softmax."""
    if get_backend() == Backend.CPU:
        pytest.skip("Skipping benchmark on CPU")

    torch.manual_seed(42)
    B, N, D = 4, 4096, 4096  # Typical attention map
    x = torch.randn(B * N, D, dtype=torch.float32)

    if torch.cuda.is_available():
        x = x.cuda()

    from ops.softmax import online_safe_softmax

    # Warmup
    for _ in range(5):
        _ = online_safe_softmax(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None

    iters = 30
    start = time.perf_counter()
    for _ in range(iters):
        _ = online_safe_softmax(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    custom_time = (time.perf_counter() - start) / iters

    start = time.perf_counter()
    for _ in range(iters):
        _ = torch.softmax(x, dim=-1)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
    torch_time = (time.perf_counter() - start) / iters

    print(f"\n  Custom safe_softmax: {custom_time*1000:.2f} ms")
    print(f"  Torch softmax:       {torch_time*1000:.2f} ms")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
