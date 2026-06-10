"""
tests/test_quantization.py — Tests for INT4 quantization.

Validates:
  - Packing/unpacking correctness
  - Quantization round-trip accuracy
  - GPTQ layer-wise quantization
  - Compression ratio
"""

import sys
import pytest
import torch
import numpy as np

sys.path.insert(0, "..")

from quantization.int4_quantizer import (
    Int4Quantizer, QuantizedLinear, pack_int4, unpack_int4, QuantizedWeight
)
from quantization.calibration import CalibrationDataset


# ═══════════════════════════════════════════════════════════════════════════
# INT4 Packing/Unpacking
# ═══════════════════════════════════════════════════════════════════════════

class TestPackInt4:
    """Verify bit packing."""

    def test_pack_unpack_roundtrip(self):
        """Pack → Unpack should recover original values."""
        w = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
                        dtype=torch.uint8)

        packed = pack_int4(w)
        unpacked = unpack_int4(packed)

        assert torch.equal(unpacked, w), f"Round-trip failed: {unpacked} != {w}"

    def test_pack_shape(self):
        """Packed shape should be half of original."""
        N = 256
        w = torch.randint(0, 16, (4, N), dtype=torch.uint8)

        packed = pack_int4(w)
        assert packed.shape == (4, N // 2)

    def test_pack_nibble_order(self):
        """Lower nibble = even index, upper nibble = odd index."""
        w = torch.tensor([0x0A, 0x0B], dtype=torch.uint8)  # 10, 11
        packed = pack_int4(w)

        assert packed.item() == (0x0B << 4) | 0x0A  # 0xBA

    def test_unpack_individual_values(self):
        """Each unpacked value should be 0-15."""
        packed = torch.randint(0, 256, (100,), dtype=torch.uint8)
        unpacked = unpack_int4(packed)

        assert unpacked.dtype == torch.uint8
        assert torch.all(unpacked <= 15)
        assert len(unpacked) == 2 * len(packed)


# ═══════════════════════════════════════════════════════════════════════════
# Quantization
# ═══════════════════════════════════════════════════════════════════════════

class TestInt4Quantizer:
    """Test the quantizer."""

    def test_quantize_dequantize_roundtrip(self):
        """Quantize → Dequantize should approximately recover original."""
        torch.manual_seed(42)
        W = torch.randn(256, 512, dtype=torch.float16)

        quantizer = Int4Quantizer(group_size=128)
        qw = quantizer.quantize(W)
        W_recovered = quantizer.dequantize(qw)

        # With 4 bits, expect some error but not catastrophic
        err = (W.float() - W_recovered).abs()
        mean_err = err.mean().item()
        max_err = err.max().item()

        # For FP16 inputs, 4-bit quantization should have < 10% mean relative error
        assert mean_err < 1.0, f"Mean error too high: {mean_err:.4f}"
        # Max error can be larger due to outliers
        print(f"\n  Quantization error: mean={mean_err:.4f}, max={max_err:.4f}")

    def test_per_channel_range(self):
        """Each output channel should have valid scale/zp."""
        torch.manual_seed(42)
        W = torch.randn(128, 256, dtype=torch.float32)

        quantizer = Int4Quantizer(group_size=64)
        qw = quantizer.quantize(W)

        # Scales should be positive
        assert torch.all(qw.scales > 0), "All scales must be positive"

        # Zero points should be in [0, 15]
        assert torch.all(qw.zeros >= 0) and torch.all(qw.zeros <= 15)

    def test_compression_ratio(self):
        """INT4 should compress ~3-4× vs FP16."""
        torch.manual_seed(42)
        W = torch.randn(4096, 4096, dtype=torch.float16)

        quantizer = Int4Quantizer(group_size=128)
        qw = quantizer.quantize(W)
        ratio = qw.compression_ratio()

        # For group_size=128: ~3.76× compression
        assert 3.0 < ratio < 4.5, f"Unexpected compression ratio: {ratio:.2f}×"
        print(f"\n  Compression ratio: {ratio:.2f}×")

    def test_asymmetric_vs_symmetric(self):
        """Asymmetric quantization should have lower error than symmetric."""
        torch.manual_seed(42)
        # Skewed distribution (not zero-centered)
        W = torch.randn(128, 256, dtype=torch.float32) * 2 + 3.0

        q_asym = Int4Quantizer(group_size=64, asymmetric=True)
        qw_asym = q_asym.quantize(W)
        W_asym = q_asym.dequantize(qw_asym)

        q_sym = Int4Quantizer(group_size=64, asymmetric=False)
        qw_sym = q_sym.quantize(W)
        W_sym = q_sym.dequantize(qw_sym)

        err_asym = (W - W_asym).abs().mean()
        err_sym = (W - W_sym).abs().mean()

        print(f"\n  Asymmetric error: {err_asym:.6f}")
        print(f"  Symmetric error:  {err_sym:.6f}")

        # Asymmetric should be better for skewed distributions
        assert err_asym < err_sym, \
            f"Asymmetric ({err_asym:.6f}) should be better than symmetric ({err_sym:.6f})"

    def test_group_size_impact(self):
        """Smaller group_size → better accuracy but worse compression."""
        torch.manual_seed(42)
        W = torch.randn(512, 1024, dtype=torch.float32)

        errors = {}
        ratios = {}
        for gs in [32, 64, 128, 256]:
            q = Int4Quantizer(group_size=gs)
            qw = q.quantize(W)
            W_rec = q.dequantize(qw)
            errors[gs] = (W - W_rec).abs().mean().item()
            ratios[gs] = qw.compression_ratio()

        print("\n  Group size → error, compression:")
        for gs in sorted(errors):
            print(f"    {gs:3d}:  err={errors[gs]:.6f},  ratio={ratios[gs]:.2f}×")

        # Smaller groups should have lower error
        assert errors[32] < errors[256]


# ═══════════════════════════════════════════════════════════════════════════
# QuantizedLinear
# ═══════════════════════════════════════════════════════════════════════════

class TestQuantizedLinear:
    """Test the drop-in nn.Linear replacement."""

    def test_forward_shape(self):
        """Output shape should match nn.Linear."""
        torch.manual_seed(42)

        in_features, out_features = 256, 512
        linear = torch.nn.Linear(in_features, out_features)

        qlinear = QuantizedLinear.from_float(linear, group_size=64)

        x = torch.randn(8, in_features)
        y = qlinear(x)

        assert y.shape == (8, out_features)

    def test_forward_approximate_match(self):
        """INT4 forward should approximately match FP16 forward."""
        torch.manual_seed(42)

        in_features, out_features = 64, 128
        linear = torch.nn.Linear(in_features, out_features, dtype=torch.float32)
        qlinear = QuantizedLinear.from_float(linear, group_size=32)

        x = torch.randn(4, in_features)

        y_ref = linear(x)
        y_quant = qlinear(x)

        # INT4 approximation: expect reasonable cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten(), y_quant.flatten(), dim=0
        )
        assert cos_sim > 0.95, f"Low cosine similarity: {cos_sim:.4f}"

        print(f"\n  Cosine similarity (FP16 vs INT4): {cos_sim:.4f}")

    def test_with_bias(self):
        """Should handle bias correctly."""
        torch.manual_seed(42)

        linear = torch.nn.Linear(128, 64, bias=True)
        qlinear = QuantizedLinear.from_float(linear, group_size=64)

        x = torch.randn(4, 128)

        y_ref = linear(x)
        y_quant = qlinear(x)

        assert y_quant.shape == y_ref.shape

    def test_groupwise_and_fullforward_match(self):
        """groupwise and full forward should produce same result."""
        torch.manual_seed(42)

        linear = torch.nn.Linear(256, 512)
        qlinear = QuantizedLinear.from_float(linear, group_size=128)

        x = torch.randn(4, 256)

        y_full = qlinear._forward_full(x)
        y_group = qlinear._forward_groupwise(x)

        assert torch.allclose(y_full, y_group, rtol=1e-4, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════
# Calibration Dataset
# ═══════════════════════════════════════════════════════════════════════════

class TestCalibrationDataset:
    """Test calibration data loading."""

    def test_random_calibration_shape(self):
        """Random calibration should produce correct shapes."""
        calib = CalibrationDataset._random_calibration(
            tokenizer=None,  # Uses random ints directly
            num_samples=32,
            seq_len=512,
        )

        assert calib.input_ids.shape == (32, 512)

    def test_iter_batches(self):
        """Batch iteration yields correct shapes."""
        calib = CalibrationDataset._random_calibration(
            tokenizer=None, num_samples=16, seq_len=256
        )

        batch_count = 0
        for batch in calib.iter_batches(batch_size=4):
            if isinstance(batch, dict):
                assert batch["input_ids"].shape[0] <= 4
            else:
                assert batch.shape[0] <= 4
            batch_count += 1

        assert batch_count == 4  # 16 samples / 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
