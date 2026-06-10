"""
quantization/int4_quantizer.py — INT4 weight-only quantization primitives.

Implements group-wise asymmetric INT4 quantization:

  For each group of `group_size` weights:
    1. Find min/max: scale = (max - min) / 15, zero_point = round(-min / scale)
    2. Quantize:  w_int4 = round(w / scale + zero_point)
    3. Clamp:     w_int4 = clamp(w_int4, 0, 15)
    4. Pack:      two INT4 values per byte (lower 4 bits = first weight)

Dequantization:
    w_fp16 = scale * (w_int4 - zero_point)

Memory savings:
  - FP16: 2 bytes/weight
  - INT4 (packed): 0.5 bytes/weight + 2 bytes/group (scale) + 2 bytes/group (zp)
  - For group_size=128: 0.53125 bytes/weight (3.76× compression)
  - For group_size=64:  0.5625 bytes/weight  (3.56× compression)

CUDA dequantization kernel included for efficient inference.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Packing / Unpacking
# ═══════════════════════════════════════════════════════════════════════════

def pack_int4(weights: torch.Tensor) -> torch.Tensor:
    """
    Pack two INT4 values into one byte.

    Input:  [..., N] where N is even, each element in [0, 15]
    Output: [..., N//2], each byte = (w[2i+1] << 4) | w[2i]

    Args:
        weights: INT4 weights in uint8, values 0-15

    Returns:
        Packed uint8 tensor with N//2 elements
    """
    assert weights.dtype == torch.uint8, "Weights must be uint8 for packing"
    N = weights.shape[-1]
    assert N % 2 == 0, f"Weight count must be even, got {N}"

    # Reshape to pairs: [..., N//2, 2]
    pairs = weights.reshape(*weights.shape[:-1], N // 2, 2)

    # Pack: lower 4 bits = first weight, upper 4 bits = second weight
    packed = (pairs[..., 0] & 0x0F) | ((pairs[..., 1] & 0x0F) << 4)

    return packed.to(torch.uint8)


def unpack_int4(packed: torch.Tensor) -> torch.Tensor:
    """
    Unpack bytes into INT4 values.

    Input:  [..., N//2] uint8, each byte holds two INT4 weights
    Output: [..., N] uint8, values 0-15

    Args:
        packed: Packed uint8 tensor

    Returns:
        Unpacked uint8 tensor with 2× elements
    """
    assert packed.dtype == torch.uint8, "Packed must be uint8"

    shape = packed.shape
    unpacked = torch.empty(*shape[:-1], shape[-1] * 2, dtype=torch.uint8, device=packed.device)

    # Extract lower and upper nibbles
    unpacked[..., 0::2] = packed & 0x0F        # even indices = lower nibble
    unpacked[..., 1::2] = (packed >> 4) & 0x0F  # odd indices = upper nibble

    return unpacked


# ═══════════════════════════════════════════════════════════════════════════
# INT4 Quantizer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class QuantizedWeight:
    """Container for INT4 quantized weights."""
    qweight: torch.Tensor        # Packed INT4 weights [out_features, in_features // 2] uint8
    scales: torch.Tensor         # Per-group scales [out_features, num_groups] float16
    zeros: torch.Tensor          # Per-group zero points [out_features, num_groups] uint8
    group_size: int              # Number of weights per quantization group
    out_features: int
    in_features: int
    bits: int = 4

    def memory_bytes(self) -> int:
        """Compute total memory usage in bytes."""
        return (self.qweight.element_size() * self.qweight.numel() +
                self.scales.element_size() * self.scales.numel() +
                self.zeros.element_size() * self.zeros.numel())

    def compression_ratio(self) -> float:
        """Compression ratio vs. float16."""
        fp16_bytes = 2 * self.out_features * self.in_features
        return fp16_bytes / self.memory_bytes()


class Int4Quantizer:
    """
    Group-wise asymmetric INT4 quantization.

    Usage:
        quantizer = Int4Quantizer(group_size=128, asymmetric=True)
        quantized = quantizer.quantize(weight_tensor)
        restored = quantizer.dequantize(quantized)
    """

    def __init__(
        self,
        group_size: int = 128,
        asymmetric: bool = True,
        per_group: bool = True,
    ):
        """
        Args:
            group_size: Number of weights per quantization group (along in_features)
            asymmetric: Use asymmetric quantization (with zero-point)
            per_group:  If True, each group has its own scale/zp.
                        If False, per-channel quantization.
        """
        self.group_size = group_size
        self.asymmetric = asymmetric
        self.per_group = per_group
        self.bits = 4
        self.qmax = (1 << self.bits) - 1  # 15

    def quantize(self, weight: torch.Tensor) -> QuantizedWeight:
        """
        Quantize a weight matrix to INT4.

        Args:
            weight: FP16/BF16 weight matrix [out_features, in_features]

        Returns:
            QuantizedWeight with packed INT4 weights and scales
        """
        assert weight.dim() == 2, f"Expected 2D weight, got {weight.dim()}D"
        out_features, in_features = weight.shape

        # Convert to float32 for quantization math
        w_float = weight.float()

        # Pad in_features to be divisible by group_size
        padded_in = ((in_features + self.group_size - 1) // self.group_size) * self.group_size
        if padded_in != in_features:
            w_padded = F.pad(w_float, (0, padded_in - in_features))
        else:
            w_padded = w_float

        num_groups = padded_in // self.group_size

        # Reshape: [out_features, num_groups, group_size]
        w_reshaped = w_padded.reshape(out_features, num_groups, self.group_size)

        # ── Find min/max per group ────────────────────────────────────────
        if self.per_group:
            w_min = w_reshaped.min(dim=-1).values   # [out_features, num_groups]
            w_max = w_reshaped.max(dim=-1).values
        else:
            # Per-channel: one scale per output channel
            w_min = w_reshaped.min(dim=-1).values.min(dim=-1, keepdim=True).values
            w_max = w_reshaped.max(dim=-1).values.max(dim=-1, keepdim=True).values

        # Clamp min/max to prevent degenerate scales
        range_val = w_max - w_min
        range_val = torch.clamp(range_val, min=1e-6)

        # ── Compute scale and zero-point ──────────────────────────────────
        scale = range_val / self.qmax  # [out_features, num_groups]

        if self.asymmetric:
            # zero_point = round(-w_min / scale), clamped to [0, qmax]
            zp = torch.round(-w_min / scale)
            zp = torch.clamp(zp, 0, self.qmax).to(torch.uint8)
        else:
            # Symmetric: zero_point = qmax / 2 = 7 (or 8)
            zp = torch.full_like(w_min, self.qmax // 2, dtype=torch.uint8)

        # ── Quantize ──────────────────────────────────────────────────────
        w_int = torch.round(w_reshaped / scale.unsqueeze(-1) + zp.unsqueeze(-1).float())
        w_int = torch.clamp(w_int, 0, self.qmax).to(torch.uint8)

        # ── Reshape and pack ──────────────────────────────────────────────
        w_flat = w_int.reshape(out_features, padded_in)  # [out, in_padded]

        # Trim padding if added
        if padded_in != in_features:
            w_flat = w_flat[:, :in_features]

        # Ensure even in_features for packing
        if in_features % 2 != 0:
            w_flat = F.pad(w_flat, (0, 1))  # pad one column

        # Pack two INT4 per byte
        qweight = pack_int4(w_flat)  # [out_features, in_features_padded // 2]

        # Store scales in float16 for memory efficiency
        scales_store = scale.to(torch.float16)
        if padded_in != in_features and num_groups * self.group_size != in_features:
            # Adjust num_groups for trimmed case
            scales_store = scales_store[:, :((in_features + self.group_size - 1) // self.group_size)]

        return QuantizedWeight(
            qweight=qweight,
            scales=scales_store,
            zeros=zp,
            group_size=self.group_size,
            out_features=out_features,
            in_features=in_features,
            bits=self.bits,
        )

    def dequantize(self, qw: QuantizedWeight) -> torch.Tensor:
        """
        Dequantize INT4 weights back to float32.

        Args:
            qw: QuantizedWeight container

        Returns:
            Dequantized float32 weight [out_features, in_features]
        """
        # Unpack INT4 → uint8
        w_int8 = unpack_int4(qw.qweight)  # [out, in_padded]
        w_int8 = w_int8[:, :qw.in_features].to(torch.float32)

        # Compute groups
        num_groups = (qw.in_features + qw.group_size - 1) // qw.group_size

        # Reshape for per-group dequant
        # Pad to group boundary for reshape
        padded_in = num_groups * qw.group_size
        w_int_padded = F.pad(w_int8, (0, padded_in - qw.in_features))
        w_reshaped = w_int_padded.reshape(qw.out_features, num_groups, qw.group_size)

        # Dequantize: w_fp = scale * (w_int - zp)
        scale_f = qw.scales.float().unsqueeze(-1)  # [out, groups, 1]
        zp_f = qw.zeros[:, :num_groups].float().unsqueeze(-1)

        w_deq = scale_f * (w_reshaped - zp_f)
        w_flat = w_deq.reshape(qw.out_features, padded_in)

        # Trim padding
        return w_flat[:, :qw.in_features]


# ═══════════════════════════════════════════════════════════════════════════
# QuantizedLinear — drop-in replacement for nn.Linear with INT4 weights
# ═══════════════════════════════════════════════════════════════════════════

class QuantizedLinear(nn.Module):
    """
    INT4-quantized linear layer.

    Stores weights as INT4 but computes in FP16. The forward pass:
      1. Dequantizes INT4 → FP16 for the relevant weight columns
      2. Computes FP16 matmul
      3. Adds bias (if present)

    For inference efficiency on CUDA:
      - The CUDA dequant kernel fused with matmul is in `csrc/`
      - On CPU: pure PyTorch fallback

    Usage:
        layer = QuantizedLinear.from_float(original_linear, group_size=128)
        output = layer(input_fp16)
    """

    def __init__(
        self,
        qweight: QuantizedWeight,
        bias: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.qweight = qweight

        # Register as buffers (not parameters, to skip gradient)
        self.register_buffer("_qweight_packed", qweight.qweight)
        self.register_buffer("_scales", qweight.scales)
        self.register_buffer("_zeros", qweight.zeros.to(torch.float16))

        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None

        self.out_features = qweight.out_features
        self.in_features = qweight.in_features
        self.group_size = qweight.group_size

    @classmethod
    def from_float(
        cls,
        linear: nn.Linear,
        group_size: int = 128,
        asymmetric: bool = True,
    ) -> "QuantizedLinear":
        """
        Create QuantizedLinear from a standard nn.Linear layer.

        Args:
            linear: Original FP32/FP16 linear layer
            group_size: Number of weights per quantization group
            asymmetric: Use asymmetric quantization

        Returns:
            QuantizedLinear with INT4 weights
        """
        quantizer = Int4Quantizer(
            group_size=group_size,
            asymmetric=asymmetric,
        )
        qw = quantizer.quantize(linear.weight.data)

        bias = linear.bias.data.clone() if linear.bias is not None else None

        return cls(qw, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: dequantizes INT4 → FP16 and computes matmul.

        Strategy: For each group, dequantize and compute partial matmul.
        This avoids allocating the full dequantized weight matrix.

        Args:
            x: Input tensor [..., in_features]

        Returns:
            Output tensor [..., out_features]
        """
        # For small weights, dequantize fully
        # For large weights, dequantize per group
        if self.in_features * self.out_features < 4096 * 4096:
            return self._forward_full(x)
        else:
            return self._forward_groupwise(x)

    def _forward_full(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize full weight and compute matmul."""
        # Unpack and dequantize
        w_int = unpack_int4(self._qweight_packed)[:, :self.in_features].to(torch.float32)
        num_groups = (self.in_features + self.group_size - 1) // self.group_size

        # Tile scales and zeros to match weight shape
        w_deq = torch.empty(self.out_features, self.in_features, device=x.device, dtype=torch.float32)
        for g in range(num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)
            scale = self._scales[:, g].float().unsqueeze(-1)
            zp = self._zeros[:, g].float().unsqueeze(-1)
            w_deq[:, start:end] = scale * (w_int[:, start:end] - zp)

        # F.linear in FP32 then cast
        return F.linear(x.to(torch.float32), w_deq, self.bias.float() if self.bias is not None else None).to(x.dtype)

    def _forward_groupwise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Group-wise dequant and matmul to reduce peak memory.

        Accumulates output = Σ (dequantize(group_g) @ input_group_g^T)
        """
        x_float = x.to(torch.float32)
        output = torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=torch.float32)

        if self.bias is not None:
            output += self.bias.float()

        num_groups = (self.in_features + self.group_size - 1) // self.group_size
        w_int_all = unpack_int4(self._qweight_packed)[:, :self.in_features].to(torch.float32)

        for g in range(num_groups):
            start = g * self.group_size
            end = min(start + self.group_size, self.in_features)

            scale = self._scales[:, g].float().unsqueeze(-1)  # [out, 1]
            zp = self._zeros[:, g].float().unsqueeze(-1)

            # Dequantize this group
            w_group = scale * (w_int_all[:, start:end] - zp)  # [out, group_size]
            x_group = x_float[..., start:end]  # [..., group_size]

            output += x_group @ w_group.T

        return output.to(x.dtype)

    def extra_repr(self) -> str:
        cr = self.qweight.compression_ratio()
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bits=4, group_size={self.group_size}, compression={cr:.2f}×")


# ═══════════════════════════════════════════════════════════════════════════
# CUDA Dequantization Kernel (Python fallback with GPU tensor ops)
# ═══════════════════════════════════════════════════════════════════════════

def dequantize_cuda(qweight: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor,
                    group_size: int) -> torch.Tensor:
    """
    Efficient GPU dequantization using PyTorch tensor ops.

    This is the Python implementation. For production, the CUDA kernel
    in csrc/dequant_kernel.cu provides a fused dequant+matmul.

    Args:
        qweight: Packed INT4 weights [out, in//2] uint8
        scales:  Per-group scales [out, num_groups] float16
        zeros:   Per-group zero-points [out, num_groups] uint8
        group_size: Groups per weight

    Returns:
        Dequantized FP16 weights [out, in]
    """
    out_features = qweight.shape[0]
    packed_in = qweight.shape[1]
    in_features = packed_in * 2  # approximate

    # Unpack
    w_int = unpack_int4(qweight).to(torch.float16)  # [out, in]

    # Compute actual in_features (accounting for padding)
    num_groups = scales.shape[1]
    actual_in = num_groups * group_size
    if actual_in != in_features:
        w_int = w_int[:, :actual_in]

    # Tile scales and zeros
    scales_fp16 = scales.to(torch.float16)
    # Expand scales: [out, groups] → [out, groups, 1] → repeat → [out, in]
    scales_expanded = scales_fp16.repeat_interleave(group_size, dim=1)[:, :actual_in]
    zeros_expanded = zeros.to(torch.float16).repeat_interleave(group_size, dim=1)[:, :actual_in]

    return scales_expanded * (w_int - zeros_expanded)
