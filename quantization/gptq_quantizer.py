"""
quantization/gptq_quantizer.py — GPTQ-style INT4 quantization.

Implements the GPTQ (Optimal Brain Quantization) algorithm for post-training
weight-only quantization:

  1. Compute Hessian diagonal (H) from calibration data
  2. For each weight column (or block of columns):
     a. Quantize the weight
     b. Compute quantization error: d = (w_quantized - w_original)
     c. Update remaining weights: w_remaining -= d * H_inv[:, col] / H[col, col]
  3. Optionally reorder columns by activation magnitude (act-order)

This preserves model quality much better than round-to-nearest quantization,
especially at 4-bit precision.

References:
  - Frantar et al. "GPTQ: Accurate Post-Training Quantization for
    Generative Pre-trained Transformers." ICLR 2023.
  - Original repo: https://github.com/IST-DASLab/gptq
  - AutoGPTQ: https://github.com/AutoGPTQ/AutoGPTQ
"""

import time
import math
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
import torch.nn as nn
import numpy as np

from .int4_quantizer import Int4Quantizer, QuantizedWeight, QuantizedLinear
from .calibration import CalibrationDataset


# ═══════════════════════════════════════════════════════════════════════════
# GPTQ Quantizer
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GPTQConfig:
    """Configuration for GPTQ quantization."""
    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01       # Hessian diagonal dampening
    act_order: bool = True           # Reorder by activation magnitude
    per_group: bool = True           # Per-group quantization
    asymmetric: bool = True          # Asymmetric (with zero-point)
    true_sequential: bool = True     # Sequential column processing
    static_groups: bool = False      # Pre-compute groups vs dynamic


class GPTQQuantizer:
    """
    GPTQ post-training quantization for LLM linear layers.

    Algorithm (per layer):

      X = calibration inputs  [N_samples, in_features]
      H = X^T @ X / N          # Hessian approximation
      H += damp * mean(diag(H)) * I   # dampening

      For each column col in [0, in_features):
        1. Compute quantized weight w_q[col]
        2. Compute error: err = (w_q[col] - w[col]) / H_inv[col, col]
        3. Update remaining weights: w[col+1:] -= err * H_inv[col+1:, col]

    Usage:
        config = GPTQConfig(bits=4, group_size=128)
        quantizer = GPTQQuantizer(config)
        quantizer.quantize_model(model, calibration_data)
        model.save_pretrained("llama-7b-int4-gptq")
    """

    def __init__(self, config: GPTQConfig):
        self.config = config
        self.quantizer = Int4Quantizer(
            group_size=config.group_size,
            asymmetric=config.asymmetric,
            per_group=config.per_group,
        )
        self.bits = config.bits

    def quantize_model(
        self,
        model: nn.Module,
        calib_dataset: CalibrationDataset,
        target_modules: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Quantize all linear layers in the model using GPTQ.

        Args:
            model: HuggingFace model (LLaMA/LLaVA)
            calib_dataset: Calibration data for Hessian computation
            target_modules: Specific module names to quantize (e.g., ["q_proj", "k_proj", ...])
                            If None, quantizes all nn.Linear layers.

        Returns:
            Model with linear layers replaced by QuantizedLinear
        """
        model.eval()

        # Collect layers to quantize
        layers_to_quantize = self._find_linear_layers(model, target_modules)
        print(f"[GPTQ] Found {len(layers_to_quantize)} linear layers to quantize")

        # ── Cache calibration inputs at each layer ────────────────────────
        layer_inputs = self._collect_layer_inputs(model, calib_dataset, layers_to_quantize)

        # ── Quantize each layer ───────────────────────────────────────────
        for layer_name, layer in tqdm(layers_to_quantize.items(), desc="GPTQ quantizing"):
            inp = layer_inputs[layer_name]

            if inp is None:
                print(f"  Skipping {layer_name}: no calibration data")
                continue

            # Compute Hessian
            H = self._compute_hessian(inp)

            # Apply GPTQ algorithm
            qw = self._quantize_weight_gptq(layer.weight.data, H)

            # Replace layer
            quantized_layer = QuantizedLinear(
                qweight=qw,
                bias=layer.bias.data.clone() if layer.bias is not None else None,
            )

            # Find parent module and replace
            self._replace_module(model, layer_name, quantized_layer)

        print(f"[GPTQ] Quantization complete. Model memory reduced.")
        return model

    def _quantize_weight_gptq(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
    ) -> QuantizedWeight:
        """
        Apply GPTQ algorithm to a single weight matrix.

        Args:
            W: Original weight [out_features, in_features]
            H: Hessian matrix [in_features, in_features]

        Returns:
            QuantizedWeight
        """
        out_features, in_features = W.shape
        dtype = W.dtype
        dev = W.device

        W_fp = W.float().clone()
        H = H.float()

        # ── Hessian dampening ─────────────────────────────────────────────
        damp = self.config.damp_percent * torch.mean(torch.diag(H))
        H += damp * torch.eye(in_features, device=dev, dtype=torch.float32)

        # ── Cholesky decomposition of H inverse ───────────────────────────
        # H_inv is used to compute optimal weight compensation
        try:
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))
        except RuntimeError:
            # If Cholesky fails (numerical issue), use pseudoinverse
            H = H + 1e-3 * torch.eye(in_features, device=dev, dtype=torch.float32)
            H_inv = torch.cholesky_inverse(torch.linalg.cholesky(H))

        # ── Determine quantization order ──────────────────────────────────
        if self.config.act_order:
            # Reorder by activation magnitude (descending diagonal of H)
            act_magnitude = torch.diag(H)
            _, perm = torch.sort(act_magnitude, descending=True)
            inv_perm = torch.argsort(perm)
        else:
            perm = torch.arange(in_features, device=dev)
            inv_perm = torch.arange(in_features, device=dev)

        W_fp = W_fp[:, perm]  # Reorder columns

        # ── GPTQ: Sequential column quantization with error compensation ──
        Q = torch.zeros_like(W_fp)  # Quantized weights
        dead = torch.zeros(in_features, dtype=torch.bool, device=dev)

        # Process all output channels simultaneously, one input column at a time
        for col_idx in range(in_features):
            if dead[col_idx]:
                continue

            # Current weight column
            w_col = W_fp[:, col_idx]  # [out_features]

            # Quantize this column
            # Note: GPTQ processes columns sequentially, but we need all columns
            # in a group for proper quantization. Simplified here: quantize per-column
            # and handle group-wise scale/zp at the end.
            w_min = w_col.min()
            w_max = w_col.max()
            scale = max((w_max - w_min).item() / 15, 1e-6)
            zp = torch.round(-w_min / scale).clamp(0, 15)
            w_q = torch.round(w_col / scale + zp).clamp(0, 15)
            w_dq = scale * (w_q.float() - zp.float())

            # Quantization error
            err = (w_dq - w_col) / H_inv[col_idx, col_idx]

            Q[:, col_idx] = w_q

            # Update remaining columns (error compensation)
            remaining_mask = torch.arange(col_idx + 1, in_features, device=dev)
            if len(remaining_mask) > 0:
                W_fp[:, perm[col_idx + 1:]] -= torch.outer(
                    err,
                    H_inv[perm[col_idx + 1:], perm[col_idx]]
                )

        # ── Restore original column order ─────────────────────────────────
        if self.config.act_order:
            Q = Q[:, inv_perm]

        # ── Final quantization with proper group-wise scales ─────────────
        qw = self._group_quantize(Q.to(dtype), in_features)

        return qw

    def _group_quantize(self, W_int: torch.Tensor, in_features: int) -> QuantizedWeight:
        """
        Compute per-group scales and zero-points from already-quantized weights,
        then pack into QuantizedWeight format.
        """
        out_features = W_int.shape[0]
        group_size = self.config.group_size

        # Pad to group boundary
        num_groups = (in_features + group_size - 1) // group_size
        padded_in = num_groups * group_size

        w_padded = torch.nn.functional.pad(W_int.float(), (0, padded_in - in_features))
        w_reshaped = w_padded.reshape(out_features, num_groups, group_size)

        # Compute per-group min/max from quantized values
        w_min = w_reshaped.min(dim=-1).values
        w_max = w_reshaped.max(dim=-1).values
        range_val = torch.clamp(w_max - w_min, min=1e-6)

        scales = (range_val / 15.0).to(torch.float16)
        zp = torch.round(-w_min / range_val * 15.0).clamp(0, 15).to(torch.uint8)

        # Make sure in_features is even for packing
        if in_features % 2 != 0:
            W_int = torch.nn.functional.pad(W_int, (0, 1))

        from .int4_quantizer import pack_int4
        qweight = pack_int4(W_int.to(torch.uint8))

        return QuantizedWeight(
            qweight=qweight,
            scales=scales,
            zeros=zp,
            group_size=group_size,
            out_features=out_features,
            in_features=in_features,
            bits=self.bits,
        )

    def _compute_hessian(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian matrix approximation: H ≈ X^T @ X / N

        Where X is the set of input activations [N_samples, in_features].
        This approximates the Fisher information matrix for linear layers.

        Args:
            inp: Calibration inputs [N, in_features]

        Returns:
            Hessian matrix [in_features, in_features]
        """
        N, in_features = inp.shape
        X = inp.float().T  # [in_features, N]
        H = (X @ X.T) / N  # [in_features, in_features]

        # Ensure symmetry
        H = (H + H.T) / 2

        return H

    def _find_linear_layers(
        self,
        model: nn.Module,
        target_modules: Optional[List[str]] = None,
    ) -> Dict[str, nn.Linear]:
        """
        Find all nn.Linear layers in the model, optionally filtering
        by name patterns in target_modules.
        """
        layers = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if target_modules is None:
                    layers[name] = module
                elif any(t in name for t in target_modules):
                    layers[name] = module
        return layers

    def _collect_layer_inputs(
        self,
        model: nn.Module,
        calib_dataset: CalibrationDataset,
        layers: Dict[str, nn.Linear],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Run calibration data through the model and capture inputs to each
        linear layer using forward hooks.

        Returns a dict mapping layer names to their calibration inputs.
        """
        layer_inputs: Dict[str, List[torch.Tensor]] = {name: [] for name in layers}
        hooks = []

        def make_hook(name):
            def hook(module, inp, out):
                # inp[0] is the input tensor to the linear layer
                layer_inputs[name].append(inp[0].detach().cpu())
            return hook

        # Register hooks
        for name, layer in layers.items():
            hooks.append(layer.register_forward_hook(make_hook(name)))

        # Run calibration data
        model_device = next(model.parameters()).device
        with torch.no_grad():
            for batch in calib_dataset.iter_batches():
                # Move to model device
                if isinstance(batch, dict):
                    batch = {k: v.to(model_device) for k, v in batch.items()}
                    model(**batch)
                else:
                    batch = batch.to(model_device)
                    model(batch)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Concatenate inputs for each layer
        result = {}
        for name, inputs in layer_inputs.items():
            if inputs:
                result[name] = torch.cat(inputs, dim=0)  # [total_samples, in_features]
            else:
                result[name] = None

        return result

    def _replace_module(
        self,
        model: nn.Module,
        target_name: str,
        new_module: nn.Module,
    ):
        """
        Replace a module identified by dotted name in the model tree.

        Args:
            model: Root model
            target_name: Dotted path like "model.layers.0.self_attn.q_proj"
            new_module: Replacement module
        """
        path = target_name.split(".")
        parent = model
        for part in path[:-1]:
            parent = getattr(parent, part)
        setattr(parent, path[-1], new_module)
