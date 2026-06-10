"""
quantization — INT4 weight-only quantization for LLM inference.

Implements GPTQ-style group-wise asymmetric INT4 quantization with:
  - Hessian-based optimal scaling factor search
  - Activation-order channel reordering
  - Weight packing (two INT4 weights per byte)
  - Dequantization kernels compatible with CUDA and CPU

Supports:
  - LLaMA / Llama-2 architectures
  - LLaVA-1.5 multimodal model
  - Integration with vLLM's INT4 weight loading
"""

from .int4_quantizer import Int4Quantizer, QuantizedLinear, pack_int4, unpack_int4
from .gptq_quantizer import GPTQQuantizer
from .calibration import CalibrationDataset, load_calibration_data

__all__ = [
    "Int4Quantizer",
    "QuantizedLinear",
    "pack_int4",
    "unpack_int4",
    "GPTQQuantizer",
    "CalibrationDataset",
    "load_calibration_data",
]
