"""
vllm_deploy/config.py — Serving configuration for LLaVA-1.5 + vLLM.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path


@dataclass
class ModelConfig:
    """LLaVA-1.5 model configuration."""
    # HuggingFace model ID or local path
    model_name: str = "liuhaotian/llava-v1.5-7b"
    # Alternative: llava-hf/llava-1.5-7b-hf (pure HuggingFace version)
    trust_remote_code: bool = True

    # Model dimensions
    hidden_size: int = 4096
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32
    intermediate_size: int = 11008

    # Vision tower
    vision_model: str = "openai/clip-vit-large-patch14-336"
    image_size: int = 336
    patch_size: int = 14
    num_vision_tokens: int = 576  # (336/14)^2 = 576

    # Multimodal projector
    mm_projector_type: str = "mlp2x_gelu"  # LLaVA-1.5 default

    # Sequence
    max_model_len: int = 4096

    # Data type
    dtype: str = "float16"

    # Quantization (set to "int4" to load INT4 quantized weights)
    quantization: Optional[str] = None


@dataclass
class SchedulerConfig:
    """Continuous batching scheduler configuration."""
    # Maximum number of sequences batched concurrently
    max_num_seqs: int = 256

    # Maximum number of batched tokens per iteration
    max_num_batched_tokens: int = 2048

    # Maximum number of tokens in a single prefill step
    max_num_prefill_tokens: int = 8192

    # Block size for PagedAttention KV-cache management
    block_size: int = 16

    # Enable chunked prefill (splits long prefills into chunks)
    enable_chunked_prefill: bool = True

    # Enable prefix caching (reuses KV-cache for shared prefixes)
    enable_prefix_caching: bool = True

    # Number of GPU blocks to allocate for KV cache
    # (computed automatically from gpu_memory_utilization)
    num_gpu_blocks_override: Optional[int] = None


@dataclass
class ServingConfig:
    """Top-level serving configuration."""
    # Model
    model: ModelConfig = field(default_factory=ModelConfig)

    # Scheduler
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # GPU memory
    gpu_memory_utilization: float = 0.90
    swap_space: int = 4  # GB of CPU memory for KV-cache offload
    enforce_eager: bool = False  # True to disable CUDA graphs

    # Logging
    log_level: str = "INFO"

    # Image preprocessing
    max_image_size: int = 336
    image_mean: List[float] = field(default_factory=lambda: [0.48145466, 0.4578275, 0.40821073])
    image_std: List[float] = field(default_factory=lambda: [0.26862954, 0.26130258, 0.27577711])

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ServingConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)

        config = cls()
        if "model" in data:
            config.model = ModelConfig(**{k: v for k, v in data["model"].items()
                                          if k in ModelConfig.__dataclass_fields__})
        if "vllm" in data:
            vllm_data = data["vllm"]
            # Map vLLM config keys to scheduler
            sched_data = {k: v for k, v in vllm_data.items()
                         if k in SchedulerConfig.__dataclass_fields__}
            config.scheduler = SchedulerConfig(**sched_data)
            # Server-level keys
            for key in ("host", "port", "gpu_memory_utilization", "swap_space"):
                if key in vllm_data:
                    setattr(config, key, vllm_data[key])

        return config


@dataclass
class BatchRequest:
    """A single request in a batch."""
    request_id: str
    prompt: str
    image_data: Optional[bytes] = None       # Raw image bytes
    image_tensor: Optional["torch.Tensor"] = None  # Preprocessed image
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False


@dataclass
class GenerationOutput:
    """Output from generation."""
    request_id: str
    text: str
    finish_reason: str  # "stop", "length", "abort"
    tokens_generated: int
    prompt_tokens: int
    total_time_ms: float
    tokens_per_second: float
