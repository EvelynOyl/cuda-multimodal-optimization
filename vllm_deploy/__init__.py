"""
vllm_deploy — vLLM-powered LLaVA-1.5 inference serving.

Provides:
  - LLaVA multimodal worker with vision tower integration
  - Continuous batching scheduler for maximal throughput
  - FastAPI inference server with streaming support
  - Chat-compatible API endpoint
"""

from .config import ServingConfig
from .llava_worker import LLaVAWorker
from .continuous_batching import ContinuousBatchingScheduler
from .api_server import create_app

__all__ = [
    "ServingConfig",
    "LLaVAWorker",
    "ContinuousBatchingScheduler",
    "create_app",
]
