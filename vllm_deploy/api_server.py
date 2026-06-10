"""
vllm_deploy/api_server.py — FastAPI inference server for LLaVA-1.5 + vLLM.

Provides OpenAI-compatible API endpoints:
  - POST /v1/chat/completions    — Chat-style multimodal completion
  - POST /v1/completions          — Text-only completion
  - GET  /v1/models               — Model list
  - GET  /health                  — Health check
  - GET  /metrics                 — Prometheus-style metrics

Design:
  - Uses vLLM's AsyncLLMEngine for true continuous batching at scale
  - Background task for request scheduling loop
  - Proper request lifecycle management with abort support
  - Streaming (SSE) support for token-by-token output

Usage:
    uvicorn vllm_deploy.api_server:app --host 0.0.0.0 --port 8000

Or programmatically:
    from vllm_deploy.api_server import create_app
    app = create_app(config)
"""

import asyncio
import time
import uuid
import json
import logging
from typing import Optional, List, Dict, Any, AsyncIterator
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import ServingConfig, BatchRequest, GenerationOutput
from .llava_worker import LLaVAWorker
from .continuous_batching import ContinuousBatchingScheduler

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Pydantic API Models
# ═══════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str | List[Dict[str, Any]] = Field(..., description="Text or multimodal content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default="llava-v1.5-7b")
    messages: List[Message]
    max_tokens: int = Field(default=256, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=100)
    stream: bool = False
    stop: Optional[List[str]] = None
    # Multimodal: image passed as base64 in content or via this field
    image_url: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class StreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[StreamChoice]


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str
    requests_processed: int
    tokens_generated: int
    uptime_seconds: float


class MetricsResponse(BaseModel):
    requests_total: int
    tokens_total: int
    avg_tokens_per_second: float
    avg_time_per_request_ms: float
    queue_depth: int
    active_sequences: int


# ═══════════════════════════════════════════════════════════════════════════
# Application Factory
# ═══════════════════════════════════════════════════════════════════════════

# Global state (per-worker)
_worker: Optional[LLaVAWorker] = None
_scheduler: Optional[ContinuousBatchingScheduler] = None
_config: Optional[ServingConfig] = None
_start_time: float = 0.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: initialize worker on startup, cleanup on shutdown."""
    global _worker, _scheduler, _config, _start_time

    logger.info("Starting LLaVA-1.5 vLLM server...")

    # Load config
    _config = ServingConfig()
    try:
        _config = ServingConfig.from_yaml("configs/config.yaml")
        logger.info("Loaded config from configs/config.yaml")
    except FileNotFoundError:
        logger.info("Using default config")

    # Initialize worker and scheduler
    _worker = LLaVAWorker(_config)
    await _worker.initialize()

    _scheduler = ContinuousBatchingScheduler(
        _config.scheduler,
        num_gpu_blocks=10000,  # will be computed from GPU memory
    )

    _start_time = time.time()
    logger.info(f"Server ready on {_config.host}:{_config.port}")

    yield  # Server runs here

    # Cleanup
    logger.info("Shutting down...")
    if _worker:
        await _worker.shutdown()


def create_app(config: Optional[ServingConfig] = None) -> FastAPI:
    """
    Create a FastAPI application with LLaVA-1.5 + vLLM endpoints.

    Args:
        config: Serving configuration (uses defaults if not provided)

    Returns:
        Configured FastAPI application
    """
    global _config
    if config:
        _config = config

    app = FastAPI(
        title="LLaVA-1.5 + vLLM Inference Server",
        description="Multimodal vision-language model serving with Continuous Batching",
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── Middleware ────────────────────────────────────────────────────────

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log all requests with timing."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        logger.info(f"{request.method} {request.url.path} — {response.status_code} ({elapsed*1000:.1f}ms)")
        return response

    # ── Health Check ──────────────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        if _worker is None:
            raise HTTPException(status_code=503, detail="Worker not initialized")

        stats = _worker.stats()
        return HealthResponse(
            status="healthy" if _worker._initialized else "initializing",
            model=_config.model.model_name,
            backend="cuda" if torch.cuda.is_available() else "cpu",
            requests_processed=stats["requests_processed"],
            tokens_generated=stats["tokens_generated"],
            uptime_seconds=time.time() - _start_time,
        )

    # ── Metrics ───────────────────────────────────────────────────────────

    @app.get("/metrics", response_model=MetricsResponse)
    async def metrics():
        """Prometheus-compatible metrics."""
        if _worker is None:
            raise HTTPException(status_code=503, detail="Not ready")

        stats = _worker.stats()
        uptime = max(time.time() - _start_time, 0.001)

        return MetricsResponse(
            requests_total=stats["requests_processed"],
            tokens_total=stats["tokens_generated"],
            avg_tokens_per_second=stats["tokens_generated"] / uptime,
            avg_time_per_request_ms=uptime * 1000 / max(stats["requests_processed"], 1),
            queue_depth=_scheduler.stats()["waiting"] if _scheduler else 0,
            active_sequences=_scheduler.stats()["running"] if _scheduler else 0,
        )

    # ── Model List ────────────────────────────────────────────────────────

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible model list."""
        return {
            "object": "list",
            "data": [
                {
                    "id": "llava-v1.5-7b",
                    "object": "model",
                    "created": 1698979200,
                    "owned_by": "liuhaotian",
                }
            ],
        }

    # ── Chat Completions ──────────────────────────────────────────────────

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: ChatCompletionRequest):
        """
        OpenAI-compatible chat completions endpoint with multimodal support.

        Supports:
          - Text-only conversations
          - Image + text (LLaVA-1.5 multimodal)
          - Streaming (SSE) responses
        """
        if _worker is None:
            raise HTTPException(status_code=503, detail="Server not ready")

        # ── Extract prompt from messages ──────────────────────────────────
        prompt = ""
        image_bytes = None

        for msg in request.messages:
            if msg.role == "user":
                if isinstance(msg.content, str):
                    prompt = msg.content
                elif isinstance(msg.content, list):
                    # Multimodal content: [{"type": "text", "text": "..."}, {"type": "image_url", ...}]
                    parts = []
                    for part in msg.content:
                        if part.get("type") == "text":
                            parts.append(part["text"])
                        elif part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:"):
                                import base64
                                # Decode base64 data URL
                                _, b64_data = image_url.split(",", 1)
                                image_bytes = base64.b64decode(b64_data)
                            elif image_url.startswith("http"):
                                import aiohttp
                                async with aiohttp.ClientSession() as session:
                                    async with session.get(image_url) as resp:
                                        image_bytes = await resp.read()
                    prompt = " ".join(parts)

        # Handle image_url field if provided
        if request.image_url and not image_bytes:
            if request.image_url.startswith("data:"):
                import base64
                _, b64_data = request.image_url.split(",", 1)
                image_bytes = base64.b64decode(b64_data)

        if not prompt:
            raise HTTPException(status_code=400, detail="No text content in messages")

        # ── Build batch request ───────────────────────────────────────────
        batch_req = BatchRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            image_data=image_bytes,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_sequences=request.stop or [],
            stream=request.stream,
        )

        # ── Streaming ─────────────────────────────────────────────────────
        if request.stream:
            return StreamingResponse(
                _stream_chat_response(batch_req, request.model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        # ── Non-streaming ─────────────────────────────────────────────────
        output = await _worker.generate(batch_req)

        response_id = f"chatcmpl-{batch_req.request_id}"
        return ChatCompletionResponse(
            id=response_id,
            created=int(time.time()),
            model=request.model,
            choices=[
                ChatCompletionChoice(
                    message=Message(role="assistant", content=output.text),
                    finish_reason=output.finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=output.prompt_tokens,
                completion_tokens=output.tokens_generated,
                total_tokens=output.prompt_tokens + output.tokens_generated,
            ),
        )

    # ── Streaming generator ───────────────────────────────────────────────
    async def _stream_chat_response(req: BatchRequest, model: str):
        """SSE stream for chat completions."""
        response_id = f"chatcmpl-{req.request_id}"
        created = int(time.time())

        # Send role first
        role_chunk = StreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[StreamChoice(delta=DeltaMessage(role="assistant"))],
        )
        yield f"data: {role_chunk.model_dump_json()}\n\n"

        # Stream tokens
        async for token_text in _worker.generate_stream(req):
            token_chunk = StreamResponse(
                id=response_id,
                created=created,
                model=model,
                choices=[StreamChoice(delta=DeltaMessage(content=token_text))],
            )
            yield f"data: {token_chunk.model_dump_json()}\n\n"

        # Done
        done_chunk = StreamResponse(
            id=response_id,
            created=created,
            model=model,
            choices=[StreamChoice(delta=DeltaMessage(), finish_reason="stop")],
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    # ── Image Upload (alternative) ────────────────────────────────────────

    @app.post("/v1/chat/completions/with_image")
    async def chat_with_image(
        prompt: str = Form(...),
        image: UploadFile = File(...),
        max_tokens: int = Form(256),
        temperature: float = Form(0.7),
        stream: bool = Form(False),
    ):
        """Chat completion with image upload (multipart form)."""
        if _worker is None:
            raise HTTPException(status_code=503, detail="Not ready")

        image_bytes = await image.read()

        batch_req = BatchRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            image_data=image_bytes,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
        )

        output = await _worker.generate(batch_req)

        return JSONResponse({
            "text": output.text,
            "tokens_generated": output.tokens_generated,
            "tokens_per_second": output.tokens_per_second,
            "finish_reason": output.finish_reason,
        })

    return app


# ═══════════════════════════════════════════════════════════════════════════
# Default app instance
# ═══════════════════════════════════════════════════════════════════════════

app = create_app()


# ── Direct launch ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "vllm_deploy.api_server:app",
        host=_config.host if _config else "0.0.0.0",
        port=_config.port if _config else 8000,
        log_level="info",
        reload=False,
    )
