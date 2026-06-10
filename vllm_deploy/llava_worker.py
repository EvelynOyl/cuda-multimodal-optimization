"""
vllm_deploy/llava_worker.py — LLaVA-1.5 multimodal worker for vLLM.

Integrates the CLIP vision tower, multimodal projector, and Llama-2 language
model into the vLLM serving framework. Handles:
  - Image preprocessing (CLIP-standard normalization)
  - Vision feature extraction
  - Multimodal embedding construction (image tokens + text tokens)
  - Token generation via vLLM engine

The worker is designed to be instantiated per-GPU (tensor-parallel across
multiple GPUs if needed).

References:
  - LLaVA-1.5: https://github.com/haotian-liu/LLaVA
  - vLLM multimodal: https://docs.vllm.ai/en/latest/features/multimodal_inputs.html
"""

import asyncio
import time
from typing import Optional, List, Dict, Any, AsyncIterator
from dataclasses import dataclass
from io import BytesIO

import torch
import numpy as np
from PIL import Image
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModel,
    AutoTokenizer,
    AutoConfig,
    LlavaForConditionalGeneration,
)
from vllm import LLM, SamplingParams
from vllm.multimodal import MultiModalData

from .config import ServingConfig, BatchRequest, GenerationOutput


class ImagePreprocessor:
    """CLIP-compatible image preprocessing for LLaVA-1.5."""

    def __init__(self, config: ServingConfig):
        self.image_size = config.model.image_size  # 336
        self.mean = np.array(config.image_mean, dtype=np.float32)
        self.std = np.array(config.image_std, dtype=np.float32)

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Preprocess a PIL image into CLIP input format.

        Steps:
          1. Resize to 336x336 (CLIP-ViT-L/14 expected input)
          2. Center-crop
          3. Normalize with CLIP mean/std
          4. Return as [3, 336, 336] tensor
        """
        # Resize shortest edge to 336, preserve aspect ratio
        image = image.convert("RGB")
        w, h = image.size
        scale = self.image_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.BICUBIC)

        # Center crop
        left = (new_w - self.image_size) // 2
        top = (new_h - self.image_size) // 2
        image = image.crop((left, top, left + self.image_size, top + self.image_size))

        # Normalize
        img_array = np.array(image, dtype=np.float32) / 255.0
        img_array = (img_array - self.mean) / self.std

        # CHW format
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)  # [1, 3, 336, 336]
        return tensor

    def preprocess_bytes(self, image_bytes: bytes) -> torch.Tensor:
        """Preprocess raw image bytes."""
        image = Image.open(BytesIO(image_bytes))
        return self.preprocess(image)


class LLaVAWorker:
    """
    LLaVA-1.5 inference worker backed by vLLM.

    Usage pattern:
        worker = LLaVAWorker(config)
        await worker.initialize()

        # Single request
        output = await worker.generate(request)

        # Batch
        outputs = await worker.generate_batch(requests)

        await worker.shutdown()
    """

    def __init__(self, config: ServingConfig, device_id: int = 0):
        self.config = config
        self.device_id = device_id
        self.device = torch.device(f"cuda:{device_id}")

        # Components (loaded in initialize())
        self.llm: Optional[LLM] = None
        self.tokenizer = None
        self.image_preprocessor: Optional[ImagePreprocessor] = None
        self.vision_tower: Optional[CLIPVisionModel] = None
        self.vision_processor: Optional[CLIPImageProcessor] = None
        self.mm_projector: Optional[torch.nn.Module] = None

        # Statistics
        self.num_requests_processed = 0
        self.total_tokens_generated = 0
        self._initialized = False

    async def initialize(self):
        """Load model, vision tower, and tokenizer."""

        print(f"[LLaVAWorker] Initializing on GPU {self.device_id}...")
        print(f"  Model: {self.config.model.model_name}")
        print(f"  Max seq len: {self.config.model.max_model_len}")
        print(f"  Quantization: {self.config.model.quantization or 'none'}")

        # ── Initialize vLLM engine ─────────────────────────────────────────
        # vLLM provides PagedAttention, continuous batching, and KV-cache mgmt
        self.llm = LLM(
            model=self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
            max_model_len=self.config.model.max_model_len,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            dtype=self.config.model.dtype,
            quantization=self.config.model.quantization,
            enforce_eager=self.config.enforce_eager,
            max_num_seqs=self.config.scheduler.max_num_seqs,
            max_num_batched_tokens=self.config.scheduler.max_num_batched_tokens,
            enable_prefix_caching=self.config.scheduler.enable_prefix_caching,
            enable_chunked_prefill=self.config.scheduler.enable_chunked_prefill,
            block_size=self.config.scheduler.block_size,
            swap_space=self.config.swap_space,
        )

        # ── Tokenizer ──────────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.model_name,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        # LLaVA uses a special image token
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

        # ── Vision preprocessor ────────────────────────────────────────────
        self.image_preprocessor = ImagePreprocessor(self.config)

        # ── Vision tower (CLIP-ViT-L/14) ──────────────────────────────────
        self.vision_processor = CLIPImageProcessor.from_pretrained(
            self.config.model.vision_model
        )
        self.vision_tower = CLIPVisionModel.from_pretrained(
            self.config.model.vision_model,
            torch_dtype=torch.float16,
        ).to(self.device)
        self.vision_tower.eval()

        # ── Multimodal projector ──────────────────────────────────────────
        # Load the complete LLaVA model to extract projector weights
        llava_model = LlavaForConditionalGeneration.from_pretrained(
            self.config.model.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=self.config.model.trust_remote_code,
        )
        self.mm_projector = llava_model.multi_modal_projector.to(self.device)
        self.mm_projector.eval()

        # Free the full model (keep only projector)
        del llava_model
        torch.cuda.empty_cache()

        self._initialized = True
        print(f"[LLaVAWorker] Initialized. Vision tower: {self.config.model.vision_model}")

    def _encode_image(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract image features and project to language embedding space.

        Pipeline:
          image [1, 3, 336, 336]
            → CLIP ViT-L → patch features [1, 577, 1024]
            → mm_projector → image embeddings [1, 576, 4096]
        """
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device, dtype=torch.float16)

            # CLIP vision encoder
            vision_outputs = self.vision_tower(pixel_values=image_tensor)
            # Use the last hidden state (before pooling)
            image_features = vision_outputs.last_hidden_state  # [1, 577, 1024]

            # Select patch features (exclude CLS token)
            # CLIP ViT output: [CLS, patch_1, patch_2, ..., patch_576]
            image_features = image_features[:, 1:, :]  # [1, 576, 1024]

            # Multimodal projection → language embedding space
            image_embeddings = self.mm_projector(image_features)  # [1, 576, 4096]

        return image_embeddings  # [1, num_patches, hidden_size]

    def _build_prompt(
        self,
        text: str,
        has_image: bool = True,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Build the LLaVA conversation prompt.

        LLaVA-1.5 uses a specific conversation template with <image> tokens.
        """
        if system_prompt is None:
            system_prompt = (
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions."
            )

        if has_image:
            # LLaVA-1.5 image prompt format
            prompt = (
                f"{system_prompt} "
                f"USER: <image>\n{text} "
                f"ASSISTANT:"
            )
        else:
            prompt = (
                f"{system_prompt} "
                f"USER: {text} "
                f"ASSISTANT:"
            )

        return prompt

    async def generate(self, request: BatchRequest) -> GenerationOutput:
        """
        Generate a response for a single request.

        Args:
            request: BatchRequest with prompt, optional image, and sampling params

        Returns:
            GenerationOutput with generated text and metrics
        """
        start_time = time.perf_counter()

        # ── Preprocess image (if provided) ─────────────────────────────────
        image_embeddings = None
        has_image = False

        if request.image_data is not None:
            image_tensor = self.image_preprocessor.preprocess_bytes(request.image_data)
            image_embeddings = self._encode_image(image_tensor)
            has_image = True
        elif request.image_tensor is not None:
            image_embeddings = self._encode_image(request.image_tensor)
            has_image = True

        # ── Build prompt ───────────────────────────────────────────────────
        prompt = self._build_prompt(request.prompt, has_image=has_image)

        # ── Configure sampling ─────────────────────────────────────────────
        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop_sequences or None,
        )

        # ── Run generation via vLLM ───────────────────────────────────────
        # vLLM handles KV-cache + continuous batching internally
        if image_embeddings is not None:
            # Pass image embeddings as multi-modal data
            mm_data = MultiModalData(
                image_embeds=image_embeddings,
            )
            outputs = self.llm.generate(
                [prompt],
                sampling_params=sampling_params,
                multi_modal_data=mm_data,
            )
        else:
            outputs = self.llm.generate(
                [prompt],
                sampling_params=sampling_params,
            )

        # ── Extract output ─────────────────────────────────────────────────
        output = outputs[0]
        generated_text = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        completion_tokens = len(output.outputs[0].token_ids)

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        tokens_per_sec = completion_tokens / (elapsed_ms / 1000) if elapsed_ms > 0 else 0

        # Determine finish reason
        finish_reason = output.outputs[0].finish_reason or "stop"
        if completion_tokens >= request.max_tokens:
            finish_reason = "length"

        self.num_requests_processed += 1
        self.total_tokens_generated += completion_tokens

        return GenerationOutput(
            request_id=request.request_id,
            text=generated_text,
            finish_reason=finish_reason,
            tokens_generated=completion_tokens,
            prompt_tokens=prompt_tokens,
            total_time_ms=elapsed_ms,
            tokens_per_second=tokens_per_sec,
        )

    async def generate_batch(
        self,
        requests: List[BatchRequest],
    ) -> List[GenerationOutput]:
        """
        Generate responses for a batch of requests.

        vLLM automatically applies continuous batching internally —
        this method just submits all requests and lets vLLM schedule them.
        """
        start_time = time.perf_counter()

        # ── Process all images in batch ────────────────────────────────────
        prompts = []
        all_image_embeddings = []

        for req in requests:
            image_emb = None
            has_image = False

            if req.image_data is not None:
                img_tensor = self.image_preprocessor.preprocess_bytes(req.image_data)
                image_emb = self._encode_image(img_tensor)
                has_image = True
            elif req.image_tensor is not None:
                image_emb = self._encode_image(req.image_tensor)
                has_image = True

            prompt = self._build_prompt(req.prompt, has_image=has_image)
            prompts.append(prompt)
            all_image_embeddings.append(image_emb)

        # ── Sampling params (shared across batch) ──────────────────────────
        # Each request can have different params — use per-request params
        sampling_params_list = [
            SamplingParams(
                max_tokens=r.max_tokens,
                temperature=r.temperature,
                top_p=r.top_p,
                top_k=r.top_k,
                stop=r.stop_sequences or None,
            )
            for r in requests
        ]

        # ── Generate with vLLM continuous batching ─────────────────────────
        # vLLM internally interleaves prefill and decode phases across
        # all sequences, maximizing GPU utilization.
        results = []
        for i, (prompt, img_emb, sp) in enumerate(
            zip(prompts, all_image_embeddings, sampling_params_list)
        ):
            if img_emb is not None:
                mm_data = MultiModalData(image_embeds=img_emb)
                output = self.llm.generate([prompt], sampling_params=sp, multi_modal_data=mm_data)
            else:
                output = self.llm.generate([prompt], sampling_params=sp)

            out = output[0]
            gen_text = out.outputs[0].text
            prompt_tok = len(out.prompt_token_ids)
            comp_tok = len(out.outputs[0].token_ids)

            results.append(GenerationOutput(
                request_id=requests[i].request_id,
                text=gen_text,
                finish_reason=out.outputs[0].finish_reason or "stop",
                tokens_generated=comp_tok,
                prompt_tokens=prompt_tok,
                total_time_ms=(time.perf_counter() - start_time) * 1000,
                tokens_per_second=comp_tok / ((time.perf_counter() - start_time)) if comp_tok > 0 else 0,
            ))

        self.num_requests_processed += len(requests)
        self.total_tokens_generated += sum(r.tokens_generated for r in results)

        return results

    async def generate_stream(
        self,
        request: BatchRequest,
    ) -> AsyncIterator[str]:
        """
        Stream generated tokens one by one.

        Uses vLLM's streaming API for token-by-token generation.
        """
        image_embeddings = None
        has_image = False

        if request.image_data is not None:
            image_tensor = self.image_preprocessor.preprocess_bytes(request.image_data)
            image_embeddings = self._encode_image(image_tensor)
            has_image = True
        elif request.image_tensor is not None:
            image_embeddings = self._encode_image(request.image_tensor)
            has_image = True

        prompt = self._build_prompt(request.prompt, has_image=has_image)

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop=request.stop_sequences or None,
        )

        # vLLM streaming generator
        if image_embeddings is not None:
            mm_data = MultiModalData(image_embeds=image_embeddings)
            stream = self.llm.generate(
                [prompt],
                sampling_params=sampling_params,
                multi_modal_data=mm_data,
                use_tqdm=False,
            )
        else:
            stream = self.llm.generate(
                [prompt],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

        # Yield each token
        # Note: vLLM's streaming works per-request; for true token-level
        # streaming, use AsyncLLMEngine (see api_server.py)
        for output in stream:
            for token in output.outputs[0].token_ids:
                text = self.tokenizer.decode(token, skip_special_tokens=True)
                yield text

    def stats(self) -> Dict[str, Any]:
        """Return worker statistics."""
        return {
            "requests_processed": self.num_requests_processed,
            "tokens_generated": self.total_tokens_generated,
            "gpu_utilization": torch.cuda.utilization(self.device_id) if torch.cuda.is_available() else 0,
            "gpu_memory_used": torch.cuda.memory_allocated(self.device_id) / 1024**3 if torch.cuda.is_available() else 0,
        }

    async def shutdown(self):
        """Clean up resources."""
        print("[LLaVAWorker] Shutting down...")
        if self.llm is not None:
            del self.llm
        if self.vision_tower is not None:
            self.vision_tower.cpu()
        if self.mm_projector is not None:
            self.mm_projector.cpu()
        torch.cuda.empty_cache()
        self._initialized = False
        print("[LLaVAWorker] Shutdown complete.")
