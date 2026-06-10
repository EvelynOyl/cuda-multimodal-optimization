"""
vllm_deploy/continuous_batching.py — Continuous (in-flight) batching scheduler.

This module implements the scheduling logic for continuous batching — the key
technique that makes vLLM 10-20× higher throughput than static batching.

Core idea:
  Instead of waiting for ALL sequences in a batch to finish decoding before
  starting the next batch, we dynamically add new sequences as soon as slots
  free up. The scheduler manages three phase transitions per sequence:

    1. WAITING   → PREFILL   (KV-cache allocated, prompt processed)
    2. PREFILL   → DECODE    (one token generated per step)
    3. DECODE    → FINISHED  (EOS token or max_tokens reached)

  At each iteration, the scheduler decides:
    - Which waiting sequences to admit (subject to KV-cache budget)
    - How to pack prefill + decode tokens within the token budget
    - When to evict finished sequences

References:
  - Yu et al. "Orca: A Distributed Serving System for Transformer-Based
    Generative Models." OSDI 2022.
  - Kwon et al. "Efficient Memory Management for Large Language Model
    Serving with PagedAttention." SOSP 2023.
  - vLLM source: https://github.com/vllm-project/vllm
"""

import time
import heapq
import asyncio
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Deque
from collections import deque

from .config import SchedulerConfig, BatchRequest, GenerationOutput


class SequenceState(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class Sequence:
    """Metadata for one sequence in the batch."""
    request: BatchRequest
    state: SequenceState = SequenceState.WAITING

    # Token IDs (accumulated during prefill + decode)
    prompt_token_ids: List[int] = field(default_factory=list)
    output_token_ids: List[int] = field(default_factory=list)

    # KV-cache tracking
    num_cached_tokens: int = 0         # tokens already in KV-cache
    num_computed_tokens: int = 0       # total tokens computed so far
    num_kv_blocks: int = 0             # KV-cache blocks allocated

    # Timing
    arrival_time: float = 0.0
    prefill_start_time: float = 0.0
    first_token_time: Optional[float] = None  # Time to first token (TTFT)

    # Completion
    finish_reason: str = ""
    finished: bool = False

    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    def num_total_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def is_finished(self) -> bool:
        return self.state in (SequenceState.FINISHED, SequenceState.ABORTED)


class ContinuousBatchingScheduler:
    """
    Implements continuous batching scheduling strategy.

    Design notes:
      - vLLM internally handles most of this logic via its Scheduler class.
        This is a reference implementation to illustrate the algorithm.
      - In production, use vLLM's built-in AsyncLLMEngine which includes
        a production-hardened continuous batching scheduler.
      - This scheduler demonstrates the key concepts: dynamic admission,
        prefill-decode interleaving, and KV-cache budget management.

    Key parameters:
      - max_num_seqs: Hard cap on concurrent sequences
      - max_num_batched_tokens: Total tokens processed per step (prefill + decode)
      - block_size: KV-cache block size for paged attention
    """

    def __init__(self, config: SchedulerConfig, num_gpu_blocks: int = 10000):
        self.config = config
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_num_prefill_tokens = config.max_num_prefill_tokens
        self.block_size = config.block_size
        self.enable_chunked_prefill = config.enable_chunked_prefill
        self.enable_prefix_caching = config.enable_prefix_caching

        # KV-cache block pool
        self.num_gpu_blocks = num_gpu_blocks
        self.free_blocks: Deque[int] = deque(range(num_gpu_blocks))
        self.allocated_blocks: Dict[str, List[int]] = {}  # request_id → block_ids

        # Sequences
        self.waiting_queue: Deque[Sequence] = deque()
        self.running_sequences: Dict[str, Sequence] = {}
        self.finished_sequences: Dict[str, Sequence] = {}

        # Prefix cache: hash(prompt_prefix) → KV block addresses
        self.prefix_cache: Dict[str, Tuple[int, List[int]]] = {}

        # Stats
        self.num_prefills_completed = 0
        self.num_tokens_prefilled = 0
        self.num_tokens_decoded = 0
        self.total_iterations = 0

    # ═══════════════════════════════════════════════════════════════════════
    # Scheduler Step (called once per iteration)
    # ═══════════════════════════════════════════════════════════════════════

    def schedule(self) -> Dict[str, List[int]]:
        """
        Run one scheduling iteration.

        Returns a dict mapping request_id → [new_token_ids to process].

        The scheduling algorithm:

        1. EVICT finished sequences → free KV blocks
        2. ADMIT waiting sequences (subject to budget)
           - Check if enough free KV blocks exist
           - Check max_num_seqs cap
           - Allocate blocks for new sequences
        3. SELECT sequences for this step
           - Running decode sequences each contribute 1 token
           - Newly admitted sequences need prefill (many tokens)
           - Respect max_num_batched_tokens budget
           - If chunked prefill: split long prefills across steps
        4. COMPUTE token budget allocation
        """
        self.total_iterations += 1

        # ── Step 1: Evict finished sequences ─────────────────────────────
        self._evict_finished()

        # ── Step 2: Admit new sequences ──────────────────────────────────
        self._admit_sequences()

        # ── Step 3: Build the schedule ────────────────────────────────────
        schedule = {}
        remaining_token_budget = self.max_num_batched_tokens

        # Phase A: Running decode sequences (1 token each)
        for seq_id, seq in list(self.running_sequences.items()):
            if seq.state == SequenceState.DECODE:
                if remaining_token_budget >= 1:
                    schedule[seq_id] = [1]  # decode: 1 token per step
                    remaining_token_budget -= 1

        # Phase B: Prefill for newly admitted sequences
        for seq_id, seq in list(self.running_sequences.items()):
            if seq.state == SequenceState.PREFILL:
                remaining_prompt_tokens = (
                    seq.num_prompt_tokens() - seq.num_computed_tokens
                )

                if self.enable_chunked_prefill:
                    # Chunked prefill: process up to max_num_prefill_tokens
                    tokens_to_prefill = min(
                        remaining_prompt_tokens,
                        self.max_num_prefill_tokens,
                        remaining_token_budget,
                    )
                else:
                    # Full prefill: all or nothing
                    tokens_to_prefill = (
                        remaining_prompt_tokens
                        if remaining_prompt_tokens <= remaining_token_budget
                        else 0
                    )

                if tokens_to_prefill > 0:
                    schedule[seq_id] = [tokens_to_prefill]
                    remaining_token_budget -= tokens_to_prefill

        return schedule

    def _evict_finished(self):
        """Free KV-cache blocks for finished sequences."""
        for seq_id, seq in list(self.running_sequences.items()):
            if seq.state in (SequenceState.FINISHED, SequenceState.ABORTED):
                # Move to finished
                self.finished_sequences[seq_id] = seq

                # Free KV blocks
                if seq_id in self.allocated_blocks:
                    for block_id in self.allocated_blocks.pop(seq_id):
                        self.free_blocks.append(block_id)

                del self.running_sequences[seq_id]

    def _admit_sequences(self):
        """
        Admit waiting sequences if budget allows.

        Admission criteria:
          1. num_running < max_num_seqs
          2. Enough free KV blocks for the new sequence's prompt
        """
        while self.waiting_queue and len(self.running_sequences) < self.max_num_seqs:
            seq = self.waiting_queue.popleft()

            # Estimate KV blocks needed for this sequence
            blocks_needed = self._estimate_blocks(seq)

            if len(self.free_blocks) >= blocks_needed:
                # Allocate blocks
                allocated = []
                for _ in range(blocks_needed):
                    if self.free_blocks:
                        allocated.append(self.free_blocks.popleft())
                self.allocated_blocks[seq.request.request_id] = allocated
                seq.num_kv_blocks = len(allocated)

                # Transition to PREFILL
                seq.state = SequenceState.PREFILL
                seq.prefill_start_time = time.time()
                self.running_sequences[seq.request.request_id] = seq

                # Track prompt tokens (would be set during tokenization)
                if not seq.prompt_token_ids:
                    # Placeholder: tokenize would fill this
                    seq.prompt_token_ids = list(range(min(100, seq.request.max_tokens)))

            else:
                # Not enough blocks — put back in front
                self.waiting_queue.appendleft(seq)
                break  # Can't admit more

    def _estimate_blocks(self, seq: Sequence) -> int:
        """
        Estimate the number of KV-cache blocks needed.

        Each block stores `block_size` tokens worth of KV cache.
        We need blocks for the max total tokens (prompt + max_tokens).

        Returns:
            Number of GPU memory blocks needed
        """
        max_total_tokens = seq.num_prompt_tokens() + seq.request.max_tokens
        blocks = (max_total_tokens + self.block_size - 1) // self.block_size
        return max(blocks, 1)

    # ═══════════════════════════════════════════════════════════════════════
    # Request Lifecycle
    # ═══════════════════════════════════════════════════════════════════════

    def add_request(self, request: BatchRequest):
        """
        Add a new request to the waiting queue.

        This is called by the API server when a client submits a request.
        """
        seq = Sequence(
            request=request,
            state=SequenceState.WAITING,
            arrival_time=time.time(),
        )
        self.waiting_queue.append(seq)
        return seq

    def mark_step_complete(
        self,
        seq_id: str,
        tokens_processed: int,
        new_token_ids: Optional[List[int]] = None,
        finish_reason: Optional[str] = None,
    ):
        """
        Called after each step to update sequence state.

        Args:
            seq_id: Request ID
            tokens_processed: Number of tokens processed this step
            new_token_ids: New output token IDs (for decode steps)
            finish_reason: If set, marks the sequence as finished
        """
        if seq_id not in self.running_sequences:
            return

        seq = self.running_sequences[seq_id]

        if seq.state == SequenceState.PREFILL:
            seq.num_computed_tokens += tokens_processed

            # Check if prefill complete
            if seq.num_computed_tokens >= seq.num_prompt_tokens():
                # Transition to DECODE
                seq.state = SequenceState.DECODE
                seq.first_token_time = time.time()
                self.num_prefills_completed += 1

        elif seq.state == SequenceState.DECODE:
            if new_token_ids:
                seq.output_token_ids.extend(new_token_ids)
            seq.num_computed_tokens += tokens_processed
            self.num_tokens_decoded += 1

            # Check stop conditions
            if finish_reason:
                seq.finish_reason = finish_reason
                seq.state = SequenceState.FINISHED
            elif len(seq.output_token_ids) >= seq.request.max_tokens:
                seq.finish_reason = "length"
                seq.state = SequenceState.FINISHED

    def abort_request(self, seq_id: str):
        """Abort a running request."""
        if seq_id in self.running_sequences:
            self.running_sequences[seq_id].state = SequenceState.ABORTED

    # ── Prefix Caching (experimental) ──────────────────────────────────────

    def _hash_prompt_prefix(self, prefix_token_ids: List[int]) -> str:
        """Hash a token prefix for caching."""
        # Simple hash — production would use a proper rolling hash
        return str(hash(tuple(prefix_token_ids[:32])))  # first 32 tokens

    def try_prefix_match(self, prompt_token_ids: List[int]) -> Optional[int]:
        """
        Check if any prefix of `prompt_token_ids` is already cached.
        Returns the number of cached tokens if found.

        This enables sharing KV-cache across requests with common
        prefixes (e.g., system prompts).
        """
        if not self.enable_prefix_caching:
            return None

        # Check progressively shorter prefixes
        for length in range(min(len(prompt_token_ids), 128), 0, -32):
            prefix = prompt_token_ids[:length]
            prefix_hash = self._hash_prompt_prefix(prefix)
            if prefix_hash in self.prefix_cache:
                return length

        return None

    # ── Diagnostic / Debug ─────────────────────────────────────────────────

    def stats(self) -> Dict:
        """Return scheduler statistics."""
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_sequences),
            "finished": len(self.finished_sequences),
            "free_blocks": len(self.free_blocks),
            "free_block_pct": len(self.free_blocks) / self.num_gpu_blocks * 100,
            "num_sequences": len(self.running_sequences),
            "max_concurrency": self.max_num_seqs,
            "total_iterations": self.total_iterations,
            "prefills_completed": self.num_prefills_completed,
            "tokens_decoded": self.num_tokens_decoded,
        }

    def running_summary(self) -> str:
        """Human-readable summary of running state."""
        states = {}
        for seq in self.running_sequences.values():
            states[seq.state.value] = states.get(seq.state.value, 0) + 1
        return (
            f"Scheduler[waiting={len(self.waiting_queue)}, "
            f"running={len(self.running_sequences)} "
            f"({', '.join(f'{k}={v}' for k, v in states.items())}), "
            f"free_blocks={len(self.free_blocks)}/{self.num_gpu_blocks}]"
        )
