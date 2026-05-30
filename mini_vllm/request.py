"""Request and Sequence state.

A ``Request`` is what a user submits. A ``Sequence`` is the engine's mutable
bookkeeping for that request as it moves WAITING -> RUNNING -> FINISHED (with
possible detours through SWAPPED / back to WAITING on preemption).

Token accounting follows vLLM's mental model:

* ``length``       = prompt_len + num_generated  (all tokens that exist)
* ``num_computed`` = leading tokens whose KV is materialized in the cache
* prefill is "remaining" while num_computed < length (true initially, and again
  after a recompute-preemption, when generated tokens must be recomputed).
* during decode, num_computed == length after each step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class SeqStatus(Enum):
    WAITING = "waiting"
    RUNNING = "running"
    SWAPPED = "swapped"
    FINISHED = "finished"


@dataclass
class Request:
    """A unit of work submitted to the engine."""

    request_id: str
    prompt_len: int
    max_tokens: int
    arrival: float = 0.0
    # Optional concrete prompt tokens. Only needed to exercise prefix caching;
    # workloads that don't care can leave this None.
    token_ids: Optional[List[int]] = None


@dataclass
class Sequence:
    """Mutable per-request engine state."""

    request_id: str
    prompt_len: int
    max_tokens: int
    arrival: float = 0.0
    token_ids: Optional[List[int]] = None

    status: SeqStatus = SeqStatus.WAITING
    num_computed: int = 0
    num_generated: int = 0
    block_table: List[int] = field(default_factory=list)  # physical GPU block ids

    # metrics (simulated-clock timestamps, seconds)
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    num_cached_tokens: int = 0      # prompt tokens served from the prefix cache
    num_preemptions: int = 0

    @classmethod
    def from_request(cls, req: Request) -> "Sequence":
        return cls(
            request_id=req.request_id,
            prompt_len=req.prompt_len,
            max_tokens=req.max_tokens,
            arrival=req.arrival,
            token_ids=req.token_ids,
        )

    # --- derived token accounting ---
    @property
    def length(self) -> int:
        """Total tokens that exist for this sequence (prompt + generated)."""
        return self.prompt_len + self.num_generated

    @property
    def num_kv_tokens(self) -> int:
        """Tokens currently materialized in the KV cache."""
        return self.num_computed

    @property
    def prefill_remaining(self) -> int:
        return self.length - self.num_computed

    @property
    def in_prefill(self) -> bool:
        return self.prefill_remaining > 0

    @property
    def is_finished(self) -> bool:
        return self.num_generated >= self.max_tokens

    def reset_for_recompute(self) -> None:
        """Drop materialized KV; keep generated tokens (they become prefill)."""
        self.num_computed = 0
        self.block_table = []
        self.status = SeqStatus.WAITING
        self.num_preemptions += 1
