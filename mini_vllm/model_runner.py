"""Simulated execution backend.

We do **not** run a real transformer — there is no GPU and the interesting parts
of a serving engine are memory management and scheduling, not the matmuls. So we
model a forward pass with a deterministic latency function whose *shape* matches
real inference:

* **prefill** is roughly linear in the number of tokens processed (compute
  bound): ``prefill_ms_per_token * tokens``.
* **decode** is memory-bandwidth bound: the per-step cost grows only mildly with
  batch size (``decode_ms_base + decode_ms_per_seq * batch``), so per-token cost
  *falls* as the batch grows. That sub-linearity is exactly why batching works,
  and it's the property the throughput benchmark measures.

Everything is deterministic given ``ModelConfig.seed``, so benchmark numbers are
reproducible run to run.
"""
from __future__ import annotations

from .config import ModelConfig
from .scheduler import StepWork


class ModelRunner:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg

    def step_latency_ms(self, work: StepWork) -> float:
        """Wall-clock cost (ms) of executing one mixed prefill+decode batch."""
        if work.is_empty:
            return 0.0
        c = self.cfg
        latency = c.step_overhead_ms
        latency += c.prefill_ms_per_token * work.num_prefill_tokens
        d = work.num_decode_seqs
        if d > 0:
            latency += c.decode_ms_base + c.decode_ms_per_seq * d
        return latency
