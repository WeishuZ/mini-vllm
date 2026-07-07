"""Configuration objects for the mini-vLLM engine.

The numbers here intentionally mirror the *shape* of a real serving stack: a
fixed pool of fixed-size KV-cache blocks (the "GPU"), an optional CPU swap pool,
and a per-step token budget for the scheduler. The latency model is simulated
(see :mod:`mini_vllm.model_runner`) so the whole thing runs on a laptop with no
GPU, while the memory management and scheduling logic are real.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CacheConfig:
    """KV-cache layout.

    Attributes:
        block_size: tokens stored per physical block (vLLM default is 16).
        num_gpu_blocks: size of the fast (GPU) block pool. This is the hard cap
            on how many token-slots of KV state can be resident at once.
        num_cpu_blocks: size of the CPU pool used for swap-based preemption.
        enable_prefix_caching: share identical leading prompt blocks across
            sequences (the page-cache analogue).
        prefix_cache_max_blocks: maximum number of GPU blocks the prefix cache
            may retain. ``None`` means the cache can grow to the GPU pool size;
            active, referenced blocks are pinned and may temporarily exceed a
            smaller budget until they become idle.
    """

    block_size: int = 16
    num_gpu_blocks: int = 512
    num_cpu_blocks: int = 1024
    enable_prefix_caching: bool = False
    prefix_cache_max_blocks: int | None = None

    @property
    def total_gpu_slots(self) -> int:
        return self.block_size * self.num_gpu_blocks


@dataclass
class SchedulerConfig:
    """Admission / batching policy.

    Attributes:
        max_num_seqs: max sequences running (decoding) concurrently.
        max_num_batched_tokens: per-step compute budget. Caps prefill tokens +
            decode tokens scheduled in a single forward pass.
        enable_chunked_prefill: split long prefills across steps so they don't
            block ongoing decodes (Sarathi-Serve style).
        policy: "continuous" (admit/finish requests every step) or "static"
            (classic fixed-batch: drain the whole batch before admitting more).
        preemption_mode: "recompute" (drop KV, re-prefill later — vLLM default)
            or "swap" (page KV out to the CPU pool).
    """

    max_num_seqs: int = 32
    max_num_batched_tokens: int = 2048
    enable_chunked_prefill: bool = True
    policy: str = "continuous"
    preemption_mode: str = "recompute"
    # Keep this fraction of the KV pool free when admitting new prefills, so
    # running sequences have headroom to grow before they must be preempted.
    # This is the single most important knob for avoiding preemption thrashing
    # (vLLM calls it the watermark; default there is ~1%).
    watermark: float = 0.04


@dataclass
class ModelConfig:
    """Simulated cost model (milliseconds).

    A forward pass costs a linear term in the number of prefill tokens plus a
    decode term that is sub-linear *per token* in the batch size — this is what
    makes batching pay off, and it's the property the benchmarks probe.
    """

    prefill_ms_per_token: float = 0.10
    decode_ms_base: float = 6.0
    decode_ms_per_seq: float = 0.35
    step_overhead_ms: float = 0.5
    seed: int = 0
