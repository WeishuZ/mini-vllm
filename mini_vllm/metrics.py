"""Metrics collected over a simulated run. Times are in milliseconds on the
engine's simulated clock."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


def percentile(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    if len(s) == 1:
        return s[0]
    pos = q / 100.0 * (len(s) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


@dataclass
class StepStat:
    t_ms: float
    running: int
    decode_seqs: int
    prefill_tokens: int
    gpu_util: float
    waiting: int = 0
    swapped: int = 0
    finished: int = 0
    preempted: int = 0
    swapped_in: int = 0
    swapped_out: int = 0
    prefix_hits: int = 0
    prefix_cache_evictions: int = 0


@dataclass
class EngineMetrics:
    num_requests: int = 0
    num_completed: int = 0
    total_generated_tokens: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    makespan_ms: float = 0.0
    num_steps: int = 0

    throughput_tok_s: float = 0.0

    queue_ms_mean: float = 0.0
    queue_ms_p50: float = 0.0
    queue_ms_p99: float = 0.0

    ttft_ms_mean: float = 0.0
    ttft_ms_p50: float = 0.0
    ttft_ms_p99: float = 0.0

    itl_ms_mean: float = 0.0
    itl_ms_p50: float = 0.0
    itl_ms_p99: float = 0.0

    tpot_ms_mean: float = 0.0
    tpot_ms_p50: float = 0.0
    tpot_ms_p99: float = 0.0

    e2e_ms_mean: float = 0.0
    e2e_ms_p50: float = 0.0
    e2e_ms_p99: float = 0.0

    num_preemptions: int = 0
    num_swaps: int = 0
    num_cow: int = 0
    peak_gpu_util: float = 0.0
    prefix_cache_hit_rate: float = 0.0
    prefix_cache_saved_tokens: int = 0
    prefix_cache_evictions: int = 0
    prefix_cache_blocks: int = 0
    prefix_cache_pinned_blocks: int = 0
    prefix_cache_evictable_blocks: int = 0

    def as_dict(self) -> dict:
        return {k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.__dict__.items()}

    def summary(self) -> str:
        return (
            f"requests={self.num_completed}/{self.num_requests}  "
            f"throughput={self.throughput_tok_s:7.1f} tok/s  "
            f"queue p50={self.queue_ms_p50:6.1f} ms  "
            f"TTFT p50/p99={self.ttft_ms_p50:6.1f}/{self.ttft_ms_p99:6.1f} ms  "
            f"ITL p50={self.itl_ms_p50:5.1f} ms  "
            f"e2e p50={self.e2e_ms_p50:7.1f} ms  "
            f"preempt={self.num_preemptions}  swap={self.num_swaps}  "
            f"peakKV={self.peak_gpu_util*100:4.1f}%  "
            f"cache_hit={self.prefix_cache_hit_rate*100:4.1f}%  "
            f"saved_prefill={self.prefix_cache_saved_tokens}"
        )
