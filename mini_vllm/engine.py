"""The serving loop: glue the block manager, scheduler, and (simulated) runner
into something you can feed requests and read metrics from.

Usage::

    engine = LLMEngine(CacheConfig(), SchedulerConfig(), ModelConfig())
    for r in requests:
        engine.add_request(r)
    metrics = engine.run()
    print(metrics.summary())
"""
from __future__ import annotations

from typing import Dict, List

from .block_manager import BlockManager
from .config import CacheConfig, ModelConfig, SchedulerConfig
from .metrics import EngineMetrics, StepStat, percentile
from .model_runner import ModelRunner
from .request import Request, Sequence
from .scheduler import Scheduler, StepWork


class LLMEngine:
    def __init__(
        self,
        cache_config: CacheConfig | None = None,
        scheduler_config: SchedulerConfig | None = None,
        model_config: ModelConfig | None = None,
    ):
        self.cache_config = cache_config or CacheConfig()
        self.scheduler_config = scheduler_config or SchedulerConfig()
        self.model_config = model_config or ModelConfig()

        self.block_manager = BlockManager(self.cache_config)
        self.scheduler = Scheduler(self.scheduler_config, self.block_manager)
        self.runner = ModelRunner(self.model_config)

        self.clock_ms: float = 0.0
        self.num_steps: int = 0
        self.total_prefill_tokens: int = 0
        self.total_decode_tokens: int = 0
        self.num_preemptions: int = 0
        self.num_swaps: int = 0

        self._pending: List[Sequence] = []          # not yet arrived
        self.sequences: Dict[str, Sequence] = {}
        self.completed: List[Sequence] = []
        self.history: List[StepStat] = []

    # ----------------------------------------------------------- submission
    def add_request(self, req: Request) -> None:
        seq = Sequence.from_request(req)
        self.sequences[seq.request_id] = seq
        self._pending.append(seq)

    def add_requests(self, reqs: List[Request]) -> None:
        for r in reqs:
            self.add_request(r)

    def _release_arrivals(self) -> None:
        still: List[Sequence] = []
        for seq in self._pending:
            if seq.arrival <= self.clock_ms:
                self.scheduler.add(seq)
            else:
                still.append(seq)
        self._pending = still

    # ------------------------------------------------------------------ step
    def step(self) -> StepWork:
        work = self.scheduler.schedule()

        # First admission timestamp. This is queue time, not TTFT: a sequence
        # can spend one or more steps in prefill before it emits a token.
        for seq in self.scheduler.running:
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = self.clock_ms

        # Apply prefill progress (KV for these tokens is now materialized).
        for seq, chunk in work.prefill:
            seq.num_computed += chunk
            self.block_manager._register_full_prompt_blocks(seq)
            self.total_prefill_tokens += chunk

        # Apply decode: each running, prefill-complete sequence emits one token.
        for seq in work.decode:
            seq.num_generated += 1
            seq.num_computed += 1
            self.total_decode_tokens += 1

        # Advance the simulated clock by this batch's latency.
        self.clock_ms += self.runner.step_latency_ms(work)
        self.num_steps += 1
        self.num_preemptions += work.preempted
        self.num_swaps += work.swapped_out

        finished: List[Sequence] = []
        for seq in work.decode:
            seq.decode_token_times.append(self.clock_ms)
            if seq.num_generated == 1 and seq.first_token_time is None:
                seq.first_token_time = self.clock_ms
            if seq.is_finished:
                seq.finish_time = self.clock_ms
                finished.append(seq)
        if finished:
            self.scheduler.drop_finished(finished)
            self.completed.extend(finished)

        self.history.append(
            StepStat(
                t_ms=self.clock_ms,
                running=len(self.scheduler.running),
                decode_seqs=work.num_decode_seqs,
                prefill_tokens=work.num_prefill_tokens,
                gpu_util=self.block_manager.gpu_utilization,
                waiting=len(self.scheduler.waiting),
                swapped=len(self.scheduler.swapped),
                finished=len(self.completed),
                preempted=work.preempted,
                swapped_in=work.swapped_in,
                swapped_out=work.swapped_out,
                prefix_hits=len(work.prefix_hits),
                prefix_cache_evictions=self.block_manager.prefix_cache_evictions,
            )
        )
        return work

    # ------------------------------------------------------------------- run
    def run(self, max_steps: int = 5_000_000) -> EngineMetrics:
        self._pending.sort(key=lambda s: s.arrival)
        consecutive_empty = 0
        while (self._pending or self.scheduler.has_unfinished) and self.num_steps < max_steps:
            self._release_arrivals()
            if not self.scheduler.has_unfinished:
                # idle gap: fast-forward to the next arrival
                if self._pending:
                    self.clock_ms = max(self.clock_ms, self._pending[0].arrival)
                    continue
                break
            work = self.step()
            if work.is_empty:
                consecutive_empty += 1
                if consecutive_empty > 3:
                    break  # no forward progress possible (mis-sized config)
            else:
                consecutive_empty = 0
        return self.metrics()

    # --------------------------------------------------------------- metrics
    def metrics(self) -> EngineMetrics:
        m = EngineMetrics()
        m.num_requests = len(self.sequences)
        m.num_completed = len(self.completed)
        m.total_generated_tokens = self.total_decode_tokens
        m.total_prefill_tokens = self.total_prefill_tokens
        m.total_decode_tokens = self.total_decode_tokens
        m.makespan_ms = self.clock_ms
        m.num_steps = self.num_steps
        if self.clock_ms > 0:
            m.throughput_tok_s = self.total_decode_tokens / (self.clock_ms / 1000.0)

        queues = [
            s.first_scheduled_time - s.arrival
            for s in self.completed
            if s.first_scheduled_time is not None
        ]
        ttfts = [
            s.first_token_time - s.arrival
            for s in self.completed
            if s.first_token_time is not None
        ]
        e2es = [
            s.finish_time - s.arrival
            for s in self.completed
            if s.finish_time is not None
        ]
        itls = [
            times[i] - times[i - 1]
            for s in self.completed
            for times in [s.decode_token_times]
            for i in range(1, len(times))
        ]
        tpots = [
            (s.finish_time - s.first_token_time) / max(1, s.num_generated - 1)
            for s in self.completed
            if s.finish_time is not None
            and s.first_token_time is not None
            and s.num_generated > 1
        ]
        if queues:
            m.queue_ms_mean = sum(queues) / len(queues)
            m.queue_ms_p50 = percentile(queues, 50)
            m.queue_ms_p99 = percentile(queues, 99)
        if ttfts:
            m.ttft_ms_mean = sum(ttfts) / len(ttfts)
            m.ttft_ms_p50 = percentile(ttfts, 50)
            m.ttft_ms_p99 = percentile(ttfts, 99)
        if itls:
            m.itl_ms_mean = sum(itls) / len(itls)
            m.itl_ms_p50 = percentile(itls, 50)
            m.itl_ms_p99 = percentile(itls, 99)
        if tpots:
            m.tpot_ms_mean = sum(tpots) / len(tpots)
            m.tpot_ms_p50 = percentile(tpots, 50)
            m.tpot_ms_p99 = percentile(tpots, 99)
        if e2es:
            m.e2e_ms_mean = sum(e2es) / len(e2es)
            m.e2e_ms_p50 = percentile(e2es, 50)
            m.e2e_ms_p99 = percentile(e2es, 99)

        m.num_preemptions = self.num_preemptions
        m.num_swaps = self.num_swaps
        m.num_cow = self.block_manager.num_cow
        m.peak_gpu_util = self.block_manager.peak_gpu_blocks_used / self.block_manager.num_gpu_blocks
        if self.block_manager.cache_query_blocks > 0:
            m.prefix_cache_hit_rate = (
                self.block_manager.cache_hit_blocks / self.block_manager.cache_query_blocks
            )
        m.prefix_cache_saved_tokens = self.block_manager.prefix_cache_saved_tokens
        m.prefix_cache_evictions = self.block_manager.prefix_cache_evictions
        m.prefix_cache_blocks = self.block_manager.num_prefix_cache_blocks
        m.prefix_cache_pinned_blocks = self.block_manager.num_pinned_prefix_blocks
        m.prefix_cache_evictable_blocks = self.block_manager.num_evictable_prefix_blocks
        return m
