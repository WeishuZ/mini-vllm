"""Continuous-batching scheduler.

Every step the scheduler decides, under a fixed compute budget and a fixed pool
of KV blocks, which sequences make progress:

* **continuous** policy (the interesting one): in-flight decodes are advanced
  first, then new requests are admitted into the same batch with whatever budget
  is left. Finished sequences free their blocks immediately, so a newly arrived
  request can start the moment one slot opens — no waiting for a whole batch.
* **static** policy (the baseline): a batch is admitted only when the engine is
  idle and reserves worst-case KV up front; no new request joins until the whole
  batch drains. This reproduces head-of-line blocking and the decode-time
  "batch decay" that continuous batching was invented to fix.

When the KV pool is exhausted, the lowest-priority running sequence is
**preempted** — either dropped and recomputed later (vLLM's default) or paged
out to the CPU pool (``swap``). Preempted requests re-enter at the front of the
queue so they keep their place in line.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Tuple

from .block_manager import BlockManager, cdiv
from .config import SchedulerConfig
from .request import Sequence, SeqStatus


@dataclass
class StepWork:
    """The plan for one engine step. Block reservations are already applied;
    token counters are advanced by the engine when it 'runs' the batch."""

    prefill: List[Tuple[Sequence, int]] = field(default_factory=list)  # (seq, n_tokens)
    decode: List[Sequence] = field(default_factory=list)
    preempted: int = 0
    swapped_in: int = 0
    swapped_out: int = 0
    preempted_seq_ids: List[str] = field(default_factory=list)
    swapped_in_seq_ids: List[str] = field(default_factory=list)
    swapped_out_seq_ids: List[str] = field(default_factory=list)
    # (request_id, cached prompt tokens, shared physical block ids)
    prefix_hits: List[Tuple[str, int, List[int]]] = field(default_factory=list)

    @property
    def num_prefill_tokens(self) -> int:
        return sum(n for _, n in self.prefill)

    @property
    def num_decode_seqs(self) -> int:
        return len(self.decode)

    @property
    def is_empty(self) -> bool:
        return not self.prefill and not self.decode


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, block_manager: BlockManager):
        self.cfg = cfg
        self.bm = block_manager
        self.waiting: Deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self.swapped: Deque[Sequence] = deque()
        # swap is only lossless when every block is privately owned, i.e. when
        # prefix-cache sharing is off. Otherwise fall back to recompute.
        self._can_swap = (
            cfg.preemption_mode == "swap" and not block_manager.enable_prefix_caching
        )

    # ------------------------------------------------------------ queue admin
    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    @property
    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running or self.swapped)

    @property
    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running) + len(self.swapped)

    def drop_finished(self, finished: List[Sequence]) -> None:
        fin = set(id(s) for s in finished)
        for s in finished:
            self.bm.free(s)
            s.status = SeqStatus.FINISHED
        self.running = [s for s in self.running if id(s) not in fin]

    # --------------------------------------------------------------- preempt
    def _preempt(self, seq: Sequence, work: StepWork) -> None:
        if self._can_swap and len(seq.block_table) <= self.bm.num_free_cpu_blocks:
            self.bm.swap_out(seq)
            seq.status = SeqStatus.SWAPPED
            seq.num_preemptions += 1
            self.swapped.append(seq)
            work.swapped_out += 1
            work.swapped_out_seq_ids.append(seq.request_id)
        else:
            self.bm.free(seq)
            seq.reset_for_recompute()       # keeps generated tokens; drops KV
            self.waiting.appendleft(seq)    # re-enter at the front
            work.preempted += 1
            work.preempted_seq_ids.append(seq.request_id)

    # ----------------------------------------------------------- core passes
    def _resume_swapped(self, work: StepWork) -> None:
        while self.swapped and len(self.running) < self.cfg.max_num_seqs:
            seq = self.swapped[0]
            if not self.bm.can_swap_in(seq):
                break
            self.bm.swap_in(seq)
            self.swapped.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            work.swapped_in += 1
            work.swapped_in_seq_ids.append(seq.request_id)

    def _advance_running(self, work: StepWork, budget: int, allow_preempt: bool) -> int:
        """Advance every running sequence one step (a prefill chunk or one decode
        token), preempting the newest sequences when the KV pool is exhausted.
        Returns the prefill-token budget left over."""
        self.running.sort(key=lambda s: (s.arrival, s.request_id))  # oldest = priority
        survivors: List[Sequence] = []
        queue: Deque[Sequence] = deque(self.running)
        while queue:
            seq = queue.popleft()
            if seq.in_prefill:
                need = min(seq.prefill_remaining, budget)
                if need <= 0:
                    survivors.append(seq)   # out of budget; idle this step
                    continue
            else:
                need = 1

            while not self.bm.can_grow(seq, need) and allow_preempt and queue:
                self._preempt(queue.pop(), work)   # evict newest first

            if self.bm.can_grow(seq, need):
                self.bm.grow(seq, need)
                if seq.in_prefill:
                    work.prefill.append((seq, need))
                    budget -= need
                else:
                    work.decode.append(seq)
                survivors.append(seq)
            elif allow_preempt:
                self._preempt(seq, work)          # nothing left to evict for it
            else:
                survivors.append(seq)
        self.running = survivors
        return budget

    def _admit_waiting(self, work: StepWork, budget: int) -> int:
        watermark_blocks = int(self.cfg.watermark * self.bm.num_gpu_blocks)
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            if self.bm.num_available_gpu_blocks <= watermark_blocks:
                break  # leave headroom for running sequences to grow
            seq = self.waiting[0]
            before_blocks = len(seq.block_table)
            covered = self.bm.admit_prefix(seq)   # share cached prefix blocks
            shared_blocks = list(seq.block_table[before_blocks:])
            need = seq.prefill_remaining
            if need <= 0:                         # whole prompt was cached
                self.waiting.popleft()
                seq.status = SeqStatus.RUNNING
                self.running.append(seq)
                if covered:
                    work.prefix_hits.append((seq.request_id, covered, shared_blocks))
                continue
            chunk = min(need, budget) if self.cfg.enable_chunked_prefill else need
            if chunk <= 0 or chunk > budget or not self.bm.can_grow(seq, chunk):
                self.bm.free(seq)                 # undo prefix share; try later
                break
            self.bm.grow(seq, chunk)
            work.prefill.append((seq, chunk))
            budget -= chunk
            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)
            if covered:
                work.prefix_hits.append((seq.request_id, covered, shared_blocks))
        return budget

    def _admit_batch_static(self) -> None:
        """Baseline: fill a fresh batch, reserving worst-case KV per sequence."""
        while self.waiting and len(self.running) < self.cfg.max_num_seqs:
            seq = self.waiting[0]
            worst_case_tokens = seq.prompt_len + seq.max_tokens
            blocks_needed = cdiv(worst_case_tokens, self.bm.block_size)
            if blocks_needed > self.bm.num_available_gpu_blocks:
                break
            self.waiting.popleft()
            self.bm.grow(seq, worst_case_tokens)  # pre-reserve; never grows again
            seq.status = SeqStatus.RUNNING
            self.running.append(seq)

    # ------------------------------------------------------------- schedule
    def schedule(self) -> StepWork:
        work = StepWork()
        budget = self.cfg.max_num_batched_tokens

        if self.cfg.policy == "static":
            if not self.running:                  # only admit when idle
                self._admit_batch_static()
            self._advance_running(work, budget, allow_preempt=False)
            return work

        # continuous
        self._resume_swapped(work)
        budget = self._advance_running(work, budget, allow_preempt=True)
        # Only pull in new requests when we're not already shedding load: nothing
        # was preempted/swapped this step and no preempted work is parked. This,
        # plus the watermark, is what keeps the engine out of a thrashing loop.
        under_pressure = work.preempted > 0 or work.swapped_out > 0 or bool(self.swapped)
        if not under_pressure:
            self._admit_waiting(work, budget)
        return work
