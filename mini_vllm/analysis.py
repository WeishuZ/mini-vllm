"""Static memory-capacity analysis: how many sequences fit under contiguous
reservation vs paged allocation, and how much KV is wasted.

This is the quantitative core of the PagedAttention argument. A contiguous
allocator must reserve ``max_seq_len`` slots per sequence up front (it cannot
know the final length and needs one contiguous run), so every sequence wastes
``max_seq_len - actual_len`` slots. A paged allocator reserves
``ceil(actual_len / block_size)`` blocks on demand, wasting at most one
partially-filled block per sequence (internal fragmentation only).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .block_manager import cdiv


@dataclass
class CapacityResult:
    fit: int                 # sequences that fit
    used_slots: int          # token-slots actually holding KV
    reserved_slots: int      # token-slots reserved (incl. waste)
    wasted_slots: int        # reserved but unused
    waste_fraction: float    # wasted / reserved

    @property
    def utilization(self) -> float:
        return 0.0 if self.reserved_slots == 0 else self.used_slots / self.reserved_slots


def contiguous_capacity(total_slots: int, max_seq_len: int, seq_lens: List[int]) -> CapacityResult:
    """Reserve ``max_seq_len`` per sequence (no paging)."""
    fit = 0
    used = 0
    reserved = 0
    for L in seq_lens:
        if reserved + max_seq_len > total_slots:
            break
        reserved += max_seq_len
        used += min(L, max_seq_len)
        fit += 1
    wasted = reserved - used
    return CapacityResult(fit, used, reserved, wasted,
                          0.0 if reserved == 0 else wasted / reserved)


def paged_capacity(num_blocks: int, block_size: int, seq_lens: List[int]) -> CapacityResult:
    """Reserve ``ceil(L / block_size)`` blocks per sequence (on demand)."""
    total_slots = num_blocks * block_size
    blocks_left = num_blocks
    fit = 0
    used = 0
    reserved = 0
    for L in seq_lens:
        need = cdiv(L, block_size)
        if need > blocks_left:
            break
        blocks_left -= need
        reserved += need * block_size
        used += L
        fit += 1
    wasted = reserved - used
    return CapacityResult(fit, used, reserved, wasted,
                          0.0 if reserved == 0 else wasted / reserved)
