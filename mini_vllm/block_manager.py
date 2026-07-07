"""Paged KV-cache block manager.

This is the core of the project and the part that maps most directly onto
operating-system virtual memory:

============================  ===================================
OS concept                    mini-vLLM here
============================  ===================================
physical frame                physical block (fixed ``block_size`` tokens)
virtual page                  logical block (an index into ``block_table``)
page table                    ``Sequence.block_table`` (logical -> physical)
demand paging                 blocks allocated as tokens are computed
internal fragmentation        partially-filled last block
page cache / dedup            prefix caching (shared, hashed full blocks)
cache replacement             LRU eviction of unpinned prefix blocks
copy-on-write                 forking a shared partial block before a write
swapping to disk              ``swap_out`` / ``swap_in`` to the CPU pool
============================  ===================================

Because blocks are fixed-size and mapped through a per-sequence table, there is
**no external fragmentation** — any free block can back any sequence's next
logical block. That is the whole point of PagedAttention, and the memory
benchmark quantifies what it buys you versus reserving contiguous KV per
sequence.
"""
from __future__ import annotations

from collections import OrderedDict
import hashlib
from typing import Dict, List

from .config import CacheConfig
from .request import Sequence


def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(-a // b)


class BlockManager:
    def __init__(self, cfg: CacheConfig):
        self.block_size = cfg.block_size
        self.num_gpu_blocks = cfg.num_gpu_blocks
        self.num_cpu_blocks = cfg.num_cpu_blocks
        self.enable_prefix_caching = cfg.enable_prefix_caching
        self.prefix_cache_max_blocks = (
            cfg.prefix_cache_max_blocks
            if cfg.prefix_cache_max_blocks is not None
            else cfg.num_gpu_blocks
        )
        self.prefix_cache_max_blocks = max(0, self.prefix_cache_max_blocks)

        # Free pools as stacks of physical block ids.
        self._free_gpu: List[int] = list(range(self.num_gpu_blocks))
        self._free_cpu: List[int] = list(range(self.num_cpu_blocks))

        # Reference counts let blocks be shared (prefix cache) and freed safely.
        self.ref_count: Dict[int, int] = {}

        # Prefix cache: hash(prefix tokens) -> physical gpu block id, and back.
        # Idle cached blocks remain resident with ref_count == 0 until the LRU
        # evictor reclaims them for budget or allocation pressure.
        self._hash_to_block: Dict[str, int] = {}
        self._block_to_hash: Dict[int, str] = {}
        self._cache_lru: OrderedDict[str, int] = OrderedDict()

        # request_id -> list of cpu block ids, for swapped-out sequences.
        self._swapped: Dict[str, List[int]] = {}

        # metrics
        self.cache_query_blocks = 0
        self.cache_hit_blocks = 0
        self.prefix_cache_saved_tokens = 0
        self.prefix_cache_evictions = 0
        self.num_cow = 0
        self.peak_gpu_blocks_used = 0

    # ------------------------------------------------------------------ pools
    @property
    def num_free_gpu_blocks(self) -> int:
        return len(self._free_gpu)

    @property
    def num_used_gpu_blocks(self) -> int:
        return self.num_gpu_blocks - len(self._free_gpu)

    @property
    def num_prefix_cache_blocks(self) -> int:
        return len(self._hash_to_block)

    @property
    def num_evictable_prefix_blocks(self) -> int:
        return sum(
            1
            for bid in self._hash_to_block.values()
            if self.ref_count.get(bid, 0) == 0
        )

    @property
    def num_pinned_prefix_blocks(self) -> int:
        return self.num_prefix_cache_blocks - self.num_evictable_prefix_blocks

    @property
    def num_available_gpu_blocks(self) -> int:
        return self.num_free_gpu_blocks + self.num_evictable_prefix_blocks

    @property
    def gpu_utilization(self) -> float:
        return self.num_used_gpu_blocks / self.num_gpu_blocks

    @property
    def num_free_cpu_blocks(self) -> int:
        return len(self._free_cpu)

    def _track_peak(self) -> None:
        self.peak_gpu_blocks_used = max(
            self.peak_gpu_blocks_used, self.num_used_gpu_blocks
        )

    def _alloc_gpu(self) -> int:
        if not self._free_gpu and not self._evict_one_prefix_block():
            raise MemoryError("out of GPU KV blocks")
        bid = self._free_gpu.pop()
        self.ref_count[bid] = 1
        self._track_peak()
        return bid

    def _decref_gpu(self, bid: int) -> None:
        self.ref_count[bid] -= 1
        if self.ref_count[bid] <= 0:
            if bid in self._block_to_hash:
                # Idle prefix-cache blocks stay resident and reusable until LRU
                # eviction or allocation pressure reclaims them.
                self.ref_count[bid] = 0
                self._enforce_prefix_cache_budget()
            else:
                del self.ref_count[bid]
                self._free_gpu.append(bid)

    def _evict_prefix_block(self, h: str) -> bool:
        bid = self._hash_to_block.get(h)
        if bid is None or self.ref_count.get(bid, 0) > 0:
            return False
        del self._hash_to_block[h]
        self._block_to_hash.pop(bid, None)
        self._cache_lru.pop(h, None)
        self.ref_count.pop(bid, None)
        self._free_gpu.append(bid)
        self.prefix_cache_evictions += 1
        return True

    def _evict_one_prefix_block(self) -> bool:
        for h in list(self._cache_lru.keys()):
            if self._evict_prefix_block(h):
                return True
        return False

    def _enforce_prefix_cache_budget(self) -> None:
        while self.num_prefix_cache_blocks > self.prefix_cache_max_blocks:
            if not self._evict_one_prefix_block():
                break

    # --------------------------------------------------------- prefix caching
    def _block_hash(self, token_ids: List[int], end: int) -> str:
        """Content hash of the prefix tokens ``[0, end)`` (vLLM-style: a block's
        identity includes everything before it, so only identical prefixes
        collide)."""
        h = hashlib.blake2b(digest_size=16)
        h.update(end.to_bytes(4, "little"))
        h.update(b",".join(str(t).encode() for t in token_ids[:end]))
        return h.hexdigest()

    def _shareable_prefix_blocks(self, seq: Sequence) -> List[int]:
        """Leading full prompt blocks already present in the cache, as a
        contiguous run from the start. No ref-count side effects."""
        if not self.enable_prefix_caching or seq.token_ids is None:
            return []
        shared: List[int] = []
        num_full = seq.prompt_len // self.block_size
        for b in range(num_full):
            h = self._block_hash(seq.token_ids, (b + 1) * self.block_size)
            bid = self._hash_to_block.get(h)
            if bid is None or bid not in self.ref_count:
                break  # prefix sharing must be contiguous from token 0
            shared.append(bid)
            self._cache_lru.move_to_end(h)
        return shared

    def _register_full_prompt_blocks(self, seq: Sequence) -> None:
        """After computing prefill, publish any now-complete *prompt* blocks to
        the cache so concurrent/later sequences can share them."""
        if not self.enable_prefix_caching or seq.token_ids is None:
            return
        num_cacheable = min(seq.num_computed, seq.prompt_len) // self.block_size
        for b in range(num_cacheable):
            bid = seq.block_table[b]
            if bid in self._block_to_hash:
                continue
            if self.ref_count.get(bid, 0) != 1:
                continue  # only publish blocks we exclusively own
            h = self._block_hash(seq.token_ids, (b + 1) * self.block_size)
            if h not in self._hash_to_block:
                self._hash_to_block[h] = bid
                self._block_to_hash[bid] = h
                self._cache_lru[h] = bid
                self._enforce_prefix_cache_budget()

    # ------------------------------------------------------------ allocation
    def admit_prefix(self, seq: Sequence) -> int:
        """Attach shareable cached prefix blocks to a fresh sequence. Returns the
        number of prompt tokens covered (which prefill can therefore skip)."""
        shared = self._shareable_prefix_blocks(seq)
        num_full = seq.prompt_len // self.block_size
        self.cache_query_blocks += num_full
        for bid in shared:
            self.ref_count[bid] += 1
            seq.block_table.append(bid)
        self.cache_hit_blocks += len(shared)
        covered = len(shared) * self.block_size
        self.prefix_cache_saved_tokens += covered
        seq.num_computed = covered
        seq.num_cached_tokens = covered
        return covered

    def _extra_blocks_to_grow(self, seq: Sequence, num_new: int) -> int:
        """Physical blocks needed to grow KV by ``num_new`` tokens, including a
        possible copy-on-write of a shared partial tail block."""
        cur = seq.num_kv_tokens
        capacity = len(seq.block_table) * self.block_size
        deficit = max(0, (cur + num_new) - capacity)
        extra = cdiv(deficit, self.block_size)
        if (
            seq.block_table
            and cur % self.block_size != 0
            and self.ref_count.get(seq.block_table[-1], 1) > 1
        ):
            extra += 1  # COW the shared, partially-filled tail before writing
        return extra

    def can_grow(self, seq: Sequence, num_new: int) -> bool:
        return self._extra_blocks_to_grow(seq, num_new) <= self.num_available_gpu_blocks

    def grow(self, seq: Sequence, num_new: int) -> None:
        """Reserve blocks so the sequence can hold ``num_new`` more computed
        tokens. Performs copy-on-write if the tail block is shared."""
        cur = seq.num_kv_tokens
        if (
            seq.block_table
            and cur % self.block_size != 0
            and self.ref_count.get(seq.block_table[-1], 1) > 1
        ):
            old = seq.block_table[-1]
            new = self._alloc_gpu()           # private copy
            self._decref_gpu(old)
            seq.block_table[-1] = new
            self.num_cow += 1
        target = cur + num_new
        while len(seq.block_table) * self.block_size < target:
            seq.block_table.append(self._alloc_gpu())

    def free(self, seq: Sequence) -> None:
        for bid in seq.block_table:
            self._decref_gpu(bid)
        seq.block_table = []
        cpu = self._swapped.pop(seq.request_id, None)
        if cpu:
            self._free_cpu.extend(cpu)

    def fork(self, parent: Sequence, child: Sequence) -> None:
        """Share the parent's blocks with a new child sequence (beam search /
        parallel sampling). Subsequent divergent writes trigger COW."""
        child.block_table = list(parent.block_table)
        child.num_computed = parent.num_computed
        child.num_generated = parent.num_generated
        child.token_ids = list(parent.token_ids) if parent.token_ids else None
        for bid in child.block_table:
            self.ref_count[bid] += 1

    # ----------------------------------------------------------------- swap
    def can_swap_in(self, seq: Sequence) -> bool:
        return len(self._swapped.get(seq.request_id, [])) <= self.num_available_gpu_blocks

    def swap_out(self, seq: Sequence) -> None:
        """Page a sequence's blocks GPU -> CPU. Assumes private blocks (the swap
        preemption path is only enabled without prefix caching, where every
        block is exclusively owned)."""
        cpu_ids: List[int] = []
        for bid in seq.block_table:
            self._decref_gpu(bid)
            cpu_ids.append(self._free_cpu.pop())
        self._swapped[seq.request_id] = cpu_ids
        seq.block_table = []
        seq.status = seq.status  # status set by caller

    def swap_in(self, seq: Sequence) -> None:
        cpu_ids = self._swapped.pop(seq.request_id)
        new_table: List[int] = []
        for cid in cpu_ids:
            self._free_cpu.append(cid)
            new_table.append(self._alloc_gpu())
        seq.block_table = new_table
