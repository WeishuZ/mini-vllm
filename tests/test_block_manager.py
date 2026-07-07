"""Unit tests for the paged block manager: allocation, growth, COW, swap,
and prefix-cache dedup."""
from mini_vllm import BlockManager, CacheConfig, Sequence


def seq(rid, prompt, gen=8, tokens=None):
    return Sequence(rid, prompt_len=prompt, max_tokens=gen, token_ids=tokens)


def test_alloc_and_free_no_leak():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=10))
    s = seq("a", 0)
    assert bm.can_grow(s, 10)
    bm.grow(s, 10)
    s.num_computed = 10
    assert len(s.block_table) == 3          # ceil(10/4)
    assert bm.num_free_gpu_blocks == 7
    bm.free(s)
    assert bm.num_free_gpu_blocks == 10
    assert s.block_table == []


def test_incremental_paged_growth():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=10))
    s = seq("a", 0)
    bm.grow(s, 3); s.num_computed = 3
    assert len(s.block_table) == 1
    bm.grow(s, 1); s.num_computed = 4       # fills the block exactly
    assert len(s.block_table) == 1
    bm.grow(s, 1); s.num_computed = 5       # spills into a new block
    assert len(s.block_table) == 2


def test_out_of_memory_is_reported():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=2))
    s = seq("a", 0)
    assert bm.can_grow(s, 8) is True        # exactly 2 blocks
    bm.grow(s, 8); s.num_computed = 8
    assert bm.num_free_gpu_blocks == 0
    s2 = seq("b", 0)
    assert bm.can_grow(s2, 1) is False


def test_copy_on_write_on_fork():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=10))
    parent = seq("p", 0)
    bm.grow(parent, 6); parent.num_computed = 6      # 2 blocks, tail partially full
    child = seq("c", 0)
    bm.fork(parent, child)
    assert child.block_table == parent.block_table   # shared
    assert bm.num_free_gpu_blocks == 8               # still only 2 distinct blocks

    cow_before = bm.num_cow
    bm.grow(child, 1); child.num_computed = 7         # write into shared partial tail
    assert bm.num_cow == cow_before + 1
    assert child.block_table[-1] != parent.block_table[-1]   # tail copied
    assert child.block_table[:-1] == parent.block_table[:-1] # prefix still shared


def test_swap_out_in_roundtrip():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=10, num_cpu_blocks=10))
    s = seq("a", 0)
    bm.grow(s, 8); s.num_computed = 8                 # 2 blocks
    bm.swap_out(s)
    assert s.block_table == []
    assert bm.num_free_gpu_blocks == 10              # GPU memory reclaimed
    assert bm.num_free_cpu_blocks == 8               # parked on CPU
    assert bm.can_swap_in(s)
    bm.swap_in(s)
    assert len(s.block_table) == 2
    assert bm.num_free_gpu_blocks == 8
    assert bm.num_free_cpu_blocks == 10


def test_prefix_cache_dedup():
    cfg = CacheConfig(block_size=4, num_gpu_blocks=50, enable_prefix_caching=True)
    bm = BlockManager(cfg)
    toks = list(range(1, 13))                         # 12 tokens -> 3 full blocks
    s1 = seq("a", 12, tokens=toks)
    assert bm.admit_prefix(s1) == 0                   # cold cache
    bm.grow(s1, 12); s1.num_computed = 12
    bm._register_full_prompt_blocks(s1)

    s2 = seq("b", 12, tokens=toks)
    covered = bm.admit_prefix(s2)
    assert covered == 12                              # all 3 blocks shared
    assert bm.cache_hit_blocks >= 3
    # the shared blocks are the very same physical blocks
    assert s2.block_table == s1.block_table[:3]


def test_prefix_cache_partial_match():
    cfg = CacheConfig(block_size=4, num_gpu_blocks=50, enable_prefix_caching=True)
    bm = BlockManager(cfg)
    base = list(range(1, 13))
    s1 = seq("a", 12, tokens=base)
    bm.admit_prefix(s1); bm.grow(s1, 12); s1.num_computed = 12
    bm._register_full_prompt_blocks(s1)

    # shares first two blocks (tokens 1..8), diverges at the third
    other = list(range(1, 9)) + [99, 99, 99, 99]
    s2 = seq("b", 12, tokens=other)
    covered = bm.admit_prefix(s2)
    assert covered == 8                               # 2 full shared blocks


def test_prefix_cache_survives_sequence_free():
    cfg = CacheConfig(block_size=4, num_gpu_blocks=10, enable_prefix_caching=True)
    bm = BlockManager(cfg)
    toks = list(range(1, 9))                          # 2 full blocks
    s1 = seq("a", 8, tokens=toks)
    bm.admit_prefix(s1)
    bm.grow(s1, 8); s1.num_computed = 8
    bm._register_full_prompt_blocks(s1)

    bm.free(s1)
    assert bm.num_prefix_cache_blocks == 2
    assert bm.num_evictable_prefix_blocks == 2
    assert bm.num_free_gpu_blocks == 8                # cached blocks stay resident

    s2 = seq("b", 8, tokens=toks)
    assert bm.admit_prefix(s2) == 8
    assert bm.prefix_cache_saved_tokens == 8
    assert bm.num_pinned_prefix_blocks == 2


def test_prefix_cache_lru_budget_eviction():
    cfg = CacheConfig(
        block_size=4,
        num_gpu_blocks=10,
        enable_prefix_caching=True,
        prefix_cache_max_blocks=2,
    )
    bm = BlockManager(cfg)
    a = list(range(1, 9))
    b = list(range(101, 109))

    s1 = seq("a", 8, tokens=a)
    bm.admit_prefix(s1)
    bm.grow(s1, 8); s1.num_computed = 8
    bm._register_full_prompt_blocks(s1)
    bm.free(s1)
    assert bm.num_prefix_cache_blocks == 2

    s2 = seq("b", 8, tokens=b)
    bm.admit_prefix(s2)
    bm.grow(s2, 8); s2.num_computed = 8
    bm._register_full_prompt_blocks(s2)
    bm.free(s2)

    assert bm.num_prefix_cache_blocks == 2
    assert bm.prefix_cache_evictions == 2
    assert bm.admit_prefix(seq("a2", 8, tokens=a)) == 0
    assert bm.admit_prefix(seq("b2", 8, tokens=b)) == 8


def test_allocation_pressure_evicts_idle_prefix_cache():
    cfg = CacheConfig(block_size=4, num_gpu_blocks=2, enable_prefix_caching=True)
    bm = BlockManager(cfg)
    toks = list(range(1, 9))
    s1 = seq("a", 8, tokens=toks)
    bm.admit_prefix(s1)
    bm.grow(s1, 8); s1.num_computed = 8
    bm._register_full_prompt_blocks(s1)
    bm.free(s1)

    assert bm.num_free_gpu_blocks == 0
    assert bm.num_available_gpu_blocks == 2

    s2 = seq("b", 0)
    assert bm.can_grow(s2, 8)
    bm.grow(s2, 8); s2.num_computed = 8
    assert bm.num_prefix_cache_blocks == 0
    assert bm.prefix_cache_evictions == 2
    assert len(s2.block_table) == 2
