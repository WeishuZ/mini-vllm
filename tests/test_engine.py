"""End-to-end engine tests: correctness, preemption, swap, prefix caching, and
the continuous-vs-static throughput story."""
from mini_vllm import (
    CacheConfig,
    LLMEngine,
    ModelConfig,
    Request,
    SchedulerConfig,
    workloads,
)


def build(policy="continuous", n=40, blocks=300, seed=1, **sched):
    cache = CacheConfig(block_size=16, num_gpu_blocks=blocks)
    scfg = SchedulerConfig(policy=policy, max_num_seqs=32,
                           max_num_batched_tokens=2048, **sched)
    e = LLMEngine(cache, scfg, ModelConfig())
    e.add_requests(workloads.burst(n=n, prompt_mean=128, gen_mean=96, seed=seed))
    return e


def test_all_requests_complete_and_token_exact():
    for policy in ("continuous", "static"):
        e = build(policy=policy)
        m = e.run()
        assert m.num_completed == 40, policy
        assert m.total_generated_tokens == sum(s.max_tokens for s in e.completed)
        for s in e.completed:
            assert s.num_generated == s.max_tokens
            assert s.first_token_time is not None
            assert s.finish_time >= s.first_token_time


def test_continuous_beats_static_throughput():
    mc = build(policy="continuous").run()
    ms = build(policy="static").run()
    assert mc.throughput_tok_s > ms.throughput_tok_s
    # continuous also admits sooner -> lower tail TTFT
    assert mc.ttft_ms_p99 <= ms.ttft_ms_p99


def test_recompute_preemption_completes_under_pressure():
    cache = CacheConfig(block_size=16, num_gpu_blocks=80)
    e = LLMEngine(cache, SchedulerConfig(preemption_mode="recompute",
                                         max_num_seqs=32, max_num_batched_tokens=2048),
                  ModelConfig())
    e.add_requests(workloads.burst(n=30, prompt_mean=128, gen_mean=64, seed=5))
    m = e.run()
    assert m.num_completed == 30
    assert m.num_preemptions > 0          # memory pressure forced preemption
    assert m.num_swaps == 0


def test_swap_preemption_completes():
    cache = CacheConfig(block_size=16, num_gpu_blocks=80, num_cpu_blocks=400,
                        enable_prefix_caching=False)
    e = LLMEngine(cache, SchedulerConfig(preemption_mode="swap",
                                         max_num_seqs=32, max_num_batched_tokens=2048),
                  ModelConfig())
    e.add_requests(workloads.burst(n=30, prompt_mean=128, gen_mean=64, seed=5))
    m = e.run()
    assert m.num_completed == 30
    assert m.num_swaps > 0


def test_prefix_cache_hits_when_warmed():
    cache = CacheConfig(block_size=16, num_gpu_blocks=200, enable_prefix_caching=True)
    e = LLMEngine(cache, SchedulerConfig(max_num_seqs=8, max_num_batched_tokens=4096),
                  ModelConfig())
    system = list(range(1, 257))          # 256 tokens == 16 full blocks
    e.add_request(Request("A", prompt_len=256, max_tokens=300, token_ids=system))
    for _ in range(5):                    # warm: A prefills and registers blocks
        e._release_arrivals(); e.step()
    assert e.block_manager.cache_hit_blocks == 0

    e.add_request(Request("B", prompt_len=264, max_tokens=50,
                          arrival=e.clock_ms, token_ids=system + [9_999] * 8))
    m = e.run()
    assert e.block_manager.cache_hit_blocks >= 16     # B shared the 16 system blocks
    assert m.num_completed == 2


def test_prefix_cache_lowers_prefill_work():
    # Staggered arrivals so a leader warms the cache before followers admit.
    system = list(range(1, 257))
    reqs = [
        Request(f"r{i}", prompt_len=264, max_tokens=120,
                arrival=i * 30.0, token_ids=system + [10_000 + i] * 8)
        for i in range(12)
    ]

    def run(pc):
        cache = CacheConfig(block_size=16, num_gpu_blocks=400,
                            enable_prefix_caching=pc)
        e = LLMEngine(cache, SchedulerConfig(max_num_seqs=8,
                                             max_num_batched_tokens=2048),
                      ModelConfig())
        for r in reqs:
            e.add_request(Request(r.request_id, r.prompt_len, r.max_tokens,
                                  r.arrival, list(r.token_ids)))
        return e, e.run()

    e_off, m_off = run(False)
    e_on, m_on = run(True)
    assert e_on.total_prefill_tokens < e_off.total_prefill_tokens
    assert m_on.prefix_cache_hit_rate > 0
    assert m_on.num_completed == m_off.num_completed == 12
