from mini_vllm import (
    BlockManager,
    CacheConfig,
    LLMEngine,
    ModelConfig,
    Request,
    Scheduler,
    SchedulerConfig,
    SeqStatus,
    Sequence,
)


def test_static_scheduler_does_not_admit_while_batch_is_running():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=100))
    scheduler = Scheduler(
        SchedulerConfig(policy="static", max_num_seqs=1, max_num_batched_tokens=100),
        bm,
    )
    first = Sequence("a", prompt_len=4, max_tokens=8)
    second = Sequence("b", prompt_len=4, max_tokens=8)
    scheduler.add(first)
    scheduler.add(second)

    work = scheduler.schedule()

    assert first.status == SeqStatus.RUNNING
    assert second.status == SeqStatus.WAITING
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1
    assert work.num_prefill_tokens == 4

    scheduler.schedule()

    assert second.status == SeqStatus.WAITING
    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1


def test_continuous_scheduler_mixes_decode_and_new_prefill():
    engine = LLMEngine(
        CacheConfig(block_size=4, num_gpu_blocks=100),
        SchedulerConfig(policy="continuous", max_num_seqs=2, max_num_batched_tokens=100),
        ModelConfig(),
    )
    engine.add_request(Request("a", prompt_len=4, max_tokens=3))
    engine._release_arrivals()
    engine.step()

    engine.add_request(Request("b", prompt_len=4, max_tokens=3, arrival=engine.clock_ms))
    engine._release_arrivals()
    work = engine.step()

    assert [seq.request_id for seq in work.decode] == ["a"]
    assert [(seq.request_id, tokens) for seq, tokens in work.prefill] == [("b", 4)]


def test_watermark_limits_new_admissions():
    bm = BlockManager(CacheConfig(block_size=4, num_gpu_blocks=10))
    scheduler = Scheduler(
        SchedulerConfig(
            policy="continuous",
            max_num_seqs=10,
            max_num_batched_tokens=100,
            watermark=0.8,
        ),
        bm,
    )
    scheduler.add(Sequence("a", prompt_len=16, max_tokens=1))
    scheduler.add(Sequence("b", prompt_len=16, max_tokens=1))

    scheduler.schedule()

    assert [seq.request_id for seq in scheduler.running] == ["a"]
    assert [seq.request_id for seq in scheduler.waiting] == ["b"]

