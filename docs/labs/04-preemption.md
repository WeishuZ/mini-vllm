# Lab 04: Preemption 和 admission control

## 目标

理解显存压力下 scheduler 如何通过 recompute 或 swap 让系统继续前进,
以及 watermark 为什么能避免 preemption thrashing。

## 阅读

- `mini_vllm/scheduler.py`
- `mini_vllm/block_manager.py`
- `mini_vllm/config.py`
- `tests/test_engine.py`

重点看:

- `Scheduler._preempt`
- `Scheduler._resume_swapped`
- `Scheduler._admit_waiting`
- `BlockManager.swap_out`
- `BlockManager.swap_in`

## 实验 1: 制造压力

把 GPU block pool 调小:

```python
CacheConfig(block_size=16, num_gpu_blocks=80)
SchedulerConfig(max_num_seqs=32, preemption_mode="recompute")
```

记录:

- `num_preemptions`
- `peak_gpu_util`
- throughput
- TTFT p99

## 实验 2: recompute vs swap

对比:

```python
SchedulerConfig(preemption_mode="recompute")
SchedulerConfig(preemption_mode="swap")
```

注意: 当前实现中 swap path 只在 prefix caching 关闭时启用, 因为每个 block
必须是 private ownership 才能 lossless swap。

## 实验 3: watermark

对比:

```python
SchedulerConfig(watermark=0.00)
SchedulerConfig(watermark=0.04)
SchedulerConfig(watermark=0.10)
```

观察 admission 是否过于激进, preemption 次数是否上升。

## 验收

你应该能解释:

- recompute preemption 丢 KV, 保留已经生成的 tokens。
- swap preemption 把 KV blocks 移到 CPU pool, 但需要额外内存和搬运成本。
- watermark 留出 KV headroom, 避免刚 admit 的请求立刻把 running 请求挤出去。

