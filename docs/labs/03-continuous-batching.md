# Lab 03: Continuous batching

## 目标

理解 continuous batching 如何避免 static batching 的 head-of-line blocking,
并保持 decode batch 更饱满。

## 阅读

- `mini_vllm/scheduler.py`
- `benchmarks/bench_throughput.py`
- `benchmarks/bench_latency.py`
- `tests/test_engine.py`

重点看:

- `Scheduler.schedule`
- `Scheduler._advance_running`
- `Scheduler._admit_waiting`
- `Scheduler._admit_batch_static`

## 实验 1: policy 对比

在同一 workload 下分别运行:

```python
SchedulerConfig(policy="static", max_num_seqs=128, max_num_batched_tokens=2048)
SchedulerConfig(policy="continuous", max_num_seqs=128, max_num_batched_tokens=2048)
```

记录:

- throughput
- TTFT p50/p99
- E2E p50/p99

## 实验 2: chunked prefill

对比:

```python
SchedulerConfig(enable_chunked_prefill=True)
SchedulerConfig(enable_chunked_prefill=False)
```

使用长 prompt workload, 观察 decode 是否被长 prefill 阻塞。

## 观察问题

- static batching 为什么要等整个 batch drain 完才 admit 新请求?
- continuous batching 为什么能在同一步混合 prefill 和 decode?
- `max_num_batched_tokens` 是如何在 prefill 和 decode 之间形成 tradeoff 的?

## 验收

你应该能解释:

- static batching 的尾延迟来自 head-of-line blocking。
- continuous batching 的吞吐收益来自更稳定的 decode batch occupancy。
- chunked prefill 防止长 prompt 独占整个 step budget。

