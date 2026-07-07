# Lab 06: Benchmark 和可视化表达

## 目标

学会把 mini-vLLM 的行为讲成可验证的系统结论, 而不是只贴几组数字。

## 阅读

- `mini_vllm/metrics.py`
- `benchmarks/run_all.py`
- `benchmarks/_style.py`
- `mini_vllm/trace_viewer.py`

## 实验 1: 跑全量 benchmark

```bash
make bench
```

产物:

- `docs/results.json`
- `docs/assets/mem_capacity.png`
- `docs/assets/throughput.png`
- `docs/assets/latency.png`
- `docs/assets/prefix_cache.png`

## 实验 2: 写 headline conclusion

为每个图写一句结论:

- Memory: paged KV 相比 contiguous reserve 带来什么收益?
- Throughput: continuous batching 的 ceiling 为什么更高?
- Latency: static batching 的 p99 TTFT 为什么恶化?
- Prefix: shared prompt 下 prefill tokens 为什么减少?

## 实验 3: 用 trace viewer 解释一帧

```bash
make trace
```

打开 `docs/trace.html`, 找一个有 preemption 或 prefix hit 的 step。

记录:

- 这个 step 前有哪些 waiting/running/swapped/finished requests?
- scheduler 安排了多少 prefill 和 decode?
- KV blocks 如何变化?
- 有没有 prefix hit、preemption、swap 或 eviction?

## 验收

你应该能把 benchmark 讲成因果链:

```text
机制 -> 改变了什么资源使用 -> 指标如何变化 -> 适用场景是什么
```

示例:

```text
Prefix caching 复用共享 system prompt 的 full KV blocks,
减少 follower requests 的 prefill tokens, 所以 TTFT 下降。
但 decode 仍然逐 token 生成, 所以输出很长时 E2E latency 仍会受 decode 影响。
```

