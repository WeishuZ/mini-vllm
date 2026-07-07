# mini-vLLM 学习路径

这份路径把 mini-vLLM 当作一个 LLM serving control plane 教具来使用。
目标不是只跑通 demo, 而是能解释 vLLM 为什么快, 并能用本项目的实验复现
KV cache 管理、continuous batching、preemption 和 prefix caching 的收益。

## 使用方式

先准备环境:

```bash
make setup
make test
make demo
make bench
make trace
```

如果不想用 `make`, 对应命令是:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m mini_vllm.demo
.venv/bin/python benchmarks/run_all.py
.venv/bin/python -m mini_vllm.trace_viewer --output docs/trace.html
```

## Stage 0: 全局地图

目标: 知道这个项目模拟的是 vLLM 的哪一层。

阅读:

- `README.md`
- `mini_vllm/engine.py`
- `mini_vllm/config.py`
- `mini_vllm/model_runner.py`

练习:

- 跑 `make demo`, 观察每一步的 running、prefill、decode、KV 利用率。
- 跑 `make trace`, 打开 `docs/trace.html`, 用 slider 看队列和 KV blocks。

验收:

- 能解释为什么本项目没有真实 transformer 权重和 GPU kernel。
- 能画出 `Request -> Sequence -> Scheduler -> BlockManager -> ModelRunner -> Metrics`。

## Stage 1: Token accounting 和请求生命周期

目标: 理解一个请求从提交到完成时, token 计数如何变化。

阅读:

- `mini_vllm/request.py`
- `mini_vllm/engine.py`
- `docs/labs/01-token-accounting.md`

验收:

- 能解释 `prompt_len`, `max_tokens`, `num_computed`, `num_generated`。
- 能解释 prefill 和 decode 的区别。
- 能解释 TTFT 和 E2E latency 是如何计算的。

## Stage 2: Paged KV cache

目标: 理解 PagedAttention 的内存管理思想。

阅读:

- `mini_vllm/block_manager.py`
- `mini_vllm/analysis.py`
- `benchmarks/bench_memory.py`
- `docs/labs/02-paged-kv.md`

验收:

- 能解释 physical block、logical block、block table。
- 能解释 paged KV 为什么没有外部碎片。
- 能解释 block size 对内部碎片的影响。

## Stage 3: Continuous batching

目标: 理解 scheduler 如何同时改善吞吐和 TTFT。

阅读:

- `mini_vllm/scheduler.py`
- `benchmarks/bench_throughput.py`
- `benchmarks/bench_latency.py`
- `docs/labs/03-continuous-batching.md`

验收:

- 能解释 static batching 的 head-of-line blocking。
- 能解释 continuous batching 为什么能持续 refill decode batch。
- 能解释 chunked prefill 为什么保护 decode latency。

## Stage 4: Preemption 和 admission control

目标: 理解显存压力下 serving engine 如何避免崩溃和 thrashing。

阅读:

- `mini_vllm/scheduler.py`
- `mini_vllm/block_manager.py`
- `docs/labs/04-preemption.md`

验收:

- 能解释 recompute preemption 和 swap preemption 的差别。
- 能解释 watermark 的作用。
- 能用 benchmark 说明过度 admission 为什么会造成 preemption 风暴。

## Stage 5: Prefix caching

目标: 理解共享 system prompt、RAG、agent workload 为什么能受益。

阅读:

- `mini_vllm/block_manager.py`
- `benchmarks/bench_prefix.py`
- `docs/labs/05-prefix-cache.md`

验收:

- 能解释 prefix cache 为什么必须从 token 0 连续命中。
- 能解释 block hash 为什么包含整个 prefix。
- 能解释 prefix cache 主要节省 prefill, 不是 decode。

## Stage 6: Benchmark 和可视化表达

目标: 从“代码能跑”升级到“结果能讲清楚”。

阅读:

- `mini_vllm/metrics.py`
- `benchmarks/run_all.py`
- `mini_vllm/trace_viewer.py`
- `docs/labs/06-benchmarking.md`

验收:

- 能设计一张 benchmark 表, 包含 throughput、TTFT、E2E、KV util、preemption。
- 能解释每个 benchmark 的因果链。
- 能用 trace viewer 找到一个 step, 说明发生了哪些调度和内存事件。

## Stage 7: 项目增强

目标: 亲手扩展一个真实教学功能。

推荐选题:

- 增加 queue time / ITL / TPOT metrics。
- 给 trace viewer 增加 event log。
- 新增 `bench_block_size.py`, 研究 block size 对 waste 的影响。
- 新增 `bench_watermark.py`, 研究 watermark 对 preemption 的影响。
- 给 prefix cache 增加更细的 eviction 统计。

验收:

- 有代码改动。
- 有测试或 benchmark。
- README 或 docs 中能解释这个改动为什么有意义。

## Stage 8: 映射到真实 vLLM

目标: 把 mini-vLLM 的概念迁移到真实部署。

阅读:

- `docs/vllm-mapping.md`
- `docs/real-vllm-runbook.md`

验收:

- 能启动一个 OpenAI-compatible vLLM server。
- 能用 OpenAI SDK 调用本地服务。
- 能解释 `max_model_len`, `gpu_memory_utilization`, `max_num_batched_tokens`,
  prefix caching, chunked prefill 和 mini-vLLM 里的哪些概念对应。

