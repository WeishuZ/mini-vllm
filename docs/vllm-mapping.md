# mini-vLLM 到真实 vLLM 的概念映射

mini-vLLM 不是 vLLM 的源码缩小版, 而是一个教学模型: 它保留 serving
control plane 中最关键的机制, 把真实 GPU kernel 和模型权重替换成确定性的
latency model。这样学习者可以在 laptop 上观察调度和内存管理行为。

## 总体映射

| mini-vLLM | 真实 vLLM 概念 | 说明 |
|---|---|---|
| `LLMEngine` | engine / engine core | 驱动 serving loop, 接收请求, 调 scheduler, 收集 metrics。 |
| `Scheduler` | continuous batching scheduler | 决定每一步哪些 sequence 做 prefill, 哪些 sequence decode, 是否 admit/preempt。 |
| `BlockManager` | KV cache manager / block table | 管理固定大小 KV blocks, prefix cache, ref count, swap/recompute。 |
| `Sequence` | sequence / request state | 保存 token accounting, block table, status, timing metrics。 |
| `ModelRunner` | model runner / GPU worker forward pass | 真实 vLLM 会跑 transformer forward; mini-vLLM 用 latency function 模拟。 |
| `EngineMetrics` | production metrics / benchmark result | 汇总 throughput、TTFT、E2E、KV utilization、cache hit 等指标。 |
| `workloads` | synthetic traffic / benchmark dataset | 产生 burst、Poisson、shared prefix 请求。 |
| `trace_viewer` | observability / debugging aid | 用可视化方式展示每个 scheduler step 的状态变化。 |
| `memory_estimator` | deployment sizing / capacity planning | 把模型形状和 GPU GB 粗略换算成 KV block 数。 |

## Token lifecycle

mini-vLLM 中:

- `prompt_len` 是请求已有输入 tokens。
- `max_tokens` 是最多生成 tokens。
- `num_computed` 是已经 materialized KV 的 tokens。
- `num_generated` 是已经 decode 出来的 output tokens。
- `prefill_remaining = prompt_len + num_generated - num_computed`。

真实 vLLM 中也有同样的基本状态: 已存在 tokens、已计算 tokens、等待 prefill
的 tokens、正在 decode 的 sequences。具体类名和实现细节会随版本变化, 但心智模型相同。

## Paged KV / PagedAttention

mini-vLLM 中:

- `CacheConfig.block_size` 控制一个 physical block 容纳多少 tokens。
- `Sequence.block_table` 保存 logical block 到 physical block 的映射。
- `BlockManager.grow` 按需分配 blocks。
- `BlockManager.free` 在请求完成后释放或保留 prefix cache blocks。

真实 vLLM 中:

- KV cache 被切成 blocks。
- 每个 sequence 通过 block table 找到自己的 KV blocks。
- scheduler 在 token budget 和 block budget 下安排 prefill/decode。
- PagedAttention kernel 使用 block table 读取非连续物理 KV blocks。

mini-vLLM 没有实现 attention kernel, 但实现了 kernel 依赖的内存控制面。

## Continuous batching

mini-vLLM 中:

- `policy="continuous"` 每个 step 先推进 running sequences, 再用剩余 budget admit waiting requests。
- `policy="static"` 是 baseline: batch drain 完再 admit 新 batch。
- `max_num_batched_tokens` 是每个 step 的 token budget。
- `enable_chunked_prefill` 允许长 prompt 分多步 prefill。

真实 vLLM 中:

- online serving 依靠 continuous batching 持续把请求填入 batch。
- chunked prefill 用于在长 prefill 和 decode latency 之间做权衡。
- `max_num_batched_tokens` 是真实服务里非常重要的调优旋钮。

## Preemption

mini-vLLM 中:

- `preemption_mode="recompute"` 释放 KV, 之后重新 prefill。
- `preemption_mode="swap"` 把 KV blocks 换到 CPU pool。
- `watermark` 给 running sequences 留 KV headroom, 防止 admission 过猛。

真实 vLLM 中:

- 当 KV cache 压力过大时, scheduler 也需要 preempt 或延迟 admission。
- recompute 和 swap 都是用更多计算或更多搬运换取显存压力缓解。
- 生产调优中, 过高并发、过长 context、过小 KV cache 都会触发类似问题。

## Prefix caching

mini-vLLM 中:

- `enable_prefix_caching=True` 打开 prefix cache。
- `BlockManager._block_hash` 对完整 prefix 内容做 hash。
- `admit_prefix` 只接受从 token 0 开始的连续命中。
- idle cached blocks 通过 LRU eviction 回收。

真实 vLLM 中:

- Automatic prefix caching 可以复用相同前缀的 KV blocks。
- 对 system prompt、RAG context、多轮对话、agent tool schema 等 workload 有帮助。
- 它主要减少 prefill 工作, 对 decode 阶段帮助有限。

## mini-vLLM 没有建模的内容

mini-vLLM 为了教学刻意省略:

- 真实 transformer weights。
- CUDA/HIP kernels。
- tokenizer 和 chat template。
- sampling、logits processor、structured outputs。
- tensor parallel / pipeline parallel。
- quantization。
- multi-node serving。
- OpenAI-compatible HTTP server。

这些属于真实部署阶段, 可以在理解 control plane 后再学。

## 对照学习建议

学习真实 vLLM 时, 可以按这个顺序做映射:

1. 用 mini-vLLM 理解 token accounting。
2. 用 mini-vLLM 解释 paged KV cache 为什么省显存。
3. 用 mini-vLLM benchmark 理解 continuous batching 的吞吐和 TTFT。
4. 用 mini-vLLM trace viewer 看 preemption 和 prefix cache。
5. 再去真实 vLLM 中找对应参数和 metrics。

这样读真实文档时, 不会只记命令, 而是能看出每个参数背后的系统 tradeoff。
