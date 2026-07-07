# Web Presentation Outline

> **主题**：`blueprint` — 工程蓝图风，适合系统架构、状态表、队列流和 block grid。
> **脚本状态**：approved by user in thread
> **章节数**：8 章 / 61 steps
> **核心观众**：能读 Python 源码，希望从 miniVLLM 过渡到真实 vLLM 开发的工程学习者。

---

## 1. inference-physics — LLM 推理的物理形态（7 steps）

**对应脚本段**：
> 一个 decoder-only LLM 生成文本时，是 autoregressive 的。prefill 和 decode 都叫 forward，但系统性质完全不同。

**信息池**：
- 机制：decoder-only autoregressive generation，一步只生成一个新 token —— `script.md`
- 公式：KV bytes per token = `2 * num_layers * num_kv_heads * head_dim * dtype_bytes` —— `script.md`
- 模型：prefill 更偏 compute-bound，decode 更偏 memory-bandwidth-bound —— `mini_vllm/model_runner.py`
- 源码：`ModelRunner.step_latency_ms(work)` 把两类成本压成 deterministic latency model —— `mini_vllm/model_runner.py`

**Technical Deep Dive 补充**：
- 硬问题：为什么 prefill 和 decode 不能用同一种调度直觉？
- 源码入口：`mini_vllm/model_runner.py::ModelRunner.step_latency_ms`，`mini_vllm/config.py::ModelConfig`
- 核心状态：`num_prefill_tokens`，`num_decode_seqs`，`prefill_ms_per_token`，`decode_ms_base`，`decode_ms_per_seq`
- 不变量：decode 阶段每个 active sequence 每 step 最多生成一个 token；prefill 可一次处理多个 prompt tokens。
- 机制链：autoregressive 约束 -> decode 单步化 -> KV read 成为瓶颈 -> serving scheduler 必须混合 prefill/decode。
- 生产映射：真实 vLLM 的 model runner、attention backend、CUDA kernels、chunked prefill policy。

**开发计划**：
- step 1 — 舞台中央显示 autoregressive token timeline，只有最后一个 token 被高亮为“next token”。
- step 2 — timeline 分裂为 prefill 大块和 decode 单 token 流，两个阶段并排对照。
- step 3 — prefill 面板出现“大矩阵 / many tokens / compute-bound”，decode 面板出现“1 token / KV reads / memory-bound”。
- step 4 — KV memory formula 固定在右侧，K/V 两条带状内存随 token 数增长。
- step 5 — `ModelRunner.step_latency_ms` 的简化公式出现，prefill 和 decode 成本项分别点亮。
- step 6 — 画面叠出核心矛盾：prefill wants large chunks，decode wants wide batches。
- step 7 — 右下角出现真实 vLLM 映射：model runner -> attention backend -> scheduler policy。

**素材**：
- ✓ 源码：`mini_vllm/model_runner.py`
- ✓ 配置：`mini_vllm/config.py`
- ⚠️ 无真实 GPU trace；用 blueprint 风格的抽象 timeline 和 memory strips。

---

## 2. engine-step — `LLMEngine.step()` 的提交边界（7 steps）

**对应脚本段**：
> `LLMEngine.step()` 是一次调度、执行、记账和回收。`StepWork` 是控制面和执行面的分界线。

**信息池**：
- 源码：`LLMEngine.step()` 调用 `scheduler.schedule()`，再提交 prefill/decode 状态 —— `mini_vllm/engine.py`
- 边界对象：`StepWork.prefill` 和 `StepWork.decode` —— `mini_vllm/scheduler.py`
- 状态提交：prefill 增加 `num_computed`；decode 增加 `num_generated` 和 `num_computed` —— `mini_vllm/engine.py`
- 观测：每步写入 `StepStat`，包括 `running`、`decode_seqs`、`prefill_tokens`、`gpu_util` —— `mini_vllm/metrics.py`

**Technical Deep Dive 补充**：
- 硬问题：为什么 scheduler 只产出计划，而不是直接修改 token 和时钟？
- 源码入口：`mini_vllm/engine.py::LLMEngine.step`，`mini_vllm/scheduler.py::StepWork`
- 核心状态：`work.prefill`，`work.decode`，`clock_ms`，`total_prefill_tokens`，`total_decode_tokens`，`history`
- 不变量：Scheduler 决策先于 token 状态提交；latency 在 token work 确定后计算；finished sequence 在 decode 后回收。
- 机制链：调度计划 -> 状态提交 -> latency 推进 -> finished 回收 -> metrics 落点。
- 生产映射：真实 vLLM engine core 的 scheduler output、worker execution、request state update、metrics/logging。

**开发计划**：
- step 1 — 中央显示 `LLMEngine.step()` 作为最小闭环，周围四个模块暗显。
- step 2 — Scheduler 发出 `StepWork`，画面只显示计划，不更新 Sequence 状态。
- step 3 — prefill commit 动画：`num_computed += chunk`，prefill token counter 增长。
- step 4 — decode commit 动画：`num_generated += 1` 与 `num_computed += 1` 同时增长。
- step 5 — ModelRunner latency bar 按 prefill/decode 混合长度增长，`clock_ms` 前进。
- step 6 — finished sequence 被 drop，block 回收，completed list 增长。
- step 7 — `StepStat` 行写入 history，四层状态表合并成一次 step trace。

**素材**：
- ✓ 源码：`mini_vllm/engine.py`、`mini_vllm/scheduler.py`、`mini_vllm/metrics.py`
- ⚠️ 无真实 trace screenshot；用自绘 step trace table。

---

## 3. sequence-accounting — Sequence 的 token/KV 不变量（7 steps）

**对应脚本段**：
> Sequence 不是 request 的简单包装。`length = prompt_len + num_generated`，KV 状态由 `num_computed` 决定。

**信息池**：
- 源码：`Request` 和 `Sequence` 的字段定义 —— `mini_vllm/request.py`
- 公式：`length = prompt_len + num_generated`
- 公式：`prefill_remaining = length - num_computed`
- 语义：recompute preemption 保留 `num_generated`，清空 `num_computed` 和 `block_table` —— `Sequence.reset_for_recompute`

**Technical Deep Dive 补充**：
- 硬问题：KV cache 为什么由 `num_computed` 驱动，而不是由 sequence length 驱动？
- 源码入口：`mini_vllm/request.py::Sequence`，`Sequence.reset_for_recompute`
- 核心状态：`prompt_len`，`num_generated`，`num_computed`，`prefill_remaining`，`block_table`
- 不变量：`num_computed <= prompt_len + num_generated`；decode 前必须 `num_computed == length`；decode 后 KV 要跟上新的 token。
- 机制链：autoregressive context -> token exists != KV exists -> prefill debt -> decode readiness -> recompute cost。
- 生产映射：真实 vLLM request/sequence state、computed tokens、context materialization、preemption recompute。

**开发计划**：
- step 1 — 展示 Request 变 Sequence，分成逻辑 token、KV materialization、生命周期三层。
- step 2 — `prompt_len` 与 `num_generated` 组成 `length`，token tape 上 prompt 和 generated 使用不同纹理。
- step 3 — `num_computed` 覆盖 token tape 的前缀，未覆盖区标成 prefill debt。
- step 4 — `prefill_remaining = length - num_computed` 公式出现，欠 KV 的区段被高亮。
- step 5 — decode gate 显示：只有 `num_computed == length` 才能生成 next token。
- step 6 — decode 后 `num_generated` 和 `num_computed` 同步增加，不变量面板保持绿色。
- step 7 — recompute preemption 切走 KV：`num_generated` 保留，`num_computed` 清零，prefill debt 膨胀。

**素材**：
- ✓ 源码：`mini_vllm/request.py`
- ⚠️ 无真实 token ids；用抽象 token tape。

---

## 4. paged-kv — Paged KV 与 BlockManager（8 steps）

**对应脚本段**：
> PagedAttention 不是让 attention 数学变了，而是让 KV cache 从 contiguous reservation 变成 paged allocation。

**信息池**：
- 源码：`BlockManager.grow`，`can_grow`，`free` —— `mini_vllm/block_manager.py`
- 分析：contiguous reserve `max_seq_len`，paged reserve `ceil(L / block_size)` —— `mini_vllm/analysis.py`
- 默认：`block_size=16` —— `mini_vllm/config.py`
- 概念：logical block -> physical block id via `Sequence.block_table`
- 机制：COW 条件是 shared partial tail —— `BlockManager.grow`

**Technical Deep Dive 补充**：
- 硬问题：PagedAttention 为什么本质上是 KV cache 内存管理？
- 源码入口：`mini_vllm/block_manager.py::BlockManager`，`mini_vllm/analysis.py`
- 核心状态：`_free_gpu`，`ref_count`，`Sequence.block_table`，`block_size`，`num_available_gpu_blocks`
- 不变量：logical sequence 连续，physical blocks 可不连续；block free 必须遵守 ref count；shared partial tail 写入前必须 COW。
- 机制链：KV 动态增长 -> contiguous over-reservation -> paged allocation -> partial block waste -> sharing/COW complexity。
- 生产映射：真实 vLLM KV cache manager、block tables、PagedAttention block mapping、KV block profiling。

**开发计划**：
- step 1 — 左侧 contiguous 长条 reservation，右侧 paged block grid；同一请求实际只占 5 个 blocks。
- step 2 — `capacity = len(block_table) * block_size` 公式出现，block table 映射到 scattered physical blocks。
- step 3 — `grow(seq, num_new)` 分支图出现：target 是否超过 capacity。
- step 4 — contiguous vs paged 容量面板显示 `max_seq_len` 支配 vs actual length 支配。
- step 5 — partial tail waste 用最后一个半填 block 表示，external fragmentation 被划掉。
- step 6 — ref count 层出现，两个 sequence 指向同一 full prefix blocks。
- step 7 — shared partial tail 被写入前触发 COW，child 获得私有 block。
- step 8 — 小结面板：收益是内存利用率，代价是 block table、ref count、COW 状态复杂度。

**素材**：
- ✓ 源码：`mini_vllm/block_manager.py`、`mini_vllm/analysis.py`
- ✓ Benchmark 结果可来自 `docs/results.json` 和 `docs/assets/mem_capacity.png`
- ⚠️ 不直接嵌 PNG；优先用自绘 block grid。

---

## 5. scheduler — Continuous Batching 与 token budget（8 steps）

**对应脚本段**：
> Scheduler 的问题不是下一条请求是谁，而是这一轮 token budget 怎么分给 prefill 和 decode。

**信息池**：
- 源码：`Scheduler.schedule`，`_resume_swapped`，`_advance_running`，`_admit_waiting` —— `mini_vllm/scheduler.py`
- 配置：`max_num_seqs`，`max_num_batched_tokens`，`enable_chunked_prefill`，`policy` —— `mini_vllm/config.py`
- 对照：static batching 只有 running 为空时 admit，且预留 worst-case KV —— `Scheduler._admit_batch_static`
- Benchmark：continuous batching 在 README 中展示更高 saturated throughput 和更低 TTFT —— `README.md`

**Technical Deep Dive 补充**：
- 硬问题：continuous batching 为什么比 static batching 更适合 online LLM serving？
- 源码入口：`mini_vllm/scheduler.py::Scheduler.schedule`，`Scheduler._advance_running`，`Scheduler._admit_waiting`
- 核心状态：`waiting`，`running`，`swapped`，`budget`，`work.prefill`，`work.decode`
- 不变量：running 优先于 waiting；decode 每 sequence 一 token；prefill 受 token budget 和 KV capacity 双重约束。
- 机制链：variable-length generation -> static batch decay/head-of-line blocking -> continuous refill -> better throughput/TTFT -> prefill/decode tradeoff。
- 生产映射：真实 vLLM scheduler、chunked prefill、`max_num_batched_tokens`、request admission。

**开发计划**：
- step 1 — 展示 Scheduler 的两个预算：compute token budget 和 KV block budget。
- step 2 — continuous policy 三段顺序出现：resume swapped -> advance running -> admit waiting。
- step 3 — running lane 中 prefill chunk 和 decode token 混合进入 `StepWork`。
- step 4 — token budget 仪表显示 prefill 吃大块、decode 按 sequence 数扣一格。
- step 5 — chunked prefill 把长 prompt 切成多个 step，decode lane 得以继续点亮。
- step 6 — static batching 对照：batch 必须 drain，waiting 请求被挡在门外。
- step 7 — batch decay 动画：短 sequence 结束后 static batch 变窄，continuous refills 空位。
- step 8 — 参数映射面板：`max_num_batched_tokens` 影响 TTFT、ITL、throughput 和混合形态。

**素材**：
- ✓ 源码：`mini_vllm/scheduler.py`、`mini_vllm/config.py`
- ✓ Benchmark 结果可来自 `docs/assets/throughput.png`、`docs/assets/latency.png`
- ⚠️ 优先自绘队列和 budget 仪表，而不是静态图。

---

## 6. memory-pressure — Preemption、Swap 与 Watermark（8 steps）

**对应脚本段**：
> block pool 是有限的。即使 admission 时放得下，running sequence 后面也可能放不下。

**信息池**：
- 源码：`Scheduler._preempt`，`BlockManager.swap_out`，`BlockManager.swap_in` —— `mini_vllm/scheduler.py`、`mini_vllm/block_manager.py`
- 配置：`preemption_mode="recompute" | "swap"`，`watermark` —— `mini_vllm/config.py`
- 语义：recompute 释放 KV 但增加未来 prefill；swap 转移到 CPU block pool —— `scheduler.py` 注释
- 限制：swap 只在 prefix caching 关闭时启用，因为共享 block 语义复杂 —— `Scheduler.__init__`

**Technical Deep Dive 补充**：
- 硬问题：KV pressure 下系统为什么需要 admission control，而不是把 block 塞满？
- 源码入口：`Scheduler._advance_running`，`Scheduler._preempt`，`Scheduler._admit_waiting`，`BlockManager.swap_out/swap_in`
- 核心状态：`num_available_gpu_blocks`，`watermark_blocks`，`preempted_seq_ids`，`swapped`，`num_preemptions`，`num_swaps`
- 不变量：preempt 必须释放 GPU KV；recompute 后 generated tokens 保留；swap path 只用于私有 block 语义。
- 机制链：running KV grows -> can_grow false -> preempt/swap -> immediate pressure relief -> future recompute/swap cost -> watermark avoids thrashing。
- 生产映射：真实 vLLM preemption、recompute/swap policy、KV watermark、overload control。

**开发计划**：
- step 1 — GPU block pool 接近满，running sequence 的 next token 触发 `can_grow=false`。
- step 2 — priority 排序显示 arrival 老请求在前，新请求从队尾被 preempt。
- step 3 — recompute path：block table 清空，sequence 回到 waiting 队头，prefill debt 增加。
- step 4 — swap path：GPU blocks 释放，CPU pool blocks 被占用，sequence 进入 swapped queue。
- step 5 — prefix cache sharing 警示层出现：shared blocks 让 swap 语义变复杂。
- step 6 — watermark gate 出现，available blocks 低于水位线时 admission 停止。
- step 7 — thrashing 对照：无 watermark 的 admit/preempt 循环 vs 有 headroom 的稳定推进。
- step 8 — 机制链收束为 admission control：牺牲瞬时并发换 running set 稳定推进。

**素材**：
- ✓ 源码：`mini_vllm/scheduler.py`、`mini_vllm/block_manager.py`
- ✓ README design notes 中的 thrashing 描述可作为信息池来源 —— `README.md`
- ⚠️ 不模拟 PCIe 时间；视觉只呈现控制面语义。

---

## 7. prefix-cache — Prefix Cache 的 hash、共享与生命周期（8 steps）

**对应脚本段**：
> prefix cache 缓存的是已经 materialized 的 full prompt KV block，不是字符串，也不是输出。

**信息池**：
- 源码：`_block_hash`，`_shareable_prefix_blocks`，`admit_prefix`，`_register_full_prompt_blocks` —— `mini_vllm/block_manager.py`
- 状态：`_hash_to_block`，`_block_to_hash`，`_cache_lru`，`ref_count`
- 限制：只连续命中 leading full prompt blocks；hash 是 `[0, end)` 的完整 prefix —— `block_manager.py`
- 生命周期：pinned vs evictable；cache budget 和 allocation pressure 触发 LRU eviction —— `block_manager.py`

**Technical Deep Dive 补充**：
- 硬问题：prefix cache 为什么只能连续命中，并且为什么省 prefill 不省 decode？
- 源码入口：`BlockManager.admit_prefix`，`BlockManager._block_hash`，`BlockManager._register_full_prompt_blocks`，`BlockManager._evict_one_prefix_block`
- 核心状态：`cache_query_blocks`，`cache_hit_blocks`，`prefix_cache_saved_tokens`，`num_pinned_prefix_blocks`，`num_evictable_prefix_blocks`
- 不变量：cache hit 必须从 token 0 连续；cached block active 时不可 evict；ref count 为 0 的 cached block 仍 resident 直到 eviction。
- 机制链：shared prompt -> repeated prefill waste -> block hash/cache -> shared physical blocks -> lower TTFT/prefill work -> resident cache pressure。
- 生产映射：真实 vLLM automatic prefix caching、block hash、cache eviction、shared KV semantics。

**开发计划**：
- step 1 — shared system prompt 的多请求 workload 出现，重复 prefix 被标成浪费 prefill。
- step 2 — `_register_full_prompt_blocks` 发布 full prompt blocks，hash table 连接到 physical blocks。
- step 3 — `_block_hash(token_ids, end)` 显示 hash 范围 `[0, end)`，不是单个 block。
- step 4 — `admit_prefix` 从 token 0 连续命中，block table 直接 append shared physical block ids。
- step 5 — `num_computed` 跳到 covered tokens，saved prefill counter 增长，decode lane 保持未省。
- step 6 — partial mismatch 展示：第 3 块 miss 后，后面 token 即使相同也不能共享。
- step 7 — lifecycle 展示：active cached block pinned，sequence free 后变 evictable 但仍占 GPU block。
- step 8 — LRU eviction 在 cache budget 或 allocation pressure 下回收 idle cached block。

**素材**：
- ✓ 源码：`mini_vllm/block_manager.py`
- ✓ Benchmark 结果可来自 `benchmarks/bench_prefix.py`、`docs/assets/prefix_cache.png`
- ⚠️ 视觉上用 hash table + block grid，不伪造真实 KV tensor。

---

## 8. observability-to-vllm — Metrics、Trace 与真实 vLLM 映射（8 steps）

**对应脚本段**：
> 真实系统开发不是看能跑，而是看指标能不能定位瓶颈。miniVLLM 的抽象对应真实 vLLM 的 control plane。

**信息池**：
- 源码：`EngineMetrics`，`StepStat` —— `mini_vllm/metrics.py`
- Trace：`build_trace()` 每帧记录 queues、blocks、work、requests —— `mini_vllm/trace_viewer.py`
- 映射：`LLMEngine` -> engine core，`Scheduler` -> request scheduling，`BlockManager` -> KV cache manager，`ModelRunner` -> model execution —— `script.md`
- 指标：throughput、TTFT、E2E、peak KV、preemption、swap、prefix hit、saved prefill —— `mini_vllm/metrics.py`

**Technical Deep Dive 补充**：
- 硬问题：为什么 metrics 必须组合解读，才能定位 serving 瓶颈？
- 源码入口：`mini_vllm/metrics.py::EngineMetrics`，`mini_vllm/trace_viewer.py::build_trace`
- 核心状态：`throughput_tok_s`，`ttft_ms_p99`，`peak_gpu_util`，`prefix_cache_hit_rate`，`prefix_cache_saved_tokens`，queues and block owners
- 不变量：aggregate metrics 必须能追溯到 step-level work；高 throughput 不能掩盖差 TTFT/ITL；cache hit 必须结合 saved tokens。
- 机制链：aggregate metric anomaly -> trace frame -> queue/block/work state -> bottleneck classification -> production debugging.
- 生产映射：真实 vLLM metrics endpoint、Prometheus/Grafana、scheduler logs、engine traces、worker metrics。

**开发计划**：
- step 1 — 指标面板分成用户体验、吞吐、显存压力、缓存收益四列。
- step 2 — TTFT 高的因果树展开：queue time、prefill、prefix miss、recompute preemption。
- step 3 — throughput 高但 streaming 差的对照出现：average tok/s 不能代表 ITL。
- step 4 — trace frame 展开 queues、work、block owners，aggregate metric 被追溯到 step-level。
- step 5 — miniVLLM -> vLLM 映射矩阵出现：engine、scheduler、block manager、model runner、attention backend。
- step 6 — 最终系统剖面图合拢，显示 KV、调度、执行、观测四条主线。
- step 7 — 真实复杂度展示为机制叠加：paging、sharing、preemption、scheduling、execution。
- step 8 — 最终收束为四条主线：KV、调度、执行、观测。

**素材**：
- ✓ 源码：`mini_vllm/metrics.py`、`mini_vllm/trace_viewer.py`
- ✓ 现有 trace artifact：`docs/trace.html`
- ⚠️ 不伪造 Prometheus 截图；用抽象 observability panel。

---

## 素材清单

### 1. inference-physics
- ✓ 源码：`mini_vllm/model_runner.py`、`mini_vllm/config.py`
- ✓ 公式：来自 `script.md`
- ⚠️ GPU kernel trace：缺，用抽象 compute/memory timeline。

### 2. engine-step
- ✓ 源码：`mini_vllm/engine.py`、`mini_vllm/scheduler.py`、`mini_vllm/metrics.py`
- ⚠️ 真实 trace screenshot：缺，用自绘 step trace。

### 3. sequence-accounting
- ✓ 源码：`mini_vllm/request.py`
- ⚠️ 实际 token ids：不需要，用抽象 token tape。

### 4. paged-kv
- ✓ 源码：`mini_vllm/block_manager.py`、`mini_vllm/analysis.py`
- ✓ 参考图：`docs/assets/mem_capacity.png`
- ⚠️ 真实 KV tensor：不需要，用 block grid。

### 5. scheduler
- ✓ 源码：`mini_vllm/scheduler.py`、`mini_vllm/config.py`
- ✓ 参考图：`docs/assets/throughput.png`、`docs/assets/latency.png`

### 6. memory-pressure
- ✓ 源码：`mini_vllm/scheduler.py`、`mini_vllm/block_manager.py`
- ✓ README design notes：`README.md`

### 7. prefix-cache
- ✓ 源码：`mini_vllm/block_manager.py`
- ✓ 参考图：`docs/assets/prefix_cache.png`

### 8. observability-to-vllm
- ✓ 源码：`mini_vllm/metrics.py`、`mini_vllm/trace_viewer.py`
- ✓ 现有 artifact：`docs/trace.html`
- ⚠️ 真实 vLLM dashboard：缺，不伪造，用 production mapping panel。
