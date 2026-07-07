# Spoken Script

## Notes
- Audience: 已经能读 Python 源码，目标是通过 miniVLLM 深入掌握 LLM serving control plane，并具备继续阅读、改动真实 vLLM 的能力。
- Tone: 硬核源码讲解。重点讲系统约束、状态不变量、算法分支、机制引入原因和效果。
- Assumptions: 这是配套学习网站的长教程脚本。网页后续按章节组织，每章是“问题背景 -> LLM 约束 -> miniVLLM 抽象 -> vLLM 映射 -> 工程权衡”。

---

先从最小闭环开始。

`LLMEngine.step()`。

这不是一个普通循环。

这是一个 LLM serving engine 在一个调度周期里做的全部事情。

调度。

分配 KV。

推进 prefill。

推进 decode。

更新时间。

回收结束请求。

写入指标。

---

如果你能解释清楚这一行。

后面所有机制都能挂上去。

PagedAttention 是为了解决 KV 怎么放。

Continuous batching 是为了解决请求怎么混。

Preemption 是为了解决 KV 不够时谁让路。

Prefix caching 是为了解决重复前缀为什么不要重算。

---

先看 LLM 推理本身。

一个 decoder-only LLM 生成文本时，是 autoregressive 的。

也就是下一个 token 依赖前面所有 token。

所以生成 100 个 token，不是一次 forward 出 100 个。

而是一步一步生成。

---

这就把推理天然分成两个阶段。

prefill。

decode。

prefill 处理已有 prompt。

decode 每一步生成一个新 token。

这两个阶段看起来都叫 forward。

但系统性质完全不同。

---

prefill 的输入通常是一整段 prompt。

比如 512 个 token。

模型可以一次性并行处理这些 token。

矩阵乘法规模大。

GPU 利用率容易打满。

它更偏 compute-bound。

---

decode 就不一样。

每条 sequence 每次只生成一个 token。

算这个 token 的 attention 时。

它要读取历史上所有 token 的 K 和 V。

新计算很少。

读 KV 很多。

它更偏 memory-bandwidth-bound。

---

所以 LLM serving 的核心矛盾来了。

prefill 喜欢大块 token。

decode 喜欢很多 sequence 一起 batch。

prefill 太大，会挡住 decode。

decode batch 太小，GPU 又吃不满。

---

miniVLLM 的 `ModelRunner` 把这个差异压成一个简单 cost model。

prefill cost 和 token 数线性相关。

decode cost 和 sequence 数相关，但 per-token 成本随 batch 变大而下降。

这不是随便造的。

它是在模拟真实推理里 prefill 和 decode 的不同瓶颈。

---

真实 vLLM 里，这个差异会落到更复杂的地方。

attention kernel。

KV cache layout。

scheduler policy。

CUDA graph。

multi-step scheduling。

chunked prefill。

但是最高层的问题没变。

怎么把 prefill 和 decode 混在一起，吞吐高，尾延迟还不能爆。

---

回到 `LLMEngine.step()`。

它先调用 `scheduler.schedule()`。

Scheduler 返回一个 `StepWork`。

`StepWork` 是控制面和执行面的分界线。

它只描述这一步计划做什么。

不直接改 token。

不直接改时钟。

---

`StepWork.prefill` 里是若干 `(sequence, chunk_size)`。

意思是这些 sequence 要推进 prefill。

每个推进多少 token。

`StepWork.decode` 里是若干 sequence。

意思是这些 sequence 各生成一个 token。

---

Engine 拿到计划以后，才提交状态。

prefill 会让 `num_computed += chunk`。

decode 会让 `num_generated += 1`。

同时 `num_computed += 1`。

然后 clock 按 `ModelRunner.step_latency_ms(work)` 前进。

---

这套分层非常重要。

Scheduler 做的是“应该跑谁”。

BlockManager 做的是“KV 放不放得下”。

Engine 做的是“把计划变成事实”。

ModelRunner 做的是“这批 work 需要多长时间”。

Metrics 做的是“这个事实怎么被观测”。

---

现在看 Sequence。

Sequence 不是 request 的简单包装。

它是 request 在 serving engine 里的运行态。

它承载三类状态。

逻辑 token 状态。

KV materialization 状态。

调度生命周期状态。

---

逻辑 token 状态由两个数决定。

`prompt_len`。

`num_generated`。

所以当前 sequence 已经存在的 token 数是：

`length = prompt_len + num_generated`。

---

KV 状态由另一个数决定。

`num_computed`。

它表示前多少个 token 的 KV 已经算好，并且应该在 KV cache 里可用。

注意，是前缀。

不是任意 token 集合。

---

于是 prefill 的定义就很精确。

只要 `num_computed < length`。

这个 sequence 就欠 KV。

欠多少？

`length - num_computed`。

这就是 `prefill_remaining`。

---

decode 的前提也很精确。

只有当 `num_computed == length`。

这个 sequence 的已有上下文才全部 materialized。

这时才能生成下一个 token。

生成以后，逻辑长度加一。

KV 也要跟着加一。

---

这里有一个很硬的不变量。

对任何 active sequence。

`num_computed` 不能大于 `length`。

也不能在 decode 后落后于新的 `length`。

否则你就生成了一个 token。

但没有给这个 token 存 KV。

下一步 attention 就没法正确读历史。

---

这个不变量解释了 recompute preemption。

当 KV 被抢掉时。

`num_generated` 不能清零。

因为用户已经看到了这些 token。

但 `num_computed` 必须回到 0。

因为 KV 真的没了。

---

所以 preempted sequence 回来以后。

它不是从原始 prompt 开始。

而是要对 `prompt + generated_so_far` 重新 prefill。

这就是 recompute 的系统成本。

它不是抽象的“慢一点”。

它会直接增加 prefill token work。

---

接下来讲 KV cache 为什么会成为核心瓶颈。

Transformer 每层 attention 都要存 K 和 V。

每个 token 都会留下 KV。

所以 KV cache 大小和生成中的 token 总数线性增长。

---

粗略公式是：

每 token KV bytes 等于：

`2 * num_layers * num_kv_heads * head_dim * dtype_bytes`。

这里的 2 是 K 和 V。

如果是 GQA，`num_kv_heads` 会小于 attention heads。

但它仍然很大。

---

所以 serving 里最贵的资源，不只是模型权重。

权重基本是固定成本。

KV cache 是随并发、上下文长度、输出长度动态增长的成本。

同一个模型，同一张 GPU。

你能服务多少请求，很多时候被 KV cache 决定。

---

这就是为什么 PagedAttention 重要。

它不是让 attention 数学变了。

它是让 KV cache 的内存管理从 contiguous reservation。

变成 paged allocation。

它解决的是内存利用率问题。

---

在 miniVLLM 里。

`BlockManager` 就是这个 KV memory manager。

每个 physical block 存固定数量 token 的 KV。

`block_size` 默认 16。

每个 sequence 的 `block_table` 把 logical block 映射到 physical block。

---

逻辑上，sequence 的 token 是连续的。

第 0 个 block。

第 1 个 block。

第 2 个 block。

但物理上，这些 block 可以散落在任意 free block id 上。

这就是 page table 的作用。

---

`grow(seq, num_new)` 是 BlockManager 的核心。

它先看当前 KV token 数。

也就是 `seq.num_kv_tokens`。

再看当前 capacity。

也就是 `len(seq.block_table) * block_size`。

如果 target 超过 capacity，就补 block。

---

这里的设计效果非常具体。

如果一个请求只有 80 个 token。

block size 是 16。

它就吃 5 个 block。

不是因为模型最大上下文是 2048。

就给它预留 2048 个 token 的 KV。

---

contiguous allocation 的问题在这里。

它按最大长度保守预留。

max context 越大，单请求 reservation 越大。

同样 KV budget 下，能并发的请求数就越少。

---

paged allocation 的容量由实际长度分布决定。

短请求不会因为系统支持长上下文而被迫占一大段 reservation。

浪费主要是最后一个 block 没填满。

也就是 internal fragmentation。

---

这就是 miniVLLM memory benchmark 的核心解释。

contiguous 被 `max_seq_len` 控制。

paged 被实际 token 使用量控制。

所以上下文上限越长。

两者差距越明显。

---

PagedAttention 的真正收益不是“没有浪费”。

而是把浪费从灾难性的 external fragmentation 和 over-reservation。

压缩到可控的 partial block waste。

这个差别在 serving 场景里非常大。

因为请求长度分布通常很不均匀。

---

BlockManager 还必须处理共享。

共享一旦出现，就需要 ref count。

一个 physical block 可能被多个 sequence 的 block table 指向。

释放时不能简单 free。

必须 decref。

ref count 归零，才可能回到 free pool。

---

但共享也带来写入问题。

如果共享的是 full prefix block。

它已经填满。

语义稳定。

后续不会往这个 block 里追加 token。

所以继续共享是安全的。

---

如果共享的是 partial tail。

就危险了。

因为下一个 token 可能要写进这个 tail block。

如果直接写。

所有共享这个 physical block 的 sequence 都会看到这次写入。

---

所以 `grow()` 里有 copy-on-write。

条件很窄。

当前 block table 非空。

当前 KV token 数不在 block 边界。

尾块 ref count 大于 1。

满足这些，才 fork 出一个私有 block。

---

这个分支的效果是：

共享 prefix 的内存收益保住。

分叉后继续生成的正确性也保住。

这和操作系统里的 COW 很像。

读共享。

写复制。

---

现在讲调度。

LLM serving 的调度目标，不是简单最大化 batch size。

而是在 TTFT、ITL、throughput 和 KV pressure 之间找平衡。

这几个指标天然互相拉扯。

---

prefill 喜欢吞大 token chunk。

这样 GPU 算得爽。

但大 prefill 会占用 step latency。

正在 decode 的请求就要等。

用户看到的 stream 会卡。

---

decode 喜欢把很多 sequence 合在一起。

因为每条 sequence 一步只生成一个 token。

单条 decode 太小。

GPU 利用率差。

合起来以后，per-token decode cost 下降。

---

所以 Scheduler 的问题不是“下一条请求是谁”。

而是：

这一轮 token budget 怎么分给 prefill 和 decode。

这一轮 KV block 够不够支撑这些选择。

哪些请求可以进入 running。

哪些必须等。

---

miniVLLM 的 continuous policy 顺序很关键。

先 `_resume_swapped`。

再 `_advance_running`。

最后 `_admit_waiting`。

这个顺序体现了 serving 的优先级。

已经被系统接收的运行中请求，比新请求优先。

---

`_advance_running` 先处理 running sequence。

如果 sequence 还在 prefill。

它拿一个 prefill chunk。

如果 prefill 完成。

它进入 decode，一次生成一个 token。

---

这里的 chunked prefill 是关键机制。

没有 chunking。

一个超长 prompt 可能一次吃完整个 step。

decode 请求就被挡住。

有了 chunking。

长 prompt 被切开。

decode 可以在多个 step 之间继续穿插推进。

---

chunked prefill 的效果不是免费变快。

它是把长 prompt 的 prefill 分散到多个 step。

换来 decode latency 更稳定。

也就是牺牲部分单请求 prefill 连续性。

换整体 serving 的交互体验。

---

`max_num_batched_tokens` 就是在调这个权衡。

它不是普通 batch size。

它是一次 scheduler step 的 token budget。

prefill token 和 decode token 都要从里面扣。

---

如果这个值很小。

长 prompt 会被切得很碎。

TTFT 可能上升。

如果这个值很大。

prefill 更容易吞大块。

throughput 可能变好。

但 decode 间隔可能变差。

---

真实 vLLM 里的 `max_num_batched_tokens` 也是非常关键的调优参数。

它影响 TTFT。

影响 ITL。

影响吞吐。

也影响 prefill/decode 的混合形态。

理解 miniVLLM 里的这个参数，后面读真实 vLLM scheduler 会轻很多。

---

再看 static batching。

它在 miniVLLM 里是 baseline。

只有当 running 为空，才 admit 一批。

而且 admission 时按 `prompt_len + max_tokens` 预留 worst-case KV。

---

这个策略为什么差？

第一，它制造 head-of-line blocking。

新请求到了，但当前 batch 没清空。

它不能加入。

所以 TTFT 尾部会炸。

---

第二，它制造 batch decay。

同一个 batch 里，短输出先结束。

batch 逐渐变窄。

但 static 不补新请求。

所以 decode 后半段 GPU 利用率越来越差。

---

continuous batching 的核心效果，就是持续 refill。

sequence 完成后，KV 释放。

running 空位出现后，新请求可以进入。

decode batch 宽度被维持住。

系统吞吐更接近稳定流，而不是一批一批衰减。

---

这就是 vLLM serving 快的一个关键原因。

不是单条请求更神奇。

而是多请求并发下，调度器让 GPU 少空转。

让 KV 少浪费。

让 decode batch 更持续。

---

现在讲 memory pressure。

只要是 paged KV，就会面对一个事实。

block pool 是有限的。

running sequence 每生成一个 token，KV 就继续增长。

即使 admission 时放得下，后面也可能放不下。

---

所以 Scheduler 每推进一个 sequence 前。

都要问 BlockManager：

`can_grow(seq, need)`。

如果不能 grow。

系统必须释放别人的 KV。

这就是 preemption。

---

miniVLLM 里，running sequence 按 arrival 排序。

老请求优先。

当需要腾 block 时。

从队尾 preempt 更新的 sequence。

这是一个简单但有效的优先级模型。

---

preemption 的第一种模式是 recompute。

它直接 free 这个 sequence 的 GPU block。

然后 `reset_for_recompute()`。

sequence 回到 waiting 队头。

---

为什么放回队头？

因为它不是新来的请求。

它已经被系统 admit 过。

只是 KV 被抢了。

放回队头可以避免它被后续新请求无限插队。

---

recompute 的效果很明确。

它降低了当前 GPU KV pressure。

但把成本推迟到了未来。

未来它重新运行时，要把已有上下文重新 prefill。

所以 total prefill tokens 会增加。

---

第二种模式是 swap。

swap 不丢逻辑状态。

它把 GPU block 对应的占用转移到 CPU block pool。

之后有空间时再 swap in。

---

真实系统里，swap 的问题是带宽和延迟。

GPU 到 CPU 的迁移不便宜。

如果频繁 swap，可能比 recompute 更糟。

miniVLLM 没模拟搬运时间。

但它保留了队列语义和资源语义。

---

还有一个容易忽略的约束。

miniVLLM 里 swap 只在 prefix caching 关闭时启用。

因为 prefix caching 会引入共享 block。

共享 block 的 swap 语义更复杂。

你不能简单假设每个 block 都是 sequence 私有的。

---

这点很适合训练真实 vLLM 开发直觉。

一个机制单独看很简单。

两个机制叠在一起，状态空间会暴涨。

prefix sharing、ref count、swap、COW、LRU eviction。

组合起来就很难写对。

---

memory pressure 还有另一个保护。

watermark。

admission 不会把所有 available block 用光。

它会保留一部分 headroom。

---

为什么需要 headroom？

因为 running sequence 不是静态的。

它们 decode 时还会继续长。

如果 admission 把空间吃满。

下一步 running sequence 需要 grow 时，就只能 preempt。

---

这会导致 thrashing。

刚 admit。

马上 preempt。

preempt 后又有空位。

又 admit。

然后继续 preempt。

吞吐、TTFT、prefill work 都会变差。

---

watermark 的效果，是牺牲一部分瞬时 admission。

换 running set 的稳定推进。

它不是保守主义。

它是过载控制。

在真实 serving 里，这类 admission control 非常重要。

---

现在讲 prefix caching。

这是另一个 KV 层机制。

它解决的问题不是 KV 不够。

而是相同前缀为什么要重复 prefill。

---

很多 workload 有共享前缀。

固定 system prompt。

同一个 RAG 文档。

agent 的工具说明。

多轮对话里稳定的历史上下文。

这些 token 的 KV，如果每个请求都重新算，就很浪费。

---

miniVLLM 的 prefix cache 缓存的是 full prompt block。

不是字符串。

不是 token list。

也不是输出。

而是已经 materialized 的 KV block。

---

`_register_full_prompt_blocks()` 会在 prefill 之后发布 cacheable block。

只有完整 prompt block 才能发布。

而且必须是当前 sequence 独占的 block。

如果 ref count 不等于 1，就不会发布。

---

这个限制是为了保证 cache entry 的语义清楚。

cache 里保存的是某个 prefix 对应的一段物理 KV。

它必须稳定。

不能是一个还在被别的共享写入语义影响的 block。

---

`admit_prefix()` 发生在新 sequence admission 时。

它从 token 0 开始查。

能连续命中多少 full blocks，就共享多少。

然后把这些 physical block append 到新 sequence 的 block table。

ref count 加一。

---

命中以后，`seq.num_computed` 直接跳到 covered token 数。

这意味着这些 token 不再需要 prefill。

所以 prefix cache 的直接效果是：

减少 prefill token work。

降低 TTFT。

减少 repeated prompt 的计算成本。

---

它为什么不直接减少 decode？

因为 decode 是生成新 token。

新 token 的 KV 还不存在。

prefix cache 只能复用已有前缀的 KV。

后续生成部分仍然要一步一步算。

---

prefix cache 最关键的是 hash 语义。

不能只 hash 当前 block token。

因为 attention 的 KV 语义依赖完整历史。

同样的 16 个 token，出现在不同前文后面。

它们对应的 KV 不一样。

---

所以 `_block_hash(token_ids, end)` hash 的是 `[0, end)`。

也就是从开头到这个 block 结束的整个 prefix。

这保证了只有完整前缀相同，才会命中。

---

这也解释了为什么只能连续命中。

如果第 3 个 block miss。

那第 4 个 block 就算 token 内容碰巧一样。

也不能共享。

因为第 4 个 block 的 KV 依赖第 3 个 block 之前的完整上下文。

---

prefix cache 的生命周期也很关键。

active sequence 引用 cached block 时。

这个 block 是 pinned。

ref count 大于 0。

不能 eviction。

---

sequence 结束后。

如果 block 属于 prefix cache。

它不会立刻回到 free pool。

它会 resident。

ref count 变成 0。

状态变成 evictable。

---

evictable cached block 仍然占 GPU KV pool。

这就是 prefix cache 的成本。

它换来了未来命中。

但占用了当前显存。

所以必须有 LRU 和 cache budget。

---

当 `prefix_cache_max_blocks` 被超过。

或者 allocation pressure 需要 block。

BlockManager 会驱逐 idle cached block。

驱逐以后，它才真正回到 free pool。

---

这就是 prefix caching 的完整权衡。

共享前缀越长。

请求复用越多。

收益越高。

但 cache resident set 越大。

对 active KV 的挤压也越明显。

---

接下来讲 metrics。

真实系统开发不是看“能跑”。

是看指标能不能定位瓶颈。

miniVLLM 里的指标虽然简化。

但已经覆盖了几类核心问题。

---

throughput 看总产出。

TTFT 看用户第一次看到 token 的等待。

E2E 看请求完整完成时间。

peak KV utilization 看显存压力。

preemption 和 swap 看过载行为。

prefix cache hit 和 saved tokens 看复用收益。

---

这些指标必须组合解释。

TTFT 高，不一定是 scheduler 差。

可能是 queue time 高。

可能是 prefill 太长。

可能是 prefix cache 没命中。

也可能是 recompute preemption 反复打断。

---

throughput 高，也不一定好。

如果 throughput 高，但 ITL 很差。

用户看到的 streaming 会卡。

如果 TTFT p99 爆。

说明尾部请求体验很差。

serving 不是只优化平均值。

---

prefix cache hit rate 也不能单独看。

一个 workload 可能 hit rate 很高。

但共享 prefix 很短。

saved prefill tokens 很少。

TTFT 改善就有限。

---

preemption count 也要和 total prefill tokens 一起读。

recompute preemption 多，会把 prefill work 放大。

swap 多，则说明 GPU KV 压力持续存在。

它们都不是单纯的“事件计数”。

而是系统进入压力区的信号。

---

trace viewer 在这里的作用，是把 aggregate metrics 拆回 step 级别。

某个请求为什么 TTFT 高。

看它在 waiting 里待了多久。

看它是否被 preempt。

看它拿到的 prefill chunk 有没有被切碎。

---

这也是阅读真实 vLLM trace 和 logs 的基本方法。

不要只看最终吞吐。

要看每个 step 的 running set。

waiting queue。

KV usage。

prefill/decode mix。

preemption 或 eviction 事件。

---

现在把 miniVLLM 映射到真实 vLLM。

`LLMEngine` 对应 engine 层的主循环和状态管理。

`Scheduler` 对应 request scheduling 和 token budget 决策。

`BlockManager` 对应 KV cache manager。

`ModelRunner` 对应真实 worker 里的 model execution。

---

miniVLLM 省掉了真实模型执行。

真实 vLLM 里，ModelRunner 后面还有很多层。

model executor。

attention backend。

CUDA kernel。

sampling。

distributed executor。

但 control plane 的结构仍然能对应上。

---

真实 vLLM 开发时，你会遇到更多细节。

KV cache dtype。

KV cache block profiling。

CUDA graph capture。

prefix cache hash 和 eviction。

chunked prefill policy。

speculative decoding。

LoRA adapter。

multi-modal input。

---

但如果你没有先掌握 miniVLLM 里的这几个不变量。

真实代码会很难读。

因为真实 vLLM 的复杂度，不是来自一个神秘算法。

而是来自多个机制叠加。

---

KV cache 要分页。

分页以后要 block table。

block table 以后有 ref count。

ref count 以后有 sharing。

sharing 以后有 COW。

prefix cache 以后有 pinned 和 evictable。

preemption 以后又要 recompute 或 swap。

---

Scheduler 也是一样。

有 waiting。

有 running。

有 swapped。

有 finished。

有 token budget。

有 memory budget。

有 admission control。

有 prefill/decode 混合。

这些状态组合起来，才是 serving engine。

---

所以这个学习网站的最终结构，应该是一层层剖开。

第一层，LLM inference 的物理形态。

第二层，Engine step 的状态提交。

第三层，KV cache 的分页内存管理。

第四层，Scheduler 的 continuous batching。

第五层，memory pressure 下的 preemption。

第六层，prefix cache 的共享和生命周期。

第七层，metrics 如何定位瓶颈。

第八层，映射到真实 vLLM 开发。

---

每一章都围绕一个硬问题。

不是“这个模块做什么”。

而是“为什么必须有这个模块”。

没有它会出什么问题。

加了它解决什么问题。

又引入了什么新状态和新风险。

---

学完以后，要能回答这些问题。

prefill 和 decode 为什么不能用同一种调度直觉。

KV cache 为什么会比权重更影响并发。

PagedAttention 为什么本质上是内存管理。

Continuous batching 为什么能提高饱和吞吐。

Watermark 为什么能减少 thrashing。

Prefix cache 为什么只能连续命中。

Metrics 为什么必须组合解读。

---

如果这些问题能讲清楚。

再去读真实 vLLM。

你就不会迷路。

你看到的不再是一堆文件。

而是一套围绕 KV、调度、执行和观测展开的 serving 系统。

