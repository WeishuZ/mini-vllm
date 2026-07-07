# 真实 vLLM 部署 Runbook

这份 runbook 用来把 mini-vLLM 的学习结果迁移到真实 vLLM server。命令会随
vLLM 版本变化, 实际使用时应以对应版本官方文档为准。

## 1. 环境准备

建议在单独目录或虚拟环境中安装真实 vLLM, 不要把它作为 mini-vLLM 的项目依赖。

```bash
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

如果不用 `uv`, 请按你的 CUDA/ROCm/PyTorch 环境选择官方推荐安装方式。

## 2. 启动一个小模型

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

默认服务通常在:

```text
http://localhost:8000/v1
```

验证:

```bash
curl http://localhost:8000/v1/models
```

## 3. OpenAI-compatible client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY",
)

resp = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "用三句话解释 vLLM 为什么快。"}],
    temperature=0.2,
    max_tokens=256,
)

print(resp.choices[0].message.content)
```

## 4. 参数和 mini-vLLM 概念映射

| 真实 vLLM 参数/概念 | mini-vLLM 对应 | 作用 |
|---|---|---|
| `max_model_len` | request 最大 prompt+generation 长度假设 | 影响 KV cache 容量规划。 |
| `gpu_memory_utilization` | `CacheConfig.num_gpu_blocks` 的真实来源 | 决定多少 GPU 显存可用于 KV cache。 |
| `max_num_batched_tokens` | `SchedulerConfig.max_num_batched_tokens` | 控制每步 prefill/decode token budget。 |
| `max_num_seqs` | `SchedulerConfig.max_num_seqs` | 控制同时 running 的 sequence 数量。 |
| chunked prefill | `enable_chunked_prefill` | 长 prompt 和 decode latency 的折中。 |
| prefix caching | `enable_prefix_caching` | 复用相同前缀 KV blocks。 |
| preemption | `preemption_mode` | KV 压力下释放或迁移 sequence state。 |
| metrics endpoint | `EngineMetrics` | 观察 TTFT、ITL、throughput、KV cache usage。 |

## 4.1 KV cache 粗估

可以用 mini-vLLM 的估算器把真实模型/GPU 预算换成 block 数:

```bash
.venv/bin/python examples/kv_sizing.py
```

核心公式:

```text
KV bytes/token/GPU =
  2 * layers * kv_heads_per_gpu * head_dim * dtype_bytes
```

它不是真实 vLLM 的精确容量规划器, 但足够帮助你理解
`gpu_memory_utilization`, `max_model_len`, tensor parallel 和 KV blocks 的关系。

## 5. Benchmark checklist

每次 benchmark 至少记录:

- 模型名和版本。
- GPU 型号和数量。
- vLLM 版本。
- `max_model_len`。
- `gpu_memory_utilization`。
- `max_num_batched_tokens`。
- `max_num_seqs`。
- prompt length 分布。
- output length 分布。
- request rate 或 concurrency。
- TTFT p50/p99。
- ITL/TPOT p50/p99。
- E2E latency p50/p99。
- output tokens/s。
- KV cache usage。
- running/waiting requests。
- preemption 次数或相关告警。

## 6. 调优顺序

推荐顺序:

1. 先确定模型和硬件是否能稳定启动。
2. 固定 workload, 跑 baseline benchmark。
3. 检查是否 KV cache 压力过大。
4. 调 `max_model_len` 和 `gpu_memory_utilization`。
5. 调 `max_num_batched_tokens`, 在 TTFT、ITL、throughput 之间取舍。
6. 对长 prompt workload 观察 chunked prefill。
7. 对 shared prefix workload 打开 prefix caching。
8. 需要更大模型时再考虑 tensor parallel、pipeline parallel、量化。

## 7. 常见现象

TTFT 高:

- queue time 高, admission 压力大。
- prompt 太长, prefill 占用 token budget。
- `max_num_batched_tokens` 太小或 batch 策略不合适。

ITL/TPOT 高:

- decode batch 太大或 GPU memory bandwidth 压力大。
- 输出很长, decode 成为主要瓶颈。

吞吐低:

- batch occupancy 不够。
- request rate 低, 没有压到服务容量。
- CPU/tokenizer/network overhead 明显。

KV cache 压力高:

- context 太长。
- concurrency 太高。
- `max_model_len` 配得过大。
- prefix cache 或其他常驻 blocks 占用太多。

## 8. 回到 mini-vLLM 验证直觉

当真实 vLLM 指标异常时, 可以用 mini-vLLM 做一个简化复现实验:

- TTFT 高: 跑 `bench_latency.py`, 对比 static/continuous 和 chunked prefill。
- KV 压力高: 减小 `num_gpu_blocks`, 观察 preemption。
- shared prompt 收益不明显: 跑 `bench_prefix.py`, 检查前缀是否真的相同。
- block waste 疑问: 跑 `bench_memory.py`, 改 `block_size` 和长度分布。

这个 loop 是 mini-vLLM 作为教具最有价值的地方: 用小模型解释真实系统的大现象。
