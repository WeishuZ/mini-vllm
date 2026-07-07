# Lab 01: Token accounting 和请求生命周期

## 目标

理解一个 `Request` 如何变成 mutable 的 `Sequence`, 以及 prefill/decode
阶段如何推进 `num_computed` 和 `num_generated`。

## 阅读

- `mini_vllm/request.py`
- `mini_vllm/engine.py`
- `mini_vllm/model_runner.py`

重点看:

- `Sequence.length`
- `Sequence.prefill_remaining`
- `Sequence.in_prefill`
- `LLMEngine.step`
- `LLMEngine.metrics`

## 实验

新建一个临时脚本, 或在 Python REPL 里运行:

```python
from mini_vllm import CacheConfig, LLMEngine, ModelConfig, Request, SchedulerConfig

engine = LLMEngine(
    CacheConfig(block_size=4, num_gpu_blocks=16),
    SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=8),
    ModelConfig(),
)
engine.add_request(Request("a", prompt_len=10, max_tokens=3))
engine.add_request(Request("b", prompt_len=6, max_tokens=2))

for _ in range(8):
    engine._release_arrivals()
    work = engine.step()
    print("step", engine.num_steps, "prefill", work.num_prefill_tokens, "decode", work.num_decode_seqs)
    for seq in engine.sequences.values():
        print(seq.request_id, seq.status.value, seq.num_computed, seq.num_generated, seq.block_table)
```

## 观察问题

- 哪一步开始 decode?
- 第一个 token 生成时, `first_token_time` 何时写入?
- `num_computed` 为什么在 decode 时也会增加?
- 如果 `max_num_batched_tokens` 调小, TTFT 如何变化?

## 验收

你应该能解释:

- prefill 是“把已有 prompt token 的 KV materialize 出来”。
- decode 是“每个 running sequence 每步最多生成一个新 token”。
- TTFT 是 `first_token_time - arrival`。
- E2E latency 是 `finish_time - arrival`。

