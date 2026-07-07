# vLLM Serve Command Examples

Small local model:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct
```

Constrain context and KV budget:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --max-num-batched-tokens 4096
```

Shared-prefix workload:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --enable-prefix-caching \
  --max-num-batched-tokens 4096
```

Tensor parallel example:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90
```

These commands are intentionally starting points. Pin your vLLM version and
check the matching official documentation before production use.

