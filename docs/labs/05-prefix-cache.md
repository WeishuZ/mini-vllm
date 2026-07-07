# Lab 05: Prefix caching

## 目标

理解 shared system prompt、RAG、agent workload 如何通过 prefix caching
减少 prefill work 和 TTFT。

## 阅读

- `mini_vllm/block_manager.py`
- `benchmarks/bench_prefix.py`
- `tests/test_block_manager.py`

重点看:

- `BlockManager._block_hash`
- `BlockManager._shareable_prefix_blocks`
- `BlockManager.admit_prefix`
- `BlockManager._register_full_prompt_blocks`
- `BlockManager._enforce_prefix_cache_budget`

## 实验 1: 复现 prefix benchmark

```bash
.venv/bin/python benchmarks/bench_prefix.py
```

观察:

- prefill tokens computed
- TTFT p50
- cache hit rate

## 实验 2: shared prefix 长度

在 `benchmarks/bench_prefix.py` 中修改:

```python
SYSTEM_LENS = [128, 256, 512, 1024]
```

加入更短或更长的 system prompt, 观察收益曲线。

## 实验 3: cache budget

在 engine config 中设置:

```python
CacheConfig(enable_prefix_caching=True, prefix_cache_max_blocks=32)
CacheConfig(enable_prefix_caching=True, prefix_cache_max_blocks=128)
```

观察 eviction 和 hit rate。

## 观察问题

- 为什么 prefix sharing 必须从 token 0 连续命中?
- 为什么 block hash 包含从开头到当前 block 的所有 prefix tokens?
- 为什么 prefix cache 降低 prefill work, 但 decode tokens 还是要逐步生成?

## 验收

你应该能解释:

- prefix cache 是 page cache / dedup 的类比。
- active shared blocks 通过 ref count pinned。
- idle cached blocks 可被 LRU evictor 回收。
- prefix caching 对长共享前缀最有价值。

