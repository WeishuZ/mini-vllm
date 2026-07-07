# Lab 02: Paged KV cache

## 目标

理解 PagedAttention 的内存管理模型: 固定大小 physical blocks,
每个 sequence 用 block table 做 logical-to-physical 映射。

## 阅读

- `mini_vllm/block_manager.py`
- `mini_vllm/analysis.py`
- `benchmarks/bench_memory.py`
- `tests/test_block_manager.py`

重点看:

- `BlockManager.grow`
- `BlockManager.free`
- `BlockManager.fork`
- `contiguous_capacity`
- `paged_capacity`

## 实验 1: 复现 memory benchmark

```bash
make bench
```

或只跑:

```bash
.venv/bin/python benchmarks/bench_memory.py
```

观察 `docs/assets/mem_capacity.png` 和 `docs/results.json`。

## 实验 2: 改 block size

在 `benchmarks/bench_memory.py` 中临时修改:

```python
BLOCK_SIZE = 4
BLOCK_SIZE = 8
BLOCK_SIZE = 16
BLOCK_SIZE = 32
```

记录:

- paged fit
- paged utilization
- paged waste fraction

## 观察问题

- block size 变大时, 内部碎片如何变化?
- contiguous allocator 为什么必须按 `max_seq_len` 预留?
- paged allocator 为什么只需要按实际长度增长?

## 验收

你应该能解释:

- physical block 类似 OS 里的 physical frame。
- `Sequence.block_table` 类似 page table。
- paged KV 没有外部碎片, 但有尾块内部碎片。
- copy-on-write 只在 shared partial block 被写入时触发。

