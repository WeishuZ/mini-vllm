"""Estimate mini-vLLM KV blocks from a real model/GPU budget."""
from __future__ import annotations

from mini_vllm import ModelMemorySpec, ServingMemoryBudget, estimate_kv_cache_blocks


def main() -> None:
    # Rough Qwen2.5-1.5B-like shape. Adjust for the model you deploy.
    model = ModelMemorySpec(
        num_layers=28,
        hidden_size=1536,
        num_attention_heads=12,
        num_kv_heads=2,
        dtype="float16",
    )
    budget = ServingMemoryBudget(
        gpu_memory_gb=24,
        gpu_memory_utilization=0.90,
        model_weights_gb=3.5,
        non_kv_overhead_gb=2.0,
        tensor_parallel_size=1,
        block_size=16,
    )
    estimate = estimate_kv_cache_blocks(model, budget)
    print(estimate)
    print("mini-vLLM CacheConfig:", estimate.cache_config())


if __name__ == "__main__":
    main()

