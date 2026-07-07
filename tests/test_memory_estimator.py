import pytest

from mini_vllm import (
    ModelMemorySpec,
    ServingMemoryBudget,
    dtype_size_bytes,
    estimate_kv_cache_blocks,
)


def test_dtype_size_bytes():
    assert dtype_size_bytes("float16") == 2
    assert dtype_size_bytes("BF16") == 2
    assert dtype_size_bytes("fp8") == 1
    with pytest.raises(ValueError):
        dtype_size_bytes("weird")


def test_estimate_kv_cache_blocks_for_mha():
    model = ModelMemorySpec(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        dtype="float16",
    )
    budget = ServingMemoryBudget(
        gpu_memory_gb=1,
        gpu_memory_utilization=1.0,
        model_weights_gb=0,
        non_kv_overhead_gb=0,
        block_size=16,
    )

    est = estimate_kv_cache_blocks(model, budget)

    assert est.bytes_per_token_per_gpu == 2 * 2 * 4 * 4 * 2
    assert est.block_bytes_per_gpu == est.bytes_per_token_per_gpu * 16
    assert est.num_gpu_blocks > 0
    assert est.total_gpu_slots == est.num_gpu_blocks * 16
    assert est.cache_config().num_gpu_blocks == est.num_gpu_blocks


def test_tensor_parallel_splits_kv_heads():
    model = ModelMemorySpec(
        num_layers=4,
        hidden_size=32,
        num_attention_heads=8,
        num_kv_heads=8,
    )
    one = estimate_kv_cache_blocks(
        model,
        ServingMemoryBudget(gpu_memory_gb=1, tensor_parallel_size=1, non_kv_overhead_gb=0),
    )
    two = estimate_kv_cache_blocks(
        model,
        ServingMemoryBudget(gpu_memory_gb=1, tensor_parallel_size=2, non_kv_overhead_gb=0),
    )

    assert two.bytes_per_token_per_gpu == one.bytes_per_token_per_gpu // 2
    assert two.num_gpu_blocks >= one.num_gpu_blocks * 2 - 1


def test_invalid_memory_estimator_inputs():
    with pytest.raises(ValueError):
        ModelMemorySpec(num_layers=1, hidden_size=10, num_attention_heads=3).head_dim
    with pytest.raises(ValueError):
        estimate_kv_cache_blocks(
            ModelMemorySpec(num_layers=1, hidden_size=8, num_attention_heads=2),
            ServingMemoryBudget(gpu_memory_gb=1, gpu_memory_utilization=1.5),
        )
