"""Memory sizing helpers for connecting mini-vLLM to real deployments.

The engine itself works in abstract KV blocks. Real vLLM users usually think in
GPU GB, model size, dtype, layers, and heads. This module bridges those units so
the teaching examples can answer: "roughly how many mini-vLLM blocks would this
real GPU budget buy me?"
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from .config import CacheConfig


DTYPE_BYTES = {
    "fp32": 4,
    "float32": 4,
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "half": 2,
    "fp8": 1,
    "int8": 1,
}


def dtype_size_bytes(dtype: str) -> int:
    """Return bytes per scalar for common inference dtypes."""
    try:
        return DTYPE_BYTES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {dtype}") from exc


@dataclass(frozen=True)
class ModelMemorySpec:
    """Transformer KV-cache shape.

    ``num_kv_heads`` defaults to ``num_attention_heads`` for standard MHA. Set
    it lower for GQA/MQA models.
    """

    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int | None = None
    dtype: str = "float16"

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_heads(self) -> int:
        return self.num_kv_heads or self.num_attention_heads

    @property
    def dtype_bytes(self) -> int:
        return dtype_size_bytes(self.dtype)


@dataclass(frozen=True)
class ServingMemoryBudget:
    """Per-GPU serving memory budget."""

    gpu_memory_gb: float
    gpu_memory_utilization: float = 0.90
    model_weights_gb: float = 0.0
    non_kv_overhead_gb: float = 1.0
    tensor_parallel_size: int = 1
    block_size: int = 16


@dataclass(frozen=True)
class KVCacheEstimate:
    bytes_per_token_per_gpu: int
    block_bytes_per_gpu: int
    available_kv_bytes_per_gpu: int
    num_gpu_blocks: int
    total_gpu_slots: int

    def cache_config(self, *, num_cpu_blocks: int = 1024) -> CacheConfig:
        """Create a matching mini-vLLM ``CacheConfig``."""
        return CacheConfig(
            block_size=self.block_bytes_per_gpu // self.bytes_per_token_per_gpu,
            num_gpu_blocks=self.num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
        )


def gb_to_bytes(gb: float) -> int:
    return int(gb * 1024**3)


def estimate_kv_cache_blocks(
    model: ModelMemorySpec,
    budget: ServingMemoryBudget,
) -> KVCacheEstimate:
    """Estimate per-GPU KV capacity as mini-vLLM blocks.

    The formula is intentionally transparent:

    ``KV bytes/token = 2 * layers * kv_heads_per_gpu * head_dim * dtype_bytes``

    The factor of 2 is for key and value. For tensor parallelism, KV heads are
    split approximately evenly across GPUs.
    """
    if budget.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size must be >= 1")
    if budget.block_size < 1:
        raise ValueError("block_size must be >= 1")
    if not 0 < budget.gpu_memory_utilization <= 1:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")

    kv_heads_per_gpu = ceil(model.kv_heads / budget.tensor_parallel_size)
    bytes_per_token = (
        2
        * model.num_layers
        * kv_heads_per_gpu
        * model.head_dim
        * model.dtype_bytes
    )
    block_bytes = bytes_per_token * budget.block_size
    available = (
        gb_to_bytes(budget.gpu_memory_gb) * budget.gpu_memory_utilization
        - gb_to_bytes(budget.model_weights_gb)
        - gb_to_bytes(budget.non_kv_overhead_gb)
    )
    available = max(0, int(available))
    num_blocks = available // block_bytes if block_bytes else 0
    return KVCacheEstimate(
        bytes_per_token_per_gpu=bytes_per_token,
        block_bytes_per_gpu=block_bytes,
        available_kv_bytes_per_gpu=available,
        num_gpu_blocks=num_blocks,
        total_gpu_slots=num_blocks * budget.block_size,
    )

