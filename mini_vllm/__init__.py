"""mini-vLLM: a from-scratch, GPU-free model of the systems core of a modern
LLM serving engine — paged KV-cache, continuous batching, preemption, and
prefix caching — with reproducible benchmarks.

The compute is simulated; the memory management and scheduling are real,
tested implementations. See the README for the OS-concept mapping.
"""
from .analysis import contiguous_capacity, paged_capacity
from .block_manager import BlockManager, cdiv
from .config import CacheConfig, ModelConfig, SchedulerConfig
from .engine import LLMEngine
from .memory_estimator import (
    KVCacheEstimate,
    ModelMemorySpec,
    ServingMemoryBudget,
    dtype_size_bytes,
    estimate_kv_cache_blocks,
)
from .metrics import EngineMetrics
from .model_runner import ModelRunner
from .request import Request, Sequence, SeqStatus
from .scheduler import Scheduler, StepWork

__version__ = "0.1.0"

__all__ = [
    "CacheConfig",
    "SchedulerConfig",
    "ModelConfig",
    "Request",
    "Sequence",
    "SeqStatus",
    "BlockManager",
    "cdiv",
    "Scheduler",
    "StepWork",
    "ModelRunner",
    "LLMEngine",
    "EngineMetrics",
    "contiguous_capacity",
    "paged_capacity",
    "ModelMemorySpec",
    "ServingMemoryBudget",
    "KVCacheEstimate",
    "dtype_size_bytes",
    "estimate_kv_cache_blocks",
]
