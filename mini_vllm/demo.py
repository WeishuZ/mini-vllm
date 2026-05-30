"""A tiny end-to-end demo. Run with::

    python -m mini_vllm.demo

It serves a burst of requests through a small KV pool so you can watch
continuous batching, preemption, and (optionally) prefix caching at work.
"""
from __future__ import annotations

from . import workloads
from .config import CacheConfig, ModelConfig, SchedulerConfig
from .engine import LLMEngine


def main() -> None:
    print("mini-vLLM demo — continuous batching on a deliberately small KV pool\n")

    cache = CacheConfig(block_size=16, num_gpu_blocks=180, num_cpu_blocks=512)
    sched = SchedulerConfig(
        max_num_seqs=24,
        max_num_batched_tokens=1024,
        enable_chunked_prefill=True,
        policy="continuous",
        preemption_mode="recompute",
    )
    engine = LLMEngine(cache, sched, ModelConfig())
    engine.add_requests(workloads.burst(n=64, prompt_mean=200, gen_mean=120, seed=7))

    # step manually so we can narrate a few steps, then drain
    for _ in range(8):
        if not engine.scheduler.has_unfinished and not engine._pending:
            break
        engine._release_arrivals()
        w = engine.step()
        print(
            f"step {engine.num_steps:>3}  "
            f"running={len(engine.scheduler.running):>2}  "
            f"prefill_tok={w.num_prefill_tokens:>4}  "
            f"decode={w.num_decode_seqs:>2}  "
            f"KV={engine.block_manager.gpu_utilization*100:5.1f}%  "
            f"preempt={w.preempted}  done={len(engine.completed)}"
        )

    metrics = engine.run()
    print("\nfinal:", metrics.summary())


if __name__ == "__main__":
    main()
