"""Admission watermark sweep under KV pressure."""
from __future__ import annotations

from typing import Dict

from mini_vllm import CacheConfig, LLMEngine, ModelConfig, SchedulerConfig, workloads

WATERMARKS = [0.00, 0.02, 0.04, 0.08, 0.12]


def _run(watermark: float) -> Dict:
    e = LLMEngine(
        CacheConfig(block_size=16, num_gpu_blocks=120),
        SchedulerConfig(
            max_num_seqs=32,
            max_num_batched_tokens=1024,
            preemption_mode="recompute",
            watermark=watermark,
        ),
        ModelConfig(),
    )
    e.add_requests(workloads.burst(n=48, prompt_mean=180, gen_mean=100, seed=11))
    m = e.run()
    return {
        "completed": m.num_completed,
        "preemptions": m.num_preemptions,
        "throughput": m.throughput_tok_s,
        "ttft_p99": m.ttft_ms_p99,
        "peak_kv": m.peak_gpu_util,
    }


def run() -> Dict:
    rows = {w: _run(w) for w in WATERMARKS}
    best = min(WATERMARKS, key=lambda w: rows[w]["preemptions"])
    return {
        "watermarks": WATERMARKS,
        "preemptions": [rows[w]["preemptions"] for w in WATERMARKS],
        "throughput": [round(rows[w]["throughput"], 1) for w in WATERMARKS],
        "ttft_p99": [round(rows[w]["ttft_p99"], 1) for w in WATERMARKS],
        "completed": [rows[w]["completed"] for w in WATERMARKS],
        "best_by_preemptions": best,
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, GREEN, RED

    plt = apply()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.1))
    x = [w * 100 for w in data["watermarks"]]
    ax1.plot(x, data["preemptions"], "o-", color=RED, label="preemptions")
    ax1.set_xlabel("admission watermark (% of GPU blocks)")
    ax1.set_ylabel("preemptions")

    ax2 = ax1.twinx()
    ax2.plot(x, data["throughput"], "o-", color=GREEN, label="throughput")
    ax2.set_ylabel("throughput (tok/s)")
    ax2.tick_params(colors=GREEN)
    ax2.spines["right"].set_color(GREEN)

    ax1.set_title("Watermark leaves KV headroom and reduces pressure", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os

    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/watermark.png")

