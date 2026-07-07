"""Recompute vs swap preemption under KV pressure."""
from __future__ import annotations

from typing import Dict

from mini_vllm import CacheConfig, LLMEngine, ModelConfig, SchedulerConfig, workloads

MODES = ["recompute", "swap"]


def _run(mode: str) -> Dict:
    e = LLMEngine(
        CacheConfig(
            block_size=16,
            num_gpu_blocks=80,
            num_cpu_blocks=500,
            enable_prefix_caching=False,
        ),
        SchedulerConfig(
            max_num_seqs=32,
            max_num_batched_tokens=1024,
            preemption_mode=mode,
        ),
        ModelConfig(),
    )
    e.add_requests(workloads.burst(n=30, prompt_mean=128, gen_mean=64, seed=5))
    m = e.run()
    return {
        "completed": m.num_completed,
        "throughput": m.throughput_tok_s,
        "ttft_p99": m.ttft_ms_p99,
        "preemptions": m.num_preemptions,
        "swaps": m.num_swaps,
        "prefill_tokens": m.total_prefill_tokens,
    }


def run() -> Dict:
    rows = {mode: _run(mode) for mode in MODES}
    return {
        "modes": MODES,
        "completed": [rows[m]["completed"] for m in MODES],
        "throughput": [round(rows[m]["throughput"], 1) for m in MODES],
        "ttft_p99": [round(rows[m]["ttft_p99"], 1) for m in MODES],
        "preemptions": [rows[m]["preemptions"] for m in MODES],
        "swaps": [rows[m]["swaps"] for m in MODES],
        "prefill_tokens": [rows[m]["prefill_tokens"] for m in MODES],
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, AMBER, CYAN, GREEN, RED

    plt = apply()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = range(len(data["modes"]))
    width = 0.34
    ax.bar([i - width / 2 for i in x], data["preemptions"], width, color=RED, label="recompute preemptions")
    ax.bar([i + width / 2 for i in x], data["swaps"], width, color=AMBER, label="swap outs")
    for i, thr in enumerate(data["throughput"]):
        ax.text(i, max(data["preemptions"][i], data["swaps"][i]) + 0.5, f"{thr:.0f} tok/s",
                color=GREEN if i == 0 else CYAN, ha="center", fontsize=9)
    ax.set_xticks(list(x))
    ax.set_xticklabels(data["modes"])
    ax.set_ylabel("events")
    ax.set_title("Preemption mode changes the recovery cost under KV pressure", fontsize=10.5)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os

    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/preemption.png")

