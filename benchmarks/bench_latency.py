"""Time-to-first-token under streaming (Poisson) load, swept across the
*sustainable* load range (offered load below the throughput ceiling, so both
policies serve the same work). The difference is pure latency: static batching
makes an arriving request wait for the current batch to drain (head-of-line
blocking), so its TTFT — especially the tail — is enormous; continuous batching
admits into the running batch immediately and keeps TTFT flat.
"""
from __future__ import annotations

from typing import Dict

from mini_vllm import (CacheConfig, LLMEngine, ModelConfig, SchedulerConfig,
                       workloads)

RATES = [2, 3, 4, 5, 6, 7]
N_REQUESTS = 200
HEADLINE_RATE = 6


def _run(policy: str, rate: float) -> Dict:
    e = LLMEngine(
        CacheConfig(block_size=16, num_gpu_blocks=700),
        SchedulerConfig(policy=policy, max_num_seqs=128, max_num_batched_tokens=2048),
        ModelConfig(),
    )
    e.add_requests(workloads.poisson(n=N_REQUESTS, rate_rps=rate,
                                     prompt_mean=256, gen_mean=128, seed=7))
    m = e.run()
    return {"p50": m.ttft_ms_p50, "p99": m.ttft_ms_p99, "thr": m.throughput_tok_s}


def run() -> Dict:
    static = {r: _run("static", r) for r in RATES}
    cont = {r: _run("continuous", r) for r in RATES}
    h = HEADLINE_RATE
    return {
        "rates": RATES,
        "n_requests": N_REQUESTS,
        "static_ttft_p50": [round(static[r]["p50"], 1) for r in RATES],
        "static_ttft_p99": [round(static[r]["p99"], 1) for r in RATES],
        "continuous_ttft_p50": [round(cont[r]["p50"], 1) for r in RATES],
        "continuous_ttft_p99": [round(cont[r]["p99"], 1) for r in RATES],
        # both policies sustain ~the same throughput here (fair comparison)
        "static_throughput": [round(static[r]["thr"], 1) for r in RATES],
        "continuous_throughput": [round(cont[r]["thr"], 1) for r in RATES],
        "headline_rate": h,
        "headline_static_p99": round(static[h]["p99"], 1),
        "headline_continuous_p99": round(cont[h]["p99"], 1),
        "headline_p99_improvement": round(static[h]["p99"] / max(1e-9, cont[h]["p99"]), 1),
        "continuous_worst_p99": round(max(cont[r]["p99"] for r in RATES), 1),
        "static_worst_p99": round(max(static[r]["p99"] for r in RATES), 1),
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, GREEN, MUTED, RED, CYAN
    plt = apply()

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    x = data["rates"]
    ax.plot(x, data["static_ttft_p99"], "o-", color=RED, label="static  p99")
    ax.plot(x, data["static_ttft_p50"], "o--", color=MUTED, label="static  p50")
    ax.plot(x, data["continuous_ttft_p99"], "o-", color=GREEN, label="continuous  p99")
    ax.plot(x, data["continuous_ttft_p50"], "o--", color=CYAN, label="continuous  p50")
    ax.set_yscale("log")
    ax.set_xlabel("offered load (requests / s)  ·  matched throughput")
    ax.set_ylabel("time to first token (ms, log scale)")
    ax.set_title(
        f"Head-of-line blocking: static p99 TTFT {data['headline_p99_improvement']:.0f}x worse @ "
        f"{data['headline_rate']} req/s",
        fontsize=10.5,
    )
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os
    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/latency.png")
