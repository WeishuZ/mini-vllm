"""Sustained throughput vs offered load. Both policies track the offered rate
while they have headroom; then static batching hits its ceiling (drain barriers
+ batch decay leave the accelerator idle) while continuous batching keeps packing
the decode batch and sustains a markedly higher maximum throughput.

The "ceiling" is the mean throughput across the saturated tail of the sweep
(offered load well above capacity), not a transient peak.
"""
from __future__ import annotations

from typing import Dict

from mini_vllm import (CacheConfig, LLMEngine, ModelConfig, SchedulerConfig,
                       workloads)

RATES = [5, 10, 15, 20, 25, 30, 40, 60, 80]
SATURATED = [30, 40, 60, 80]      # offered load past capacity -> steady state
N_REQUESTS = 300


def _run(policy: str, rate: float) -> float:
    e = LLMEngine(
        CacheConfig(block_size=16, num_gpu_blocks=600),
        SchedulerConfig(policy=policy, max_num_seqs=128, max_num_batched_tokens=2048),
        ModelConfig(),
    )
    e.add_requests(workloads.poisson(n=N_REQUESTS, rate_rps=rate,
                                     prompt_mean=256, gen_mean=128, seed=7))
    return e.run().throughput_tok_s


def _ceiling(by_rate: Dict[int, float]) -> float:
    vals = [by_rate[r] for r in SATURATED]
    return sum(vals) / len(vals)


def run() -> Dict:
    static = {r: _run("static", r) for r in RATES}
    cont = {r: _run("continuous", r) for r in RATES}
    sc, cc = _ceiling(static), _ceiling(cont)
    return {
        "rates": RATES,
        "n_requests": N_REQUESTS,
        "static_tok_s": [round(static[r], 1) for r in RATES],
        "continuous_tok_s": [round(cont[r], 1) for r in RATES],
        "static_ceiling_tok_s": round(sc, 1),
        "continuous_ceiling_tok_s": round(cc, 1),
        "ceiling_speedup": round(cc / sc, 2),
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, GREEN, MUTED
    plt = apply()

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = data["rates"]
    ax.plot(x, data["static_tok_s"], "o-", color=MUTED, label="static batching")
    ax.plot(x, data["continuous_tok_s"], "o-", color=GREEN, label="continuous batching")

    sc, cc = data["static_ceiling_tok_s"], data["continuous_ceiling_tok_s"]
    ax.axhline(sc, color=MUTED, ls=":", lw=1)
    ax.axhline(cc, color=GREEN, ls=":", lw=1)
    ax.annotate(f"static ceiling ≈ {sc:.0f} tok/s", (x[-1], sc),
                textcoords="offset points", xytext=(-6, -14), ha="right",
                color=MUTED, fontsize=8.5)
    ax.annotate(f"continuous ceiling ≈ {cc:.0f} tok/s", (x[-1], cc),
                textcoords="offset points", xytext=(-6, 6), ha="right",
                color=GREEN, fontsize=8.5)

    ax.set_xlabel("offered load (requests / s)")
    ax.set_ylabel("sustained throughput (tokens / s)")
    ax.set_title(
        f"Continuous batching sustains {data['ceiling_speedup']}x the throughput at saturation",
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="lower right")
    ax.margins(y=0.12)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os
    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/throughput.png")
