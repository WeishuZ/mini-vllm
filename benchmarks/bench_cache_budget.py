"""Prefix-cache budget sweep."""
from __future__ import annotations

from typing import Dict

from mini_vllm import CacheConfig, LLMEngine, ModelConfig, Request, SchedulerConfig

BUDGETS = [0, 16, 32, 64, 256]
SYSTEM_LEN = 512
N_REQUESTS = 48
USER_TOKENS = 8


def _requests():
    system = list(range(1, SYSTEM_LEN + 1))
    return [
        Request(
            f"r{i}",
            prompt_len=SYSTEM_LEN + USER_TOKENS,
            max_tokens=96,
            # Space requests far enough apart that reuse depends on the
            # resident prefix cache, not just overlap with an active leader.
            arrival=i * 900.0,
            token_ids=system + [20_000 + i] * USER_TOKENS,
        )
        for i in range(N_REQUESTS)
    ]


def _run(cache_budget: int) -> Dict:
    e = LLMEngine(
        CacheConfig(
            block_size=16,
            num_gpu_blocks=1200,
            enable_prefix_caching=True,
            prefix_cache_max_blocks=cache_budget,
        ),
        SchedulerConfig(max_num_seqs=4, max_num_batched_tokens=2048),
        ModelConfig(),
    )
    e.add_requests(_requests())
    m = e.run()
    return {
        "hit_rate": m.prefix_cache_hit_rate,
        "saved_tokens": m.prefix_cache_saved_tokens,
        "evictions": m.prefix_cache_evictions,
        "ttft_p50": m.ttft_ms_p50,
        "cache_blocks": m.prefix_cache_blocks,
    }


def run() -> Dict:
    rows = {b: _run(b) for b in BUDGETS}
    return {
        "cache_budgets": BUDGETS,
        "hit_rate": [round(rows[b]["hit_rate"], 3) for b in BUDGETS],
        "saved_tokens": [rows[b]["saved_tokens"] for b in BUDGETS],
        "evictions": [rows[b]["evictions"] for b in BUDGETS],
        "ttft_p50": [round(rows[b]["ttft_p50"], 1) for b in BUDGETS],
        "cache_blocks": [rows[b]["cache_blocks"] for b in BUDGETS],
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, AMBER, GREEN

    plt = apply()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.1))
    x = data["cache_budgets"]
    ax1.plot(x, [h * 100 for h in data["hit_rate"]], "o-", color=GREEN, label="hit rate")
    ax1.set_xlabel("prefix cache budget (blocks)")
    ax1.set_ylabel("block hit rate (%)")

    ax2 = ax1.twinx()
    ax2.plot(x, data["evictions"], "o-", color=AMBER, label="evictions")
    ax2.set_ylabel("evictions")
    ax2.tick_params(colors=AMBER)
    ax2.spines["right"].set_color(AMBER)

    ax1.set_title("Prefix-cache budget controls reuse vs eviction", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os

    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/cache_budget.png")
