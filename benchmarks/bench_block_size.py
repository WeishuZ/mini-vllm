"""Block-size sweep: internal fragmentation vs capacity.

Paged KV avoids external fragmentation, but the final partially-filled block of
each sequence is still internal fragmentation. Larger blocks reduce metadata
overhead in real systems, but waste more token slots in this simplified model.
"""
from __future__ import annotations

import random
from typing import Dict

from mini_vllm import paged_capacity

TOTAL_SLOTS = 32_768
BLOCK_SIZES = [4, 8, 16, 32, 64]


def _lengths(n: int = 4000, mean: int = 240, sigma: float = 0.6, seed: int = 3):
    import math

    rng = random.Random(seed)
    mu = math.log(mean)
    return [max(8, min(2048, int(rng.lognormvariate(mu, sigma)))) for _ in range(n)]


def run() -> Dict:
    lens = _lengths()
    by_size = {}
    for block_size in BLOCK_SIZES:
        cap = paged_capacity(TOTAL_SLOTS // block_size, block_size, lens)
        by_size[block_size] = cap
    best_fit = max(by_size.values(), key=lambda r: r.fit)
    lowest_waste = min(by_size.values(), key=lambda r: r.waste_fraction)
    return {
        "block_sizes": BLOCK_SIZES,
        "fit": [by_size[b].fit for b in BLOCK_SIZES],
        "utilization": [round(by_size[b].utilization, 4) for b in BLOCK_SIZES],
        "waste_fraction": [round(by_size[b].waste_fraction, 4) for b in BLOCK_SIZES],
        "best_fit": best_fit.fit,
        "lowest_waste_fraction": round(lowest_waste.waste_fraction, 4),
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, CYAN, GREEN

    plt = apply()
    fig, ax1 = plt.subplots(figsize=(7.2, 4.1))
    x = data["block_sizes"]
    ax1.plot(x, data["fit"], "o-", color=GREEN, label="sequences fit")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in x])
    ax1.set_xlabel("block size (tokens)")
    ax1.set_ylabel("sequences that fit")

    ax2 = ax1.twinx()
    ax2.plot(x, [w * 100 for w in data["waste_fraction"]], "o-", color=CYAN, label="waste")
    ax2.set_ylabel("internal fragmentation (%)")
    ax2.tick_params(colors=CYAN)
    ax2.spines["right"].set_color(CYAN)

    ax1.set_title("Block size trades metadata granularity for tail waste", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os

    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/block_size.png")

