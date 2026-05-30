"""Paged vs contiguous KV allocation: how many sequences fit, and how much KV
memory is wasted. Pure analysis (no engine run), so it's exact and instant.
"""
from __future__ import annotations

import random
from typing import Dict

from mini_vllm import contiguous_capacity, paged_capacity

BLOCK_SIZE = 16
NUM_BLOCKS = 2048                 # 2048 * 16 = 32,768 token-slots of KV
TOTAL_SLOTS = NUM_BLOCKS * BLOCK_SIZE
MAX_SEQ_LEN = 2048


def _lengths(n: int = 4000, mean: int = 240, sigma: float = 0.6, seed: int = 0):
    import math
    rng = random.Random(seed)
    mu = math.log(mean)
    return [max(8, min(MAX_SEQ_LEN, int(rng.lognormvariate(mu, sigma)))) for _ in range(n)]


def run() -> Dict:
    lens = _lengths()
    cont = contiguous_capacity(TOTAL_SLOTS, MAX_SEQ_LEN, lens)
    paged = paged_capacity(NUM_BLOCKS, BLOCK_SIZE, lens)

    # how capacity scales as you support longer max context
    sweep_max = [512, 1024, 2048, 4096, 8192]
    cont_fit, paged_fit = [], []
    for m in sweep_max:
        ls = [min(x, m) for x in lens]
        cont_fit.append(contiguous_capacity(TOTAL_SLOTS, m, ls).fit)
        paged_fit.append(paged_capacity(NUM_BLOCKS, BLOCK_SIZE, ls).fit)

    return {
        "config": {"block_size": BLOCK_SIZE, "num_blocks": NUM_BLOCKS,
                   "total_slots": TOTAL_SLOTS, "max_seq_len": MAX_SEQ_LEN,
                   "mean_len": 240},
        "contiguous_fit": cont.fit,
        "paged_fit": paged.fit,
        "fit_speedup": round(paged.fit / max(1, cont.fit), 2),
        "contiguous_util": round(cont.utilization, 4),
        "paged_util": round(paged.utilization, 4),
        "contiguous_waste_frac": round(cont.waste_fraction, 4),
        "paged_waste_frac": round(paged.waste_fraction, 4),
        "sweep_max_seq_len": sweep_max,
        "sweep_contiguous_fit": cont_fit,
        "sweep_paged_fit": paged_fit,
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, CYAN, GREEN, MUTED
    plt = apply()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # Panel A: sequences that fit + KV utilization
    labels = ["contiguous\n(reserve max)", "paged\n(on demand)"]
    fits = [data["contiguous_fit"], data["paged_fit"]]
    bars = ax1.bar(labels, fits, color=[MUTED, GREEN], width=0.6)
    ax1.set_ylabel("concurrent sequences that fit")
    ax1.set_title(f"Same KV budget, {data['fit_speedup']}x more sequences", fontsize=11)
    for b, f, u in zip(bars, fits, [data["contiguous_util"], data["paged_util"]]):
        ax1.text(b.get_x() + b.get_width() / 2, f, f"{f}\n{u*100:.0f}% util",
                 ha="center", va="bottom", fontsize=9)
    ax1.margins(y=0.18)

    # Panel B: capacity vs max context length
    x = data["sweep_max_seq_len"]
    ax2.plot(x, data["sweep_contiguous_fit"], "o-", color=MUTED, label="contiguous")
    ax2.plot(x, data["sweep_paged_fit"], "o-", color=CYAN, label="paged")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(x); ax2.set_xticklabels([str(v) for v in x])
    ax2.set_xlabel("supported max context length (tokens)")
    ax2.set_ylabel("concurrent sequences that fit")
    ax2.set_title("Contiguous collapses as context grows", fontsize=11)
    ax2.legend(frameon=False)

    fig.suptitle("Paged KV cache  ·  no external fragmentation",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os
    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/mem_capacity.png")
