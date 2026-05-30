"""Prefix caching on a shared-system-prompt workload (the RAG / agent / chat
pattern). Requests share an identical leading system prompt; with prefix caching
those blocks are computed once and shared, so prefill compute and TTFT drop.
"""
from __future__ import annotations

from typing import Dict

from mini_vllm import CacheConfig, LLMEngine, ModelConfig, Request, SchedulerConfig

N_REQUESTS = 64
SYSTEM_LENS = [128, 256, 512, 1024]
USER_TOKENS = 8
GEN = 160


def _make(system_len: int):
    system = list(range(1, system_len + 1))
    return [
        Request(f"r{i}", prompt_len=system_len + USER_TOKENS, max_tokens=GEN,
                arrival=i * 20.0, token_ids=system + [10_000 + i] * USER_TOKENS)
        for i in range(N_REQUESTS)
    ]


def _run(enable: bool, system_len: int) -> Dict:
    e = LLMEngine(
        CacheConfig(block_size=16, num_gpu_blocks=2000, enable_prefix_caching=enable),
        SchedulerConfig(max_num_seqs=24, max_num_batched_tokens=2048),
        ModelConfig(),
    )
    for r in _make(system_len):
        e.add_request(r)
    m = e.run()
    return {"prefill_tokens": e.total_prefill_tokens, "ttft_p50": m.ttft_ms_p50,
            "throughput": m.throughput_tok_s, "hit": m.prefix_cache_hit_rate}


def run() -> Dict:
    off = [_run(False, s) for s in SYSTEM_LENS]
    on = [_run(True, s) for s in SYSTEM_LENS]
    j = SYSTEM_LENS.index(512)
    return {
        "system_lens": SYSTEM_LENS,
        "n_requests": N_REQUESTS,
        "off_prefill_tokens": [o["prefill_tokens"] for o in off],
        "on_prefill_tokens": [o["prefill_tokens"] for o in on],
        "off_ttft_p50": [round(o["ttft_p50"], 1) for o in off],
        "on_ttft_p50": [round(o["ttft_p50"], 1) for o in on],
        "on_hit_rate": [round(o["hit"], 3) for o in on],
        "headline_system_len": 512,
        "headline_hit_rate": round(on[j]["hit"], 3),
        "headline_prefill_reduction": round(1 - on[j]["prefill_tokens"] / off[j]["prefill_tokens"], 3),
        "headline_ttft_reduction": round(1 - on[j]["ttft_p50"] / off[j]["ttft_p50"], 3),
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, GREEN, MUTED, CYAN
    plt = apply()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    x = data["system_lens"]
    xs = [str(v) for v in x]

    width = 0.38
    pos = list(range(len(x)))
    ax1.bar([p - width / 2 for p in pos], [t / 1000 for t in data["off_prefill_tokens"]],
            width, color=MUTED, label="cache off")
    ax1.bar([p + width / 2 for p in pos], [t / 1000 for t in data["on_prefill_tokens"]],
            width, color=GREEN, label="cache on")
    ax1.set_xticks(pos); ax1.set_xticklabels(xs)
    ax1.set_xlabel("shared system-prompt length (tokens)")
    ax1.set_ylabel("prefill tokens computed (thousands)")
    ax1.set_title("Prefill work avoided", fontsize=11)
    ax1.legend(frameon=False)

    ax2.plot(x, data["off_ttft_p50"], "o-", color=MUTED, label="cache off")
    ax2.plot(x, data["on_ttft_p50"], "o-", color=CYAN, label="cache on")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(x); ax2.set_xticklabels(xs)
    ax2.set_xlabel("shared system-prompt length (tokens)")
    ax2.set_ylabel("median TTFT (ms)")
    ax2.set_title("Time to first token", fontsize=11)
    ax2.legend(frameon=False)

    hr = data["headline_hit_rate"] * 100
    fig.suptitle(f"Prefix caching  ·  {hr:.0f}% block hit-rate @ 512-token system prompt",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os
    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/prefix_cache.png")
