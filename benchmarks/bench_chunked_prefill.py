"""Chunked prefill sweep with prompts larger than the per-step token budget."""
from __future__ import annotations

from typing import Dict, List

from mini_vllm import CacheConfig, LLMEngine, ModelConfig, Request, SchedulerConfig

PROMPT_LENS = [256, 512, 1024]
N_REQUESTS = 36
TOKEN_BUDGET = 512


def _requests(prompt_len: int) -> List[Request]:
    return [
        Request(f"r{i}", prompt_len=prompt_len, max_tokens=64, arrival=i * 12.0)
        for i in range(N_REQUESTS)
    ]


def _run(prompt_len: int, chunked: bool) -> Dict:
    e = LLMEngine(
        CacheConfig(block_size=16, num_gpu_blocks=900),
        SchedulerConfig(
            max_num_seqs=24,
            max_num_batched_tokens=TOKEN_BUDGET,
            enable_chunked_prefill=chunked,
        ),
        ModelConfig(),
    )
    e.add_requests(_requests(prompt_len))
    m = e.run()
    return {
        "completed": m.num_completed,
        "ttft_p50": m.ttft_ms_p50,
        "ttft_p99": m.ttft_ms_p99,
        "throughput": m.throughput_tok_s,
        "prefill_tokens": m.total_prefill_tokens,
    }


def run() -> Dict:
    on = {p: _run(p, True) for p in PROMPT_LENS}
    off = {p: _run(p, False) for p in PROMPT_LENS}
    return {
        "prompt_lens": PROMPT_LENS,
        "token_budget": TOKEN_BUDGET,
        "chunked_completed": [on[p]["completed"] for p in PROMPT_LENS],
        "unchunked_completed": [off[p]["completed"] for p in PROMPT_LENS],
        "chunked_ttft_p50": [round(on[p]["ttft_p50"], 1) for p in PROMPT_LENS],
        "unchunked_ttft_p50": [round(off[p]["ttft_p50"], 1) for p in PROMPT_LENS],
        "chunked_throughput": [round(on[p]["throughput"], 1) for p in PROMPT_LENS],
        "unchunked_throughput": [round(off[p]["throughput"], 1) for p in PROMPT_LENS],
    }


def plot(data: Dict, path: str) -> None:
    from _style import apply, CYAN, GREEN, MUTED, RED

    plt = apply()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))
    x = data["prompt_lens"]
    ax1.plot(x, data["chunked_completed"], "o-", color=GREEN, label="chunked")
    ax1.plot(x, data["unchunked_completed"], "o-", color=RED, label="unchunked")
    ax1.set_xscale("log", base=2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in x])
    ax1.set_xlabel("prompt length")
    ax1.set_ylabel("completed requests")
    ax1.set_title("Progress with fixed token budget", fontsize=10.5)
    ax1.legend(frameon=False)

    ax2.plot(x, data["chunked_ttft_p50"], "o-", color=CYAN, label="chunked")
    ax2.plot(x, data["unchunked_ttft_p50"], "o-", color=MUTED, label="unchunked")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(v) for v in x])
    ax2.set_xlabel("prompt length")
    ax2.set_ylabel("TTFT p50 (ms)")
    ax2.set_title("Chunking keeps long prefills schedulable", fontsize=10.5)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    print(f"wrote {path}")


if __name__ == "__main__":
    import json, os

    d = run()
    print(json.dumps(d, indent=2))
    os.makedirs("docs/assets", exist_ok=True)
    plot(d, "docs/assets/chunked_prefill.png")

