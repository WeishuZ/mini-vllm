"""Run every benchmark, write plots to docs/assets/ and a machine-readable
docs/results.json. Reproducible: `python benchmarks/run_all.py`.
"""
from __future__ import annotations

import json
import os
import sys

# allow running both as `python benchmarks/run_all.py` and `-m`
sys.path.insert(0, os.path.dirname(__file__))

import bench_latency
import bench_memory
import bench_prefix
import bench_throughput

ASSETS = "docs/assets"


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    results = {}

    print("\n[1/4] memory: paged vs contiguous")
    results["memory"] = bench_memory.run()
    bench_memory.plot(results["memory"], f"{ASSETS}/mem_capacity.png")

    print("\n[2/4] throughput: continuous vs static (saturated)")
    results["throughput"] = bench_throughput.run()
    bench_throughput.plot(results["throughput"], f"{ASSETS}/throughput.png")

    print("\n[3/4] latency: TTFT under Poisson load")
    results["latency"] = bench_latency.run()
    bench_latency.plot(results["latency"], f"{ASSETS}/latency.png")

    print("\n[4/4] prefix cache: shared system prompt")
    results["prefix"] = bench_prefix.run()
    bench_prefix.plot(results["prefix"], f"{ASSETS}/prefix_cache.png")

    with open("docs/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n================ headline numbers ================")
    m, t, l, p = (results["memory"], results["throughput"],
                  results["latency"], results["prefix"])
    print(f"  paged KV:        {m['fit_speedup']}x more sequences vs contiguous "
          f"({m['paged_util']*100:.0f}% vs {m['contiguous_util']*100:.0f}% KV utilization)")
    print(f"  continuous batch:{t['ceiling_speedup']}x sustained throughput at saturation "
          f"({t['continuous_ceiling_tok_s']:.0f} vs {t['static_ceiling_tok_s']:.0f} tok/s)")
    print(f"  TTFT ({l['headline_rate']} req/s): {l['headline_p99_improvement']:.0f}x lower p99 "
          f"({l['headline_continuous_p99']:.0f} vs {l['headline_static_p99']:.0f} ms)")
    print(f"  prefix cache:    {p['headline_prefill_reduction']*100:.0f}% less prefill, "
          f"{p['headline_ttft_reduction']*100:.0f}% lower TTFT "
          f"({p['headline_hit_rate']*100:.0f}% block hit-rate)")
    print("  wrote docs/results.json and docs/assets/*.png")


if __name__ == "__main__":
    main()
