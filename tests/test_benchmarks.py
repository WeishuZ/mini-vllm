import os
import sys

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks")
sys.path.insert(0, BENCH_DIR)

import bench_block_size
import bench_cache_budget
import bench_chunked_prefill
import bench_preemption
import bench_watermark


def test_new_benchmark_runs_smoke():
    block = bench_block_size.run()
    water = bench_watermark.run()
    chunked = bench_chunked_prefill.run()
    preemption = bench_preemption.run()
    cache = bench_cache_budget.run()

    assert block["block_sizes"]
    assert len(water["watermarks"]) == len(water["preemptions"])
    assert chunked["chunked_completed"][-1] > chunked["unchunked_completed"][-1]
    assert preemption["completed"] == [30, 30]
    assert max(cache["hit_rate"]) > 0
