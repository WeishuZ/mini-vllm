"""Generate a standalone HTML trace viewer for mini-vLLM.

The viewer is intentionally dependency-free: it embeds a deterministic run as
JSON and renders the scheduler queues, token work, KV-block grid, and memory
events in the browser. This makes it easy to publish from ``docs/`` or attach
to a resume/portfolio without running a backend.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import CacheConfig, ModelConfig, SchedulerConfig
from .engine import LLMEngine
from .request import Request, Sequence
from .scheduler import StepWork


DEFAULT_OUTPUT = Path("docs/trace.html")


def trace_workload(
    n: int = 44,
    system_len: int = 128,
    seed: int = 17,
) -> List[Request]:
    """A deterministic workload tuned to show the interesting mechanisms.

    One leader arrives first and warms the prefix cache. The rest arrive shortly
    after with the same system prompt, then compete for a deliberately small KV
    pool so preemption becomes visible.
    """
    rng = random.Random(seed)
    system = list(range(1, system_len + 1))
    requests: List[Request] = []
    for i in range(n):
        user_len = 16 + (i * 7) % 44
        user = [10_000 + i * 200 + rng.randrange(200) for _ in range(user_len)]
        prompt = system + user
        gen_len = 44 + (i * 13) % 96
        arrival = 0.0 if i == 0 else 28.0 + (i % 9) * 3.5 + (i // 9) * 4.0
        requests.append(
            Request(
                request_id=f"r{i}",
                prompt_len=len(prompt),
                max_tokens=gen_len,
                arrival=arrival,
                token_ids=prompt,
            )
        )
    return requests


def build_demo_engine(
    n_requests: int = 44,
    enable_prefix_caching: bool = True,
    preemption_mode: str = "recompute",
) -> LLMEngine:
    cache = CacheConfig(
        block_size=16,
        num_gpu_blocks=128,
        num_cpu_blocks=512,
        enable_prefix_caching=enable_prefix_caching,
    )
    scheduler = SchedulerConfig(
        max_num_seqs=22,
        max_num_batched_tokens=512,
        enable_chunked_prefill=True,
        policy="continuous",
        preemption_mode=preemption_mode,
        watermark=0.04,
    )
    engine = LLMEngine(cache, scheduler, ModelConfig())
    engine.add_requests(trace_workload(n=n_requests))
    return engine


def _ids(seqs: Iterable[Sequence]) -> List[str]:
    return [s.request_id for s in seqs]


def _seq_sort_key(seq: Sequence) -> tuple:
    try:
        suffix = int(seq.request_id.lstrip("r"))
    except ValueError:
        suffix = 0
    return (seq.arrival, suffix, seq.request_id)


def _sequence_rows(engine: LLMEngine) -> List[Dict]:
    pending_ids = {s.request_id for s in engine._pending}
    rows = []
    for seq in sorted(engine.sequences.values(), key=_seq_sort_key):
        status = "pending" if seq.request_id in pending_ids else seq.status.value
        rows.append(
            {
                "id": seq.request_id,
                "status": status,
                "arrival": round(seq.arrival, 1),
                "queue_ms": (
                    round(seq.first_scheduled_time - seq.arrival, 1)
                    if seq.first_scheduled_time is not None
                    else None
                ),
                "prompt": seq.prompt_len,
                "generated": seq.num_generated,
                "max": seq.max_tokens,
                "computed": seq.num_computed,
                "cached": seq.num_cached_tokens,
                "blocks": len(seq.block_table),
                "preemptions": seq.num_preemptions,
            }
        )
    return rows


def _block_rows(engine: LLMEngine) -> List[Dict]:
    owners_by_block: Dict[int, List[str]] = {}
    for seq in engine.sequences.values():
        for block_id in seq.block_table:
            owners_by_block.setdefault(block_id, []).append(seq.request_id)

    cached_blocks = set(engine.block_manager._block_to_hash.keys())
    rows = []
    for block_id in range(engine.block_manager.num_gpu_blocks):
        owners = sorted(owners_by_block.get(block_id, []))
        cached = block_id in cached_blocks
        rows.append(
            {
                "id": block_id,
                "owners": owners,
                "cached": cached,
                "used": bool(owners) or cached,
                "shared": len(owners) > 1,
            }
        )
    return rows


def _work_dict(work: Optional[StepWork]) -> Dict:
    if work is None:
        return {
            "prefill": [],
            "decode": [],
            "prefill_tokens": 0,
            "decode_tokens": 0,
            "preempted": [],
            "swapped_in": [],
            "swapped_out": [],
            "prefix_hits": [],
        }
    return {
        "prefill": [
            {"id": seq.request_id, "tokens": tokens}
            for seq, tokens in work.prefill
        ],
        "decode": _ids(work.decode),
        "prefill_tokens": work.num_prefill_tokens,
        "decode_tokens": work.num_decode_seqs,
        "preempted": list(work.preempted_seq_ids),
        "swapped_in": list(work.swapped_in_seq_ids),
        "swapped_out": list(work.swapped_out_seq_ids),
        "prefix_hits": [
            {"id": rid, "tokens": tokens, "blocks": blocks}
            for rid, tokens, blocks in work.prefix_hits
        ],
    }


def _frame(engine: LLMEngine, work: Optional[StepWork]) -> Dict:
    pending = sorted(engine._pending, key=_seq_sort_key)
    completed = sorted(engine.completed, key=_seq_sort_key)
    queues = {
        "pending": _ids(pending),
        "waiting": _ids(engine.scheduler.waiting),
        "running": _ids(engine.scheduler.running),
        "swapped": _ids(engine.scheduler.swapped),
        "finished": _ids(completed),
    }
    return {
        "step": engine.num_steps,
        "time_ms": round(engine.clock_ms, 2),
        "queues": queues,
        "counts": {name: len(values) for name, values in queues.items()},
        "gpu_used_blocks": engine.block_manager.num_used_gpu_blocks,
        "gpu_total_blocks": engine.block_manager.num_gpu_blocks,
        "gpu_util": round(engine.block_manager.gpu_utilization, 4),
        "cache_hit_rate": round(engine.metrics().prefix_cache_hit_rate, 4),
        "work": _work_dict(work),
        "requests": _sequence_rows(engine),
        "blocks": _block_rows(engine),
    }


def build_trace(engine: Optional[LLMEngine] = None, max_steps: int = 180) -> Dict:
    """Run an engine and return serializable trace data."""
    engine = engine or build_demo_engine()
    engine._pending.sort(key=lambda s: s.arrival)
    frames: List[Dict] = []
    consecutive_empty = 0

    while (engine._pending or engine.scheduler.has_unfinished) and len(frames) < max_steps:
        engine._release_arrivals()
        if not engine.scheduler.has_unfinished:
            if engine._pending:
                engine.clock_ms = max(engine.clock_ms, engine._pending[0].arrival)
                continue
            break

        work = engine.step()
        frames.append(_frame(engine, work))
        if work.is_empty:
            consecutive_empty += 1
            if consecutive_empty > 3:
                break
        else:
            consecutive_empty = 0

    metrics = engine.metrics()
    return {
        "metadata": {
            "title": "mini-vLLM scheduler trace",
            "description": (
                "Paged KV-cache, continuous batching, preemption, and prefix "
                "caching in one deterministic run."
            ),
            "max_steps": max_steps,
            "captured_steps": len(frames),
            "block_size": engine.cache_config.block_size,
            "token_budget": engine.scheduler_config.max_num_batched_tokens,
            "max_num_seqs": engine.scheduler_config.max_num_seqs,
            "preemption_mode": engine.scheduler_config.preemption_mode,
            "prefix_caching": engine.cache_config.enable_prefix_caching,
            "prefix_cache_max_blocks": engine.block_manager.prefix_cache_max_blocks,
        },
        "summary": metrics.as_dict(),
        "frames": frames,
    }


def render_html(trace: Dict) -> str:
    payload = json.dumps(trace, separators=(",", ":")).replace("</", "<\\/")
    return HTML_TEMPLATE.replace("__TRACE_JSON__", payload)


def write_trace_html(trace: Dict, path: Path = DEFAULT_OUTPUT) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_html(trace), encoding="utf-8")
    return path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Generate docs/trace.html")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--requests", type=int, default=44)
    parser.add_argument(
        "--preemption-mode",
        choices=("recompute", "swap"),
        default="recompute",
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="Disable prefix caching. Useful with --preemption-mode swap.",
    )
    args = parser.parse_args(argv)

    engine = build_demo_engine(
        n_requests=args.requests,
        enable_prefix_caching=not args.no_prefix_cache,
        preemption_mode=args.preemption_mode,
    )
    path = write_trace_html(build_trace(engine, max_steps=args.steps), args.output)
    print(f"wrote {path}")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>mini-vLLM scheduler trace</title>
<style>
:root {
  --bg: #f6f7f9;
  --panel: #ffffff;
  --ink: #18202f;
  --muted: #667085;
  --line: #d9dee8;
  --soft: #edf1f6;
  --blue: #2563eb;
  --cyan: #0891b2;
  --green: #0f8b6f;
  --amber: #b45309;
  --red: #be123c;
  --violet: #6d28d9;
  --shadow: 0 14px 34px rgba(24, 32, 47, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
    "Segoe UI", sans-serif;
  line-height: 1.45;
}
button, input { font: inherit; }
.app {
  min-height: 100vh;
  display: grid;
  grid-template-rows: auto auto 1fr;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 24px;
  padding: 22px 28px 14px;
  border-bottom: 1px solid var(--line);
  background: rgba(255, 255, 255, 0.82);
  backdrop-filter: blur(16px);
  position: sticky;
  top: 0;
  z-index: 5;
}
.title h1 {
  margin: 0;
  font-size: 24px;
  line-height: 1.1;
  font-weight: 750;
  letter-spacing: 0;
}
.title p {
  margin: 6px 0 0;
  color: var(--muted);
  max-width: 760px;
  font-size: 13px;
}
.controls {
  display: grid;
  grid-template-columns: auto auto auto minmax(180px, 320px);
  gap: 8px;
  align-items: center;
}
.icon-button {
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  width: 54px;
  height: 34px;
  border-radius: 7px;
  cursor: pointer;
  box-shadow: 0 1px 0 rgba(24, 32, 47, 0.04);
}
.icon-button:hover { border-color: #aab4c5; }
.step-range {
  width: 100%;
  accent-color: var(--blue);
}
.statusbar {
  display: grid;
  grid-template-columns: repeat(6, minmax(110px, 1fr));
  gap: 1px;
  background: var(--line);
  border-bottom: 1px solid var(--line);
}
.stat {
  background: var(--panel);
  padding: 12px 18px;
  min-width: 0;
}
.stat label {
  display: block;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.stat strong {
  display: block;
  margin-top: 3px;
  font-size: 18px;
  line-height: 1.15;
}
.main {
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(360px, 0.75fr);
  gap: 18px;
  padding: 18px;
}
.left-stack,
.right-stack {
  min-width: 0;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  box-shadow: var(--shadow);
  min-width: 0;
}
.panel h2 {
  margin: 0;
  padding: 14px 16px 0;
  font-size: 14px;
  line-height: 1.2;
}
.panel-sub {
  color: var(--muted);
  font-size: 12px;
  padding: 4px 16px 0;
}
.charts {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: 18px;
}
.chart svg {
  width: 100%;
  height: 176px;
  display: block;
  padding: 12px 14px 16px;
}
.chart text {
  fill: var(--muted);
  font-size: 10px;
}
.chart .axis { stroke: var(--line); stroke-width: 1; }
.chart .kv-line { fill: none; stroke: var(--green); stroke-width: 2.4; }
.chart .prefill { fill: rgba(37, 99, 235, 0.72); }
.chart .decode { fill: rgba(8, 145, 178, 0.72); }
.chart .cursor { stroke: var(--red); stroke-width: 1.5; }
.block-grid {
  display: grid;
  grid-template-columns: repeat(32, minmax(0, 1fr));
  gap: 4px;
  padding: 14px 16px 16px;
}
.block {
  aspect-ratio: 1 / 1;
  border-radius: 3px;
  background: var(--soft);
  border: 1px solid #dbe2ed;
  min-width: 0;
}
.block.used { background: #9db7f6; border-color: #6f93e8; }
.block.cached { background: #7dd3c7; border-color: #35a99b; }
.block.shared { background: #c4b5fd; border-color: #8b5cf6; }
.block.hot { outline: 2px solid var(--red); outline-offset: 1px; }
.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 0 16px 14px;
  color: var(--muted);
  font-size: 12px;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.swatch {
  width: 12px;
  height: 12px;
  border-radius: 3px;
  border: 1px solid var(--line);
  display: inline-block;
}
.queues {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
  padding: 14px 16px 16px;
}
.queue h3 {
  margin: 0 0 8px;
  font-size: 12px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  min-height: 34px;
  align-content: flex-start;
}
.chip {
  border: 1px solid var(--line);
  background: #f8fafc;
  border-radius: 6px;
  padding: 3px 6px;
  font-size: 11px;
  color: var(--ink);
}
.chip.running { border-color: rgba(37, 99, 235, 0.45); background: #eff5ff; }
.chip.finished { border-color: rgba(15, 139, 111, 0.36); background: #eefaf6; }
.chip.swapped { border-color: rgba(180, 83, 9, 0.42); background: #fff7ed; }
.event-list {
  padding: 12px 16px 16px;
  display: grid;
  gap: 8px;
}
.event {
  display: grid;
  grid-template-columns: 86px 1fr;
  gap: 10px;
  align-items: start;
  border-top: 1px solid var(--line);
  padding-top: 8px;
  font-size: 12px;
}
.event:first-child { border-top: 0; padding-top: 0; }
.event strong {
  color: var(--ink);
  font-size: 12px;
}
.event span { color: var(--muted); }
.table-wrap {
  max-height: 430px;
  overflow: auto;
  border-top: 1px solid var(--line);
  margin-top: 10px;
  min-width: 0;
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
th, td {
  padding: 8px 9px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  white-space: nowrap;
}
th {
  position: sticky;
  top: 0;
  background: #f8fafc;
  color: var(--muted);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  z-index: 1;
}
.status {
  font-weight: 650;
}
.status.running { color: var(--blue); }
.status.finished { color: var(--green); }
.status.swapped { color: var(--amber); }
.status.waiting { color: var(--violet); }
.status.pending { color: var(--muted); }
.right-stack {
  display: grid;
  gap: 18px;
  align-content: start;
}
.empty {
  color: var(--muted);
  font-size: 12px;
}
@media (max-width: 1020px) {
  .topbar { align-items: stretch; flex-direction: column; }
  .controls { grid-template-columns: auto auto auto 1fr; }
  .statusbar { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .main { grid-template-columns: 1fr; }
}
@media (max-width: 680px) {
  .topbar { padding: 18px 14px 12px; }
  .main { padding: 12px; }
  .statusbar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .charts { grid-template-columns: 1fr; }
  .queues { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .block-grid { grid-template-columns: repeat(16, minmax(0, 1fr)); }
}
</style>
</head>
<body>
<div class="app">
  <header class="topbar">
    <div class="title">
      <h1>mini-vLLM scheduler trace</h1>
      <p>Step through continuous batching, paged KV-cache pressure, recompute preemption, and prefix-cache block sharing in a deterministic run.</p>
    </div>
    <div class="controls" aria-label="Trace controls">
      <button class="icon-button" id="prev" title="Previous step" aria-label="Previous step">&lt;</button>
      <button class="icon-button" id="play" title="Play trace" aria-label="Play trace">Play</button>
      <button class="icon-button" id="next" title="Next step" aria-label="Next step">&gt;</button>
      <input class="step-range" id="slider" type="range" min="0" max="0" value="0" aria-label="Trace step">
    </div>
  </header>
  <section class="statusbar" id="statusbar"></section>
  <main class="main">
    <section class="left-stack">
      <div class="charts">
        <section class="panel chart">
          <h2>KV utilization over time</h2>
          <div class="panel-sub">Used GPU blocks as a fraction of the block pool.</div>
          <svg id="kvChart" viewBox="0 0 420 176" role="img" aria-label="KV utilization chart"></svg>
        </section>
        <section class="panel chart">
          <h2>Token work per step</h2>
          <div class="panel-sub">Prefill tokens and decode tokens scheduled under the per-step budget.</div>
          <svg id="workChart" viewBox="0 0 420 176" role="img" aria-label="Token work chart"></svg>
        </section>
      </div>
      <section class="panel">
        <h2>GPU KV block pool</h2>
        <div class="panel-sub">Each square is one physical block. Shared/cached blocks are the prefix-cache story.</div>
        <div class="block-grid" id="blockGrid"></div>
        <div class="legend">
          <span><i class="swatch"></i>free</span>
          <span><i class="swatch" style="background:#9db7f6;border-color:#6f93e8"></i>owned</span>
          <span><i class="swatch" style="background:#7dd3c7;border-color:#35a99b"></i>cached</span>
          <span><i class="swatch" style="background:#c4b5fd;border-color:#8b5cf6"></i>shared</span>
          <span><i class="swatch" style="outline:2px solid #be123c;outline-offset:1px"></i>touched this step</span>
        </div>
      </section>
      <section class="panel">
        <h2>Queues by timestep</h2>
        <div class="queues" id="queues"></div>
      </section>
      <section class="panel">
        <h2>Requests</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>id</th><th>status</th><th>arrival</th><th>queue</th><th>prompt</th>
                <th>gen</th><th>computed</th><th>cached</th><th>blocks</th><th>preempt</th>
              </tr>
            </thead>
            <tbody id="requestRows"></tbody>
          </table>
        </div>
      </section>
    </section>
    <aside class="right-stack">
      <section class="panel">
        <h2>Step events</h2>
        <div class="event-list" id="events"></div>
      </section>
      <section class="panel">
        <h2>Run configuration</h2>
        <div class="event-list" id="config"></div>
      </section>
    </aside>
  </main>
</div>
<script>
const TRACE = __TRACE_JSON__;
const frames = TRACE.frames;
let index = 0;
let timer = null;

const els = {
  slider: document.getElementById("slider"),
  play: document.getElementById("play"),
  prev: document.getElementById("prev"),
  next: document.getElementById("next"),
  statusbar: document.getElementById("statusbar"),
  kvChart: document.getElementById("kvChart"),
  workChart: document.getElementById("workChart"),
  blockGrid: document.getElementById("blockGrid"),
  queues: document.getElementById("queues"),
  events: document.getElementById("events"),
  requestRows: document.getElementById("requestRows"),
  config: document.getElementById("config")
};

els.slider.max = Math.max(0, frames.length - 1);
els.slider.addEventListener("input", event => setIndex(Number(event.target.value)));
els.prev.addEventListener("click", () => setIndex(index - 1));
els.next.addEventListener("click", () => setIndex(index + 1));
els.play.addEventListener("click", togglePlay);

function clamp(value, lo, hi) {
  return Math.max(lo, Math.min(hi, value));
}

function setIndex(next) {
  index = clamp(next, 0, frames.length - 1);
  els.slider.value = index;
  render();
}

function togglePlay() {
  if (timer) {
    clearInterval(timer);
    timer = null;
    els.play.textContent = "Play";
    els.play.title = "Play trace";
    return;
  }
  els.play.textContent = "Stop";
  els.play.title = "Stop trace";
  timer = setInterval(() => {
    if (index >= frames.length - 1) {
      togglePlay();
    } else {
      setIndex(index + 1);
    }
  }, 420);
}

function fmtMs(ms) {
  if (ms >= 1000) return (ms / 1000).toFixed(2) + " s";
  return ms.toFixed(1) + " ms";
}

function pct(value) {
  return (value * 100).toFixed(1) + "%";
}

function stat(label, value) {
  return `<div class="stat"><label>${label}</label><strong>${value}</strong></div>`;
}

function renderStatus(frame) {
  const summary = TRACE.summary;
  els.statusbar.innerHTML = [
    stat("step", `${frame.step} / ${TRACE.metadata.captured_steps}`),
    stat("time", fmtMs(frame.time_ms)),
    stat("running", frame.counts.running),
    stat("KV used", `${frame.gpu_used_blocks}/${frame.gpu_total_blocks}`),
    stat("preemptions", summary.num_preemptions),
    stat("cache hit", pct(frame.cache_hit_rate))
  ].join("");
}

function scaleX(i, n, left, width) {
  if (n <= 1) return left;
  return left + (i / (n - 1)) * width;
}

function renderKvChart() {
  const w = 420, h = 176, left = 34, top = 14, width = 366, height = 130;
  const points = frames.map((f, i) => [
    scaleX(i, frames.length, left, width),
    top + height - f.gpu_util * height
  ]);
  const path = points.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(" ");
  const cursorX = scaleX(index, frames.length, left, width);
  els.kvChart.innerHTML = `
    <line class="axis" x1="${left}" y1="${top + height}" x2="${left + width}" y2="${top + height}"></line>
    <line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${top + height}"></line>
    <text x="4" y="${top + 6}">100%</text>
    <text x="10" y="${top + height}">0%</text>
    <path class="kv-line" d="${path}"></path>
    <line class="cursor" x1="${cursorX}" y1="${top}" x2="${cursorX}" y2="${top + height}"></line>
  `;
}

function renderWorkChart() {
  const w = 420, h = 176, left = 34, top = 14, width = 366, height = 130;
  const budget = TRACE.metadata.token_budget;
  const barGap = 2;
  const barWidth = Math.max(2, (width / frames.length) - barGap);
  let bars = "";
  frames.forEach((f, i) => {
    const x = scaleX(i, frames.length, left, width) - barWidth / 2;
    const prefillH = Math.min(height, (f.work.prefill_tokens / budget) * height);
    const decodeH = Math.min(height - prefillH, (f.work.decode_tokens / budget) * height);
    const base = top + height;
    bars += `<rect class="prefill" x="${x.toFixed(1)}" y="${(base - prefillH).toFixed(1)}" width="${barWidth.toFixed(1)}" height="${prefillH.toFixed(1)}"></rect>`;
    bars += `<rect class="decode" x="${x.toFixed(1)}" y="${(base - prefillH - decodeH).toFixed(1)}" width="${barWidth.toFixed(1)}" height="${decodeH.toFixed(1)}"></rect>`;
  });
  const cursorX = scaleX(index, frames.length, left, width);
  els.workChart.innerHTML = `
    <line class="axis" x1="${left}" y1="${top + height}" x2="${left + width}" y2="${top + height}"></line>
    <line class="axis" x1="${left}" y1="${top}" x2="${left}" y2="${top + height}"></line>
    <text x="2" y="${top + 6}">${budget}</text>
    ${bars}
    <line class="cursor" x1="${cursorX}" y1="${top}" x2="${cursorX}" y2="${top + height}"></line>
  `;
}

function renderBlocks(frame) {
  const touched = new Set();
  frame.work.prefix_hits.forEach(hit => hit.blocks.forEach(block => touched.add(block)));
  els.blockGrid.innerHTML = frame.blocks.map(block => {
    const classes = ["block"];
    if (block.used) classes.push("used");
    if (block.cached) classes.push("cached");
    if (block.shared) classes.push("shared");
    if (touched.has(block.id)) classes.push("hot");
    const owners = block.owners.length ? block.owners.join(", ") : (block.cached ? "cached, idle" : "free");
    return `<span class="${classes.join(" ")}" title="block ${block.id}: ${owners}"></span>`;
  }).join("");
}

function renderQueues(frame) {
  const order = ["pending", "waiting", "running", "swapped", "finished"];
  els.queues.innerHTML = order.map(name => {
    const chips = frame.queues[name].slice(0, 36).map(id => `<span class="chip ${name}">${id}</span>`).join("");
    const rest = frame.queues[name].length > 36 ? `<span class="chip">+${frame.queues[name].length - 36}</span>` : "";
    return `<div class="queue"><h3>${name} ${frame.counts[name]}</h3><div class="chips">${chips}${rest || '<span class="empty">empty</span>'}</div></div>`;
  }).join("");
}

function renderEvents(frame) {
  const w = frame.work;
  const rows = [];
  if (w.prefill.length) rows.push(["prefill", w.prefill.map(x => `${x.id} +${x.tokens}`).join(", ")]);
  if (w.decode.length) rows.push(["decode", `${w.decode.length} seqs: ${w.decode.slice(0, 18).join(", ")}${w.decode.length > 18 ? ", ..." : ""}`]);
  if (w.preempted.length) rows.push(["recompute", w.preempted.join(", ")]);
  if (w.swapped_out.length) rows.push(["swap out", w.swapped_out.join(", ")]);
  if (w.swapped_in.length) rows.push(["swap in", w.swapped_in.join(", ")]);
  if (w.prefix_hits.length) {
    rows.push(["prefix hits", w.prefix_hits.map(hit => `${hit.id}: ${hit.tokens} tok / blocks ${hit.blocks.join(",")}`).join("; ")]);
  }
  if (!rows.length) rows.push(["idle", "no scheduled token work"]);
  els.events.innerHTML = rows.map(([label, body]) => `<div class="event"><strong>${label}</strong><span>${body}</span></div>`).join("");
}

function renderRequests(frame) {
  els.requestRows.innerHTML = frame.requests.map(row => `
    <tr>
      <td>${row.id}</td>
      <td><span class="status ${row.status}">${row.status}</span></td>
      <td>${row.arrival}</td>
      <td>${row.queue_ms === null ? "-" : row.queue_ms}</td>
      <td>${row.prompt}</td>
      <td>${row.generated}/${row.max}</td>
      <td>${row.computed}</td>
      <td>${row.cached}</td>
      <td>${row.blocks}</td>
      <td>${row.preemptions}</td>
    </tr>
  `).join("");
}

function renderConfig() {
  const meta = TRACE.metadata;
  const summary = TRACE.summary;
  const rows = [
    ["policy", "continuous batching"],
    ["block size", `${meta.block_size} tokens`],
    ["GPU blocks", frames[0] ? frames[0].gpu_total_blocks : 0],
    ["token budget", meta.token_budget],
    ["max seqs", meta.max_num_seqs],
    ["preemption", meta.preemption_mode],
    ["prefix cache", meta.prefix_caching ? "enabled" : "disabled"],
    ["cache budget", `${meta.prefix_cache_max_blocks} blocks`],
    ["cache blocks", `${summary.prefix_cache_blocks} total / ${summary.prefix_cache_pinned_blocks} pinned`],
    ["cache evictions", summary.prefix_cache_evictions],
    ["saved prefill", `${summary.prefix_cache_saved_tokens} tokens`],
    ["completed", `${summary.num_completed}/${summary.num_requests}`],
    ["throughput", `${summary.throughput_tok_s.toFixed(1)} tok/s`]
  ];
  els.config.innerHTML = rows.map(([label, body]) => `<div class="event"><strong>${label}</strong><span>${body}</span></div>`).join("");
}

function render() {
  if (!frames.length) return;
  const frame = frames[index];
  renderStatus(frame);
  renderKvChart();
  renderWorkChart();
  renderBlocks(frame);
  renderQueues(frame);
  renderEvents(frame);
  renderRequests(frame);
  renderConfig();
}

render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
