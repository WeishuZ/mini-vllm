import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./SchedulerChapter.css";

const diveSteps: DiveStep[] = [
  {
    title: "Scheduler 管两个预算",
    body: "LLM serving 不是按 request 数调度。每一步都同时受 compute token budget 和 KV block budget 限制。",
    visual: "queues",
    states: [
      { label: "compute", value: "max_num_batched_tokens", tone: "hot" },
      { label: "memory", value: "GPU block pool" },
      { label: "output", value: "StepWork" },
    ],
    chain: ["waiting requests", "token budget", "KV budget", "StepWork"],
  },
  {
    title: "running 优先",
    body: "continuous policy 的顺序是 resume swapped、advance running、admit waiting。已经进入系统的请求比新请求优先。",
    visual: "queues",
    code: "_resume_swapped(work)\n_advance_running(work, budget)\n_admit_waiting(work, budget)",
    states: [
      { label: "priority", value: "running first", tone: "hot" },
      { label: "new work", value: "waiting last" },
      { label: "reason", value: "protect streaming latency" },
    ],
    chain: ["resume", "advance", "admit", "stable stream"],
  },
  {
    title: "prefill 和 decode 混合",
    body: "`_advance_running` 里，欠 KV 的 sequence 拿 prefill chunk；KV 完整的 sequence 做一次 decode。",
    visual: "queues",
    states: [
      { label: "in_prefill", value: "prefill chunk", tone: "warn" },
      { label: "ready", value: "one decode token", tone: "ok" },
      { label: "same step", value: "mixed batch", tone: "hot" },
    ],
    chain: ["running set", "prefill debt", "decode-ready", "mixed work"],
  },
  {
    title: "chunked prefill 保住 decode",
    body: "长 prompt 如果一次吃完整个 step，会挡住流式 decode。chunked prefill 把大 prompt 分散到多个 step，换取更稳定的 decode latency。",
    visual: "timeline",
    formula: "chunk = min(prefill_remaining, budget)",
    states: [
      { label: "without chunking", value: "long prompt blocks step", tone: "warn" },
      { label: "with chunking", value: "decode interleaves", tone: "ok" },
      { label: "tradeoff", value: "single prompt may stretch", tone: "hot" },
    ],
    chain: ["long prefill", "step latency", "chunk", "decode stability"],
  },
  {
    title: "token budget 不是 batch size",
    body: "它是一次 scheduler step 的 token budget。prefill token 和 decode token 都从这里扣，所以它直接改变 TTFT、ITL 和 throughput。",
    visual: "timeline",
    formula: "budget = prefill_tokens + decode_seqs",
    states: [
      { label: "too small", value: "fragmented prefill", tone: "warn" },
      { label: "too large", value: "decode gaps", tone: "warn" },
      { label: "target", value: "workload-dependent", tone: "hot" },
    ],
    chain: ["budget", "prefill/decode mix", "latency", "throughput"],
  },
  {
    title: "static 制造 head-of-line",
    body: "static batching 只有 running 为空才 admit 新 batch。新请求到来时，如果当前 batch 没清空，只能等。",
    visual: "queues",
    states: [
      { label: "admission", value: "only when idle", tone: "warn" },
      { label: "tail", value: "TTFT p99 explodes", tone: "hot" },
      { label: "baseline", value: "static policy" },
    ],
    chain: ["fixed batch", "arrival waits", "head-of-line", "tail latency"],
  },
  {
    title: "static 还会 batch decay",
    body: "同一个 batch 里短 sequence 先结束，decode 宽度逐渐变窄。但 static policy 不补新请求，后半段 GPU 利用率下降。",
    visual: "queues",
    states: [
      { label: "short seq", value: "finishes early" },
      { label: "batch width", value: "decays", tone: "warn" },
      { label: "throughput", value: "lower ceiling", tone: "hot" },
    ],
    chain: ["variable lengths", "empty slots", "no refill", "lower utilization"],
  },
  {
    title: "continuous 是持续流",
    body: "continuous batching 的效果是不断 refill running set。完成就回收 KV，有预算就 admit，decode batch 宽度更稳定。",
    visual: "queues",
    states: [
      { label: "finished", value: "free KV blocks", tone: "ok" },
      { label: "waiting", value: "admit into gaps", tone: "hot" },
      { label: "effect", value: "higher saturated throughput" },
    ],
    chain: ["finish", "free", "admit", "refill", "sustain"],
  },
];

export default function SchedulerChapter({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="sc-scene"
      step={step}
      eyebrow="05 / scheduler"
      title="Continuous Batching 与 token budget"
      hardQuestion="continuous batching 为什么比 static batching 更适合 online LLM serving？"
      source="scheduler.py / config.py"
      invariant="running 优先于 waiting；prefill 和 decode 共同消耗 token budget。"
      production="vLLM scheduler / chunked prefill / max_num_batched_tokens"
      steps={diveSteps}
    />
  );
}
