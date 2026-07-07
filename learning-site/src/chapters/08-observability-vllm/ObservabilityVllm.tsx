import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./ObservabilityVllm.css";

const diveSteps: DiveStep[] = [
  {
    title: "能跑不是目标",
    body: "serving 系统开发要能定位瓶颈。miniVLLM 的 metrics 分成用户体验、吞吐、显存压力和缓存收益四类。",
    visual: "metrics",
    states: [
      { label: "UX", value: "TTFT / E2E" },
      { label: "output", value: "throughput", tone: "ok" },
      { label: "pressure", value: "KV / preempt / cache", tone: "hot" },
    ],
    chain: ["run", "measure", "classify", "debug"],
  },
  {
    title: "TTFT 要拆因果",
    body: "TTFT 高不一定是 scheduler 差。可能是 waiting queue 长、prefill 太重、prefix miss，或者 recompute preemption 反复打断。",
    visual: "metrics",
    states: [
      { label: "queue", value: "waiting time", tone: "warn" },
      { label: "prefill", value: "prompt work" },
      { label: "preempt", value: "recompute debt", tone: "hot" },
    ],
    chain: ["TTFT high", "queue?", "prefill?", "preempt?", "cache?"],
  },
  {
    title: "throughput 不能单看",
    body: "throughput 高但 ITL 差，用户看到的 streaming 会卡。serving 不是只优化平均 tokens/s。",
    visual: "metrics",
    states: [
      { label: "aggregate", value: "tokens/s", tone: "ok" },
      { label: "streaming", value: "ITL / TPOT", tone: "hot" },
      { label: "tail", value: "p99 matters", tone: "warn" },
    ],
    chain: ["high throughput", "decode gaps", "bad stream", "tail latency"],
  },
  {
    title: "cache hit 也要看 saved tokens",
    body: "prefix cache hit rate 高，不一定收益大。如果共享 prefix 很短，saved prefill tokens 少，TTFT 改善也有限。",
    visual: "prefix",
    states: [
      { label: "hit rate", value: "block fraction" },
      { label: "saved work", value: "tokens skipped", tone: "hot" },
      { label: "effect", value: "TTFT delta" },
    ],
    chain: ["hit blocks", "covered tokens", "prefill saved", "latency effect"],
  },
  {
    title: "trace 把指标拆回 step",
    body: "`trace_viewer` 每帧记录 queues、work、requests 和 blocks。aggregate metrics 的异常要回到 step-level 状态解释。",
    visual: "engine",
    code: "frame = { queues, work, requests, blocks, gpu_util }",
    states: [
      { label: "aggregate", value: "EngineMetrics" },
      { label: "frame", value: "build_trace()", tone: "hot" },
      { label: "debug target", value: "queues + block owners" },
    ],
    chain: ["metric anomaly", "trace frame", "queue/block/work", "bottleneck"],
  },
  {
    title: "miniVLLM 到 vLLM",
    body: "miniVLLM 省掉真实 kernel，但 control plane 能对应到真实 vLLM：engine、scheduler、KV cache manager、model runner 和 attention backend。",
    visual: "metrics",
    states: [
      { label: "LLMEngine", value: "engine core", tone: "hot" },
      { label: "Scheduler", value: "request scheduling" },
      { label: "BlockManager", value: "KV cache manager" },
    ],
    chain: ["mini abstraction", "engine core", "worker", "attention backend", "real vLLM"],
  },
  {
    title: "真实复杂度来自叠加",
    body: "真实 vLLM 难读，不是因为一个神秘算法，而是 KV paging、ref count、prefix cache、preemption、chunked prefill、sampling 和 distributed execution 叠在一起。",
    visual: "metrics",
    states: [
      { label: "mechanisms", value: "stacked", tone: "hot" },
      { label: "risk", value: "state interaction", tone: "warn" },
      { label: "skill", value: "read invariants" },
    ],
    chain: ["paging", "sharing", "preemption", "scheduling", "execution"],
  },
  {
    title: "最后看四条主线",
    body: "读完这个系统，真实 vLLM 就不再是一堆文件。它是一套围绕 KV、调度、执行和观测展开的 serving control plane。",
    visual: "metrics",
    states: [
      { label: "KV", value: "where memory goes", tone: "hot" },
      { label: "scheduler", value: "who progresses" },
      { label: "metrics", value: "why it behaves" },
    ],
    chain: ["KV", "scheduling", "execution", "observability", "production"],
  },
];

export default function ObservabilityVllm({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="ov-scene"
      step={step}
      eyebrow="08 / observability to vLLM"
      title="Metrics、Trace 与真实 vLLM 映射"
      hardQuestion="为什么 metrics 必须组合解读，才能定位 serving 瓶颈？"
      source="metrics.py / trace_viewer.py"
      invariant="aggregate metrics 必须能追溯到 step-level queues、work 和 block ownership。"
      production="vLLM metrics endpoint / scheduler logs / engine traces"
      steps={diveSteps}
    />
  );
}
