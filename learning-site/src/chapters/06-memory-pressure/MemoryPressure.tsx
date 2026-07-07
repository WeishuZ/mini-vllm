import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./MemoryPressure.css";

const diveSteps: DiveStep[] = [
  {
    title: "admission 之后还会增长",
    body: "paged KV 的 block pool 是有限的。请求 admission 时放得下，不代表后续 decode 继续增长时还放得下。",
    visual: "pressure",
    states: [
      { label: "resource", value: "GPU block pool", tone: "hot" },
      { label: "growth", value: "one KV token per decode" },
      { label: "risk", value: "can_grow false", tone: "warn" },
    ],
    chain: ["admit", "decode grows", "KV pressure", "preemption"],
  },
  {
    title: "can_grow 是压力门",
    body: "Scheduler 推进 sequence 前必须问 BlockManager：这次 prefill chunk 或 decode token 需要的 KV 能不能长出来。",
    visual: "pressure",
    code: "if not bm.can_grow(seq, need):\n    preempt someone",
    states: [
      { label: "need", value: "prefill chunk or 1 decode" },
      { label: "gate", value: "can_grow(seq, need)", tone: "hot" },
      { label: "failure", value: "release KV" },
    ],
    chain: ["need KV", "check blocks", "fail", "choose victim"],
  },
  {
    title: "preempt 新请求",
    body: "running sequence 按 arrival 排序。老请求优先，新请求从队尾被抢掉。这不是完整公平性方案，但能避免老请求一直被饿死。",
    visual: "queues",
    states: [
      { label: "priority", value: "older first", tone: "ok" },
      { label: "victim", value: "newer tail", tone: "hot" },
      { label: "goal", value: "forward progress" },
    ],
    chain: ["sort running", "protect old", "evict tail", "progress"],
  },
  {
    title: "recompute 释放现在",
    body: "recompute path 直接 free GPU blocks，把 sequence 放回 waiting 队头。当前压力解决了，未来要重新 prefill 已有上下文。",
    visual: "pressure",
    code: "bm.free(seq)\nseq.reset_for_recompute()\nwaiting.appendleft(seq)",
    states: [
      { label: "immediate", value: "GPU blocks free", tone: "ok" },
      { label: "future", value: "more prefill work", tone: "warn" },
      { label: "queue", value: "waiting front", tone: "hot" },
    ],
    chain: ["free KV", "reset computed", "keep generated", "re-prefill later"],
  },
  {
    title: "swap 转移压力",
    body: "swap path 不丢逻辑状态，而是把 GPU block 占用转移到 CPU block pool。miniVLLM 建模队列语义，不模拟真实搬运时间。",
    visual: "pressure",
    code: "bm.swap_out(seq)\nseq.status = SWAPPED\nswapped.append(seq)",
    states: [
      { label: "GPU", value: "blocks reclaimed", tone: "ok" },
      { label: "CPU", value: "swap pool used", tone: "hot" },
      { label: "real cost", value: "bandwidth/latency", tone: "warn" },
    ],
    chain: ["GPU pressure", "CPU pool", "swapped queue", "swap in later"],
  },
  {
    title: "sharing 让 swap 变难",
    body: "miniVLLM 只在 prefix caching 关闭时允许 swap。原因是共享 block、ref count、LRU 和 COW 叠在一起后，swap 语义不再是私有 block 迁移。",
    visual: "blocks",
    states: [
      { label: "simple case", value: "private blocks", tone: "ok" },
      { label: "complex case", value: "shared prefix blocks", tone: "warn" },
      { label: "guard", value: "swap disabled with prefix cache", tone: "hot" },
    ],
    chain: ["sharing", "ref counts", "swap ownership", "state explosion"],
  },
  {
    title: "watermark 是 admission control",
    body: "admission 不把 available blocks 用光。它保留 headroom，给 running sequence 后续 decode 增长留空间。",
    visual: "pressure",
    formula: "available_blocks > watermark_blocks",
    states: [
      { label: "headroom", value: "reserved free blocks", tone: "hot" },
      { label: "cost", value: "lower instant admission" },
      { label: "benefit", value: "stable running set", tone: "ok" },
    ],
    chain: ["available blocks", "watermark gate", "headroom", "less preempt"],
  },
  {
    title: "thrashing 是机制交互失败",
    body: "没有 watermark 时，系统可能刚 admit 就 preempt，preempt 后又 admit，然后继续 preempt。吞吐、TTFT 和 prefill work 都会变差。",
    visual: "pressure",
    states: [
      { label: "loop", value: "admit -> preempt -> admit", tone: "hot" },
      { label: "symptom", value: "preemption spike", tone: "warn" },
      { label: "fix", value: "headroom", tone: "ok" },
    ],
    chain: ["over-admit", "no growth room", "preempt", "thrash", "watermark"],
  },
];

export default function MemoryPressure({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="mp-scene"
      step={step}
      eyebrow="06 / memory pressure"
      title="Preemption、Swap 与 Watermark"
      hardQuestion="KV pressure 下系统为什么需要 admission control，而不是把 block 塞满？"
      source="scheduler.py / block_manager.py / config.py"
      invariant="preemption 必须释放 GPU KV；recompute 保留 generated tokens；swap 只走私有 block 语义。"
      production="vLLM preemption / swap policy / KV watermark / overload control"
      steps={diveSteps}
    />
  );
}
