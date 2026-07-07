import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./EngineStep.css";

const diveSteps: DiveStep[] = [
  {
    title: "一个 step 是一次事务",
    body: "`LLMEngine.step()` 不是简单 tick。它把 scheduler plan、KV 分配、token 状态提交、clock 推进、finished 回收和 metrics 写入放在同一个调度周期里。",
    visual: "engine",
    states: [
      { label: "entry", value: "LLMEngine.step()", tone: "hot" },
      { label: "scope", value: "schedule -> commit -> observe" },
      { label: "unit", value: "one serving cycle" },
    ],
    chain: ["schedule", "commit", "latency", "reclaim", "metrics"],
  },
  {
    title: "Scheduler 只产出计划",
    body: "Scheduler 返回 `StepWork`。它描述哪些 sequence 做 prefill，哪些 sequence 做 decode，但不直接推进 token，不更新时间。",
    visual: "engine",
    code: "work = self.scheduler.schedule()",
    states: [
      { label: "boundary object", value: "StepWork", tone: "hot" },
      { label: "prefill", value: "[(seq, chunk)]" },
      { label: "decode", value: "[seq]" },
    ],
    chain: ["policy", "StepWork", "no token mutation", "engine commit"],
  },
  {
    title: "prefill 提交 KV 进度",
    body: "Engine 看到 `work.prefill` 后才更新 `seq.num_computed`。这一步表示这些 token 的 KV 已经 materialized，可以进入 block table 语义。",
    visual: "sequence",
    code: "seq.num_computed += chunk\nself.total_prefill_tokens += chunk",
    states: [
      { label: "mutated", value: "num_computed", tone: "hot" },
      { label: "counter", value: "total_prefill_tokens" },
      { label: "cache hook", value: "_register_full_prompt_blocks" },
    ],
    chain: ["prefill work", "KV exists", "cache publish", "metrics"],
  },
  {
    title: "decode 同步两个计数",
    body: "decode 后，逻辑 token 多了一个，KV 也必须多一个。所以 `num_generated` 和 `num_computed` 同时增加。",
    visual: "sequence",
    code: "seq.num_generated += 1\nseq.num_computed += 1",
    states: [
      { label: "logical token", value: "num_generated + 1", tone: "hot" },
      { label: "KV token", value: "num_computed + 1", tone: "ok" },
      { label: "invariant", value: "KV catches up" },
    ],
    chain: ["decode token", "logical length grows", "KV stored", "next attention valid"],
  },
  {
    title: "latency 在 work 后计算",
    body: "clock 不能在 schedule 前推进。只有这一步的 prefill/decode mix 确定后，ModelRunner 才能给出这次 batch 的 simulated latency。",
    visual: "engine",
    code: "self.clock_ms += self.runner.step_latency_ms(work)",
    states: [
      { label: "input", value: "StepWork" },
      { label: "output", value: "clock_ms", tone: "hot" },
      { label: "model", value: "prefill + decode cost" },
    ],
    chain: ["work shape", "cost model", "clock advance", "TTFT/E2E"],
  },
  {
    title: "finished 之后才能回收",
    body: "请求完成判断发生在 decode 后。只有生成 token 数达到 `max_tokens`，Engine 才 drop finished 并调用 BlockManager free。",
    visual: "blocks",
    code: "if seq.is_finished:\n    seq.finish_time = self.clock_ms\n    finished.append(seq)",
    states: [
      { label: "finish gate", value: "num_generated == max_tokens" },
      { label: "cleanup", value: "drop_finished", tone: "ok" },
      { label: "resource", value: "free KV blocks", tone: "hot" },
    ],
    chain: ["decode", "finish_time", "drop", "free blocks"],
  },
  {
    title: "metrics 是事实落点",
    body: "`StepStat` 把这一步的 running、decode count、prefill tokens 和 GPU utilization 固化下来。之后 benchmark 和 trace 都从这些事实解释系统行为。",
    visual: "metrics",
    code: "StepStat(t_ms, running, decode_seqs, prefill_tokens, gpu_util)",
    states: [
      { label: "history", value: "per-step facts", tone: "hot" },
      { label: "aggregate", value: "EngineMetrics" },
      { label: "debug", value: "trace frame" },
    ],
    chain: ["state commit", "StepStat", "history", "benchmark"],
  },
];

export default function EngineStep({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="es-scene"
      step={step}
      eyebrow="02 / engine step"
      title="LLMEngine.step() 的提交边界"
      hardQuestion="为什么 scheduler 只产出计划，而不是直接修改 token 和时钟？"
      source="engine.py / scheduler.py / metrics.py"
      invariant="Scheduler 先决策；Engine 再提交 token、clock、finished 和 metrics。"
      production="vLLM engine core / scheduler output / worker execution"
      steps={diveSteps}
    />
  );
}
