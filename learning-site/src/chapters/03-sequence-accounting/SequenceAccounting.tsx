import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./SequenceAccounting.css";

const diveSteps: DiveStep[] = [
  {
    title: "Sequence 是运行态",
    body: "Request 是用户提交的工作，Sequence 是 engine 内部的可变状态。它同时承载逻辑 token、KV materialization 和调度生命周期。",
    visual: "sequence",
    states: [
      { label: "input", value: "Request" },
      { label: "runtime", value: "Sequence", tone: "hot" },
      { label: "lifecycle", value: "WAITING/RUNNING/SWAPPED/FINISHED" },
    ],
    chain: ["request", "sequence", "queues", "metrics"],
  },
  {
    title: "length 是逻辑长度",
    body: "`length` 不是 KV 数量。它表示当前已经存在的 token：原始 prompt 加上已经生成给用户的 tokens。",
    visual: "sequence",
    formula: "length = prompt_len + num_generated",
    states: [
      { label: "prompt", value: "prompt_len" },
      { label: "visible output", value: "num_generated" },
      { label: "logical length", value: "prompt + generated", tone: "hot" },
    ],
    chain: ["prompt exists", "tokens generated", "length grows", "context grows"],
  },
  {
    title: "num_computed 是 KV 前缀",
    body: "KV 状态由 `num_computed` 决定。它不是任意集合，而是从 token 0 开始连续 materialized 的前缀。",
    visual: "sequence",
    formula: "KV prefix = tokens[0:num_computed]",
    states: [
      { label: "materialized", value: "num_computed", tone: "hot" },
      { label: "shape", value: "prefix only" },
      { label: "not allowed", value: "holes in KV", tone: "warn" },
    ],
    chain: ["computed prefix", "KV cache", "attention reads", "decode readiness"],
  },
  {
    title: "prefill 是偿还 KV 债",
    body: "只要 `num_computed < length`，sequence 就欠 KV。欠多少就是 `prefill_remaining`。",
    visual: "sequence",
    formula: "prefill_remaining = length - num_computed",
    states: [
      { label: "gate", value: "num_computed < length", tone: "warn" },
      { label: "debt", value: "prefill_remaining", tone: "hot" },
      { label: "work type", value: "prefill" },
    ],
    chain: ["length grows", "KV missing", "prefill debt", "materialize"],
  },
  {
    title: "decode 要求 KV 完整",
    body: "decode 的前提是 `num_computed == length`。否则新 token 的 attention 读不到完整历史。",
    visual: "sequence",
    formula: "decode_ready iff num_computed == length",
    states: [
      { label: "gate", value: "closed until equal", tone: "hot" },
      { label: "failure", value: "missing history KV", tone: "warn" },
      { label: "effect", value: "safe next token" },
    ],
    chain: ["full context", "decode gate", "next token", "KV append"],
  },
  {
    title: "decode 后双计数同步",
    body: "生成一个 token 会让逻辑长度增加。这个 token 的 KV 也必须立刻存在，所以 `num_generated` 和 `num_computed` 同步增长。",
    visual: "sequence",
    code: "num_generated += 1\nnum_computed += 1",
    states: [
      { label: "logical", value: "+1 token", tone: "hot" },
      { label: "KV", value: "+1 token", tone: "ok" },
      { label: "invariant", value: "computed catches length" },
    ],
    chain: ["emit token", "append KV", "next step valid", "stream continues"],
  },
  {
    title: "recompute 解释了代价",
    body: "preemption 抢掉 KV 时，用户已经看到的 generated tokens 不能消失。但 `num_computed` 必须清零，因为 KV 确实被释放了。",
    visual: "pressure",
    code: "reset_for_recompute():\n    num_computed = 0\n    block_table = []",
    states: [
      { label: "kept", value: "num_generated", tone: "ok" },
      { label: "reset", value: "num_computed", tone: "hot" },
      { label: "future cost", value: "re-prefill prompt + generated", tone: "warn" },
    ],
    chain: ["KV freed", "visible tokens kept", "computed reset", "prefill work grows"],
  },
];

export default function SequenceAccounting({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="sa-scene"
      step={step}
      eyebrow="03 / sequence accounting"
      title="Sequence 的 token/KV 不变量"
      hardQuestion="KV cache 为什么由 num_computed 驱动，而不是由 sequence length 驱动？"
      source="request.py::Sequence"
      invariant="num_computed <= prompt_len + num_generated；decode 前必须 materialize 完整前缀。"
      production="vLLM sequence state / computed tokens / recompute preemption"
      steps={diveSteps}
    />
  );
}
