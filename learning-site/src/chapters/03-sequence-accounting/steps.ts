import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "Runtime state", note: "Sequence is mutable engine state." },
  { title: "Logical length", note: "Prompt plus generated tokens." },
  { title: "Computed prefix", note: "KV exists for a contiguous prefix." },
  { title: "Prefill debt", note: "Missing KV becomes prefill work." },
  { title: "Decode gate", note: "Decode requires full context KV." },
  { title: "Synchronized counters", note: "Generated and computed advance together." },
  { title: "Recompute cost", note: "Generated tokens stay while KV resets." },
];
