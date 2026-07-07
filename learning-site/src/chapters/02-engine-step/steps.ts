import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "Step transaction", note: "One serving cycle." },
  { title: "StepWork boundary", note: "Plan before mutation." },
  { title: "Prefill commit", note: "KV materialization advances." },
  { title: "Decode commit", note: "Generated and computed move together." },
  { title: "Latency commit", note: "Clock advances after work shape is known." },
  { title: "Finished cleanup", note: "KV blocks are freed after decode completion." },
  { title: "Metrics facts", note: "StepStat becomes the debugging surface." },
];
