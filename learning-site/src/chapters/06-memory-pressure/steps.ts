import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "KV keeps growing", note: "Admission is not the last memory check." },
  { title: "can_grow gate", note: "Every progress step needs KV capacity." },
  { title: "Victim priority", note: "Newer running sequences are preempted first." },
  { title: "Recompute path", note: "Free now, repay with prefill later." },
  { title: "Swap path", note: "Move pressure to CPU pool." },
  { title: "Sharing interaction", note: "Shared blocks complicate swap semantics." },
  { title: "Watermark", note: "Admission leaves growth headroom." },
  { title: "Thrashing", note: "Headroom prevents admit/preempt loops." },
];
