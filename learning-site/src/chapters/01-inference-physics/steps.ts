import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "Autoregressive shape", note: "Token generation is stepwise." },
  { title: "Prefill cost", note: "Prompt processing is chunk-shaped." },
  { title: "Decode cost", note: "Decode is narrow but KV-read heavy." },
  { title: "KV memory", note: "KV grows with live tokens." },
  { title: "Serving conflict", note: "Prefill and decode prefer different batch shapes." },
  { title: "Cost model", note: "ModelRunner preserves the bottleneck shape." },
  { title: "Production map", note: "Map the abstraction to real vLLM execution." },
];
