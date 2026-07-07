import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "Metric groups", note: "Metrics classify pressure and experience." },
  { title: "TTFT causes", note: "High TTFT has multiple roots." },
  { title: "Throughput trap", note: "Average tokens/s can hide bad streaming." },
  { title: "Cache metrics", note: "Hit rate must pair with saved tokens." },
  { title: "Trace frames", note: "Aggregate anomalies map to step state." },
  { title: "vLLM mapping", note: "Mini subsystems map to real vLLM layers." },
  { title: "Stacked mechanisms", note: "Real complexity is interaction complexity." },
  { title: "Four threads", note: "KV, scheduling, execution, observability." },
];
