import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "Two budgets", note: "Scheduling is bounded by compute and KV." },
  { title: "Running first", note: "Continuous policy protects in-flight work." },
  { title: "Mixed work", note: "Prefill and decode share a step." },
  { title: "Chunked prefill", note: "Long prompts are sliced to protect decode." },
  { title: "Token budget", note: "The scheduler budget is not batch size." },
  { title: "Head-of-line", note: "Static batching delays new arrivals." },
  { title: "Batch decay", note: "Static decode width shrinks over time." },
  { title: "Refill", note: "Continuous batching sustains the running set." },
];
