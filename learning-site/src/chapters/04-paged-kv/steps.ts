import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "KV as memory system", note: "PagedAttention is allocation control." },
  { title: "Block table", note: "Logical continuity maps to scattered blocks." },
  { title: "Demand paging", note: "Grow only when target exceeds capacity." },
  { title: "Contiguous failure", note: "Max context dominates reservation." },
  { title: "Paged effect", note: "Actual length dominates capacity." },
  { title: "Ref count", note: "Sharing requires safe lifetime management." },
  { title: "Copy-on-write", note: "Partial shared tail must fork before writes." },
  { title: "Tradeoff", note: "Utilization rises while state complexity grows." },
];
