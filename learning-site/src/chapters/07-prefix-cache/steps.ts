import type { StepDef } from "../../registry/types";

export const steps: StepDef[] = [
  { title: "KV block cache", note: "The cache stores materialized KV blocks." },
  { title: "Stable publish", note: "Only exclusive full prompt blocks publish." },
  { title: "Full-prefix hash", note: "Hash covers [0, end)." },
  { title: "Contiguous hits", note: "First miss stops sharing." },
  { title: "Computed jump", note: "Hits append physical blocks and advance num_computed." },
  { title: "Prefill only", note: "Prefix cache skips prompt work, not future decode." },
  { title: "Pinned lifecycle", note: "Cached blocks move from pinned to evictable." },
  { title: "Cache pressure", note: "LRU controls resident KV cost." },
];
