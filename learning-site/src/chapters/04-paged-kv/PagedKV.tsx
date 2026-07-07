import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./PagedKV.css";

const diveSteps: DiveStep[] = [
  {
    title: "KV cache 是内存系统",
    body: "PagedAttention 在这层不是改 attention 数学，而是把 KV cache 从连续预留变成分页分配。核心问题是实际 token 怎么映射到有限 physical blocks。",
    visual: "blocks",
    states: [
      { label: "object", value: "KV cache" },
      { label: "manager", value: "BlockManager", tone: "hot" },
      { label: "unit", value: "physical block" },
    ],
    chain: ["KV grows", "block pool finite", "allocation policy", "capacity"],
  },
  {
    title: "block table 是页表",
    body: "sequence 逻辑上连续，但 physical blocks 可以不连续。`Sequence.block_table` 把 logical block index 映射到 physical block id。",
    visual: "blocks",
    formula: "logical block i -> block_table[i] -> physical block id",
    states: [
      { label: "logical", value: "contiguous sequence" },
      { label: "physical", value: "scattered blocks", tone: "ok" },
      { label: "mapping", value: "block_table", tone: "hot" },
    ],
    chain: ["logical page", "block table", "physical block", "no contiguous run"],
  },
  {
    title: "grow() 是 demand paging",
    body: "`grow(seq, num_new)` 看当前 KV token 和 block capacity。target 超过 capacity 时才补 block。",
    visual: "blocks",
    formula: "capacity = len(block_table) * block_size",
    code: "deficit = max(0, target - capacity)\nextra = ceil(deficit / block_size)",
    states: [
      { label: "current", value: "num_kv_tokens" },
      { label: "capacity", value: "blocks * block_size" },
      { label: "allocate when", value: "target > capacity", tone: "hot" },
    ],
    chain: ["need tokens", "check capacity", "allocate blocks", "track peak"],
  },
  {
    title: "contiguous 被 max context 支配",
    body: "连续预留会按 `max_seq_len` 为每条 sequence 留空间。请求实际很短，也会吃掉长上下文 reservation。",
    visual: "blocks",
    formula: "contiguous_fit = total_slots // max_seq_len",
    states: [
      { label: "reserve", value: "max_seq_len", tone: "warn" },
      { label: "actual use", value: "often much shorter" },
      { label: "waste", value: "over-reservation", tone: "hot" },
    ],
    chain: ["support long ctx", "large reservation", "few sequences", "low util"],
  },
  {
    title: "paged 被实际长度支配",
    body: "paged allocation 只按实际 materialized token 分配 blocks。浪费主要来自最后一个没填满的 partial block。",
    visual: "blocks",
    formula: "paged_blocks = ceil(actual_len / block_size)",
    states: [
      { label: "reserve", value: "actual blocks", tone: "ok" },
      { label: "waste", value: "partial tail" },
      { label: "external fragmentation", value: "removed", tone: "hot" },
    ],
    chain: ["actual length", "ceil blocks", "tail waste", "higher concurrency"],
  },
  {
    title: "共享必须 ref count",
    body: "prefix cache 或 fork 会让多个 sequence 指向同一个 physical block。释放时不能直接 free，只能 decref。",
    visual: "blocks",
    states: [
      { label: "shared block", value: "ref_count > 1", tone: "hot" },
      { label: "free rule", value: "only when zero" },
      { label: "risk", value: "premature free corrupts readers", tone: "warn" },
    ],
    chain: ["sharing", "ref count", "safe free", "resident cache"],
  },
  {
    title: "COW 只保护 partial tail",
    body: "full shared prefix 已经填满，不会再写。shared partial tail 还可能追加 token，所以写之前必须 copy-on-write。",
    visual: "blocks",
    code: "cur % block_size != 0\nand ref_count[tail] > 1\n=> allocate private tail",
    states: [
      { label: "safe share", value: "full prefix", tone: "ok" },
      { label: "danger", value: "partial tail", tone: "warn" },
      { label: "fix", value: "copy-on-write", tone: "hot" },
    ],
    chain: ["shared tail", "next write", "would pollute parent", "private copy"],
  },
  {
    title: "收益换来状态复杂度",
    body: "paged KV 的收益是把 over-reservation 压成 partial block waste。代价是 block table、ref count、COW、cache residency 都要一起维护。",
    visual: "blocks",
    states: [
      { label: "effect", value: "higher KV utilization", tone: "ok" },
      { label: "new state", value: "tables + refs + COW", tone: "hot" },
      { label: "real vLLM", value: "KV cache manager" },
    ],
    chain: ["less waste", "more mappings", "more invariants", "real system complexity"],
  },
];

export default function PagedKV({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="pk-scene"
      step={step}
      eyebrow="04 / paged kv"
      title="Paged KV 与 BlockManager"
      hardQuestion="PagedAttention 为什么本质上是 KV cache 内存管理？"
      source="block_manager.py / analysis.py"
      invariant="logical sequence 连续；physical blocks 可不连续；free 必须遵守 ref count。"
      production="vLLM KV cache manager / block table / PagedAttention"
      steps={diveSteps}
    />
  );
}
