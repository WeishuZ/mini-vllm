import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./PrefixCacheChapter.css";

const diveSteps: DiveStep[] = [
  {
    title: "缓存的是 KV block",
    body: "prefix cache 不是缓存 prompt 字符串，也不是缓存输出。它缓存的是已经 materialized 的 full prompt KV block。",
    visual: "prefix",
    states: [
      { label: "not", value: "text / token list / output" },
      { label: "cache object", value: "physical KV block", tone: "hot" },
      { label: "scope", value: "full prompt blocks" },
    ],
    chain: ["shared prefix", "computed KV", "publish block", "reuse"],
  },
  {
    title: "publish 要求语义稳定",
    body: "`_register_full_prompt_blocks()` 只发布完整 prompt block，而且要求当前 sequence 独占这个 block。这样 cache entry 才有稳定语义。",
    visual: "prefix",
    code: "if ref_count[bid] != 1:\n    continue",
    states: [
      { label: "publish when", value: "full prompt block", tone: "ok" },
      { label: "ownership", value: "exclusive", tone: "hot" },
      { label: "avoid", value: "unstable shared write", tone: "warn" },
    ],
    chain: ["prefill done", "full block", "exclusive", "cache entry"],
  },
  {
    title: "hash 是完整前缀",
    body: "`_block_hash(token_ids, end)` hash 的是 `[0, end)`，不是当前 block。因为同样的 block token，在不同历史前缀下 KV 不一样。",
    visual: "prefix",
    formula: "hash = H(tokens[0:end])",
    states: [
      { label: "wrong", value: "hash current block only", tone: "warn" },
      { label: "right", value: "hash full prefix", tone: "hot" },
      { label: "reason", value: "attention depends on history" },
    ],
    chain: ["same local tokens", "different prefix", "different KV", "full-prefix hash"],
  },
  {
    title: "admit 只能连续命中",
    body: "`admit_prefix()` 从 token 0 开始查。中间一个 block miss，后面就算 token 内容相同，也不能跳着共享。",
    visual: "prefix",
    states: [
      { label: "query", value: "leading full blocks" },
      { label: "hit rule", value: "contiguous from zero", tone: "hot" },
      { label: "break", value: "first miss stops" },
    ],
    chain: ["token 0", "block 1", "block 2", "miss", "stop"],
  },
  {
    title: "命中直接推进 computed",
    body: "命中后，新 sequence 的 block table 直接 append shared physical block ids，`num_computed` 跳到 covered token 数。",
    visual: "prefix",
    code: "seq.block_table.append(bid)\nseq.num_computed = covered",
    states: [
      { label: "shared", value: "same physical block", tone: "ok" },
      { label: "computed", value: "covered tokens", tone: "hot" },
      { label: "counter", value: "saved_prefill += covered" },
    ],
    chain: ["cache hit", "ref_count++", "block table append", "skip prefill"],
  },
  {
    title: "省 prefill，不省 decode",
    body: "prefix cache 只能复用已有前缀的 KV。后续新生成 token 的 KV 还不存在，所以 decode 仍然要一步一步做。",
    visual: "timeline",
    states: [
      { label: "saved", value: "prompt prefill", tone: "ok" },
      { label: "not saved", value: "future decode", tone: "warn" },
      { label: "metric", value: "saved prefill tokens", tone: "hot" },
    ],
    chain: ["shared prompt", "skip prefill", "first token sooner", "decode unchanged"],
  },
  {
    title: "pinned 和 evictable",
    body: "active sequence 引用 cached block 时，它是 pinned。sequence 结束后，ref count 可以变 0，但 block 仍 resident，变成 evictable。",
    visual: "prefix",
    states: [
      { label: "active", value: "pinned", tone: "hot" },
      { label: "idle cached", value: "evictable", tone: "warn" },
      { label: "still", value: "occupies GPU block" },
    ],
    chain: ["active ref", "free sequence", "resident cache", "evictable"],
  },
  {
    title: "收益也占显存",
    body: "LRU eviction 在 cache budget 超限或 allocation pressure 时回收 idle cached block。prefix cache 的收益是降低未来 prefill，代价是占用 resident KV。",
    visual: "prefix",
    states: [
      { label: "benefit", value: "less repeated prefill", tone: "ok" },
      { label: "cost", value: "resident KV pressure", tone: "warn" },
      { label: "control", value: "LRU + budget", tone: "hot" },
    ],
    chain: ["retain", "future hit", "memory pressure", "evict idle"],
  },
];

export default function PrefixCacheChapter({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="pc-scene"
      step={step}
      eyebrow="07 / prefix cache"
      title="Prefix Cache 的 hash、共享与生命周期"
      hardQuestion="prefix cache 为什么只能连续命中，并且为什么省 prefill 不省 decode？"
      source="block_manager.py::_block_hash / admit_prefix"
      invariant="cache hit 必须从 token 0 连续；active cached block 不可 eviction。"
      production="vLLM automatic prefix caching / block hash / cache eviction"
      steps={diveSteps}
    />
  );
}
