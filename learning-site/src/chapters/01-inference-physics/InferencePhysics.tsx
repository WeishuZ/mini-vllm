import { DeepDiveChapter, type DiveStep } from "../../components/DeepDiveChapter";
import type { ChapterStepProps } from "../../registry/types";
import "./InferencePhysics.css";

const diveSteps: DiveStep[] = [
  {
    title: "生成不是批量吐出",
    body: "decoder-only LLM 是 autoregressive 的。生成 100 个 token，不是一次 forward 出 100 个，而是每一步把已有上下文 materialize 后再预测下一个 token。",
    visual: "timeline",
    formula: "next_token = f(prompt + generated_so_far)",
    states: [
      { label: "unit", value: "one token / step", tone: "hot" },
      { label: "dependency", value: "full prefix" },
      { label: "risk", value: "history KV missing -> invalid decode", tone: "warn" },
    ],
    chain: ["autoregressive", "full prefix", "one token", "stateful KV"],
  },
  {
    title: "prefill 是大块计算",
    body: "prefill 处理已有 prompt。它一次吃很多 token，矩阵乘法规模大，GPU 更容易被算力喂饱，所以更接近 compute-bound。",
    visual: "timeline",
    formula: "prefill_cost ~= prefill_ms_per_token * tokens",
    states: [
      { label: "shape", value: "many prompt tokens" },
      { label: "bottleneck", value: "compute-bound", tone: "ok" },
      { label: "scheduler cost", value: "large chunk latency", tone: "warn" },
    ],
    chain: ["prompt block", "parallel matmul", "large chunk", "TTFT impact"],
  },
  {
    title: "decode 是窄步读取",
    body: "decode 每条 sequence 一次只生成一个 token。新计算很少，但 attention 要读历史 K/V，所以瓶颈更像 memory bandwidth。",
    visual: "timeline",
    formula: "decode_cost ~= decode_ms_base + decode_ms_per_seq * batch",
    states: [
      { label: "shape", value: "1 token / sequence" },
      { label: "bottleneck", value: "KV reads", tone: "hot" },
      { label: "batch effect", value: "per-token cost drops", tone: "ok" },
    ],
    chain: ["one token", "read history", "memory bandwidth", "wide batch"],
  },
  {
    title: "KV 是动态成本",
    body: "权重是固定成本，KV cache 随并发、上下文和输出长度增长。很多 serving 场景里，并发上限不是权重决定的，而是 KV resident set 决定的。",
    visual: "timeline",
    formula: "KV/token = 2 * layers * kv_heads * head_dim * dtype_bytes",
    states: [
      { label: "weights", value: "fixed" },
      { label: "kv cache", value: "dynamic", tone: "hot" },
      { label: "capacity", value: "concurrency bound" },
    ],
    chain: ["more requests", "more tokens", "more KV", "lower capacity"],
  },
  {
    title: "两种 forward，两种系统性格",
    body: "prefill 喜欢大 token chunk，decode 喜欢很多 sequence 一起 batch。一个服务端调度器必须同时照顾吞吐和交互延迟。",
    visual: "timeline",
    states: [
      { label: "prefill wants", value: "large chunks" },
      { label: "decode wants", value: "wide batch" },
      { label: "conflict", value: "TTFT vs ITL vs throughput", tone: "warn" },
    ],
    chain: ["prefill chunk", "decode stream", "mixed batch", "policy tradeoff"],
  },
  {
    title: "ModelRunner 是瓶颈抽象",
    body: "miniVLLM 不跑真实 transformer，但 `ModelRunner` 保留了 prefill 线性成本和 decode batch sublinear 成本。这个 cost model 是为了让调度结果可解释。",
    visual: "timeline",
    code: "latency += prefill_ms_per_token * work.num_prefill_tokens\nlatency += decode_ms_base + decode_ms_per_seq * d",
    states: [
      { label: "source", value: "model_runner.py" },
      { label: "prefill", value: "linear tokens" },
      { label: "decode", value: "batch-shaped" },
    ],
    chain: ["simulated cost", "deterministic run", "explain scheduler", "benchmarkable"],
  },
  {
    title: "真实 vLLM 只是层更多",
    body: "真实 vLLM 会把这些约束落到 model runner、attention backend、CUDA kernel、scheduler policy 和 chunked prefill。核心问题仍然是怎么混合 prefill 和 decode。",
    visual: "metrics",
    states: [
      { label: "mini", value: "ModelRunner" },
      { label: "real", value: "worker + attention backend", tone: "ok" },
      { label: "shared problem", value: "prefill/decode mix", tone: "hot" },
    ],
    chain: ["mini model", "real worker", "kernels", "scheduler policy"],
  },
];

export default function InferencePhysics({ step }: ChapterStepProps) {
  return (
    <DeepDiveChapter
      className="ip-scene"
      step={step}
      eyebrow="01 / inference physics"
      title="LLM 推理的物理形态"
      hardQuestion="为什么 prefill 和 decode 不能用同一种调度直觉？"
      source="model_runner.py / config.py"
      invariant="decode 每条 sequence 每 step 最多生成一个 token；KV 必须覆盖完整前缀。"
      production="vLLM model runner / attention backend / chunked prefill"
      steps={diveSteps}
    />
  );
}
