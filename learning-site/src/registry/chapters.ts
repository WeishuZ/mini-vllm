import type { ChapterDef } from "./types";
import InferencePhysics from "../chapters/01-inference-physics/InferencePhysics";
import { steps as inferencePhysicsSteps } from "../chapters/01-inference-physics/steps";
import EngineStep from "../chapters/02-engine-step/EngineStep";
import { steps as engineStepSteps } from "../chapters/02-engine-step/steps";
import SequenceAccounting from "../chapters/03-sequence-accounting/SequenceAccounting";
import { steps as sequenceAccountingSteps } from "../chapters/03-sequence-accounting/steps";
import PagedKV from "../chapters/04-paged-kv/PagedKV";
import { steps as pagedKVSteps } from "../chapters/04-paged-kv/steps";
import SchedulerChapter from "../chapters/05-scheduler/SchedulerChapter";
import { steps as schedulerSteps } from "../chapters/05-scheduler/steps";
import MemoryPressure from "../chapters/06-memory-pressure/MemoryPressure";
import { steps as memoryPressureSteps } from "../chapters/06-memory-pressure/steps";
import PrefixCacheChapter from "../chapters/07-prefix-cache/PrefixCacheChapter";
import { steps as prefixCacheSteps } from "../chapters/07-prefix-cache/steps";
import ObservabilityVllm from "../chapters/08-observability-vllm/ObservabilityVllm";
import { steps as observabilitySteps } from "../chapters/08-observability-vllm/steps";

/**
 * Order = order of presentation.
 *
 * Each chapter MUST provide a `steps: StepDef[]` array. Its length is the
 * chapter's step count, so the runtime stepper and the chapter `.tsx`
 * switch on `step` cannot drift apart.
 *
 * Visual styling (color, fonts) comes entirely from the active theme —
 * chapters never hard-code palette / font names. See THEMES.md.
 */
export const CHAPTERS: ChapterDef[] = [
  {
    id: "inference-physics",
    title: "LLM 推理的物理形态",
    steps: inferencePhysicsSteps,
    Component: InferencePhysics,
  },
  {
    id: "engine-step",
    title: "LLMEngine.step() 的提交边界",
    steps: engineStepSteps,
    Component: EngineStep,
  },
  {
    id: "sequence-accounting",
    title: "Sequence 的 token/KV 不变量",
    steps: sequenceAccountingSteps,
    Component: SequenceAccounting,
  },
  {
    id: "paged-kv",
    title: "Paged KV 与 BlockManager",
    steps: pagedKVSteps,
    Component: PagedKV,
  },
  {
    id: "scheduler",
    title: "Continuous Batching 与 token budget",
    steps: schedulerSteps,
    Component: SchedulerChapter,
  },
  {
    id: "memory-pressure",
    title: "Preemption、Swap 与 Watermark",
    steps: memoryPressureSteps,
    Component: MemoryPressure,
  },
  {
    id: "prefix-cache",
    title: "Prefix Cache 的 hash、共享与生命周期",
    steps: prefixCacheSteps,
    Component: PrefixCacheChapter,
  },
  {
    id: "observability-vllm",
    title: "Metrics、Trace 与真实 vLLM 映射",
    steps: observabilitySteps,
    Component: ObservabilityVllm,
  },
];
