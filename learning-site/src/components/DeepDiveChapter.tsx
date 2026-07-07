import type { ChapterStepProps } from "../registry/types";
import "./DeepDiveChapter.css";

export type VisualKind =
  | "timeline"
  | "engine"
  | "sequence"
  | "blocks"
  | "queues"
  | "pressure"
  | "prefix"
  | "metrics";

export interface DiveStep {
  title: string;
  body: string;
  visual: VisualKind;
  code?: string;
  formula?: string;
  states?: Array<{ label: string; value: string; tone?: "hot" | "ok" | "warn" }>;
  chain?: string[];
  chips?: string[];
}

export interface DiveChapterProps extends ChapterStepProps {
  className?: string;
  eyebrow: string;
  title: string;
  hardQuestion: string;
  source: string;
  invariant: string;
  production: string;
  steps: DiveStep[];
}

const clampStep = (step: number, total: number) =>
  Math.max(0, Math.min(step, Math.max(0, total - 1)));

export function DeepDiveChapter({
  step,
  eyebrow,
  title,
  hardQuestion,
  source,
  invariant,
  production,
  steps,
  className,
}: DiveChapterProps) {
  const index = clampStep(step, steps.length);
  const item = steps[index]!;

  return (
    <section className={`dd-scene scene-pad ${className ?? ""}`}>
      <header className="dd-topbar">
        <div>
          <div className="label-mono">{eyebrow}</div>
          <h1>{title}</h1>
        </div>
        <div className="dd-counter label-mono">
          {String(index + 1).padStart(2, "0")} / {String(steps.length).padStart(2, "0")}
        </div>
      </header>

      <main className="dd-main">
        <article className="dd-brief">
          <div className="dd-hard">
            <span>Hard question</span>
            <strong>{hardQuestion}</strong>
          </div>

          <div className="dd-step-copy">
            <div className="dd-step-label label-mono">{item.visual}</div>
            <h2>{item.title}</h2>
            <p>{item.body}</p>
          </div>

          {item.formula && <div className="dd-formula">{item.formula}</div>}
          {item.code && <pre className="dd-code">{item.code}</pre>}

          {item.chips && (
            <div className="dd-chips">
              {item.chips.map((chip) => (
                <span key={chip}>{chip}</span>
              ))}
            </div>
          )}
        </article>

        <div className="dd-visual">
          <Visual kind={item.visual} index={index} step={item} />
        </div>
      </main>

      <footer className="dd-footer">
        <div>
          <span className="label-mono">source</span>
          <strong>{source}</strong>
        </div>
        <div>
          <span className="label-mono">invariant</span>
          <strong>{invariant}</strong>
        </div>
        <div>
          <span className="label-mono">production map</span>
          <strong>{production}</strong>
        </div>
      </footer>
    </section>
  );
}

function Visual({
  kind,
  index,
  step,
}: {
  kind: VisualKind;
  index: number;
  step: DiveStep;
}) {
  return (
    <div className={`dd-v dd-v-${kind}`}>
      <MechanismChain chain={step.chain} index={index} />
      {kind === "timeline" && <Timeline index={index} />}
      {kind === "engine" && <EngineLoop index={index} />}
      {kind === "sequence" && <TokenTape index={index} />}
      {kind === "blocks" && <BlockGrid index={index} />}
      {kind === "queues" && <QueueFlow index={index} />}
      {kind === "pressure" && <PressurePools index={index} />}
      {kind === "prefix" && <PrefixCache index={index} />}
      {kind === "metrics" && <MetricsMap index={index} />}
      <StateBoard states={step.states} />
    </div>
  );
}

function MechanismChain({ chain, index }: { chain?: string[]; index: number }) {
  if (!chain?.length) return null;
  return (
    <div className="dd-chain">
      {chain.map((node, i) => (
        <div
          key={node}
          className={`dd-chain-node ${i <= index % chain.length ? "is-on" : ""}`}
        >
          {node}
        </div>
      ))}
    </div>
  );
}

function StateBoard({
  states,
}: {
  states?: Array<{ label: string; value: string; tone?: "hot" | "ok" | "warn" }>;
}) {
  if (!states?.length) return null;
  return (
    <div className="dd-state-board">
      {states.map((s) => (
        <div key={s.label} className={`dd-state ${s.tone ?? ""}`}>
          <span>{s.label}</span>
          <strong>{s.value}</strong>
        </div>
      ))}
    </div>
  );
}

function Timeline({ index }: { index: number }) {
  const prefill = Math.min(7, 3 + index);
  const decode = Math.min(9, 2 + Math.max(0, index - 1));
  return (
    <div className="dd-timeline">
      <div className="dd-lane">
        <span className="label-mono">prefill</span>
        <div className="dd-token-row">
          {Array.from({ length: 10 }).map((_, i) => (
            <b key={i} className={i < prefill ? "fill" : ""} />
          ))}
        </div>
      </div>
      <div className="dd-lane">
        <span className="label-mono">decode</span>
        <div className="dd-token-row decode">
          {Array.from({ length: 10 }).map((_, i) => (
            <b key={i} className={i < decode ? "fill" : ""} />
          ))}
        </div>
      </div>
      <div className="dd-band">
        <span>compute-bound</span>
        <span>memory-bandwidth-bound</span>
      </div>
    </div>
  );
}

function EngineLoop({ index }: { index: number }) {
  const nodes = ["Scheduler", "StepWork", "Engine commit", "ModelRunner", "Metrics"];
  return (
    <div className="dd-loop">
      {nodes.map((node, i) => (
        <div key={node} className={`dd-loop-node ${i <= index % nodes.length ? "is-on" : ""}`}>
          <span>{String(i + 1).padStart(2, "0")}</span>
          <strong>{node}</strong>
        </div>
      ))}
      <div className="dd-loop-core">LLMEngine.step()</div>
    </div>
  );
}

function TokenTape({ index }: { index: number }) {
  const generated = Math.min(5, Math.max(0, index - 1));
  const computed = index >= 6 ? 0 : Math.min(13, 6 + index);
  const total = 14;
  return (
    <div className="dd-token-tape">
      <div className="dd-tape-labels">
        <span>prompt_len</span>
        <span>num_generated</span>
      </div>
      <div className="dd-tape">
        {Array.from({ length: total }).map((_, i) => {
          const isGenerated = i >= 9 && i < 9 + generated;
          const isComputed = i < computed;
          return (
            <b
              key={i}
              className={`${isGenerated ? "gen" : "prompt"} ${isComputed ? "computed" : ""}`}
            />
          );
        })}
      </div>
      <div className="dd-gate">
        <span>decode gate</span>
        <strong>{computed >= 9 + generated ? "open" : "blocked"}</strong>
      </div>
    </div>
  );
}

function BlockGrid({ index }: { index: number }) {
  const used = Math.min(19, 5 + index * 2);
  const shared = index >= 5;
  const cow = index >= 6;
  return (
    <div className="dd-blocks">
      <div className="dd-block-table">
        {["L0", "L1", "L2", "tail"].map((l, i) => (
          <div key={l} className={cow && i === 3 ? "cow" : shared && i < 2 ? "shared" : ""}>
            <span>{l}</span>
            <strong>P{12 + i * 7}</strong>
          </div>
        ))}
      </div>
      <div className="dd-grid">
        {Array.from({ length: 32 }).map((_, i) => (
          <b
            key={i}
            className={`${i < used ? "used" : ""} ${shared && (i === 4 || i === 11) ? "shared" : ""} ${cow && i === 27 ? "cow" : ""}`}
          />
        ))}
      </div>
    </div>
  );
}

function QueueFlow({ index }: { index: number }) {
  const lanes = [
    ["waiting", 6 - Math.min(3, index)],
    ["running", 3 + Math.min(4, index)],
    ["swapped", index > 5 ? 1 : 0],
    ["finished", Math.max(0, index - 3)],
  ] as const;
  return (
    <div className="dd-queues">
      {lanes.map(([name, count]) => (
        <div key={name} className="dd-queue">
          <span className="label-mono">{name}</span>
          <div>
            {Array.from({ length: Math.max(0, Math.min(8, count)) }).map((_, i) => (
              <b key={i}>{name[0]}{i}</b>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function PressurePools({ index }: { index: number }) {
  const gpu = Math.min(96, 58 + index * 7);
  const cpu = index >= 3 ? Math.min(54, 14 + index * 5) : 4;
  const watermark = 18;
  return (
    <div className="dd-pressure">
      <Pool label="gpu kv pool" value={gpu} watermark={watermark} />
      <Pool label="cpu swap pool" value={cpu} />
      <div className="dd-pressure-note">
        <span>admission gate</span>
        <strong>{gpu > 100 - watermark ? "closed" : "open"}</strong>
      </div>
    </div>
  );
}

function Pool({
  label,
  value,
  watermark,
}: {
  label: string;
  value: number;
  watermark?: number;
}) {
  return (
    <div className="dd-pool">
      <div>
        <span className="label-mono">{label}</span>
        <strong>{value}%</strong>
      </div>
      <div className="dd-pool-bar">
        <b style={{ width: `${value}%` }} />
        {watermark && <i style={{ right: `${watermark}%` }} />}
      </div>
    </div>
  );
}

function PrefixCache({ index }: { index: number }) {
  const hit = index >= 3;
  const evict = index >= 7;
  return (
    <div className="dd-prefix">
      <div className="dd-hash">
        {["h[0,16)", "h[0,32)", "h[0,48)", "miss"].map((h, i) => (
          <div key={h} className={`${hit && i < 3 ? "hit" : ""} ${evict && i === 1 ? "evict" : ""}`}>
            <span>{h}</span>
            <strong>{i < 3 ? `P${20 + i}` : "break"}</strong>
          </div>
        ))}
      </div>
      <div className="dd-prefix-tape">
        {Array.from({ length: 12 }).map((_, i) => (
          <b key={i} className={`${hit && i < 9 ? "hit" : ""} ${i >= 9 ? "unique" : ""}`} />
        ))}
      </div>
      <div className="dd-cache-life">
        <span className={index < 6 ? "on" : ""}>pinned</span>
        <span className={index >= 6 && !evict ? "on" : ""}>evictable</span>
        <span className={evict ? "on" : ""}>free pool</span>
      </div>
    </div>
  );
}

function MetricsMap({ index }: { index: number }) {
  const metrics = ["TTFT", "ITL", "throughput", "peak KV", "preempt", "saved prefill"];
  return (
    <div className="dd-metrics">
      <div className="dd-metric-grid">
        {metrics.map((m, i) => (
          <div key={m} className={i <= index % metrics.length ? "is-on" : ""}>
            <span>{m}</span>
            <strong>{["p99", "gap", "tok/s", "%", "count", "tokens"][i]}</strong>
          </div>
        ))}
      </div>
      <div className="dd-map">
        {["LLMEngine", "Scheduler", "BlockManager", "ModelRunner", "vLLM"].map((n, i) => (
          <b key={n} className={i <= Math.min(index, 4) ? "is-on" : ""}>{n}</b>
        ))}
      </div>
    </div>
  );
}
