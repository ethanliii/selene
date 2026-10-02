/**
 * Analyst brief: renders a markdown-ish string (headings '#', bullets '-'/'*', blank-line paragraphs,
 * **bold**, `code`) without a markdown dependency, plus a provenance / "generated <UTC>" footer and the scenario
 * disclaimer. Before the story reaches its brief event the tab shows an "in progress" brief: classification
 * line, the story beats reached so far and a progress bar to the auto-generation time (the outcome is not spoiled).
 * Only the trusted backend (or the browser mock) produces the brief string.
 */
import { useMemo, type ReactNode } from 'react';
import { useEndpointStatus } from '../api/client';
import { BEATS, BEAT_LABELS, beatOfEvent, deriveHeadline, isBriefKind, type Beat } from '../demo/headline';
import { startDemo } from '../demo/useDemoDriver';
import { fmtElapsed, fmtUtc, useSelene } from '../store/useSelene';
import { severityColor } from './severity';

type Block = { type: 'h'; level: number; text: string } | { type: 'p'; text: string } | { type: 'ul'; items: string[] };

export function parseBrief(src: string): Block[] {
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] = [];
  const flush = () => {
    if (para.length) blocks.push({ type: 'p', text: para.join(' ') });
    if (list.length) blocks.push({ type: 'ul', items: list });
    para = [];
    list = [];
  };
  for (const raw of src.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) {
      flush();
      continue;
    }
    const h = /^(#{1,6})\s+(.*)$/.exec(line);
    if (h) {
      flush();
      blocks.push({ type: 'h', level: h[1].length, text: h[2] });
      continue;
    }
    const li = /^[-*•]\s+(.*)$/.exec(line);
    if (li) {
      if (para.length) {
        blocks.push({ type: 'p', text: para.join(' ') });
        para = [];
      }
      list.push(li[1]);
      continue;
    }
    if (list.length) {
      blocks.push({ type: 'ul', items: list });
      list = [];
    }
    para.push(line);
  }
  flush();
  return blocks;
}

/** Inline **bold** and `code`. */
function inline(text: string): ReactNode[] {
  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
  return parts.map((p, i) => {
    if (p.startsWith('**') && p.endsWith('**')) return <strong key={i}>{p.slice(2, -2)}</strong>;
    if (p.startsWith('`') && p.endsWith('`')) return <code key={i} className="mono">{p.slice(1, -1)}</code>;
    return <span key={i}>{p}</span>;
  });
}

/** Time at which the brief becomes available: the scenario's brief event, else the end of the span. */
function briefReleaseTime(events: { kind: string; t: number }[], t1: number): number {
  const e = events.find((x) => isBriefKind(x.kind));
  return e ? e.t : t1;
}

function Provenance({ live }: { live: boolean }) {
  return (
    <div className="prov">
      source: <b>{live ? 'backend /api/demo/scenario — DE440s N-body truth, UKF custody, backend tasker' : 'browser mock — CR3BP dynamics and emulated filter in the browser'}</b>
    </div>
  );
}

function InProgress({ releaseT }: { releaseT: number }) {
  const events = useSelene((s) => s.events);
  const t0 = useSelene((s) => s.t0Sec);
  // Coarse cursor (1 min) so the progress bar re-renders a few times a second, not every frame.
  const tSec = useSelene((s) => Math.floor(s.tSec / 60) * 60);
  const reached = new Map<Beat, (typeof events)[number]>();
  for (const e of events) {
    const b = beatOfEvent(e);
    if (b && e.t <= tSec && !reached.has(b)) reached.set(b, e);
  }
  const pct = Math.max(0, Math.min(100, (100 * (tSec - t0)) / Math.max(1, releaseT - t0)));
  return (
    <div className="brief">
      <h3 className="panel-title">Analyst brief · in progress</h3>
      <p className="class-line">UNCLASSIFIED // DEMONSTRATION — SIMULATED, NOTIONAL ACTOR</p>
      <p className="muted small" style={{ margin: '0 0 6px' }}>
        Story beats reached so far; the brief auto-generates once the maneuver is characterised.
      </p>
      <ul className="beats">
        {BEATS.map((b) => {
          const e = reached.get(b);
          return (
            <li key={b} className={e ? '' : 'pending'} style={{ ['--sev' as string]: e ? severityColor(e.severity) : 'var(--muted)' }}>
              <span className="dot" />
              <span>{e ? (e.headline ?? deriveHeadline(e)) : `${BEAT_LABELS[b]} — pending`}</span>
              <span className="t">{e ? `T${fmtElapsed(e.t)}` : ''}</span>
            </li>
          );
        })}
      </ul>
      <div className="brief-progress">
        <div className="bar">
          <span style={{ width: `${pct}%` }} />
        </div>
        <div className="lbl">
          brief auto-generates at T{fmtElapsed(releaseT)} · {pct.toFixed(0)}%
        </div>
      </div>
      <p className="muted small">Use the Events tab's "Upcoming" toggle for a presenter preview of the remaining beats.</p>
    </div>
  );
}

/** Before any scenario is loaded: what the brief is, where it comes from, and the beats it will cover. */
function Idle() {
  const ready = useSelene((s) => !!s.ephemeris);
  return (
    <div className="brief-idle">
      <p className="class-line">UNCLASSIFIED // DEMONSTRATION — SIMULATED, NOTIONAL ACTOR</p>
      <p>The analyst brief is auto-generated by the backend scenario engine once the maneuver has been characterised: the detection, the custody gap, the reachability call and the recovery, in plain English with the numbers behind each.</p>
      <ul className="beats">
        {BEATS.map((b) => (
          <li key={b} className="pending" style={{ ['--sev' as string]: 'var(--muted)' }}>
            <span className="dot" />
            <span>{BEAT_LABELS[b]}</span>
          </li>
        ))}
      </ul>
      <button className="primary wide" onClick={() => void startDemo()} disabled={!ready} title={ready ? 'Play the scripted 2-minute scenario' : 'Waiting for the ephemeris basis'}>
        ▶ Play the 2-minute scenario
      </button>
    </div>
  );
}

export function AnalystBrief({ text }: { text?: string }) {
  const storeBrief = useSelene((s) => s.brief);
  const scenario = useSelene((s) => s.scenario);
  const finished = useSelene((s) => s.scenarioFinished);
  // Boolean selector: re-renders only when the brief is released/withdrawn, not on every tick.
  const released = useSelene((s) => !s.scenario || s.scenarioFinished || s.tSec >= briefReleaseTime(s.events, s.t1Sec));
  const releaseT = useSelene((s) => briefReleaseTime(s.events, s.t1Sec));
  const endpointLive = useEndpointStatus().demo === 'live';
  // The bundle's own provenance tag wins; the endpoint registry is the fallback for a scenario loaded elsewhere.
  const live = scenario?.meta.source ? scenario.meta.source === 'backend' : endpointLive;
  const src = text ?? storeBrief;
  const blocks = useMemo(() => parseBrief(src), [src]);
  const generated = scenario ? (scenario.meta.brief_generated_utc ? new Date(scenario.meta.brief_generated_utc) : new Date(Date.parse(scenario.meta.t0_utc) + scenario.meta.duration_s * 1000)) : null;
  if (blocks.length > 0 && !released && !text) return <InProgress releaseT={releaseT} />;
  return (
    <div className="brief">
      <h3 className="panel-title">Analyst brief</h3>
      {blocks.length === 0 ? (
        <Idle />
      ) : (
        <>
          {scenario && !finished && <p className="muted small">Generated at T{fmtElapsed(releaseT)}; the story is still playing.</p>}
          {blocks.map((b, i) =>
            b.type === 'h' ? (
              b.level <= 1 ? (
                <h3 key={i} className="brief-h1">
                  {inline(b.text)}
                </h3>
              ) : (
                <h4 key={i}>{inline(b.text)}</h4>
              )
            ) : b.type === 'ul' ? (
              <ul key={i}>
                {b.items.map((it, j) => (
                  <li key={j}>{inline(it)}</li>
                ))}
              </ul>
            ) : (
              <p key={i}>{inline(b.text)}</p>
            ),
          )}
          <div className="brief-footer mono">
            {generated && <div>generated {fmtUtc(generated)}</div>}
            {scenario && <Provenance live={live} />}
            {scenario?.meta.disclaimer && <div className="muted">{scenario.meta.disclaimer}</div>}
          </div>
        </>
      )}
    </div>
  );
}
