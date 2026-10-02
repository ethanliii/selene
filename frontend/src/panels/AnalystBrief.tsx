/**
 * Analyst brief: renders a markdown-ish string (headings '#', bullets '-'/'*', blank-line paragraphs,
 * **bold**, `code`) without a markdown dependency, plus a "generated <UTC>" footer and the scenario disclaimer.
 * Only the trusted backend (or the browser mock) produces this string.
 */
import { useMemo, type ReactNode } from 'react';
import { fmtElapsed, fmtUtc, useSelene } from '../store/useSelene';

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

/** Time at which the brief becomes available: the scenario's 'brief' event, else the end of the span. */
function briefReleaseTime(events: { kind: string; t: number }[], t1: number): number {
  const e = events.find((x) => x.kind === 'brief');
  return e ? e.t : t1;
}

export function AnalystBrief({ text }: { text?: string }) {
  const storeBrief = useSelene((s) => s.brief);
  const scenario = useSelene((s) => s.scenario);
  const finished = useSelene((s) => s.scenarioFinished);
  // Boolean selector: re-renders only when the brief is released/withdrawn, not on every tick.
  const released = useSelene((s) => !s.scenario || s.scenarioFinished || s.tSec >= briefReleaseTime(s.events, s.t1Sec));
  const releaseT = useSelene((s) => briefReleaseTime(s.events, s.t1Sec));
  const src = text ?? storeBrief;
  const blocks = useMemo(() => parseBrief(src), [src]);
  const generated = scenario ? (scenario.meta.brief_generated_utc ? new Date(scenario.meta.brief_generated_utc) : new Date(Date.parse(scenario.meta.t0_utc) + scenario.meta.duration_s * 1000)) : null;
  if (blocks.length > 0 && !released && !text) {
    return (
      <div className="brief">
        <h3 className="panel-title">Analyst brief</h3>
        <p className="empty">
          The brief is generated automatically when the story reaches T{fmtElapsed(releaseT)}; it is not shown earlier so the outcome is not spoiled. Use the Events tab's "Upcoming" toggle for a presenter preview.
        </p>
      </div>
    );
  }
  return (
    <div className="brief">
      <h3 className="panel-title">Analyst brief</h3>
      {blocks.length === 0 ? (
        <p className="empty">The auto-generated brief appears here at the end of the demo scenario.</p>
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
            {scenario?.meta.disclaimer && <div className="muted">{scenario.meta.disclaimer}</div>}
          </div>
        </>
      )}
    </div>
  );
}
