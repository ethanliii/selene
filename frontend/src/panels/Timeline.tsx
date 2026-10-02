/**
 * Bottom timeline: play/pause, rewind, speed, UTC readout with the current / next event beside it, and a scrubber
 * with hour gridlines + labels, coloured event ticks (hover tooltips), a custody ribbon coloured by the
 * protagonist's custody per scenario frame and labelled pips for the six story beats
 * (Burn · Detected · Lost · Tasked · Regained · Brief).
 */
import { useMemo, useState } from 'react';
import type { SeleneEvent } from '../api/types';
import { BEAT_LABELS, beatOfEvent, deriveHeadline, kindLabel, type Beat } from '../demo/headline';
import { SPEEDS, cursorDate, fmtAge, fmtElapsed, fmtUtc, useSelene } from '../store/useSelene';
import { Icon } from './icons';
import { severityColor } from './severity';

function speedLabel(s: number): string {
  return s >= 1000 ? `${Math.round(s).toLocaleString('en-US')}×` : `${s}×`;
}

const BEAT_COLOR: Record<Beat, string> = {
  burn: 'var(--muted)',
  detected: 'var(--alert)',
  lost: 'var(--alert)',
  tasked: 'var(--accent)',
  regained: 'var(--ok)',
  brief: 'var(--accent-2)',
};

interface Pip {
  x: number;
  t: number;
  beats: Beat[];
  color: string;
  objectId?: string;
  /** Label row (0–2): neighbours closer than 7 % of the span stack their labels on alternating rows. */
  row: number;
}

export function Timeline() {
  const t0Iso = useSelene((s) => s.t0Iso);
  const tSec = useSelene((s) => s.tSec);
  const t0 = useSelene((s) => s.t0Sec);
  const t1 = useSelene((s) => s.t1Sec);
  const playing = useSelene((s) => s.playing);
  const speed = useSelene((s) => s.speed);
  const events = useSelene((s) => s.events);
  const scenario = useSelene((s) => s.scenario);
  const setT = useSelene((s) => s.setT);
  const toggle = useSelene((s) => s.togglePlaying);
  const setSpeed = useSelene((s) => s.setSpeed);
  const setPlaying = useSelene((s) => s.setPlaying);
  const setFinished = useSelene((s) => s.setScenarioFinished);
  const finished = useSelene((s) => s.scenarioFinished);
  const select = useSelene((s) => s.selectObject);
  const [hover, setHover] = useState<{ e: SeleneEvent; x: number } | null>(null);

  const span = Math.max(1, t1 - t0);
  const scripted = scenario?.meta.playback_speed;
  const options = [...SPEEDS];
  if (scripted && !options.includes(scripted)) options.push(scripted);
  if (!options.includes(speed)) options.push(speed);
  options.sort((a, b) => a - b);
  const xOf = (t: number) => (100 * (t - t0)) / span;

  // Hour gridlines: minor every 6 h, labelled every 6 h for spans ≤ 2 d, every 24 h beyond.
  const grid = useMemo(() => {
    const base = Date.parse(t0Iso);
    const minor = 6 * 3600;
    const majorEvery = span <= 2 * 86400 + 1 ? 6 * 3600 : 24 * 3600;
    const out: { x: number; major: boolean; label: string | null }[] = [];
    const firstMs = Math.ceil((base + t0 * 1000) / (minor * 1000)) * minor * 1000;
    for (let ms = firstMs; ms <= base + t1 * 1000 + 1; ms += minor * 1000) {
      const t = (ms - base) / 1000;
      const d = new Date(ms);
      const major = ms % (majorEvery * 1000) === 0;
      const label = major ? (d.getUTCHours() === 0 ? d.toISOString().slice(5, 10) : `${String(d.getUTCHours()).padStart(2, '0')}:00`) : null;
      out.push({ x: xOf(t), major, label });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [t0Iso, t0, t1, span]);

  // Custody ribbon from the protagonist's custody in each scenario frame (runs merged into segments).
  const ribbon = useMemo(() => {
    if (!scenario) return [] as { x0: number; x1: number; c: string; t1: number }[];
    const pid = scenario.meta.protagonist_id ?? scenario.frames[0]?.objects[0]?.id;
    const segs: { x0: number; x1: number; c: string; t1: number }[] = [];
    const fr = scenario.frames;
    for (let i = 0; i < fr.length; i++) {
      const c = fr[i].objects.find((o) => o.id === pid)?.custody ?? 'unknown';
      const tA = fr[i].t;
      const tB = i + 1 < fr.length ? fr[i + 1].t : t1;
      const last = segs[segs.length - 1];
      if (last && last.c === c) {
        last.x1 = xOf(tB);
        last.t1 = tB;
      } else segs.push({ x0: xOf(tA), x1: xOf(tB), c, t1: tB });
    }
    return segs;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario, t0, t1]);

  // Story-beat pips (merged when closer than 2.5 % of the span so their labels do not collide).
  const pips = useMemo(() => {
    if (!scenario) return [] as Pip[];
    const seen = new Set<Beat>();
    const raw: Pip[] = [];
    for (const e of events) {
      const b = beatOfEvent(e);
      if (!b || seen.has(b)) continue;
      seen.add(b);
      raw.push({ x: xOf(e.t), t: e.t, beats: [b], color: BEAT_COLOR[b], objectId: e.object_id, row: 0 });
    }
    raw.sort((a, b) => a.x - b.x);
    // Same-time beats share one pip ("Tasked · Regained"); near neighbours stack their labels on rows 0/1/2.
    const merged: Pip[] = [];
    for (const p of raw) {
      const last = merged[merged.length - 1];
      if (last && p.x - last.x < 0.6) {
        last.beats.push(...p.beats);
        last.color = p.color;
      } else merged.push({ ...p, beats: [...p.beats] });
    }
    for (let i = 1; i < merged.length; i++) {
      const prev = merged.filter((q, k) => k < i && merged[i].x - q.x < 7);
      const used = new Set(prev.map((q) => q.row));
      merged[i].row = [0, 1, 2].find((r) => !used.has(r)) ?? i % 3;
    }
    return merged;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scenario, events, t0, t1]);

  const reveal = !scenario || finished;
  const skipToEvent = useSelene((s) => s.skipToEvent);
  const enqueueCaptions = useSelene((s) => s.enqueueCaptions);
  /** Jump to the next/previous story event (observations skipped) and narrate just that one. */
  const skipEvent = (dir: 1 | -1) => {
    const ev = skipToEvent(dir);
    if (ev) enqueueCaptions([ev], true);
  };
  // "Now" = the event being narrated (one at a time, so coincident beats each get their turn), else the last past one.
  const narrated = useSelene((s) => s.caption);
  const pastEvents = events.filter((e) => e.t <= tSec && e.kind !== 'observation');
  const cur = narrated && narrated.t <= tSec ? narrated : pastEvents[pastEvents.length - 1];
  const next = events.find((e) => e.t > tSec && e.kind !== 'observation');
  const curPast = (e: SeleneEvent) => e.t <= tSec;

  return (
    <div className="timeline-inner">
      <div className="controls">
        <button
          className="toggle"
          title="Rewind to start"
          onClick={() => {
            setPlaying(false);
            setFinished(false);
            setT(t0);
          }}
        >
          <Icon name="rewind" style={{ marginRight: 0, verticalAlign: '-2px' }} />
        </button>
        <button className="primary" onClick={toggle} style={{ minWidth: 92 }}>
          {playing ? <Icon name="pause" /> : <Icon name="play" />}
          {playing ? 'Pause' : 'Play'}
        </button>
        {scenario && (
          <>
            <button className="toggle skip" title="Back to the previous story event" onClick={() => skipEvent(-1)} aria-label="Previous event">
              ‹ ev
            </button>
            <button className="toggle skip" title={next ? `Skip to the next story event (${reveal ? kindLabel(next.kind).toLowerCase() : 'in ' + fmtAge(next.t - tSec)})` : 'No further events'} onClick={() => skipEvent(1)} disabled={!next} aria-label="Next event">
              ev ›
            </button>
          </>
        )}
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} title={`Playback speed: ${speedLabel(speed)} sim seconds per wall-clock second${scripted ? ` · scripted story = ${speedLabel(scripted)}` : ''}`}>
          {options.map((s) => (
            <option key={s} value={s} title={`${speedLabel(s)} sim seconds per wall-clock second`}>
              {s === scripted ? `Story pace (${speedLabel(s)})` : speedLabel(s)}
            </option>
          ))}
        </select>
        <span className="utc">{fmtUtc(cursorDate(t0Iso, tSec))}</span>
        <span className="elapsed">T{fmtElapsed(tSec)}</span>
        {scenario && (cur || next) && (
          <span className="now-ev" style={{ ['--sev' as string]: cur ? severityColor(cur.severity) : 'var(--muted)' }}>
            {cur && (
              <>
                <span className="kind">{kindLabel(cur.kind)}</span>
                <span className="h">{cur.headline ?? deriveHeadline(cur)}</span>
              </>
            )}
            {next && <span className="next">· next {reveal ? kindLabel(next.kind).toLowerCase() : 'event'} in {fmtAge(next.t - tSec)}</span>}
          </span>
        )}
        <span className="spacer" style={{ flex: 1 }} />
        {scenario ? (
          <span className="ribbon-key" title="Custody ribbon: protagonist custody per scenario frame">
            <span>
              <i style={{ background: 'var(--held)' }} />
              held
            </span>
            <span>
              <i style={{ background: 'var(--degraded)' }} />
              degraded
            </span>
            <span>
              <i style={{ background: 'var(--lost)' }} />
              lost
            </span>
          </span>
        ) : (
          <span className="elapsed">
            span {fmtElapsed(t0).slice(1)} → {fmtElapsed(t1).slice(1)} · {events.length} events
          </span>
        )}
      </div>
      <div className="scrubber" onMouseLeave={() => setHover(null)}>
        <div className="hgrid">
          {grid.map((g, i) => (
            <span key={i} className={`hl${g.major ? ' major' : ''}`} style={{ left: `${g.x}%` }} />
          ))}
          {grid
            .filter((g) => g.label)
            .map((g, i, arr) => (
              <span key={i} className={`hlbl${g.x < 2 ? ' first' : ''}${g.x > 98 ? ' last' : ''}`} style={{ left: `${g.x}%`, display: i > 0 && g.x - arr[i - 1].x < 4 ? 'none' : undefined }}>
                {g.label}
              </span>
            ))}
        </div>
        <div className="marks">
          {events.map((e, i) => {
            const x = xOf(e.t);
            const beat = beatOfEvent(e) !== null;
            return (
              <span
                key={i}
                className={`mark${curPast(e) ? ' past' : ''}${beat ? ' beat' : ''}`}
                style={{ left: `${x}%`, ['--sev' as string]: severityColor(e.severity) }}
                onMouseEnter={() => setHover({ e, x })}
                onClick={() => {
                  setT(e.t);
                  if (e.object_id) select(e.object_id);
                }}
              />
            );
          })}
          <span className="cursor" style={{ left: `${xOf(tSec)}%` }} />
        </div>
        <input type="range" min={t0} max={t1} step={Math.max(1, span / 4000)} value={tSec} onChange={(e) => setT(Number(e.target.value))} aria-label="Time cursor" />
        {scenario && (
          <>
            <div className="ribbon" aria-hidden>
              {ribbon.map((s, i) => (
                <span key={i} className={`seg ${s.c}${s.t1 > tSec && s.x0 >= xOf(tSec) ? ' future' : ''}`} style={{ left: `${s.x0}%`, width: `${Math.max(0.05, s.x1 - s.x0)}%` }} />
              ))}
            </div>
            <div className="pips">
              {pips.map((p, i) => {
                const past = p.t <= tSec;
                const name = p.beats.map((b) => BEAT_LABELS[b]).join(' · ');
                return (
                  <span
                    key={i}
                    className={`pip${past ? ' past' : ' future'}`}
                    style={{ left: `${p.x}%`, ['--pip' as string]: p.color }}
                    title={reveal || past ? `${name} · T${fmtElapsed(p.t)}` : `Upcoming beat · T${fmtElapsed(p.t)}`}
                    onClick={() => {
                      setT(p.t);
                      if (p.objectId) select(p.objectId);
                    }}
                  >
                    <span className={`lbl${reveal || past ? '' : ' hidden-name'}`} style={{ top: 9 + p.row * 13 }}>
                      {reveal || past ? name : '·'}
                    </span>
                  </span>
                );
              })}
            </div>
          </>
        )}
        {hover &&
          (hover.e.t <= tSec || !scenario || finished ? (
            <div className="tip" style={{ left: `${hover.x}%`, ['--sev' as string]: severityColor(hover.e.severity) }}>
              <div className="meta">
                <span className="kind">{kindLabel(hover.e.kind)}</span>
                <span>T{fmtElapsed(hover.e.t)}</span>
                <span>{fmtUtc(cursorDate(t0Iso, hover.e.t))}</span>
              </div>
              <div style={{ fontWeight: 600 }}>{hover.e.headline ?? deriveHeadline(hover.e)}</div>
              <div className="muted" style={{ fontSize: 12, marginTop: 2 }}>
                {hover.e.text}
              </div>
            </div>
          ) : (
            <div className="tip" style={{ left: `${hover.x}%`, ['--sev' as string]: 'var(--muted)' }}>
              <div className="meta">
                <span className="kind">upcoming</span>
                <span>T{fmtElapsed(hover.e.t)}</span>
                <span>{fmtUtc(cursorDate(t0Iso, hover.e.t))}</span>
              </div>
              <div className="muted">Event not reached yet — click to jump there.</div>
            </div>
          ))}
      </div>
    </div>
  );
}
