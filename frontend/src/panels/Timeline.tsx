/** Bottom timeline: play/pause, rewind, speed, UTC readout, scrubber with coloured event ticks + hover tooltips. */
import { useState } from 'react';
import type { SeleneEvent } from '../api/types';
import { SPEEDS, cursorDate, fmtElapsed, fmtUtc, useSelene } from '../store/useSelene';
import { severityColor } from './severity';

function speedLabel(s: number): string {
  return s >= 1000 ? `${Math.round(s).toLocaleString('en-US')}×` : `${s}×`;
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
          |◀
        </button>
        <button className="primary" onClick={toggle} style={{ minWidth: 76 }}>
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} title="Playback speed (sim seconds per wall-clock second)">
          {options.map((s) => (
            <option key={s} value={s}>
              {speedLabel(s)}
              {s === scripted ? ' (scripted)' : ''}
            </option>
          ))}
        </select>
        <span className="utc">{fmtUtc(cursorDate(t0Iso, tSec))}</span>
        <span className="elapsed">T{fmtElapsed(tSec)}</span>
        <span className="spacer" style={{ flex: 1 }} />
        <span className="elapsed">
          span {fmtElapsed(t0).slice(1)} → {fmtElapsed(t1).slice(1)} · {events.length} events
        </span>
      </div>
      <div className="scrubber" onMouseLeave={() => setHover(null)}>
        <div className="marks">
          {events.map((e, i) => {
            const x = (100 * (e.t - t0)) / span;
            return (
              <span
                key={i}
                className={`mark${e.t <= tSec ? ' past' : ''}`}
                style={{ left: `${x}%`, ['--sev' as string]: severityColor(e.severity) }}
                onMouseEnter={() => setHover({ e, x })}
                onClick={() => {
                  setT(e.t);
                  if (e.object_id) select(e.object_id);
                }}
              />
            );
          })}
          <span className="cursor" style={{ left: `${(100 * (tSec - t0)) / span}%` }} />
        </div>
        <input type="range" min={t0} max={t1} step={Math.max(1, span / 4000)} value={tSec} onChange={(e) => setT(Number(e.target.value))} />
        {hover &&
          (hover.e.t <= tSec || !scenario || finished ? (
            <div className="tip" style={{ left: `${hover.x}%`, ['--sev' as string]: severityColor(hover.e.severity) }}>
              <div className="meta">
                <span className="kind">{hover.e.kind.replace(/_/g, ' ')}</span>
                <span>T{fmtElapsed(hover.e.t)}</span>
                <span>{fmtUtc(cursorDate(t0Iso, hover.e.t))}</span>
              </div>
              <div>{hover.e.text}</div>
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
