/** Bottom timeline: play/pause, speed, UTC readout, scrubber with event marks. */
import { SPEEDS, cursorDate, fmtElapsed, fmtUtc, useSelene, type PlaybackSpeed } from '../store/useSelene';
import { severityColor } from './severity';

export function Timeline() {
  const t0Iso = useSelene((s) => s.t0Iso);
  const tSec = useSelene((s) => s.tSec);
  const t0 = useSelene((s) => s.t0Sec);
  const t1 = useSelene((s) => s.t1Sec);
  const playing = useSelene((s) => s.playing);
  const speed = useSelene((s) => s.speed);
  const events = useSelene((s) => s.events);
  const setT = useSelene((s) => s.setT);
  const toggle = useSelene((s) => s.togglePlaying);
  const setSpeed = useSelene((s) => s.setSpeed);

  const span = Math.max(1, t1 - t0);
  return (
    <div className="timeline-inner">
      <div className="controls">
        <button className="primary" onClick={toggle} style={{ minWidth: 72 }}>
          {playing ? '❚❚ Pause' : '▶ Play'}
        </button>
        <select value={speed} onChange={(e) => setSpeed(Number(e.target.value) as PlaybackSpeed)} title="Playback speed (sim seconds per wall second)">
          {SPEEDS.map((s) => (
            <option key={s} value={s}>
              {s}×
            </option>
          ))}
        </select>
        <span className="utc">{fmtUtc(cursorDate(t0Iso, tSec))}</span>
        <span className="elapsed">T{fmtElapsed(tSec)}</span>
        <span className="spacer" style={{ flex: 1 }} />
        <span className="elapsed">
          span {fmtElapsed(t0).slice(1)} → {fmtElapsed(t1).slice(1)}
        </span>
      </div>
      <div className="scrubber">
        <div className="marks">
          {events.map((e, i) => (
            <span
              key={i}
              className="mark"
              title={`${e.kind}: ${e.text}`}
              style={{ left: `${(100 * (e.t - t0)) / span}%`, ['--sev' as string]: severityColor(e.severity) }}
            />
          ))}
        </div>
        <input type="range" min={t0} max={t1} step={Math.max(1, span / 2000)} value={tSec} onChange={(e) => setT(Number(e.target.value))} />
      </div>
    </div>
  );
}
