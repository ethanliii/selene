/**
 * Right dock: KPI strip (scenario only: custody · σ_pos · last obs · tasked sensors, live from the current frame)
 * above the Object / Events / Brief tabs (active tab lives in the store so the demo driver can switch it).
 *
 * The strip reads the scenario frame at the cursor directly (objects' custody/σ, sensors, per-frame observation
 * rows) rather than mounting a second useScenarioFrame(): it needs none of the trail rebuilding that hook does.
 * Custody thresholds are the scenario's own (meta.custody_km / lost_km, see demo/normalize.ts), never a UI constant.
 */
import { useEffect, useLayoutEffect, useRef } from 'react';
import type { DemoFrame } from '../api/types';
import { beatOfEvent, sensorName, sensorShort } from '../demo/headline';
import { frameIndexAt } from '../demo/useScenarioFrame';
import { fmtAge, useSelene, type DockTab } from '../store/useSelene';
import { AnalysisPanel } from './analysis/AnalysisPanel';
import { AnalystBrief } from './AnalystBrief';
import { EventsFeed } from './EventsFeed';
import { ObjectPanel } from './ObjectPanel';

const TABS: { id: DockTab; label: string; title?: string }[] = [
  { id: 'object', label: 'Object' },
  { id: 'events', label: 'Events' },
  { id: 'analysis', label: 'Analysis', title: 'Run the OD, maneuver-detection, reachability and tasking engines on the selected object (live backend)' },
  { id: 'brief', label: 'Brief' },
];

const fmtKm = (v: number) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));

/** A frame row counts as a tracklet when it realised a measurement (tasking "looks" without one carry a note). */
function isTracklet(o: NonNullable<DemoFrame['observations']>[number]): boolean {
  if (typeof o.residual_arcsec === 'number' && Number.isFinite(o.residual_arcsec)) return true;
  return !o.note || !/no measurement/i.test(o.note);
}

/** Latest tracklet on `id` at or before frame `idx`: from the frames' observation rows (every tracklet), falling
 *  back to observation EVENTS when the bundle carries no per-frame rows (the feed thins follow-ups to one per 6 h). */
function lastObservation(frames: DemoFrame[], idx: number, id: string, events: { kind: string; t: number; object_id?: string; data?: Record<string, unknown> }[], tSec: number): { t: number; sensor: string } | null {
  let anyRows = false;
  for (let k = idx; k >= 0; k--) {
    const rows = frames[k].observations;
    if (!rows?.length) continue;
    anyRows = true;
    const hit = rows.filter((o) => o.object_id === id && isTracklet(o));
    if (hit.length) return { t: frames[k].t, sensor: hit[hit.length - 1].sensor_id };
  }
  if (anyRows) return null;
  for (let k = events.length - 1; k >= 0; k--) {
    const e = events[k];
    if (e.kind === 'observation' && e.object_id === id && e.t <= tSec) return { t: e.t, sensor: String(e.data?.sensor_id ?? '') };
  }
  return null;
}

function KpiStrip() {
  const scenario = useSelene((s) => s.scenario);
  const selected = useSelene((s) => s.selectedObjectId);
  const events = useSelene((s) => s.events);
  // Throttled cursor (≈5 Hz while playing) so the strip does not re-render every animation frame.
  // Throttled cursor for the "since obs" age (≈5 Hz at full speed, capped at 5 sim-minutes so the crawl during
  // narration — 1/40 speed — still refreshes every few seconds). The FRAME is picked from the EXACT cursor below,
  // so custody / σ never lag a frame behind the badges, caption and ribbon at a frame boundary.
  const tSec = useSelene((s) => {
    const q = s.playing ? Math.min(300, Math.max(1, s.speed * 0.2)) : 1;
    return Math.min(s.t1Sec, Math.floor(s.tSec / q) * q);
  });
  const idx = useSelene((s) => (s.scenario ? frameIndexAt(s.scenario.frames, s.tSec) : -1));
  if (!scenario) return null;
  const frames = scenario.frames;
  const frame = idx >= 0 ? frames[idx] : null;
  const objects = frame?.objects ?? [];
  const sensors = frame?.sensors ?? [];
  const id = selected && objects.some((o) => o.id === selected) ? selected : (scenario.meta.protagonist_id ?? objects[0]?.id ?? null);
  const obj = objects.find((o) => o.id === id) ?? null;
  const custody = obj?.custody ?? 'unknown';
  const sev = custody === 'held' ? 'ok' : custody === 'degraded' ? 'warn' : custody === 'lost' ? 'alert' : 'muted';
  const sigma = obj?.sigma_pos_km;
  const ck = scenario.meta.custody_km, lk = scenario.meta.lost_km;
  // σ tile colour follows the scenario's thresholds when known, else the frame's own custody label.
  const sigmaSev = sigma === undefined ? 'muted' : ck && lk ? (sigma >= lk ? 'alert' : sigma >= ck ? 'warn' : 'ok') : sev;
  const sigmaBand = sigma === undefined ? '' : ck && lk ? (sigma >= lk ? `lost ≥ ${fmtKm(lk)} km` : sigma >= ck ? `degraded ${fmtKm(ck)}–${fmtKm(lk)} km` : `held < ${fmtKm(ck)} km`) : '√tr P_pos';
  const last = id ? lastObservation(frames, idx, id, events, tSec) : null;
  const lastAge = last ? Math.max(0, tSec - last.t) : NaN;
  const tasked = sensors.filter((s) => s.target_id === id && (s.active || s.target_id));
  const lastTask = [...events].reverse().find((e) => beatOfEvent(e) === 'tasked' && e.t <= tSec);
  const recentlyTasked = lastTask ? tSec - lastTask.t < 2 * 3600 : false;
  const taskedLabel = tasked.length === 0 ? '—' : tasked.length === 1 ? sensorShort(tasked[0].id) : `${tasked.length} sensors`;
  const lastSev = !Number.isFinite(lastAge) ? 'muted' : lastAge > 12 * 3600 ? 'alert' : lastAge > 4 * 3600 ? 'warn' : 'ok';
  return (
    <div className="kpi-strip" title={id ? `Live from the scenario frame at the cursor for ${id}` : undefined}>
      <div className={`kpi ${sev}`} title={ck && lk ? `Custody from the particle-cloud σ_pos: held < ${fmtKm(ck)} km, degraded ${fmtKm(ck)}–${fmtKm(lk)} km, lost ≥ ${fmtKm(lk)} km (scenario thresholds)` : 'Custody status from the scenario engine'}>
        <div className="k">Custody</div>
        <div className="v">{custody === 'unknown' ? 'N/A' : custody.toUpperCase()}</div>
        <div className="s">{id ?? ''}</div>
      </div>
      <div className={`kpi ${sigmaSev}`} title="σ_pos = √tr(P_pos) of the estimate, km">
        <div className="k">σ_pos</div>
        <div className="v">
          {sigma === undefined ? '—' : fmtKm(sigma)}
          {sigma !== undefined && <small> km</small>}
        </div>
        <div className="s">{sigmaBand}</div>
      </div>
      <div className={`kpi ${lastSev}`} title={last ? `${sensorName(last.sensor)} tracklet at T+${fmtAge(last.t)} (every tracklet in the frames counts, not only the feed entries)` : 'No tracklet yet in the window'}>
        <div className="k">Since obs</div>
        <div className="v">{Number.isFinite(lastAge) ? fmtAge(lastAge) : '—'}</div>
        <div className="s">{last ? sensorShort(last.sensor) : 'none in window'}</div>
      </div>
      <div className={`kpi ${tasked.length ? 'accent' : 'muted'}`} title={tasked.length ? tasked.map((s) => sensorName(s.id)).join(', ') : 'No sensor currently tasked on this object'}>
        <div className="k">Tasked</div>
        <div className={`v${recentlyTasked && tasked.length ? ' pulse' : ''}`}>{taskedLabel}</div>
        <div className="s">{tasked.length === 0 ? 'no sensor on target' : tasked.length === 1 ? 'on target' : tasked.map((s) => sensorShort(s.id)).join(' · ')}</div>
      </div>
    </div>
  );
}

export function Dock() {
  const tab = useSelene((s) => s.dockTab);
  const setTab = useSelene((s) => s.setDockTab);
  const selected = useSelene((s) => s.selectedObjectId);
  const source = useSelene((s) => s.selectSource);
  const finished = useSelene((s) => s.scenarioFinished);
  const briefReady = useSelene((s) => s.brief.length > 0);
  const body = useRef<HTMLDivElement>(null);
  // Each tab opens at its top: without this the dock body keeps the previous tab's scroll offset, and the Brief
  // (switched in by the demo driver at brief_ready / end of story) would open mid-glossary instead of at the BLUF.
  useLayoutEffect(() => {
    if (body.current) body.current.scrollTop = 0;
  }, [tab]);
  // A USER selection in the scene brings the Object tab forward; the demo driver's automatic selections do not
  // (the scripted story keeps the Events feed in front and surfaces events as toasts).
  useEffect(() => {
    if (selected && source === 'user' && !finished) setTab('object');
  }, [selected, source, finished, setTab]);
  return (
    <>
      <KpiStrip />
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)} title={t.title}>
            {t.label}
            {t.id === 'brief' && briefReady && finished && <span className="dot" />}
          </button>
        ))}
      </div>
      <div className="dock-body" ref={body}>
        {tab === 'object' && <ObjectPanel />}
        {tab === 'events' && <EventsFeed />}
        {tab === 'analysis' && <AnalysisPanel />}
        {tab === 'brief' && <AnalystBrief />}
      </div>
    </>
  );
}
