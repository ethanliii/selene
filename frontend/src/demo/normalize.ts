/**
 * Normalise a /api/demo/scenario payload (or the browser mock) into the shape the scene consumes:
 *  - cloud points → Float32Array `points` (rows `points_rot` are packed; flat arrays are converted),
 *  - sensor `pointing_rot` / `boresight_rot` aliases are both filled,
 *  - frames sorted by t, events sorted by t,
 *  - meta.playback_speed derived from playback_s (default 120 s wall-clock) when absent,
 *  - meta.protagonist_id derived from the first alert/warn event when absent.
 */
import type { DemoFrame, DemoScenario, FrameCloud, FrameObject, FrameObservation, FrameSensor, SeleneEvent, Vec3 } from '../api/types';
import { deriveHeadline, normCustody } from './headline';

export function packRows(rows: Vec3[]): Float32Array {
  const out = new Float32Array(rows.length * 3);
  for (let i = 0; i < rows.length; i++) {
    out[3 * i] = rows[i][0];
    out[3 * i + 1] = rows[i][1];
    out[3 * i + 2] = rows[i][2];
  }
  return out;
}

export function toFloat32(points: number[] | Float32Array | undefined, rows?: Vec3[]): Float32Array {
  if (points instanceof Float32Array) return points;
  if (points && points.length) return Float32Array.from(points);
  if (rows && rows.length) return packRows(rows);
  return new Float32Array(0);
}

function normCloud(c: FrameCloud): FrameCloud {
  const points = toFloat32(c.points, c.points_rot);
  return { object_id: c.object_id, points, sigma_km: c.sigma_km };
}

function normSensor(s: FrameSensor): FrameSensor {
  const bs = s.boresight_rot ?? s.pointing_rot;
  return { ...s, boresight_rot: bs, pointing_rot: bs, active: s.active ?? !!s.target_id };
}

function normObject(o: FrameObject): FrameObject {
  const custody = normCustody(o.custody_ui ?? o.custody);
  const sigma = o.sigma_pos_km ?? o.sigma_km;
  return { ...o, custody, sigma_pos_km: sigma };
}

function normEvent(e: SeleneEvent): SeleneEvent {
  return { ...e, headline: deriveHeadline(e) };
}

function normObservation(o: FrameObservation): FrameObservation {
  return { sensor_id: o.sensor_id, object_id: o.object_id, residual_arcsec: o.residual_arcsec ?? null, magnitude: o.magnitude ?? null, note: o.note };
}

function normFrame(f: DemoFrame): DemoFrame {
  const out: DemoFrame = {
    t: f.t,
    objects: (f.objects ?? []).map(normObject),
    clouds: (f.clouds ?? []).map(normCloud),
    sensors: (f.sensors ?? []).map(normSensor),
  };
  // Every tracklet stays available per frame (the KPI strip's "last obs" reads these, not the thinned feed).
  if (f.observations?.length) out.observations = f.observations.map(normObservation);
  if (f.events) out.events = f.events;
  if (f.reachable) out.reachable = f.reachable;
  return out;
}

export function normalizeScenario(raw: DemoScenario, source: 'backend' | 'browser-mock' = 'backend'): DemoScenario {
  const frames = [...(raw.frames ?? [])].sort((a, b) => a.t - b.t).map(normFrame);
  const events = [...(raw.events ?? [])].sort((a, b) => a.t - b.t).map(normEvent);
  const duration = raw.meta.duration_s || (frames.length ? frames[frames.length - 1].t : 86400);
  const playback_s = raw.meta.playback_s ?? 120;
  const playback_speed = raw.meta.playback_speed ?? Math.max(1, duration / playback_s);
  const protagonist_id =
    raw.meta.protagonist_id ??
    events.find((e) => (e.severity === 'alert' || e.severity === 'warn') && e.object_id)?.object_id ??
    events.find((e) => e.object_id)?.object_id ??
    frames[0]?.objects[0]?.id;
  const { custody_km, lost_km } = custodyThresholds(raw, events);
  const title = raw.meta.title ?? 'Demo scenario';
  return {
    meta: { ...raw.meta, duration_s: duration, playback_s, playback_speed, protagonist_id, title: source === 'browser-mock' ? `[OFFLINE MOCK] ${title}` : title, custody_km, lost_km, source },
    frames,
    events,
    brief: raw.brief ?? '',
    // Backend run metrics (custody/σ timelines, detection latency, Δv truth vs estimate) feed the metrics card.
    metrics: raw.metrics,
  };
}

const numOr = (v: unknown, d: number | undefined) => (typeof v === 'number' && Number.isFinite(v) && v > 0 ? v : d);

/**
 * Custody σ thresholds [km] the scenario was computed with: meta (mock) → backend metrics.filter → the numbers
 * quoted by the custody_degraded / custody_lost events ("sigma_pos 102 km > 100 km") → undefined (the UI then shows
 * no threshold band rather than an invented one).
 */
export function custodyThresholds(raw: DemoScenario, events: SeleneEvent[]): { custody_km?: number; lost_km?: number } {
  const filt = (raw.metrics?.filter ?? {}) as Record<string, unknown>;
  let custody_km = numOr(raw.meta.custody_km, numOr(filt.custody_km, undefined));
  let lost_km = numOr(raw.meta.lost_km, numOr(filt.lost_km, undefined));
  const quoted = (kind: string) => {
    const e = events.find((x) => x.kind === kind);
    const m = e ? /[>≥]\s*([\d,.]+)\s*km/.exec(e.text) : null;
    return m ? Number(m[1].replace(/,/g, '')) : NaN;
  };
  if (custody_km === undefined) custody_km = numOr(quoted('custody_degraded'), undefined);
  if (lost_km === undefined) lost_km = numOr(quoted('custody_lost'), undefined);
  return { custody_km, lost_km };
}
