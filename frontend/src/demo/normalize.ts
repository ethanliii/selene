/**
 * Normalise a /api/demo/scenario payload (or the browser mock) into the shape the scene consumes:
 *  - cloud points → Float32Array `points` (rows `points_rot` are packed; flat arrays are converted),
 *  - sensor `pointing_rot` / `boresight_rot` aliases are both filled,
 *  - frames sorted by t, events sorted by t,
 *  - meta.playback_speed derived from playback_s (default 120 s wall-clock) when absent,
 *  - meta.protagonist_id derived from the first alert/warn event when absent.
 */
import type { DemoFrame, DemoScenario, FrameCloud, FrameSensor, Vec3 } from '../api/types';

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

function normFrame(f: DemoFrame): DemoFrame {
  const out: DemoFrame = {
    t: f.t,
    objects: f.objects ?? [],
    clouds: (f.clouds ?? []).map(normCloud),
    sensors: (f.sensors ?? []).map(normSensor),
  };
  if (f.events) out.events = f.events;
  if (f.reachable) out.reachable = f.reachable;
  return out;
}

export function normalizeScenario(raw: DemoScenario): DemoScenario {
  const frames = [...(raw.frames ?? [])].sort((a, b) => a.t - b.t).map(normFrame);
  const events = [...(raw.events ?? [])].sort((a, b) => a.t - b.t);
  const duration = raw.meta.duration_s || (frames.length ? frames[frames.length - 1].t : 86400);
  const playback_s = raw.meta.playback_s ?? 120;
  const playback_speed = raw.meta.playback_speed ?? Math.max(1, duration / playback_s);
  const protagonist_id =
    raw.meta.protagonist_id ??
    events.find((e) => (e.severity === 'alert' || e.severity === 'warn') && e.object_id)?.object_id ??
    events.find((e) => e.object_id)?.object_id ??
    frames[0]?.objects[0]?.id;
  return {
    meta: { ...raw.meta, duration_s: duration, playback_s, playback_speed, protagonist_id, title: raw.meta.title ?? 'Demo scenario' },
    frames,
    events,
    brief: raw.brief ?? '',
  };
}
