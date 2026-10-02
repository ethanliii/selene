/**
 * Trajectory tracks: a time-sorted set of 6-states from `/api/catalog/objects/{id}/trajectory`, evaluated at
 * arbitrary times with cubic Hermite interpolation (positions + the served velocities as tangents). This is what
 * the Ops scene reads for object motion in live mode: the backend propagates (DE440s N-body + SRP for SIMULATED
 * objects; Horizons splines for real ones), the browser only interpolates between samples.
 *
 * Units: `t` seconds since `t0Ms`; `p` in the track's frame units (nondimensional for rot_nd, km for gcrf_km);
 * `v` in units of `p` per second (rot_nd velocities are divided by T* on ingest).
 */
import type { TrajectoryFrame, TrajectoryResponse, Vec3 } from '../api/types';
import { T_STAR_S } from './cr3bp';

export interface Track {
  id: string;
  frame: TrajectoryFrame;
  t0Ms: number;
  n: number;
  /** Seconds since t0Ms, strictly increasing. */
  t: Float64Array;
  /** 3n positions. */
  p: Float64Array;
  /** 3n velocities per second (null when the response had no usable velocities). */
  v: Float64Array | null;
}

export function trackFromTrajectory(resp: TrajectoryResponse, t0Ms: number): Track {
  const n = resp.states.length;
  const t = new Float64Array(n);
  const p = new Float64Array(3 * n);
  const v = new Float64Array(3 * n);
  const velScale = resp.frame === 'rot_nd' ? 1 / T_STAR_S : 1;
  let hasVel = true;
  for (let i = 0; i < n; i++) {
    t[i] = (Date.parse(resp.epochs_utc[i]) - t0Ms) / 1000;
    const s = resp.states[i];
    p[3 * i] = s[0];
    p[3 * i + 1] = s[1];
    p[3 * i + 2] = s[2];
    if (s.length >= 6 && Number.isFinite(s[3]) && Number.isFinite(s[4]) && Number.isFinite(s[5])) {
      v[3 * i] = s[3] * velScale;
      v[3 * i + 1] = s[4] * velScale;
      v[3 * i + 2] = s[5] * velScale;
    } else hasVel = false;
  }
  return { id: resp.object_id, frame: resp.frame, t0Ms, n, t, p, v: hasVel ? v : null };
}

/** Index i of the segment [t_i, t_{i+1}] containing tSec (clamped to [0, n−2]). */
export function trackSegment(tr: Track, tSec: number): number {
  const t = tr.t;
  const n = tr.n;
  if (n < 2 || tSec <= t[0]) return 0;
  if (tSec >= t[n - 1]) return n - 2;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= tSec) lo = mid;
    else hi = mid;
  }
  return lo;
}

/** True when tSec lies within the sampled span (with a small tolerance of one step at each end). */
export function trackCovers(tr: Track, tSec: number): boolean {
  if (tr.n < 2) return false;
  const step = (tr.t[tr.n - 1] - tr.t[0]) / (tr.n - 1);
  return tSec >= tr.t[0] - step && tSec <= tr.t[tr.n - 1] + step;
}

/**
 * Evaluate position (and optionally velocity per second) at tSec. Cubic Hermite inside the span; the end segments
 * extrapolate linearly outside it. Writes into `outP` (length-3 array-like) and returns it.
 */
export function trackEval<T extends { [k: number]: number }>(tr: Track, tSec: number, outP: T, outV?: { [k: number]: number }): T {
  const i = trackSegment(tr, tSec);
  const j = i + 1;
  const t0 = tr.t[i], t1 = tr.t[j];
  const h = t1 - t0 || 1;
  const s = (tSec - t0) / h;
  const p = tr.p, v = tr.v;
  if (!v || s < 0 || s > 1) {
    // Linear (also used for extrapolation beyond the ends).
    for (let k = 0; k < 3; k++) {
      const a = p[3 * i + k], b = p[3 * j + k];
      outP[k] = a + (b - a) * s;
      if (outV) outV[k] = (b - a) / h;
    }
    return outP;
  }
  const s2 = s * s, s3 = s2 * s;
  const h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s, h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
  const d00 = 6 * s2 - 6 * s, d10 = 3 * s2 - 4 * s + 1, d01 = -6 * s2 + 6 * s, d11 = 3 * s2 - 2 * s;
  for (let k = 0; k < 3; k++) {
    const p0 = p[3 * i + k], p1 = p[3 * j + k], m0 = v[3 * i + k] * h, m1 = v[3 * j + k] * h;
    outP[k] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
    if (outV) outV[k] = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h;
  }
  return outP;
}

/** Position at tSec as a fresh Vec3. */
export function trackPos(tr: Track, tSec: number): Vec3 {
  return trackEval(tr, tSec, [0, 0, 0] as Vec3);
}

/**
 * Sampled polyline over [tA, tB] (≤ maxPts points, evenly spaced in time, Hermite-evaluated so curved segments
 * stay smooth), clamped to the track span. Used for trails.
 */
export function trackSlice(tr: Track, tA: number, tB: number, maxPts = 160): Vec3[] {
  if (tr.n < 2) return [];
  const a = Math.max(tA, tr.t[0]), b = Math.min(tB, tr.t[tr.n - 1]);
  if (!(b > a)) return [];
  const step = (tr.t[tr.n - 1] - tr.t[0]) / (tr.n - 1);
  const n = Math.max(2, Math.min(maxPts, Math.ceil((b - a) / Math.max(1, step / 2)) + 1));
  const out: Vec3[] = new Array(n);
  for (let k = 0; k < n; k++) out[k] = trackPos(tr, a + ((b - a) * k) / (n - 1));
  return out;
}
