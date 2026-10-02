/**
 * Display-level ephemeris helpers: the Earth–Moon rotating-frame basis, the Sun direction and Earth's orientation
 * as functions of UTC, used to (a) place the synodic group for the 'inertial' view, (b) light the Earth's night
 * side, (c) place and orient ground sites and (d) convert rotating-frame samples to GCRF km.
 *
 * Two levels of fidelity, selected automatically:
 *
 * EXACT (live backend): `/api/ephemeris/bodies` carries, per epoch, the rotating-frame basis R (3×3, columns
 *   x̂ ŷ ẑ in GCRF), the Earth–Moon distance d and the barycentre r_bary (GCRF km), all from JPL DE440s, so
 *       r_gcrf = r_bary + d · R · r_rot_nd        r_rot_nd = Rᵀ (r_gcrf − r_bary) / d.
 *   Between epochs R is slerped as a quaternion and d, r_bary, Sun are interpolated linearly (hourly samples:
 *   the frame turns ≈0.55°/h, so the slerp error is negligible). Earth-fixed → rotating uses Rᵀ·R_z(GMST)
 *   (precession/nutation/polar motion neglected).
 *
 * MEAN (mock / backend down): the planar model below. ECL = ecliptic-aligned inertial frame, ROT = synodic frame
 *   with x̂ along the Earth→Moon line and ẑ = ecliptic pole, θ = ecliptic longitude of the Moon; GCRF = R_x(ε)·ECL
 *   with ε = 23.439291° (IAU 1976 J2000 mean obliquity, Meeus ch. 22); Earth-fixed: v_GCRF = R_z(GMST)·v_ECEF.
 *   Mean-element formulas (Meeus, Astronomical Algorithms, 2nd ed.):
 *     Moon mean longitude  L' = 218.3164477° + 481267.88123421° T   (ch. 47, eq. 47.1)
 *     Sun  mean longitude  L0 = 280.46646°   +  36000.76983°   T   (ch. 25, eq. 25.2)
 *     GMST (deg)              = 280.46061837 + 360.98564736629 (JD − 2451545.0)  (ch. 12, eq. 12.4)
 *   with T = Julian centuries from J2000.0 and UT1 ≈ UTC. These are MEAN angles (errors of several degrees for the
 *   Moon) and are only used for display continuity when the backend is down. When a mock ephemeris table is loaded
 *   (no basis) the Moon/Sun longitudes are taken from it instead of the mean formulas.
 */
import * as THREE from 'three';
import type { EphemerisBodies, State6, Vec3 } from '../api/types';
import { L_STAR_KM, MU, T_STAR_S, V_STAR_KMS } from './cr3bp';

const DEG = Math.PI / 180;
const J2000_MS = Date.UTC(2000, 0, 1, 12, 0, 0);
export const SIDEREAL_MONTH_S = 27.321661 * 86400;
export const MEAN_MOON_RATE_RAD_S = (2 * Math.PI) / SIDEREAL_MONTH_S;
/** J2000 mean obliquity of the ecliptic [rad] (IAU 1976: 23°26′21.448″). */
export const OBLIQUITY_RAD = 23.439291 * DEG;
/** Astronomical unit [km] (IAU 2012). */
export const AU_KM = 149597870.7;

function wrap(a: number): number {
  a = a % (2 * Math.PI);
  return a < 0 ? a + 2 * Math.PI : a;
}

/** Mean Moon longitude [rad] (Meeus 47.1). */
export function meanMoonAngle(ms: number): number {
  const T = (ms - J2000_MS) / (36525 * 86400e3);
  return wrap((218.3164477 + 481267.88123421 * T) * DEG);
}
/** Mean Sun longitude [rad] (Meeus 25.2). */
export function meanSunAngle(ms: number): number {
  const T = (ms - J2000_MS) / (36525 * 86400e3);
  return wrap((280.46646 + 36000.76983 * T) * DEG);
}
/** Greenwich mean sidereal time [rad] (Meeus 12.4). */
export function gmst(ms: number): number {
  const d = (ms - J2000_MS) / 86400e3;
  return wrap((280.46061837 + 360.98564736629 * d) * DEG);
}

// ---- small rotation helpers (right-handed, active rotations of the vector) -------------------------
export const rotZ = (v: Vec3, a: number): Vec3 => {
  const c = Math.cos(a), s = Math.sin(a);
  return [c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]];
};
export const rotX = (v: Vec3, a: number): Vec3 => {
  const c = Math.cos(a), s = Math.sin(a);
  return [v[0], c * v[1] - s * v[2], s * v[1] + c * v[2]];
};
/** ECL → GCRF (equatorial). */
export const eclToGcrf = (v: Vec3): Vec3 => rotX(v, OBLIQUITY_RAD);
/** GCRF → ECL. */
export const gcrfToEcl = (v: Vec3): Vec3 => rotX(v, -OBLIQUITY_RAD);

interface Tables {
  tMs: number[];
  moonTheta: number[]; // unwrapped ecliptic longitude of the Moon
  sunLambda: number[]; // unwrapped ecliptic longitude of the Sun
  moonDistKm: number[];
}

/** Exact per-epoch basis from the backend (see header). */
interface BasisTables {
  tMs: number[];
  /** Quaternion of R (rotating → GCRF) per epoch; sign-continuous so slerp takes the short arc. */
  q: THREE.Quaternion[];
  dKm: number[];
  rBary: Vec3[];
  sunGcrf: Vec3[];
  omega: number[];
}

let tables: Tables | null = null;
let basis: BasisTables | null = null;
let source: 'ephemeris' | 'mean' = 'mean';

function unwrapSeries(a: number[]): number[] {
  const out = [a[0]];
  for (let i = 1; i < a.length; i++) {
    let d = a[i] - a[i - 1];
    while (d > Math.PI) d -= 2 * Math.PI;
    while (d < -Math.PI) d += 2 * Math.PI;
    out.push(out[i - 1] + d);
  }
  return out;
}

const m4 = new THREE.Matrix4();
function quatFromRowMajor(R: number[]): THREE.Quaternion {
  // R is row-major: R[3*i + j] = element (i, j); columns are x̂, ŷ, ẑ in GCRF.
  m4.set(R[0], R[1], R[2], 0, R[3], R[4], R[5], 0, R[6], R[7], R[8], 0, 0, 0, 0, 1);
  return new THREE.Quaternion().setFromRotationMatrix(m4).normalize();
}

/** Install (or clear) the backend ephemeris as the angle/basis source. */
export function setEphemeris(eph: EphemerisBodies | null): void {
  basis = null;
  if (!eph || eph.epochs.length < 2) {
    tables = null;
    source = 'mean';
    return;
  }
  const tMs = eph.epochs.map((e) => Date.parse(e));
  const lon = (v: Vec3) => {
    const e = gcrfToEcl(v);
    return Math.atan2(e[1], e[0]);
  };
  tables = {
    tMs,
    moonTheta: unwrapSeries(eph.moon.map(lon)),
    sunLambda: unwrapSeries(eph.sun.map(lon)),
    moonDistKm: eph.moon.map((m) => Math.hypot(m[0], m[1], m[2])),
  };
  source = 'ephemeris';
  const b = eph.basis;
  if (b && Array.isArray(b.R) && b.R.length === tMs.length && b.R.every((r) => Array.isArray(r) && r.length === 9)) {
    const q = b.R.map(quatFromRowMajor);
    for (let i = 1; i < q.length; i++) if (q[i - 1].dot(q[i]) < 0) q[i].set(-q[i].x, -q[i].y, -q[i].z, -q[i].w);
    basis = { tMs, q, dKm: b.d_km, rBary: b.r_bary_km, sunGcrf: eph.sun, omega: b.omega_rad_s ?? [] };
  }
}

export function ephemerisSource(): 'ephemeris' | 'mean' {
  return source;
}
/**
 * Range guard. The loaded tables cover one timeline span (hourly samples); a query outside it must NOT be clamped
 * or extrapolated silently (a frozen March basis applied to an October scenario froze the Earth orientation and
 * Sun direction and broke the mock demo's ground visibility). Outside the span (± one sample interval) every
 * accessor falls back to the MEAN model, exactly as if no ephemeris were loaded, and `ephemerisCovers(ms)` lets
 * the HUD say so.
 */
function covers(ms: number, tMs: number[]): boolean {
  const n = tMs.length;
  const tol = (tMs[n - 1] - tMs[0]) / (n - 1);
  return ms >= tMs[0] - tol && ms <= tMs[n - 1] + tol;
}
function basisFor(ms: number): BasisTables | null {
  return basis && covers(ms, basis.tMs) ? basis : null;
}
function tablesFor(ms: number): Tables | null {
  return tables && covers(ms, tables.tMs) ? tables : null;
}
/** True when the exact DE440s rotating-frame basis is loaded (live backend). */
export function hasBasis(): boolean {
  return basis !== null;
}
/** True when the exact basis is loaded AND covers `ms` (what the inertial view / ground geometry actually use). */
export function hasBasisAt(ms: number): boolean {
  return basisFor(ms) !== null;
}
/** True when any loaded ephemeris table (exact or mock) covers `ms`; false → mean-element formulas are in use. */
export function ephemerisCovers(ms: number): boolean {
  return tablesFor(ms) !== null;
}
/** Time range [ms] covered by the loaded ephemeris tables (null when none). */
export function ephemerisRangeMs(): [number, number] | null {
  return tables ? [tables.tMs[0], tables.tMs[tables.tMs.length - 1]] : null;
}
/** Key describing the installed ephemeris state (source, basis, span) for caches that depend on it. */
export function ephemerisKey(): string {
  const r = ephemerisRangeMs();
  return `${source}|${basis ? 'basis' : 'nobasis'}|${r ? `${r[0]}-${r[1]}` : 'none'}`;
}

function bracket(t: number, xs: number[]): { lo: number; hi: number; w: number } {
  const n = xs.length;
  if (t <= xs[0]) return { lo: 0, hi: 1, w: (t - xs[0]) / (xs[1] - xs[0]) };
  if (t >= xs[n - 1]) return { lo: n - 2, hi: n - 1, w: (t - xs[n - 2]) / (xs[n - 1] - xs[n - 2]) };
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= t) lo = mid;
    else hi = mid;
  }
  return { lo, hi, w: (t - xs[lo]) / (xs[hi] - xs[lo]) };
}
function interp(t: number, xs: number[], ys: number[]): number {
  const { lo, hi, w } = bracket(t, xs);
  return ys[lo] + (ys[hi] - ys[lo]) * w;
}
function interpV(t: number, xs: number[], ys: Vec3[]): Vec3 {
  const { lo, hi, w } = bracket(t, xs);
  const a = ys[lo], b = ys[hi];
  return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
}

/** Interpolated exact basis at `ms`: `q` = rotating → GCRF rotation, `dKm`, `rBary` (GCRF km). */
export interface BasisSample {
  q: THREE.Quaternion;
  dKm: number;
  rBary: Vec3;
}
const bsTmp: BasisSample = { q: new THREE.Quaternion(), dKm: L_STAR_KM, rBary: [0, 0, 0] };
/** Fills `out` (default: a shared scratch object) and returns it; null when no exact basis is loaded. */
export function basisAt(ms: number, out: BasisSample = bsTmp): BasisSample | null {
  const bs = basisFor(ms);
  if (!bs) return null;
  const { lo, hi, w } = bracket(ms, bs.tMs);
  // w is within [−1, 2] here (one-interval tolerance); clamp only that sliver.
  const wc = Math.min(1, Math.max(0, w));
  out.q.copy(bs.q[lo]).slerp(bs.q[hi], wc);
  out.dKm = bs.dKm[lo] + (bs.dKm[hi] - bs.dKm[lo]) * wc;
  const a = bs.rBary[lo], b = bs.rBary[hi];
  out.rBary[0] = a[0] + (b[0] - a[0]) * wc;
  out.rBary[1] = a[1] + (b[1] - a[1]) * wc;
  out.rBary[2] = a[2] + (b[2] - a[2]) * wc;
  return out;
}

const v3a = new THREE.Vector3(), v3b = new THREE.Vector3(), qTmp = new THREE.Quaternion(), qZ = new THREE.Quaternion(), Zaxis = new THREE.Vector3(0, 0, 1);

/** Ecliptic longitude of the Earth→Moon line [rad] (= planar rotating-frame angle θ); drives the fallback inertial view. */
export function emAngleAt(ms: number): number {
  const tb = tablesFor(ms);
  if (tb) return interp(ms, tb.tMs, tb.moonTheta);
  return meanMoonAngle(ms);
}
/** Ecliptic longitude of the Earth→Sun line [rad]. */
export function sunAngleAt(ms: number): number {
  const tb = tablesFor(ms);
  if (tb) return interp(ms, tb.tMs, tb.sunLambda);
  return meanSunAngle(ms);
}
/** Instantaneous rotation rate of the Earth–Moon line [rad/s]. */
export function emRateAt(ms: number): number {
  const bs = basisFor(ms);
  if (bs && bs.omega.length === bs.tMs.length) return interp(ms, bs.tMs, bs.omega);
  if (!tablesFor(ms)) return MEAN_MOON_RATE_RAD_S;
  return (emAngleAt(ms + 3600e3) - emAngleAt(ms - 3600e3)) / 7200;
}
/** Unit vector toward the Sun expressed in the rotating frame (exact Rᵀ·(sun − r_bary) when the basis is loaded). */
export function sunDirRot(ms: number): Vec3 {
  const bs = basisFor(ms);
  const b = bs ? basisAt(ms) : null;
  if (bs && b) {
    const s = interpV(ms, bs.tMs, bs.sunGcrf);
    v3a.set(s[0] - b.rBary[0], s[1] - b.rBary[1], s[2] - b.rBary[2]).applyQuaternion(qTmp.copy(b.q).invert()).normalize();
    return [v3a.x, v3a.y, v3a.z];
  }
  const a = sunAngleAt(ms) - emAngleAt(ms);
  return [Math.cos(a), Math.sin(a), 0];
}
/** Sun position in the rotating frame, km from the barycenter (1 AU along sunDirRot). */
export function sunPosRotKm(ms: number): Vec3 {
  const d = sunDirRot(ms);
  return [d[0] * AU_KM, d[1] * AU_KM, d[2] * AU_KM];
}
/**
 * Earth's spin angle about the rotating frame's +z as if the equator lay in the ecliptic (GMST − θ) [rad].
 * Kept for callers that only need a planar approximation; prefer {@link ecefToRot}.
 */
export function earthSpinRot(ms: number): number {
  return gmst(ms) - emAngleAt(ms);
}
/** Earth-fixed unit vector (or position) → rotating-frame direction: Rᵀ · R_z(GMST) · v (exact) or the planar model. */
export function ecefToRot(v: Vec3, ms: number): Vec3 {
  const b = basisAt(ms);
  if (b) {
    v3a.set(v[0], v[1], v[2]).applyAxisAngle(Zaxis, gmst(ms)).applyQuaternion(qTmp.copy(b.q).invert());
    return [v3a.x, v3a.y, v3a.z];
  }
  return rotZ(rotX(rotZ(v, gmst(ms)), -OBLIQUITY_RAD), -emAngleAt(ms));
}
/** Euler angles (ZXZ, applied as R_z(a)·R_x(b)·R_z(c)) of the Earth's orientation in the planar rotating frame. */
export function earthOrientationRot(ms: number): { thetaEM: number; obliquity: number; gmst: number } {
  return { thetaEM: emAngleAt(ms), obliquity: OBLIQUITY_RAD, gmst: gmst(ms) };
}
/**
 * Earth orientation in the rotating frame as a quaternion: Rᵀ · R_z(GMST) (exact) or R_z(−θ)·R_x(−ε)·R_z(GMST).
 * `spin = false` omits the GMST term (equator/pole orientation only).
 */
export function earthQuaternionRot(out: THREE.Quaternion, ms: number, spin: boolean): THREE.Quaternion {
  const b = basisAt(ms);
  if (b) {
    out.copy(b.q).invert();
  } else {
    out.setFromAxisAngle(Zaxis, -emAngleAt(ms)).multiply(qTmp.setFromAxisAngle(new THREE.Vector3(1, 0, 0), -OBLIQUITY_RAD));
  }
  if (spin) out.multiply(qZ.setFromAxisAngle(Zaxis, gmst(ms)));
  return out;
}

/** Earth–Moon distance [km] at `ms` (ephemeris if available, else mean L*). */
export function moonDistanceKm(ms: number): number {
  const b = basisAt(ms);
  if (b) return b.dKm;
  const tb = tablesFor(ms);
  if (!tb) return L_STAR_KM;
  return interp(ms, tb.tMs, tb.moonDistKm);
}

/**
 * GCRF (km, km/s) → rotating nondimensional state. Positions are exact when the basis is loaded; velocities use
 * the frame rate ω about ẑ_rot (scale-rate term neglected). Fallback: planar mean-element model.
 */
export function gcrfToRotState(s: State6, epochMs: number): State6 {
  const b = basisAt(epochMs);
  if (b) {
    const qi = qTmp.copy(b.q).invert();
    const r = v3a.set(s[0] - b.rBary[0], s[1] - b.rBary[1], s[2] - b.rBary[2]).applyQuaternion(qi);
    const v = v3b.set(s[3], s[4], s[5]).applyQuaternion(qi);
    const om = emRateAt(epochMs);
    // v_rot = v_in − ω ẑ × r  (both in rotating axes)
    const vr: Vec3 = [v.x + om * r.y, v.y - om * r.x, v.z];
    const vscale = V_STAR_KMS * (b.dKm / L_STAR_KM);
    return [r.x / b.dKm, r.y / b.dKm, r.z / b.dKm, vr[0] / vscale, vr[1] / vscale, vr[2] / vscale];
  }
  const th = emAngleAt(epochMs);
  const Lkm = moonDistanceKm(epochMs);
  const om = emRateAt(epochMs);
  const r = gcrfToEcl([s[0], s[1], s[2]]);
  const v = gcrfToEcl([s[3], s[4], s[5]]);
  const c = Math.cos(th), sn = Math.sin(th);
  // Barycenter offset along the Earth→Moon line.
  const bx = MU * Lkm * c, by = MU * Lkm * sn;
  const rx = r[0] - bx, ry = r[1] - by, rz = r[2];
  const xr = (c * rx + sn * ry) / Lkm;
  const yr = (-sn * rx + c * ry) / Lkm;
  const zr = rz / Lkm;
  const vxr = v[0] + om * ry, vyr = v[1] - om * rx, vzr = v[2];
  const vscale = V_STAR_KMS * (Lkm / L_STAR_KM);
  return [xr, yr, zr, (c * vxr + sn * vyr) / vscale, (-sn * vxr + c * vyr) / vscale, vzr / vscale];
}

/** Inverse of {@link gcrfToRotState} (same approximations). */
export function rotToGcrfState(s: State6, epochMs: number): State6 {
  const b = basisAt(epochMs);
  if (b) {
    const om = emRateAt(epochMs);
    const vscale = V_STAR_KMS * (b.dKm / L_STAR_KM);
    const rx = s[0] * b.dKm, ry = s[1] * b.dKm, rz = s[2] * b.dKm;
    const vx = s[3] * vscale - om * ry, vy = s[4] * vscale + om * rx, vz = s[5] * vscale;
    const r = v3a.set(rx, ry, rz).applyQuaternion(b.q);
    const v = v3b.set(vx, vy, vz).applyQuaternion(b.q);
    return [r.x + b.rBary[0], r.y + b.rBary[1], r.z + b.rBary[2], v.x, v.y, v.z];
  }
  const th = emAngleAt(epochMs);
  const Lkm = moonDistanceKm(epochMs);
  const om = emRateAt(epochMs);
  const c = Math.cos(th), sn = Math.sin(th);
  const rx = (c * s[0] - sn * s[1]) * Lkm, ry = (sn * s[0] + c * s[1]) * Lkm, rz = s[2] * Lkm;
  const vscale = V_STAR_KMS * (Lkm / L_STAR_KM);
  const vxr = (c * s[3] - sn * s[4]) * vscale, vyr = (sn * s[3] + c * s[4]) * vscale, vzr = s[5] * vscale;
  const r = eclToGcrf([rx + MU * Lkm * c, ry + MU * Lkm * sn, rz]);
  const v = eclToGcrf([vxr - om * ry, vyr + om * rx, vzr]);
  return [r[0], r[1], r[2], v[0], v[1], v[2]];
}

/** Rotating-frame position (nondim) → GCRF km (Earth-centered). Exact with the basis. */
export function rotPosToGcrfKm(p: Vec3, epochMs: number): Vec3 {
  const b = basisAt(epochMs);
  if (b) {
    v3a.set(p[0] * b.dKm, p[1] * b.dKm, p[2] * b.dKm).applyQuaternion(b.q);
    return [v3a.x + b.rBary[0], v3a.y + b.rBary[1], v3a.z + b.rBary[2]];
  }
  const th = emAngleAt(epochMs);
  const Lkm = moonDistanceKm(epochMs);
  const c = Math.cos(th), sn = Math.sin(th);
  return eclToGcrf([(c * p[0] - sn * p[1]) * Lkm + MU * Lkm * c, (sn * p[0] + c * p[1]) * Lkm + MU * Lkm * sn, p[2] * Lkm]);
}
/** GCRF km → rotating-frame position (nondim). Exact with the basis; planar model otherwise. */
export function gcrfKmToRotPos(r: Vec3, epochMs: number): Vec3 {
  const b = basisAt(epochMs);
  if (b) {
    v3a.set(r[0] - b.rBary[0], r[1] - b.rBary[1], r[2] - b.rBary[2]).applyQuaternion(qTmp.copy(b.q).invert());
    return [v3a.x / b.dKm, v3a.y / b.dKm, v3a.z / b.dKm];
  }
  const s = gcrfToRotState([r[0], r[1], r[2], 0, 0, 0], epochMs);
  return [s[0], s[1], s[2]];
}

/**
 * Inertial-view display transform for the synodic scene group (scene unit = L*).
 * The display frame is Earth-centred and inertial, with axes equal to the rotating frame's axes at `t0Ms`, so the
 * two views coincide at t0 and the Earth–Moon line then sweeps around the display z axis (≈ the lunar orbit pole).
 *   p_display = Q₀ᵀ · (r_bary(t) + d(t) · Q(t) · p_rot) / L*  +  (−μ, 0, 0)
 * i.e. uniform scale d/L*, rotation Q₀ᵀQ(t), translation Q₀ᵀ r_bary/L* + (−μ,0,0). The Earth stays at (−μ,0,0)
 * and the Moon rides its true ephemeris track at its true distance. Returns false (and leaves `out` untouched)
 * when no exact basis is loaded; callers then fall back to the planar rotation about z.
 */
const b0 = { q: new THREE.Quaternion(), dKm: L_STAR_KM, rBary: [0, 0, 0] as Vec3 };
const bt = { q: new THREE.Quaternion(), dKm: L_STAR_KM, rBary: [0, 0, 0] as Vec3 };
const qRel = new THREE.Quaternion(), tr = new THREE.Vector3(), sc = new THREE.Vector3();
export function inertialDisplayMatrix(ms: number, t0Ms: number, out: THREE.Matrix4): boolean {
  if (!basisAt(t0Ms, b0) || !basisAt(ms, bt)) return false;
  const q0inv = qTmp.copy(b0.q).invert();
  qRel.copy(q0inv).multiply(bt.q);
  tr.set(bt.rBary[0], bt.rBary[1], bt.rBary[2]).applyQuaternion(q0inv).multiplyScalar(1 / L_STAR_KM);
  tr.x -= MU;
  const s = bt.dKm / L_STAR_KM;
  sc.set(s, s, s);
  out.compose(tr, qRel, sc);
  return true;
}

/** Seconds → nondimensional time. */
export const secToNd = (s: number): number => s / T_STAR_S;
export const ndToSec = (tau: number): number => tau * T_STAR_S;
