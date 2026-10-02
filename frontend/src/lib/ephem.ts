/**
 * Display-level ephemeris helpers: the Earth–Moon line angle, the Sun direction and Earth's orientation as
 * functions of UTC, used to (a) rotate the synodic group for the 'inertial' view, (b) light the Earth's
 * night side, (c) place and orient ground sites and (d) convert rotating-frame samples to GCRF-like km.
 *
 * Frame model (one model for the whole UI, so the scene, the Object panel and the mock scenario agree):
 *   ECL  ecliptic-aligned inertial frame (x toward the equinox, z = ecliptic pole), J2000 mean obliquity.
 *   ROT  synodic frame: x̂ along the Earth→Moon line, ẑ = ecliptic pole, angle θ = ecliptic longitude of the
 *        Moon. The scene's z axis IS this ẑ. The lunar orbit inclination (5.1°) is neglected.
 *   GCRF equatorial inertial frame: v_GCRF = R_x(ε) · v_ECL with ε = 23.439291° (IAU 1976 J2000 mean
 *        obliquity, Meeus ch. 22). Earth-fixed: v_GCRF = R_z(GMST) · v_ECEF.
 * So a ground site's rotating-frame position is R_z(−θ) · R_x(−ε) · R_z(GMST) · r_ECEF, which gives the Moon
 * a declination of up to ±ε at the sites (real: ±28.6°, the missing 5° being the lunar inclination).
 *
 * Source of truth when available: `/api/ephemeris/bodies` (DE440s via the backend). The Moon/Sun GCRF vectors
 * are rotated to ECL and their longitudes tabulated and linearly interpolated.
 *
 * Fallback (mock mode): mean-element formulas (Meeus, Astronomical Algorithms, 2nd ed.):
 *   Moon mean longitude  L' = 218.3164477° + 481267.88123421° T   (ch. 47, eq. 47.1)
 *   Sun  mean longitude  L0 = 280.46646°   +  36000.76983°   T   (ch. 25, eq. 25.2)
 *   GMST (deg)              = 280.46061837 + 360.98564736629 (JD − 2451545.0)  (ch. 12, eq. 12.4)
 * with T = Julian centuries from J2000.0 and UT1 ≈ UTC. These are MEAN angles (no periodic terms, errors of
 * several degrees for the Moon) and are only used for display continuity when the backend is down.
 */
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

let tables: Tables | null = null;
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

/** Install (or clear) the backend ephemeris as the angle source. */
export function setEphemeris(eph: EphemerisBodies | null): void {
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
}

export function ephemerisSource(): 'ephemeris' | 'mean' {
  return source;
}

function interp(t: number, xs: number[], ys: number[]): number {
  const n = xs.length;
  if (t <= xs[0]) return ys[0] + ((ys[1] - ys[0]) * (t - xs[0])) / (xs[1] - xs[0]);
  if (t >= xs[n - 1]) return ys[n - 1] + ((ys[n - 1] - ys[n - 2]) * (t - xs[n - 1])) / (xs[n - 1] - xs[n - 2]);
  let lo = 0, hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xs[mid] <= t) lo = mid;
    else hi = mid;
  }
  const w = (t - xs[lo]) / (xs[hi] - xs[lo]);
  return ys[lo] + (ys[hi] - ys[lo]) * w;
}

/** Ecliptic longitude of the Earth→Moon line [rad] (= rotating-frame angle θ); drives the inertial view. */
export function emAngleAt(ms: number): number {
  if (tables) return interp(ms, tables.tMs, tables.moonTheta);
  return meanMoonAngle(ms);
}
/** Ecliptic longitude of the Earth→Sun line [rad]. */
export function sunAngleAt(ms: number): number {
  if (tables) return interp(ms, tables.tMs, tables.sunLambda);
  return meanSunAngle(ms);
}
/** Instantaneous rotation rate of the Earth–Moon line [rad/s]. */
export function emRateAt(ms: number): number {
  if (!tables) return MEAN_MOON_RATE_RAD_S;
  return (emAngleAt(ms + 3600e3) - emAngleAt(ms - 3600e3)) / 7200;
}
/** Unit vector toward the Sun expressed in the rotating frame (Sun in the ecliptic plane). */
export function sunDirRot(ms: number): Vec3 {
  const a = sunAngleAt(ms) - emAngleAt(ms);
  return [Math.cos(a), Math.sin(a), 0];
}
/** Sun position in the rotating frame, km from the barycenter (1 AU along sunDirRot). */
export function sunPosRotKm(ms: number): Vec3 {
  const d = sunDirRot(ms);
  return [d[0] * AU_KM, d[1] * AU_KM, 0];
}
/**
 * Earth's spin angle about the rotating frame's +z as if the equator lay in the ecliptic (GMST − θ) [rad].
 * Kept for callers that only need a planar approximation; prefer {@link ecefToRot}.
 */
export function earthSpinRot(ms: number): number {
  return gmst(ms) - emAngleAt(ms);
}
/** Earth-fixed unit vector (or position) → rotating-frame direction: R_z(−θ) · R_x(−ε) · R_z(GMST) · v. */
export function ecefToRot(v: Vec3, ms: number): Vec3 {
  return rotZ(rotX(rotZ(v, gmst(ms)), -OBLIQUITY_RAD), -emAngleAt(ms));
}
/** Euler angles (ZXZ, applied as R_z(a)·R_x(b)·R_z(c)) of the Earth's orientation in the rotating frame. */
export function earthOrientationRot(ms: number): { thetaEM: number; obliquity: number; gmst: number } {
  return { thetaEM: emAngleAt(ms), obliquity: OBLIQUITY_RAD, gmst: gmst(ms) };
}

/** Earth–Moon distance [km] at `ms` (ephemeris if available, else mean L*). */
export function moonDistanceKm(ms: number): number {
  if (!tables) return L_STAR_KM;
  return interp(ms, tables.tMs, tables.moonDistKm);
}

/**
 * GCRF (km, km/s) → rotating nondimensional state. Display-level only: mean-element Earth–Moon line
 * (ephemeris angle when available), lunar inclination neglected, scale-rate term neglected. Velocities are
 * nondimensionalised with the CR3BP unit V* = L*·T*⁻¹ scaled by the current Earth–Moon distance (same
 * convention as lib/cr3bp.ts); the ω×r term uses the actual (ephemeris or mean) frame rate.
 */
export function gcrfToRotState(s: State6, epochMs: number): State6 {
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

/** Rotating-frame position (nondim) → GCRF km (Earth-centered). */
export function rotPosToGcrfKm(p: Vec3, epochMs: number): Vec3 {
  const th = emAngleAt(epochMs);
  const Lkm = moonDistanceKm(epochMs);
  const c = Math.cos(th), sn = Math.sin(th);
  return eclToGcrf([(c * p[0] - sn * p[1]) * Lkm + MU * Lkm * c, (sn * p[0] + c * p[1]) * Lkm + MU * Lkm * sn, p[2] * Lkm]);
}

/** Seconds → nondimensional time. */
export const secToNd = (s: number): number => s / T_STAR_S;
export const ndToSec = (tau: number): number => tau * T_STAR_S;
