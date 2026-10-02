/**
 * Browser-side geometric visibility model used by the MOCK fallbacks of the Coverage page and the
 * Architecture Studio when the FastAPI backend is offline.
 *
 * Everything here is a deliberately simple but physically grounded surrogate of the backend
 * sensor model (PLAN.md §2.5). It is NOT the flight-fidelity model: orbits are schematic
 * parametrisations (not integrated CR3BP/ephemeris trajectories), the Sun direction follows the
 * mean synodic phase, and ground sites are placed on the equator. Every result produced from it is
 * labelled MOCK in the UI.
 *
 * Frame and units: Earth–Moon rotating (synodic) frame, barycentric origin, +x toward the Moon,
 * +z along the orbital angular momentum. Positions in km, times in seconds since t0, angles in rad.
 */
import type { BlindReason, CandidateOrbit, Vec3 } from './types';

// ---- constants --------------------------------------------------------------------------------
/** Earth–Moon characteristic length, km (CR3BP nondimensional unit, PLAN.md §2.1). */
export const L_STAR_KM = 384400;
/** CR3BP mass ratio from DE440 GMs (GM_E 398600.435507, GM_M 4902.800118 km³/s²). */
export const MU = 4902.800118 / (398600.435507 + 4902.800118);
/** IAU 2015 nominal Earth equatorial radius, km; IAU WGCCRE mean lunar radius, km. */
export const R_EARTH_KM = 6378.1366;
export const R_MOON_KM = 1737.4;
/** Astronomical unit, km (IAU 2012). */
export const AU_KM = 1.495978707e8;
/** Mean synodic month and mean sidereal month, s (Meeus, Astronomical Algorithms, ch. 49). */
export const T_SYNODIC_S = 29.530589 * 86400;
export const T_SIDEREAL_S = 27.321661 * 86400;
/** Sidereal rotation period of the Earth, s (IERS). */
export const T_EARTH_ROT_S = 86164.0905;
/** Geostationary radius, km. */
export const R_GEO_KM = 42164;
/** Solar radius, km (IAU 2015 nominal). */
export const R_SUN_KM = 695700;
/** Apparent visual magnitude of the Sun (Willmer 2018, ApJS 236, 47: V = −26.74). */
export const M_SUN = -26.74;
/**
 * Reference mean new Moon: 2000-01-06 18:14 UTC (Meeus ch. 49, k = 0: JDE 2451550.09766).
 * Used only to set the mean Sun direction in the rotating frame; true phase differs by up to ~14 h
 * (≈ 7° of Sun direction) because of lunar orbit eccentricity.
 */
export const NEW_MOON_REF_MS = Date.UTC(2000, 0, 6, 18, 14, 0);

export const EARTH_ROT_KM: Vec3 = [-MU * L_STAR_KM, 0, 0];
export const MOON_ROT_KM: Vec3 = [(1 - MU) * L_STAR_KM, 0, 0];
/** Collinear points from the CR3BP quintic (PLAN.md §2.1); L4/L5 equilateral. Nondimensional. */
export const LAGRANGE_ND: { name: 'L1' | 'L2' | 'L3' | 'L4' | 'L5'; pos: Vec3 }[] = [
  { name: 'L1', pos: [0.836915, 0, 0] },
  { name: 'L2', pos: [1.155682, 0, 0] },
  { name: 'L3', pos: [-1.005063, 0, 0] },
  { name: 'L4', pos: [0.5 - MU, 0.866025, 0] },
  { name: 'L5', pos: [0.5 - MU, -0.866025, 0] },
];

const DEG = Math.PI / 180;

// ---- small vector helpers ----------------------------------------------------------------------
const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: Vec3) => Math.sqrt(dot(a, a));
const unit = (a: Vec3): Vec3 => {
  const n = norm(a) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};
/** Angle between two vectors, rad (robust acos). */
const angle = (a: Vec3, b: Vec3) => Math.acos(Math.max(-1, Math.min(1, dot(a, b) / ((norm(a) || 1) * (norm(b) || 1)))));

// ---- Sun direction ------------------------------------------------------------------------------
/**
 * Unit vector from the barycenter toward the Sun in the rotating frame at absolute epoch `ms`
 * (JS milliseconds). At new Moon the Sun lies along +x (behind the Moon); it then regresses
 * clockwise at the synodic rate because the frame rotates faster than the Earth–Sun line.
 */
export function sunDirAt(ms: number): Vec3 {
  const phase = ((ms - NEW_MOON_REF_MS) / 1000 / T_SYNODIC_S) % 1;
  const th = -2 * Math.PI * phase;
  return [Math.cos(th), Math.sin(th), 0];
}

// ---- schematic orbits ---------------------------------------------------------------------------
/** Solve Kepler's equation M = E − e sin E for E (Newton, rad). */
function keplerE(M: number, e: number): number {
  let E = e < 0.8 ? M : Math.PI;
  for (let k = 0; k < 12; k++) {
    const f = E - e * Math.sin(E) - M;
    const fp = 1 - e * Math.cos(E);
    const d = f / fp;
    E -= d;
    if (Math.abs(d) < 1e-10) break;
  }
  return E;
}

/** Earth-rotation angle relative to the rotating frame (rad): sidereal spin minus frame rotation. */
export function earthSpinAngle(tSec: number, phase0: number): number {
  return phase0 + 2 * Math.PI * tSec * (1 / T_EARTH_ROT_S - 1 / T_SIDEREAL_S);
}

export interface OrbitShape {
  /** Position in the rotating frame, km, at time tSec with phase fraction `phase`. */
  pos(tSec: number, phase: number): Vec3;
  /** Period, s (for drawing one revolution). */
  period_s: number;
  label: string;
}

/**
 * Schematic host/target orbits. Amplitudes are representative mid-family values quoted to the
 * nearest 10 000 km; they are placeholders until the backend orbit library (M3) supplies integrated
 * samples. Halo/DRO/NRHO numbers follow the ranges tabulated in Zimovan-Spreen, Howell & Davis,
 * "Near rectilinear halo orbits and nearby higher-period dynamical structures", Celest. Mech. Dyn.
 * Astron. 132:28 (2020) and Grebow (2006, Purdue MS thesis) for halo amplitudes.
 */
export const ORBITS: Record<CandidateOrbit | 'NRHO_9_2' | 'ground', OrbitShape> = {
  ground: {
    label: 'Ground site',
    period_s: T_EARTH_ROT_S,
    pos: (t, phase) => {
      const psi = earthSpinAngle(t, 2 * Math.PI * phase);
      return [EARTH_ROT_KM[0] + R_EARTH_KM * Math.cos(psi), R_EARTH_KM * Math.sin(psi), 0];
    },
  },
  GEO: {
    label: 'GEO',
    period_s: T_EARTH_ROT_S,
    pos: (t, phase) => {
      const psi = earthSpinAngle(t, 2 * Math.PI * phase);
      return [EARTH_ROT_KM[0] + R_GEO_KM * Math.cos(psi), R_GEO_KM * Math.sin(psi), 0];
    },
  },
  L1_halo: {
    // Schematic L1 halo: Ax ≈ 0.06 L*, Ay ≈ 0.20 L*, Az ≈ 0.10 L*, period ≈ 12 d (northern).
    label: 'L1 halo',
    period_s: 12.0 * 86400,
    pos: (t, phase) => {
      const tau = 2 * Math.PI * (t / (12.0 * 86400) + phase);
      return [(0.836915 - 0.06 * Math.cos(tau)) * L_STAR_KM, 0.2 * Math.sin(tau) * L_STAR_KM, 0.1 * Math.cos(tau) * L_STAR_KM];
    },
  },
  L2_halo: {
    // Schematic L2 southern halo: Ax ≈ 0.08 L*, Ay ≈ 0.25 L*, Az ≈ −0.15 L*, period ≈ 14.5 d.
    label: 'L2 halo (S)',
    period_s: 14.5 * 86400,
    pos: (t, phase) => {
      const tau = 2 * Math.PI * (t / (14.5 * 86400) + phase);
      return [(1.155682 + 0.08 * Math.cos(tau)) * L_STAR_KM, 0.25 * Math.sin(tau) * L_STAR_KM, -0.15 * Math.cos(tau) * L_STAR_KM];
    },
  },
  DRO: {
    // Schematic planar DRO: x half-width 70 000 km, y half-width ≈ 2× (DROs are y-elongated),
    // retrograde (clockwise) in the rotating frame, period ≈ 13 d.
    label: 'DRO',
    period_s: 13.0 * 86400,
    pos: (t, phase) => {
      const tau = 2 * Math.PI * (t / (13.0 * 86400) + phase);
      return [MOON_ROT_KM[0] + 70000 * Math.cos(tau), -135000 * Math.sin(tau), 0];
    },
  },
  resonant_3_1: {
    // Earth-centred Kepler ellipse with period T_sidereal/3 (a = 3^(−2/3) L* ≈ 0.48 L*), e = 0.55,
    // expressed in the rotating frame by un-rotating the inertial position at the frame rate.
    label: '3:1 resonant',
    period_s: T_SIDEREAL_S, // the pattern closes after one sidereal month (3 lobes)
    pos: (t, phase) => {
      const Torb = T_SIDEREAL_S / 3;
      const a = Math.pow(1 / 3, 2 / 3) * L_STAR_KM;
      const e = 0.55;
      const M = 2 * Math.PI * ((t / Torb + phase) % 1);
      const E = keplerE(M, e);
      const xi = a * (Math.cos(E) - e);
      const yi = a * Math.sqrt(1 - e * e) * Math.sin(E);
      const th = -(2 * Math.PI * t) / T_SIDEREAL_S; // frame rotation un-done
      const c = Math.cos(th);
      const s = Math.sin(th);
      return [EARTH_ROT_KM[0] + xi * c - yi * s, xi * s + yi * c, 0];
    },
  },
  NRHO_9_2: {
    // Schematic 9:2 NRHO surrogate: Moon-centred Kepler-like ellipse with r_p ≈ 3 250 km,
    // r_a ≈ 70 000 km, period 6.56 d, apolune toward −z (southern family), slight x tilt.
    label: '9:2 NRHO',
    period_s: 6.56 * 86400,
    pos: (t, phase) => {
      const rp = 3250;
      const ra = 70000;
      const a = (rp + ra) / 2;
      const e = (ra - rp) / (ra + rp);
      const M = 2 * Math.PI * ((t / (6.56 * 86400) + phase) % 1);
      const E = keplerE(M, e);
      const r = a * (1 - e * Math.cos(E));
      const nu = Math.atan2(Math.sqrt(1 - e * e) * Math.sin(E), Math.cos(E) - e);
      // perilune over the north pole (+z), apolune south (−z); in-plane direction tilted 10° toward −x
      const zc = r * Math.cos(nu);
      const xc = r * Math.sin(nu);
      const tilt = 10 * DEG;
      return [MOON_ROT_KM[0] - xc * Math.cos(tilt), xc * Math.sin(tilt) * 0.3, zc];
    },
  },
};

/** Sample one closed curve of an orbit, nondimensional xy, for drawing. */
export function orbitOutlineND(o: OrbitShape, n = 240): [number, number][] {
  const out: [number, number][] = [];
  for (let k = 0; k <= n; k++) {
    const p = o.pos((o.period_s * k) / n, 0);
    out.push([p[0] / L_STAR_KM, p[1] / L_STAR_KM]);
  }
  return out;
}

// ---- sensors -------------------------------------------------------------------------------------
export interface ModelSensor {
  id: string;
  kind: 'ground' | 'space';
  orbit: CandidateOrbit | 'ground';
  phase: number;
  limiting_mag: number;
  /** Exclusion half-angles, rad. */
  sun_excl: number;
  moon_excl: number;
  earth_excl: number;
  /** Ground only: elevation mask and Sun-depression requirement, rad. */
  min_el: number;
  sun_el_max: number;
}

/** Three notional ground sites, 120° apart in longitude, 1-m class (m_lim 19.5), equatorial approximation. */
export function groundNetwork(): ModelSensor[] {
  return [0, 1, 2].map((k) => ({
    id: `GND-${k + 1}`,
    kind: 'ground',
    orbit: 'ground',
    phase: k / 3,
    limiting_mag: 19.5,
    sun_excl: 0,
    moon_excl: 25 * DEG, // full-Moon avoidance; phase-scaled 5–25° in checkVisibility
    earth_excl: 0,
    min_el: 20 * DEG,
    sun_el_max: -12 * DEG, // nautical twilight
  }));
}

export function spaceSensor(id: string, orbit: CandidateOrbit, limiting_mag: number, phase = 0): ModelSensor {
  return {
    id,
    kind: 'space',
    orbit,
    phase,
    limiting_mag,
    sun_excl: 40 * DEG,
    moon_excl: 10 * DEG,
    earth_excl: 10 * DEG,
    min_el: 0,
    sun_el_max: 0,
  };
}

export function sensorPosition(s: ModelSensor, tSec: number): Vec3 {
  return ORBITS[s.orbit].pos(tSec, s.phase);
}

// ---- photometry ------------------------------------------------------------------------------------
/**
 * Apparent visual magnitude of a diffuse (Lambertian) sphere of radius `radius_m`, geometric albedo
 * `albedo`, at range `range_km`, phase angle `phi` (rad):
 *   m = M_SUN − 2.5 log10( albedo · A · F(φ) / R² ),  A = π ρ²,  F(φ) = (2/(3π²)) [(π−φ) cos φ + sin φ].
 * McCue, Williams & Morford, "Optical characteristics of artificial satellites", Planet. Space Sci.
 * 19 (1971) 851–868; same form as PLAN.md §2.5 with A written out. ρ and R in the same units (m).
 */
export function apparentMag(radius_m: number, albedo: number, range_km: number, phi: number): number {
  const A = Math.PI * radius_m * radius_m;
  const F = (2 / (3 * Math.PI * Math.PI)) * ((Math.PI - phi) * Math.cos(phi) + Math.sin(phi));
  const R = range_km * 1000;
  const flux = (albedo * A * Math.max(F, 1e-12)) / (R * R);
  return M_SUN - 2.5 * Math.log10(flux);
}

// ---- visibility -----------------------------------------------------------------------------------
export interface TargetSpec {
  radius_m: number;
  albedo: number;
}

/** Order used to pick the "dominant" (least blocking) reason across sensors. Lower = closer to seeing. */
const REASON_RANK: Record<BlindReason, number> = {
  covered: -1,
  too_faint: 0,
  lunar_glare: 1,
  earth_exclusion: 2,
  sun_exclusion: 3,
  horizon: 4,
  daylight: 5,
  shadow: 6,
  out_of_fov: 7,
};

/** Umbra cone test: radius shrinks linearly from R_body to 0 at the apex (R_body · AU / (R_sun − R_body)). */
function inUmbra(target: Vec3, body: Vec3, bodyRadius: number, sunDir: Vec3): boolean {
  const d = sub(target, body);
  const along = dot(d, sunDir);
  if (along >= 0) return false; // sunward side
  const rUmbra = bodyRadius - (-along * (R_SUN_KM - bodyRadius)) / AU_KM;
  if (rUmbra <= 0) return false; // beyond the umbra apex (~1.38e6 km for Earth)
  const perp2 = dot(d, d) - along * along;
  return perp2 < rUmbra * rUmbra;
}

/**
 * Illuminated fraction of the Moon seen from Earth, from the barycentric Sun direction: at new Moon
 * the Sun is along +x (same side as the Moon), so k = (1 − cos θ)/2 with θ the Sun angle from +x.
 */
export function moonIlluminatedFraction(sunDir: Vec3): number {
  return (1 - sunDir[0]) / 2;
}

/**
 * Can sensor `s` detect a target at `target` (km, rotating frame) at time tSec? Returns the first
 * failing constraint, in the order: shadow → (ground: Sun depression, elevation) → Moon exclusion →
 * (space: Sun, Earth exclusion) → photometric limit. `sunDir` is the barycentric Sun unit vector.
 */
export function checkVisibility(s: ModelSensor, target: Vec3, tSec: number, sunDir: Vec3, spec: TargetSpec): BlindReason {
  if (inUmbra(target, EARTH_ROT_KM, R_EARTH_KM, sunDir) || inUmbra(target, MOON_ROT_KM, R_MOON_KM, sunDir)) return 'shadow';
  const obs = sensorPosition(s, tSec);
  const los = sub(target, obs);
  const range = norm(los);
  if (range < 1) return 'covered';
  if (s.kind === 'ground') {
    const zenith = unit(sub(obs, EARTH_ROT_KM));
    const sunEl = Math.asin(Math.max(-1, Math.min(1, dot(zenith, sunDir))));
    if (sunEl > s.sun_el_max) return 'daylight';
    const el = Math.asin(Math.max(-1, Math.min(1, dot(zenith, los) / range)));
    if (el < s.min_el) {
      // Below this site's mask. If the target sits in the day/twilight sky (solar elongation < 108°)
      // no night-side site can ever have it up, so the physical cause is daylight.
      return angle(los, sunDir) < 108 * DEG ? 'daylight' : 'horizon';
    }
  }
  const toMoon = sub(MOON_ROT_KM, obs);
  // Ground telescopes: moonlight sky background scales with the illuminated fraction (Krisciunas &
  // Schaefer 1991, PASP 103, 1033), so the avoidance angle is phase-scaled: 5° (new) → 25° (full).
  const moonExcl = s.kind === 'ground' ? (5 + 20 * moonIlluminatedFraction(sunDir)) * DEG : s.moon_excl;
  if (norm(toMoon) > 1 && angle(los, toMoon) < moonExcl) return 'lunar_glare';
  if (s.kind === 'space') {
    if (angle(los, sunDir) < s.sun_excl) return 'sun_exclusion';
    const toEarth = sub(EARTH_ROT_KM, obs);
    if (angle(los, toEarth) < s.earth_excl) return 'earth_exclusion';
  }
  // phase angle at the target: between the Sun and the observer directions
  const phi = angle(sunDir, [-los[0], -los[1], -los[2]]);
  if (apparentMag(spec.radius_m, spec.albedo, range, phi) > s.limiting_mag) return 'too_faint';
  return 'covered';
}

/** Count detecting sensors and the dominant blind reason at a point. */
export function evaluatePoint(sensors: ModelSensor[], target: Vec3, tSec: number, sunDir: Vec3, spec: TargetSpec): { count: number; reason: BlindReason } {
  let count = 0;
  let best: BlindReason = 'shadow';
  for (const s of sensors) {
    const r = checkVisibility(s, target, tSec, sunDir, spec);
    if (r === 'covered') count++;
    else if (REASON_RANK[r] < REASON_RANK[best]) best = r;
  }
  return { count, reason: count > 0 ? 'covered' : best };
}

/** Deterministic PRNG (mulberry32) for reproducible mock Monte Carlo. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
