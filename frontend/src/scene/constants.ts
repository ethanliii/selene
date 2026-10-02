/**
 * Scene units and physical constants used by the 3D view.
 * Scene unit = 1 Earth–Moon characteristic length L* = 384 400 km (CR3BP nondimensional length).
 * All radii below are converted to scene units; sources: IAU 2015 nominal Earth equatorial radius
 * 6378.1366 km (IERS Conventions 2010), mean lunar radius 1737.4 km (IAU WGCCRE 2015), DE440 GMs.
 */
import type { Vec3 } from '../api/types';

export const L_STAR_KM = 384400;
export const kmToScene = (km: number): number => km / L_STAR_KM;

export const EARTH_RADIUS = kmToScene(6378.1366);
export const MOON_RADIUS = kmToScene(1737.4);

/** CR3BP mass ratio μ = GM_M/(GM_E+GM_M) with DE440 GMs (PLAN.md §2.1). */
export const MU = 4902.800118 / (398600.435507 + 4902.800118); // ≈ 0.0121505

/** Rotating-frame positions of the primaries (nondimensional). */
export const EARTH_ROT: Vec3 = [-MU, 0, 0];
export const MOON_ROT: Vec3 = [1 - MU, 0, 0];

/** Libration points in the rotating frame (PLAN.md §2.1; L1–L3 from the collinear quintic). */
export const LAGRANGE_ROT: { name: 'L1' | 'L2' | 'L3' | 'L4' | 'L5'; pos: Vec3 }[] = [
  { name: 'L1', pos: [0.836915, 0, 0] },
  { name: 'L2', pos: [1.155682, 0, 0] },
  { name: 'L3', pos: [-1.005063, 0, 0] },
  { name: 'L4', pos: [0.5 - MU, 0.866025, 0] },
  { name: 'L5', pos: [0.5 - MU, -0.866025, 0] },
];

/**
 * PLACEHOLDER synodic rotation rate used for the inertial-frame view until the ephemeris API
 * (/api/ephemeris/bodies -> true Earth–Moon line direction vs. time) is wired by a later agent.
 * Mean synodic month 29.53 d would be the Sun-relative rate; the task spec asks for 2π/(27.28 d)
 * here (close to the sidereal month 27.3217 d). Rotation is about +z (ecliptic-ish pole).
 */
export const SYNODIC_RATE_RAD_S = (2 * Math.PI) / (27.28 * 86400);

/** Camera defaults (scene units). */
export const CAMERA = {
  position: [1.9, -1.6, 1.15] as Vec3,
  near: 1e-4,
  far: 50,
  fov: 40,
};

/** Palette mirrors theme.css tokens for materials (three.js needs literal colors). */
export const COLORS = {
  accent: '#4cc9f0',
  accent2: '#9b5de5',
  ok: '#2dd4bf',
  warn: '#f5b700',
  alert: '#ff4d4f',
  sim: '#ffb86b',
  muted: '#7c8da3',
  text: '#d6e2f0',
  grid: '#1c2733',
  earth: '#0e3a5a',
  earthEmissive: '#0a2a44',
  atmosphere: '#2a9fd6',
  moon: '#8a8f99',
};
