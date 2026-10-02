/**
 * Scene units and physical constants used by the 3D view.
 * Scene unit = 1 Earth–Moon characteristic length L* = 384 400 km (CR3BP nondimensional length).
 * All radii below are converted to scene units; sources: IAU 2015 nominal Earth equatorial radius
 * 6378.1366 km (IERS Conventions 2010), mean lunar radius 1737.4 km (IAU WGCCRE 2015), DE440 GMs.
 */
import type { Vec3 } from '../api/types';

export const L_STAR_KM = 384400;
export const kmToScene = (km: number): number => km / L_STAR_KM;

export const EARTH_RADIUS_KM = 6378.1366;
export const MOON_RADIUS_KM = 1737.4;
export const EARTH_RADIUS = kmToScene(EARTH_RADIUS_KM);
export const MOON_RADIUS = kmToScene(MOON_RADIUS_KM);

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
 * PLACEHOLDER synodic rotation rate kept for reference only; the inertial view uses the DE440s basis when the
 * backend is up (lib/ephem.ts) and the mean lunar longitude otherwise.
 */
export const SYNODIC_RATE_RAD_S = (2 * Math.PI) / (27.28 * 86400);

/**
 * Camera defaults (scene units). The overview looks at the Earth–Moon system from the south, 31° above the plane,
 * far enough back that Earth, Moon, L1/L2 and the L4/L5 triangle all fit a 16:9 viewport (the Earth keeps a
 * minimum on-screen size through its glow sprite, Bodies.tsx).
 */
/** Default framing: Earth, Moon, L1–L5 (L4/L5 at y = ±0.866) all inside the 16:9 viewport above the timeline. */
export const CAMERA = {
  position: [0.5, -1.9, 1.08] as Vec3,
  target: [0.5, -0.1, -0.03] as Vec3,
  near: 1e-4,
  far: 50,
  fov: 38,
};

/** Orbit-family colours (by canonical family name, else keyword, else by index). Shared by the scene and the legend. */
export const FAMILY_PALETTE = ['#4cc9f0', '#9b5de5', '#f2cc8f', '#81b29a', '#e07a5f', '#f5b700', '#2dd4bf', '#ff8fab'];
const FAMILY_COLORS: Record<string, string> = {
  l1_lyapunov: '#4cc9f0',
  l2_lyapunov: '#9b5de5',
  l1_halo_n: '#e07a5f',
  l1_halo_s: '#c96a62',
  l2_halo_n: '#f2cc8f',
  l2_halo_s: '#e8b85c',
  dro: '#81b29a',
  'resonant_3:1': '#ff8fab',
  'resonant_2:1': '#c86fa8',
};
/** Highlight colour of the 9:2 NRHO member. */
export const NRHO_COLOR = '#ffe59a';

export function familyColor(name: string, index: number): string {
  const n = name.toLowerCase();
  if (FAMILY_COLORS[n]) return FAMILY_COLORS[n];
  if (n.includes('dro')) return FAMILY_COLORS.dro;
  if (n.includes('nrho') || (n.includes('l2') && n.includes('halo'))) return FAMILY_COLORS.l2_halo_s;
  if (n.includes('l1') && n.includes('halo')) return FAMILY_COLORS.l1_halo_n;
  if (n.includes('l1')) return FAMILY_COLORS.l1_lyapunov;
  if (n.includes('l2')) return FAMILY_COLORS.l2_lyapunov;
  if (n.includes('resonant')) return FAMILY_COLORS['resonant_3:1'];
  if (n.includes('frozen') || n.includes('elfo')) return '#f5b700';
  return FAMILY_PALETTE[index % FAMILY_PALETTE.length];
}

/** Human label for a family name: "L2_halo_S" → "L2 halo (S)", "resonant_3:1" → "3:1 resonant". */
export function familyLabel(name: string): string {
  const m = /^(L[12])_(lyapunov|halo)(?:_([NS]))?$/i.exec(name);
  if (m) return `${m[1].toUpperCase()} ${m[2].toLowerCase() === 'lyapunov' ? 'Lyapunov' : 'halo'}${m[3] ? ` (${m[3].toUpperCase()})` : ''}`;
  const r = /^resonant_(\d+:\d+)$/i.exec(name);
  if (r) return `${r[1]} resonant`;
  return name.replace(/_/g, ' ');
}

/** Camera presets (rotating-frame coordinates, scene units). */
export const CAMERA_PRESETS: Record<'overview' | 'earth' | 'moon' | 'l1' | 'l2', { position: Vec3; target: Vec3; label: string }> = {
  overview: { position: CAMERA.position, target: CAMERA.target, label: 'Earth–Moon' },
  earth: { position: [-MU + 0.16, -0.2, 0.11], target: [-MU, 0, 0], label: 'Earth' },
  moon: { position: [1 - MU + 0.22, -0.3, 0.17], target: [1 - MU, 0, 0], label: 'Moon' },
  l1: { position: [0.836915 + 0.1, -0.16, 0.09], target: [0.836915, 0, 0], label: 'L1' },
  l2: { position: [1.155682 + 0.12, -0.2, 0.11], target: [1.155682, 0, 0], label: 'L2' },
};

/** Palette mirrors theme.css tokens for materials (three.js needs literal colors). */
export const COLORS = {
  accent: '#4cc9f0',
  accent2: '#9b5de5',
  ok: '#2dd4bf',
  warn: '#f5b700',
  alert: '#ff4d4f',
  sim: '#c9b27a',
  muted: '#7c8da3',
  text: '#d6e2f0',
  grid: '#1c2733',
  earth: '#0e3a5a',
  earthEmissive: '#0a2a44',
  atmosphere: '#2a9fd6',
  moon: '#8a8f99',
};
