/**
 * Minimal Earth–Moon CR3BP toolkit for the browser (mock-mode orbit shapes and display-level motion).
 *
 * Nondimensional rotating (synodic) frame: barycenter at the origin, Earth at (−μ,0,0), Moon at (1−μ,0,0),
 * unit length L* = 384 400 km, unit time T* ≈ 3.7519e5 s (so one rotation = 2π), unit speed L*·T*⁻¹ ≈ 1.0245 km/s.
 *
 * Equations of motion (PLAN.md §2.1):
 *   ẍ − 2ẏ = ∂U/∂x,  ÿ + 2ẋ = ∂U/∂y,  z̈ = ∂U/∂z,
 *   U = ½(x²+y²) + (1−μ)/r₁ + μ/r₂,  r₁ = |(x+μ,y,z)|, r₂ = |(x−1+μ,y,z)|.
 * Jacobi constant C = 2U − v² is conserved and is used as the integrator health check.
 *
 * Integrators: fixed-step RK4 (cheap, deterministic, used for orbit sampling) and an adaptive
 * Dormand–Prince 5(4) (used for long display-level propagations). Periodic orbits are found with a
 * symmetric single-shooting differential corrector (perpendicular x–z-plane crossing) with a
 * finite-difference Jacobian, exactly like the backend's `orbits/shooting.py` but without the STM.
 *
 * This file is NOT the system of record for physics results; the backend library (DE440s, numba, STM-based
 * corrector) is. It exists so the UI has real orbit shapes when the API is down and so object motion in
 * mock mode is dynamically plausible.
 */
import type { State6, Vec3 } from '../api/types';

/** Earth–Moon mass ratio (DE440 GMs). */
export const MU = 0.012150585;
/** Characteristic length [km] and time [s] (PLAN.md §2.1). */
export const L_STAR_KM = 384400;
export const T_STAR_S = 375190.26; // sqrt(L*³ / (GM_E + GM_M)) with DE440 GMs
export const V_STAR_KMS = L_STAR_KM / T_STAR_S;

export type Deriv = (s: Float64Array, out: Float64Array) => void;

/** CR3BP equations of motion, state (x,y,z,vx,vy,vz) → derivative. */
export function eom(s: Float64Array, out: Float64Array, mu = MU): void {
  const x = s[0], y = s[1], z = s[2];
  const dx1 = x + mu, dx2 = x - 1 + mu;
  const r1sq = dx1 * dx1 + y * y + z * z;
  const r2sq = dx2 * dx2 + y * y + z * z;
  const r1i3 = 1 / (r1sq * Math.sqrt(r1sq));
  const r2i3 = 1 / (r2sq * Math.sqrt(r2sq));
  const ax = x - (1 - mu) * dx1 * r1i3 - mu * dx2 * r2i3;
  const ay = y - (1 - mu) * y * r1i3 - mu * y * r2i3;
  const az = -(1 - mu) * z * r1i3 - mu * z * r2i3;
  out[0] = s[3];
  out[1] = s[4];
  out[2] = s[5];
  out[3] = ax + 2 * s[4];
  out[4] = ay - 2 * s[3];
  out[5] = az;
}

/** Jacobi constant C = 2U − v². */
export function jacobi(s: ArrayLike<number>, mu = MU): number {
  const x = s[0], y = s[1], z = s[2];
  const r1 = Math.hypot(x + mu, y, z);
  const r2 = Math.hypot(x - 1 + mu, y, z);
  const U = 0.5 * (x * x + y * y) + (1 - mu) / r1 + mu / r2;
  return 2 * U - (s[3] * s[3] + s[4] * s[4] + s[5] * s[5]);
}

/** Collinear libration points from the quintic ∂U/∂x = 0 (Newton on x), plus L4/L5. */
export function lagrangePoints(mu = MU): Record<'L1' | 'L2' | 'L3' | 'L4' | 'L5', Vec3> {
  const f = (x: number) => x - ((1 - mu) * (x + mu)) / Math.abs(x + mu) ** 3 - (mu * (x - 1 + mu)) / Math.abs(x - 1 + mu) ** 3;
  const newton = (x0: number) => {
    let x = x0;
    for (let i = 0; i < 60; i++) {
      const h = 1e-7;
      const d = (f(x + h) - f(x - h)) / (2 * h);
      const step = f(x) / d;
      x -= step;
      if (Math.abs(step) < 1e-14) break;
    }
    return x;
  };
  return {
    L1: [newton(0.84), 0, 0],
    L2: [newton(1.16), 0, 0],
    L3: [newton(-1.0), 0, 0],
    L4: [0.5 - mu, Math.sqrt(3) / 2, 0],
    L5: [0.5 - mu, -Math.sqrt(3) / 2, 0],
  };
}

// ---------------------------------------------------------------------------------------------
// Integrators

const K1 = new Float64Array(6), K2 = new Float64Array(6), K3 = new Float64Array(6), K4 = new Float64Array(6), TMP = new Float64Array(6);

/** One classical RK4 step in place. */
export function rk4Step(s: Float64Array, h: number, mu = MU): void {
  eom(s, K1, mu);
  for (let i = 0; i < 6; i++) TMP[i] = s[i] + 0.5 * h * K1[i];
  eom(TMP, K2, mu);
  for (let i = 0; i < 6; i++) TMP[i] = s[i] + 0.5 * h * K2[i];
  eom(TMP, K3, mu);
  for (let i = 0; i < 6; i++) TMP[i] = s[i] + h * K3[i];
  eom(TMP, K4, mu);
  for (let i = 0; i < 6; i++) s[i] += (h / 6) * (K1[i] + 2 * K2[i] + 2 * K3[i] + K4[i]);
}

/** Propagate `s` (copied) by `tau` nondimensional time units with fixed-step RK4 (n steps). */
export function propagateRK4(s0: ArrayLike<number>, tau: number, nSteps: number, mu = MU): Float64Array {
  const s = Float64Array.from(s0 as ArrayLike<number>);
  const h = tau / nSteps;
  for (let i = 0; i < nSteps; i++) rk4Step(s, h, mu);
  return s;
}

/**
 * Sample a trajectory at `n` equally spaced epochs over [0, tau] (first sample = s0).
 * Returns positions (Vec3[]) and full states; `stepsPer` RK4 sub-steps between samples.
 */
export function sampleTrajectory(s0: ArrayLike<number>, tau: number, n: number, stepsPer = 4, mu = MU): { pos: Vec3[]; states: State6[] } {
  const s = Float64Array.from(s0 as ArrayLike<number>);
  const pos: Vec3[] = [[s[0], s[1], s[2]]];
  const states: State6[] = [Array.from(s) as State6];
  const h = tau / (n - 1) / stepsPer;
  for (let i = 1; i < n; i++) {
    for (let k = 0; k < stepsPer; k++) rk4Step(s, h, mu);
    pos.push([s[0], s[1], s[2]]);
    states.push(Array.from(s) as State6);
  }
  return { pos, states };
}

// Dormand–Prince 5(4) coefficients (Hairer, Nørsett & Wanner, Solving ODEs I, Table 5.2).
const A21 = 1 / 5;
const A31 = 3 / 40, A32 = 9 / 40;
const A41 = 44 / 45, A42 = -56 / 15, A43 = 32 / 9;
const A51 = 19372 / 6561, A52 = -25360 / 2187, A53 = 64448 / 6561, A54 = -212 / 729;
const A61 = 9017 / 3168, A62 = -355 / 33, A63 = 46732 / 5247, A64 = 49 / 176, A65 = -5103 / 18656;
const B1 = 35 / 384, B3 = 500 / 1113, B4 = 125 / 192, B5 = -2187 / 6784, B6 = 11 / 84;
const E1 = 71 / 57600, E3 = -71 / 16695, E4 = 71 / 1920, E5 = -17253 / 339200, E6 = 22 / 525, E7 = -1 / 40;

/**
 * Adaptive Dormand–Prince 5(4) propagation from t=0 to t=tau. Calls `onStep(t, s)` after each accepted step
 * (s is a reused buffer: copy if you keep it). Returns the final state.
 */
export function propagateDP45(
  s0: ArrayLike<number>,
  tau: number,
  opts: { rtol?: number; atol?: number; h0?: number; hmax?: number; onStep?: (t: number, s: Float64Array) => boolean | void; mu?: number } = {},
): Float64Array {
  const { rtol = 1e-9, atol = 1e-11, hmax = 0.05, onStep, mu = MU } = opts;
  const dir = Math.sign(tau) || 1;
  const s = Float64Array.from(s0 as ArrayLike<number>);
  const k1 = new Float64Array(6), k2 = new Float64Array(6), k3 = new Float64Array(6), k4 = new Float64Array(6);
  const k5 = new Float64Array(6), k6 = new Float64Array(6), k7 = new Float64Array(6), y = new Float64Array(6), yn = new Float64Array(6);
  let t = 0;
  let h = dir * Math.min(opts.h0 ?? 1e-3, Math.abs(tau), hmax);
  const T = Math.abs(tau);
  let guard = 0;
  eom(s, k1, mu);
  while (Math.abs(t) < T && guard++ < 300_000) {
    if (Math.abs(t + h) > T) h = dir * (T - Math.abs(t));
    for (let i = 0; i < 6; i++) y[i] = s[i] + h * A21 * k1[i];
    eom(y, k2, mu);
    for (let i = 0; i < 6; i++) y[i] = s[i] + h * (A31 * k1[i] + A32 * k2[i]);
    eom(y, k3, mu);
    for (let i = 0; i < 6; i++) y[i] = s[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
    eom(y, k4, mu);
    for (let i = 0; i < 6; i++) y[i] = s[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
    eom(y, k5, mu);
    for (let i = 0; i < 6; i++) y[i] = s[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
    eom(y, k6, mu);
    for (let i = 0; i < 6; i++) yn[i] = s[i] + h * (B1 * k1[i] + B3 * k3[i] + B4 * k4[i] + B5 * k5[i] + B6 * k6[i]);
    eom(yn, k7, mu);
    let err = 0;
    for (let i = 0; i < 6; i++) {
      const ei = h * (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k7[i]);
      const sc = atol + rtol * Math.max(Math.abs(s[i]), Math.abs(yn[i]));
      err += (ei / sc) ** 2;
    }
    err = Math.sqrt(err / 6);
    if (err <= 1 || Math.abs(h) < 1e-12) {
      t += h;
      s.set(yn);
      k1.set(k7); // FSAL
      if (onStep?.(t, s) === true) break;
    }
    const fac = Math.min(5, Math.max(0.2, 0.9 * Math.pow(Math.max(err, 1e-16), -0.2)));
    h = dir * Math.min(hmax, Math.abs(h) * fac);
    // Safety: a collision with a primary drives h → 0; bail out instead of spinning (state is then stale).
    if (Math.abs(h) < 1e-11 || !Number.isFinite(err)) break;
  }
  return s;
}

// ---------------------------------------------------------------------------------------------
// Symmetric periodic-orbit corrector

interface Crossing {
  /** State at the next y = 0 crossing (after leaving the x–z plane). */
  s: Float64Array;
  /** Time of the crossing (half period for symmetric orbits). */
  t: number;
}

/**
 * Integrate to the next y = 0 crossing (sign change) with adaptive DP45, then refine the crossing time by
 * bisection on the last step. Adaptive stepping matters for NRHO-class orbits whose perilune passes
 * (r₂ ≈ 0.01 L*) would be badly resolved by a fixed step.
 */
export function nextXZCrossing(s0: ArrayLike<number>, opts: { tmax?: number; rtol?: number; atol?: number; hmax?: number } = {}): Crossing | null {
  const { tmax = 12, rtol = 1e-10, atol = 1e-12, hmax = 0.01 } = opts;
  const prev = Float64Array.from(s0 as ArrayLike<number>);
  let prevT = 0;
  let left = false;
  let hit: { t: number } | null = null;
  propagateDP45(s0, tmax, {
    rtol,
    atol,
    hmax,
    onStep: (t, s) => {
      if (!left) {
        if (Math.abs(s[1]) > 1e-7) left = true;
        prev.set(s);
        prevT = t;
        return;
      }
      if (Math.sign(s[1]) !== Math.sign(prev[1])) {
        hit = { t };
        return true;
      }
      prev.set(s);
      prevT = t;
      return;
    },
  });
  if (!hit) return null;
  // Bisection on the sub-step length from `prev` (y is monotone across one short step).
  let lo = 0;
  let hi = (hit as { t: number }).t - prevT;
  let trial: Float64Array = Float64Array.from(prev);
  for (let i = 0; i < 40; i++) {
    const mid = 0.5 * (lo + hi);
    trial = propagateDP45(prev, mid, { rtol, atol, hmax });
    if (Math.sign(trial[1]) === Math.sign(prev[1]) || trial[1] === 0) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-13) break;
  }
  return { s: trial, t: prevT + 0.5 * (lo + hi) };
}

/**
 * Tabulate positions at `n` uniform epochs over [0, tau] using DP45 dense steps + linear interpolation.
 * Suitable for display (frames every few minutes); not for estimation.
 */
export function tabulate(s0: ArrayLike<number>, tau: number, n: number, opts: { hmax?: number; rtol?: number } = {}): Vec3[] {
  const ts: number[] = [0];
  const xs: number[] = [s0[0]], ys: number[] = [s0[1]], zs: number[] = [s0[2]];
  propagateDP45(s0, tau, {
    rtol: opts.rtol ?? 1e-9,
    atol: 1e-11,
    hmax: opts.hmax ?? 0.01,
    onStep: (t, s) => {
      ts.push(t);
      xs.push(s[0]);
      ys.push(s[1]);
      zs.push(s[2]);
    },
  });
  const out: Vec3[] = [];
  let k = 0;
  for (let i = 0; i < n; i++) {
    const t = (tau * i) / Math.max(1, n - 1);
    while (k < ts.length - 2 && ts[k + 1] < t) k++;
    const t0 = ts[k], t1 = ts[k + 1] ?? ts[k];
    const w = t1 > t0 ? Math.min(1, Math.max(0, (t - t0) / (t1 - t0))) : 0;
    const j = Math.min(k + 1, ts.length - 1);
    out.push([xs[k] + (xs[j] - xs[k]) * w, ys[k] + (ys[j] - ys[k]) * w, zs[k] + (zs[j] - zs[k]) * w]);
  }
  return out;
}

export interface PeriodicOrbit {
  ic: State6;
  period: number;
  jacobi: number;
  /** |ẋ|+|ż| at the half-period crossing (closure residual). */
  residual: number;
  converged: boolean;
}

/**
 * Symmetric single shooting. Free variables:
 *   'vy'    → planar (Lyapunov/DRO): drive ẋ(T/2) = 0 with ẏ₀.
 *   'x,vy'  → halo with fixed z₀: drive ẋ(T/2) = ż(T/2) = 0 with (x₀, ẏ₀).
 *   'z,vy'  → halo with fixed x₀: drive ẋ(T/2) = ż(T/2) = 0 with (z₀, ẏ₀).
 */
export function correctSymmetric(x0: number, z0: number, vy0: number, free: 'vy' | 'x,vy' | 'z,vy', opts: { iters?: number; tol?: number } = {}): PeriodicOrbit {
  const { iters = 30, tol = 1e-8 } = opts;
  let p = free === 'vy' ? [vy0] : free === 'x,vy' ? [x0, vy0] : [z0, vy0];
  const state = (q: number[]): State6 =>
    free === 'vy' ? [x0, 0, z0, 0, q[0], 0] : free === 'x,vy' ? [q[0], 0, z0, 0, q[1], 0] : [x0, 0, q[0], 0, q[1], 0];
  const F = (q: number[]): { f: number[]; t: number } | null => {
    const c = nextXZCrossing(state(q));
    if (!c) return null;
    return free === 'vy' ? { f: [c.s[3]], t: c.t } : { f: [c.s[3], c.s[5]], t: c.t };
  };
  let last: { f: number[]; t: number } | null = F(p);
  let converged = false;
  for (let it = 0; it < iters && last; it++) {
    const n = Math.hypot(...last.f);
    if (n < tol) {
      converged = true;
      break;
    }
    // Finite-difference Jacobian.
    const m = p.length;
    const J: number[][] = Array.from({ length: m }, () => Array(m).fill(0));
    let bad = false;
    for (let j = 0; j < m; j++) {
      const d = 1e-6 * Math.max(1, Math.abs(p[j]));
      const q = p.slice();
      q[j] += d;
      const Fq = F(q);
      if (!Fq) {
        bad = true;
        break;
      }
      for (let i = 0; i < m; i++) J[i][j] = (Fq.f[i] - last.f[i]) / d;
    }
    if (bad) break;
    let dp: number[];
    if (m === 1) dp = [-last.f[0] / J[0][0]];
    else {
      const det = J[0][0] * J[1][1] - J[0][1] * J[1][0];
      if (Math.abs(det) < 1e-14) break;
      dp = [-(J[1][1] * last.f[0] - J[0][1] * last.f[1]) / det, -(-J[1][0] * last.f[0] + J[0][0] * last.f[1]) / det];
    }
    // Damped step + backtracking line search: keeps sensitive (NRHO) orbits from diverging and stops the
    // planar corrector from overshooting onto a different x-axis crossing.
    let scale = Math.min(1, 0.05 / Math.max(1e-12, Math.hypot(...dp)));
    let next: { f: number[]; t: number } | null = null;
    let q = p;
    for (let k = 0; k < 6; k++) {
      q = p.map((v, i) => v + scale * dp[i]);
      next = F(q);
      if (next && Math.hypot(...next.f) < n) break;
      scale *= 0.5;
    }
    if (!next) break;
    p = q;
    last = next;
  }
  const ic = state(p);
  const residual = last ? Math.hypot(...last.f) : Infinity;
  const period = last ? 2 * last.t : NaN;
  return { ic, period, jacobi: jacobi(ic), residual, converged: converged || residual < 1e-6 };
}

// ---------------------------------------------------------------------------------------------
// Family seeds (literature initial conditions) and generation

export interface FamilyMember {
  id: string;
  ic: State6;
  period: number;
  jacobi: number;
  stability: number;
  samples_rot: Vec3[];
  approximate: boolean;
}

export interface Family {
  name: string;
  members: FamilyMember[];
}

/**
 * Literature seeds (nondimensional, Earth–Moon):
 *  - L1 Lyapunov: x₀ = 0.8234, ẏ₀ = 0.1263, T ≈ 2.743 — Koon, Lo, Marsden & Ross, "Dynamical Systems, the
 *    Three-Body Problem and Space Mission Design" (2011), §4.
 *  - L2 Lyapunov: x₀ = 1.1809, ẏ₀ = −0.1559, T ≈ 3.41 — approximate seed; corrected here.
 *  - L2 southern halo / NRHO-like: x₀ = 1.0221, z₀ = −0.1821, ẏ₀ = −0.1033, T ≈ 1.511 — approximate 9:2 NRHO IC
 *    (cf. Zimovan-Spreen, Howell & Davis 2020; Lee 2019 Gateway white paper); corrected here, labelled approx.
 *  - DRO: x₀ = 0.80, ẏ₀ ≈ 0.52 — approximate seed for a large DRO; corrected here.
 */
export const SEEDS = {
  l1Lyapunov: { x0: 0.8234, z0: 0, vy0: 0.1263, period: 2.743 },
  l2Lyapunov: { x0: 1.1809, z0: 0, vy0: -0.1559, period: 3.41 },
  l2SouthHalo: { x0: 1.0221, z0: -0.1821, vy0: -0.1033, period: 1.511 },
  dro: { x0: 0.8, z0: 0, vy0: 0.52, period: 3.2 },
} as const;

/** Sample one full period of a corrected orbit (n points, closed back to the first). */
export function sampleOrbit(o: PeriodicOrbit, n = 240): Vec3[] {
  const pos = tabulate(o.ic, o.period, n);
  pos[pos.length - 1] = pos[0]; // force visual closure (residual is reported separately)
  return pos;
}

/** Crude stability index from a finite-difference monodromy estimate on the (x, vy) planar subspace. */
function stabilityIndex(o: PeriodicOrbit): number {
  const d = 1e-6;
  const cols: number[][] = [];
  const base = propagateDP45(o.ic, o.period, { rtol: 1e-10, atol: 1e-12 });
  for (let j = 0; j < 6; j++) {
    const s = o.ic.slice() as State6;
    s[j] += d;
    const f = propagateDP45(s, o.period, { rtol: 1e-10, atol: 1e-12 });
    cols.push(Array.from({ length: 6 }, (_, i) => (f[i] - base[i]) / d));
  }
  // Largest eigenvalue magnitude via power iteration on M = cols^T (columns are M e_j).
  let v = [1, 0.3, 0.2, 0.1, 0.5, 0.4];
  let lam = 1;
  for (let it = 0; it < 60; it++) {
    const w = Array(6).fill(0);
    for (let j = 0; j < 6; j++) for (let i = 0; i < 6; i++) w[i] += cols[j][i] * v[j];
    lam = Math.hypot(...w);
    if (!Number.isFinite(lam) || lam === 0) return 1;
    v = w.map((x) => x / lam);
  }
  return 0.5 * (lam + 1 / lam);
}

function continueFamily(
  name: string,
  prefix: string,
  seed: { x0: number; z0: number; vy0: number },
  free: 'vy' | 'x,vy' | 'z,vy',
  steps: { param: 'x0' | 'z0'; values: number[] },
  approximate: boolean,
): Family {
  const members: FamilyMember[] = [];
  let x0 = seed.x0, z0 = seed.z0, vy0 = seed.vy0;
  let prev: PeriodicOrbit | null = null;
  for (let i = 0; i < steps.values.length; i++) {
    if (steps.param === 'x0') x0 = steps.values[i];
    else z0 = steps.values[i];
    const o = correctSymmetric(x0, z0, vy0, free);
    if (!o.converged || !Number.isFinite(o.period)) continue;
    // Natural-parameter continuation: carry the converged free variables forward.
    x0 = o.ic[0];
    z0 = o.ic[2];
    vy0 = o.ic[4];
    prev = o;
    members.push({
      id: `${prefix}-${String(i + 1).padStart(2, '0')}`,
      ic: o.ic,
      period: o.period,
      jacobi: o.jacobi,
      stability: stabilityIndex(o),
      samples_rot: sampleOrbit(o),
      approximate,
    });
  }
  void prev;
  return { name, members };
}

let cached: Family[] | null = null;

/** Generate the mock orbit library once (≈100 ms). Families are labelled "(browser CR3BP)". */
export function mockFamilies(): Family[] {
  if (cached) return cached;
  const L = lagrangePoints();
  const lin = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
  const fams: Family[] = [
    continueFamily('L1 Lyapunov', 'L1LY', SEEDS.l1Lyapunov, 'vy', { param: 'x0', values: lin(SEEDS.l1Lyapunov.x0, L.L1[0] - 0.004, 6) }, false),
    continueFamily('L2 Lyapunov', 'L2LY', SEEDS.l2Lyapunov, 'vy', { param: 'x0', values: lin(SEEDS.l2Lyapunov.x0, L.L2[0] + 0.004, 6) }, true),
    continueFamily('L2 southern halo / NRHO', 'L2HS', SEEDS.l2SouthHalo, 'z,vy', { param: 'x0', values: lin(SEEDS.l2SouthHalo.x0, 1.10, 6) }, true),
    continueFamily('DRO', 'DRO', SEEDS.dro, 'vy', { param: 'x0', values: lin(0.8, 0.92, 7) }, true),
  ];
  cached = fams.filter((f) => f.members.length > 0);
  return cached;
}

/** Position on a sampled closed orbit at phase φ ∈ [0,1) (linear interpolation between samples). */
export function orbitPointAtPhase(samples: Vec3[], phase: number): Vec3 {
  const n = samples.length - 1;
  const u = ((phase % 1) + 1) % 1;
  const f = u * n;
  const i = Math.floor(f);
  const a = samples[i], b = samples[(i + 1) % (n + 1)];
  const w = f - i;
  return [a[0] + (b[0] - a[0]) * w, a[1] + (b[1] - a[1]) * w, a[2] + (b[2] - a[2]) * w];
}
