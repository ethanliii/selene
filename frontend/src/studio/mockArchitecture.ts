/**
 * MOCK evaluator for POST /api/architecture/evaluate (backend offline). Deterministic (seeded),
 * labelled MOCK in the UI. It is a browser-side Monte Carlo over a notional target population and
 * random maneuver epochs, using the schematic orbit + visibility model in ./model.ts.
 *
 * Definitions (documented in the page's explainer; the backend's real evaluator supersedes these):
 *  - Epoch grid: `horizon_days` at 1 h steps from a fixed reference epoch (2026-03-01T00:00Z).
 *  - Target population: notional objects (SIMULATED, notional actor) drawn with equal probability
 *    from {DRO, 9:2 NRHO, L1 halo, L2 southern halo} with uniform random phase; diffuse sphere of
 *    `target_radius_m` (default 1.5 m) and `target_albedo` (default 0.2).
 *  - coverage %: fraction of (cell, epoch) samples of a coarse rotating-frame xy grid
 *    (x ∈ [−0.2, 1.4], y ∈ [−0.8, 0.8], 24 × 20 cells) with ≥ 1 detecting sensor.
 *  - observation: an object is "observed" at an epoch if ≥ 1 sensor can detect it (FOV/slew are
 *    assumed to be schedulable within the hour; the real tasker enforces them).
 *  - custody %: fraction of (object, epoch) samples whose time since last observation ≤ 12 h
 *    (surrogate for the covariance-trace threshold used by the backend; the first observation
 *    establishes custody).
 *  - revisit h: mean gap between consecutive observation epochs over all objects; objects never
 *    observed contribute the full horizon (censored, counted in never_observed_pct).
 *  - detection latency h: time from a uniformly random maneuver epoch in [0, horizon − 48 h] to the
 *    SECOND post-maneuver observation (two independent angle fixes are needed to attribute a
 *    residual to a burn rather than noise); censored at the horizon (undetected_pct). Mean and p95.
 *  - Common random numbers: the trial set (object class, phase, burn epoch) is drawn ONCE from `seed`
 *    and reused for every architecture, so architectures are compared on identical draws and a given
 *    architecture scores the same regardless of its position in the list.
 */
import { checkVisibility, evaluatePoint, groundNetwork, L_STAR_KM, mulberry32, ORBITS, spaceSensor, sunDirAt, type ModelSensor } from './model';
import type { ArchitectureEvaluateRequest, ArchitectureEvaluateResponse, ArchitectureScore, Vec3 } from './types';

export const MOCK_T0_MS = Date.UTC(2026, 2, 1, 0, 0, 0);
const TARGET_CLASSES = ['DRO', 'NRHO_9_2', 'L1_halo', 'L2_halo'] as const;
const CUSTODY_WINDOW_H = 12;

function percentile(sorted: number[], p: number): number {
  if (!sorted.length) return 0;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(p * sorted.length) - 1));
  return sorted[idx];
}

export function mockArchitectureEvaluate(req: ArchitectureEvaluateRequest): ArchitectureEvaluateResponse {
  const horizonH = Math.max(24, Math.round(req.horizon_days * 24));
  const nMc = Math.max(4, req.n_mc);
  const seed = req.seed ?? 20260301;
  const SPEC = { radius_m: req.target_radius_m ?? 1.5, albedo: req.target_albedo ?? 0.2 };

  // Common random numbers: one trial set shared by all architectures.
  const rng = mulberry32(seed);
  const trials = Array.from({ length: nMc }, () => ({
    cls: TARGET_CLASSES[Math.floor(rng() * TARGET_CLASSES.length)],
    phase: rng(),
    burnH: Math.floor(rng() * Math.max(1, horizonH - 48)),
  }));

  const scores: ArchitectureScore[] = req.architectures.map((arch) => {
    const sensors: ModelSensor[] = arch.ground_network ? groundNetwork() : [];
    arch.sensors.forEach((s, k) => sensors.push(spaceSensor(`${arch.name}-${k}`, s.orbit, s.limiting_mag, s.phase)));

    if (sensors.length === 0) {
      return { name: arch.name, coverage_pct: 0, custody_pct: 0, revisit_h: horizonH, detect_latency_h_mean: horizonH, detect_latency_h_p95: horizonH, n_mc: nMc, n_sensors: 0, undetected_pct: 100, never_observed_pct: 100 };
    }

    // --- volume coverage on a coarse grid, every 3 h to bound runtime
    const nx = 24;
    const ny = 20;
    let covered = 0;
    let total = 0;
    for (let h = 0; h < horizonH; h += 3) {
      const ms = MOCK_T0_MS + h * 3600e3;
      const sun = sunDirAt(ms);
      for (let ix = 0; ix < nx; ix++) {
        for (let iy = 0; iy < ny; iy++) {
          const p: Vec3 = [(-0.2 + (1.6 * (ix + 0.5)) / nx) * L_STAR_KM, (-0.8 + (1.6 * (iy + 0.5)) / ny) * L_STAR_KM, 0];
          if (evaluatePoint(sensors, p, h * 3600, sun, SPEC).count > 0) covered++;
          total++;
        }
      }
    }

    // --- object Monte Carlo
    let custodyHits = 0;
    let custodyTotal = 0;
    const gaps: number[] = [];
    let neverObserved = 0;
    const latencies: number[] = [];
    let undetected = 0;

    for (const { cls, phase, burnH } of trials) {
      const orbit = ORBITS[cls];
      const obsEpochs: number[] = [];
      let lastObs = -Infinity;
      for (let h = 0; h < horizonH; h++) {
        const tSec = h * 3600;
        const target = orbit.pos(tSec, phase);
        const sun = sunDirAt(MOCK_T0_MS + h * 3600e3);
        let seen = false;
        for (const s of sensors) {
          if (checkVisibility(s, target, tSec, sun, SPEC) === 'covered') {
            seen = true;
            break;
          }
        }
        if (seen) {
          if (obsEpochs.length) gaps.push(h - obsEpochs[obsEpochs.length - 1]);
          obsEpochs.push(h);
          lastObs = h;
        }
        if (lastObs > -Infinity) {
          custodyTotal++;
          if (h - lastObs <= CUSTODY_WINDOW_H) custodyHits++;
        } else {
          custodyTotal++;
        }
      }
      if (!obsEpochs.length) {
        neverObserved++;
        gaps.push(horizonH);
      }
      const post = obsEpochs.filter((h) => h > burnH);
      if (post.length >= 2) latencies.push(post[1] - burnH);
      else {
        undetected++;
        latencies.push(horizonH - burnH);
      }
    }

    const sortedLat = [...latencies].sort((a, b) => a - b);
    const mean = (a: number[]) => (a.length ? a.reduce((s, v) => s + v, 0) / a.length : 0);
    return {
      name: arch.name,
      coverage_pct: (100 * covered) / Math.max(1, total),
      custody_pct: (100 * custodyHits) / Math.max(1, custodyTotal),
      revisit_h: mean(gaps),
      detect_latency_h_mean: mean(latencies),
      detect_latency_h_p95: percentile(sortedLat, 0.95),
      n_mc: nMc,
      n_sensors: sensors.length,
      undetected_pct: (100 * undetected) / nMc,
      never_observed_pct: (100 * neverObserved) / nMc,
    };
  });

  return { scores, mock: true, method: 'browser-side schematic Monte Carlo (studio/mockArchitecture.ts)' };
}
