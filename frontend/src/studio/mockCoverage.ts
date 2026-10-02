/**
 * MOCK generator for POST /api/coverage (used when the backend is offline).
 *
 * Heuristics (all evaluated with the shared geometric model in ./model.ts):
 *  - Grid: rotating-frame xy slice (z = 0), nondimensional, cell index i = ix * ny + iy.
 *  - Sensors: 3 notional equatorial ground sites (m_lim 19.5, el > 20°, Sun < −12°, Moon avoidance
 *    5–25° scaled with the illuminated fraction, see model.ts checkVisibility)
 *    plus, per preset, a GEO observer (m_lim 18.0), an L2 southern-halo observer (18.5) and/or a DRO
 *    observer (18.5), each with Sun 40° / Moon 10° / Earth 10° exclusion.
 *  - Target: diffuse sphere (radius, albedo from the request), McCue et al. (1971) magnitude law.
 *  - Sun direction from the mean synodic phase relative to the 2000-01-06 reference new Moon.
 *  - values[t][i] = number of sensors able to detect; reasons[t][i] = dominant blind reason code.
 * Expected structure: a dayside blind lobe toward the Sun for the ground network, a lunar-glare
 * hole around the Moon, a "too faint" rim at large Earth range for small targets, and large
 * recoveries when space observers near the Moon are added.
 */
import { evaluatePoint, groundNetwork, L_STAR_KM, spaceSensor, sunDirAt, type ModelSensor } from './model';
import { REASON_CODES, toStudioPreset, type CoverageRequest, type CoverageResponse, type NetworkPreset, type Vec3 } from './types';

export function presetSensors(preset: NetworkPreset): ModelSensor[] {
  const s = groundNetwork();
  if (preset === 'ground+geo' || preset === 'ground+all') s.push(spaceSensor('SPC-GEO-1', 'GEO', 18.0, 0.37));
  if (preset === 'ground+l2' || preset === 'ground+all') s.push(spaceSensor('SPC-L2H-1', 'L2_halo', 18.5, 0.1));
  if (preset === 'ground+dro' || preset === 'ground+all') s.push(spaceSensor('SPC-DRO-1', 'DRO', 18.5, 0.2));
  return s;
}

export function mockCoverage(req: CoverageRequest): CoverageResponse {
  const { nx, ny, xmin, xmax, ymin, ymax } = req.grid;
  const x = Array.from({ length: nx }, (_, i) => xmin + ((xmax - xmin) * i) / Math.max(1, nx - 1));
  const y = Array.from({ length: ny }, (_, j) => ymin + ((ymax - ymin) * j) / Math.max(1, ny - 1));
  const t0ms = Date.parse(req.t0);
  const t1ms = Date.parse(req.t1);
  const nT = Math.max(1, req.n_t);
  const epochs = Array.from({ length: nT }, (_, k) => new Date(t0ms + ((t1ms - t0ms) * k) / Math.max(1, nT - 1)).toISOString());
  const sensors = presetSensors(toStudioPreset(req.network));
  const spec = { radius_m: req.target_radius_m ?? 1.0, albedo: req.target_albedo ?? 0.2 };
  const codeOf = new Map(REASON_CODES.map((r, i) => [r, i] as const));

  const values: number[][] = [];
  const reasons: number[][] = [];
  for (let k = 0; k < nT; k++) {
    const ms = t0ms + ((t1ms - t0ms) * k) / Math.max(1, nT - 1);
    const tSec = (ms - t0ms) / 1000;
    const sun = sunDirAt(ms);
    const v = new Array<number>(nx * ny);
    const r = new Array<number>(nx * ny);
    for (let ix = 0; ix < nx; ix++) {
      for (let iy = 0; iy < ny; iy++) {
        const p: Vec3 = [x[ix] * L_STAR_KM, y[iy] * L_STAR_KM, 0];
        const e = evaluatePoint(sensors, p, tSec, sun, spec);
        v[ix * ny + iy] = e.count;
        r[ix * ny + iy] = codeOf.get(e.reason) ?? 0;
      }
    }
    values.push(v);
    reasons.push(r);
  }
  return { grid: { ...req.grid, x, y }, values, epochs, reasons, sensors_used: sensors.map((s) => s.id) };
}

/**
 * Client-side estimate of blind reasons for a backend response that carries `values` but no
 * `reasons` (the §5 contract does not require them). Uses the same surrogate model, so it is an
 * estimate and is labelled as such in the UI.
 */
export function estimateReasons(req: CoverageRequest, res: CoverageResponse): number[][] {
  const { x, y } = res.grid;
  const ny = y.length;
  const t0ms = Date.parse(req.t0);
  const sensors = presetSensors(toStudioPreset(req.network));
  const spec = { radius_m: req.target_radius_m ?? 1.0, albedo: req.target_albedo ?? 0.2 };
  const codeOf = new Map(REASON_CODES.map((r, i) => [r, i] as const));
  return res.values.map((v, k) => {
    const ms = Date.parse(res.epochs[k] ?? req.t0);
    const sun = sunDirAt(ms);
    const tSec = (ms - t0ms) / 1000;
    return v.map((count, i) => {
      if (count > 0) return 0;
      const ix = Math.floor(i / ny);
      const iy = i % ny;
      const e = evaluatePoint(sensors, [x[ix] * L_STAR_KM, y[iy] * L_STAR_KM, 0], tSec, sun, spec);
      return codeOf.get(e.reason === 'covered' ? 'too_faint' : e.reason) ?? 0;
    });
  });
}
