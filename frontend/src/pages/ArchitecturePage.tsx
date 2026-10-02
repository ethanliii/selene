/**
 * Architecture Trade Studio.
 * Left: candidate orbits (add sensors, edit specs) and up to 4 saved architectures.
 * Centre: "Map | Results" segmented view — the 2D rotating-frame plot of the selected architecture against the
 *         notional target population, or a 2×2 grid of metric charts once a run exists (the view switches to
 *         Results when a run completes).
 * Right: full-width Run Monte Carlo button + run parameters, side-by-side table (best row highlighted), summary,
 *        Export JSON, metric explainer (collapsed).
 * Data: POST /api/architecture/evaluate (backend Monte Carlo; studio/api.ts adapts the contract). When the backend
 * is unreachable the browser schematic model answers and the results carry a muted SCHEMATIC MODEL tag.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, ErrorBar, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import { Icon } from '../panels/icons';
import { TopBar } from '../panels/TopBar';
import { studioApi } from '../studio/api';
import { ARCH_COLORS } from '../studio/ramp';
import { SystemPlot } from '../studio/SystemPlot';
import {
  CANDIDATE_ORBITS,
  type ArchitectureDef,
  type ArchitectureEvaluateRequest,
  type ArchitectureEvaluateResponse,
  type ArchitectureScore,
  type CandidateOrbit,
  type SensorSpec,
} from '../studio/types';
import '../studio/studio.css';

const MAX_ARCH = 4;
let uid = 0;
const nextId = (p: string) => `${p}-${++uid}-${Date.now().toString(36)}`;

function makeSensor(orbit: CandidateOrbit, phase = 0): SensorSpec {
  const info = CANDIDATE_ORBITS.find((c) => c.id === orbit)!;
  return { id: nextId('S'), orbit, aperture_m: 0.5, limiting_mag: info.default_limiting_mag, fov_deg: 2.0, slew_rate_dps: 1.0, phase };
}

function defaultArchitectures(): ArchitectureDef[] {
  return [
    { id: nextId('A'), name: 'A · Ground only', ground_network: true, sensors: [], slot: 0 },
    { id: nextId('A'), name: 'B · Ground + GEO', ground_network: true, sensors: [makeSensor('GEO', 0.3)], slot: 1 },
    { id: nextId('A'), name: 'C · Ground + L2 + DRO', ground_network: true, sensors: [makeSensor('L2_halo', 0.1), makeSensor('DRO', 0.2)], slot: 2 },
  ];
}

/** First colour slot (0..MAX_ARCH-1) not held by a live architecture; slots survive deletes of others. */
function freeSlot(list: ArchitectureDef[]): number {
  const used = new Set(list.map((a) => a.slot));
  for (let k = 0; k < MAX_ARCH; k++) if (!used.has(k)) return k;
  return list.length % ARCH_COLORS.length;
}

/** First letter A.. not already used as a "X · name" prefix, so names stay unique after deletes. */
function freeLetter(list: ArchitectureDef[]): string {
  const used = new Set(list.map((a) => a.name.split('·')[0].trim().toUpperCase()));
  for (let k = 0; k < 26; k++) {
    const L = String.fromCharCode(65 + k);
    if (!used.has(L)) return L;
  }
  return String(list.length + 1);
}

/** Short label for charts/table: the letter prefix of "X · name" (falls back to the first word). */
const shortName = (name: string) => name.split('·')[0].trim().slice(0, 6);

/** Round up to 1, 2, 2.5 or 5 × 10^k so axis ends land on a clean tick (avoids 186.29999999999998). */
function niceCeil(v: number): number {
  if (!(v > 0)) return 1;
  const exp = Math.floor(Math.log10(v));
  const base = Math.pow(10, exp);
  for (const m of [1, 2, 2.5, 5, 10]) if (m * base >= v - 1e-9) return m * base;
  return 10 * base;
}

interface RunState {
  req: ArchitectureEvaluateRequest;
  res: ArchitectureEvaluateResponse;
  /** colour slot per score row, aligned with the architecture list at run time */
  colors: string[];
  mock: boolean;
  elapsed_ms: number;
  at: string;
}

type MetricKey = 'coverage_pct' | 'custody_pct' | 'revisit_h' | 'detect_latency_h_mean';
const METRICS: { key: MetricKey; label: string; unit: string; higherBetter: boolean; fmt: (v: number) => string }[] = [
  { key: 'coverage_pct', label: 'Coverage', unit: '%', higherBetter: true, fmt: (v) => v.toFixed(1) },
  { key: 'custody_pct', label: 'Custody', unit: '%', higherBetter: true, fmt: (v) => v.toFixed(1) },
  { key: 'revisit_h', label: 'Mean revisit', unit: 'h', higherBetter: false, fmt: (v) => v.toFixed(1) },
  { key: 'detect_latency_h_mean', label: 'Detect latency (mean)', unit: 'h', higherBetter: false, fmt: (v) => v.toFixed(1) },
];

function bestIndex(scores: ArchitectureScore[], key: keyof ArchitectureScore, higherBetter: boolean): number {
  let bi = -1;
  let bv = higherBetter ? -Infinity : Infinity;
  scores.forEach((s, i) => {
    const v = s[key] as number;
    if (higherBetter ? v > bv : v < bv) {
      bv = v;
      bi = i;
    }
  });
  return bi;
}

const tickStyle = { fill: '#8d9db3', fontSize: 12 };

function MetricChart({ title, unit, scores, colors, dataKey, secondKey, loKey, hiKey, domainMax, tall }: { title: string; unit: string; scores: ArchitectureScore[]; colors: string[]; dataKey: keyof ArchitectureScore; secondKey?: keyof ArchitectureScore; loKey?: keyof ArchitectureScore; hiKey?: keyof ArchitectureScore; domainMax?: number; tall?: boolean }) {
  const fmtLabel = (v: number) => v.toFixed(v >= 100 ? 0 : 1);
  const span = domainMax ?? Math.max(1, ...scores.map((s) => s[dataKey] as number));
  const whisk = !!loKey && !!hiKey && scores.some((s) => typeof s[loKey] === 'number' && typeof s[hiKey] === 'number');
  const data = scores.map((s, i) => {
    const v = s[dataKey] as number;
    const v2 = secondKey ? (s[secondKey] as number) : undefined;
    // the p95 direct label is dropped when it would sit on top of the mean label (< 8 % of the axis
    // apart); the value stays in the tooltip and the table
    const v2label = v2 === undefined ? '' : Math.abs(v2 - v) / span < 0.08 ? '' : fmtLabel(v2);
    const lo = loKey ? (s[loKey] as number | undefined) : undefined;
    const hi = hiKey ? (s[hiKey] as number | undefined) : undefined;
    // asymmetric whisker [below, above] the mean: the p05–p95 band over the Monte Carlo draws
    const err = typeof lo === 'number' && typeof hi === 'number' ? [Math.max(0, v - lo), Math.max(0, hi - v)] : undefined;
    const pad = typeof hi === 'number' ? Math.max(0, hi - v) : 0;
    return { name: s.name, short: shortName(s.name), v, v2, v2label, color: colors[i], err, lo, hi, pad };
  });
  return (
    <div className="metric-chart">
      <div className="legend-row" style={{ justifyContent: 'space-between' }}>
        <span style={{ color: 'var(--text)', fontWeight: 600 }}>
          {title} ({unit})
        </span>
        {secondKey && (
          <span>
            <span className="sw" style={{ background: 'var(--text)' }} /> mean &nbsp;
            <span className="sw hollow" /> p95
          </span>
        )}
        {whisk && <span className="muted">whisker = p05–p95 over draws</span>}
      </div>
      <div className={`chart-box${tall ? ' grid-cell' : ''}`}>
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -18 }} barCategoryGap="22%" barGap={2}>
            <CartesianGrid stroke="#1c2733" vertical={false} />
            <XAxis dataKey="short" tick={tickStyle} stroke="#1c2733" interval={0} />
            <YAxis domain={[0, domainMax ?? 'auto']} tick={tickStyle} stroke="#1c2733" />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.04)' }}
              content={({ active, payload }) => {
                // series are looked up by dataKey: the stacked label-carrier bar ("pad") must not show up
                const pv = payload?.find((p) => p.dataKey === 'v');
                const p2 = payload?.find((p) => p.dataKey === 'v2');
                const row = pv?.payload as { name: string; lo?: number; hi?: number } | undefined;
                return active && pv && row ? (
                  <div className="studio-tip">
                    <div>{row.name}</div>
                    <div>
                      {secondKey ? 'mean ' : ''}
                      {(pv.value as number).toFixed(2)} {unit}
                    </div>
                    {secondKey && p2 && (
                      <div>
                        p95 {(p2.value as number).toFixed(2)} {unit}
                      </div>
                    )}
                    {whisk && typeof row.lo === 'number' && typeof row.hi === 'number' && (
                      <div>
                        p05–p95 {row.lo.toFixed(1)}–{row.hi.toFixed(1)} {unit}
                      </div>
                    )}
                  </div>
                ) : null;
              }}
            />
            <Bar dataKey="v" stackId="v" isAnimationActive={false} radius={[3, 3, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.color} />
              ))}
              {whisk && <ErrorBar dataKey="err" width={6} strokeWidth={1.5} stroke="#d6e2f0" direction="y" />}
            </Bar>
            {/* Invisible bar stacked on the mean up to the p95 whisker tip: it carries the value label, so the label
                sits ABOVE the whisker instead of being struck through by it. Zero-height when there is no whisker. */}
            <Bar dataKey="pad" stackId="v" isAnimationActive={false} fill="transparent" stroke="none" legendType="none" tooltipType="none">
              <LabelList dataKey="v" position="top" formatter={fmtLabel} style={{ fill: '#d6e2f0', fontSize: 12, fontFamily: 'IBM Plex Mono, monospace' }} />
            </Bar>
            {secondKey && (
              <Bar dataKey="v2" isAnimationActive={false} radius={[3, 3, 0, 0]} fillOpacity={0} strokeWidth={2}>
                {data.map((d, i) => (
                  <Cell key={i} stroke={d.color} />
                ))}
                <LabelList dataKey="v2label" position="top" style={{ fill: '#8d9db3', fontSize: 12, fontFamily: 'IBM Plex Mono, monospace' }} />
              </Bar>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function SourceTag({ mock }: { mock: boolean }) {
  return mock ? (
    <span className="tag schematic" title="Browser-side schematic model: the backend Monte Carlo (DE440s dynamics, UKF custody, backend tasker) was not reachable. Figures are for layout and relative comparison only.">
      SCHEMATIC MODEL
    </span>
  ) : (
    <span className="tag live-src" title="Backend Monte Carlo: /api/architecture/evaluate (DE440s dynamics, UKF custody, backend tasker).">
      LIVE · BACKEND MONTE CARLO
    </span>
  );
}

export function ArchitecturePage() {
  const [archs, setArchs] = useState<ArchitectureDef[]>(() => defaultArchitectures());
  const [selId, setSelId] = useState<string>(() => archs[archs.length - 1]?.id ?? '');
  const [nMc, setNMc] = useState(8);
  const [horizonDays, setHorizonDays] = useState(3);
  const [targetRadius, setTargetRadius] = useState(1.5);
  const [albedo, setAlbedo] = useState(0.2);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [run, setRun] = useState<RunState | null>(null);
  const [view, setView] = useState<'map' | 'results'>('map');
  const [presetBusy, setPresetBusy] = useState(false);
  const [presetNote, setPresetNote] = useState<string | null>(null);

  /** Replace the saved architectures with the backend's reference presets (GET /api/architecture/presets). */
  async function loadPresets() {
    setPresetBusy(true);
    setPresetNote(null);
    try {
      const p = await api.architecturePresets();
      const toOrbit: Record<string, CandidateOrbit> = { geo: 'GEO', l1_halo: 'L1_halo', l2_halo_S: 'L2_halo', dro: 'DRO', resonant_3_1: 'resonant_3_1' };
      const skipped: string[] = [];
      const list: ArchitectureDef[] = p.presets.slice(0, MAX_ARCH).map((a, i) => ({
        id: nextId('A'),
        name: `${String.fromCharCode(65 + i)} · ${a.name}`,
        ground_network: a.ground,
        slot: i % ARCH_COLORS.length,
        sensors: a.sensors.flatMap((s) => {
          const orbit = toOrbit[s.platform];
          if (!orbit) {
            skipped.push(`${a.name}: ${s.platform}`);
            return [];
          }
          const base = makeSensor(orbit, s.phase ?? 0);
          return [{ ...base, aperture_m: s.aperture_m ?? base.aperture_m, limiting_mag: s.limiting_mag ?? base.limiting_mag, fov_deg: s.fov_deg, slew_rate_dps: s.slew_rate_deg_s, ...(typeof s.lon_deg === 'number' ? { lon_deg: s.lon_deg } : {}) }];
        }),
      }));
      setArchs(list);
      setSelId(list[0]?.id ?? '');
      setView('map');
      setPresetNote(`${list.length} backend presets loaded${p.presets.length > MAX_ARCH ? ` (first ${MAX_ARCH} of ${p.presets.length})` : ''}${skipped.length ? `; skipped sensors on platforms the map cannot draw: ${skipped.join(', ')}` : ''}`);
    } catch (e) {
      setPresetNote(`presets unavailable: ${e instanceof Error ? e.message : String(e)}`);
    } finally {
      setPresetBusy(false);
    }
  }

  const sel = archs.find((a) => a.id === selId) ?? archs[0] ?? null;
  const colorOf = (id: string) => ARCH_COLORS[(archs.find((a) => a.id === id)?.slot ?? 0) % ARCH_COLORS.length];

  const updateArch = (id: string, f: (a: ArchitectureDef) => ArchitectureDef) => setArchs((list) => list.map((a) => (a.id === id ? f(a) : a)));

  function addSensor(orbit: CandidateOrbit) {
    if (!sel) return;
    const n = sel.sensors.filter((s) => s.orbit === orbit).length;
    updateArch(sel.id, (a) => ({ ...a, sensors: [...a.sensors, makeSensor(orbit, (n * 0.5) % 1)] }));
  }
  function updateSensor(sid: string, patch: Partial<SensorSpec>) {
    if (!sel) return;
    updateArch(sel.id, (a) => ({ ...a, sensors: a.sensors.map((s) => (s.id === sid ? { ...s, ...patch } : s)) }));
  }
  function removeSensor(sid: string) {
    if (!sel) return;
    updateArch(sel.id, (a) => ({ ...a, sensors: a.sensors.filter((s) => s.id !== sid) }));
  }
  function addArch() {
    if (archs.length >= MAX_ARCH) return;
    const a: ArchitectureDef = { id: nextId('A'), name: `${freeLetter(archs)} · New architecture`, ground_network: true, sensors: [], slot: freeSlot(archs) };
    setArchs([...archs, a]);
    setSelId(a.id);
    setView('map');
  }
  function dupArch(id: string) {
    if (archs.length >= MAX_ARCH) return;
    const src = archs.find((a) => a.id === id);
    if (!src) return;
    const base = src.name.includes('·') ? src.name.slice(src.name.indexOf('·') + 1).trim() : src.name;
    const a: ArchitectureDef = { ...src, id: nextId('A'), name: `${freeLetter(archs)} · ${base} (copy)`, sensors: src.sensors.map((s) => ({ ...s, id: nextId('S') })), slot: freeSlot(archs) };
    setArchs([...archs, a]);
    setSelId(a.id);
  }
  function delArch(id: string) {
    const n = archs.filter((a) => a.id !== id);
    setArchs(n);
    if (selId === id) setSelId(n[0]?.id ?? '');
  }

  async function runMc() {
    setBusy(true);
    setErr(null);
    const req: ArchitectureEvaluateRequest = {
      architectures: archs.map((a) => ({
        name: a.name,
        ground_network: a.ground_network,
        sensors: a.sensors.map(({ orbit, aperture_m, limiting_mag, fov_deg, slew_rate_dps, phase, lon_deg }) => ({ orbit, aperture_m, limiting_mag, fov_deg, slew_rate_dps, phase, ...(typeof lon_deg === 'number' ? { lon_deg } : {}) })),
      })),
      n_mc: nMc,
      horizon_days: horizonDays,
      seed: 20260301,
      target_radius_m: targetRadius,
      target_albedo: albedo,
    };
    try {
      await new Promise((r) => setTimeout(r, 20));
      const { data, mock, elapsed_ms } = await studioApi.architectureEvaluate(req);
      setRun({ req, res: data, colors: archs.map((a) => colorOf(a.id)), mock: mock || !!data.mock, elapsed_ms, at: new Date().toISOString() });
      setView('results');
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function exportJson() {
    if (!run) return;
    const blob = new Blob([JSON.stringify({ generated_at: run.at, source: run.mock ? 'browser schematic model (relative comparison only)' : 'SELENE backend Monte Carlo (/api/architecture/evaluate)', method: run.res.method, request: run.req, response: run.res }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `selene-architecture-study-${run.at.slice(0, 19).replace(/[:T]/g, '-')}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Deep link: /architecture?autorun=1 runs the saved architectures on mount (ref guard: StrictMode
  // double-invokes effects in dev).
  const didAutorun = useRef(false);
  useEffect(() => {
    if (didAutorun.current) return;
    didAutorun.current = true;
    if (new URLSearchParams(window.location.search).get('autorun') === '1') void runMc();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const summary = useMemo(() => {
    if (!run) return null;
    const s = run.res.scores;
    if (!s.length) return null;
    const bc = bestIndex(s, 'coverage_pct', true);
    const bu = bestIndex(s, 'custody_pct', true);
    const bl = bestIndex(s, 'detect_latency_h_mean', false);
    const br = bestIndex(s, 'revisit_h', false);
    const worstCust = bestIndex(s, 'custody_pct', false);
    // Overall best = most "best" cells (ties → custody).
    const wins = s.map((_, i) => [bc, bu, bl, br].filter((x) => x === i).length);
    let overall = 0;
    wins.forEach((w, i) => {
      if (w > wins[overall] || (w === wins[overall] && s[i].custody_pct > s[overall].custody_pct)) overall = i;
    });
    return { bc, bu, bl, br, worstCust, overall, s };
  }, [run]);

  const scores = run?.res.scores ?? [];
  const latencyMax = niceCeil(Math.max(1, ...scores.map((x) => Math.max(x.detect_latency_h_mean, x.detect_latency_h_p95))) * 1.15);
  const liveNMc = scores[0]?.n_mc;

  return (
    <div className="studio">
      <TopBar controls={false} />
      <div className="studio-body arch">
        {/* ---------------- left ---------------- */}
        <aside className="studio-col left">
          <h2>Architecture Trade Studio</h2>
          <p className="lede">Place hypothetical space-based sensors on candidate orbits, save up to {MAX_ARCH} architectures, and score them by Monte Carlo.</p>

          <h3>Architectures ({archs.length}/{MAX_ARCH})</h3>
          <ul className="arch-list">
            {archs.map((a) => (
              <li key={a.id} className={a.id === sel?.id ? 'active' : ''} onClick={() => setSelId(a.id)}>
                <span className="dot" style={{ background: colorOf(a.id) }} aria-hidden />
                <span>
                  <input type="text" value={a.name} aria-label="Architecture name" onChange={(e) => updateArch(a.id, (x) => ({ ...x, name: e.target.value }))} onClick={(e) => e.stopPropagation()} />
                  <div className="meta">
                    {a.sensors.length} space sensor{a.sensors.length === 1 ? '' : 's'}
                    {a.ground_network ? ' + ground network' : ''}
                  </div>
                </span>
                <span className="acts">
                  <button className="sm" title="Duplicate" aria-label="Duplicate architecture" disabled={archs.length >= MAX_ARCH} onClick={(e) => (e.stopPropagation(), dupArch(a.id))}>
                    <Icon name="copy" size={13} style={{ marginRight: 0, verticalAlign: '-2px' }} />
                  </button>
                  <button className="sm danger" title="Delete" aria-label="Delete architecture" onClick={(e) => (e.stopPropagation(), delArch(a.id))}>
                    <Icon name="close" size={13} style={{ marginRight: 0, verticalAlign: '-2px' }} />
                  </button>
                </span>
              </li>
            ))}
          </ul>
          <div className="btn-row" style={{ marginTop: 4 }}>
            <button onClick={addArch} disabled={archs.length >= MAX_ARCH}>
              + New architecture
            </button>
            <button onClick={() => void loadPresets()} disabled={presetBusy} title="Replace the list with the backend's reference architectures (GET /api/architecture/presets)">
              {presetBusy ? 'Loading…' : 'Load backend presets'}
            </button>
          </div>
          {presetNote && <p className="hint">{presetNote}</p>}
          {sel && (
            <label className="check">
              <input type="checkbox" checked={sel.ground_network} onChange={(e) => updateArch(sel.id, (x) => ({ ...x, ground_network: e.target.checked }))} />
              Include the notional ground optical network in <b>{sel.name}</b>
            </label>
          )}

          <h3>Candidate orbits {sel ? `· editing ${sel.name.split('·')[0].trim()}` : ''}</h3>
          {CANDIDATE_ORBITS.map((c) => {
            const mine = sel?.sensors.filter((s) => s.orbit === c.id) ?? [];
            return (
              <div className="orbit-card" key={c.id}>
                <div className="title">
                  <b>{c.label}</b>
                  <button className="sm" onClick={() => addSensor(c.id)} disabled={!sel}>
                    + Add sensor
                  </button>
                </div>
                <p className="blurb">{c.blurb}</p>
                {mine.map((s) => {
                  const idx = sel!.sensors.findIndex((x) => x.id === s.id) + 1;
                  return (
                    <div className="sensor-row" key={s.id}>
                      <div className="head">
                        <span>S{idx} on {c.label}</span>
                        <button className="sm danger" onClick={() => removeSensor(s.id)} title="Remove sensor">
                          remove
                        </button>
                      </div>
                      <div className="ctl">
                        <label htmlFor={`${s.id}-ap`}>Aperture m</label>
                        <input id={`${s.id}-ap`} type="number" min={0.05} max={4} step={0.05} value={s.aperture_m} onChange={(e) => updateSensor(s.id, { aperture_m: Number(e.target.value) })} />
                        <label htmlFor={`${s.id}-ml`}>Limit. mag</label>
                        <input id={`${s.id}-ml`} type="number" min={10} max={24} step={0.1} value={s.limiting_mag} onChange={(e) => updateSensor(s.id, { limiting_mag: Number(e.target.value) })} />
                        <label htmlFor={`${s.id}-fov`}>FOV deg</label>
                        <input id={`${s.id}-fov`} type="number" min={0.1} max={30} step={0.1} value={s.fov_deg} onChange={(e) => updateSensor(s.id, { fov_deg: Number(e.target.value) })} />
                        <label htmlFor={`${s.id}-sl`}>Slew deg/s</label>
                        <input id={`${s.id}-sl`} type="number" min={0.01} max={10} step={0.1} value={s.slew_rate_dps} onChange={(e) => updateSensor(s.id, { slew_rate_dps: Number(e.target.value) })} />
                        <label htmlFor={`${s.id}-ph`}>Phase</label>
                        <input id={`${s.id}-ph`} type="number" min={0} max={0.99} step={0.05} value={s.phase} onChange={(e) => updateSensor(s.id, { phase: Math.min(0.99, Math.max(0, Number(e.target.value))) })} />
                      </div>
                    </div>
                  );
                })}
              </div>
            );
          })}
          <p className="hint">
            {run && !run.mock
              ? 'Backend evaluator: aperture sets the limiting magnitude (m_lim = 18.5 + 5 log₁₀(D / 0.5 m)) when none is given; FOV, slew and phase drive the tasking model.'
              : 'Schematic model: only limiting magnitude affects the scores; aperture, FOV and slew are recorded in the definition and the exported JSON.'}
          </p>
        </aside>

        {/* ---------------- centre ---------------- */}
        <section className="studio-col center">
          <div className="plot-head">
            <span className="seg seg-view" role="tablist" aria-label="Centre view">
              <button role="tab" aria-selected={view === 'map'} className={view === 'map' ? 'active' : ''} onClick={() => setView('map')}>
                Map
              </button>
              <button role="tab" aria-selected={view === 'results'} className={view === 'results' ? 'active' : ''} onClick={() => setView('results')} disabled={!run} title={run ? 'Metric charts for the last run' : 'Run Monte Carlo first'}>
                Results
              </button>
            </span>
            {view === 'map' ? (
              <>
                <span>
                  SELECTED <b>{sel?.name ?? '—'}</b>
                </span>
                <span>
                  SENSORS <b>{sel ? sel.sensors.length + (sel.ground_network ? (run && !run.mock ? 9 : 3) : 0) : 0}</b>
                </span>
                <span className="spacer" />
                <span className="legend-row">
                  <span>
                    <span className="sw" style={{ background: 'var(--accent)' }} />
                    sensor orbits
                  </span>
                  <span>
                    <span className="sw" style={{ background: 'var(--sim)' }} />
                    SIMULATED targets (notional actor)
                  </span>
                </span>
              </>
            ) : (
              <>
                <span className="spacer" />
                {run && (
                  <span className="legend-row">
                    {scores.map((s, i) => (
                      <span key={i}>
                        <span className="sw" style={{ background: run.colors[i] }} />
                        {s.name}
                      </span>
                    ))}
                  </span>
                )}
              </>
            )}
          </div>
          {view === 'map' || !run ? (
            <SystemPlot arch={sel} />
          ) : (
            <div className="chart-grid">
              {METRICS.slice(0, 3).map((m) => (
                <MetricChart key={m.key} title={m.label} unit={m.unit} scores={scores} colors={run.colors} dataKey={m.key} loKey={`${m.key}_p05` as keyof ArchitectureScore} hiKey={`${m.key}_p95` as keyof ArchitectureScore} domainMax={m.unit === '%' ? 100 : undefined} tall />
              ))}
              <MetricChart title="Maneuver-detection latency" unit="h" scores={scores} colors={run.colors} dataKey="detect_latency_h_mean" secondKey="detect_latency_h_p95" domainMax={latencyMax} tall />
            </div>
          )}
          {busy && <div className="progress-thin" role="progressbar" aria-label="Running Monte Carlo" />}
          {busy && <div className="center-busy quiet">running Monte Carlo…</div>}
        </section>

        {/* ---------------- right ---------------- */}
        <aside className="studio-col right">
          <button className="primary wide run-btn" onClick={runMc} disabled={busy || archs.length === 0}>
            {busy ? (
              'Running…'
            ) : (
              <>
                <Icon name="play" size={15} />
                Run Monte Carlo · {archs.length} architecture{archs.length === 1 ? '' : 's'}
              </>
            )}
          </button>
          <div className="ctl params" style={{ gridTemplateColumns: 'auto 1fr auto 1fr' }}>
            <label htmlFor="mc-n" title="Monte Carlo draws per architecture (backend cap 16)">
              n_mc
            </label>
            <input id="mc-n" type="number" min={1} max={200} step={1} value={nMc} onChange={(e) => setNMc(Math.max(1, Number(e.target.value) || 1))} />
            <label htmlFor="mc-h" title="Evaluation horizon, days (backend cap 7)">
              days
            </label>
            <input id="mc-h" type="number" min={1} max={60} step={1} value={horizonDays} onChange={(e) => setHorizonDays(Math.max(1, Number(e.target.value) || 1))} />
            <label htmlFor="mc-r" title="Reference target radius, m">
              ρ m
            </label>
            <input id="mc-r" type="number" min={0.1} max={10} step={0.1} value={targetRadius} onChange={(e) => setTargetRadius(Math.max(0.05, Number(e.target.value) || 0.05))} />
            <label htmlFor="mc-a">albedo</label>
            <input id="mc-a" type="number" min={0.02} max={1} step={0.02} value={albedo} onChange={(e) => setAlbedo(Math.min(1, Math.max(0.01, Number(e.target.value) || 0.01)))} />
          </div>
          {err && <div className="status-line err">ERROR: {err}</div>}

          <h3>
            Results {run && <SourceTag mock={run.mock} />}
          </h3>
          {!run && <p className="hint">Run Monte Carlo to compare the saved architectures side by side. The backend draws notional objects from the SIMULATED population, injects unannounced burns and scores coverage, custody, revisit and detection latency.</p>}
          {run && (
            <>
              <div className="status-line" style={{ marginTop: 0 }}>
                n_mc {liveNMc ?? run.req.n_mc}
                {liveNMc !== undefined && liveNMc !== run.req.n_mc ? ` (requested ${run.req.n_mc}, backend cap)` : ''} · horizon {run.req.horizon_days}&nbsp;d · target ρ&nbsp;{run.req.target_radius_m}&nbsp;m, a&nbsp;{run.req.target_albedo} · <span className="nowrap">{(run.elapsed_ms / 1000).toFixed(1)}&nbsp;s</span>
              </div>
              <div className="table-scroll">
                <table className="results-table">
                  <thead>
                    <tr>
                      <th title="Letter = architecture (see legend below)">Arch</th>
                      <th title="Coverage % of the xy slice">Cov %</th>
                      <th title="Custody % of object-hours">Cust %</th>
                      <th title="Mean revisit, hours">Rev h</th>
                      <th title="Maneuver-detection latency, mean hours">Lat h</th>
                      <th title="Maneuver-detection latency, 95th percentile hours">p95 h</th>
                    </tr>
                  </thead>
                  <tbody>
                    {scores.map((s, i) => (
                      <tr key={i} className={summary && summary.overall === i ? 'best-row' : ''} title={summary && summary.overall === i ? 'Best overall (most best-in-class metrics)' : undefined}>
                        <td title={`${s.name}${s.n_sensors !== undefined ? ` — ${s.n_sensors} space sensors` : ''}`}>
                          <span className="sw" style={{ background: run.colors[i] }} />
                          {shortName(s.name)}
                        </td>
                        <td className={bestIndex(scores, 'coverage_pct', true) === i ? 'best' : ''}>{s.coverage_pct.toFixed(1)}</td>
                        <td className={bestIndex(scores, 'custody_pct', true) === i ? 'best' : ''}>{s.custody_pct.toFixed(1)}</td>
                        <td className={bestIndex(scores, 'revisit_h', false) === i ? 'best' : ''}>{s.revisit_h.toFixed(1)}</td>
                        <td className={bestIndex(scores, 'detect_latency_h_mean', false) === i ? 'best' : ''}>{s.detect_latency_h_mean.toFixed(1)}</td>
                        <td>{s.detect_latency_h_p95.toFixed(1)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="legend-row" style={{ margin: '8px 0 4px' }}>
                {scores.map((s, i) => (
                  <span key={i}>
                    <span className="sw" style={{ background: run.colors[i] }} />
                    {s.name}
                  </span>
                ))}
              </div>
              {run.res.method && <p className="hint method-line">{run.res.method}</p>}
              {run.res.caps_applied && run.res.caps_applied.length > 0 && <p className="hint">Backend caps: {run.res.caps_applied.join('; ')}</p>}
              {scores.some((s) => (s.undetected_pct ?? 0) > 0 || (s.never_observed_pct ?? 0) > 0) && (
                <p className="hint">
                  Censored (latency = remaining horizon):{' '}
                  {scores.map((s) => `${shortName(s.name)} ${(s.undetected_pct ?? 0).toFixed(0)}% burns undetected${s.never_observed_pct !== undefined ? ` / ${s.never_observed_pct.toFixed(0)}% objects never revisited` : ''}`).join(' · ')}
                </p>
              )}

              {summary && (
                <>
                  <h3>Summary</h3>
                  <p className="summary">
                    Best overall: <b>{summary.s[summary.overall].name}</b>. Best coverage: <b>{summary.s[summary.bc].name}</b> ({summary.s[summary.bc].coverage_pct.toFixed(1)}%). Best custody: <b>{summary.s[summary.bu].name}</b> ({summary.s[summary.bu].custody_pct.toFixed(1)}% of
                    object-hours, vs {summary.s[summary.worstCust].custody_pct.toFixed(1)}% for {summary.s[summary.worstCust].name}). Fastest maneuver detection: <b>{summary.s[summary.bl].name}</b>, mean {summary.s[summary.bl].detect_latency_h_mean.toFixed(1)} h / p95{' '}
                    {summary.s[summary.bl].detect_latency_h_p95.toFixed(1)} h.
                    {run.mock ? ' Schematic-model figures: relative comparison only.' : ''}
                  </p>
                </>
              )}
              <div className="btn-row">
                <button onClick={exportJson}>
                  <Icon name="download" />
                  Export JSON
                </button>
              </div>
            </>
          )}

          <details className="explainer-box">
            <summary>How the metrics are computed{run?.res.method_notes?.length ? ' · backend method notes' : ''}</summary>
            {run?.res.method_notes && run.res.method_notes.length > 0 && (
              <ul className="method-notes">
                {run.res.method_notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            )}
            <dl className="explainer">
              <dt>Monte Carlo</dt>
              <dd>
                {run && !run.mock
                  ? 'Each draw samples a start epoch inside the cached DE440s span and injects unannounced burns (1–20 m/s, Poisson rate per object) into the SIMULATED population; each architecture is scored with the real visibility model (photometry, Sun/Moon/Earth exclusion, shadow, site night/elevation), a UKF custody filter and the information-gain tasker.'
                  : 'n_mc trials; each draws a SIMULATED object (notional actor) from the target population — DRO, 9:2 NRHO, L1 halo, L2 southern halo — at a random phase, plus a random unannounced maneuver epoch. Visibility is evaluated hourly over the horizon with photometric detectability, Sun/Moon/Earth exclusion, shadow, and ground-site night/elevation constraints.'}
              </dd>
              <dt>Coverage %</dt>
              <dd>Fraction of (cell, epoch) samples of the rotating-frame xy slice where ≥ 1 sensor could detect the reference object.</dd>
              <dt>Custody %</dt>
              <dd>{run && !run.mock ? 'Fraction of object-hours in which the filter position uncertainty (√tr P) stays under the custody threshold (100 km).' : 'Fraction of object-hours in which time since last observation is within a 12 h custody window (schematic surrogate for the filter-based definition).'}</dd>
              <dt>Mean revisit (h)</dt>
              <dd>Mean gap between consecutive observation opportunities per object; never-observed objects contribute the full horizon.</dd>
              <dt>Detection latency (h)</dt>
              <dd>Time from the maneuver epoch to the detection (NIS gate or miss on re-acquisition); undetected burns are censored at the horizon. Mean and 95th percentile.</dd>
            </dl>
          </details>
        </aside>
      </div>
    </div>
  );
}
