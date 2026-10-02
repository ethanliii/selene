/**
 * Architecture Trade Studio.
 * Left: candidate orbits (add sensors, edit specs) and up to 4 saved architectures.
 * Centre: 2D rotating-frame plot of the selected architecture against the notional target population,
 *         Run Monte Carlo (POST /api/architecture/evaluate, browser mock fallback with MOCK badge).
 * Right: side-by-side table, grouped bar charts, summary, Export JSON, metric explainer.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
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

const tickStyle = { fill: '#8d9db3', fontSize: 11 };

function MetricChart({ title, unit, scores, colors, dataKey, secondKey, domainMax }: { title: string; unit: string; scores: ArchitectureScore[]; colors: string[]; dataKey: keyof ArchitectureScore; secondKey?: keyof ArchitectureScore; domainMax?: number }) {
  const fmtLabel = (v: number) => v.toFixed(v >= 100 ? 0 : 1);
  const span = domainMax ?? Math.max(1, ...scores.map((s) => s[dataKey] as number));
  const data = scores.map((s, i) => {
    const v = s[dataKey] as number;
    const v2 = secondKey ? (s[secondKey] as number) : undefined;
    // the p95 direct label is dropped when it would sit on top of the mean label (< 8 % of the axis
    // apart); the value stays in the tooltip and the table
    const v2label = v2 === undefined ? '' : Math.abs(v2 - v) / span < 0.08 ? '' : fmtLabel(v2);
    return { name: s.name, short: shortName(s.name), v, v2, v2label, color: colors[i] };
  });
  return (
    <div>
      <div className="legend-row" style={{ justifyContent: 'space-between' }}>
        <span style={{ color: 'var(--text)' }}>
          {title} ({unit})
        </span>
        {secondKey && (
          <span>
            <span className="sw" style={{ background: 'var(--text)' }} /> mean &nbsp;
            <span className="sw hollow" /> p95
          </span>
        )}
      </div>
      <div className="chart-box">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} margin={{ top: 16, right: 8, bottom: 0, left: -22 }} barCategoryGap="22%" barGap={2}>
            <CartesianGrid stroke="#1c2733" vertical={false} />
            <XAxis dataKey="short" tick={tickStyle} stroke="#1c2733" interval={0} />
            <YAxis domain={[0, domainMax ?? 'auto']} tick={tickStyle} stroke="#1c2733" />
            <Tooltip
              cursor={{ fill: 'rgba(255,255,255,0.04)' }}
              content={({ active, payload }) =>
                active && payload && payload.length ? (
                  <div className="studio-tip">
                    <div>{(payload[0].payload as { name: string }).name}</div>
                    <div>
                      {secondKey ? 'mean ' : ''}
                      {(payload[0].value as number).toFixed(2)} {unit}
                    </div>
                    {secondKey && payload[1] && (
                      <div>
                        p95 {(payload[1].value as number).toFixed(2)} {unit}
                      </div>
                    )}
                  </div>
                ) : null
              }
            />
            <Bar dataKey="v" isAnimationActive={false} radius={[3, 3, 0, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.color} />
              ))}
              <LabelList dataKey="v" position="top" formatter={fmtLabel} style={{ fill: '#d6e2f0', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace' }} />
            </Bar>
            {secondKey && (
              <Bar dataKey="v2" isAnimationActive={false} radius={[3, 3, 0, 0]} fillOpacity={0} strokeWidth={2}>
                {data.map((d, i) => (
                  <Cell key={i} stroke={d.color} />
                ))}
                <LabelList dataKey="v2label" position="top" style={{ fill: '#8d9db3', fontSize: 11, fontFamily: 'IBM Plex Mono, monospace' }} />
              </Bar>
            )}
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function ArchitecturePage() {
  const [archs, setArchs] = useState<ArchitectureDef[]>(() => defaultArchitectures());
  const [selId, setSelId] = useState<string>(() => archs[archs.length - 1]?.id ?? '');
  const [nMc, setNMc] = useState(100);
  const [horizonDays, setHorizonDays] = useState(7);
  const [targetRadius, setTargetRadius] = useState(1.5);
  const [albedo, setAlbedo] = useState(0.2);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [run, setRun] = useState<RunState | null>(null);

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
        sensors: a.sensors.map(({ orbit, aperture_m, limiting_mag, fov_deg, slew_rate_dps, phase }) => ({ orbit, aperture_m, limiting_mag, fov_deg, slew_rate_dps, phase })),
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
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function exportJson() {
    if (!run) return;
    const blob = new Blob([JSON.stringify({ generated_at: run.at, source: run.mock ? 'MOCK browser-side Monte Carlo' : 'SELENE backend /api/architecture/evaluate', request: run.req, response: run.res }, null, 2)], { type: 'application/json' });
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
    const worstCust = bestIndex(s, 'custody_pct', false);
    return { bc, bu, bl, worstCust, s };
  }, [run]);

  const scores = run?.res.scores ?? [];
  const latencyMax = niceCeil(Math.max(1, ...scores.map((x) => Math.max(x.detect_latency_h_mean, x.detect_latency_h_p95))) * 1.15);

  return (
    <div className="studio">
      <TopBar controls={false} />
      <div className="studio-body">
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
                  <button className="sm" title="Duplicate" disabled={archs.length >= MAX_ARCH} onClick={(e) => (e.stopPropagation(), dupArch(a.id))}>
                    ⧉
                  </button>
                  <button className="sm danger" title="Delete" onClick={(e) => (e.stopPropagation(), delArch(a.id))}>
                    ✕
                  </button>
                </span>
              </li>
            ))}
          </ul>
          <div className="btn-row" style={{ marginTop: 4 }}>
            <button onClick={addArch} disabled={archs.length >= MAX_ARCH}>
              + New architecture
            </button>
          </div>
          {sel && (
            <label className="check">
              <input type="checkbox" checked={sel.ground_network} onChange={(e) => updateArch(sel.id, (x) => ({ ...x, ground_network: e.target.checked }))} />
              Include ground network (3 notional 1-m sites) in <b>{sel.name}</b>
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
            Only limiting magnitude affects the scores today (mock and the planned backend evaluator both detect on m ≤ m_lim). Aperture, FOV and slew are recorded in the architecture definition and the exported JSON for the tasking scheduler; they do not change
            these results.
          </p>
        </aside>

        {/* ---------------- centre ---------------- */}
        <section className="studio-col center">
          <div className="plot-head">
            <span>
              SELECTED <b>{sel?.name ?? '—'}</b>
            </span>
            <span>
              SENSORS <b>{sel ? sel.sensors.length + (sel.ground_network ? 3 : 0) : 0}</b>
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
          </div>
          <SystemPlot arch={sel} />
          <div className="plot-head">
            <span className="ctl" style={{ gridTemplateColumns: 'auto 70px auto 60px auto 60px auto 60px', padding: 0 }}>
              <label htmlFor="mc-n">n_mc</label>
              <input id="mc-n" type="number" min={10} max={2000} step={10} value={nMc} onChange={(e) => setNMc(Math.max(4, Number(e.target.value) || 4))} />
              <label htmlFor="mc-h">days</label>
              <input id="mc-h" type="number" min={1} max={60} step={1} value={horizonDays} onChange={(e) => setHorizonDays(Math.max(1, Number(e.target.value) || 1))} />
              <label htmlFor="mc-r">ρ m</label>
              <input id="mc-r" type="number" min={0.1} max={10} step={0.1} value={targetRadius} onChange={(e) => setTargetRadius(Math.max(0.05, Number(e.target.value) || 0.05))} />
              <label htmlFor="mc-a">albedo</label>
              <input id="mc-a" type="number" min={0.02} max={1} step={0.02} value={albedo} onChange={(e) => setAlbedo(Math.min(1, Math.max(0.01, Number(e.target.value) || 0.01)))} />
            </span>
            <span className="spacer" />
            <button className="primary" onClick={runMc} disabled={busy || archs.length === 0}>
              {busy ? 'Running…' : `▶ Run Monte Carlo (${archs.length} arch.)`}
            </button>
          </div>
          {err && <div className="status-line err">ERROR: {err}</div>}
          {busy && <div className="center-busy">RUNNING MONTE CARLO…</div>}
        </section>

        {/* ---------------- right ---------------- */}
        <aside className="studio-col right">
          <h3>
            Results {run && (run.mock ? <span className="badge-mock">MOCK</span> : <span className="badge-live">LIVE</span>)}
          </h3>
          {!run && <p className="hint">Run Monte Carlo to compare the saved architectures side by side.</p>}
          {run && (
            <>
              <div className="status-line" style={{ marginTop: 0 }}>
                n_mc {run.res.scores[0]?.n_mc ?? run.req.n_mc} · horizon {run.req.horizon_days} d · target ρ {run.req.target_radius_m} m, a {run.req.target_albedo} · {run.elapsed_ms.toFixed(0)} ms
                {run.mock && <div>{run.res.method}</div>}
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
                      <tr key={i}>
                        <td title={`${s.name}${s.n_sensors !== undefined ? ` — ${s.n_sensors} sensors` : ''}`}>
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
              {scores.some((s) => (s.undetected_pct ?? 0) > 0 || (s.never_observed_pct ?? 0) > 0) && (
                <p className="hint">
                  Censored samples (latency = remaining horizon):{' '}
                  {scores.map((s) => `${shortName(s.name)} ${(s.undetected_pct ?? 0).toFixed(0)}% undetected / ${(s.never_observed_pct ?? 0).toFixed(0)}% never observed`).join(' · ')}
                </p>
              )}

              <div className="legend-row" style={{ margin: '8px 0 4px' }}>
                {scores.map((s, i) => (
                  <span key={i}>
                    <span className="sw" style={{ background: run.colors[i] }} />
                    {s.name}
                  </span>
                ))}
              </div>
              {METRICS.slice(0, 3).map((m) => (
                <MetricChart key={m.key} title={m.label} unit={m.unit} scores={scores} colors={run.colors} dataKey={m.key} domainMax={m.unit === '%' ? 100 : undefined} />
              ))}
              <MetricChart title="Maneuver-detection latency" unit="h" scores={scores} colors={run.colors} dataKey="detect_latency_h_mean" secondKey="detect_latency_h_p95" domainMax={latencyMax} />

              {summary && (
                <>
                  <h3>Summary</h3>
                  <p className="summary">
                    Best coverage: <b>{summary.s[summary.bc].name}</b> ({summary.s[summary.bc].coverage_pct.toFixed(1)}% of the xy slice). Best custody: <b>{summary.s[summary.bu].name}</b> ({summary.s[summary.bu].custody_pct.toFixed(1)}%
                    of object-hours, vs {summary.s[summary.worstCust].custody_pct.toFixed(1)}% for {summary.s[summary.worstCust].name}). Fastest maneuver detection: <b>{summary.s[summary.bl].name}</b>, mean{' '}
                    {summary.s[summary.bl].detect_latency_h_mean.toFixed(1)} h / p95 {summary.s[summary.bl].detect_latency_h_p95.toFixed(1)} h.
                    {run.mock ? ' Figures are from the browser-side schematic model and are for layout and relative comparison only.' : ''}
                  </p>
                </>
              )}
              <div className="btn-row">
                <button onClick={exportJson}>⤓ Export JSON</button>
              </div>
            </>
          )}

          <h3>How the metrics are computed</h3>
          <dl className="explainer">
            <dt>Monte Carlo</dt>
            <dd>
              n_mc trials; each draws a SIMULATED object (notional actor) from the target population — DRO, 9:2 NRHO, L1 halo, L2 southern halo — at a random phase, plus a random unannounced maneuver epoch. Visibility is evaluated hourly over the horizon
              with photometric detectability (size, albedo, phase angle, range vs. limiting magnitude), Sun/Moon/Earth exclusion, shadow, and ground-site night/elevation constraints.
            </dd>
            <dt>Coverage %</dt>
            <dd>Fraction of (cell, epoch) samples of the rotating-frame xy slice where ≥ 1 sensor could detect the reference object.</dd>
            <dt>Custody %</dt>
            <dd>Fraction of object-hours in which time since last observation is within a 12 h custody window (mock surrogate; a filter-based evaluator would use the position-covariance trace instead — no such backend route exists yet).</dd>
            <dt>Mean revisit (h)</dt>
            <dd>Mean gap between consecutive observation opportunities per object; never-observed objects contribute the full horizon.</dd>
            <dt>Detection latency (h)</dt>
            <dd>Time from the maneuver epoch to the second post-maneuver observation (two fixes are needed to attribute a residual to a burn); undetected trials are censored at the horizon. Mean and 95th percentile.</dd>
          </dl>
        </aside>
      </div>
    </div>
  );
}
