/**
 * Coverage & blind-spot page.
 * Left: network preset, time window, reference target, Run (POST /api/coverage, mock fallback).
 * Centre: canvas heatmap over the rotating-frame xy slice with overlays, per-epoch / time-averaged
 * toggle and a time slider. Right: "why blind" breakdown by dominant reason and coverage % vs time.
 *
 * Per-epoch fields: the browser mock returns them directly. The live backend serves only the
 * time-averaged product, so in live mode per-epoch frames are fetched lazily (1-second window,
 * n_t = 2, ~15–25 ms each) when the user switches to "Per epoch": the displayed epoch first, then
 * the rest in the background with a small worker pool. Frames are never mixed with mock data.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api/client';
import type { CoveragePresets } from '../api/types';
import { Icon } from '../panels/icons';
import { TopBar } from '../panels/TopBar';
import { studioApi } from '../studio/api';
import { CoverageHeatmap } from '../studio/CoverageHeatmap';
import { estimateReasons } from '../studio/mockCoverage';
import { sunDirAt } from '../studio/model';
import { backendPresetLabel, NETWORK_PRESETS, REASON_CODES, REASON_LABELS, type BlindReason, type CoverageFrame, type CoverageRequest, type CoverageResponse, type NetworkPreset } from '../studio/types';
import '../studio/studio.css';

const WINDOWS: { id: string; label: string; hours: number; n_t: number }[] = [
  { id: '24h', label: '24 h', hours: 24, n_t: 25 },
  { id: '72h', label: '72 h', hours: 72, n_t: 37 },
  { id: '7d', label: '7 d', hours: 168, n_t: 57 },
  { id: 'syn', label: '29.5 d', hours: 29.530589 * 24, n_t: 60 },
];
/**
 * Mock grid: same extent as the backend's default 2-D grid (x ∈ [−1.6, 1.6], y ∈ [−1.4, 1.4] L*,
 * backend/selene/sensors/coverage.py make_grid) so that L3 (x = −1.005) and L4/L5 (y = ±0.866) are
 * inside the slice in both modes; 96 × 84 cells (1/30 L* ≈ 12 800 km per cell).
 */
const GRID = { xmin: -1.6, xmax: 1.6, ymin: -1.4, ymax: 1.4, nx: 96, ny: 84 };
const DEFAULT_T0 = '2026-03-01T00:00';
const FRAME_WORKERS = 3;

interface RunState {
  req: CoverageRequest;
  res: CoverageResponse;
  reasons: number[][];
  reasonsEstimated: boolean;
  mock: boolean;
  elapsed_ms: number;
}

/** Lazily loaded per-epoch frames for a live run. */
interface FrameStore {
  key: number;
  frames: (CoverageFrame | undefined)[];
  inflight: Set<number>;
  ctrl: AbortController;
  loaded: number;
}

function fmtUtc(iso: string): string {
  return iso.replace('T', ' ').slice(0, 16) + 'Z';
}

/** Optional deep-link defaults: ?network=ground%2Ball (or ground+all) &mode=avg&window=7d */
function initialParams() {
  const q = new URLSearchParams(window.location.search);
  // URLSearchParams decodes '+' as a space; preset ids contain '+', so map spaces back.
  const network = q.get('network')?.replace(/ /g, '+') ?? null;
  const mode = q.get('mode');
  const window_ = q.get('window');
  return {
    preset: (NETWORK_PRESETS.some((p) => p.id === network) || (network && /^[a-z0-9_]+$/.test(network)) ? network : 'ground') as NetworkPreset,
    mode: (mode === 'avg' ? 'avg' : 'epoch') as 'epoch' | 'avg',
    win: WINDOWS.some((w) => w.id === window_) ? (window_ as string) : '7d',
  };
}

let frameStoreCounter = 0;

export function CoveragePage() {
  const [init] = useState(initialParams);
  const [preset, setPreset] = useState<string>(init.preset);
  /** Preset list from GET /api/coverage/presets when the backend is up (the studio's 5 presets otherwise). */
  const [livePresets, setLivePresets] = useState<CoveragePresets | null>(null);
  useEffect(() => {
    let alive = true;
    api
      .coveragePresets()
      .then((p) => {
        if (!alive || !p || Object.keys(p).length === 0) return;
        setLivePresets(p);
        // Translate a studio preset id to its backend name so the active button matches the live list.
        setPreset((cur) => (cur in p ? cur : (NETWORK_PRESETS.find((x) => x.id === cur) ? ({ ground: 'ground_only', 'ground+geo': 'ground_plus_geo', 'ground+l2': 'ground_plus_l2_halo', 'ground+dro': 'ground_plus_dro', 'ground+all': 'full' } as Record<string, string>)[cur] ?? cur : cur)));
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, []);
  const presetList = useMemo<{ id: string; label: string; hint: string; live_hint: string }[]>(() => {
    if (!livePresets) return NETWORK_PRESETS;
    return Object.entries(livePresets).map(([id, p]) => {
      const hint = `backend ${id}: ${p.ground_ids.length ? `${p.ground_ids.length} ground sites` : 'no ground sites'}${p.space.length ? ` + ${p.space.join(', ')}` : ''}`;
      return { id, label: backendPresetLabel(id), hint, live_hint: hint };
    });
  }, [livePresets]);
  const [win, setWin] = useState(init.win);
  const [t0, setT0] = useState(DEFAULT_T0);
  const [radius, setRadius] = useState(1.0);
  const [albedo, setAlbedo] = useState(0.2);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [run, setRun] = useState<RunState | null>(null);
  const [tIdx, setTIdx] = useState(0);
  const [mode, setMode] = useState<'epoch' | 'avg'>(init.mode);
  const [families, setFamilies] = useState(true);
  const framesRef = useRef<FrameStore | null>(null);
  const [frameTick, setFrameTick] = useState(0);
  const [frameErr, setFrameErr] = useState<string | null>(null);

  async function doRun() {
    setBusy(true);
    setErr(null);
    const w = WINDOWS.find((x) => x.id === win) ?? WINDOWS[2];
    const t0ms = Date.parse(t0 + ':00Z');
    if (!Number.isFinite(t0ms)) {
      setErr('Invalid start epoch');
      setBusy(false);
      return;
    }
    const req: CoverageRequest = {
      t0: new Date(t0ms).toISOString(),
      t1: new Date(t0ms + w.hours * 3600e3).toISOString(),
      n_t: w.n_t,
      grid: GRID,
      network: preset as NetworkPreset,
      target_radius_m: radius,
      target_albedo: albedo,
    };
    try {
      // yield a frame so the busy overlay paints before the (synchronous) mock computes
      await new Promise((r) => setTimeout(r, 20));
      const { data, mock, elapsed_ms } = await studioApi.coverage(req);
      const live = data.values.length === 0 && !!data.averaged;
      const reasonsEstimated = !live && !data.reasons;
      const reasons = live ? [] : data.reasons ?? estimateReasons(req, data);
      framesRef.current?.ctrl.abort();
      framesRef.current = { key: ++frameStoreCounter, frames: new Array(data.epochs.length), inflight: new Set(), ctrl: new AbortController(), loaded: 0 };
      setFrameErr(null);
      setFrameTick(0);
      setRun({ req, res: data, reasons, reasonsEstimated, mock, elapsed_ms });
      setTIdx(0);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  // Run once on mount so the page is never empty (ref guard: StrictMode double-invokes effects).
  const didInit = useRef(false);
  useEffect(() => {
    if (didInit.current) return;
    didInit.current = true;
    void doRun();
    return () => framesRef.current?.ctrl.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /** Live run: per-epoch fields are fetched lazily. */
  const live = !!run && run.res.values.length === 0 && !!run.res.averaged;

  // ---- lazy per-epoch frames (live mode only) ----------------------------------------------
  async function fetchFrame(i: number) {
    const store = framesRef.current;
    if (!run || !store || store.frames[i] || store.inflight.has(i)) return;
    store.inflight.add(i);
    try {
      const f = await studioApi.coverageFrame(run.req, run.res.epochs[i], store.ctrl.signal);
      if (framesRef.current?.key !== store.key) return;
      store.frames[i] = f;
      store.loaded++;
      setFrameTick((t) => t + 1);
    } catch (e) {
      if (!store.ctrl.signal.aborted) setFrameErr(e instanceof Error ? e.message : String(e));
    } finally {
      store.inflight.delete(i);
    }
  }

  // Priority: the epoch on screen.
  useEffect(() => {
    if (live && mode === 'epoch') void fetchFrame(tIdx);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, live, mode, tIdx]);

  // Background: all remaining epochs with a small worker pool (idempotent; aborted on a new run).
  useEffect(() => {
    if (!live || mode !== 'epoch' || !run) return;
    const store = framesRef.current;
    if (!store) return;
    const n = run.res.epochs.length;
    const order = Array.from({ length: n }, (_, k) => (tIdx + k) % n);
    let cursor = 0;
    const worker = async () => {
      while (cursor < order.length && framesRef.current?.key === store.key && !store.ctrl.signal.aborted) {
        const i = order[cursor++];
        await fetchFrame(i);
      }
    };
    void Promise.all(Array.from({ length: FRAME_WORKERS }, worker));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [run, live, mode]);

  const frame = live && mode === 'epoch' ? framesRef.current?.frames[tIdx] : undefined;
  const framesLoaded = framesRef.current?.loaded ?? 0;
  // frameTick is read so that the memos below recompute when frames arrive.
  void frameTick;

  const derived = useMemo(() => {
    if (!run) return null;
    const { values, grid, epochs } = run.res;
    const nCells = grid.x.length * grid.y.length;
    if (run.res.averaged && values.length === 0) {
      const a = run.res.averaged;
      const perEpochPct = a.per_time_pct;
      const meanPct = (a.coverage.reduce((s, v) => s + v, 0) / Math.max(1, nCells)) * 100;
      const series = epochs.map((e, k) => ({ k, t: fmtUtc(e), pct: perEpochPct[k] ?? 0 }));
      let vmaxCount = 1;
      for (const f of framesRef.current?.frames ?? []) if (f) for (const c of f.values) if (c > vmaxCount) vmaxCount = c;
      return { perEpochPct, avg: Float32Array.from(a.coverage), avgReason: Uint8Array.from(a.reasons), vmaxCount, meanPct, series };
    }
    const nT = values.length;
    const perEpochPct = values.map((v) => (100 * v.reduce((s, c) => s + (c > 0 ? 1 : 0), 0)) / nCells);
    const avg = new Float32Array(nCells);
    for (const v of values) for (let i = 0; i < nCells; i++) if (v[i] > 0) avg[i] += 1 / nT;
    // dominant blind reason per cell over time = most frequent blind reason code (0 = covered)
    const avgReason = new Uint8Array(nCells);
    const counts = new Uint32Array(REASON_CODES.length);
    for (let i = 0; i < nCells; i++) {
      counts.fill(0);
      for (let t = 0; t < nT; t++) counts[run.reasons[t][i]]++;
      let best = 0;
      for (let c = 1; c < counts.length; c++) if (counts[c] > counts[best]) best = c;
      avgReason[i] = best;
    }
    let vmaxCount = 1;
    for (const v of values) for (const c of v) if (c > vmaxCount) vmaxCount = c;
    const meanPct = perEpochPct.reduce((s, v) => s + v, 0) / Math.max(1, nT);
    const series = epochs.map((e, k) => ({ k, t: fmtUtc(e), pct: perEpochPct[k] }));
    return { perEpochPct, avg, avgReason, vmaxCount, meanPct, series };
  }, [run, frameTick]);

  const legend = useMemo(() => {
    if (!run) return [];
    const hist = new Array<number>(REASON_CODES.length).fill(0);
    let n = 0;
    if (live) {
      if (mode === 'epoch' && frame) {
        for (const c of frame.reasons) {
          hist[c]++;
          n++;
        }
      } else {
        // Whole window: split cell-epochs into covered (mean coverage) and blind, attributing each
        // cell's blind fraction (1 − coverage) to its dominant reason over the window.
        const a = run.res.averaged!;
        a.coverage.forEach((c, i) => {
          hist[0] += c;
          hist[a.reasons[i]] += 1 - c;
          n++;
        });
      }
    } else {
      const rows = mode === 'epoch' ? [run.reasons[tIdx] ?? []] : run.reasons;
      for (const row of rows)
        for (const c of row) {
          hist[c]++;
          n++;
        }
    }
    // rows below 0.05 % would print as "0.0 %", so they are dropped (except the Covered row)
    return REASON_CODES.map((r, i) => ({ reason: r as BlindReason, pct: (100 * hist[i]) / Math.max(1, n) })).filter((x) => x.pct >= 0.05 || x.reason === 'covered');
  }, [run, live, mode, tIdx, frame]);

  const topBlind = legend.filter((r) => r.reason !== 'covered').sort((a, b) => b.pct - a.pct)[0];
  const epochIso = run?.res.epochs[tIdx];
  const sunDir = epochIso && mode === 'epoch' ? sunDirAt(Date.parse(epochIso)) : undefined;
  const presetInfo = presetList.find((p) => p.id === preset) ?? presetList[0];
  const showEpochField = mode === 'epoch' && (!live || !!frame);
  const epochValues = live ? frame?.values : run?.res.values[tIdx];
  const epochReasons = live ? frame?.reasons : run?.reasons[tIdx];
  const nEpochs = run?.res.epochs.length ?? 0;
  const barTicks = derived ? (derived.vmaxCount <= 8 ? Array.from({ length: derived.vmaxCount }, (_, i) => i + 1) : Array.from({ length: Math.ceil(derived.vmaxCount / 2) }, (_, i) => Math.min(derived.vmaxCount, 2 * (i + 1)))) : undefined;

  return (
    <div className="studio">
      <TopBar controls={false} />
      <div className="studio-body">
        {/* ---------------- left: controls ---------------- */}
        <aside className="studio-col left">
          <h2>Coverage &amp; Blind Spots</h2>
          <p className="lede">Where can the network detect a reference object, and why not elsewhere. Blind spots are the market problem, not something to hide.</p>

          <h3>Sensor network</h3>
          <div className="seg" role="radiogroup" aria-label="Network preset">
            {presetList.map((p) => (
              <button key={p.id} role="radio" aria-checked={preset === p.id} className={preset === p.id ? 'active' : ''} onClick={() => setPreset(p.id)} title={p.hint}>
                {p.label}
              </button>
            ))}
          </div>
          <p className="hint">
            {run && !run.mock ? presetInfo.live_hint : presetInfo.hint}
            {livePresets && <span className="muted"> · backend preset list</span>}
          </p>

          <h3>Time window</h3>
          <div className="seg" role="radiogroup" aria-label="Time window">
            {WINDOWS.map((w) => (
              <button key={w.id} role="radio" aria-checked={win === w.id} className={win === w.id ? 'active' : ''} onClick={() => setWin(w.id)}>
                {w.label}
              </button>
            ))}
          </div>
          <div className="ctl">
            <label htmlFor="cov-t0">Start (UTC)</label>
            <input id="cov-t0" type="text" value={t0} onChange={(e) => setT0(e.target.value)} placeholder="YYYY-MM-DDTHH:MM" spellCheck={false} />
          </div>
          <p className="hint">
            Grid: rotating-frame xy slice (z = 0), x ∈ [−1.6, 1.6], y ∈ [−1.4, 1.4] L*; mock {GRID.nx}×{GRID.ny} cells, live backend 64×56 (rendered as served).
          </p>

          <h3>Reference object</h3>
          <div className="ctl">
            <label htmlFor="cov-r">Radius (m)</label>
            <input id="cov-r" type="number" min={0.1} max={10} step={0.1} value={radius} onChange={(e) => setRadius(Math.max(0.05, Number(e.target.value) || 0.05))} />
            <label htmlFor="cov-a">Albedo</label>
            <input id="cov-a" type="number" min={0.02} max={1} step={0.02} value={albedo} onChange={(e) => setAlbedo(Math.min(1, Math.max(0.01, Number(e.target.value) || 0.01)))} />
          </div>
          <p className="hint">Diffuse sphere, m = −26.74 − 2.5 log₁₀(a·πρ²·F(φ)/R²); detectable when m ≤ m_lim of the sensor.</p>

          <div className="btn-row">
            <button className="primary wide" onClick={doRun} disabled={busy}>
              {busy ? (
                'Running…'
              ) : (
                <>
                  <Icon name="play" size={15} />
                  Run coverage
                </>
              )}
            </button>
          </div>
          <div className="status-line">
            {err && <div className="err">ERROR: {err}</div>}
            {run && (
              <>
                <div>
                  source:{' '}
                  {run.mock ? (
                    <span className="tag schematic" title="Browser-side schematic model: the backend coverage engine was not reachable. Relative patterns only.">
                      SCHEMATIC MODEL
                    </span>
                  ) : (
                    <span className="tag live-src" title="Backend coverage engine (/api/coverage): DE440s geometry, named sensor catalogue">
                      LIVE · BACKEND
                    </span>
                  )}{' '}
                </div>
                <div>
                  <span className="nowrap">{run.elapsed_ms.toFixed(0)}&nbsp;ms</span> · epochs {nEpochs} · cells {run.res.grid.x.length * run.res.grid.y.length}
                </div>
                <div>sensors: {run.res.sensors_used?.join(', ') ?? '(backend default)'}</div>
              </>
            )}
          </div>

          <details className="explainer-box">
            <summary>{run && !run.mock ? 'Assumptions (backend model)' : 'Assumptions (schematic browser model)'}</summary>
            {run && !run.mock ? (
              <p className="hint">
                Sensor catalogue and constraint set are the backend's: the named ground sites and space observers listed above, each with its own elevation mask, twilight, Sun/Moon/Earth exclusion, shadow and photometric tests; the reference object is a
                diffuse sphere of the radius and albedo set here. {run.res.averaged?.note ?? ''} Sun compass on the map is the mean synodic phase (±7°); the backend itself uses DE440s.
              </p>
            ) : (
              <p className="hint">
                3 notional equatorial ground sites 120° apart (m_lim 19.5, elevation &gt; 20°, Sun &lt; −12°, Moon avoidance 5–25° scaled with illuminated fraction). Space observers: Sun 40°, Moon 10°, Earth 10° exclusion; m_lim 18.0 (GEO) / 18.5 (L2 halo, DRO). Umbra
                cones for Earth and Moon. Sun direction from the mean synodic phase (±7°). Schematic host orbits; FOV/slew assumed schedulable within an epoch step.
              </p>
            )}
          </details>
        </aside>

        {/* ---------------- centre: heatmap ---------------- */}
        <section className="studio-col center">
          <div className="heat-head">
            <span>
              <b>{mode === 'epoch' ? 'SENSOR COUNT' : 'TIME-AVERAGED'}</b>
            </span>
            <span className="seg">
              <button className={mode === 'epoch' ? 'active' : ''} onClick={() => setMode('epoch')} title={live ? 'Sensor count per epoch (frames fetched from the backend on demand)' : 'Sensor count per epoch'}>
                Per epoch
              </button>
              <button className={mode === 'avg' ? 'active' : ''} onClick={() => setMode('avg')}>
                Time-averaged
              </button>
              <button className={families ? 'active' : ''} onClick={() => setFamilies((f) => !f)} title="Toggle DRO / NRHO / halo outlines">
                Orbit outlines
              </button>
            </span>
            {run?.mock && (
              <span className="tag schematic" title="Browser-side schematic model (backend unreachable)">
                SCHEMATIC MODEL
              </span>
            )}
            {live && mode === 'epoch' && framesLoaded < nEpochs && (
              <span className="muted mono" style={{ fontSize: 12 }}>
                frames {framesLoaded}/{nEpochs}
              </span>
            )}
            <span className="spacer" />
            {derived && (
              <span className="kpi-tiles">
                <span className="tile" title="Mean coverage over the window: fraction of slice cells with ≥ 1 detecting sensor, averaged over epochs">
                  <span className="k">{showEpochField ? 'coverage · this epoch' : 'mean coverage'}</span>
                  <span className="v">{(showEpochField ? (derived.perEpochPct[tIdx] ?? derived.meanPct) : derived.meanPct).toFixed(1)}%</span>
                </span>
                <span className="tile warn" title="Largest blind-spot cause by share of slice cells">
                  <span className="k">top blind reason</span>
                  <span className="v">{topBlind ? `${REASON_LABELS[topBlind.reason].replace(/\s*\(.*\)$/, '')} · ${topBlind.pct.toFixed(0)}%` : '—'}</span>
                </span>
              </span>
            )}
          </div>

          {busy && <div className="progress-thin" role="progressbar" aria-label="Computing coverage" />}
          {run && derived ? (
            <CoverageHeatmap
              x={run.res.grid.x}
              y={run.res.grid.y}
              values={showEpochField && epochValues ? epochValues : derived.avg}
              reasons={showEpochField && epochReasons ? epochReasons : derived.avgReason}
              vmax={showEpochField ? derived.vmaxCount : 1}
              caption={showEpochField ? 'sensors able to detect the reference object' : 'fraction of epochs with ≥ 1 sensor'}
              format={showEpochField ? (v) => `${Math.round(v)} sensor${Math.round(v) === 1 ? '' : 's'}` : (v) => `${(100 * v).toFixed(0)}%`}
              ticks={showEpochField ? barTicks : undefined}
              tickFormat={showEpochField ? (v) => String(Math.round(v)) : undefined}
              sunDir={sunDir}
              overlayFamilies={families}
            />
          ) : (
            <div className="center-empty">{busy ? '' : 'Run coverage to render the field.'}</div>
          )}

          {run && (
            <div className="time-row">
              <span className="muted mono" style={{ fontSize: 12 }}>
                {mode === 'epoch' ? `EPOCH ${tIdx + 1}/${nEpochs}` : `AVERAGE OF ${nEpochs} EPOCHS`}
              </span>
              <input type="range" min={0} max={Math.max(0, nEpochs - 1)} value={tIdx} onChange={(e) => setTIdx(Number(e.target.value))} disabled={mode !== 'epoch'} aria-label="Epoch" />
              <span className="utc">{mode === 'epoch' && epochIso ? fmtUtc(epochIso) : nEpochs ? `${fmtUtc(run.res.epochs[0])} → ${fmtUtc(run.res.epochs[nEpochs - 1])}` : '—'}</span>
            </div>
          )}
          {frameErr && mode === 'epoch' && live && <div className="status-line err">Per-epoch frame failed: {frameErr}</div>}
          {busy && <div className="center-busy quiet" />}
          {!busy && live && mode === 'epoch' && !frame && <div className="progress-thin" role="progressbar" aria-label="Loading epoch" />}
        </section>

        {/* ---------------- right: why blind + series ---------------- */}
        <aside className="studio-col right">
          <h3>Why blind {showEpochField ? '· this epoch' : '· whole window'}</h3>
          {run?.reasonsEstimated && <p className="hint">Backend returned no per-cell reasons; breakdown estimated client-side from the geometric model.</p>}
          {live && !showEpochField && <p className="hint">Live backend, whole window: blind cell-epochs attributed to each cell's dominant reason over the window.</p>}
          <ul className="reasons">
            {legend.map((row) => (
              <li key={row.reason} className={row.reason === 'covered' ? 'covered' : ''}>
                <span className="name">{REASON_LABELS[row.reason]}</span>
                <span className="pct">{row.pct.toFixed(1)}%</span>
                <span className="bar">
                  <span style={{ width: `${Math.min(100, row.pct)}%` }} />
                </span>
              </li>
            ))}
            {!run && <li className="muted">—</li>}
          </ul>
          <p className="hint">Percent of slice cells. Dominant reason = the constraint that blocks the sensor closest to detecting (shadow → daylight → Sun/Earth exclusion → lunar glare → photometric limit).</p>

          <h3>Coverage % vs time</h3>
          <div className="chart-box tall">
            {derived && (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={derived.series} margin={{ top: 8, right: 12, bottom: 4, left: -14 }}>
                  <CartesianGrid stroke="#1c2733" vertical={false} />
                  <XAxis dataKey="k" tick={{ fill: '#8d9db3', fontSize: 12 }} tickFormatter={(k: number) => derived.series[k]?.t.slice(5, 11) ?? ''} stroke="#1c2733" minTickGap={28} />
                  <YAxis domain={[0, 100]} tick={{ fill: '#8d9db3', fontSize: 12 }} stroke="#1c2733" unit="%" />
                  <Tooltip
                    content={({ active, payload }) =>
                      active && payload && payload.length ? (
                        <div className="studio-tip">
                          <div>{(payload[0].payload as { t: string }).t}</div>
                          <div>coverage {(payload[0].value as number).toFixed(1)}%</div>
                        </div>
                      ) : null
                    }
                  />
                  {mode === 'epoch' && <ReferenceLine x={tIdx} stroke="#f5b700" strokeDasharray="3 3" />}
                  <Line type="monotone" dataKey="pct" stroke="#4cc9f0" strokeWidth={2} dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            )}
          </div>
          <p className="chart-note">Fraction of slice cells with ≥ 1 detecting sensor, per epoch. Yellow marker = epoch shown on the map.</p>

          <h3>Reading the map</h3>
          <p className="hint">
            Ground-only networks lose the sunward half of cislunar space to daylight, a disc around the Moon to glare, and distant cells to the photometric limit. Space-based observers near the Moon recover the glare hole and the far side, at the price of their own Sun-exclusion cone. Use
            Time-averaged to see persistent blind regions; Per epoch to watch them rotate with the Sun.
          </p>
        </aside>
      </div>
    </div>
  );
}
