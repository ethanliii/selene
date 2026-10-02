/**
 * OD panel: POST /api/od/run for the analysis target with a preset from GET /api/od/presets.
 * Shows the observation yield (and why epochs were dropped), the IOD / batch / UKF summary against the SIMULATED
 * truth, the σ_pos + truth-error series, the NIS / NEES consistency series with their χ² bounds, and the particle-
 * cloud custody decay (slider over the propagated frames; the chosen frame is drawn in the scene by
 * scene/AnalysisOverlays.tsx). Everything shown is the backend's answer; errors are inline.
 */
import { useEffect, useMemo, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../../api/client';
import type { OdRunRequest } from '../../api/analysisTypes';
import { useAnalysis } from '../../store/useAnalysis';
import { Ctl, fmtKm, fmtNum, GRID, hoursSince, LiveBadge, Note, RunBar, shortUtc, TICK, Tile, Tiles, TIP_STYLE, useAnalysisTarget } from './common';
import { SERIES } from './palette';

const SENSOR_SETS: { id: string; label: string }[] = [
  { id: 'all', label: 'all 15 sensors' },
  { id: 'ground', label: 'ground sites only' },
  { id: 'space', label: 'space observers only' },
];

export function OdPanel() {
  const { objectId, isReal, t0Iso, inScenario, ready, clamped } = useAnalysisTarget();
  const state = useAnalysis((s) => s.od);
  const presets = useAnalysis((s) => s.odPresets);
  const setPresets = useAnalysis((s) => s.setOdPresets);
  const run = useAnalysis((s) => s.run);
  const odFrame = useAnalysis((s) => s.odFrame);
  const setOdFrame = useAnalysis((s) => s.setOdFrame);
  const showCloud = useAnalysis((s) => s.showOdCloud);
  const setShowCloud = useAnalysis((s) => s.setShowOdCloud);
  const [preset, setPreset] = useState('ui_quick');
  const [sensors, setSensors] = useState('all');
  const [windowH, setWindowH] = useState(24);
  const [presetErr, setPresetErr] = useState<string | null>(null);

  useEffect(() => {
    if (presets) return;
    api
      .odPresets()
      .then(setPresets)
      .catch((e) => setPresetErr(e instanceof Error ? e.message : String(e)));
  }, [presets, setPresets]);

  const p = presets?.presets[preset];
  useEffect(() => {
    if (p?.sensors && typeof p.sensors === 'string') setSensors(p.sensors);
  }, [p]);

  function go() {
    if (!objectId) return;
    const t1 = new Date(Date.parse(t0Iso) + windowH * 3600e3).toISOString().replace(/\.\d{3}Z$/, 'Z');
    const req: OdRunRequest = {
      object_id: objectId,
      t0: t0Iso,
      t1,
      sensors,
      cadence_min: p?.cadence_min ?? 120,
      method: p?.method ?? 'all',
      particles: { n: p?.particles?.n ?? 1000, t_grid_h: p?.particles?.t_grid_h ?? 72, step_h: p?.particles?.step_h ?? 6, max_export: 400 },
      seed: 0,
    };
    void run('od', objectId, t0Iso, (signal) => api.odRun(req, signal));
  }

  const d = state.data;
  const sigmaSeries = useMemo(() => {
    if (!d?.ukf) return [];
    const base = d.t0_utc;
    return d.ukf.epochs.map((ep, i) => ({ h: Math.round(hoursSince(ep, base) * 100) / 100, sigma: d.ukf!.sigma_pos_km[i], err: d.truth_error_km?.[i] ?? null, pred: d.ukf!.sigma_pos_pred_km?.[i] ?? null }));
  }, [d]);
  const nisSeries = useMemo(() => {
    if (!d?.ukf) return [];
    return d.ukf.nis.map((v, i) => ({ i: i + 1, nis: Math.max(1e-3, v), nees: d.nees ? Math.max(1e-3, d.nees[i]) : null }));
  }, [d]);
  const neesIn = useMemo(() => {
    if (!d?.nees || !d.nees_bounds_95_single_epoch) return null;
    const [lo, hi] = d.nees_bounds_95_single_epoch;
    const n = d.nees.filter((v) => v >= lo && v <= hi).length;
    return { frac: n / d.nees.length, lo, hi };
  }, [d]);
  const part = d?.particles ?? null;
  const frameIdx = part ? Math.min(odFrame, part.frames.length - 1) : 0;
  const decay = useMemo(() => (part ? part.metrics.hours.map((h, i) => ({ h, sigma: part.metrics.sigma_pos_km[i] })) : []), [part]);
  const dropped = d ? Object.entries(d.observations.dropped_by_reason).sort((a, b) => b[1] - a[1]) : [];

  return (
    <div className="an-panel">
      <p className="muted small">
        Angles-only OD on <b className="mono">{objectId ?? '—'}</b> from the time cursor <span className="mono">{shortUtc(t0Iso)}</span>: simulated RA/Dec tracklets through the full visibility model, then IOD → batch least squares → UKF, and a particle cloud propagated with no further observations.
      </p>
      <div className="an-ctls">
        <Ctl label="Preset" title={p?.note ?? 'Presets from GET /api/od/presets'}>
          <select value={preset} onChange={(e) => setPreset(e.target.value)} disabled={!presets}>
            {presets ? Object.keys(presets.presets).map((k) => <option key={k} value={k}>{k.replace(/_/g, ' ')}</option>) : <option>{presetErr ? 'presets unavailable' : 'loading presets…'}</option>}
          </select>
        </Ctl>
        <Ctl label="Sensors">
          <select value={sensors} onChange={(e) => setSensors(e.target.value)}>
            {SENSOR_SETS.map((s) => (
              <option key={s.id} value={s.id}>
                {s.label}
              </option>
            ))}
          </select>
        </Ctl>
        <Ctl label="Window" title="Observation window after t0 (the particle cloud starts at its end)">
          <select value={windowH} onChange={(e) => setWindowH(Number(e.target.value))}>
            {[12, 24, 48, 72].map((h) => (
              <option key={h} value={h}>
                {h} h
              </option>
            ))}
          </select>
        </Ctl>
      </div>
      {presetErr && <div className="an-error">Presets: {presetErr}</div>}
      <RunBar label="Run OD" state={state} onRun={go} disabled={!ready || !objectId || isReal} disabledReason={!ready ? 'Waiting for the catalogue epoch from the backend…' : isReal ? 'OD runs only for SIMULATED objects (they have a synthetic truth); real Horizons objects never get synthetic observations.' : 'Select an object'} backendS={d?.timing.total_s} />
      {isReal && <Note>Real (JPL Horizons) object selected: the backend refuses synthetic OD on real spacecraft. Select a SIMULATED object.</Note>}
      {inScenario && <Note>The scenario is loaded: this run uses the catalogue truth of the object at the cursor epoch, independent of the story's injected burn.</Note>}

      {clamped && <Note>The time cursor is outside the backend's cached truth window for this object; the run epoch was clamped to {shortUtc(t0Iso)}.</Note>}
      {d && (
        <>
          <h4 className="an-h">
            Observations <LiveBadge />
          </h4>
          <Tiles>
            <Tile k="Usable" v={`${d.observations.n_used} / ${d.observations.n_candidates}`} s={`${(100 * d.observations.fraction_observed).toFixed(0)}% of candidate epochs×sensors`} sev={d.observations.n_used === 0 ? 'alert' : d.observations.fraction_observed < 0.25 ? 'warn' : 'ok'} />
            <Tile k="Blind" v={dropped[0] ? `${dropped[0][1]}` : '0'} s={dropped[0] ? dropped[0][0].replace(/_/g, ' ') : 'nothing dropped'} sev="muted" title={dropped.map(([k, v]) => `${k}: ${v}`).join(' · ')} />
            <Tile k="Noise" v={`${d.observations.sigma_arcsec}″`} s="1σ astrometric" sev="muted" />
          </Tiles>
          {d.notes.length > 0 && (
            <ul className="an-notes">
              {d.notes.map((n, i) => (
                <li key={i}>{n}</li>
              ))}
            </ul>
          )}

          <h4 className="an-h">Solutions vs SIMULATED truth</h4>
          <table className="tbl an-tbl">
            <thead>
              <tr>
                <th>Method</th>
                <th className="num">pos err km</th>
                <th className="num">σ_pos km</th>
                <th className="num">RMS ″</th>
                <th>note</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>IOD</td>
                <td className="mono num">{d.iod?.best ? fmtKm(d.iod.best.truth_error_pos_km, 2) : '—'}</td>
                <td className="mono num">{d.iod?.best ? fmtKm(d.iod.best.sigma_pos_km, 2) : '—'}</td>
                <td className="mono num">{d.iod?.best ? fmtNum(d.iod.best.rms_arcsec) : '—'}</td>
                <td className="muted">{d.iod ? `${d.iod.quality}, ${d.iod.n_obs} obs` : 'not run'}</td>
              </tr>
              <tr>
                <td>Batch LS</td>
                <td className="mono num">{d.batch ? fmtKm(d.batch.truth_error_pos_km, 2) : '—'}</td>
                <td className="mono num">{d.batch ? fmtKm(d.batch.sigma_pos_km, 2) : '—'}</td>
                <td className="mono num">{d.batch ? fmtNum(d.batch.weighted_rms) : '—'}</td>
                <td className="muted">{d.batch ? `${d.batch.converged ? 'converged' : 'NOT converged'} in ${d.batch.iterations} it · NEES ${fmtNum(d.batch.truth_nees, 1)}` : 'not run'}</td>
              </tr>
              <tr>
                <td>{d.ukf?.meta.filter.toUpperCase() ?? 'UKF'}</td>
                <td className="mono num">{d.truth_error_km ? fmtKm(d.truth_error_km[d.truth_error_km.length - 1], 2) : '—'}</td>
                <td className="mono num">{d.ukf ? fmtKm(d.ukf.sigma_pos_km[d.ukf.sigma_pos_km.length - 1], 2) : '—'}</td>
                <td className="mono num">{d.ukf ? `NIS ${fmtNum(d.ukf.meta.mean_nis, 1)}` : '—'}</td>
                <td className="muted">{d.ukf ? `${d.ukf.n} updates · final values` : 'not run'}</td>
              </tr>
            </tbody>
          </table>

          {sigmaSeries.length > 1 && (
            <>
              <h4 className="an-h">σ_pos and truth error · km vs T+h</h4>
              <div className="an-chart">
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={sigmaSeries} margin={{ top: 8, right: 8, bottom: 0, left: -6 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="h" type="number" domain={['dataMin', 'dataMax']} tick={TICK} stroke={GRID} tickFormatter={(v: number) => `${v}h`} />
                    <YAxis scale="log" domain={['auto', 'auto']} tick={TICK} stroke={GRID} width={44} tickFormatter={(v: number) => (v >= 100 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2))} />
                    <Tooltip contentStyle={TIP_STYLE} labelFormatter={(v) => `T+${Number(v).toFixed(1)} h`} formatter={(v: number, name: string) => [`${fmtKm(v, 2)} km`, name === 'sigma' ? 'σ_pos (posterior)' : name === 'err' ? 'error vs truth' : 'σ_pos (predicted)']} />
                    <Line type="monotone" dataKey="sigma" stroke={SERIES.sigma} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="err" stroke={SERIES.truth} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="an-legend">
                  <span>
                    <i style={{ background: SERIES.sigma }} /> σ_pos posterior
                  </span>
                  <span>
                    <i style={{ background: SERIES.truth }} /> |error| vs truth
                  </span>
                </div>
              </div>
            </>
          )}

          {nisSeries.length > 1 && (
            <>
              <h4 className="an-h">Filter consistency · NIS (χ²₂, 99 % gate 9.21) and NEES (χ²₆) per update</h4>
              <div className="an-chart">
                <ResponsiveContainer width="100%" height={150}>
                  <LineChart data={nisSeries} margin={{ top: 8, right: 8, bottom: 0, left: -6 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="i" type="number" domain={['dataMin', 'dataMax']} tick={TICK} stroke={GRID} tickFormatter={(v: number) => `#${v}`} />
                    <YAxis scale="log" domain={[0.01, 'auto']} tick={TICK} stroke={GRID} width={44} tickFormatter={(v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(2))} allowDataOverflow />
                    <Tooltip contentStyle={TIP_STYLE} labelFormatter={(v) => `update #${v}`} formatter={(v: number, name: string) => [fmtNum(v, 2), name === 'nis' ? 'NIS' : 'NEES']} />
                    <ReferenceLine y={9.21} stroke={SERIES.threshold} strokeDasharray="4 3" />
                    {neesIn && <ReferenceLine y={neesIn.hi} stroke={SERIES.nees} strokeDasharray="2 3" />}
                    {neesIn && <ReferenceLine y={neesIn.lo} stroke={SERIES.nees} strokeDasharray="2 3" />}
                    <Line type="monotone" dataKey="nis" stroke={SERIES.nis} strokeWidth={2} dot={false} isAnimationActive={false} />
                    {d.nees && <Line type="monotone" dataKey="nees" stroke={SERIES.nees} strokeWidth={2} dot={false} isAnimationActive={false} />}
                  </LineChart>
                </ResponsiveContainer>
                <div className="an-legend">
                  <span>
                    <i style={{ background: SERIES.nis }} /> NIS
                  </span>
                  <span>
                    <i style={{ background: SERIES.nees }} /> NEES vs truth
                  </span>
                  <span className="muted">dashed: χ²₂ 99% gate · χ²₆ 95% band</span>
                </div>
              </div>
              {neesIn && (
                <Note>
                  NEES inside the 95% single-epoch band [{neesIn.lo}, {neesIn.hi}] for {(100 * neesIn.frac).toFixed(0)}% of updates — {neesIn.frac >= 0.85 ? 'covariance is realistic' : neesIn.frac >= 0.6 ? 'covariance is mildly optimistic/pessimistic' : 'covariance is NOT consistent with the truth error'}.
                </Note>
              )}
            </>
          )}

          {part && (
            <>
              <h4 className="an-h">Custody decay · particle cloud ({part.n_particles.toLocaleString('en-US')} particles, {part.n_exported} drawn)</h4>
              <Tiles>
                <Tile k="σ now" v={`${fmtKm(part.metrics.sigma_pos_km[frameIdx], 2)} km`} s={`T+${part.metrics.hours[frameIdx]} h after the last update`} sev={part.metrics.sigma_pos_km[frameIdx] >= 1000 ? 'alert' : part.metrics.sigma_pos_km[frameIdx] >= 100 ? 'warn' : 'ok'} />
                <Tile k="Growth" v={part.custody.growth_factor ? `×${fmtNum(part.custody.growth_factor, 1)}` : '—'} s={`${fmtKm(part.custody.sigma_pos_km_start, 2)} → ${fmtKm(part.custody.sigma_pos_km_end, 2)} km`} sev="muted" />
                <Tile k="Lost at" v={part.custody.hours_to_1000km !== null ? `${part.custody.hours_to_1000km} h` : `> ${part.metrics.hours[part.metrics.hours.length - 1]} h`} s="σ > 1000 km" sev={part.custody.hours_to_1000km !== null ? 'warn' : 'ok'} />
              </Tiles>
              <div className="an-chart">
                <ResponsiveContainer width="100%" height={110}>
                  <LineChart data={decay} margin={{ top: 8, right: 8, bottom: 0, left: -6 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="h" type="number" domain={['dataMin', 'dataMax']} tick={TICK} stroke={GRID} tickFormatter={(v: number) => `+${v}h`} />
                    <YAxis scale="log" domain={['auto', 'auto']} tick={TICK} stroke={GRID} width={44} tickFormatter={(v: number) => (v >= 100 ? v.toFixed(0) : v.toFixed(1))} />
                    <Tooltip contentStyle={TIP_STYLE} labelFormatter={(v) => `+${v} h`} formatter={(v: number) => [`${fmtKm(v, 2)} km`, 'σ_pos']} />
                    <ReferenceLine x={part.metrics.hours[frameIdx]} stroke={SERIES.cursor} strokeDasharray="3 3" />
                    <Line type="monotone" dataKey="sigma" stroke={SERIES.sigma} strokeWidth={2} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="an-slider">
                <input type="range" min={0} max={part.frames.length - 1} step={1} value={frameIdx} onChange={(e) => setOdFrame(Number(e.target.value))} aria-label="Particle-cloud horizon" />
                <span className="mono">{shortUtc(part.frames[frameIdx]?.epoch)}</span>
                <label className="an-check">
                  <input type="checkbox" checked={showCloud} onChange={(e) => setShowCloud(e.target.checked)} /> show in scene
                </label>
              </div>
              <Note>
                {part.source} propagated with {d.force_model.filter}; {part.custody.note}. The cloud is drawn in blue at the slider epoch (the amber cloud belongs to the scenario story).
              </Note>
            </>
          )}
          {d.caps_applied.length > 0 && <Note>Caps applied by the backend: {d.caps_applied.join('; ')}</Note>}
        </>
      )}
    </div>
  );
}
