/**
 * Maneuver panel: POST /api/maneuver/detect on the analysis target with an injectable SIMULATED burn (magnitude,
 * direction, epoch offset after t0). Shows the NIS series with the χ² gates and the burn/detection epochs, the
 * detection list (first exceedance per test), the Δv estimate against the injected truth, and the latency.
 * The injected burn is attributed to a notional actor; nothing here concerns a real spacecraft.
 */
import { useMemo, useState } from 'react';
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Scatter, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../../api/client';
import { MANEUVER_DIRECTIONS, type ManeuverDetectRequest, type ManeuverDirection } from '../../api/analysisTypes';
import { sensorName } from '../../demo/headline';
import { useAnalysis } from '../../store/useAnalysis';
import { Ctl, fmtNum, GRID, hoursSince, LiveBadge, Note, RunBar, shortUtc, TICK, Tile, Tiles, TIP_STYLE, useAnalysisTarget } from './common';
import { SERIES } from './palette';

const TEST_LABEL: Record<string, string> = { nis: 'NIS', nis_window: 'windowed NIS', gap_refit: 'gap re-fit', cusum: 'CUSUM', nees: 'NEES (truth)' };

export function ManeuverPanel() {
  const { objectId, isReal, t0Iso, ready, clamped } = useAnalysisTarget();
  const state = useAnalysis((s) => s.maneuver);
  const run = useAnalysis((s) => s.run);
  const [mag, setMag] = useState(30);
  const [dir, setDir] = useState<ManeuverDirection>('prograde');
  const [burnH, setBurnH] = useState(24);
  const [spanD, setSpanD] = useState(3);
  const [cadence, setCadence] = useState(120);
  const [inject, setInject] = useState(true);

  function go() {
    if (!objectId) return;
    const t1 = new Date(Date.parse(t0Iso) + spanD * 86400e3).toISOString().replace(/\.\d{3}Z$/, 'Z');
    const tb = new Date(Date.parse(t0Iso) + burnH * 3600e3).toISOString().replace(/\.\d{3}Z$/, 'Z');
    const req: ManeuverDetectRequest = { object_id: objectId, t0: t0Iso, t1, cadence_min: cadence, alpha: 0.01, injected: inject && mag > 0 ? { t_burn_utc: tb, magnitude_mps: mag, direction: dir } : null, seed: 0, estimate: true };
    void run('maneuver', objectId, t0Iso, (signal) => api.maneuverDetect(req, signal));
  }

  const d = state.data;
  const base = state.t0 ?? t0Iso;
  const series = useMemo(() => (d ? d.filter.epochs_utc.map((ep, i) => ({ h: Math.round(hoursSince(ep, base) * 100) / 100, nis: Math.max(1e-3, d.filter.nis[i]), sensor: d.filter.sensor_ids[i] })) : []), [d, base]);
  const firstPerTest = useMemo(() => {
    if (!d) return [];
    const seen = new Set<string>();
    return d.detections.filter((x) => (seen.has(x.test) ? false : (seen.add(x.test), true))).sort((a, b) => a.t_s - b.t_s);
  }, [d]);
  const burnHTruth = d?.truth ? hoursSince(d.truth.t_burn_utc, base) : null;
  const detectH = d?.first_declared_utc ? hoursSince(d.first_declared_utc, base) : d?.first_detection_utc ? hoursSince(d.first_detection_utc, base) : null;
  const detectPts = useMemo(() => firstPerTest.map((x) => ({ h: Math.round(hoursSince(x.t_utc, base) * 100) / 100, nis: Math.max(1e-3, x.test === 'nis' ? x.statistic : d?.filter.nis[x.index] ?? 1), test: x.test })), [firstPerTest, base, d]);
  const est = d?.dv_estimate ?? null;
  const err = d?.truth?.estimate_error ?? null;
  const latencyH = d?.truth?.detection_latency_s != null ? d.truth.detection_latency_s / 3600 : null;
  const sev = d ? (d.declared ? (d.truth ? 'ok' : 'alert') : d.truth ? 'alert' : 'ok') : 'muted';

  return (
    <div className="an-panel">
      <p className="muted small">
        Sequential filter on <b className="mono">{objectId ?? '—'}</b> from <span className="mono">{shortUtc(t0Iso)}</span> with NIS · windowed-NIS · gap re-fit · CUSUM tests (false-alarm rate α = 0.01, family-wise corrected). Inject a SIMULATED burn by a notional actor and see whether, when and how well it is caught.
      </p>
      <div className="an-ctls">
        <label className="an-check an-ctl-wide">
          <input type="checkbox" checked={inject} onChange={(e) => setInject(e.target.checked)} /> inject a test burn (SIMULATED)
        </label>
        <Ctl label={`Δv ${mag} m/s`} title="Impulsive burn magnitude">
          <input type="range" min={0} max={100} step={1} value={mag} onChange={(e) => setMag(Number(e.target.value))} disabled={!inject} />
        </Ctl>
        <Ctl label="Direction">
          <select value={dir} onChange={(e) => setDir(e.target.value as ManeuverDirection)} disabled={!inject}>
            {MANEUVER_DIRECTIONS.map((x) => (
              <option key={x} value={x}>
                {x.replace(/_/g, ' ')}
              </option>
            ))}
          </select>
        </Ctl>
        <Ctl label={`Burn at T+${burnH} h`} title="Burn epoch after t0 (inside the window)">
          <input type="range" min={6} max={spanD * 24 - 12} step={1} value={Math.min(burnH, spanD * 24 - 12)} onChange={(e) => setBurnH(Number(e.target.value))} disabled={!inject} />
        </Ctl>
        <Ctl label="Window">
          <select value={spanD} onChange={(e) => setSpanD(Number(e.target.value))}>
            {[2, 3, 5, 7].map((x) => (
              <option key={x} value={x}>
                {x} d
              </option>
            ))}
          </select>
        </Ctl>
        <Ctl label="Cadence">
          <select value={cadence} onChange={(e) => setCadence(Number(e.target.value))}>
            {[60, 120, 180, 360].map((x) => (
              <option key={x} value={x}>
                {x} min
              </option>
            ))}
          </select>
        </Ctl>
      </div>
      <RunBar label="Detect" state={state} onRun={go} disabled={!ready || !objectId || isReal} disabledReason={!ready ? 'Waiting for the catalogue epoch from the backend…' : isReal ? 'Residual tests against a real Horizons ephemeris would report model mismatch as a "maneuver" of a real spacecraft — the backend refuses it.' : 'Select an object'} backendS={d?.timing.total_s} />
      {isReal && <Note>Real (JPL Horizons) object selected: maneuver tests run only on SIMULATED objects.</Note>}

      {clamped && <Note>The time cursor is outside the backend's cached truth window for this object; the run epoch was clamped to {shortUtc(t0Iso)}.</Note>}
      {d && (
        <>
          <Tiles>
            <Tile k="Verdict" v={d.declared ? 'DECLARED' : d.status.replace(/_/g, ' ')} s={d.truth ? (d.declared ? 'injected burn caught' : 'injected burn MISSED') : d.declared ? 'false alarm (no burn injected)' : 'quiet, as injected'} sev={sev} title={d.status_note} />
            <Tile k="Latency" v={latencyH !== null ? `${fmtNum(latencyH, 1)} h` : '—'} s={d.truth ? `burn ${shortUtc(d.truth.t_burn_utc)}` : 'no injected burn'} sev={latencyH === null ? 'muted' : latencyH <= 6 ? 'ok' : 'warn'} />
            <Tile k="Updates" v={`${d.summary.n_tested}/${d.summary.n_updates}`} s={`tested · mean NIS ${fmtNum(d.summary.mean_nis, 1)}`} sev="muted" />
          </Tiles>
          {series.length > 1 && (
            <>
              <h4 className="an-h">
                NIS per update · χ²₂(1−α) gate {d.summary.threshold_nis.toFixed(1)} <LiveBadge />
              </h4>
              <div className="an-chart">
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={series} margin={{ top: 8, right: 8, bottom: 0, left: -6 }}>
                    <CartesianGrid stroke={GRID} vertical={false} />
                    <XAxis dataKey="h" type="number" domain={['dataMin', 'dataMax']} tick={TICK} stroke={GRID} tickFormatter={(v: number) => `${v}h`} />
                    <YAxis scale="log" domain={[0.01, 'auto']} tick={TICK} stroke={GRID} width={44} tickFormatter={(v: number) => (v >= 10 ? v.toFixed(0) : v.toFixed(2))} allowDataOverflow />
                    <Tooltip contentStyle={TIP_STYLE} labelFormatter={(v) => `T+${Number(v).toFixed(1)} h`} formatter={(v: number, _n: string, item) => [`${fmtNum(v, 2)} · ${sensorName(String((item.payload as { sensor?: string }).sensor ?? ''))}`, 'NIS']} />
                    <ReferenceLine y={d.summary.threshold_nis} stroke={SERIES.threshold} strokeDasharray="4 3" />
                    <ReferenceLine y={d.summary.threshold_nis_familywise} stroke={SERIES.familywise} strokeDasharray="2 3" />
                    {burnHTruth !== null && <ReferenceLine x={burnHTruth} stroke={SERIES.burn} strokeDasharray="3 3" label={{ value: 'burn', fill: '#8d9db3', fontSize: 11, position: 'insideTopRight' }} />}
                    {detectH !== null && <ReferenceLine x={detectH} stroke={SERIES.threshold} label={{ value: 'declared', fill: '#ff4d4f', fontSize: 11, position: 'insideBottomRight' }} />}
                    <Line type="monotone" dataKey="nis" stroke={SERIES.nis} strokeWidth={2} dot={false} isAnimationActive={false} />
                    <Scatter data={detectPts} dataKey="nis" fill={SERIES.threshold} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
                <div className="an-legend">
                  <span>
                    <i style={{ background: SERIES.nis }} /> NIS
                  </span>
                  <span>
                    <i style={{ background: SERIES.threshold }} /> gate / first exceedance
                  </span>
                  <span>
                    <i style={{ background: SERIES.familywise }} /> family-wise gate {d.summary.threshold_nis_familywise.toFixed(1)}
                  </span>
                </div>
              </div>
            </>
          )}

          <h4 className="an-h">Detections ({d.detections.length} exceedances, first per test)</h4>
          {firstPerTest.length === 0 ? (
            <Note>No test exceeded its threshold in the window.</Note>
          ) : (
            <table className="tbl an-tbl">
              <thead>
                <tr>
                  <th>T+h</th>
                  <th>test</th>
                  <th className="num">stat</th>
                  <th className="num">gate</th>
                  <th>sensor</th>
                </tr>
              </thead>
              <tbody>
                {firstPerTest.map((x) => (
                  <tr key={x.test}>
                    <td className="mono">{fmtNum(hoursSince(x.t_utc, base), 1)}</td>
                    <td>{TEST_LABEL[x.test] ?? x.test}</td>
                    <td className="mono num">{x.statistic >= 1000 ? x.statistic.toExponential(1) : fmtNum(x.statistic, 1)}</td>
                    <td className="mono num muted">{fmtNum(x.threshold, 1)}</td>
                    <td>{sensorName(x.sensor_id)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h4 className="an-h">Δv estimate vs injected truth</h4>
          <table className="tbl an-tbl dv-tbl">
            <thead>
              <tr>
                <th></th>
                <th className="num">estimate</th>
                <th className="num">truth</th>
                <th className="num">error</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>|Δv| m/s</td>
                <td className="mono num wrap">{est ? `${fmtNum(est.magnitude_mps, 2)} ± ${fmtNum(est.magnitude_sigma_mps, 2)}` : '—'}</td>
                <td className="mono num">{d.truth ? fmtNum(d.truth.magnitude_mps, 1) : '—'}</td>
                <td className="mono num wrap">{err ? `${err.magnitude_error_mps >= 0 ? '+' : ''}${fmtNum(err.magnitude_error_mps, 2)} (${fmtNum(err.magnitude_error_pct, 1)}%)` : '—'}</td>
              </tr>
              <tr>
                <td>direction</td>
                <td className="mono num">{est ? `σ ${fmtNum(est.direction_sigma_deg, 2)}°` : '—'}</td>
                <td className="mono num wrap">{d.truth ? dir.replace(/_/g, ' ') : '—'}</td>
                <td className="mono num">{err ? `${fmtNum(err.direction_error_deg, 2)}°` : '—'}</td>
              </tr>
              <tr>
                <td>burn epoch</td>
                <td className="mono num wrap">{est ? `${shortUtc(est.t_burn_utc)} ± ${fmtNum(est.t_burn_sigma_s, 0)} s` : '—'}</td>
                <td className="mono num wrap">{d.truth ? shortUtc(d.truth.t_burn_utc) : '—'}</td>
                <td className="mono num">{err ? `${fmtNum(err.t_burn_error_s, 0)} s` : '—'}</td>
              </tr>
            </tbody>
          </table>
          {est?.classification && (
            <Note>
              Heuristic class: <b>{est.classification.primary}</b> (descriptive only; no intent is inferred). Fit: {est.n_obs} post-burn obs, residual RMS {fmtNum(est.residual_rms_arcsec, 2)}″, reduced χ² {fmtNum(est.reduced_chi2, 2)}.
            </Note>
          )}
          {!est && d.declared && <Note>Declared, but no Δv estimate converged in this window (too few post-burn observations).</Note>}
          {d.summary.filter_health?.warnings?.length ? <Note>Filter health: {d.summary.filter_health.warnings.join('; ')}</Note> : null}
        </>
      )}
    </div>
  );
}
