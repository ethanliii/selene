/**
 * Tasking panel: POST /api/tasking/schedule (method greedy · milp · random · round_robin · compare) with a sensor
 * preset from GET /api/tasking/presets over the default xGEO object set (+ the selected object when it is not in
 * it). Shows custody % / mean time-since-last-observation overall and per object, a Gantt-like strip per sensor
 * (one block per assigned slot, coloured by object in fixed order), and for 'compare' the policy comparison bars.
 */
import { useEffect, useMemo, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../../api/client';
import { TASKING_METHODS, type TaskingMethod, type TaskingRequest } from '../../api/analysisTypes';
import { sensorShort } from '../../demo/headline';
import { useAnalysis } from '../../store/useAnalysis';
import { Ctl, fmtKm, fmtNum, fmtPct, GRID, LiveBadge, Note, RunBar, shortUtc, TICK, Tile, Tiles, TIP_STYLE, useAnalysisTarget } from './common';
import { categorical, OTHER } from './palette';

const METHOD_LABEL: Record<TaskingMethod, string> = { greedy: 'greedy (information gain)', milp: 'MILP (windowed optimisation)', random: 'random (baseline)', round_robin: 'round robin (baseline)', compare: 'compare all four' };

export function TaskingPanel() {
  const { objectId, t0Iso, ready, clamped } = useAnalysisTarget();
  const state = useAnalysis((s) => s.tasking);
  const presets = useAnalysis((s) => s.taskingPresets);
  const setPresets = useAnalysis((s) => s.setTaskingPresets);
  const run = useAnalysis((s) => s.run);
  const [method, setMethod] = useState<TaskingMethod>('greedy');
  const [preset, setPreset] = useState('default');
  const [horizon, setHorizon] = useState(48);
  const [slot, setSlot] = useState(60);
  const [presetErr, setPresetErr] = useState<string | null>(null);

  useEffect(() => {
    if (presets) return;
    api
      .taskingPresets()
      .then(setPresets)
      .catch((e) => setPresetErr(e instanceof Error ? e.message : String(e)));
  }, [presets, setPresets]);

  function go() {
    const defaults = presets?.default_object_ids ?? [];
    const ids = objectId && objectId.startsWith('SIM') && !defaults.includes(objectId) && defaults.length ? [...defaults, objectId] : 'default';
    const t1 = new Date(Date.parse(t0Iso) + horizon * 3600e3).toISOString().replace(/\.\d{3}Z$/, 'Z');
    const req: TaskingRequest = { object_ids: ids, preset, t0: t0Iso, t1, slot_min: slot, method, time_budget_s: 15, include_series: true, seed: 0 };
    void run('tasking', objectId ?? 'default', t0Iso, (signal) => api.taskingSchedule(req, signal));
  }

  const d = state.data;
  const objects = d?.config.object_ids ?? [];
  const colorOf = (id: string) => categorical(objects.indexOf(id));
  const rows = useMemo(() => {
    if (!d) return [];
    const by = new Map<string, typeof d.schedule>();
    for (const sid of d.config.sensor_ids) by.set(sid, []);
    for (const a of d.schedule) (by.get(a.sensor_id) ?? by.set(a.sensor_id, []).get(a.sensor_id)!).push(a);
    return [...by.entries()];
  }, [d]);
  const nSlots = d?.config.n_slots ?? 1;
  const cmp = d?.comparison ?? [];
  const SHORT: Record<string, string> = { greedy: 'greedy', milp: 'MILP', random: 'random', round_robin: 'RR' };
  const cmpData = cmp.map((r) => ({ name: SHORT[r.method] ?? r.method, custody: r.custody_pct, tslo: r.mean_tslo_h, trace: r.summed_trace_mean_km2 }));
  const presetInfo = presets?.sensor_presets[preset];

  return (
    <div className="an-panel">
      <p className="muted small">
        Schedule a heterogeneous network over {horizon} h from <span className="mono">{shortUtc(t0Iso)}</span> to keep custody of the default xGEO set{objectId && objectId.startsWith('SIM') && presets && !presets.default_object_ids.includes(objectId) ? ` plus ${objectId}` : ''}: linear-covariance analysis, one angles-only tracklet per assigned slot, FOV acquisition model, slew limits.
      </p>
      <div className="an-ctls">
        <Ctl label="Method">
          <select value={method} onChange={(e) => setMethod(e.target.value as TaskingMethod)}>
            {TASKING_METHODS.map((m) => (
              <option key={m} value={m} disabled={m === 'milp' && presets?.milp_available === false}>
                {METHOD_LABEL[m]}
              </option>
            ))}
          </select>
        </Ctl>
        <Ctl label="Sensors" title={presetInfo ? `${presetInfo.ground_ids.length} ground sites + ${presetInfo.space.join(', ') || 'no space observers'}` : 'Presets from GET /api/tasking/presets'}>
          <select value={preset} onChange={(e) => setPreset(e.target.value)} disabled={!presets}>
            {presets ? Object.keys(presets.sensor_presets).map((k) => <option key={k} value={k}>{k.replace(/_/g, ' ')}</option>) : <option>{presetErr ? 'presets unavailable' : 'loading presets…'}</option>}
          </select>
        </Ctl>
        <Ctl label="Horizon">
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            {[24, 48, 72].map((h) => (
              <option key={h} value={h}>
                {h} h
              </option>
            ))}
          </select>
        </Ctl>
        <Ctl label="Slot">
          <select value={slot} onChange={(e) => setSlot(Number(e.target.value))}>
            {[20, 30, 60].map((h) => (
              <option key={h} value={h}>
                {h} min
              </option>
            ))}
          </select>
        </Ctl>
      </div>
      {presetErr && <div className="an-error">Presets: {presetErr}</div>}
      <RunBar label={method === 'compare' ? 'Run & compare' : 'Schedule'} state={state} onRun={go} disabled={!ready || (!presets && !presetErr)} disabledReason="Waiting for the catalogue epoch from the backend…" backendS={d?.timing.total_s} />

      {clamped && <Note>The time cursor is outside the backend's cached truth window for this object; the run epoch was clamped to {shortUtc(t0Iso)}.</Note>}
      {d && (
        <>
          <Tiles>
            <Tile k="Custody" v={fmtPct(d.overall.custody_pct, 1)} s={`null (no obs) ${fmtPct(d.overall.custody_pct_null, 0)} · σ < ${d.config.custody_threshold_km} km`} sev={d.overall.custody_pct >= 90 ? 'ok' : d.overall.custody_pct >= 60 ? 'warn' : 'alert'} title={d.overall.custody_note} />
            <Tile k="Mean TSLO" v={`${fmtNum(d.overall.mean_tslo_h, 2)} h`} s={`max ${fmtNum(d.overall.max_tslo_h, 1)} h`} sev={d.overall.mean_tslo_h <= 2 ? 'ok' : d.overall.mean_tslo_h <= 6 ? 'warn' : 'alert'} />
            <Tile k="Tracklets" v={d.overall.n_observations} s={`${d.overall.n_sensors} sensors · ${d.overall.n_slots} slots`} sev="muted" />
          </Tiles>
          {d.overall.budget_exhausted && <Note>MILP time budget exhausted: remaining slots were filled greedily.</Note>}

          <h4 className="an-h">
            Schedule · {d.method} <LiveBadge />
          </h4>
          <div className="gantt" role="img" aria-label="Sensor schedule">
            {rows.map(([sid, list]) => (
              <div className="gantt-row" key={sid}>
                <span className="gantt-lbl" title={sid}>
                  {sensorShort(sid)}
                </span>
                <div className="gantt-strip">
                  {list.map((a, i) => (
                    <span key={i} className="gantt-blk" style={{ left: `${(100 * a.slot) / nSlots}%`, width: `${Math.max(0.6, 100 / nSlots - 0.3)}%`, background: colorOf(a.object_id) }} title={`${sensorShort(sid)} → ${a.object_id} · slot ${a.slot} · σ ${fmtKm(a.sigma_before_km, 1)} → ${fmtKm(a.sigma_after_km, 1)} km · p_acq ${fmtNum(a.p_acq, 2)}`} />
                  ))}
                  {list.length === 0 && <span className="gantt-empty">no assignable slot (blocked)</span>}
                </div>
              </div>
            ))}
            <div className="gantt-axis">
              <span>{shortUtc(d.config.t0_utc)}</span>
              <span>{shortUtc(d.config.t1_utc)}</span>
            </div>
          </div>
          <div className="an-legend wrap">
            {objects.map((id, i) => (
              <span key={id}>
                <i style={{ background: i < 8 ? categorical(i) : OTHER }} /> {id}
              </span>
            ))}
          </div>

          <h4 className="an-h">Per object</h4>
          <table className="tbl an-tbl">
            <thead>
              <tr>
                <th>object</th>
                <th className="num">custody</th>
                <th className="num">TSLO h</th>
                <th className="num">obs</th>
                <th className="num">σ end km</th>
              </tr>
            </thead>
            <tbody>
              {d.per_object.map((o) => (
                <tr key={o.id} className={o.id === objectId ? 'best' : ''}>
                  <td>
                    <i className="sw" style={{ background: colorOf(o.id) }} />
                    {o.id}
                  </td>
                  <td className={`mono num${o.custody_pct < 60 ? ' warn' : ''}`}>{fmtPct(o.custody_pct, 0)}</td>
                  <td className="mono num">{fmtNum(o.mean_tslo_h, 1)}</td>
                  <td className="mono num">{o.never_observed ? <span className="warn">0</span> : o.n_obs}</td>
                  <td className="mono num">{fmtKm(o.final_sigma_km, 1)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          {cmp.length > 0 && (
            <>
              <h4 className="an-h">Policy comparison (same scenario, same visibility)</h4>
              <div className="an-chart an-chart-pair">
                <div>
                  <div className="an-chart-title">custody % (σ &lt; {d.config.custody_threshold_km} km)</div>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={cmpData} margin={{ top: 16, right: 4, bottom: 0, left: -14 }} barCategoryGap="25%">
                      <CartesianGrid stroke={GRID} vertical={false} />
                      <XAxis dataKey="name" tick={{ ...TICK, fontSize: 11 }} stroke={GRID} interval={0} angle={-30} textAnchor="end" height={30} />
                      <YAxis domain={[0, 100]} tick={TICK} stroke={GRID} />
                      <Tooltip contentStyle={TIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.04)' }} formatter={(v: number) => [`${fmtNum(v, 1)}%`, 'custody']} />
                      <Bar dataKey="custody" isAnimationActive={false} radius={[3, 3, 0, 0]}>
                        {cmpData.map((_, i) => (
                          <Cell key={i} fill={categorical(i)} />
                        ))}
                        <LabelList dataKey="custody" position="top" formatter={(v: number) => v.toFixed(0)} style={{ fill: '#d6e2f0', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div>
                  <div className="an-chart-title">mean time since last obs, h (lower is better)</div>
                  <ResponsiveContainer width="100%" height={150}>
                    <BarChart data={cmpData} margin={{ top: 16, right: 4, bottom: 0, left: -14 }} barCategoryGap="25%">
                      <CartesianGrid stroke={GRID} vertical={false} />
                      <XAxis dataKey="name" tick={{ ...TICK, fontSize: 11 }} stroke={GRID} interval={0} angle={-30} textAnchor="end" height={30} />
                      <YAxis tick={TICK} stroke={GRID} />
                      <Tooltip contentStyle={TIP_STYLE} cursor={{ fill: 'rgba(255,255,255,0.04)' }} formatter={(v: number) => [`${fmtNum(v, 2)} h`, 'mean TSLO']} />
                      <Bar dataKey="tslo" isAnimationActive={false} radius={[3, 3, 0, 0]}>
                        {cmpData.map((_, i) => (
                          <Cell key={i} fill={categorical(i)} />
                        ))}
                        <LabelList dataKey="tslo" position="top" formatter={(v: number) => v.toFixed(1)} style={{ fill: '#d6e2f0', fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
              <table className="tbl an-tbl">
                <thead>
                  <tr>
                    <th>policy</th>
                    <th className="num">custody</th>
                    <th className="num">TSLO h</th>
                    <th className="num">Σtr P km²</th>
                    <th className="num">obs</th>
                    <th className="num">s</th>
                  </tr>
                </thead>
                <tbody>
                  {cmp.map((r, i) => (
                    <tr key={r.method}>
                      <td>
                        <i className="sw" style={{ background: categorical(i) }} />
                        {r.method.replace('_', ' ')}
                      </td>
                      <td className="mono num">{fmtPct(r.custody_pct, 1)}</td>
                      <td className="mono num">{fmtNum(r.mean_tslo_h, 2)}</td>
                      <td className="mono num">{fmtKm(r.summed_trace_mean_km2, 1)}</td>
                      <td className="mono num">{r.n_observations}</td>
                      <td className="mono num muted">{fmtNum(r.runtime_s, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <Note>When the null custody (no observations at all) is already {fmtPct(d.overall.custody_pct_null, 0)}, custody % does not discriminate policies; the summed covariance trace and TSLO do.</Note>
            </>
          )}
          <Note>{d.disclaimer}</Note>
        </>
      )}
    </div>
  );
}
