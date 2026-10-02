/**
 * Reachability panel: POST /api/reachability for the analysis target at the time cursor with an assumed impulsive
 * Δv budget (0–200 m/s) and horizon (24–168 h). The reachable endpoints are drawn in the scene coloured by the
 * first region each sample enters (scene/AnalysisOverlays.tsx); the panel lists the regions (fraction of rays,
 * earliest arrival, minimum Δv) and the backend's sensor-pointing hints. Awareness only: it says where the object
 * COULD be so sensors can be pointed; it asserts no intent.
 */
import { useMemo, useState } from 'react';
import { api } from '../../api/client';
import type { ReachRequest } from '../../api/analysisTypes';
import { sensorName } from '../../demo/headline';
import { useAnalysis } from '../../store/useAnalysis';
import { Ctl, fmtNum, fmtPct, LiveBadge, Note, RunBar, shortUtc, Tile, Tiles, useAnalysisTarget } from './common';
import { OTHER, regionColor } from './palette';

export function ReachabilityPanel() {
  const { objectId, isReal, t0Iso, ready, clamped } = useAnalysisTarget();
  const state = useAnalysis((s) => s.reach);
  const run = useAnalysis((s) => s.run);
  const show = useAnalysis((s) => s.showReach);
  const setShow = useAnalysis((s) => s.setShowReach);
  const [dv, setDv] = useState(50);
  const [horizon, setHorizon] = useState(168);
  const [nSamples, setNSamples] = useState(1000);

  function go() {
    if (!objectId) return;
    const req: ReachRequest = { object_id: objectId, t0: t0Iso, dv_budget_mps: dv, horizon_h: horizon, n_samples: nSamples, include_paths: false, include_hints: true, refine: true };
    void run('reach', objectId, t0Iso, (signal) => api.reachability(req, signal));
  }

  const d = state.data;
  const hints = useMemo(() => {
    if (!d) return [];
    return Object.values(d.sensor_hints.summary)
      .filter((h) => h.peak_fov_capture_fraction > 0)
      .sort((a, b) => b.peak_fov_capture_fraction - a.peak_fov_capture_fraction || (a.first_opportunity_h ?? 1e9) - (b.first_opportunity_h ?? 1e9))
      .slice(0, 6);
  }, [d]);
  // The regions' `fraction` is per RAY (one direction × burn epoch, all magnitudes); the tile uses the same
  // denominator so "1 % enter a region" never sits next to "L2 gateway 5 %". Samples-based when `ray` is absent.
  const hit = useMemo(() => {
    if (!d) return { n: 0, of: 0, unit: 'rays' };
    const rays = d.samples.ray;
    if (rays && rays.length === d.samples.n) {
      const hitRays = new Set<number>();
      rays.forEach((r, i) => {
        if (d.samples.hit_region[i]) hitRays.add(r);
      });
      return { n: hitRays.size, of: Math.max(1, d.config.n_dirs * d.config.burn_epochs_h.length), unit: 'rays' };
    }
    return { n: d.samples.hit_region.filter((x) => x).length, of: Math.max(1, d.samples.n), unit: 'samples' };
  }, [d]);
  const terminated = d ? d.samples.terminated.filter(Boolean).length : 0;
  const reached = d ? d.regions.filter((r) => r.fraction > 0) : [];

  return (
    <div className="an-panel">
      <p className="muted small">
        Where <b className="mono">{objectId ?? '—'}</b> could be from its state at <span className="mono">{shortUtc(t0Iso)}</span> if it burned: impulsive Δv on a Fibonacci sphere × magnitude ladder × burn epochs (0/6/12/24 h), propagated with the DE440s force model. Fractions are fractions of sampled rays.
      </p>
      <div className="an-ctls">
        <Ctl label={`Δv budget ${dv} m/s`} title="Assumed planning budget, 0–200 m/s">
          <input type="range" min={0} max={200} step={5} value={dv} onChange={(e) => setDv(Number(e.target.value))} />
        </Ctl>
        <Ctl label={`Horizon ${horizon} h`} title="24–168 h (gateways from a DRO only open on the week scale)">
          <input type="range" min={24} max={168} step={24} value={horizon} onChange={(e) => setHorizon(Number(e.target.value))} />
        </Ctl>
        <Ctl label="Samples">
          <select value={nSamples} onChange={(e) => setNSamples(Number(e.target.value))}>
            {[500, 1000, 2000, 4000].map((n) => (
              <option key={n} value={n}>
                ≈{n}
              </option>
            ))}
          </select>
        </Ctl>
      </div>
      <RunBar label="Compute reachable set" state={state} onRun={go} disabled={!ready || !objectId} disabledReason="Waiting for the catalogue epoch from the backend…" backendS={d?.timing.route_total_s}>
        {clamped && <Note>The time cursor is outside the backend's cached truth window for this object; the run epoch was clamped to {shortUtc(t0Iso)}.</Note>}
      {d && (
          <label className="an-check">
            <input type="checkbox" checked={show} onChange={(e) => setShow(e.target.checked)} /> show in scene
          </label>
        )}
      </RunBar>
      {isReal && <Note>Real object: the envelope is a hypothetical what-if for custody planning; no maneuver is simulated or implied.</Note>}

      {d && (
        <>
          <Tiles>
            <Tile k="Samples" v={d.samples.n.toLocaleString('en-US')} s={`${d.config.n_dirs} dirs × ${d.config.magnitudes_mps.length} mags × ${d.config.burn_epochs_h.length} epochs`} sev="muted" />
            <Tile k="Enter a region" v={fmtPct((100 * hit.n) / hit.of)} s={`${hit.n} of ${hit.of} ${hit.unit} reach ≥ 1 region`} sev={hit.n ? 'warn' : 'ok'} title={hit.unit === 'rays' ? 'A ray = one burn direction × burn epoch over the whole magnitude ladder; the same denominator as the region table' : undefined} />
            <Tile k="Terminated" v={terminated} s="lunar impact / Earth re-entry" sev={terminated ? 'alert' : 'muted'} />
          </Tiles>

          <h4 className="an-h">
            Regions within {d.config.horizon_h} h at ≤ {d.config.dv_budget_mps} m/s <LiveBadge />
          </h4>
          <table className="tbl an-tbl">
            <thead>
              <tr>
                <th>region</th>
                <th className="num" title="Fraction of rays (direction × burn epoch) that enter the region at some magnitude ≤ budget">rays</th>
                <th className="num">earliest</th>
                <th className="num">min Δv</th>
              </tr>
            </thead>
            <tbody>
              {d.regions.map((r) => (
                <tr key={r.key} className={r.fraction > 0 ? '' : 'dim'} title={r.why_it_matters}>
                  <td>
                    <i className="sw" style={{ background: r.fraction > 0 ? regionColor(r.key) : OTHER }} />
                    {r.name}
                    {r.newly_reachable && <span className="tag warn" style={{ marginLeft: 6 }}>NEW</span>}
                    {r.nominal_hits && <span className="tag muted" style={{ marginLeft: 6 }} title="The unperturbed (no-burn) trajectory also enters this region">nominal</span>}
                  </td>
                  <td className="mono num">{fmtPct(100 * r.fraction)}</td>
                  <td className="mono num">{r.earliest_h !== null ? `${fmtNum(r.earliest_h, 0)} h` : '—'}</td>
                  <td className="mono num">{r.min_dv_mps !== null ? `${fmtNum(r.refined_min_dv?.dv_mps ?? r.min_dv_mps, 1)} m/s` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          {reached.length === 0 && <Note>No high-value region is reachable with this budget and horizon — custody search can stay local.</Note>}

          <h4 className="an-h">Where to point</h4>
          {hints.length === 0 ? (
            <Note>No sensor can capture the set in a single field within the horizon ({d.sensor_hints.note}).</Note>
          ) : (
            <table className="tbl an-tbl">
              <thead>
                <tr>
                  <th>sensor</th>
                  <th className="num">peak capture</th>
                  <th className="num">first window</th>
                  <th className="num">steps</th>
                </tr>
              </thead>
              <tbody>
                {hints.map((h) => (
                  <tr key={h.sensor_id} className={h.sensor_id === d.sensor_hints.best_overall ? 'best' : ''}>
                    <td>
                      {sensorName(h.sensor_id)}
                      {h.sensor_id === d.sensor_hints.best_overall && <span className="tag ok" style={{ marginLeft: 6 }}>BEST</span>}
                    </td>
                    <td className="mono num">{fmtPct(100 * h.peak_fov_capture_fraction)}</td>
                    <td className="mono num">{h.first_opportunity_h !== null ? `+${fmtNum(h.first_opportunity_h, 0)} h` : 'now'}</td>
                    <td className="mono num muted">{h.n_steps_visible}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <Note>
            Peak capture = largest fraction of the active set inside one exposure at the centroid boresight{d.sensor_hints.photometry ? ` (assumed ${fmtNum(d.sensor_hints.photometry.radius_m, 1)} m radius, albedo ${fmtNum(d.sensor_hints.photometry.albedo, 2)})` : ''}. {d.config.caps_applied.length ? `Caps: ${d.config.caps_applied.join('; ')}.` : ''}
          </Note>
        </>
      )}
    </div>
  );
}
