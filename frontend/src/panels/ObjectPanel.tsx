/**
 * Object panel: identity, orbit type, actor, state vector (km, km/s, mono), custody badge, covariance sparkline,
 * last-observation age, observation history (sensor, time, residual ″), maneuver history (time, Δv, direction,
 * confidence), reachability quick-look when the current frame carries a reachable set.
 *
 * State vector: when a scenario frame is loaded the GCRF state is DERIVED from the rotating-frame samples
 * (lib/ephem.ts planar frame model + finite-difference velocity) and labelled as such; otherwise the catalog epoch
 * state is shown.
 */
import { useMemo, type ReactNode } from 'react';
import type { SeleneEvent, Vec3 } from '../api/types';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { L_STAR_KM } from '../lib/cr3bp';
import { rotPosToGcrfKm } from '../lib/ephem';
import { cursorDate, fmtAge, fmtElapsed, fmtUtc, useSelene } from '../store/useSelene';
import { SigmaSparkline, type SigmaPoint } from './SigmaSparkline';

function Field({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="field">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}

const f = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const vec = (v: Vec3 | number[], d: number) => v.map((x) => f(x, d)).join(', ');

function dirLabel(dir: unknown): string {
  if (Array.isArray(dir) && dir.length === 3 && dir.every((x) => typeof x === 'number')) {
    const [x, y, z] = dir as number[];
    const az = (Math.atan2(y, x) * 180) / Math.PI, el = (Math.asin(Math.max(-1, Math.min(1, z))) * 180) / Math.PI;
    return `az ${az.toFixed(0)}° el ${el.toFixed(0)}°`;
  }
  return typeof dir === 'string' ? dir : '—';
}

export function ObjectPanel() {
  const id = useSelene((s) => s.selectedObjectId);
  const catalog = useSelene((s) => s.catalog);
  const events = useSelene((s) => s.events);
  // Throttled cursor: quantised to ~0.2 s of wall-clock at the current speed (min 1 s of sim time), so the panel
  // and its Recharts sparkline re-render ≈5×/s while playing instead of on every animation frame.
  const tSec = useSelene((s) => {
    const q = Math.max(1, s.speed * 0.2);
    return Math.min(s.t1Sec, Math.floor(s.tSec / q) * q);
  });
  const t0Iso = useSelene((s) => s.t0Iso);
  const scenario = useSelene((s) => s.scenario);
  const select = useSelene((s) => s.selectObject);
  const { objects, frame, reachable, source } = useScenarioFrame();

  const cat = catalog.find((c) => c.id === id) ?? null;
  const live = objects.find((o) => o.id === id) ?? null;

  const sigmaSeries: SigmaPoint[] = useMemo(() => {
    if (!scenario || !id) return [];
    const out: SigmaPoint[] = [];
    for (const fr of scenario.frames) {
      const o = fr.objects.find((q) => q.id === id);
      if (o?.sigma_pos_km !== undefined) out.push({ h: Math.round((fr.t / 3600) * 10) / 10, sigma: o.sigma_pos_km });
    }
    // Thin to ≤ 200 points for the chart.
    const stride = Math.max(1, Math.ceil(out.length / 200));
    return out.filter((_, i) => i % stride === 0 || i === out.length - 1);
  }, [scenario, id]);

  if (!id) {
    return (
      <div>
        <h3 className="panel-title">Object</h3>
        <p className="muted">Select an object in the scene or from the list.</p>
        {catalog.length > 0 && (
          <ul className="events">
            {catalog.map((c) => {
              const lv = objects.find((o) => o.id === c.id);
              return (
                <li key={c.id} onClick={() => select(c.id, 'user')} style={{ ['--sev' as string]: c.kind === 'simulated' ? 'var(--sim)' : 'var(--accent)' }}>
                  <span className="bar" />
                  <div>
                    <div className="meta">
                      <span>{c.id}</span>
                      <span className="kind">{c.kind === 'simulated' ? 'SIMULATED' : 'HORIZONS'}</span>
                      {lv && <span className={`tag ${lv.custody === 'held' ? 'ok' : lv.custody === 'degraded' ? 'warn' : 'alert'}`}>{lv.custody.toUpperCase()}</span>}
                    </div>
                    <div className="text">
                      {c.name} <span className="muted">· {c.orbit_type}</span>
                    </div>
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </div>
    );
  }

  const custody = live?.custody ?? 'held';
  const custodyTag = custody === 'held' ? 'ok' : custody === 'degraded' ? 'warn' : 'alert';
  const custodyLabel = custody === 'held' ? 'CUSTODY' : custody.toUpperCase();
  const myEvents = events.filter((e) => e.object_id === id && e.t <= tSec);
  const observations = myEvents.filter((e) => e.kind === 'observation');
  const lastObs = observations[observations.length - 1];
  const maneuvers = myEvents.filter((e) => e.kind === 'maneuver_detected' || e.kind === 'maneuver_characterized');
  const nowMs = Date.parse(t0Iso) + tSec * 1000;

  // State vector: derived from rotating-frame samples when a frame is loaded.
  let rKm: Vec3 | null = null, vKms: Vec3 | null = null, derived = false;
  if (live) {
    const dt = Math.max(1, live.tNext - live.tFrame);
    const w = Math.min(1, Math.max(0, (tSec - live.tFrame) / dt));
    const p: Vec3 = [0, 1, 2].map((k) => live.pos[k] + (live.posNext[k] - live.pos[k]) * w) as Vec3;
    rKm = rotPosToGcrfKm(p, nowMs);
    const r1 = rotPosToGcrfKm(live.pos, Date.parse(t0Iso) + live.tFrame * 1000);
    const r2 = rotPosToGcrfKm(live.posNext, Date.parse(t0Iso) + live.tNext * 1000);
    vKms = [(r2[0] - r1[0]) / dt, (r2[1] - r1[1]) / dt, (r2[2] - r1[2]) / dt];
    derived = true;
  } else if (cat) {
    rKm = [cat.state_gcrf_km[0], cat.state_gcrf_km[1], cat.state_gcrf_km[2]];
    vKms = [cat.state_gcrf_km[3], cat.state_gcrf_km[4], cat.state_gcrf_km[5]];
  }
  const sigma = live?.sigmaKm;

  return (
    <div>
      <h3 className="panel-title">
        Object{' '}
        <button className="toggle" style={{ float: 'right' }} onClick={() => select(null)}>
          ✕
        </button>
      </h3>
      <div className="obj-head">
        <div className="obj-id mono">{id}</div>
        <span className={`tag ${custodyTag} big`}>{custodyLabel}</span>
      </div>
      <Field k="Name" v={cat?.name ?? '—'} />
      <Field k="Kind" v={cat?.kind === 'horizons' ? <span className="tag accent">REAL · HORIZONS</span> : <span className="tag sim">SIMULATED</span>} />
      <Field k="Actor" v={cat?.kind === 'horizons' ? cat.actor : `notional actor${cat && cat.actor !== 'notional' ? ` (${cat.actor})` : ''}`} />
      <Field k="Orbit type" v={cat?.orbit_type ?? '—'} />
      <Field k="Epoch" v={live ? `${fmtUtc(cursorDate(t0Iso, tSec))}${derived ? ' (frame)' : ''}` : (cat?.epoch_utc ?? '—')} />
      <Field k="r GCRF km" v={rKm ? <span className="mono">{vec(rKm, 1)}</span> : '—'} />
      <Field k="v GCRF km/s" v={vKms ? <span className="mono">{vec(vKms, 4)}</span> : '—'} />
      <Field k="r rot L*" v={live ? <span className="mono">{vec(live.pos, 5)}</span> : cat?.ic_rot ? <span className="mono">{vec([cat.ic_rot[0], cat.ic_rot[1], cat.ic_rot[2]], 5)}</span> : '—'} />
      {derived && <p className="muted small">GCRF values derived from rotating-frame display samples (mean-element Earth–Moon line + J2000 obliquity; lunar inclination neglected); see backend OD for filtered states.</p>}
      <Field k="σ_pos √tr P" v={sigma !== undefined ? <span className="mono">{sigma >= 100 ? sigma.toFixed(0) : sigma.toFixed(1)} km</span> : <span className="muted">—</span>} />
      <Field k="Last obs" v={lastObs ? <span className="mono">{fmtAge(tSec - lastObs.t)} ago · {String(lastObs.data?.sensor_id ?? '')}</span> : <span className="muted">none in window</span>} />
      {sigmaSeries.length > 1 && <SigmaSparkline data={sigmaSeries} cursorH={tSec / 3600} />}

      {reachable && live && (
        <>
          <h3 className="panel-title" style={{ marginTop: 14 }}>
            Reachability {reachable.dv_budget_mps ? `· Δv ≤ ${reachable.dv_budget_mps} m/s` : ''} {reachable.horizon_h ? `· ${reachable.horizon_h} h` : ''}
          </h3>
          <table className="tbl">
            <tbody>
              {reachable.regions.map((rg) => (
                <tr key={rg.name}>
                  <td>{rg.name}</td>
                  <td className="mono num">{(100 * rg.fraction).toFixed(0)}%</td>
                  <td className="mono num muted">{rg.earliest_h !== null ? `${rg.earliest_h} h` : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      <h3 className="panel-title" style={{ marginTop: 14 }}>
        Observations <span className="mono muted">({observations.length})</span>
      </h3>
      {observations.length === 0 ? (
        <p className="muted small">No observations yet in this window.</p>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>T</th>
              <th>Sensor</th>
              <th className="num">Resid ″</th>
            </tr>
          </thead>
          <tbody>
            {[...observations].reverse().slice(0, 12).map((e, i) => (
              <tr key={i}>
                <td className="mono">{fmtElapsed(e.t).slice(1, -3)}</td>
                <td>{String(e.data?.sensor_id ?? '—')}</td>
                <td className={`mono num${Number(e.data?.residual_arcsec) > 5 ? ' warn' : ''}`}>{typeof e.data?.residual_arcsec === 'number' ? (e.data.residual_arcsec as number).toFixed(2) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3 className="panel-title" style={{ marginTop: 14 }}>
        Maneuvers <span className="mono muted">({maneuvers.length})</span>
      </h3>
      {maneuvers.length === 0 ? (
        <p className="muted small">None detected.</p>
      ) : (
        <table className="tbl">
          <thead>
            <tr>
              <th>T</th>
              <th className="num">Δv m/s</th>
              <th>Direction</th>
              <th className="num">Conf.</th>
            </tr>
          </thead>
          <tbody>
            {maneuvers.map((e: SeleneEvent, i) => {
              const dv = e.data?.dv_mps, sg = e.data?.dv_sigma_mps, cf = e.data?.confidence;
              return (
                <tr key={i}>
                  <td className="mono">{fmtElapsed(e.t).slice(1, -3)}</td>
                  <td className="mono num">{typeof dv === 'number' ? `${dv.toFixed(1)}${typeof sg === 'number' ? ` ± ${sg.toFixed(1)}` : ''}` : <span className="muted">pending</span>}</td>
                  <td className="mono">{dirLabel(e.data?.direction)}</td>
                  <td className="mono num">{typeof cf === 'number' ? cf.toFixed(2) : e.kind === 'maneuver_detected' ? 'det.' : '—'}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
      {frame && source === 'idle' && <p className="muted small" style={{ marginTop: 10 }}>Idle view: positions are a browser CR3BP display propagation from the catalog epoch (1 L* = {L_STAR_KM.toLocaleString('en-US')} km).</p>}
    </div>
  );
}
