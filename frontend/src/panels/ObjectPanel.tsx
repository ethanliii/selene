/**
 * Object panel: identity, orbit type, actor, state vector at the time cursor (km, km/s, mono), custody badge,
 * covariance sparkline, last-observation age, observation history (sensor, time, residual ″), maneuver history
 * (time, Δv, direction, confidence), reachability quick-look when the current frame carries a reachable set.
 *
 * State vector sources, in order of fidelity:
 *  - LIVE: the backend GCRF trajectory of the selected object (/api/catalog/objects/{id}/trajectory?frame=gcrf_km,
 *    DE440s N-body + SRP truth for SIMULATED objects, Horizons spline for real ones), Hermite-interpolated at the
 *    cursor; the rotating-frame row comes from the rot_nd track the scene draws.
 *  - SCENARIO (mock story): GCRF derived from rotating-frame display samples with the display frame model.
 *  - Otherwise the catalog epoch state.
 * Physical parameters of SIMULATED objects are notional design values and are labelled as such; real objects
 * carry their Horizons provenance and cached data span instead.
 */
import { useMemo, type ReactNode } from 'react';
import type { CatalogObject, SeleneEvent, Vec3 } from '../api/types';
import { useObjectTrack } from '../demo/useLiveTracks';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { L_STAR_KM } from '../lib/cr3bp';
import { moonDistanceKm, rotPosToGcrfKm } from '../lib/ephem';
import { trackEval } from '../lib/tracks';
import { MOON_ROT } from '../scene/constants';
import { cursorDate, fmtAge, fmtElapsed, fmtUtc, useSelene } from '../store/useSelene';
import { SigmaSparkline, type SigmaPoint } from './SigmaSparkline';

function Field({ k, v, title }: { k: string; v: ReactNode; title?: string }) {
  return (
    <div className="field" title={title}>
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}

const f = (x: number, d = 3) => (Number.isFinite(x) ? x.toFixed(d) : '—');
const vec = (v: Vec3 | number[], d: number) => v.map((x) => f(x, d)).join(', ');
const km = (x: number) => (Number.isFinite(x) ? `${Math.round(x).toLocaleString('en-US')} km` : '—');

function dirLabel(dir: unknown): string {
  if (Array.isArray(dir) && dir.length === 3 && dir.every((x) => typeof x === 'number')) {
    const [x, y, z] = dir as number[];
    const az = (Math.atan2(y, x) * 180) / Math.PI, el = (Math.asin(Math.max(-1, Math.min(1, z))) * 180) / Math.PI;
    return `az ${az.toFixed(0)}° el ${el.toFixed(0)}°`;
  }
  return typeof dir === 'string' ? dir : '—';
}

function periodText(s: number | null | undefined): string {
  if (!s || !Number.isFinite(s)) return '—';
  if (s >= 86400) return `${(s / 86400).toFixed(2)} d`;
  return `${(s / 3600).toFixed(2)} h`;
}

function KindTag({ c }: { c: CatalogObject | null }) {
  if (!c) return <span className="tag muted">—</span>;
  return c.kind === 'horizons' ? <span className="tag accent">REAL · JPL HORIZONS</span> : <span className="tag sim">SIMULATED</span>;
}

function CatalogList() {
  const catalog = useSelene((s) => s.catalog);
  const meta = useSelene((s) => s.catalogMeta);
  const select = useSelene((s) => s.selectObject);
  const { objects, trackErrors, live, tracksLoaded, tracksTotal } = useScenarioFrame();
  const drawn = new Set(objects.map((o) => o.id));
  const groups: { title: string; kind: CatalogObject['kind']; items: CatalogObject[] }[] = [
    { title: 'SIMULATED · notional actors', kind: 'simulated', items: catalog.filter((c) => c.kind === 'simulated') },
    { title: 'REAL · JPL Horizons ephemerides', kind: 'horizons', items: catalog.filter((c) => c.kind === 'horizons') },
  ];
  return (
    <div>
      <h3 className="panel-title">Object</h3>
      <p className="muted">Select an object in the scene or from the list.</p>
      {live && tracksTotal > 0 && tracksLoaded < tracksTotal && (
        <p className="muted small">
          Loading backend trajectories… {tracksLoaded}/{tracksTotal}
        </p>
      )}
      {groups
        .filter((g) => g.items.length > 0)
        .map((g) => (
          <div key={g.kind} className="catalog-group">
            <h4 className={`catalog-head ${g.kind}`}>
              {g.title} <span className="mono muted">({g.items.length})</span>
            </h4>
            <ul className="events">
              {g.items.map((c) => {
                const lv = objects.find((o) => o.id === c.id);
                const err = trackErrors.get(c.id);
                return (
                  <li key={c.id} onClick={() => select(c.id, 'user')} style={{ ['--sev' as string]: c.kind === 'simulated' ? 'var(--sim)' : 'var(--accent)' }}>
                    <span className="bar" />
                    <div>
                      <div className="meta">
                        <span>{c.id}</span>
                        <span className="kind">{c.kind === 'simulated' ? 'SIMULATED' : 'REAL · JPL HORIZONS'}</span>
                        {lv && lv.custody !== 'unknown' && <span className={`tag ${lv.custody === 'held' ? 'ok' : lv.custody === 'degraded' ? 'warn' : 'alert'}`}>{lv.custody.toUpperCase()}</span>}
                        {lv && lv.custody === 'unknown' && (
                          <span className="tag muted" title="No OD/tasking has evaluated this object; custody is not measured in the idle catalog view">
                            NO OD
                          </span>
                        )}
                        {live && !drawn.has(c.id) && <span className={`tag ${err ? 'alert' : 'muted'}`}>{err ? 'NO DATA IN WINDOW' : 'LOADING'}</span>}
                      </div>
                      <div className="text">
                        {c.name} <span className="muted">· {c.orbit_type}</span>
                      </div>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        ))}
      {meta && <p className="muted small" style={{ marginTop: 10 }}>{meta.disclaimer}</p>}
    </div>
  );
}

export function ObjectPanel() {
  const id = useSelene((s) => s.selectedObjectId);
  const catalog = useSelene((s) => s.catalog);
  const events = useSelene((s) => s.events);
  // Throttled cursor: quantised to ~0.2 s of wall-clock at the current speed (min 1 s of sim time), so the panel
  // and its Recharts sparkline re-render ≈5×/s while playing instead of on every animation frame.
  // When paused / scrubbing the exact cursor is used, so the 'State at cursor' epoch matches the timeline.
  const tSec = useSelene((s) => {
    const q = s.playing ? Math.max(1, s.speed * 0.2) : 1;
    return Math.min(s.t1Sec, Math.floor(s.tSec / q) * q);
  });
  const t0Iso = useSelene((s) => s.t0Iso);
  const scenario = useSelene((s) => s.scenario);
  const select = useSelene((s) => s.selectObject);
  const { objects, frame, reachable, source, live: liveMode, trackErrors } = useScenarioFrame();

  const cat = catalog.find((c) => c.id === id) ?? null;
  const live = objects.find((o) => o.id === id) ?? null;
  const { track: gcrfTrack, error: gcrfErr } = useObjectTrack(id, 'gcrf_km', liveMode && !scenario);

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

  if (!id) return <CatalogList />;

  // 'unknown' (idle live mode, no OD/tasking run) is shown as N/A — never as a measured 'held'.
  const custody = live?.custody ?? 'unknown';
  const custodyTag = custody === 'held' ? 'ok' : custody === 'degraded' ? 'warn' : custody === 'lost' ? 'alert' : 'muted';
  const custodyLabel = custody === 'held' ? 'CUSTODY' : custody === 'unknown' ? 'CUSTODY N/A · NO OD' : custody.toUpperCase();
  const custodyTitle = custody === 'unknown' ? 'No orbit determination or tasking has evaluated this object (OD/tasking endpoints pending); custody is not measured in the idle catalog view.' : `Custody ${custody} (scenario estimate from the uncertainty cloud)`;
  const myEvents = events.filter((e) => e.object_id === id && e.t <= tSec);
  const observations = myEvents.filter((e) => e.kind === 'observation');
  const lastObs = observations[observations.length - 1];
  const maneuvers = myEvents.filter((e) => e.kind === 'maneuver_detected' || e.kind === 'maneuver_characterized');
  const nowMs = Date.parse(t0Iso) + tSec * 1000;

  // ---- state vector at the cursor ---------------------------------------------------------------
  let rKm: Vec3 | null = null, vKms: Vec3 | null = null, rRot: Vec3 | null = null;
  let stateSource: 'backend' | 'derived' | 'catalog' | 'none' = 'none';
  if (live?.track) rRot = trackEval(live.track, tSec, [0, 0, 0] as Vec3);
  if (gcrfTrack) {
    const r: Vec3 = [0, 0, 0], v: Vec3 = [0, 0, 0];
    trackEval(gcrfTrack, tSec, r, v);
    rKm = r;
    vKms = v;
    stateSource = 'backend';
  } else if (live) {
    const dt = Math.max(1, live.tNext - live.tFrame);
    const w = Math.min(1, Math.max(0, (tSec - live.tFrame) / dt));
    const p: Vec3 = rRot ?? ([0, 1, 2].map((k) => live.pos[k] + (live.posNext[k] - live.pos[k]) * w) as Vec3);
    if (!rRot) rRot = p;
    rKm = rotPosToGcrfKm(p, nowMs);
    const r1 = rotPosToGcrfKm(live.pos, Date.parse(t0Iso) + live.tFrame * 1000);
    const r2 = rotPosToGcrfKm(live.posNext, Date.parse(t0Iso) + live.tNext * 1000);
    vKms = [(r2[0] - r1[0]) / dt, (r2[1] - r1[1]) / dt, (r2[2] - r1[2]) / dt];
    stateSource = 'derived';
  } else if (cat) {
    rKm = [cat.state_gcrf_km[0], cat.state_gcrf_km[1], cat.state_gcrf_km[2]];
    vKms = [cat.state_gcrf_km[3], cat.state_gcrf_km[4], cat.state_gcrf_km[5]];
    rRot = cat.ic_rot ? [cat.ic_rot[0], cat.ic_rot[1], cat.ic_rot[2]] : null;
    stateSource = 'catalog';
  }
  const geoRange = rKm ? Math.hypot(rKm[0], rKm[1], rKm[2]) : NaN;
  const selRange = rRot ? Math.hypot(rRot[0] - MOON_ROT[0], rRot[1] - MOON_ROT[1], rRot[2] - MOON_ROT[2]) * moonDistanceKm(nowMs) : NaN;
  const sigma = live?.sigmaKm;
  const isReal = cat?.kind === 'horizons';
  const trackErr = trackErrors.get(id);
  const stepS = live?.track && live.track.n > 1 ? (live.track.t[live.track.n - 1] - live.track.t[0]) / (live.track.n - 1) : 0;
  const rec = cat?.orbit_record ?? null;
  const phys = cat?.physical ?? null;

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
        <span className={`tag ${custodyTag} big`} title={custodyTitle}>
          {custodyLabel}
        </span>
      </div>
      <Field k="Name" v={cat?.name ?? '—'} />
      <Field k="Kind" v={<KindTag c={cat} />} />
      <Field k="Actor" v={cat ? (isReal ? cat.actor : cat.actor.startsWith('notional') ? cat.actor : `notional actor (${cat.actor})`) : '—'} />
      {cat?.role && <Field k="Role" v={cat.role} />}
      <Field k="Orbit type" v={cat?.orbit_type ?? '—'} />
      {(cat?.orbit_ref || cat?.period_s) && (
        <Field k="Orbit ref" v={<span className="mono">{cat.orbit_ref && cat.orbit_ref !== 'keplerian_moon' ? cat.orbit_ref : (cat.orbit_ref ?? '—')}{cat.period_s ? ` · P ${periodText(cat.period_s)}` : ''}</span>} />
      )}
      {cat?.source && <p className="muted small obj-src">{cat.source}</p>}
      {cat?.description && <p className="small obj-desc">{cat.description}</p>}

      <h3 className="panel-title" style={{ marginTop: 12 }}>
        State at cursor
      </h3>
      <Field k="Epoch" v={<span className="mono">{live || gcrfTrack ? fmtUtc(cursorDate(t0Iso, tSec)) : fmtUtc(new Date(cat?.epoch_utc ?? NaN))}</span>} />
      <Field k="r GCRF km" v={rKm ? <span className="mono">{vec(rKm, 1)}</span> : '—'} />
      <Field k="v GCRF km/s" v={vKms ? <span className="mono">{vec(vKms, 4)}</span> : '—'} />
      <Field k="r rot L*" v={rRot ? <span className="mono">{vec(rRot, 5)}</span> : '—'} title="Earth–Moon rotating frame, nondimensional (unit = instantaneous Earth–Moon distance)" />
      <Field k="Range" v={<span className="mono">{km(geoRange)} geocentric · {km(selRange)} from Moon</span>} />
      {stateSource === 'backend' && (
        <p className="muted small">
          Backend trajectory ({isReal ? 'JPL Horizons spline' : 'DE440s Earth+Moon+Sun + SRP truth'}), Hermite-interpolated between {stepS ? `${Math.round(stepS)} s` : ''} samples; not a filter estimate.
        </p>
      )}
      {stateSource === 'derived' && <p className="muted small">GCRF values derived from rotating-frame display samples (display frame model); see backend OD for filtered states.</p>}
      {stateSource === 'catalog' && <p className="muted small">Catalog epoch state{liveMode ? trackErr ? ` — no trajectory in this window: ${trackErr}` : ' — trajectory loading…' : ''}.</p>}
      {gcrfErr && stateSource !== 'backend' && <p className="muted small">GCRF track: {gcrfErr}</p>}
      {(scenario || sigma !== undefined || lastObs) && (
        <>
          <Field k="σ_pos √tr P" v={sigma !== undefined ? <span className="mono">{sigma >= 100 ? sigma.toFixed(0) : sigma.toFixed(1)} km</span> : <span className="muted">—</span>} />
          <Field k="Last obs" v={lastObs ? <span className="mono">{fmtAge(tSec - lastObs.t)} ago · {String(lastObs.data?.sensor_id ?? '')}</span> : <span className="muted">none in window</span>} />
        </>
      )}
      {sigmaSeries.length > 1 && <SigmaSparkline data={sigmaSeries} cursorH={tSec / 3600} />}

      {isReal ? (
        <>
          <h3 className="panel-title" style={{ marginTop: 12 }}>
            Horizons data
          </h3>
          {cat?.span_utc && <Field k="Data span" v={<span className="mono">{fmtUtc(new Date(cat.span_utc[0])).slice(0, 16)} → {fmtUtc(new Date(cat.span_utc[1])).slice(0, 16)} UTC</span>} />}
          {cat?.horizons_id !== undefined && cat?.horizons_id !== null && <Field k="Horizons id" v={<span className="mono">{cat.horizons_id}</span>} />}
          {cat?.regime && <Field k="Regime" v={cat.regime} />}
          <Field k="Physical" v={<span className="muted">not modelled — no photometric assumptions are made for real objects</span>} />
          {(cat?.notes ?? []).map((n, i) => (
            <p key={i} className="muted small">
              {n}
            </p>
          ))}
        </>
      ) : (
        <>
          <h3 className="panel-title" style={{ marginTop: 12 }}>
            Physical assumptions <span className="tag sim">NOTIONAL</span>
          </h3>
          {phys || cat?.radius_m ? (
            <>
              <Field k="Radius" v={<span className="mono">{f(phys?.radius_m ?? cat?.radius_m ?? NaN, 2)} m</span>} />
              <Field k="Albedo" v={<span className="mono">{f(phys?.albedo ?? cat?.albedo ?? NaN, 2)}</span>} />
              {phys?.area_m2 !== undefined && <Field k="Area" v={<span className="mono">{f(phys.area_m2, 1)} m²</span>} />}
              {phys?.mass_kg !== undefined && <Field k="Mass" v={<span className="mono">{f(phys.mass_kg, 0)} kg</span>} />}
              {phys?.cr !== undefined && <Field k="SRP C_R" v={<span className="mono">{f(phys.cr, 2)}{phys.cr_area_mass_m2_kg !== undefined ? ` · C_R·A/m ${phys.cr_area_mass_m2_kg.toFixed(4)} m²/kg` : ''}</span>} />}
              <p className="muted small">{phys?.note ?? 'notional design values (SIMULATED object)'}</p>
            </>
          ) : (
            <p className="muted small">No physical parameters in the catalog record.</p>
          )}
          {rec && (
            <>
              <h3 className="panel-title" style={{ marginTop: 12 }}>
                Orbit record <span className="mono muted">{rec.id}</span>
              </h3>
              <Field k="Family" v={<span className="mono">{rec.family}{rec.branch ? ` (${rec.branch})` : ''}</span>} />
              <Field k="Period" v={<span className="mono">{rec.period_days.toFixed(3)} d</span>} />
              <Field k="Jacobi C" v={<span className="mono">{rec.jacobi.toFixed(4)}</span>} />
              <Field k="Stability ν" v={<span className="mono">{rec.stability_index.toFixed(3)}{Math.abs(rec.stability_index) <= 1 ? ' (stable)' : ' (unstable)'}</span>} />
              {rec.params?.perilune_km !== undefined && <Field k="Perilune / apolune" v={<span className="mono">{km(rec.params.perilune_km)} / {km(rec.params.apolune_km)}</span>} />}
            </>
          )}
          {(cat?.notes ?? []).map((n, i) => (
            <p key={i} className="muted small">
              {n}
            </p>
          ))}
        </>
      )}

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
        <p className="muted small">{scenario ? 'No observations yet in this window.' : 'No observation history loaded (tasking/OD endpoints pending — mock).'}</p>
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
        <p className="muted small">{isReal ? 'None — no events are ever simulated for real objects.' : 'None detected.'}</p>
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
      {frame && source === 'idle' && !liveMode && <p className="muted small" style={{ marginTop: 10 }}>Idle view: positions are a browser CR3BP display propagation from the catalog epoch (1 L* = {L_STAR_KM.toLocaleString('en-US')} km).</p>}
    </div>
  );
}
