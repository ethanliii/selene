/**
 * Object panel: identity, orbit type, state, covariance, custody, observation/maneuver history.
 * Placeholder fields are driven by the catalog + current demo frame when available.
 * TODO(later agent, M10): covariance ellipsoid (3σ axes from UKF cov), observation history table from
 *   /api/od/run, maneuver history from /api/maneuver/detect, reachability quick-look button.
 */
import type { ReactNode } from 'react';
import { useScenarioFrame } from '../demo/useScenarioFrame';
import { fmtElapsed, useSelene } from '../store/useSelene';

function Field({ k, v }: { k: string; v: ReactNode }) {
  return (
    <div className="field">
      <span className="k">{k}</span>
      <span className="v">{v}</span>
    </div>
  );
}

const fmt = (x: number, d = 3) => x.toFixed(d);

export function ObjectPanel() {
  const id = useSelene((s) => s.selectedObjectId);
  const catalog = useSelene((s) => s.catalog);
  const events = useSelene((s) => s.events);
  const tSec = useSelene((s) => s.tSec);
  const select = useSelene((s) => s.selectObject);
  const { objects } = useScenarioFrame();

  const cat = catalog.find((c) => c.id === id) ?? null;
  const live = objects.find((o) => o.id === id) ?? null;

  if (!id) {
    return (
      <div>
        <h3 className="panel-title">Object</h3>
        <p className="muted">Select an object in the scene or from the list.</p>
        {catalog.length > 0 && (
          <ul className="events">
            {catalog.map((c) => (
              <li key={c.id} onClick={() => select(c.id)} style={{ ['--sev' as string]: c.kind === 'simulated' ? 'var(--sim)' : 'var(--accent)' }}>
                <span className="bar" />
                <div>
                  <div className="meta">
                    <span>{c.id}</span>
                    <span className="kind">{c.kind}</span>
                  </div>
                  <div className="text">
                    {c.name} <span className="muted">· {c.orbit_type}</span>
                  </div>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    );
  }

  const custody = live?.custody ?? 'held';
  const custodyTag = custody === 'held' ? 'ok' : custody === 'degraded' ? 'warn' : 'alert';
  const myEvents = events.filter((e) => e.object_id === id && e.t <= tSec);
  const lastObs = [...myEvents].reverse().find((e) => e.kind === 'observation' || e.kind === 'custody_regained');
  const maneuvers = myEvents.filter((e) => e.kind === 'maneuver_detected');
  const st = cat?.state_gcrf_km;

  return (
    <div>
      <h3 className="panel-title">
        Object <button className="toggle" style={{ float: 'right' }} onClick={() => select(null)}>✕</button>
      </h3>
      <Field k="ID" v={id} />
      <Field k="Name" v={cat?.name ?? '—'} />
      <Field
        k="Kind"
        v={cat?.kind === 'horizons' ? <span className="tag accent">HORIZONS</span> : <span className="tag sim">SIMULATED · notional actor</span>}
      />
      <Field k="Orbit type" v={cat?.orbit_type ?? '—'} />
      <Field k="Epoch (UTC)" v={cat?.epoch_utc ?? '—'} />
      <Field k="r GCRF [km]" v={st ? `${fmt(st[0], 1)}, ${fmt(st[1], 1)}, ${fmt(st[2], 1)}` : '—'} />
      <Field k="v GCRF [km/s]" v={st ? `${fmt(st[3], 4)}, ${fmt(st[4], 4)}, ${fmt(st[5], 4)}` : '—'} />
      <Field k="r rot [L*]" v={live ? `${fmt(live.pos[0], 4)}, ${fmt(live.pos[1], 4)}, ${fmt(live.pos[2], 4)}` : '—'} />
      <Field k="σ_pos (√tr P)" v={<span className="muted">— km (TODO: UKF covariance)</span>} />
      <Field k="Custody" v={<span className={`tag ${custodyTag}`}>{custody.toUpperCase()}</span>} />
      <Field k="Last obs" v={lastObs ? `T${fmtElapsed(lastObs.t)}` : '—'} />
      <Field k="Maneuvers" v={maneuvers.length === 0 ? 'none detected' : maneuvers.map((m) => `T${fmtElapsed(m.t)}`).join(', ')} />
      <h3 className="panel-title" style={{ marginTop: 14 }}>
        History
      </h3>
      {myEvents.length === 0 ? (
        <p className="muted">No events for this object yet.</p>
      ) : (
        <ul className="events">
          {myEvents.map((e, i) => (
            <li key={i} className="past" style={{ ['--sev' as string]: `var(--${e.severity === 'info' ? 'accent' : e.severity})` }}>
              <span className="bar" />
              <div>
                <div className="meta">
                  <span>T{fmtElapsed(e.t)}</span>
                  <span className="kind">{e.kind.replace(/_/g, ' ')}</span>
                </div>
                <div className="text">{e.text}</div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
