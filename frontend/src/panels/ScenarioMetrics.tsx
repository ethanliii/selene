/**
 * "Scenario metrics" card: the backend run metrics of the loaded bundle (scenario.metrics) revealed as the story
 * reaches them, so the card never spoils the outcome: custody % so far (custody_timeline up to the cursor), σ_max so
 * far (sigma_timeline), detection latency once the detection has happened, time to regain once regained, Δv truth
 * (after the simulation-truth burn marker) vs estimate (after the characterisation). Values are the engines' own.
 */
import { fmtAge, useSelene } from '../store/useSelene';

/** Bundle metrics as served: timelines carry `t_rel_s` (newer bundles) or `t_s` (older), event epochs come as
 *  `*_utc` plus optional `*_rel_s` / `*_s`; `relS()` normalises all of them to seconds since the scenario t0. */
interface Metrics {
  custody_timeline?: { t_s?: number; t_rel_s?: number; t_utc?: string; status: string }[];
  sigma_timeline?: { t_s?: number; t_rel_s?: number; t_utc?: string; sigma_km: number }[];
  t_burn_rel_s?: number;
  t_detect_rel_s?: number;
  t_lost_rel_s?: number;
  t_regained_rel_s?: number;
  t_burn_utc?: string;
  t_detect_utc?: string;
  t_lost_utc?: string;
  t_regained_utc?: string;
  custody_pct?: number;
  t_burn_s?: number;
  t_detect_s?: number;
  detection_latency_h?: number;
  detect_sensor?: string;
  detect_nis?: number;
  t_lost_s?: number;
  t_regained_s?: number;
  regained_after_h?: number;
  regained_after_burn_h?: number;
  max_sigma_km?: number;
  sigma_at_detection_km?: number;
  dv_true_mps?: number;
  dv_est_mps?: number;
  dv_est_sigma_mps?: number;
  dv_est_err_pct?: number;
  dir_err_deg?: number;
  t_characterised_utc?: string;
}

const fmtKm = (v: number) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v.toFixed(1));
const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined);
/** Seconds since t0 from whichever field the bundle carries. */
const relS = (t0Ms: number, rel?: number, abs?: number, utc?: string): number | undefined => rel ?? abs ?? (utc ? (Date.parse(utc) - t0Ms) / 1000 : undefined);

export function ScenarioMetrics() {
  const scenario = useSelene((s) => s.scenario);
  const tSec = useSelene((s) => Math.floor(s.tSec / 600) * 600);
  const events = useSelene((s) => s.events);
  const m = (scenario?.metrics ?? null) as Metrics | null;
  if (!scenario || !m) return null;
  const t0Ms = Date.parse(scenario.meta.t0_utc);
  const tOf = (x: { t_s?: number; t_rel_s?: number; t_utc?: string }) => relS(t0Ms, num(x.t_rel_s), num(x.t_s), x.t_utc) ?? Infinity;
  const ct = m.custody_timeline ?? [];
  const past = ct.filter((x) => tOf(x) <= tSec);
  const pct = past.length ? (100 * past.filter((x) => x.status === 'CUSTODY').length) / past.length : null;
  const sig = (m.sigma_timeline ?? []).filter((x) => tOf(x) <= tSec);
  const sigMax = sig.length ? Math.max(...sig.map((x) => x.sigma_km)) : null;
  const tDetect = relS(t0Ms, num(m.t_detect_rel_s), num(m.t_detect_s), m.t_detect_utc);
  const tRegained = relS(t0Ms, num(m.t_regained_rel_s), num(m.t_regained_s), m.t_regained_utc);
  const tBurn = relS(t0Ms, num(m.t_burn_rel_s), num(m.t_burn_s), m.t_burn_utc);
  const tLost = relS(t0Ms, num(m.t_lost_rel_s), num(m.t_lost_s), m.t_lost_utc);
  const detected = tDetect !== undefined && tSec >= tDetect;
  const regained = tRegained !== undefined && tSec >= tRegained;
  const burnShown = tBurn !== undefined && tSec >= tBurn;
  const lost = tLost !== undefined && tSec >= tLost;
  const charEv = events.find((e) => /characteri[sz]ed/.test(e.kind));
  const characterised = charEv ? tSec >= charEv.t : false;
  const mock = scenario.meta.source === 'browser-mock';
  return (
    <>
      <h3 className="panel-title" style={{ marginTop: 12 }}>
        Scenario metrics <span className={`tag ${mock ? 'warn' : 'live-badge'}`}>{mock ? 'OFFLINE MOCK' : 'LIVE'}</span>
      </h3>
      <div className="an-tiles metrics">
        <div className={`an-tile ${pct === null ? 'muted' : pct >= 90 ? 'ok' : pct >= 60 ? 'warn' : 'alert'}`} title="Fraction of hourly frames so far in which the protagonist's cloud σ_pos is below the custody threshold">
          <div className="k">Custody so far</div>
          <div className="v mono">{pct === null ? '—' : `${pct.toFixed(0)}%`}</div>
          <div className="s">{past.length} of {ct.length} frames</div>
        </div>
        <div className={`an-tile ${sigMax === null ? 'muted' : sigMax >= 1000 ? 'alert' : sigMax >= 100 ? 'warn' : 'ok'}`} title="Largest particle-cloud σ_pos reached so far">
          <div className="k">σ max so far</div>
          <div className="v mono">{sigMax === null ? '—' : `${fmtKm(sigMax)} km`}</div>
          <div className="s">{m.max_sigma_km !== undefined && sigMax !== null && sigMax >= m.max_sigma_km - 1e-6 ? 'peak of the story' : 'still evolving'}</div>
        </div>
        <div className={`an-tile ${detected ? (m.detection_latency_h !== undefined && m.detection_latency_h <= 3 ? 'ok' : 'warn') : 'muted'}`} title="Time from the (SIMULATED) burn to the declared detection">
          <div className="k">Detect latency</div>
          <div className="v mono">{detected && m.detection_latency_h !== undefined ? `${m.detection_latency_h.toFixed(1)} h` : '—'}</div>
          <div className="s">{detected ? `${m.detect_sensor ?? ''} · NIS ${m.detect_nis !== undefined ? Math.round(m.detect_nis).toLocaleString('en-US') : ''}` : 'no detection yet'}</div>
        </div>
        <div className={`an-tile ${regained ? 'ok' : lost ? 'alert' : 'muted'}`} title="Time from the custody-loss declaration to re-acquisition by the tasked observers">
          <div className="k">Regain after loss</div>
          <div className="v mono">{regained && m.regained_after_h !== undefined ? fmtAge(m.regained_after_h * 3600) : '—'}</div>
          <div className="s">{regained ? `${(m.regained_after_burn_h ?? 0).toFixed(1)} h after the burn` : lost ? 'custody lost' : 'not lost'}</div>
        </div>
        <div className={`an-tile ${characterised ? 'ok' : 'muted'}`} title="Δv of the SIMULATED burn (truth) vs the estimate from the post-burn tracklets">
          <div className="k">Δv truth · estimate</div>
          <div className="v mono">
            {burnShown && m.dv_true_mps !== undefined ? `${m.dv_true_mps.toFixed(0)}` : '—'}
            {' · '}
            {characterised && m.dv_est_mps !== undefined ? `${m.dv_est_mps.toFixed(1)}${m.dv_est_sigma_mps !== undefined ? ` ± ${m.dv_est_sigma_mps.toFixed(2)}` : ''}` : '—'} <small>m/s</small>
          </div>
          <div className="s">{characterised ? `error ${m.dv_est_err_pct !== undefined ? `${m.dv_est_err_pct > 0 ? '+' : ''}${m.dv_est_err_pct.toFixed(2)}%` : ''}${m.dir_err_deg !== undefined ? ` · dir ${m.dir_err_deg.toFixed(2)}°` : ''}` : burnShown ? 'estimate pending' : 'no burn yet'}</div>
        </div>
      </div>
    </>
  );
}
