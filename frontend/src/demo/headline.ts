/**
 * Plain-English event headlines and the six story beats.
 *
 * The backend bundle (and the browser mock) carry analyst-grade `text` (residuals, NIS, χ² gates). Toasts, the
 * presenter caption and the feed headline need ≤ 12 words. When an event has no `headline` field one is derived
 * from its kind + object id + structured data (falling back to the first clause of `text`). Both the mock's kinds
 * (`tasking`, `brief`, `sim_truth`, `entered_region`, `maneuver_characterized`) and the backend's
 * (`tasking_update`, `brief_ready`, `maneuver`, `reachability_alert`, `maneuver_characterised`) are understood.
 */
import type { SeleneEvent } from '../api/types';

export type Beat = 'burn' | 'detected' | 'lost' | 'tasked' | 'regained' | 'brief';
export const BEATS: Beat[] = ['burn', 'detected', 'lost', 'tasked', 'regained', 'brief'];
export const BEAT_LABELS: Record<Beat, string> = { burn: 'Burn', detected: 'Detected', lost: 'Lost', tasked: 'Tasked', regained: 'Regained', brief: 'Brief' };

/** Which story beat an event kind marks (null for observations and other chatter). */
export function beatOf(kind: string): Beat | null {
  const k = kind.toLowerCase();
  if (k === 'sim_truth' || k === 'maneuver' || k === 'burn' || k === 'maneuver_truth') return 'burn';
  if (k === 'maneuver_detected') return 'detected';
  if (k === 'custody_lost') return 'lost';
  if (k === 'tasking' || k === 'tasking_update' || k === 'tasked') return 'tasked';
  if (k === 'custody_regained') return 'regained';
  if (k === 'brief' || k === 'brief_ready') return 'brief';
  return null;
}

/**
 * Negative-outcome variants the backend can emit (seed-dependent; also via POST /api/demo/rebuild?fast=true):
 * a `tasking_update` with `data.acquired === false` ("Tasked look by X did not acquire…", "No space observer could
 * re-acquire…") and a `maneuver_characterised` with `data.error` ("Maneuver NOT characterised: …"). Their headlines
 * must not assert the opposite, and a failed look is not the "Tasked" story beat (no pulse, no pip).
 */
export function isFailedOutcome(e: Pick<SeleneEvent, 'kind' | 'text' | 'data'>): boolean {
  const d = e.data ?? {};
  if (d.acquired === false) return true;
  if (typeof d.error === 'string' && d.error.trim()) return true;
  if (d.error === true) return true;
  const k = e.kind.toLowerCase();
  if ((k === 'tasking' || k === 'tasking_update') && /did not acquire|could not|no space observer|remains lost|not acquired/i.test(e.text)) return true;
  if ((k === 'maneuver_characterised' || k === 'maneuver_characterized') && /\bNOT characteri[sz]ed\b/i.test(e.text)) return true;
  return false;
}

/** Story beat of an EVENT: like beatOf(kind) but a failed tasked look / failed characterisation is not a beat. */
export function beatOfEvent(e: Pick<SeleneEvent, 'kind' | 'text' | 'data'>): Beat | null {
  const b = beatOf(e.kind);
  if (b === 'tasked' && isFailedOutcome(e)) return null;
  return b;
}

export function isBriefKind(kind: string): boolean {
  return beatOf(kind) === 'brief';
}
export function isTaskingKind(kind: string): boolean {
  return beatOf(kind) === 'tasked';
}

/** Human label for an event kind: "maneuver_detected" → "Maneuver detected". */
export function kindLabel(kind: string): string {
  const k = kind.toLowerCase();
  const table: Record<string, string> = {
    sim_truth: 'Simulation truth',
    maneuver: 'Burn (simulation truth)',
    maneuver_truth: 'Burn (simulation truth)',
    maneuver_detected: 'Maneuver detected',
    maneuver_characterized: 'Maneuver characterised',
    maneuver_characterised: 'Maneuver characterised',
    custody_nominal: 'Custody nominal',
    custody_degraded: 'Custody degraded',
    custody_lost: 'Custody lost',
    custody_regained: 'Custody regained',
    entered_region: 'Entered region',
    closest_approach: 'Closest approach (simulation truth)',
    reachability_alert: 'Reachability',
    tasking: 'Tasking',
    tasking_update: 'Tasking',
    observation: 'Observation',
    brief: 'Brief',
    brief_ready: 'Brief ready',
    info: 'Info',
    scenario: 'Scenario',
  };
  if (table[k]) return table[k];
  const s = k.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const num = (v: unknown): number | null => (typeof v === 'number' && Number.isFinite(v) ? v : typeof v === 'string' && v.trim() !== '' && Number.isFinite(Number(v)) ? Number(v) : null);
const str = (v: unknown): string | null => (typeof v === 'string' && v.trim() ? v.trim() : null);
const fmtKm = (v: number) => (v >= 100 ? Math.round(v).toLocaleString('en-US') : v >= 10 ? v.toFixed(0) : v.toFixed(1));

/** "GND-SOCORRO" → "Socorro", "siding_spring" → "Siding Spring", "dro_obs" → "DRO observer", "SPC-DRO-A" → "DRO observer A". */
export function sensorName(id: string): string {
  let s = id.replace(/^GND-/i, '').replace(/^SPC-/i, '');
  const obs = /^(.*?)[_-]?obs$/i.exec(s);
  if (obs) s = `${obs[1].toUpperCase().replace(/_/g, ' ')} observer`;
  else if (/^(dro|l1|l2|geo|nrho)[-_]/i.test(s)) {
    const m = /^(dro|l1|l2|geo|nrho)[-_](.*)$/i.exec(s)!;
    s = `${m[1].toUpperCase()} observer ${m[2].toUpperCase()}`;
  } else if (/^(geo)[_-](west|east)$/i.test(s)) {
    const m = /^(geo)[_-](west|east)$/i.exec(s)!;
    s = `GEO ${m[2].toLowerCase()}`;
  } else {
    s = s
      .toLowerCase()
      .split(/[_\s]+/)
      .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
      .join(' ');
  }
  return s;
}

/** "l1_gateway" → "L1 gateway", "nrho_corridor" → "NRHO corridor", "lunar_south_pole" → "lunar south pole". */
export function regionName(id: string): string {
  return id
    .replace(/[_-]+/g, ' ')
    .trim()
    .split(' ')
    .map((w) => (/^(l[1-5]|nrho|dro|geo|leo|xgeo)$/i.test(w) ? w.toUpperCase() : w.toLowerCase()))
    .join(' ');
}

/** Compact sensor name for KPI tiles: "geo_west" → "GEO W", "l1_halo_obs" → "L1 halo", "dro_obs" → "DRO", "siding_spring" → "Siding Spring". */
export function sensorShort(id: string): string {
  const s = id.replace(/^GND-/i, '').replace(/^SPC-/i, '');
  const gw = /^(geo)[_-](west|east)$/i.exec(s);
  if (gw) return `GEO ${gw[2].charAt(0).toUpperCase()}`;
  const obs = /^(.*?)[_-]?obs$/i.exec(s);
  if (obs) {
    const base = obs[1].toLowerCase();
    if (/^(dro|nrho|geo)$/.test(base)) return base.toUpperCase();
    return base.replace(/^(l[12])[_-]?/i, (_m, l: string) => `${l.toUpperCase()} `).replace(/_/g, ' ');
  }
  return sensorName(id);
}

/** First clause of a sentence, stripped of a leading "KIND:" label. */
function firstClause(text: string): string {
  let t = text.trim();
  t = t.replace(/^\[[^\]]*\]\s*/, ''); // "[SIMULATED truth] ..."
  t = t.replace(/^[A-Z][A-Z\s–—/()]*:\s*/, ''); // "MANEUVER DETECTED on X: ..." keeps X? no → only all-caps labels
  const m = /^(.*?)(?:[.;:]\s|\s[—–]\s|\s\(|$)/.exec(t);
  return (m ? m[1] : t).trim();
}

export function clampWords(s: string, max = 12): string {
  const tokens = s.trim().split(/\s+/).filter(Boolean);
  const isWord = (w: string) => /[\p{L}\p{N}]/u.test(w);
  let n = 0;
  for (let i = 0; i < tokens.length; i++) {
    if (isWord(tokens[i])) n++;
    if (n > max) return `${tokens.slice(0, i).join(' ').replace(/[\s\u2014\u2013·,;:-]+$/, '')}…`;
  }
  return tokens.join(' ');
}

/** ≤ 12-word plain-English headline for an event. */
export function deriveHeadline(e: Pick<SeleneEvent, 'kind' | 'text' | 'object_id' | 'data'> & { headline?: string }): string {
  if (e.headline && e.headline.trim()) return clampWords(e.headline.trim());
  const k = e.kind.toLowerCase();
  const obj = e.object_id ?? 'object';
  const d = e.data ?? {};
  const sensor = str(d.sensor_id);
  const sensorIds = Array.isArray(d.sensor_ids) ? (d.sensor_ids as unknown[]).filter((x): x is string => typeof x === 'string') : [];
  const sigma = num(d.sigma_km) ?? num(d.sigma_pos_km);
  const dv = num(d.dv_mps);
  const dvSig = num(d.dv_sigma_mps);
  let h: string | null = null;
  switch (k) {
    case 'maneuver_detected':
      h = `Maneuver detected on ${obj}${sensor ? ` by ${sensorName(sensor)}` : ''}`;
      break;
    case 'custody_lost': {
      // Say only what the event asserts: the σ threshold crossing and, when the data names it, why.
      const reason = str(d.reason);
      const why = reason === 'moon_exclusion' ? 'ground sites glare-blocked' : reason === 'no_site_available' ? 'no ground site available' : reason === 'sun_exclusion' ? 'Sun exclusion' : /no ground site|cannot observe/i.test(e.text) ? 'no ground site can observe' : null;
      h = `Custody lost on ${obj}${sigma !== null ? ` — σ ${fmtKm(sigma)} km` : ''}${why ? `, ${why}` : ''}`;
      break;
    }
    case 'custody_degraded': {
      const reason = str(d.reason);
      const why = reason === 'no_site_available' || reason === 'moon_exclusion' ? 'ground window closing' : /single .*tracklet/i.test(e.text) ? 'one post-burn tracklet' : null;
      h = `Custody degraded on ${obj}${sigma !== null ? ` — σ ${fmtKm(sigma)} km` : ''}${why ? `, ${why}` : ''}`;
      break;
    }
    case 'custody_regained':
      h = `Custody regained on ${obj}${sensor ? ` by ${sensorName(sensor)}` : ''}${sigma !== null ? ` — σ ${fmtKm(sigma)} km` : ''}`;
      break;
    case 'custody_nominal':
      h = `${obj} under routine custody`;
      break;
    case 'tasking':
    case 'tasking_update': {
      const n = sensorIds.length;
      const who = n > 1 ? `${n} space sensors` : sensor ? sensorName(sensor) : sensorIds[0] ? sensorName(sensorIds[0]) : 'a space observer';
      if (isFailedOutcome(e)) {
        // Negative outcome: never claim a redirect succeeded.
        h = /no space observer|could not/i.test(e.text) ? `No space observer could re-acquire ${obj} — custody remains lost` : `Tasked look by ${who} did not acquire ${obj}`;
      } else h = `Tasker redirects ${who} to ${str(d.target_id) ?? obj}`;
      break;
    }
    case 'entered_region': {
      // Actual (simulation-truth) entry into a sensitive region — not a reachability forecast. (An event of this
      // kind that carries a reachability `regions[]` payload is treated as the forecast it is.)
      if (Array.isArray(d.regions) && !str(d.region)) {
        h = deriveHeadline({ ...e, kind: 'reachability_alert', headline: undefined });
        break;
      }
      const region = str(d.region);
      const m = /enters?\s+(?:the\s+)?([^.;,(]+)/i.exec(e.text);
      const name = region ? regionName(region) : m ? m[1].trim() : 'a sensitive region';
      const dist = /\(([\d,.]+)\s*km from ([A-Z0-9]+)\)/.exec(e.text);
      h = `${obj} enters ${name}${dist ? ` — ${dist[1]} km from ${dist[2]}` : ''}`;
      break;
    }
    case 'closest_approach': {
      // Simulation-truth marker: how close the object really came to a region's anchor point (no transit implied).
      const region = str(d.region);
      const dist = num(d.distance_km);
      const where = region ? regionName(region).replace(/ gateway$/, '') : 'a region';
      h = `${obj} passes ${dist !== null ? `${fmtKm(dist)} km from ` : 'near '}${where}${d.transit === false ? ', no transit' : d.transit === true ? ' and transits' : ''}`;
      break;
    }
    case 'reachability_alert': {
      const regions = Array.isArray(d.regions) ? (d.regions as { name?: string; fraction?: number }[]) : [];
      const best = regions.filter((r) => typeof r.fraction === 'number' && r.fraction > 0).sort((a, b) => (b.fraction ?? 0) - (a.fraction ?? 0))[0];
      if (best?.name) h = `${obj} could reach ${best.name} (${Math.round(100 * (best.fraction ?? 0))}% of set)`;
      else {
        const m = /(\d+)\s*%[^.;]*?enters?\s+(?:the\s+)?([^.;,(]+)/i.exec(e.text);
        h = m ? `${obj} could reach ${m[2].trim()} (${m[1]}% of set)` : `Reachability: where ${obj} could go next`;
      }
      break;
    }
    case 'maneuver_characterized':
    case 'maneuver_characterised':
      if (isFailedOutcome(e)) h = `Maneuver on ${obj} not characterised`;
      // The 1σ is shown to the decimals it needs (0.03 m/s must not read as "± 0.0").
      else h = dv !== null ? `Maneuver characterised: Δv ${dv.toFixed(dvSig !== null && dvSig < 0.1 ? 2 : 1)}${dvSig !== null ? ` ± ${dvSig.toFixed(dvSig < 0.1 ? 2 : 1)}` : ''} m/s` : `Maneuver on ${obj} characterised`;
      break;
    case 'sim_truth':
    case 'maneuver':
    case 'maneuver_truth':
      h = dv !== null ? `Simulation truth: ${obj} burns ${dv.toFixed(0)} m/s, unannounced` : `Simulation truth: ${obj} performs an unannounced burn`;
      break;
    case 'observation': {
      const res = num(d.residual_arcsec);
      h = `${sensor ? sensorName(sensor) : 'Sensor'} tracklet on ${obj}${res !== null ? ` · residual ${res.toFixed(1)}″` : ''}`;
      if (d.reacquired) h = `${obj} re-acquired${sensor ? ` by ${sensorName(sensor)}` : ''}`;
      break;
    }
    case 'brief':
    case 'brief_ready':
      h = 'Analyst brief ready';
      break;
    case 'scenario':
      h = 'Scenario loaded';
      break;
    case 'info':
      if (/^scenario start/i.test(e.text)) h = `Scenario start: ${obj} quiet in custody`;
      else if (/conjunction/i.test(e.text)) {
        const m = /is\s+([\d,]+)\s*km/.exec(e.text);
        h = m ? `Conjunction screen: closest approach ${m[1]} km` : 'Conjunction screening complete';
      }
      break;
    default:
      h = null;
  }
  if (!h) h = firstClause(e.text) || kindLabel(e.kind);
  return clampWords(h);
}

/** Normalise a backend custody string ('CUSTODY' | 'DEGRADED' | 'LOST' | 'held' …) to the UI union. */
export function normCustody(v: unknown): 'held' | 'degraded' | 'lost' | 'unknown' {
  const s = String(v ?? '').toLowerCase();
  if (s === 'held' || s === 'custody' || s === 'nominal' || s === 'ok') return 'held';
  if (s === 'degraded') return 'degraded';
  if (s === 'lost') return 'lost';
  return 'unknown';
}
