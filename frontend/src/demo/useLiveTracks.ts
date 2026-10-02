/**
 * Live object motion: fetch `/api/catalog/objects/{id}/trajectory` for every catalog object over the visible
 * timeline span (rotating frame, nondimensional) and keep the resulting tracks in a small module cache keyed by
 * (object, frame, span, n). Sampling density follows the regime: lunar orbiters (periods of hours) get a 150 s
 * grid, everything else 600 s, capped at the backend's n ≤ 5000. Requests run through a 6-wide queue so a
 * catalog of ~20 objects loads in a few hundred ms against the local backend. Nothing here is computed in the
 * browser beyond Hermite interpolation (lib/tracks.ts).
 */
import { useEffect, useRef, useState } from 'react';
import { api } from '../api/client';
import type { CatalogObject, TrajectoryFrame } from '../api/types';
import { trackFromTrajectory, type Track } from '../lib/tracks';
import { useSelene } from '../store/useSelene';

const DT_LUNAR_S = 150;
const DT_DEFAULT_S = 600;
const MAX_N = 5000;
const CONCURRENCY = 6;

/** Lunar orbiters / low selenocentric objects need a much finer sample grid than the cislunar population. */
export function isLunarOrbiter(c: CatalogObject): boolean {
  if (c.regime === 'lunar_orbit') return true;
  if (c.orbit_ref === 'keplerian_moon') return true;
  if (typeof c.selenocentric_range_km === 'number' && c.selenocentric_range_km < 30000) return true;
  return false;
}

/** Number of samples for an object over `spanS` seconds. */
export function samplesFor(c: CatalogObject, spanS: number): number {
  const dt = isLunarOrbiter(c) ? DT_LUNAR_S : DT_DEFAULT_S;
  return Math.min(MAX_N, Math.max(2, Math.floor(spanS / dt) + 1));
}

/** Trail length shown behind an object [s]. */
export function trailSpanFor(c: CatalogObject | undefined): number {
  return c && isLunarOrbiter(c) ? 2 * 3600 : 24 * 3600;
}

// ---- cache + queue -----------------------------------------------------------------------------
/** Small LRU keyed by (object, frame, span, n): at most ~3 spans of a 20-object catalog are kept. */
const cache = new Map<string, Promise<Track>>();
const CACHE_MAX = 64;
function remember(key: string, p: Promise<Track>) {
  cache.set(key, p);
  while (cache.size > CACHE_MAX) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}
let active = 0;
const waiting: (() => void)[] = [];
function slot(): Promise<void> {
  if (active < CONCURRENCY) {
    active++;
    return Promise.resolve();
  }
  return new Promise<void>((r) => waiting.push(r)).then(() => {
    active++;
  });
}
function release() {
  active--;
  const next = waiting.shift();
  if (next) next();
}

export function trackKey(id: string, frame: TrajectoryFrame, t0Ms: number, t1Ms: number, n: number, originMs: number): string {
  return `${id}|${frame}|${t0Ms}|${t1Ms}|${n}|${originMs}`;
}

/**
 * Fetch (or reuse) a track; the promise is cached so concurrent callers share one request. Sample times are stored
 * relative to `originMs` = the store's timeline base (t0Iso), the same origin as `tSec`, so a non-zero window
 * start (t0Sec) cannot shift the evaluated positions.
 */
export function fetchTrack(id: string, frame: TrajectoryFrame, t0Ms: number, t1Ms: number, n: number, originMs = t0Ms): Promise<Track> {
  const key = trackKey(id, frame, t0Ms, t1Ms, n, originMs);
  let p = cache.get(key);
  if (!p) {
    p = slot()
      .then(() => api.trajectory(id, new Date(t0Ms).toISOString(), new Date(t1Ms).toISOString(), n, frame))
      .then((resp) => trackFromTrajectory(resp, originMs))
      .finally(release);
    remember(key, p);
    p.catch(() => cache.delete(key));
  }
  return p;
}

export interface TrackSet {
  /** object id → rotating-frame track for the current span. */
  tracks: Map<string, Track>;
  /** object id → error text for objects whose trajectory could not be served in this window. */
  errors: Map<string, string>;
  loaded: number;
  total: number;
}

const EMPTY: TrackSet = { tracks: new Map(), errors: new Map(), loaded: 0, total: 0 };

/**
 * Rotating-frame tracks for the whole catalog over the current timeline span. `enabled` should be false in mock
 * mode (no trajectory endpoint) and while a scenario is loaded (frames come from the scenario).
 */
export function useLiveTracks(enabled: boolean): TrackSet {
  const catalog = useSelene((s) => s.catalog);
  const t0Iso = useSelene((s) => s.t0Iso);
  const t0Sec = useSelene((s) => s.t0Sec);
  const t1Sec = useSelene((s) => s.t1Sec);
  const [set, setSet] = useState<TrackSet>(EMPTY);
  const gen = useRef(0);

  useEffect(() => {
    if (!enabled || catalog.length === 0) {
      setSet(EMPTY);
      return;
    }
    const my = ++gen.current;
    const base = Date.parse(t0Iso);
    const t0Ms = base + t0Sec * 1000;
    const t1Ms = base + t1Sec * 1000;
    const spanS = (t1Ms - t0Ms) / 1000;
    const tracks = new Map<string, Track>();
    const errors = new Map<string, string>();
    const total = catalog.length;
    let loaded = 0;
    let scheduled = false;
    const publish = () => {
      if (scheduled) return;
      scheduled = true;
      // Batch arrivals into a single React update: one animation frame when visible, a macrotask when the tab is
      // hidden (browsers pause rAF in background tabs, which would hold the tracks until the tab is foregrounded).
      const flush = () => {
        scheduled = false;
        if (gen.current === my) setSet({ tracks: new Map(tracks), errors: new Map(errors), loaded, total });
      };
      if (typeof document !== 'undefined' && document.hidden) setTimeout(flush, 0);
      else requestAnimationFrame(flush);
    };
    // Publish copies only: the working maps above are mutated as responses arrive.
    setSet({ tracks: new Map(), errors: new Map(), loaded: 0, total });
    for (const c of catalog) {
      const n = samplesFor(c, spanS);
      fetchTrack(c.id, 'rot_nd', t0Ms, t1Ms, n, base)
        .then((tr) => {
          if (gen.current !== my) return;
          tracks.set(c.id, tr);
        })
        .catch((e) => {
          if (gen.current !== my) return;
          const msg = e instanceof Error ? e.message : String(e);
          errors.set(c.id, msg.replace(/^API \d+ \S+: /, '').slice(0, 160));
        })
        .finally(() => {
          if (gen.current !== my) return;
          loaded++;
          publish();
        });
    }
    return () => {
      gen.current++;
    };
  }, [enabled, catalog, t0Iso, t0Sec, t1Sec]);

  return set;
}

/** One object's track in a given frame over the current span (for the Object panel's GCRF state). */
export function useObjectTrack(id: string | null, frame: TrajectoryFrame, enabled: boolean): { track: Track | null; error: string | null } {
  const catalog = useSelene((s) => s.catalog);
  const t0Iso = useSelene((s) => s.t0Iso);
  const t0Sec = useSelene((s) => s.t0Sec);
  const t1Sec = useSelene((s) => s.t1Sec);
  const [state, setState] = useState<{ key: string; track: Track | null; error: string | null }>({ key: '', track: null, error: null });
  const c = id ? catalog.find((x) => x.id === id) : undefined;
  const base = Date.parse(t0Iso);
  const t0Ms = base + t0Sec * 1000;
  const t1Ms = base + t1Sec * 1000;
  const n = c ? samplesFor(c, (t1Ms - t0Ms) / 1000) : 0;
  const key = c && enabled ? trackKey(c.id, frame, t0Ms, t1Ms, n, base) : '';
  useEffect(() => {
    if (!key || !c) return;
    let alive = true;
    fetchTrack(c.id, frame, t0Ms, t1Ms, n, base)
      .then((track) => alive && setState({ key, track, error: null }))
      .catch((e) => alive && setState({ key, track: null, error: e instanceof Error ? e.message : String(e) }));
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  if (!key) return { track: null, error: null };
  return state.key === key ? { track: state.track, error: state.error } : { track: null, error: null };
}
