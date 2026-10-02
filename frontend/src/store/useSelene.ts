/**
 * Global UI state (zustand). Time is kept as `tSec` = seconds since `t0Iso` (scenario start, UTC).
 * Frame mode drives the 3D scene: 'rotating' = Earth–Moon synodic frame, 'inertial' = GCRF-like view
 * (rotating group turned about +z by the Earth–Moon line angle, aligned with the rotating frame at t0).
 */
import { create } from 'zustand';
import type { CatalogMeta, CatalogObject, DemoScenario, EphemerisBodies, EventSeverity, OrbitFamilies, OrbitRecord, SeleneEvent, Sensors } from '../api/types';
import { setEphemeris as installEphemeris } from '../lib/ephem';

export type FrameMode = 'rotating' | 'inertial';

export interface Layers {
  families: boolean;
  lagrange: boolean;
  trails: boolean;
  clouds: boolean;
  fov: boolean;
  exclusion: boolean;
  grid: boolean;
  sites: boolean;
  reach: boolean;
  labels: boolean;
}

/** Preset playback speeds (sim seconds per wall-clock second). The scripted demo may use another value. */
export const SPEEDS: number[] = [1, 60, 600, 3600, 10000];
export type PlaybackSpeed = number;

export type DockTab = 'object' | 'events' | 'analysis' | 'brief';
export const DOCK_TABS: DockTab[] = ['object', 'events', 'analysis', 'brief'];
export const isDockTab = (v: unknown): v is DockTab => typeof v === 'string' && (DOCK_TABS as string[]).includes(v);

export interface Toast {
  id: number;
  /** ≤ 12-word plain-English line (what the toast shows); `text` is the full analyst detail (tooltip). */
  headline: string;
  text: string;
  severity: EventSeverity;
  kind: string;
  objectId?: string;
  t: number;
}

export interface SeleneState {
  frame: FrameMode;
  /** ISO UTC epoch corresponding to tSec = 0. */
  t0Iso: string;
  tSec: number;
  t0Sec: number;
  t1Sec: number;
  playing: boolean;
  speed: PlaybackSpeed;
  selectedObjectId: string | null;
  /** Who made the last selection: a user click/tap ('user') or the demo driver ('auto'). The dock only jumps to the
   *  Object tab for user selections, so the scripted story can keep the Events feed in front. */
  selectSource: 'user' | 'auto';
  layers: Layers;
  events: SeleneEvent[];
  catalog: CatalogObject[];
  /** Live catalog envelope (epoch, counts, disclaimer); null in mock mode. */
  catalogMeta: CatalogMeta | null;
  /** Epoch the idle timeline starts at: the live catalog epoch when known, else DEFAULT_T0_ISO. */
  baseT0Iso: string;
  sensors: Sensors | null;
  /** Orbit-library records the space observers ride (sensor id → record), fetched live. */
  observerOrbits: Record<string, OrbitRecord>;
  families: OrbitFamilies | null;
  highlightFamily: string | null;
  /** Families switched off in the legend. */
  hiddenFamilies: string[];
  /** True once the user has touched the family legend: the default "quiet" set is then no longer re-applied. */
  familiesTouched: boolean;
  ephemeris: EphemerisBodies | null;
  scenario: DemoScenario | null;
  brief: string;
  dockTab: DockTab;
  toasts: Toast[];
  /** Set once the scripted scenario has reached its end (dock switched to Brief). */
  scenarioFinished: boolean;
  /**
   * Presenter narration: story events the cursor has crossed, shown ONE AT A TIME (caption + toast) with a wall-clock
   * dwell per event, so coincident beats (detected → degraded → reachability at the same epoch) and beats that are
   * seconds apart at story speed are each readable. While `captionQueue` is non-empty or a story-beat caption is
   * within its dwell, scripted playback crawls instead of racing on (see hooks/usePlayback.ts).
   */
  captionQueue: SeleneEvent[];
  caption: SeleneEvent | null;
  /** performance.now() at which the current caption's dwell ends. */
  captionUntilMs: number;

  setFrame: (f: FrameMode) => void;
  setT: (t: number) => void;
  /** Advance by dt seconds (clamped to [t0Sec, t1Sec]; stops playback at the end). */
  advance: (dtSec: number) => void;
  setRange: (t0Sec: number, t1Sec: number, t0Iso?: string) => void;
  setPlaying: (p: boolean) => void;
  togglePlaying: () => void;
  setSpeed: (s: PlaybackSpeed) => void;
  selectObject: (id: string | null, source?: 'user' | 'auto') => void;
  toggleLayer: (k: keyof Layers) => void;
  setLayer: (k: keyof Layers, v: boolean) => void;
  setEvents: (e: SeleneEvent[]) => void;
  setCatalog: (c: CatalogObject[], meta?: CatalogMeta | null) => void;
  /** Set the idle timeline base epoch (and move the timeline there unless a scenario is loaded). */
  setBaseEpoch: (iso: string, spanS?: number) => void;
  setSensors: (s: Sensors | null) => void;
  setObserverOrbits: (o: Record<string, OrbitRecord>) => void;
  setFamilies: (f: OrbitFamilies | null) => void;
  setHighlightFamily: (name: string | null) => void;
  toggleFamilyVisible: (name: string) => void;
  setFamilyVisible: (name: string, v: boolean) => void;
  setEphemeris: (e: EphemerisBodies | null) => void;
  setBrief: (b: string) => void;
  setDockTab: (t: DockTab) => void;
  pushToast: (t: Omit<Toast, 'id'>) => void;
  dismissToast: (id: number) => void;
  setScenarioFinished: (v: boolean) => void;
  /** Queue story events for narration (chronological; duplicates by t+kind are dropped). */
  enqueueCaptions: (evs: SeleneEvent[], replace?: boolean) => void;
  /** Advance the narration: pops the next queued event when the current dwell is over. Returns the event shown. */
  tickCaptions: (nowMs: number) => SeleneEvent | null;
  clearCaptions: () => void;
  /** Jump the cursor to the next (or previous) story event — observations are skipped. Returns the event, if any. */
  skipToEvent: (dir: 1 | -1) => SeleneEvent | null;
  /** Load a (normalised) demo scenario: sets range, events, brief and rewinds to t0. */
  loadScenario: (s: DemoScenario) => void;
  reset: () => void;
}

export const DEFAULT_T0_ISO = '2026-10-01T00:00:00Z';
export const DEFAULT_SPAN_S = 7 * 86400;
const TOAST_MS = 7000;
let toastSeq = 1;

export const useSelene = create<SeleneState>((set, get) => ({
  frame: 'rotating',
  t0Iso: DEFAULT_T0_ISO,
  tSec: 0,
  t0Sec: 0,
  t1Sec: DEFAULT_SPAN_S,
  playing: false,
  speed: 600,
  selectedObjectId: null,
  selectSource: 'user',
  layers: { families: true, lagrange: true, trails: true, clouds: true, fov: true, exclusion: false, grid: true, sites: true, reach: true, labels: true },
  events: [],
  catalog: [],
  catalogMeta: null,
  baseT0Iso: DEFAULT_T0_ISO,
  sensors: null,
  observerOrbits: {},
  families: null,
  highlightFamily: null,
  hiddenFamilies: [],
  familiesTouched: false,
  ephemeris: null,
  scenario: null,
  brief: '',
  dockTab: 'object',
  toasts: [],
  scenarioFinished: false,
  captionQueue: [],
  caption: null,
  captionUntilMs: 0,

  setFrame: (frame) => set({ frame }),
  setT: (t) => {
    const { t0Sec, t1Sec } = get();
    set({ tSec: Math.min(t1Sec, Math.max(t0Sec, t)) });
  },
  advance: (dt) => {
    const { tSec, t1Sec, t0Sec } = get();
    const next = tSec + dt;
    if (next >= t1Sec) set({ tSec: t1Sec, playing: false });
    else set({ tSec: Math.max(t0Sec, next) });
  },
  setRange: (t0Sec, t1Sec, t0Iso) => set((s) => ({ t0Sec, t1Sec, t0Iso: t0Iso ?? s.t0Iso, tSec: Math.min(Math.max(s.tSec, t0Sec), t1Sec) })),
  setPlaying: (playing) => set({ playing }),
  togglePlaying: () => {
    const { playing, tSec, t1Sec, t0Sec } = get();
    // Pressing play at the end rewinds.
    if (!playing && tSec >= t1Sec) set({ tSec: t0Sec, playing: true, scenarioFinished: false });
    else set({ playing: !playing });
  },
  setSpeed: (speed) => set({ speed }),
  selectObject: (selectedObjectId, source = 'user') => set({ selectedObjectId, selectSource: source }),
  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setLayer: (k, v) => set((s) => ({ layers: { ...s.layers, [k]: v } })),
  setEvents: (events) => set({ events: [...events].sort((a, b) => a.t - b.t) }),
  setCatalog: (catalog, meta) => set((s) => ({ catalog, catalogMeta: meta === undefined ? s.catalogMeta : meta })),
  setBaseEpoch: (iso, spanS = DEFAULT_SPAN_S) =>
    set((s) => {
      if (s.scenario) return { baseT0Iso: iso };
      return { baseT0Iso: iso, t0Iso: iso, t0Sec: 0, t1Sec: spanS, tSec: Math.min(Math.max(s.tSec, 0), spanS) };
    }),
  setSensors: (sensors) => set({ sensors }),
  setObserverOrbits: (observerOrbits) => set({ observerOrbits }),
  setFamilies: (families) =>
    set((s) => {
      // Default view keeps the four families that read as "the cislunar highways" (L1 halo N, L2 halo S incl. the
      // NRHOs, DRO); the Lyapunov and resonant families stay available in the legend. Applied until the user
      // touches the legend.
      if (!families || s.familiesTouched) return { families };
      const keep = (name: string) => {
        const n = name.toLowerCase();
        if (n.includes('dro')) return true;
        if (n.includes('nrho')) return true;
        if (n.includes('halo') && n.includes('l1') && !n.includes('_s')) return true;
        if (n.includes('halo') && n.includes('l2') && !n.includes('_n')) return true;
        return false;
      };
      return { families, hiddenFamilies: families.families.map((f) => f.name).filter((n) => !keep(n)) };
    }),
  setHighlightFamily: (highlightFamily) => set({ highlightFamily }),
  toggleFamilyVisible: (name) => set((s) => ({ familiesTouched: true, hiddenFamilies: s.hiddenFamilies.includes(name) ? s.hiddenFamilies.filter((n) => n !== name) : [...s.hiddenFamilies, name] })),
  setFamilyVisible: (name, v) => set((s) => ({ familiesTouched: true, hiddenFamilies: v ? s.hiddenFamilies.filter((n) => n !== name) : s.hiddenFamilies.includes(name) ? s.hiddenFamilies : [...s.hiddenFamilies, name] })),
  setEphemeris: (ephemeris) => {
    installEphemeris(ephemeris);
    set({ ephemeris });
  },
  setBrief: (brief) => set({ brief }),
  setDockTab: (dockTab) => set({ dockTab }),
  pushToast: (t) => {
    const id = toastSeq++;
    // At most two toasts on screen (bottom-right lane): the newest two win.
    set((s) => ({ toasts: [...s.toasts.slice(-1), { ...t, id }] }));
    setTimeout(() => get().dismissToast(id), TOAST_MS);
  },
  dismissToast: (id) => set((s) => (s.toasts.some((t) => t.id === id) ? { toasts: s.toasts.filter((t) => t.id !== id) } : s)),
  setScenarioFinished: (scenarioFinished) => set({ scenarioFinished }),
  enqueueCaptions: (evs, replace = false) =>
    set((s) => {
      const base = replace ? [] : s.captionQueue;
      const key = (e: SeleneEvent) => `${e.t}|${e.kind}|${e.object_id ?? ''}`;
      const seen = new Set(base.map(key));
      if (!replace && s.caption) seen.add(key(s.caption));
      const add = evs.filter((e) => !seen.has(key(e)));
      if (add.length === 0 && !replace) return s;
      return { captionQueue: [...base, ...add].slice(-CAPTION_QUEUE_MAX), ...(replace ? { caption: null, captionUntilMs: 0 } : {}) };
    }),
  tickCaptions: (nowMs) => {
    const s = get();
    if (s.caption && nowMs < s.captionUntilMs) return null;
    const next = s.captionQueue[0];
    if (!next) {
      // A paused presenter keeps the current caption on screen; playback lets it fade after its dwell.
      if (s.caption && s.playing) set({ caption: null });
      return null;
    }
    set({ caption: next, captionQueue: s.captionQueue.slice(1), captionUntilMs: nowMs + captionDwellMs(next, s.captionQueue.length) });
    return next;
  },
  clearCaptions: () => set((s) => (s.caption || s.captionQueue.length ? { caption: null, captionQueue: [], captionUntilMs: 0 } : s)),
  skipToEvent: (dir) => {
    const { events, tSec, t0Sec, t1Sec } = get();
    const story = events.filter((e) => e.kind !== 'observation');
    // Forward: first story event strictly after the cursor; backward: the last one at least a minute before it
    // (so repeated presses step through coincident beats one group at a time).
    const target = dir > 0 ? story.find((e) => e.t > tSec + 0.5) : [...story].reverse().find((e) => e.t < tSec - 60);
    if (!target) {
      if (dir > 0) set({ tSec: t1Sec });
      else set({ tSec: t0Sec });
      return null;
    }
    // Land a hair past the event so it counts as reached (crossing registers for the driver: layers, selection).
    set({ tSec: Math.min(t1Sec, Math.max(t0Sec, target.t + 0.5)), captionQueue: [], caption: null, captionUntilMs: 0 });
    return target;
  },
  loadScenario: (scenario) =>
    set({
      scenario,
      t0Iso: scenario.meta.t0_utc,
      t0Sec: 0,
      t1Sec: scenario.meta.duration_s,
      tSec: 0,
      events: [...scenario.events].sort((a, b) => a.t - b.t),
      brief: scenario.brief,
      playing: false,
      scenarioFinished: false,
      toasts: [],
      captionQueue: [],
      caption: null,
      captionUntilMs: 0,
    }),
  reset: () =>
    set((s) => ({
      t0Iso: s.baseT0Iso,
      t0Sec: 0,
      t1Sec: DEFAULT_SPAN_S,
      tSec: 0,
      playing: false,
      speed: 600,
      events: [],
      scenario: null,
      brief: '',
      selectedObjectId: null,
      dockTab: 'object',
      toasts: [],
      scenarioFinished: false,
      highlightFamily: null,
      captionQueue: [],
      caption: null,
      captionUntilMs: 0,
    })),
}));

const CAPTION_QUEUE_MAX = 8;

/** Wall-clock dwell per narrated event: story beats and alerts 4 s, warnings 3 s, info 2.5 s (shorter when queued up). */
export function captionDwellMs(e: SeleneEvent, queued = 0): number {
  const k = e.kind.toLowerCase();
  const beat = /^(sim_truth|maneuver|burn|maneuver_truth|maneuver_detected|custody_lost|tasking|tasking_update|tasked|custody_regained|brief|brief_ready)$/.test(k);
  const base = beat || e.severity === 'alert' ? 4000 : e.severity === 'warn' ? 3000 : 2500;
  return queued >= 4 ? Math.round(base * 0.7) : base;
}

/** True while scripted playback should crawl so the audience can read the current narration. */
export function narrationHold(s: Pick<SeleneState, 'scenario' | 'speed' | 'captionQueue' | 'caption' | 'captionUntilMs'>, nowMs: number): boolean {
  if (!s.scenario || s.speed < 1000) return false;
  if (s.captionQueue.length > 0) return true;
  return !!s.caption && nowMs < s.captionUntilMs && s.caption.kind !== 'observation';
}

/** Current UTC Date for the time cursor. */
export function cursorDate(t0Iso: string, tSec: number): Date {
  return new Date(Date.parse(t0Iso) + tSec * 1000);
}

/** Current UTC epoch in ms (imperative helper for render loops). */
export function cursorMs(): number {
  const { t0Iso, tSec } = useSelene.getState();
  return Date.parse(t0Iso) + tSec * 1000;
}

/** Format as "YYYY-MM-DD HH:MM:SS UTC". */
export function fmtUtc(d: Date): string {
  if (Number.isNaN(d.getTime())) return '—';
  return d.toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ' UTC');
}

/** Format seconds as "+DDd HH:MM:SS". */
export function fmtElapsed(s: number): string {
  const sign = s < 0 ? '-' : '+';
  s = Math.abs(Math.floor(s));
  const d = Math.floor(s / 86400);
  const h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${sign}${d}d ${pad(h)}:${pad(m)}:${pad(sec)}`;
}

/** Short duration "3h 12m" / "45m" / "2d 4h". */
export function fmtAge(s: number): string {
  if (!Number.isFinite(s) || s < 0) return '—';
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60);
  if (d > 0) return `${d}d ${h}h`;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m`;
  return `${m}m`;
}
