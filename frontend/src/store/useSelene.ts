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

export type DockTab = 'object' | 'events' | 'brief';

export interface Toast {
  id: number;
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
  ephemeris: EphemerisBodies | null;
  scenario: DemoScenario | null;
  brief: string;
  dockTab: DockTab;
  toasts: Toast[];
  /** Set once the scripted scenario has reached its end (dock switched to Brief). */
  scenarioFinished: boolean;

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
  ephemeris: null,
  scenario: null,
  brief: '',
  dockTab: 'events',
  toasts: [],
  scenarioFinished: false,

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
  setFamilies: (families) => set({ families }),
  setHighlightFamily: (highlightFamily) => set({ highlightFamily }),
  toggleFamilyVisible: (name) => set((s) => ({ hiddenFamilies: s.hiddenFamilies.includes(name) ? s.hiddenFamilies.filter((n) => n !== name) : [...s.hiddenFamilies, name] })),
  setFamilyVisible: (name, v) => set((s) => ({ hiddenFamilies: v ? s.hiddenFamilies.filter((n) => n !== name) : s.hiddenFamilies.includes(name) ? s.hiddenFamilies : [...s.hiddenFamilies, name] })),
  setEphemeris: (ephemeris) => {
    installEphemeris(ephemeris);
    set({ ephemeris });
  },
  setBrief: (brief) => set({ brief }),
  setDockTab: (dockTab) => set({ dockTab }),
  pushToast: (t) => {
    const id = toastSeq++;
    set((s) => ({ toasts: [...s.toasts.slice(-4), { ...t, id }] }));
    setTimeout(() => get().dismissToast(id), TOAST_MS);
  },
  dismissToast: (id) => set((s) => (s.toasts.some((t) => t.id === id) ? { toasts: s.toasts.filter((t) => t.id !== id) } : s)),
  setScenarioFinished: (scenarioFinished) => set({ scenarioFinished }),
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
      dockTab: 'events',
      toasts: [],
      scenarioFinished: false,
      highlightFamily: null,
    })),
}));

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
