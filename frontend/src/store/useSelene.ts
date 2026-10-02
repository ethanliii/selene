/**
 * Global UI state (zustand). Time is kept as `tSec` = seconds since `t0Iso` (scenario start, UTC).
 * Frame mode drives the 3D scene: 'rotating' = Earth–Moon synodic frame, 'inertial' = GCRF-like.
 */
import { create } from 'zustand';
import type { CatalogObject, DemoScenario, SeleneEvent } from '../api/types';

export type FrameMode = 'rotating' | 'inertial';

export interface Layers {
  families: boolean;
  lagrange: boolean;
  trails: boolean;
  clouds: boolean;
  fov: boolean;
  exclusion: boolean;
  grid: boolean;
}

export type PlaybackSpeed = 1 | 60 | 600 | 3600;
export const SPEEDS: PlaybackSpeed[] = [1, 60, 600, 3600];

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
  layers: Layers;
  events: SeleneEvent[];
  catalog: CatalogObject[];
  scenario: DemoScenario | null;
  brief: string;

  setFrame: (f: FrameMode) => void;
  setT: (t: number) => void;
  /** Advance by dt seconds (clamped to [t0Sec, t1Sec]; stops playback at the end). */
  advance: (dtSec: number) => void;
  setRange: (t0Sec: number, t1Sec: number, t0Iso?: string) => void;
  setPlaying: (p: boolean) => void;
  togglePlaying: () => void;
  setSpeed: (s: PlaybackSpeed) => void;
  selectObject: (id: string | null) => void;
  toggleLayer: (k: keyof Layers) => void;
  setLayer: (k: keyof Layers, v: boolean) => void;
  setEvents: (e: SeleneEvent[]) => void;
  setCatalog: (c: CatalogObject[]) => void;
  setBrief: (b: string) => void;
  /** Load a demo scenario: sets range, events, brief and rewinds to t0. */
  loadScenario: (s: DemoScenario) => void;
  reset: () => void;
}

const DEFAULT_T0_ISO = '2026-10-01T00:00:00Z';
const DEFAULT_SPAN_S = 7 * 86400;

export const useSelene = create<SeleneState>((set, get) => ({
  frame: 'rotating',
  t0Iso: DEFAULT_T0_ISO,
  tSec: 0,
  t0Sec: 0,
  t1Sec: DEFAULT_SPAN_S,
  playing: false,
  speed: 600,
  selectedObjectId: null,
  layers: { families: true, lagrange: true, trails: true, clouds: true, fov: true, exclusion: false, grid: true },
  events: [],
  catalog: [],
  scenario: null,
  brief: '',

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
    if (!playing && tSec >= t1Sec) set({ tSec: t0Sec, playing: true });
    else set({ playing: !playing });
  },
  setSpeed: (speed) => set({ speed }),
  selectObject: (selectedObjectId) => set({ selectedObjectId }),
  toggleLayer: (k) => set((s) => ({ layers: { ...s.layers, [k]: !s.layers[k] } })),
  setLayer: (k, v) => set((s) => ({ layers: { ...s.layers, [k]: v } })),
  setEvents: (events) => set({ events: [...events].sort((a, b) => a.t - b.t) }),
  setCatalog: (catalog) => set({ catalog }),
  setBrief: (brief) => set({ brief }),
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
    }),
  reset: () =>
    set({ t0Iso: DEFAULT_T0_ISO, t0Sec: 0, t1Sec: DEFAULT_SPAN_S, tSec: 0, playing: false, events: [], scenario: null, brief: '', selectedObjectId: null }),
}));

/** Current UTC Date for the time cursor. */
export function cursorDate(t0Iso: string, tSec: number): Date {
  return new Date(Date.parse(t0Iso) + tSec * 1000);
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
