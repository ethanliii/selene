/**
 * Analysis-panel state (zustand): one run slot per engine (OD · maneuver · reachability · tasking) with status,
 * the LIVE response, the inline error and the round-trip time, plus the scene-overlay switches (OD particle frame,
 * reachable endpoints). Results survive dock-tab switches and are rendered in the 3D scene by
 * scene/AnalysisOverlays.tsx. Nothing here is ever mocked: a failed call leaves `error` set and `data` null.
 */
import { create } from 'zustand';
import type { ManeuverDetectResponse, OdPresets, OdRunResponse, ReachResponse, TaskingPresets, TaskingResponse } from '../api/analysisTypes';

export type AnalysisKey = 'od' | 'maneuver' | 'reach' | 'tasking';
export const ANALYSIS_TABS: { id: AnalysisKey; label: string; title: string }[] = [
  { id: 'od', label: 'OD', title: 'Orbit determination: IOD → batch LS → UKF, particle-cloud custody decay (POST /api/od/run)' },
  { id: 'maneuver', label: 'Maneuver', title: 'Maneuver detection with an injectable SIMULATED burn (POST /api/maneuver/detect)' },
  { id: 'reach', label: 'Reach', title: 'Reachable set under an assumed Δv budget (POST /api/reachability)' },
  { id: 'tasking', label: 'Tasking', title: 'Sensor tasking: greedy / MILP / baselines (POST /api/tasking/schedule)' },
];

export interface RunState<T> {
  status: 'idle' | 'running' | 'done' | 'error';
  data: T | null;
  error: string | null;
  /** Round-trip wall-clock, ms (the backend's own timing is inside `data`). */
  elapsedMs: number;
  /** Object and epoch the run was made for (shown next to the result so a stale result is never mistaken). */
  objectId: string | null;
  t0: string | null;
  startedAt: number;
}

const idle = <T,>(): RunState<T> => ({ status: 'idle', data: null, error: null, elapsedMs: 0, objectId: null, t0: null, startedAt: 0 });

interface AnalysisState {
  sub: AnalysisKey;
  od: RunState<OdRunResponse>;
  maneuver: RunState<ManeuverDetectResponse>;
  reach: RunState<ReachResponse>;
  tasking: RunState<TaskingResponse>;
  odPresets: OdPresets | null;
  taskingPresets: TaskingPresets | null;
  /** Index into od.data.particles.frames shown in the scene (custody-decay slider). */
  odFrame: number;
  showOdCloud: boolean;
  showReach: boolean;
  setSub: (k: AnalysisKey) => void;
  setOdFrame: (i: number) => void;
  setShowOdCloud: (v: boolean) => void;
  setShowReach: (v: boolean) => void;
  setOdPresets: (p: OdPresets | null) => void;
  setTaskingPresets: (p: TaskingPresets | null) => void;
  /** Run an engine call; a newer run of the same key supersedes (and aborts) an older one still in flight. */
  run: <K extends AnalysisKey>(key: K, objectId: string, t0: string, fn: (signal: AbortSignal) => Promise<AnalysisState[K]['data']>) => Promise<void>;
  clear: (key: AnalysisKey) => void;
}

const seq: Record<AnalysisKey, number> = { od: 0, maneuver: 0, reach: 0, tasking: 0 };
const inflight: Partial<Record<AnalysisKey, AbortController>> = {};

export const useAnalysis = create<AnalysisState>((set, get) => ({
  sub: 'od',
  od: idle(),
  maneuver: idle(),
  reach: idle(),
  tasking: idle(),
  odPresets: null,
  taskingPresets: null,
  odFrame: 0,
  showOdCloud: true,
  showReach: true,
  setSub: (sub) => set({ sub }),
  setOdFrame: (odFrame) => set({ odFrame }),
  setShowOdCloud: (showOdCloud) => set({ showOdCloud }),
  setShowReach: (showReach) => set({ showReach }),
  setOdPresets: (odPresets) => set({ odPresets }),
  setTaskingPresets: (taskingPresets) => set({ taskingPresets }),
  run: async (key, objectId, t0, fn) => {
    const my = ++seq[key];
    const started = performance.now();
    // A superseded request is aborted so the backend stops working on an answer nobody will render.
    inflight[key]?.abort();
    const ctl = new AbortController();
    inflight[key] = ctl;
    set({ [key]: { ...get()[key], status: 'running', error: null, objectId, t0, startedAt: started } } as Partial<AnalysisState>);
    try {
      const data = await fn(ctl.signal);
      if (seq[key] !== my) return;
      const extra = key === 'od' ? { odFrame: 0 } : {};
      set({ [key]: { status: 'done', data, error: null, elapsedMs: performance.now() - started, objectId, t0, startedAt: started }, ...extra } as Partial<AnalysisState>);
    } catch (e) {
      if (seq[key] !== my) return;
      const msg = e instanceof Error ? e.message : String(e);
      set({ [key]: { status: 'error', data: null, error: msg, elapsedMs: performance.now() - started, objectId, t0, startedAt: started } } as Partial<AnalysisState>);
    }
  },
  clear: (key) => {
    inflight[key]?.abort();
    seq[key]++;
    set({ [key]: idle() } as Partial<AnalysisState>);
  },
}));
