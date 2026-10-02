/**
 * Demo-scenario driver.
 *  - `startDemo()` loads /api/demo/scenario (browser CR3BP mock when offline), normalises it, resets the timeline
 *    to the scenario span, selects the protagonist, sets the scripted playback speed (≈2 min wall-clock) and plays.
 *  - `useDemoDriver()` (mounted once on the Ops page) surfaces events as the cursor crosses them, auto-selects the
 *    object of alert-level events, switches on scene layers an event asks for (`event.data.show_layers`, e.g. the
 *    exclusion cones when custody is lost) and switches the dock to the Brief when the story ends.
 *  - Narration: story events (everything except observations) go through the store's caption QUEUE and are shown
 *    one at a time (presenter caption + toast + timeline readout) with a wall-clock dwell each; the playback loop
 *    crawls while the queue drains (hooks/usePlayback.ts). Observations only get a toast, directly. A large forward
 *    jump (scrub / deep link) narrates just the last crossed event; a backward jump clears the narration.
 *  - `resetDemo()` clears the scenario and returns to the idle 7-day timeline.
 */
import { useEffect } from 'react';
import { api, ApiError } from '../api/client';
import type { DemoScenario, SeleneEvent } from '../api/types';
import { useSelene, type Layers } from '../store/useSelene';
import { beatOfEvent } from './headline';
import { SCENARIO_DURATION_S, SCENARIO_T0 } from './mockScenario';
import { normalizeScenario } from './normalize';

/** More story events than this crossing in one tick means a scrub, not playback: narrate only the last one. */
const SCRUB_THRESHOLD = 3;
const SCHEDULER_MS = 150;

/**
 * Fetch the real bundle (GET /api/demo/scenario, precomputed by the backend engines). Only when the backend is
 * unreachable does the EXPLICIT offline fallback run: the browser CR3BP story, labelled "[OFFLINE MOCK]" in its
 * title, the HUD, the brief footer and a toast. A backend error that is not "unreachable" (e.g. a 404 "bundle not
 * built, run …") is NOT masked by the mock: it is thrown so the presenter sees the real reason.
 */
async function fetchScenario(): Promise<DemoScenario> {
  const st = useSelene.getState();
  try {
    return normalizeScenario(await api.demoScenario(), 'backend');
  } catch (e) {
    // `unreachable` covers a network failure, a 502–504 gateway answer AND the Vite dev proxy's empty-body 500
    // (what `make dev` returns while uvicorn is down); a real backend answer (404 "bundle not built", 500 with a
    // detail) is re-thrown so the presenter sees the actual reason instead of a mock story.
    const unreachable = e instanceof ApiError && e.unreachable;
    if (!unreachable) throw e;
    console.warn('backend unreachable — playing the browser mock scenario (labelled OFFLINE MOCK)', e);
  }
  // The browser mock evaluates ground visibility (Earth orientation, Sun direction) through lib/ephem.ts, so
  // install the ephemeris tables for ITS span first (mean-element fallback when even that fails).
  try {
    const t1 = new Date(Date.parse(SCENARIO_T0) + SCENARIO_DURATION_S * 1000).toISOString();
    st.setEphemeris(await api.ephemerisBodies(SCENARIO_T0, t1, Math.round(SCENARIO_DURATION_S / 3600) + 1));
  } catch (e) {
    console.warn('ephemeris for the mock scenario span failed; mean-element model in use', e);
  }
  return normalizeScenario(api.demoScenarioMock(), 'browser-mock');
}

export async function startDemo(): Promise<void> {
  const st = useSelene.getState();
  const sc = await fetchScenario();
  st.loadScenario(sc);
  st.setSpeed(sc.meta.playback_speed ?? Math.max(1, sc.meta.duration_s / 120));
  st.selectObject(sc.meta.protagonist_id ?? null, 'auto');
  st.setDockTab('events');
  const mock = sc.meta.source === 'browser-mock';
  st.pushToast({
    headline: mock ? 'OFFLINE — browser mock scenario (backend unreachable)' : `Scenario loaded · ${(sc.meta.duration_s / 3600).toFixed(0)} h in about 2 minutes`,
    text: mock ? `${sc.meta.title}. CR3BP story generated in the browser because /api/demo/scenario could not be reached; numbers are not the backend engines'.` : `${sc.meta.title}. ${sc.frames.length} frames over ${(sc.meta.duration_s / 3600).toFixed(0)} h, computed by the backend engines (DE440s truth, UKF, particle clouds, reachability, tasker).`,
    severity: mock ? 'warn' : 'info',
    kind: 'scenario',
    t: 0,
  });
  st.setPlaying(true);
  // The Ops page re-fetches the ephemeris basis for the new timeline span (effect keyed on t0Iso/t0Sec/t1Sec).
}

export function resetDemo(): void {
  useSelene.getState().reset();
}

function layersRequested(data: Record<string, unknown> | undefined): (keyof Layers)[] {
  const v = data?.show_layers;
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is keyof Layers => typeof x === 'string');
}

function toastOf(e: SeleneEvent) {
  return { headline: e.headline ?? e.text, text: e.text, severity: e.severity, kind: e.kind, objectId: e.object_id, t: e.t };
}

export function useDemoDriver(): void {
  useEffect(() => {
    const unsub = useSelene.subscribe((state, prev) => {
      if (state.tSec === prev.tSec && state.scenario === prev.scenario) return;
      const { events, scenario } = state;
      if (!scenario) return;
      if (state.tSec > prev.tSec) {
        // Forward crossing: prev.tSec < e.t <= state.tSec.
        const crossed = events.filter((e) => e.t > prev.tSec && e.t <= state.tSec);
        const story = crossed.filter((e) => e.kind !== 'observation');
        const obs = crossed.filter((e) => e.kind === 'observation');
        if (story.length > SCRUB_THRESHOLD || (!state.playing && story.length > 0)) {
          // Scrub / deep link: no replay of everything skipped, just the latest event.
          state.enqueueCaptions([story[story.length - 1]], true);
        } else if (story.length) state.enqueueCaptions(story);
        // Observations: a direct toast (one per tick), unless a story event is being narrated right now.
        if (obs.length && story.length === 0 && state.captionQueue.length === 0) state.pushToast(toastOf(obs[obs.length - 1]));
        for (const e of crossed) for (const k of layersRequested(e.data)) if (k in state.layers && !state.layers[k]) state.setLayer(k, true);
        const alert = [...crossed].reverse().find((e) => e.severity === 'alert' && e.object_id);
        if (alert?.object_id && state.selectedObjectId !== alert.object_id) state.selectObject(alert.object_id, 'auto');
        // The brief lands at its event (brief_ready): bring the Brief tab forward right then, not only at the end.
        if (crossed.some((e) => beatOfEvent(e) === 'brief') && state.dockTab !== 'brief') state.setDockTab('brief');
      } else if (state.tSec < prev.tSec && state.scenario === prev.scenario) {
        state.clearCaptions();
      }
      // End of story → Brief.
      if (state.tSec >= state.t1Sec && !state.scenarioFinished) {
        state.setScenarioFinished(true);
        state.setDockTab('brief');
        if (scenario.meta.protagonist_id) state.selectObject(scenario.meta.protagonist_id, 'auto');
      }
    });
    // Narration scheduler: shows the next queued story event (caption + toast) once the current dwell is over.
    const timer = window.setInterval(() => {
      const st = useSelene.getState();
      if (!st.scenario) return;
      const shown = st.tickCaptions(performance.now());
      if (shown) st.pushToast(toastOf(shown));
    }, SCHEDULER_MS);
    return () => {
      unsub();
      window.clearInterval(timer);
    };
  }, []);
}
