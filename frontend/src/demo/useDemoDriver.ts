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
import { api, getEndpointStatus } from '../api/client';
import type { SeleneEvent } from '../api/types';
import { useSelene, type Layers } from '../store/useSelene';
import { SCENARIO_DURATION_S, SCENARIO_T0 } from './mockScenario';
import { normalizeScenario } from './normalize';

/** More story events than this crossing in one tick means a scrub, not playback: narrate only the last one. */
const SCRUB_THRESHOLD = 3;
const SCHEDULER_MS = 150;

export async function startDemo(): Promise<void> {
  const st = useSelene.getState();
  if (getEndpointStatus().demo !== 'live') {
    // The browser mock evaluates ground visibility (Earth orientation, Sun direction) through lib/ephem.ts, so
    // install the ephemeris for the SCENARIO span first: otherwise the idle span's tables are the ones loaded and
    // the generator would run on the mean-element model (lib/ephem.ts refuses to extrapolate outside its span).
    // The Ops page re-fetches the same span after loadScenario (no-op) and the idle span again on Reset.
    try {
      const t1 = new Date(Date.parse(SCENARIO_T0) + SCENARIO_DURATION_S * 1000).toISOString();
      const eph = await api.ephemerisBodies(SCENARIO_T0, t1, Math.round(SCENARIO_DURATION_S / 3600) + 1);
      st.setEphemeris(eph); // store action also installs it in lib/ephem.ts
    } catch (e) {
      console.warn('ephemeris for the scenario span failed; mock scenario uses the mean-element model', e);
    }
  }
  const raw = await api.demoScenario();
  const sc = normalizeScenario(raw);
  st.loadScenario(sc);
  st.setSpeed(sc.meta.playback_speed ?? Math.max(1, sc.meta.duration_s / 120));
  st.selectObject(sc.meta.protagonist_id ?? null, 'auto');
  st.setDockTab('events');
  st.pushToast({ headline: `Scenario loaded · ${(sc.meta.duration_s / 3600).toFixed(0)} h in about 2 minutes`, text: `${sc.meta.title}. ${sc.frames.length} frames over ${(sc.meta.duration_s / 3600).toFixed(0)} h.`, severity: 'info', kind: 'scenario', t: 0 });
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
