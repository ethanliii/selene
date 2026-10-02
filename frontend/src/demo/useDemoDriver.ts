/**
 * Demo-scenario driver.
 *  - `startDemo()` loads /api/demo/scenario (browser CR3BP mock when offline), normalises it, resets the timeline
 *    to the scenario span, selects the protagonist, sets the scripted playback speed (≈2 min wall-clock) and plays.
 *  - `useDemoDriver()` (mounted once on the Ops page) surfaces events as the cursor crosses them (toast + feed),
 *    auto-selects the object of alert-level events, switches on scene layers an event asks for
 *    (`event.data.show_layers`, e.g. the exclusion cones when custody is lost) and switches the dock to the Brief
 *    when the story ends.
 *  - `resetDemo()` clears the scenario and returns to the idle 7-day timeline.
 */
import { useEffect } from 'react';
import { api } from '../api/client';
import { useSelene, type Layers } from '../store/useSelene';
import { normalizeScenario } from './normalize';

const MAX_TOASTS_PER_TICK = 3;

export async function startDemo(): Promise<void> {
  const st = useSelene.getState();
  const raw = await api.demoScenario();
  const sc = normalizeScenario(raw);
  st.loadScenario(sc);
  st.setSpeed(sc.meta.playback_speed ?? Math.max(1, sc.meta.duration_s / 120));
  st.selectObject(sc.meta.protagonist_id ?? null, 'auto');
  st.setDockTab('events');
  st.pushToast({ text: `Scenario loaded: ${sc.meta.title}. ${sc.frames.length} frames over ${(sc.meta.duration_s / 3600).toFixed(0)} h.`, severity: 'info', kind: 'scenario', t: 0 });
  st.setPlaying(true);
  // Refresh the ephemeris angles for the scenario span (true Earth–Moon line when the backend is up).
  const t1 = new Date(Date.parse(sc.meta.t0_utc) + sc.meta.duration_s * 1000).toISOString();
  api
    .ephemerisBodies(sc.meta.t0_utc, t1, Math.min(400, Math.max(24, Math.round(sc.meta.duration_s / 3600))))
    .then((e) => useSelene.getState().setEphemeris(e))
    .catch((e) => console.warn('ephemeris failed', e));
}

export function resetDemo(): void {
  useSelene.getState().reset();
}

function layersRequested(data: Record<string, unknown> | undefined): (keyof Layers)[] {
  const v = data?.show_layers;
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is keyof Layers => typeof x === 'string');
}

export function useDemoDriver(): void {
  useEffect(() => {
    const unsub = useSelene.subscribe((state, prev) => {
      if (state.tSec === prev.tSec && state.scenario === prev.scenario) return;
      const { events, scenario } = state;
      if (!scenario) return;
      // Forward crossing: surface events with prev.tSec < e.t <= state.tSec (at most the last few).
      if (state.tSec > prev.tSec) {
        const crossed = events.filter((e) => e.t > prev.tSec && e.t <= state.tSec);
        const shown = crossed.slice(-MAX_TOASTS_PER_TICK);
        for (const e of shown) state.pushToast({ text: e.text, severity: e.severity, kind: e.kind, objectId: e.object_id, t: e.t });
        for (const e of crossed) for (const k of layersRequested(e.data)) if (k in state.layers && !state.layers[k]) state.setLayer(k, true);
        const alert = [...crossed].reverse().find((e) => e.severity === 'alert' && e.object_id);
        if (alert?.object_id && state.selectedObjectId !== alert.object_id) state.selectObject(alert.object_id, 'auto');
      }
      // End of story → Brief.
      if (state.tSec >= state.t1Sec && !state.scenarioFinished) {
        state.setScenarioFinished(true);
        state.setDockTab('brief');
        if (scenario.meta.protagonist_id) state.selectObject(scenario.meta.protagonist_id, 'auto');
      }
    });
    return unsub;
  }, []);
}
