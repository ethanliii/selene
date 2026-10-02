/**
 * requestAnimationFrame playback loop: advances store.tSec by speed * wall-clock dt while playing.
 * Mount once (OpsPage). Reads the store imperatively to avoid re-rendering on every tick.
 *
 * Narration hold: at story speed (≥ 1000×) the cursor CRAWLS (1/40 of the speed, ≈ 2 sim-minutes per wall-second
 * at 4,320×) while a story event is being narrated (store.caption / captionQueue), so "custody lost" → "tasked" →
 * "regained", which are one sim-hour apart (0.8 s of wall time at 4,320×), are each on screen for their dwell with
 * the matching KPI / badge state. The hold adds ≈ 35 s to the 2-minute story; manual speeds are never held.
 *
 * Coast: once the cursor is past the LAST story event (observations excluded) and nothing is being narrated, the
 * scripted playback runs COAST× faster so the silent tail after the brief does not stretch the story; the
 * timeline speed readout shows the coast factor. Manual speeds (< 1000×) never coast.
 */
import { useEffect } from 'react';
import { narrationHold, useSelene } from '../store/useSelene';

const CRAWL = 1 / 40;
const COAST = 3;

/** Epoch (s) of the last story event of the loaded scenario, or +∞ when there is none. */
function lastStoryEventT(events: { kind: string; t: number }[]): number {
  let last = -Infinity;
  for (const e of events) if (e.kind !== 'observation' && e.t > last) last = e.t;
  return Number.isFinite(last) ? last : Infinity;
}

export function usePlayback(): void {
  const playing = useSelene((s) => s.playing);
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      // Clamp dt so a backgrounded tab does not jump the cursor by minutes on return.
      const dt = Math.min(0.25, (now - last) / 1000);
      last = now;
      const st = useSelene.getState();
      const hold = narrationHold(st, now);
      const coast = !hold && !!st.scenario && st.speed >= 1000 && st.tSec > lastStoryEventT(st.events);
      st.advance(st.speed * dt * (hold ? CRAWL : coast ? COAST : 1));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
}
