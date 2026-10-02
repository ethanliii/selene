/**
 * requestAnimationFrame playback loop: advances store.tSec by speed * wall-clock dt while playing.
 * Mount once (OpsPage). Reads the store imperatively to avoid re-rendering on every tick.
 *
 * Narration hold: at story speed (≥ 1000×) the cursor CRAWLS (1/40 of the speed, ≈ 2 sim-minutes per wall-second
 * at 4,320×) while a story event is being narrated (store.caption / captionQueue), so "custody lost" → "tasked" →
 * "regained", which are one sim-hour apart (0.8 s of wall time at 4,320×), are each on screen for their dwell with
 * the matching KPI / badge state. The hold adds ≈ 35 s to the 2-minute story; manual speeds are never held.
 */
import { useEffect } from 'react';
import { narrationHold, useSelene } from '../store/useSelene';

const CRAWL = 1 / 40;

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
      st.advance(st.speed * dt * (hold ? CRAWL : 1));
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
}
