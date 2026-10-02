/**
 * requestAnimationFrame playback loop: advances store.tSec by speed * wall-clock dt while playing.
 * Mount once (OpsPage). Reads the store imperatively to avoid re-rendering on every tick.
 */
import { useEffect } from 'react';
import { useSelene } from '../store/useSelene';

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
      const { speed, advance } = useSelene.getState();
      advance(speed * dt);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);
}
