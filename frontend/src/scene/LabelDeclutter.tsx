/**
 * Screen-space label declutter for every registered scene label (scene/labels.ts). Runs every other render tick:
 * projects each anchor, measures the label's real DOM box, and keeps labels greedily by priority (ties: nearer to
 * the camera first). A label whose own box collides is tried mirrored about its anchor (left / above / both)
 * before it loses; losers get `visibility: hidden` (layout kept, so measurements stay valid).
 */
import { useFrame, useThree } from '@react-three/fiber';
import { useRef } from 'react';
import * as THREE from 'three';
import { useSelene } from '../store/useSelene';
import { labelEntries } from './labels';

const tmp = new THREE.Vector3();

export function LabelDeclutter() {
  const { camera, size, gl } = useThree();
  const tick = useRef(0);
  const obstacleTick = useRef(0);
  const obstacles = useRef<HTMLElement[]>([]);
  useFrame(() => {
    if (++tick.current % 2) return;
    const selected = useSelene.getState().selectedObjectId;
    type Item = { x0: number; y0: number; x1: number; y1: number; el: HTMLElement; pri: number; dist: number; behind: boolean; x?: number; y?: number; w?: number; h?: number; dx?: number; dy?: number };
    const items: Item[] = [];
    // DOM overlays (HUD, scale bar, camera presets: elements tagged data-label-obstacle) block labels too.
    const placed: Item[] = [];
    if (++obstacleTick.current % 15 === 1) obstacles.current = Array.from(document.querySelectorAll<HTMLElement>('[data-label-obstacle]'));
    const canvasRect = (gl.domElement as HTMLCanvasElement).getBoundingClientRect();
    for (const ob of obstacles.current) {
      const r = ob.getBoundingClientRect();
      if (r.width === 0 || r.height === 0) continue;
      placed.push({ x0: r.left - canvasRect.left, y0: r.top - canvasRect.top, x1: r.right - canvasRect.left, y1: r.bottom - canvasRect.top, el: ob, pri: 1000, dist: 0, behind: false });
    }
    for (const e of labelEntries()) {
      e.anchor.getWorldPosition(tmp);
      const dist = tmp.distanceTo(camera.position);
      tmp.project(camera);
      const behind = tmp.z > 1 || !Number.isFinite(tmp.x) || !Number.isFinite(tmp.y);
      const x = ((tmp.x + 1) / 2) * size.width;
      const y = ((1 - tmp.y) / 2) * size.height;
      const w = e.el.offsetWidth || 80;
      const h = e.el.offsetHeight || 16;
      // The label's CSS transform moves its TOP-LEFT corner to anchor + (dx, dy) (scene-label / obj-label styles), so
      // the measured box starts at y + dy for every variant (a 'tag' at dy = −6 straddles the anchor, not above it).
      const x0 = x + e.dx, y0 = y + e.dy;
      const pri = selected && e.id === selected ? 100 : e.priority;
      items.push({ x0, y0, x1: x0 + w, y1: y0 + h, el: e.el, pri, dist, behind, x, y, w, h, dx: e.dx, dy: e.dy });
    }
    items.sort((a, b) => b.pri - a.pri || a.dist - b.dist);
    const collides = (x0: number, y0: number, x1: number, y1: number) => {
      if (x0 < 2 || y0 < 2 || x1 > size.width - 2 || y1 > size.height - 2) return true; // cut by the viewport edge
      for (const p of placed) if (x0 < p.x1 + 4 && x1 > p.x0 - 4 && y0 < p.y1 + 2 && y1 > p.y0 - 2) return true;
      return false;
    };
    for (const it of items) {
      // Placement candidates: the label's own offset first, then the box mirrored about the anchor (left of it,
      // above/below it, both) so e.g. MOON survives next to a selected object's label instead of vanishing.
      let hidden = it.behind;
      if (!hidden) {
        const { x = 0, y = 0, w = 0, h = 0, dx = 0, dy = 0 } = it;
        const mx = -(dx + w), my = -(dy + h);
        const cands = [
          [dx, dy],
          [mx, dy],
          [dx, my],
          [mx, my],
        ];
        let ok = false;
        for (let c = 0; c < cands.length; c++) {
          const [cdx, cdy] = cands[c];
          if (c > 0 && cdx === dx && cdy === dy) continue;
          if (c > 0 && Math.abs(cdx - dx) < 1 && Math.abs(cdy - dy) < 1) continue;
          const x0 = x + cdx, y0 = y + cdy;
          if (collides(x0, y0, x0 + w, y0 + h)) continue;
          it.x0 = x0;
          it.y0 = y0;
          it.x1 = x0 + w;
          it.y1 = y0 + h;
          const tf = c === 0 ? '' : `translate(${cdx.toFixed(0)}px, ${cdy.toFixed(0)}px)`;
          if (it.el.style.transform !== tf) it.el.style.transform = tf;
          ok = true;
          break;
        }
        hidden = !ok;
      }
      if (!hidden) placed.push(it);
      const vis = hidden ? 'hidden' : 'visible';
      if (it.el.style.visibility !== vis) it.el.style.visibility = vis;
    }
  });
  return null;
}
