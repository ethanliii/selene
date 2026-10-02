/**
 * Screen-space label declutter for every registered scene label (scene/labels.ts). Runs every other render tick:
 * projects each anchor, measures the label's real DOM box, and keeps labels greedily by priority (ties: nearer to
 * the camera first). Losers get `visibility: hidden` (layout kept, so measurements stay valid).
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
    type Item = { x0: number; y0: number; x1: number; y1: number; el: HTMLElement; pri: number; dist: number; behind: boolean };
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
      const x0 = x + e.dx, y0 = y + e.dy - (e.dy < 0 ? h : 0);
      const pri = selected && e.id === selected ? 100 : e.priority;
      items.push({ x0, y0, x1: x0 + w, y1: y0 + h, el: e.el, pri, dist, behind });
    }
    items.sort((a, b) => b.pri - a.pri || a.dist - b.dist);
    for (const it of items) {
      // Hidden when behind the camera or when the box would be cut by the viewport edge (a half label is noise).
      let hidden = it.behind || it.x0 < 2 || it.y0 < 2 || it.x1 > size.width - 2 || it.y1 > size.height - 2;
      if (!hidden)
        for (const p of placed)
          if (it.x0 < p.x1 + 2 && it.x1 > p.x0 - 2 && it.y0 < p.y1 + 1 && it.y1 > p.y0 - 1) {
            hidden = true;
            break;
          }
      if (!hidden) placed.push(it);
      const vis = hidden ? 'hidden' : 'visible';
      if (it.el.style.visibility !== vis) it.el.style.visibility = vis;
    }
  });
  return null;
}
