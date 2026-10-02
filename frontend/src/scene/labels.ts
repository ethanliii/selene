/**
 * Scene-label registry for screen-space decluttering. Every HTML label in the 3D view (objects, bodies, L-points,
 * the 9:2 NRHO tag) registers its anchor Object3D, its DOM element and a priority; `LabelDeclutter` (SceneRoot)
 * projects all anchors once per tick, places labels greedily by priority (then by distance to the camera) and hides
 * any whose box would overlap an already-placed one. Priorities: selected object 100 > Earth/Moon 95 > L1/L2 90 >
 * NRHO 85 > simulated objects 70 > real objects 60 > L3/L4/L5 40.
 */
import type * as THREE from 'three';

export interface LabelEntry {
  id: string;
  priority: number;
  el: HTMLElement;
  anchor: THREE.Object3D;
  /** CSS offset of the label box from the projected anchor (px). */
  dx: number;
  dy: number;
}

const entries = new Map<string, LabelEntry>();

export function registerLabel(e: LabelEntry): void {
  entries.set(e.id, e);
}
export function unregisterLabel(id: string, el?: HTMLElement): void {
  const cur = entries.get(id);
  if (!cur || (el && cur.el !== el)) return;
  entries.delete(id);
}
export function setLabelPriority(id: string, priority: number): void {
  const cur = entries.get(id);
  if (cur) cur.priority = priority;
}
export function labelEntries(): IterableIterator<LabelEntry> {
  return entries.values();
}
