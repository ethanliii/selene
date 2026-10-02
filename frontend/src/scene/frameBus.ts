/**
 * Shared per-tick state of the synodic scene group: the matrix that currently maps rotating-frame (nondimensional)
 * coordinates into the displayed world, written by SceneRoot's FrameGroup every render tick and read by the
 * camera rig (presets are authored in rotating coordinates) and the scale bar.
 */
import * as THREE from 'three';

export const FRAME_MATRIX = new THREE.Matrix4();
export const FRAME_INFO = {
  /** 'rotating' | 'inertial-exact' (DE440s basis) | 'inertial-planar' (mean-element fallback) */
  mode: 'rotating' as 'rotating' | 'inertial-exact' | 'inertial-planar',
  /** Uniform scale of the group (d/L* in exact inertial mode, 1 otherwise). */
  scale: 1,
};

// ---- scale bar: world units per screen pixel at the orbit target, published to the DOM --------------
let worldPerPx = 0;
const listeners = new Set<() => void>();
export function publishWorldPerPixel(v: number): void {
  if (Math.abs(v - worldPerPx) < worldPerPx * 0.005) return;
  worldPerPx = v;
  listeners.forEach((l) => l());
}
export function subscribeScale(l: () => void): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}
export function getWorldPerPixel(): number {
  return worldPerPx;
}
