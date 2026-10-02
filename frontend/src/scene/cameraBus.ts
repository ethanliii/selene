/** Tiny event bus so DOM buttons (CameraPresets) can ask the Canvas-side CameraRig to fly to a preset. */
import type { CAMERA_PRESETS } from './constants';

export type PresetName = keyof typeof CAMERA_PRESETS;

type Listener = (p: PresetName) => void;
const listeners = new Set<Listener>();

export function requestCameraPreset(p: PresetName): void {
  listeners.forEach((l) => l(p));
}

export function onCameraPreset(l: Listener): () => void {
  listeners.add(l);
  return () => {
    listeners.delete(l);
  };
}
