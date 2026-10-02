/** Camera preset buttons (viewport overlay, bottom-left). */
import { requestCameraPreset, type PresetName } from '../scene/cameraBus';
import { CAMERA_PRESETS } from '../scene/constants';

const ORDER: PresetName[] = ['overview', 'earth', 'moon', 'l1', 'l2'];

export function CameraPresets() {
  return (
    <div className="cam-presets">
      <span className="label">View</span>
      {ORDER.map((k) => (
        <button key={k} className="toggle" onClick={() => requestCameraPreset(k)} title={`Fly camera to ${CAMERA_PRESETS[k].label}`}>
          {CAMERA_PRESETS[k].label}
        </button>
      ))}
    </div>
  );
}
