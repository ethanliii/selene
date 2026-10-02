/**
 * Transient event toasts: bottom-right lane of the viewport (so the Moon / L1 / L2 region stays clear), at most two
 * at a time (store), headline only (two-line clamp) with the full analyst text as the tooltip. Pushed by the demo
 * driver as the cursor crosses events. The lane registers as a label-declutter obstacle.
 */
import { kindLabel } from '../demo/headline';
import { fmtElapsed, useSelene } from '../store/useSelene';
import { severityColor } from './severity';

export function Toasts() {
  const toasts = useSelene((s) => s.toasts);
  const dismiss = useSelene((s) => s.dismissToast);
  const select = useSelene((s) => s.selectObject);
  const setT = useSelene((s) => s.setT);
  const selected = useSelene((s) => s.selectedObjectId);
  if (toasts.length === 0) return null;
  return (
    <div className="toasts" role="status" aria-live="polite" data-label-obstacle>
      {toasts.map((t) => (
        <div
          key={t.id}
          className="toast"
          style={{ ['--sev' as string]: severityColor(t.severity) }}
          title={t.text}
          onClick={() => {
            if (t.objectId) select(t.objectId);
            setT(t.t);
            dismiss(t.id);
          }}
        >
          <div className="meta">
            <span className="kind">{kindLabel(t.kind)}</span>
            <span>T{fmtElapsed(t.t)}</span>
            {t.objectId && t.objectId !== selected && <span>{t.objectId}</span>}
            <button
              className="x"
              onClick={(e) => {
                e.stopPropagation();
                dismiss(t.id);
              }}
              aria-label="dismiss"
            >
              ×
            </button>
          </div>
          <div className="h">{t.headline}</div>
        </div>
      ))}
    </div>
  );
}
