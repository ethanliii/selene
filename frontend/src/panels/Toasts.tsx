/** Transient event toasts (top-right of the viewport), pushed by the demo driver as the cursor crosses events. */
import { fmtElapsed, useSelene } from '../store/useSelene';
import { severityColor } from './severity';

export function Toasts() {
  const toasts = useSelene((s) => s.toasts);
  const dismiss = useSelene((s) => s.dismissToast);
  const select = useSelene((s) => s.selectObject);
  const setT = useSelene((s) => s.setT);
  if (toasts.length === 0) return null;
  return (
    <div className="toasts" role="status" aria-live="polite">
      {toasts.map((t) => (
        <div
          key={t.id}
          className="toast"
          style={{ ['--sev' as string]: severityColor(t.severity) }}
          onClick={() => {
            if (t.objectId) select(t.objectId);
            setT(t.t);
            dismiss(t.id);
          }}
        >
          <div className="meta">
            <span className="kind">{t.kind.replace(/_/g, ' ')}</span>
            <span>T{fmtElapsed(t.t)}</span>
            {t.objectId && <span>{t.objectId}</span>}
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
          <div className="text">{t.text}</div>
        </div>
      ))}
    </div>
  );
}
