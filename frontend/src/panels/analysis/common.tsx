/**
 * Shared pieces of the Analysis panels: the analysis target (selected object + epoch at the time cursor), the
 * run bar (CTA · spinner · LIVE badge · timings · inline error), small tiles/tables and the chart style tokens.
 * Colours for categorical series come from the validated dark palette (palette.ts); text never wears series colour.
 */
import type { ReactNode } from 'react';
import type { RunState } from '../../store/useAnalysis';
import { useSelene } from '../../store/useSelene';

/** Which object and epoch an analysis run is for: the selected object (else the protagonist / first simulated
 *  object) and the time cursor floored to the hour, as an ISO UTC string the backend parses. */
/** The backend's SIMULATED truth is cached around the catalogue epoch and extended on demand up to ±120 d; beyond
 *  that the routes fail. Analysis epochs are clamped well inside that (the demo era) and the panel says so. */
const CLAMP_BEFORE_D = 30;
const CLAMP_AFTER_D = 100;

export function useAnalysisTarget(): { objectId: string | null; name: string; isReal: boolean; t0Iso: string; inScenario: boolean; ready: boolean; clamped: boolean } {
  const selected = useSelene((s) => s.selectedObjectId);
  const catalog = useSelene((s) => s.catalog);
  const protagonist = useSelene((s) => s.scenario?.meta.protagonist_id ?? null);
  const inScenario = useSelene((s) => !!s.scenario);
  const epochIso = useSelene((s) => s.catalogMeta?.epoch_utc ?? null);
  // Hour-quantised cursor so the panel does not re-render every tick while playing.
  const hourMs = useSelene((s) => Math.floor((Date.parse(s.t0Iso) + s.tSec * 1000) / 3600e3) * 3600e3);
  const objectId = selected ?? protagonist ?? catalog.find((c) => c.kind === 'simulated')?.id ?? null;
  const cat = catalog.find((c) => c.id === objectId);
  // No run before the catalogue epoch is known: the idle timeline starts at a placeholder date until then.
  const ready = !!epochIso && catalog.length > 0;
  let ms = hourMs;
  let clamped = false;
  if (epochIso) {
    const e = Date.parse(epochIso);
    const lo = e - CLAMP_BEFORE_D * 86400e3, hi = e + CLAMP_AFTER_D * 86400e3;
    if (ms < lo || ms > hi) {
      ms = Math.min(hi, Math.max(lo, ms));
      clamped = true;
    }
  }
  return { objectId, name: cat?.name ?? objectId ?? '—', isReal: cat?.kind === 'horizons', t0Iso: new Date(ms).toISOString().replace(/\.\d{3}Z$/, 'Z'), inScenario, ready, clamped };
}

export const fmtS = (s: number | undefined) => (typeof s === 'number' && Number.isFinite(s) ? (s >= 10 ? `${s.toFixed(0)} s` : `${s.toFixed(2)} s`) : '—');
export const fmtKm = (v: number | null | undefined, d = 1) => (typeof v === 'number' && Number.isFinite(v) ? (v >= 1000 ? Math.round(v).toLocaleString('en-US') : v >= 100 ? v.toFixed(0) : v.toFixed(d)) : '—');
export const fmtNum = (v: number | null | undefined, d = 2) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(d) : '—');
export const fmtPct = (v: number | null | undefined, d = 0) => (typeof v === 'number' && Number.isFinite(v) ? `${v.toFixed(d)}%` : '—');
export const hoursSince = (iso: string, t0Iso: string) => (Date.parse(iso) - Date.parse(t0Iso)) / 3600e3;
export const shortUtc = (iso: string | null | undefined) => (iso ? iso.replace('T', ' ').slice(5, 16) + 'Z' : '—');

export function LiveBadge({ title }: { title?: string }) {
  return (
    <span className="tag live-badge" title={title ?? 'Computed by the SELENE backend for this request (not a browser model)'}>
      LIVE
    </span>
  );
}

/**
 * Is a finished run still about the CURRENT analysis target? A result is stamped with the object and epoch it was
 * computed for (store/useAnalysis.ts); selecting another object or moving the time cursor to another hour makes it
 * stale: the panel then says so (and the scene overlays of an object mismatch are hidden) instead of letting a
 * result be read as the new target's. `objectId` null (tasking without a selection) is stored as 'default'.
 */
export function useRunStale<T>(state: RunState<T>): { objectMismatch: boolean; epochMismatch: boolean; stale: boolean } {
  const { objectId, t0Iso } = useAnalysisTarget();
  if (state.status !== 'done' && state.status !== 'error') return { objectMismatch: false, epochMismatch: false, stale: false };
  const objectMismatch = state.objectId !== (objectId ?? 'default');
  const epochMismatch = !objectMismatch && !!state.t0 && state.t0 !== t0Iso;
  return { objectMismatch, epochMismatch, stale: objectMismatch || epochMismatch };
}

/** CTA + status line shared by every panel: result stamp (object @ epoch), timings, stale notice, inline error. */
export function RunBar<T>({ label, state, onRun, disabled, disabledReason, backendS, children }: { label: string; state: RunState<T>; onRun: () => void; disabled?: boolean; disabledReason?: string; backendS?: number; children?: ReactNode }) {
  const running = state.status === 'running';
  const { objectId, t0Iso } = useAnalysisTarget();
  const { objectMismatch, epochMismatch, stale } = useRunStale(state);
  return (
    <div className="an-runbar">
      <div className="an-runrow">
        <button className="primary" onClick={onRun} disabled={disabled || running} title={disabled ? disabledReason : undefined}>
          {running ? (
            <>
              <span className="spinner" aria-hidden /> Running…
            </>
          ) : (
            label
          )}
        </button>
        {state.status === 'done' && (
          <span className="an-status mono">
            <LiveBadge /> for <b>{state.objectId === 'default' ? 'default set' : state.objectId}</b> @ {shortUtc(state.t0)} · backend {fmtS(backendS)} · round trip {fmtS(state.elapsedMs / 1000)}
          </span>
        )}
        {children}
      </div>
      {stale && (
        <div className={`an-stale ${objectMismatch ? 'alert' : 'warn'}`} role="status">
          <b>{objectMismatch ? 'STALE — other object' : 'STALE — other epoch'}</b> — this result is for <span className="mono">{state.objectId === 'default' ? 'the default set' : state.objectId}</span> at <span className="mono">{shortUtc(state.t0)}</span>; the target is now <span className="mono">{objectId ?? 'default set'}</span> at <span className="mono">{shortUtc(t0Iso)}</span>.{objectMismatch ? ' Its scene overlay is hidden.' : ''}
          {!disabled && !running && (
            <button className="link" onClick={onRun}>
              Re-run for the current target
            </button>
          )}
        </div>
      )}
      {state.status === 'error' && (
        <div className="an-error" role="alert">
          <b>Backend error</b> — {state.error}
          {epochMismatch || objectMismatch ? <span className="muted"> (for {state.objectId} @ {shortUtc(state.t0)})</span> : null}
        </div>
      )}
    </div>
  );
}

export function Tiles({ children }: { children: ReactNode }) {
  return <div className="an-tiles">{children}</div>;
}
export function Tile({ k, v, s, sev, title }: { k: string; v: ReactNode; s?: ReactNode; sev?: 'ok' | 'warn' | 'alert' | 'accent' | 'muted'; title?: string }) {
  return (
    <div className={`an-tile ${sev ?? ''}`} title={title}>
      <div className="k">{k}</div>
      <div className="v mono">{v}</div>
      {s !== undefined && <div className="s">{s}</div>}
    </div>
  );
}

export function Ctl({ label, title, children }: { label: string; title?: string; children: ReactNode }) {
  return (
    <label className="an-ctl" title={title}>
      <span className="k">{label}</span>
      {children}
    </label>
  );
}

export function Note({ children }: { children: ReactNode }) {
  return <p className="muted small an-note">{children}</p>;
}

/** Chart tokens (text wears text tokens; series colours come from palette.ts). */
export const TICK = { fontSize: 12, fill: '#8d9db3' } as const;
export const GRID = '#1c2733';
export const TIP_STYLE = { background: '#0b1016', border: '1px solid #1c2733', fontSize: 12, fontFamily: 'var(--font-mono)' } as const;
