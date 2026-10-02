/**
 * Analysis dock tab: sub-tabs OD · Maneuver · Reach · Tasking, each a live call to the matching backend engine for
 * the selected object at the time cursor. Results persist in store/useAnalysis.ts while the user switches tabs and
 * are rendered in the 3D scene by scene/AnalysisOverlays.tsx.
 */
import { ANALYSIS_TABS, useAnalysis } from '../../store/useAnalysis';
import { ManeuverPanel } from './ManeuverPanel';
import { OdPanel } from './OdPanel';
import { ReachabilityPanel } from './ReachabilityPanel';
import { TaskingPanel } from './TaskingPanel';
import { useAnalysisTarget } from './common';

export function AnalysisPanel() {
  const sub = useAnalysis((s) => s.sub);
  const setSub = useAnalysis((s) => s.setSub);
  // One primitive selector per slot (a selector returning a fresh object re-renders on every store change in zustand v5).
  const odSt = useAnalysis((s) => s.od.status);
  const mvSt = useAnalysis((s) => s.maneuver.status);
  const rcSt = useAnalysis((s) => s.reach.status);
  const tkSt = useAnalysis((s) => s.tasking.status);
  const states = { od: odSt, maneuver: mvSt, reach: rcSt, tasking: tkSt } as const;
  const { objectId, name, isReal } = useAnalysisTarget();
  return (
    <div>
      <h3 className="panel-title an-head">
        <span>Analysis</span>
        <span className="an-target mono" title="Analysis target: the selected object (else the scenario protagonist) at the time cursor">
          {objectId ?? '—'} {isReal ? <span className="tag accent">REAL</span> : <span className="tag sim">SIMULATED</span>}
        </span>
      </h3>
      <p className="muted small an-target-name">{name}</p>
      <div className="an-tabs" role="tablist">
        {ANALYSIS_TABS.map((t) => (
          <button key={t.id} role="tab" aria-selected={sub === t.id} className={sub === t.id ? 'active' : ''} onClick={() => setSub(t.id)} title={t.title}>
            {t.label}
            {states[t.id] === 'running' && <span className="spinner sm" aria-hidden />}
            {states[t.id] === 'done' && <span className="dot ok" aria-hidden />}
            {states[t.id] === 'error' && <span className="dot err" aria-hidden />}
          </button>
        ))}
      </div>
      {sub === 'od' && <OdPanel />}
      {sub === 'maneuver' && <ManeuverPanel />}
      {sub === 'reach' && <ReachabilityPanel />}
      {sub === 'tasking' && <TaskingPanel />}
    </div>
  );
}
