/** Right dock with Object / Events / Brief tabs (active tab lives in the store so the demo driver can switch it). */
import { useEffect } from 'react';
import { useSelene, type DockTab } from '../store/useSelene';
import { AnalystBrief } from './AnalystBrief';
import { EventsFeed } from './EventsFeed';
import { ObjectPanel } from './ObjectPanel';

const TABS: { id: DockTab; label: string }[] = [
  { id: 'object', label: 'Object' },
  { id: 'events', label: 'Events' },
  { id: 'brief', label: 'Brief' },
];

export function Dock() {
  const tab = useSelene((s) => s.dockTab);
  const setTab = useSelene((s) => s.setDockTab);
  const selected = useSelene((s) => s.selectedObjectId);
  const source = useSelene((s) => s.selectSource);
  const finished = useSelene((s) => s.scenarioFinished);
  const briefReady = useSelene((s) => s.brief.length > 0);
  // A USER selection in the scene brings the Object tab forward; the demo driver's automatic selections do not
  // (the scripted story keeps the Events feed in front and surfaces events as toasts).
  useEffect(() => {
    if (selected && source === 'user' && !finished) setTab('object');
  }, [selected, source, finished, setTab]);
  return (
    <>
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
            {t.id === 'brief' && briefReady && finished && <span className="dot" />}
          </button>
        ))}
      </div>
      <div className="dock-body">
        {tab === 'object' && <ObjectPanel />}
        {tab === 'events' && <EventsFeed />}
        {tab === 'brief' && <AnalystBrief />}
      </div>
    </>
  );
}
