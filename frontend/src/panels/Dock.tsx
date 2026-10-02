/** Right dock with Object / Events / Brief tabs. */
import { useEffect, useState } from 'react';
import { useSelene } from '../store/useSelene';
import { AnalystBrief } from './AnalystBrief';
import { EventsFeed } from './EventsFeed';
import { ObjectPanel } from './ObjectPanel';

type Tab = 'object' | 'events' | 'brief';
const TABS: { id: Tab; label: string }[] = [
  { id: 'object', label: 'Object' },
  { id: 'events', label: 'Events' },
  { id: 'brief', label: 'Brief' },
];

export function Dock() {
  const [tab, setTab] = useState<Tab>('events');
  const selected = useSelene((s) => s.selectedObjectId);
  // Selecting an object in the scene brings the Object tab forward.
  useEffect(() => {
    if (selected) setTab('object');
  }, [selected]);
  return (
    <>
      <div className="tabs">
        {TABS.map((t) => (
          <button key={t.id} className={tab === t.id ? 'active' : ''} onClick={() => setTab(t.id)}>
            {t.label}
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
