/**
 * Events feed: chronological list coloured by severity. While a scenario is playing only events the cursor has
 * reached are listed (the story is not spoiled); a "Show upcoming" toggle reveals the rest, dimmed. Click → jump
 * the timeline to the event and select its object. Auto-scroll (toggle) keeps the newest past event in view.
 */
import { useEffect, useRef, useState } from 'react';
import { fmtElapsed, useSelene } from '../store/useSelene';
import { severityColor } from './severity';

export function EventsFeed() {
  const events = useSelene((s) => s.events);
  // Re-render only when the number of past events changes, not on every tick.
  const pastAll = useSelene((s) => s.events.filter((e) => e.t <= s.tSec).length);
  const scenario = useSelene((s) => s.scenario);
  const finished = useSelene((s) => s.scenarioFinished);
  const setT = useSelene((s) => s.setT);
  const select = useSelene((s) => s.selectObject);
  const [autoScroll, setAutoScroll] = useState(true);
  const [filterSel, setFilterSel] = useState(false);
  const [showUpcoming, setShowUpcoming] = useState(false);
  const selected = useSelene((s) => s.selectedObjectId);
  const lastPastRef = useRef<HTMLLIElement | null>(null);

  const past = events.slice(0, pastAll);
  const future = events.slice(pastAll);
  const revealFuture = !scenario || finished || showUpcoming;
  const filter = (list: typeof events) => (filterSel && selected ? list.filter((e) => e.object_id === selected) : list);
  const shownPast = filter(past);
  const shownFuture = revealFuture ? filter(future) : [];
  const lastPastIdx = shownPast.length - 1;

  useEffect(() => {
    if (autoScroll && lastPastRef.current) lastPastRef.current.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [lastPastIdx, autoScroll]);

  if (events.length === 0) {
    return (
      <div>
        <h3 className="panel-title">Events</h3>
        <p className="muted">No events loaded. Press ▶ Demo Scenario to load the scripted story.</p>
      </div>
    );
  }
  const item = (e: (typeof events)[number], i: number, isPast: boolean) => (
    <li
      key={`${e.t}-${e.kind}-${i}`}
      ref={isPast && i === lastPastIdx ? lastPastRef : undefined}
      className={`${isPast ? 'past' : 'future'}${isPast && i === lastPastIdx ? ' current' : ''}`}
      style={{ ['--sev' as string]: severityColor(e.severity) }}
      onClick={() => {
        setT(e.t);
        if (e.object_id) select(e.object_id, 'user');
      }}
      title="Jump to event"
    >
      <span className="bar" />
      <div>
        <div className="meta">
          <span>T{fmtElapsed(e.t)}</span>
          <span className="kind">{e.kind.replace(/_/g, ' ')}</span>
          {e.object_id && <span>{e.object_id}</span>}
        </div>
        <div className="text">{e.text}</div>
      </div>
    </li>
  );
  return (
    <div>
      <h3 className="panel-title feed-head">
        <span>
          Events <span className="mono muted">({shownPast.length}/{filter(events).length})</span>
        </span>
        <span className="feed-tools">
          <button className={`toggle${filterSel ? ' active' : ''}`} onClick={() => setFilterSel((v) => !v)} disabled={!selected} title="Only events for the selected object">
            Selected
          </button>
          <button className={`toggle${autoScroll ? ' active' : ''}`} onClick={() => setAutoScroll((v) => !v)} title="Follow the newest event while playing">
            Auto-scroll
          </button>
          {scenario && !finished && (
            <button className={`toggle${showUpcoming ? ' active' : ''}`} onClick={() => setShowUpcoming((v) => !v)} title="Reveal events the story has not reached yet (presenter view)">
              Upcoming
            </button>
          )}
        </span>
      </h3>
      <ul className="events">
        {shownPast.map((e, i) => item(e, i, true))}
        {shownFuture.map((e, i) => item(e, i, false))}
        {!revealFuture && future.length > 0 && (
          <li className="future placeholder">
            <span className="bar" />
            <div className="text muted">
              {future.length} upcoming event{future.length === 1 ? '' : 's'} — surfaced as the story plays.
            </div>
          </li>
        )}
      </ul>
    </div>
  );
}
