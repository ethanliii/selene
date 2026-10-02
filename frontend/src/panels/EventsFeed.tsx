/** Events feed: chronological list, colored by severity; events after the time cursor are dimmed. */
import { fmtElapsed, useSelene } from '../store/useSelene';
import { severityColor } from './severity';

export function EventsFeed() {
  const events = useSelene((s) => s.events);
  const tSec = useSelene((s) => s.tSec);
  const setT = useSelene((s) => s.setT);
  const select = useSelene((s) => s.selectObject);

  if (events.length === 0) {
    return (
      <div>
        <h3 className="panel-title">Events</h3>
        <p className="muted">No events loaded. Press ▶ Demo Scenario to load the scripted story.</p>
      </div>
    );
  }
  return (
    <div>
      <h3 className="panel-title">
        Events <span className="mono muted">({events.filter((e) => e.t <= tSec).length}/{events.length})</span>
      </h3>
      <ul className="events">
        {events.map((e, i) => (
          <li
            key={i}
            className={e.t <= tSec ? 'past' : 'future'}
            style={{ ['--sev' as string]: severityColor(e.severity) }}
            onClick={() => {
              setT(e.t);
              if (e.object_id) select(e.object_id);
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
        ))}
      </ul>
    </div>
  );
}
