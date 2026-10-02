/**
 * Events feed: chronological list coloured by severity. Each row shows a plain-English headline (≤ 12 words) with
 * the analyst detail collapsed underneath (the newest reached event is expanded; ▸ expands any row). While a
 * scenario is playing only events the cursor has reached are listed (the story is not spoiled); "Upcoming" reveals
 * the rest, dimmed. Click → jump the timeline to the event and select its object. With no events loaded the tab is
 * a call-to-action card that starts the 2-minute scenario.
 */
import { useEffect, useRef, useState } from 'react';
import { deriveHeadline, kindLabel } from '../demo/headline';
import { startDemo } from '../demo/useDemoDriver';
import { fmtElapsed, useSelene } from '../store/useSelene';
import { Icon } from './icons';
import { severityColor } from './severity';

function CtaCard() {
  const basisReady = useSelene((s) => s.ephemeris !== null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  async function go() {
    setBusy(true);
    setErr(null);
    try {
      await startDemo();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }
  return (
    <div>
      <h3 className="panel-title">Events</h3>
      <div className="cta-card">
        <button className="primary big" onClick={go} disabled={busy || !basisReady} title={basisReady ? 'Scripted 2-minute story (48–144 h of simulated time)' : 'Loading the ephemeris basis…'}>
          {busy ? (
            'Loading scenario…'
          ) : basisReady ? (
            <>
              <Icon name="play" size={16} />
              Play the 2-minute scenario
            </>
          ) : (
            'Loading…'
          )}
        </button>
        <ol>
          <li>
            A <b>notional</b> spacecraft sits quietly in a lunar DRO, then burns <b>unannounced</b>.
          </li>
          <li>
            Ground telescopes catch one bad residual, then <b>lose it in lunar glare</b> — the uncertainty cloud balloons toward the L1 gateway and an allied relay.
          </li>
          <li>
            SELENE <b>re-tasks a space-based observer</b>, regains custody, characterises the Δv and writes the analyst brief.
          </li>
        </ol>
        <p className="fine">All objects and events are SIMULATED and attributed to a notional actor. Events, toasts and the custody ribbon fill in as the story plays.</p>
        {err && <p className="fine" style={{ color: 'var(--alert)' }}>Scenario failed to load: {err}</p>}
      </div>
    </div>
  );
}

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
  const [expanded, setExpanded] = useState<Set<string>>(() => new Set());
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

  if (events.length === 0) return <CtaCard />;

  const keyOf = (e: (typeof events)[number], i: number) => `${e.t}-${e.kind}-${i}`;
  const item = (e: (typeof events)[number], i: number, isPast: boolean) => {
    const key = keyOf(e, i);
    const current = isPast && i === lastPastIdx;
    const open = current || expanded.has(key);
    const headline = e.headline ?? deriveHeadline(e);
    return (
      <li
        key={key}
        ref={current ? lastPastRef : undefined}
        className={`${isPast ? 'past' : 'future'}${current ? ' current' : ''}`}
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
            <span className="kind">{kindLabel(e.kind)}</span>
            {e.object_id && e.object_id !== selected && <span>{e.object_id}</span>}
            {!current && (
              <button
                className="chev"
                aria-label={open ? 'Collapse detail' : 'Expand detail'}
                title={open ? 'Collapse detail' : 'Show analyst detail'}
                onClick={(ev) => {
                  ev.stopPropagation();
                  setExpanded((s) => {
                    const n = new Set(s);
                    if (n.has(key)) n.delete(key);
                    else n.add(key);
                    return n;
                  });
                }}
              >
                {open ? '▾' : '▸'}
              </button>
            )}
          </div>
          <div className="h">{headline}</div>
          <div className={`detail${open ? '' : ' clamp'}`}>{e.text}</div>
        </div>
      </li>
    );
  };
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
            Follow
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
