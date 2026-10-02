/**
 * "LIVE: catalog, orbits, ephemeris, sensors · MOCK: OD, tasking, demo" — built from the per-endpoint registry in
 * api/client.ts, so the banner states exactly which parts of the screen come from the backend and which from the
 * browser mock. Endpoints that have not been touched yet are omitted.
 */
import { ENDPOINTS, useBackendStatus, useEndpointStatus, type EndpointName } from '../api/client';

export function useLiveMockLists(): { live: EndpointName[]; available: EndpointName[]; mock: EndpointName[] } {
  const st = useEndpointStatus();
  const live = ENDPOINTS.filter((e) => st[e] === 'live');
  const available = ENDPOINTS.filter((e) => st[e] === 'available');
  const mock = ENDPOINTS.filter((e) => st[e] === 'mock');
  return { live, available, mock };
}

export function LiveStatus({ compact = false }: { compact?: boolean }) {
  const backend = useBackendStatus();
  const { live, available, mock } = useLiveMockLists();
  if (backend === 'offline') return <span className="live-strip offline">BACKEND OFFLINE — ALL DATA FROM BROWSER MOCK (CR3BP in the browser)</span>;
  if (live.length === 0 && mock.length === 0 && available.length === 0) return null;
  return (
    <span
      className={`live-strip${compact ? ' compact' : ''}`}
      title="LIVE: data on screen fetched from the backend · AVAILABLE: route exists on the backend but this screen does not call it yet · MOCK: browser fallback"
    >
      {live.length > 0 && (
        <span className="live">
          LIVE: <b>{live.join(', ')}</b>
        </span>
      )}
      {live.length > 0 && available.length > 0 && <span className="sep"> · </span>}
      {available.length > 0 && (
        <span className="avail" title="Backend route exists; not wired into this screen yet">
          AVAILABLE (not wired): <b>{available.join(', ')}</b>
        </span>
      )}
      {(live.length > 0 || available.length > 0) && mock.length > 0 && <span className="sep"> · </span>}
      {mock.length > 0 && (
        <span className="mock">
          MOCK: <b>{mock.join(', ')}</b>
        </span>
      )}
    </span>
  );
}
