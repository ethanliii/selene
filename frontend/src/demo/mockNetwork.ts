/**
 * Notional sensor network used by the browser mock (catalog of sensors AND the scripted scenario).
 * Ground coordinates are those of well-known public observatory sites; the instruments described are notional
 * configurations for the aperture class (0.5 m ≈ 18 mag, 0.8–1 m ≈ 18.5–19.5 mag; same rule of thumb as the
 * backend's sites.py), NOT the specifications of any real telescope. Lunar glare is the phase-dependent 3°–15°
 * zone of lib/groundVis.ts (not a per-site constant); daylight = Sun above −12° at the site.
 */
import type { GroundSensor, SpaceSensor } from '../api/types';

export const MOCK_GROUND: GroundSensor[] = [
  { id: 'GND-MAUI', name: 'Haleakalā, Maui (notional 1 m)', lat_deg: 20.708, lon_deg: -156.257, alt_km: 3.055, limiting_mag: 19.5, fov_deg: 1.0, min_elevation_deg: 20, sun_elev_max_deg: -12, spec_note: 'ASSUMED representative specs; location public' },
  { id: 'GND-SSO', name: 'Siding Spring (notional 1 m)', lat_deg: -31.273, lon_deg: 149.064, alt_km: 1.165, limiting_mag: 19.0, fov_deg: 1.2, min_elevation_deg: 20, sun_elev_max_deg: -12, spec_note: 'ASSUMED representative specs; location public' },
  { id: 'GND-TEIDE', name: 'Teide, Tenerife (notional 0.8 m)', lat_deg: 28.3, lon_deg: -16.511, alt_km: 2.39, limiting_mag: 18.5, fov_deg: 1.5, min_elevation_deg: 20, sun_elev_max_deg: -12, spec_note: 'ASSUMED representative specs; location public' },
  { id: 'GND-SOCORRO', name: 'Socorro, NM (notional 0.5 m)', lat_deg: 33.817, lon_deg: -106.66, alt_km: 2.4, limiting_mag: 18.0, fov_deg: 2.0, min_elevation_deg: 20, sun_elev_max_deg: -12, spec_note: 'ASSUMED representative specs; location public' },
];

export interface MockSpaceObserverDef {
  id: string;
  /** Orbit-library family the platform rides ('GEO' = geostationary ring). */
  family: string | 'GEO';
  memberIndex: number;
  phase: number;
  fov_deg: number;
}

export const MOCK_SPACE_OBSERVERS: MockSpaceObserverDef[] = [
  { id: 'SPC-DRO-A', family: 'DRO', memberIndex: 3, phase: 0.78, fov_deg: 4 },
  { id: 'SPC-L1-A', family: 'L1 Lyapunov', memberIndex: 2, phase: 0.0, fov_deg: 3 },
  { id: 'SPC-GEO-A', family: 'GEO', memberIndex: 0, phase: 0.15, fov_deg: 6 },
];

export const MOCK_SPACE: SpaceSensor[] = MOCK_SPACE_OBSERVERS.map((d) => ({
  id: d.id,
  name: d.family === 'GEO' ? 'GEO-hosted observer (notional)' : d.family === 'DRO' ? 'DRO observer (notional)' : 'L1 observer (notional)',
  orbit: d.family === 'GEO' ? 'GEO' : d.family === 'DRO' ? 'DRO' : 'L1_halo',
  phase: d.phase,
  limiting_mag: d.family === 'GEO' ? 16.5 : 18.0,
  fov_deg: d.fov_deg,
  sun_exclusion_deg: 40,
  moon_exclusion_deg: 10,
  earth_exclusion_deg: 10,
  slew_rate_dps: 1.0,
}));

export const MOCK_GROUND_IDS = MOCK_GROUND.map((g) => g.id);
