/**
 * Ground-site observability of a cislunar target — browser port of the backend rules
 * (backend/selene/sensors/constraints.py and photometry.py) so the mock scenario applies the SAME tests the
 * live coverage/tasking engines use. All geometry is in the rotating frame (nondimensional, see lib/ephem.ts
 * for the frame model: ecliptic-aligned synodic frame, Earth tilted by the J2000 obliquity, GMST spin).
 *
 * Tests (all must pass; `reasons` lists the failures):
 *   daylight      Sun elevation at the site > sun_elev_max_deg (default −12°, astronomical twilight)
 *   low_elevation target elevation above the geodetic horizon < min_elevation_deg (default 20°)
 *   in_shadow     target inside the cylindrical umbra of the Earth or the Moon (unlit)
 *   lunar_glare   Moon–target separation < θ_excl = 3° + (15° − 3°)·k, k = illuminated fraction of the Moon
 *                 (k = (1 + cos α)/2, α = Sun–Moon–site phase angle, Meeus ch. 48). MODELLING ASSUMPTION
 *                 standing in for scattered-moonlight sky background (same constants as the backend).
 *   too_faint     visual magnitude > limiting_mag, Lambertian sphere:
 *                 m = −26.74 − 2.5 log10( a · π ρ² · p(φ) / R² ),  p(φ) = (2/(3π²))[(π−φ)cos φ + sin φ]
 *                 (Hejduk 2011 / Krag 1974 convention; m☉ = −26.74, Cox 2000).
 * Not modelled: refraction, atmospheric extinction, weather, lunar inclination (5.1°), Earth oblateness.
 */
import type { GroundSensor, Vec3 } from '../api/types';
import { L_STAR_KM, MU } from './cr3bp';
import { ecefToRot, sunPosRotKm } from './ephem';

const DEG = Math.PI / 180;
export const R_EARTH_KM = 6378.1366;
export const R_MOON_KM = 1737.4;
export const M_SUN = -26.74;
/** Lunar-glare exclusion half-angle at new / full Moon [deg] (assumption, identical to the backend). */
export const GLARE_THETA_MIN_DEG = 3.0;
export const GLARE_THETA_MAX_DEG = 15.0;

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const scale = (a: Vec3, k: number): Vec3 => [a[0] * k, a[1] * k, a[2] * k];
/** Angle between two vectors [rad], atan2 form (stable near 0 and π). */
export function angularSeparation(u: Vec3, v: Vec3): number {
  const c: Vec3 = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]];
  return Math.atan2(norm(c), dot(u, v));
}

/** Rotating-frame positions (km, barycentric) of the Earth and Moon centres. */
export function earthKm(): Vec3 {
  return [-MU * L_STAR_KM, 0, 0];
}
export function moonKm(ms: number): Vec3 {
  // Keep the Moon at (1−μ)·L* in scene units; the ephemeris distance only rescales magnitudes/parallax slightly.
  void ms;
  return [(1 - MU) * L_STAR_KM, 0, 0];
}

/** Site position (km, rotating, barycentric) and geodetic zenith unit vector (spherical Earth). */
export function siteRotKm(site: Pick<GroundSensor, 'lat_deg' | 'lon_deg' | 'alt_km'>, ms: number): { pos: Vec3; zenith: Vec3 } {
  const la = site.lat_deg * DEG, lo = site.lon_deg * DEG;
  const u: Vec3 = [Math.cos(la) * Math.cos(lo), Math.cos(la) * Math.sin(lo), Math.sin(la)];
  const zenith = ecefToRot(u, ms);
  const r = R_EARTH_KM + (site.alt_km ?? 0);
  const e = earthKm();
  return { pos: [e[0] + zenith[0] * r, e[1] + zenith[1] * r, e[2] + zenith[2] * r], zenith };
}

/** Illuminated fraction of the Moon as seen from `viewerKm` (default Earth centre). */
export function illuminatedFraction(ms: number, viewerKm?: Vec3): number {
  const m = moonKm(ms);
  const s = sunPosRotKm(ms);
  const v = viewerKm ?? earthKm();
  const alpha = angularSeparation(sub(s, m), sub(v, m));
  return 0.5 * (1 + Math.cos(alpha));
}

/** Phase-dependent lunar-glare exclusion half-angle [deg] (3° new Moon → 15° full Moon). */
export function lunarGlareHalfAngleDeg(ms: number, viewerKm?: Vec3): number {
  return GLARE_THETA_MIN_DEG + (GLARE_THETA_MAX_DEG - GLARE_THETA_MIN_DEG) * illuminatedFraction(ms, viewerKm);
}

/** Cylindrical-umbra test: behind the body (anti-Sun side) and within one body radius of the axis. */
export function inBodyShadow(tgtKm: Vec3, bodyKm: Vec3, bodyRadiusKm: number, sunKm: Vec3): boolean {
  const rel = sub(tgtKm, bodyKm);
  const s = sub(sunKm, bodyKm);
  const sn = norm(s) || 1;
  const along = dot(rel, s) / sn;
  const perp2 = dot(rel, rel) - along * along;
  return along < 0 && perp2 < bodyRadiusKm * bodyRadiusKm;
}

/** Lambertian-sphere phase function p(φ), φ in rad (p(0) = 2/(3π)). */
export function lambertianPhaseFunction(phi: number): number {
  const f = Math.min(Math.PI, Math.max(0, phi));
  return (2 / (3 * Math.PI * Math.PI)) * ((Math.PI - f) * Math.cos(f) + Math.sin(f));
}

/** Apparent visual magnitude of a Lambertian sphere (radius m, Lambert albedo, phase angle rad, range km). */
export function visualMagnitude(radiusM: number, albedo: number, phi: number, rangeKm: number): number {
  const p = lambertianPhaseFunction(phi);
  if (p < 1e-12) return Infinity;
  const rhoKm = radiusM * 1e-3;
  const flux = (albedo * Math.PI * rhoKm * rhoKm * p) / (rangeKm * rangeKm);
  return M_SUN - 2.5 * Math.log10(flux);
}

export interface GroundVisibility {
  visible: boolean;
  reasons: string[];
  /** Target elevation above the site horizon [deg]. */
  elevation_deg: number;
  /** Sun elevation at the site [deg] (negative = below the horizon). */
  sun_elevation_deg: number;
  /** Topocentric Moon–target separation [deg] and the glare threshold in force [deg]. */
  moon_sep_deg: number;
  glare_threshold_deg: number;
  /** Moon illuminated fraction (0–1). */
  illum: number;
  /** Sun–target separation [deg]. */
  sun_sep_deg: number;
  /** Sun–target–site phase angle [deg], apparent magnitude and range. */
  phase_deg: number;
  magnitude: number;
  range_km: number;
}

/**
 * Evaluate every ground constraint for a target at rotating-frame position `targetNd` (nondimensional) at
 * UTC epoch `ms`. Target photometry from `radiusM` / `albedo` (defaults: 1 m, 0.2).
 */
export function groundVisibility(site: GroundSensor, targetNd: Vec3, ms: number, radiusM = 1.0, albedo = 0.2): GroundVisibility {
  const { pos, zenith } = siteRotKm(site, ms);
  const tgt = scale(targetNd, L_STAR_KM);
  const sun = sunPosRotKm(ms);
  const moon = moonKm(ms);
  const los = sub(tgt, pos);
  const rangeKm = norm(los);
  const toSun = sub(sun, pos);
  const toMoon = sub(moon, pos);
  const elev = Math.asin(Math.max(-1, Math.min(1, dot(los, zenith) / rangeKm))) / DEG;
  const sunElev = Math.asin(Math.max(-1, Math.min(1, dot(toSun, zenith) / norm(toSun)))) / DEG;
  const moonSep = angularSeparation(los, toMoon) / DEG;
  const sunSep = angularSeparation(los, toSun) / DEG;
  const illum = illuminatedFraction(ms, pos);
  const thr = GLARE_THETA_MIN_DEG + (GLARE_THETA_MAX_DEG - GLARE_THETA_MIN_DEG) * illum;
  const phase = angularSeparation(sub(sun, tgt), sub(pos, tgt));
  const mag = visualMagnitude(radiusM, albedo, phase, rangeKm);
  const reasons: string[] = [];
  if (sunElev > (site.sun_elev_max_deg ?? -12)) reasons.push('daylight');
  if (elev < (site.min_elevation_deg ?? 20)) reasons.push('low_elevation');
  if (inBodyShadow(tgt, earthKm(), R_EARTH_KM, sun) || inBodyShadow(tgt, moon, R_MOON_KM, sun)) reasons.push('in_shadow');
  const moonAngRad = Math.asin(Math.min(1, R_MOON_KM / norm(toMoon))) / DEG;
  if (moonSep - moonAngRad < thr) reasons.push('lunar_glare');
  if (mag > site.limiting_mag) reasons.push('too_faint');
  return {
    visible: reasons.length === 0,
    reasons,
    elevation_deg: elev,
    sun_elevation_deg: sunElev,
    moon_sep_deg: moonSep,
    glare_threshold_deg: thr,
    illum,
    sun_sep_deg: sunSep,
    phase_deg: phase / DEG,
    magnitude: mag,
    range_km: rangeKm,
  };
}

export interface SpaceVisibility {
  visible: boolean;
  reasons: string[];
  sun_sep_deg: number;
  moon_sep_deg: number;
  earth_sep_deg: number;
  range_km: number;
  phase_deg: number;
  magnitude: number;
}

/** Space-observer exclusions (Sun, Moon-limb, Earth-limb), eclipse not modelled; angles in degrees. */
export function spaceVisibility(
  obsNd: Vec3,
  targetNd: Vec3,
  ms: number,
  cfg: { sun_exclusion_deg: number; moon_exclusion_deg: number; earth_exclusion_deg: number; limiting_mag: number },
  radiusM = 1.0,
  albedo = 0.2,
): SpaceVisibility {
  const obs = scale(obsNd, L_STAR_KM);
  const tgt = scale(targetNd, L_STAR_KM);
  const los = sub(tgt, obs);
  const sun = sunPosRotKm(ms);
  const toSun = sub(sun, obs), toMoon = sub(moonKm(ms), obs), toEarth = sub(earthKm(), obs);
  const sunSep = angularSeparation(los, toSun) / DEG;
  const moonSep = angularSeparation(los, toMoon) / DEG - Math.asin(Math.min(1, R_MOON_KM / norm(toMoon))) / DEG;
  const earthSep = angularSeparation(los, toEarth) / DEG - Math.asin(Math.min(1, R_EARTH_KM / norm(toEarth))) / DEG;
  const rangeKm = norm(los);
  const phase = angularSeparation(sub(sun, tgt), sub(obs, tgt));
  const mag = visualMagnitude(radiusM, albedo, phase, rangeKm);
  const reasons: string[] = [];
  if (sunSep < cfg.sun_exclusion_deg) reasons.push('sun_exclusion');
  if (moonSep < cfg.moon_exclusion_deg) reasons.push('moon_exclusion');
  if (earthSep < cfg.earth_exclusion_deg) reasons.push('earth_exclusion');
  if (inBodyShadow(tgt, earthKm(), R_EARTH_KM, sun) || inBodyShadow(tgt, moonKm(ms), R_MOON_KM, sun)) reasons.push('in_shadow');
  if (mag > cfg.limiting_mag) reasons.push('too_faint');
  return { visible: reasons.length === 0, reasons, sun_sep_deg: sunSep, moon_sep_deg: moonSep, earth_sep_deg: earthSep, range_km: rangeKm, phase_deg: phase / DEG, magnitude: mag };
}
