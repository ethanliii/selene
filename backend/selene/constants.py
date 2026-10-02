"""Physical constants (SI-derived, km/s units) and CR3BP normalisation.

Gravitational parameters are read from JPL ``gm_de440.tpc`` (cached under ``data/cache``)
so the CR3BP mass ratio and the ephemeris force model share one source of truth.
Fallback literal values (identical to the DE440 file) are used if the file is missing.

References
----------
* Park, Folkner, Williams, Boggs (2021), "The JPL Planetary and Lunar Ephemerides DE440 and DE441", AJ 161:105.
* IAU 2015 Resolution B3 nominal values for radii.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
DE440S_PATH = CACHE_DIR / "de440s.bsp"
GM_TPC_PATH = CACHE_DIR / "gm_de440.tpc"

# ---------------------------------------------------------------------------
# Gravitational parameters [km^3 / s^2]  (DE440)
# ---------------------------------------------------------------------------
_GM_FALLBACK = {
    "earth": 3.9860043550702266e05,
    "moon": 4.9028001184575496e03,
    "sun": 1.3271244004127942e11,
    "emb": 4.0350323562548019e05,
}
_GM_IDS = {"earth": "399", "moon": "301", "sun": "10", "emb": "3"}


def _load_gm_from_tpc(path: Path) -> dict[str, float]:
    gm = dict(_GM_FALLBACK)
    if not path.exists():
        return gm
    text = path.read_text()
    for key, bid in _GM_IDS.items():
        m = re.search(rf"BODY{bid}_GM\s*=\s*\(\s*([0-9.ED+\-]+)\s*\)", text)
        if m:
            gm[key] = float(m.group(1).replace("D", "E"))
    return gm


_GM = _load_gm_from_tpc(GM_TPC_PATH)
GM_EARTH: float = _GM["earth"]
GM_MOON: float = _GM["moon"]
GM_SUN: float = _GM["sun"]
GM_EMB: float = _GM["emb"]

# ---------------------------------------------------------------------------
# Radii [km], distances [km]
# ---------------------------------------------------------------------------
R_EARTH = 6378.1366          # IAU 2015 nominal equatorial
R_EARTH_POLAR = 6356.7519
R_MOON = 1737.4
R_SUN = 695_700.0
AU_KM = 149_597_870.7
GEO_RADIUS_KM = 42_164.0

# Solar radiation pressure at 1 AU [N/m^2] (1361 W/m^2 / c)
P_SRP_1AU = 4.56e-6

# ---------------------------------------------------------------------------
# CR3BP (Earth-Moon) normalisation
#   L* = mean Earth-Moon distance, T* = sqrt(L*^3 / (GM_E + GM_M)), V* = L*/T*
#   Units: positions in L*, time in T* (so the synodic period is 2*pi), velocities in V*.
# ---------------------------------------------------------------------------
MU: float = GM_MOON / (GM_EARTH + GM_MOON)          # ~0.0121505856
L_STAR: float = 384_400.0                           # km (conventional mean distance)
T_STAR: float = math.sqrt(L_STAR**3 / (GM_EARTH + GM_MOON))   # s, ~375190
V_STAR: float = L_STAR / T_STAR                     # km/s, ~1.0245
SYNODIC_PERIOD_S: float = 2.0 * math.pi * T_STAR    # ~27.28 d

DAY_S = 86_400.0
SECONDS_PER_DAY = DAY_S


def nd_to_km(x: float) -> float:
    return x * L_STAR


def km_to_nd(x: float) -> float:
    return x / L_STAR


def nd_to_s(t: float) -> float:
    return t * T_STAR


def s_to_nd(t: float) -> float:
    return t / T_STAR


def nd_to_kms(v: float) -> float:
    return v * V_STAR


def kms_to_nd(v: float) -> float:
    return v / V_STAR
