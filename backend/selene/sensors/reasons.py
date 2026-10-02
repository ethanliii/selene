"""Visibility-failure reason bitmask shared by every sensor constraint.

A constraint evaluation returns an integer array; each set bit names one reason the
target could not be observed at that instant.  ``0`` means "no violation" (visible).
Bits are independent, so a target can be simultaneously in daylight *and* too faint.
"""
from __future__ import annotations

import numpy as np

REASON_DAYLIGHT = 1 << 0        # ground: Sun above the astronomical-twilight limit at the site
REASON_LOW_ELEVATION = 1 << 1   # ground: target below the site's minimum elevation (or below horizon)
REASON_SUN_EXCLUSION = 1 << 2   # space: target too close to the Sun (stray light / baffle limit)
REASON_MOON_EXCLUSION = 1 << 3  # both: target too close to the (illuminated) Moon (glare / sky background)
REASON_EARTH_EXCLUSION = 1 << 4 # space: target too close to the Earth limb, or occulted by the Earth
REASON_IN_SHADOW = 1 << 5       # both: target inside the Earth's or Moon's umbral cylinder (unlit)
REASON_TOO_FAINT = 1 << 6       # both: predicted visual magnitude fainter than the limiting magnitude
REASON_OUT_OF_FOV = 1 << 7      # both: target outside the (pointed) field of view

ALL_REASONS = (
    REASON_DAYLIGHT,
    REASON_LOW_ELEVATION,
    REASON_SUN_EXCLUSION,
    REASON_MOON_EXCLUSION,
    REASON_EARTH_EXCLUSION,
    REASON_IN_SHADOW,
    REASON_TOO_FAINT,
    REASON_OUT_OF_FOV,
)

REASON_NAMES = {
    REASON_DAYLIGHT: "daylight",
    REASON_LOW_ELEVATION: "low_elevation",
    REASON_SUN_EXCLUSION: "sun_exclusion",
    REASON_MOON_EXCLUSION: "moon_exclusion",
    REASON_EARTH_EXCLUSION: "earth_exclusion",
    REASON_IN_SHADOW: "in_shadow",
    REASON_TOO_FAINT: "too_faint",
    REASON_OUT_OF_FOV: "out_of_fov",
}
REASON_BY_NAME = {v: k for k, v in REASON_NAMES.items()}

#: Reasons that describe the *sensor* being unavailable rather than the *target* being hard
#: to see.  Used to pick the most informative reason when several sensors fail (coverage.py).
AVAILABILITY_REASONS = REASON_DAYLIGHT | REASON_LOW_ELEVATION

#: Tie-break priority (most target-specific first) when choosing a single dominant reason.
REASON_PRIORITY = (
    REASON_MOON_EXCLUSION,
    REASON_SUN_EXCLUSION,
    REASON_EARTH_EXCLUSION,
    REASON_IN_SHADOW,
    REASON_TOO_FAINT,
    REASON_OUT_OF_FOV,
    REASON_LOW_ELEVATION,
    REASON_DAYLIGHT,
)

_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.int8)


def popcount(mask) -> np.ndarray:
    """Number of set reason bits for an integer array (values < 256)."""
    return _POPCOUNT[np.asarray(mask, dtype=np.int64) & 0xFF]


def reason_names(mask: int) -> list[str]:
    """Human-readable list of the reasons set in ``mask``."""
    return [REASON_NAMES[b] for b in ALL_REASONS if int(mask) & b]


def primary_reason(mask) -> np.ndarray:
    """Collapse a bitmask array to the single highest-priority reason bit (0 if none set)."""
    m = np.asarray(mask, dtype=np.int64)
    out = np.zeros_like(m)
    for bit in reversed(REASON_PRIORITY):  # lowest priority first, so highest wins
        out = np.where(m & bit, bit, out)
    return out
