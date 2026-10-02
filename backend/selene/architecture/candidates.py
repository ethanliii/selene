"""Candidate sensor platforms and architectures for the trade studio (PLAN §2.8, M8).

A *platform* is a named orbit a hypothetical space-based optical sensor could ride; a
*sensor spec* is a platform plus the instrument numbers a user can type into the studio
(aperture, limiting magnitude, field of view, slew rate, GEO longitude); an *architecture* is a
list of sensor specs plus a flag for the notional 9-site ground network of
:data:`selene.sensors.sites.DEFAULT_SITES`.

Platforms (``PLATFORMS``) map onto :class:`selene.sensors.observers.SpaceObserver`
``platform_orbit`` values, so every platform rides an orbit from the validated library
(``observers.resolve_orbit``): 9:2 synodic NRHO (``L2_halo_S_nrho_9_2``), L1 northern halo
(Az ≈ 30 000 km), L2 southern halo (Az ≈ 53 000 km), ~70 000 km DRO (Artemis-I class), the
mid-family 3:1 resonant orbit, or an ideal geostationary point.

Limiting magnitude from aperture
--------------------------------
When a spec gives an aperture but no limiting magnitude, the same aperture-class rule of thumb
as the ground network (``sites.py``: 0.5 m ≈ 18.5 mag, 1 m ≈ 19.5–20 mag, 1.5 m ≈ 20.5 mag) is
applied as ``m_lim = 18.5 + 5·log10(D / 0.5 m)`` (point-source SNR ∝ D², 1 mag per factor 2.5
in flux), clipped to [14, 23].  It is deliberately **not** credited with a space-based
sky-background advantage: these are ASSUMED representative values, not instrument
specifications.  Costs are *proxies* only (sensor count, summed aperture, summed collecting
area) so the UI can show "value per sensor"; no monetary figures are invented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Optional, Sequence

from selene.sensors.observers import SpaceObserver
from selene.sensors.sites import DEFAULT_SITES, GroundSite
from selene.sensors.visibility import Sensor

__all__ = [
    "PLATFORMS",
    "PLATFORM_INFO",
    "SensorSpec",
    "Architecture",
    "PRESET_ARCHITECTURES",
    "preset_architectures",
    "limiting_mag_from_aperture",
    "aperture_from_limiting_mag",
    "architecture_from_dict",
]

#: studio platform id -> SpaceObserver platform_orbit
PLATFORMS: dict[str, str] = {
    "geo": "geo",
    "l1_halo": "l1_halo",
    "l2_halo_S": "l2_halo",
    "dro": "dro",
    "nrho_9_2": "nrho",
    "resonant_3_1": "resonant",
}

#: human-readable facts per platform (period from the library record at resolution time; the
#: numbers here are the design-point descriptions used in the UI)
PLATFORM_INFO: dict[str, dict] = {
    "geo": {"label": "GEO-hosted sensor", "orbit": "ideal geostationary point (longitude parameter)",
            "why": "cheap hosted payload; sees the cislunar volume from 36 000 km but shares the Earth's "
                   "Sun/Moon geometry and is ~400 000 km from everything"},
    "l1_halo": {"label": "L1 halo observer", "orbit": "L1 northern halo, Az ~ 30 000 km, P ~ 12 d",
                "why": "sits between Earth and Moon: short ranges to L1 gateway traffic and the DRO belt"},
    "l2_halo_S": {"label": "L2 halo observer", "orbit": "L2 southern halo, Az ~ 53 000 km, P ~ 14 d",
                  "why": "watches the far side / L2 gateway and the NRHO corridor from behind the Moon"},
    "dro": {"label": "DRO observer", "orbit": "planar DRO, ~70 000 km from the Moon, P ~ 14.7 d",
            "why": "very stable (no station-keeping), circles the Moon with the Earth-Moon line: steady "
                   "views of lunar-vicinity objects from outside the lunar glare cone"},
    "nrho_9_2": {"label": "NRHO-hosted sensor", "orbit": "9:2 synodic L2 southern NRHO, P ~ 6.56 d, perilune ~ 3 250 km",
                 "why": "rides with Gateway-class traffic; long apolune dwell over the south pole"},
    "resonant_3_1": {"label": "3:1 resonant observer", "orbit": "3:1 Earth-Moon resonant orbit, P ~ 28 d",
                     "why": "tours the Lagrange-point region from a high-apogee Earth orbit"},
}

_MAG_REF_APERTURE_M = 0.5
_MAG_REF_LIMIT = 18.5


def limiting_mag_from_aperture(aperture_m: float) -> float:
    """Aperture-class rule of thumb (see module docstring), clipped to [14, 23]."""
    d = float(aperture_m)
    if d <= 0:
        raise ValueError("aperture_m must be positive")
    return float(min(23.0, max(14.0, _MAG_REF_LIMIT + 5.0 * math.log10(d / _MAG_REF_APERTURE_M))))


def aperture_from_limiting_mag(limiting_mag: float) -> float:
    """Inverse of :func:`limiting_mag_from_aperture` (for cost proxies when only m_lim is given)."""
    return float(_MAG_REF_APERTURE_M * 10.0 ** ((float(limiting_mag) - _MAG_REF_LIMIT) / 5.0))


@dataclass(frozen=True)
class SensorSpec:
    """One hypothetical space-based sensor as typed into the studio.

    ``aperture_m`` and ``limiting_mag`` may each be omitted; the missing one is derived with the
    aperture-class rule of thumb (both omitted -> 0.4 m / 18.0 mag, the notional default used by
    :data:`selene.sensors.observers.DEFAULT_OBSERVERS`-class sensors).
    """

    platform: str
    aperture_m: Optional[float] = None
    limiting_mag: Optional[float] = None
    fov_deg: float = 2.0
    slew_rate_deg_s: float = 1.0
    lon_deg: float = -100.0           # geo only, east-positive
    phase: float = 0.0                # fraction of the orbit period elapsed at the demo epoch
    sun_exclusion_deg: float = 30.0
    earth_exclusion_deg: float = 15.0
    moon_exclusion_deg: float = 5.0
    label: str = ""

    def __post_init__(self):
        if self.platform not in PLATFORMS:
            raise ValueError(f"unknown platform {self.platform!r}; known: {sorted(PLATFORMS)}")
        if self.aperture_m is not None and self.aperture_m <= 0:
            raise ValueError("aperture_m must be positive")
        if self.fov_deg <= 0 or self.fov_deg > 60:
            raise ValueError("fov_deg must be in (0, 60]")
        if self.slew_rate_deg_s <= 0:
            raise ValueError("slew_rate_deg_s must be positive")

    @property
    def resolved_aperture_m(self) -> float:
        if self.aperture_m is not None:
            return float(self.aperture_m)
        if self.limiting_mag is not None:
            return aperture_from_limiting_mag(self.limiting_mag)
        return 0.4

    @property
    def resolved_limiting_mag(self) -> float:
        if self.limiting_mag is not None:
            return float(self.limiting_mag)
        if self.aperture_m is not None:
            return limiting_mag_from_aperture(self.aperture_m)
        return 18.0

    def to_observer(self, index: int = 0, prefix: str = "arch") -> SpaceObserver:
        """Build the :class:`SpaceObserver` this spec describes (id ``<prefix>_<index>_<platform>``)."""
        info = PLATFORM_INFO[self.platform]
        name = self.label or f"{info['label']} ({self.resolved_aperture_m:.2f} m, notional)"
        return SpaceObserver(
            id=f"{prefix}_{index}_{self.platform}", name=name, platform_orbit=PLATFORMS[self.platform],
            limiting_mag=self.resolved_limiting_mag, fov_deg=float(self.fov_deg),
            sun_exclusion_deg=float(self.sun_exclusion_deg), earth_exclusion_deg=float(self.earth_exclusion_deg),
            moon_exclusion_deg=float(self.moon_exclusion_deg), slew_rate_deg_s=float(self.slew_rate_deg_s),
            geo_longitude_deg=float(self.lon_deg), phase=float(self.phase),
            notes=f"ASSUMED notional {self.resolved_aperture_m:.2f} m-class sensor; {info['orbit']}",
        )

    def as_dict(self) -> dict:
        return {
            "platform": self.platform, "aperture_m": self.resolved_aperture_m, "limiting_mag": self.resolved_limiting_mag,
            "fov_deg": self.fov_deg, "slew_rate_deg_s": self.slew_rate_deg_s, "lon_deg": self.lon_deg,
            "phase": self.phase, "label": self.label, "platform_orbit": PLATFORMS[self.platform],
            "orbit": PLATFORM_INFO[self.platform]["orbit"],
        }


@dataclass
class Architecture:
    """A candidate sensor architecture: space sensors + optional notional ground network."""

    name: str
    sensors: list[SensorSpec] = field(default_factory=list)
    ground: bool = True
    notes: str = ""
    ground_sites: Optional[tuple[GroundSite, ...]] = None   # None -> DEFAULT_SITES

    def observers(self) -> list[SpaceObserver]:
        return [s.to_observer(i, prefix=_slug(self.name)) for i, s in enumerate(self.sensors)]

    def all_sensors(self) -> list[Sensor]:
        out: list[Sensor] = []
        if self.ground:
            out.extend(self.ground_sites if self.ground_sites is not None else DEFAULT_SITES)
        out.extend(self.observers())
        if not out:
            raise ValueError(f"architecture {self.name!r} has no sensors (no ground network and no space sensors)")
        return out

    # -- cost proxies -------------------------------------------------------
    @property
    def n_space_sensors(self) -> int:
        return len(self.sensors)

    @property
    def total_aperture_m(self) -> float:
        return float(sum(s.resolved_aperture_m for s in self.sensors))

    @property
    def total_collecting_area_m2(self) -> float:
        return float(sum(math.pi * (0.5 * s.resolved_aperture_m) ** 2 for s in self.sensors))

    def cost_proxies(self) -> dict:
        n_ground = len(self.ground_sites if self.ground_sites is not None else DEFAULT_SITES) if self.ground else 0
        return {
            "n_space_sensors": self.n_space_sensors,
            "n_ground_sites": n_ground,
            "total_space_aperture_m": self.total_aperture_m,
            "total_space_collecting_area_m2": self.total_collecting_area_m2,
            "platforms": [s.platform for s in self.sensors],
            "note": "cost proxies only (counts and summed aperture); no monetary values are asserted",
        }

    def as_dict(self) -> dict:
        return {"name": self.name, "ground": self.ground, "notes": self.notes,
                "sensors": [s.as_dict() for s in self.sensors], **self.cost_proxies()}


def _slug(name: str) -> str:
    s = "".join(c.lower() if c.isalnum() else "_" for c in name).strip("_")
    while "__" in s:
        s = s.replace("__", "_")
    return s or "arch"


def architecture_from_dict(d: dict) -> Architecture:
    """Build an :class:`Architecture` from the API payload shape (unknown keys are rejected)."""
    allowed = {"platform", "aperture_m", "limiting_mag", "fov_deg", "slew_rate_deg_s", "lon_deg", "phase",
               "sun_exclusion_deg", "earth_exclusion_deg", "moon_exclusion_deg", "label"}
    specs = []
    for raw in d.get("sensors", ()) or ():
        if not isinstance(raw, dict):
            raise TypeError("each sensor must be an object")
        extra = set(raw) - allowed
        if extra:
            raise ValueError(f"unknown sensor fields {sorted(extra)}")
        specs.append(SensorSpec(**raw))
    name = str(d.get("name", "")).strip()
    if not name:
        raise ValueError("architecture needs a name")
    return Architecture(name=name, sensors=specs, ground=bool(d.get("ground", True)), notes=str(d.get("notes", "")))


# ---------------------------------------------------------------------------
# presets
# ---------------------------------------------------------------------------
_GEO_SPEC = SensorSpec("geo", aperture_m=0.3, limiting_mag=18.0, fov_deg=3.0, slew_rate_deg_s=1.0)
_HALO_SPEC = SensorSpec("l2_halo_S", aperture_m=0.4, limiting_mag=18.5, fov_deg=2.0, slew_rate_deg_s=1.0)
_DRO_SPEC = SensorSpec("dro", aperture_m=0.4, limiting_mag=18.5, fov_deg=2.0, slew_rate_deg_s=1.0)
_L1_SPEC = SensorSpec("l1_halo", aperture_m=0.4, limiting_mag=18.5, fov_deg=2.0, slew_rate_deg_s=1.0)

PRESET_ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture(
        "Ground only", [], ground=True,
        notes="Baseline: the notional 9-site ground optical network alone. Cheapest, but blinded by daylight, "
              "weather-free idealised skies notwithstanding, and by the lunar-glare cone that hides most xGEO "
              "objects near full Moon; this is the status quo the market problem describes.",
    ),
    Architecture(
        "Ground + 2 GEO", [replace(_GEO_SPEC, lon_deg=-100.0), replace(_GEO_SPEC, lon_deg=60.0)], ground=True,
        notes="Lowest-cost space augmentation: two 0.3 m-class hosted payloads on GEO buses (100 W, 60 E). "
              "Removes the daylight gap but keeps Earth's viewing geometry, so the Moon-glare and range "
              "limits remain largely in place.",
    ),
    Architecture(
        "Ground + L2 halo", [_HALO_SPEC], ground=True,
        notes="One 0.4 m-class sensor on an L2 southern halo: watches the L2 gateway, the far side and the NRHO "
              "corridor from behind the Moon with short ranges; needs station-keeping and a relay link.",
    ),
    Architecture(
        "Ground + DRO + L1 halo", [_DRO_SPEC, _L1_SPEC], ground=True,
        notes="Two-platform cislunar layer: a DRO sensor (stable, no station-keeping) circling the Moon plus an "
              "L1 halo sensor covering the near-side gateway. Views from outside the Earth-Moon line fill the "
              "ground network's glare and daylight blind spots from two directions.",
    ),
)


def preset_architectures() -> list[Architecture]:
    """Fresh copies of the four preset architectures."""
    return [Architecture(a.name, list(a.sensors), a.ground, a.notes) for a in PRESET_ARCHITECTURES]


def sensors_for(arch: Architecture) -> Sequence[Sensor]:   # small alias used by montecarlo.py
    return arch.all_sensors()
