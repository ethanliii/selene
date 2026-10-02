"""Validated request model for ad-hoc space observers (``POST /api/coverage`` custom networks and
``POST /api/tasking/schedule`` ``extra_sensors``).

Before this model existed both routes passed raw dicts to ``SpaceObserver(**kw)``: a dict missing
``name``/``platform_orbit`` raised ``TypeError`` (an unhandled 500 in the coverage route) and an
unknown key leaked the Python constructor signature into a 400.  Pydantic now rejects both with a
field-level message, and the studio's ``orbit`` alias / ``kind`` / ``spec_note`` echo keys are
accepted (``kind`` and ``spec_note`` are what :meth:`SpaceObserver.as_dict` emits, so a sensor
listed by ``GET /api/sensors`` can be posted straight back).
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from selene.sensors.observers import PLATFORM_ORBITS, SpaceObserver

PlatformOrbit = Literal["geo", "l1_halo", "l2_halo", "dro", "nrho", "resonant", "custom"]
assert set(PlatformOrbit.__args__) == set(PLATFORM_ORBITS)  # keep the Literal in sync with the dataclass


class SpaceObserverIn(BaseModel):
    """Keyword arguments of :class:`selene.sensors.observers.SpaceObserver`, validated."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, max_length=64)
    name: Optional[str] = Field(None, max_length=120, description="defaults to id")
    platform_orbit: PlatformOrbit = Field(..., description=f"one of {list(PLATFORM_ORBITS)} (alias: orbit)")
    orbit_ref: Optional[str] = None
    limiting_mag: float = Field(18.0, ge=8.0, le=30.0)
    fov_deg: float = Field(2.0, gt=0.0, le=180.0)
    sun_exclusion_deg: float = Field(30.0, ge=0.0, le=180.0)
    earth_exclusion_deg: float = Field(15.0, ge=0.0, le=180.0)
    moon_exclusion_deg: float = Field(5.0, ge=0.0, le=180.0)
    slew_rate_deg_s: float = Field(1.0, gt=0.0, le=360.0)
    geo_longitude_deg: float = Field(-100.0, ge=-360.0, le=360.0)
    phase: float = Field(0.0, ge=0.0, lt=1.0)
    epoch_s: Optional[float] = None
    custom_ic: Optional[list[float]] = Field(None, min_length=6, max_length=6)
    custom_period: Optional[float] = Field(None, gt=0.0)
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def _aliases(cls, v: Any) -> Any:
        if not isinstance(v, dict):
            return v
        d = dict(v)
        d.pop("kind", None)        # echoes of SpaceObserver.as_dict()
        d.pop("spec_note", None)
        d.pop("orbit_summary", None)
        if "orbit" in d:
            orbit = d.pop("orbit")
            if "platform_orbit" in d and d["platform_orbit"] != orbit:
                raise ValueError("give either 'platform_orbit' or its alias 'orbit', not both")
            d["platform_orbit"] = orbit
        return d

    @model_validator(mode="after")
    def _custom(self):
        if self.platform_orbit == "custom" and (self.custom_ic is None or self.custom_period is None):
            raise ValueError("platform_orbit 'custom' needs custom_ic (6 nondimensional rotating-frame values) and custom_period")
        return self

    def build(self) -> SpaceObserver:
        kw = self.model_dump(exclude_none=True)
        kw.setdefault("name", self.id)
        if "custom_ic" in kw:
            kw["custom_ic"] = tuple(float(x) for x in kw["custom_ic"])
        return SpaceObserver(**kw)
