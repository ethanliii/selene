"""Route registry. Each submodule exposes ``router``; add names to ROUTE_MODULES."""
from __future__ import annotations

import importlib
import logging

from fastapi import FastAPI

log = logging.getLogger(__name__)

ROUTE_MODULES = [
    "catalog", "orbits", "ephemeris", "sensors", "coverage", "od", "maneuver",
    "reachability", "tasking", "architecture", "demo",
]


def register_routes(app: FastAPI) -> None:
    for name in ROUTE_MODULES:
        try:
            mod = importlib.import_module(f"selene.api.routes.{name}")
        except ModuleNotFoundError as e:  # route not implemented yet
            if e.name and e.name.startswith("selene.api.routes"):
                continue
            raise
        app.include_router(mod.router, prefix="/api")
