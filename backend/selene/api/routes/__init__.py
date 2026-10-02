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


def register_routes(app: FastAPI) -> list[str]:
    """Include every importable router; returns the names that loaded (reported by /api/health)."""
    loaded: list[str] = []
    for name in ROUTE_MODULES:
        try:
            mod = importlib.import_module(f"selene.api.routes.{name}")
        except ModuleNotFoundError as e:  # route not implemented yet
            if e.name and e.name.startswith("selene.api.routes"):
                log.warning("route module %s not found; skipped", name)
                continue
            raise
        app.include_router(mod.router, prefix="/api")
        loaded.append(name)
    return loaded
