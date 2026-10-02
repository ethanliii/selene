"""Demo-scenario routes: ``GET /api/demo/scenario``, ``/scenario/meta``, ``/brief`` and ``POST /api/demo/rebuild``.

The scenario bundle is precomputed by ``python -m selene.scenario.demo --rebuild`` and committed as
``data/demo/scenario.json`` (+ ``scenario_meta.json``) so the demo works offline and the first request
is served from disk in well under two seconds; the parsed bundle is then cached in memory (keyed by the
file's mtime, so a rebuild on disk is picked up without a restart).  When the file is missing the
routes answer ``404`` with the exact rebuild command.

``POST /api/demo/rebuild?fast=true`` runs :func:`selene.scenario.demo.build_scenario` **synchronously
in the request** (a few seconds in fast mode on a laptop) and rewrites the bundle; it is safe to call
repeatedly (a process-wide lock serialises concurrent rebuilds and the file is written atomically).  Only
the fast build is allowed through the route (``fast=false`` answers 400 with the CLI command): the full
build is the committed artefact and is produced by ``python -m selene.scenario.demo --rebuild``.  Note
that a rebuild REPLACES the served file (``data/demo/scenario.json`` unless ``SELENE_DEMO_SCENARIO`` points
elsewhere), so after rehearsing with the route restore the full bundle with ``git checkout data/demo``.
Because of that the route answers ``403`` unless ``SELENE_ALLOW_REBUILD=1`` is set or ``SELENE_DEMO_SCENARIO``
redirects the bundle to a scratch path (``make demo`` sets neither, so a stray call during a live demo
cannot replace the story).  It is meant for development and the pitch rehearsal, not for a public deployment.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse

from selene.scenario.demo import META_PATH, SCENARIO_PATH, build_scenario, write_bundle

router = APIRouter(prefix="/demo", tags=["demo"])

_LOCK = threading.Lock()
_CACHE: dict[str, Any] = {"path": None, "mtime": None, "bundle": None, "loaded_in_s": None}

REBUILD_HINT = ("demo scenario bundle not found; build it with "
                "`cd backend && ../.venv/bin/python -m selene.scenario.demo --rebuild` (add --fast for a ~10 s build) "
                "or POST /api/demo/rebuild?fast=true")


def _paths() -> tuple[Path, Path]:
    """Bundle paths (overridable through SELENE_DEMO_SCENARIO for tests / scratch rebuilds)."""
    env = os.environ.get("SELENE_DEMO_SCENARIO")
    if env:
        p = Path(env)
        return p, p.with_name(p.stem + "_meta.json")
    return SCENARIO_PATH, META_PATH


def rebuild_allowed() -> bool:
    """``POST /api/demo/rebuild`` is unauthenticated and overwrites the served bundle, so it is off unless
    ``SELENE_ALLOW_REBUILD`` is set or the bundle path was redirected away from the committed file."""
    flag = os.environ.get("SELENE_ALLOW_REBUILD", "").strip().lower()
    return flag in ("1", "true", "yes", "on") or bool(os.environ.get("SELENE_DEMO_SCENARIO"))


def load_bundle(force: bool = False) -> dict:
    """The parsed bundle, cached in memory until the file on disk changes."""
    path, _ = _paths()
    if not path.exists():
        raise HTTPException(404, detail=REBUILD_HINT)
    mtime = path.stat().st_mtime
    with _LOCK:
        if not force and _CACHE["bundle"] is not None and _CACHE["path"] == str(path) and _CACHE["mtime"] == mtime:
            return _CACHE["bundle"]
        tic = time.perf_counter()
        try:
            bundle = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise HTTPException(500, detail=f"demo scenario bundle at {path} is not valid JSON ({e}); rebuild it")
        _CACHE.update(path=str(path), mtime=mtime, bundle=bundle, loaded_in_s=time.perf_counter() - tic)
        return bundle


@router.get("/scenario")
def scenario():
    """The full precomputed story: ``meta``, hourly ``frames``, ``events``, ``metrics`` and the ``brief``."""
    return load_bundle()


@router.get("/scenario/meta")
def scenario_meta():
    """Metadata + metrics only (small): what the frontend needs before streaming the frames."""
    path, meta_path = _paths()
    if not path.exists():
        raise HTTPException(404, detail=REBUILD_HINT)
    if meta_path.exists():
        try:
            d = json.loads(meta_path.read_text())
            d["bundle_path"] = str(path)
            d["bundle_bytes"] = path.stat().st_size if path.exists() else None
            d["served_from_cache"] = _CACHE["bundle"] is not None and _CACHE["path"] == str(path)
            return d
        except json.JSONDecodeError:
            pass
    b = load_bundle()
    return {"meta": b["meta"], "metrics": b.get("metrics", {}), "n_events": len(b.get("events", [])),
            "n_frames": len(b.get("frames", [])), "bundle_path": str(path), "bundle_bytes": path.stat().st_size,
            "served_from_cache": True}


@router.get("/brief", response_class=PlainTextResponse)
def brief():
    """The auto-generated analyst brief as Markdown (``text/markdown``)."""
    b = load_bundle()
    return PlainTextResponse(b.get("brief", ""), media_type="text/markdown; charset=utf-8")


@router.post("/rebuild")
def rebuild(fast: bool = Query(True, description="must be true: reduced sample counts (~10 s); the full build is CLI-only"),
            seed: int = Query(0, ge=0, le=10_000)):
    """Rebuild the bundle synchronously with ``fast=True`` (see module docstring) and return its meta block."""
    if not fast:
        raise HTTPException(400, detail="POST /api/demo/rebuild only runs the fast build; for the full bundle run "
                                        "`cd backend && ../.venv/bin/python -m selene.scenario.demo --rebuild`")
    if not rebuild_allowed():
        raise HTTPException(403, detail="demo rebuild over HTTP is disabled: it would overwrite the committed full-fidelity bundle "
                                        "data/demo/scenario.json with a fast build during a live demo. Enable it with "
                                        "SELENE_ALLOW_REBUILD=1, or point SELENE_DEMO_SCENARIO at a scratch path "
                                        "(rebuilds then write there), or run `python -m selene.scenario.demo --rebuild --fast --out <path>`")
    path, meta_path = _paths()
    tic = time.perf_counter()
    with _LOCK:
        try:
            bundle = build_scenario(seed=seed, fast=fast)
        except Exception as e:  # noqa: BLE001 - surfaced to the caller, never a 500 without a message
            raise HTTPException(500, detail=f"demo scenario build failed: {type(e).__name__}: {e}")
        tmp = path.with_suffix(".json.tmp")
        tmp_meta = meta_path.with_suffix(".json.tmp")
        write_bundle(bundle, tmp, tmp_meta)
        os.replace(tmp, path)
        os.replace(tmp_meta, meta_path)
        _CACHE.update(path=str(path), mtime=path.stat().st_mtime, bundle=bundle, loaded_in_s=0.0)
    m = bundle["metrics"]
    return {"status": "ok", "fast": fast, "seed": seed, "elapsed_s": round(time.perf_counter() - tic, 2),
            "note": f"replaced {path}; restore the committed full bundle with `git checkout data/demo` if this is the repository copy",
            "bundle_path": str(path), "bundle_bytes": path.stat().st_size, "n_frames": len(bundle["frames"]),
            "n_events": len(bundle["events"]), "meta": bundle["meta"],
            "headline": {"detection_latency_h": m.get("detection_latency_h"), "t_lost_utc": m.get("t_lost_utc"),
                         "t_regained_utc": m.get("t_regained_utc"), "dv_true_mps": m.get("dv_true_mps"),
                         "dv_est_mps": m.get("dv_est_mps"), "dv_est_err_pct": m.get("dv_est_err_pct"),
                         "dir_err_deg": m.get("dir_err_deg")}}
