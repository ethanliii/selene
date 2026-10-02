"""FastAPI application factory. Serves /api/* and (if built) the static frontend.

Operational contract (see README §2 "API"):

* unknown ``/api/...`` paths answer a JSON 404 (never the SPA shell), and ``/api/x/`` redirects to
  ``/api/x``;
* a missing DE440s kernel (``data/cache/de440s.bsp``) is a 503 with the download hint on every
  route that needs it, and ``/api/health`` reports ``status: degraded`` with the missing file;
* an epoch outside the kernel coverage is a 400 naming the valid span;
* CORS origins come from ``SELENE_CORS_ORIGINS`` (comma-separated; ``*`` for wide open).  The default
  is the local dev/demo origins only: the Vite dev server proxies ``/api`` and the built UI is served
  by this process, so neither needs cross-origin access.
"""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse, JSONResponse, RedirectResponse

from selene import __version__
from selene.constants import DATA_DIR, DE440S_PATH, GM_TPC_PATH

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_CORS_ORIGINS = ["http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:8000", "http://localhost:8000"]
SPA_METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"]


def cors_origins() -> list[str]:
    raw = os.environ.get("SELENE_CORS_ORIGINS", "").strip()
    if not raw:
        return list(DEFAULT_CORS_ORIGINS)
    return [o.strip() for o in raw.split(",") if o.strip()]


def data_status() -> dict:
    """Presence of the cached inputs the product needs to run offline (the health route reports this)."""
    orbits = DATA_DIR / "orbits" / "orbits.parquet"
    demo = DATA_DIR / "demo" / "scenario.json"
    horizons = DATA_DIR / "horizons" / "index.json"
    files = {
        "de440s_kernel": DE440S_PATH, "gm_de440": GM_TPC_PATH, "orbit_library": orbits,
        "horizons_cache": horizons, "demo_bundle": demo,
    }
    present = {k: p.exists() for k, p in files.items()}
    missing = [k for k, ok in present.items() if not ok]
    required_missing = [k for k in ("de440s_kernel", "orbit_library") if not present[k]]
    return {"present": present, "paths": {k: str(p) for k, p in files.items()}, "missing": missing,
            "required_missing": required_missing}


def api_paths(app: FastAPI) -> list[str]:
    """Sorted ``/api/...`` paths from the OpenAPI schema (FastAPI >= 0.142 wraps included routers in
    ``_IncludedRouter`` objects without a ``path``, so the schema is the one reliable, prefix-resolved listing;
    it is cached by FastAPI after the first call)."""
    try:
        paths = app.openapi().get("paths", {})
    except Exception:  # pragma: no cover - schema generation failure must not break /health
        return []
    return sorted(p for p in paths if p.startswith("/api/"))


def create_app() -> FastAPI:
    app = FastAPI(title="SELENE cislunar SDA API", version=__version__)
    origins = cors_origins()
    app.add_middleware(
        CORSMiddleware, allow_origins=origins, allow_methods=["*"], allow_headers=["*"],
        allow_credentials=False,
    )

    # --- error shaping ---------------------------------------------------------------------
    @app.exception_handler(FileNotFoundError)
    async def _missing_data(request: Request, exc: FileNotFoundError):
        """A cached input is missing (kernel / library / bundle): 503 with the hint, not a bare 500."""
        msg = str(exc)
        if str(DATA_DIR) in msg or "DE440s" in msg or "de440s" in msg:
            return JSONResponse(status_code=503, content={
                "detail": msg, "status": "degraded",
                "hint": "run `make setup` (downloads the JPL kernels into data/cache) or mount data/ into the container",
                "data": data_status(),
            })
        raise exc

    try:
        from jplephem.exceptions import OutOfRangeError

        @app.exception_handler(OutOfRangeError)
        async def _out_of_range(request: Request, exc: OutOfRangeError):
            return JSONResponse(status_code=400, content={"detail": f"epoch outside the DE440s ephemeris coverage: {exc}"})
    except ImportError:  # pragma: no cover
        pass

    @app.get("/api/health")
    def health():
        """Liveness + data readiness.  ``status`` is ``ok`` when every required cached input is present,
        ``degraded`` otherwise (the core routes then answer 503).  ``offline`` is *computed*: True when the
        full offline data set (kernels, orbit library, Horizons cache, demo bundle) is on disk."""
        ds = data_status()
        return {
            "status": "ok" if not ds["required_missing"] else "degraded",
            "version": __version__,
            "offline": not ds["missing"],
            "offline_meaning": "True when every cached input needed to run with no network is present (data.present)",
            "ephemeris": DE440S_PATH.name if DE440S_PATH.exists() else None,
            "data": ds,
            "routers_loaded": list(getattr(app.state, "routers_loaded", [])),
            "routes": api_paths(app),
            "cors_origins": origins,
        }

    # Routers are registered lazily so a missing optional module never breaks the app.
    from selene.api.routes import register_routes

    app.state.routers_loaded = register_routes(app)

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.api_route("/{full_path:path}", methods=SPA_METHODS, include_in_schema=False)
    def spa(full_path: str, request: Request):
        """SPA fallback for the built UI.  Never shadows the API: an unknown ``/api/...`` path is a JSON
        404 (a trailing-slash variant redirects to the canonical path first)."""
        if full_path == "api" or full_path.startswith("api/"):
            if full_path.endswith("/") and len(full_path) > 4:
                url = "/" + full_path.rstrip("/")
                if request.url.query:
                    url += "?" + request.url.query
                return RedirectResponse(url, status_code=307)
            raise HTTPException(404, detail=f"unknown API route {request.method} /{full_path}; see /docs or GET /api/health -> routes")
        if request.method not in ("GET", "HEAD"):
            raise HTTPException(405, detail="static assets accept GET/HEAD only")
        if not STATIC_DIR.exists():
            raise HTTPException(404, detail="frontend not built; run `make build` (API routes live under /api)")
        target = STATIC_DIR / full_path
        if full_path and target.is_file():
            return FileResponse(target)
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
