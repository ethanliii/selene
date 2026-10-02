"""FastAPI application factory. Serves /api/* and (if built) the static frontend."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse

from selene import __version__
from selene.constants import DE440S_PATH

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app() -> FastAPI:
    app = FastAPI(title="SELENE cislunar SDA API", version=__version__)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "offline": True,
            "ephemeris": DE440S_PATH.name if DE440S_PATH.exists() else None,
        }

    # Routers are registered lazily so a missing optional module never breaks the app.
    from selene.api.routes import register_routes

    register_routes(app)

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

        @app.get("/{full_path:path}")
        def spa(full_path: str):
            target = STATIC_DIR / full_path
            if full_path and target.is_file():
                return FileResponse(target)
            return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
