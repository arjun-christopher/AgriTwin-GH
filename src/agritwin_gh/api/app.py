"""
FastAPI application factory for AgriTwin-GH.

Responsibility
--------------
* Create and configure the ``FastAPI`` application instance.
* Register all routers under the ``/api`` prefix.
* Attach the lifespan context (startup / shutdown hooks in ``core/lifespan.py``).
* Configure CORS so the Vite dev server (port 5173) and production build can
  both reach the API.

Usage
-----
The app object is imported by ``main.py`` (repo root) which starts Uvicorn::

    import uvicorn
    from agritwin_gh.api.app import create_app

    app = create_app()

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8000)

Do not add business logic here.  All endpoint logic lives in route modules
under ``agritwin_gh.api.routes``.

Implementation note
-------------------
``create_app()`` is a factory function rather than a module-level ``app``
to make testing easier — each test can spin up a fresh isolated app instance.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from agritwin_gh.core.lifespan import lifespan
from agritwin_gh.api.routes import (
    dt,
    actuators,
    weather,
    intelligence,
    resources,
    media,
    system,
    greenhouse_3d,
)

# Absolute path to the Unity WebGL build directory.
_BUILD_DIR = Path(__file__).parent.parent / "build"

# Unity WebGL Brotli file → (media_type, Content-Encoding) mapping.
_WEBGL_BROTLI_TYPES: dict[str, str] = {
    ".js.br":   "application/javascript",
    ".wasm.br": "application/wasm",
    ".data.br": "application/octet-stream",
}
_WEBGL_GZIP_TYPES: dict[str, str] = {
    ".js.gz":   "application/javascript",
    ".wasm.gz": "application/wasm",
    ".data.gz": "application/octet-stream",
}


def _webgl_response(file_path: Path) -> FileResponse:
    """Return a FileResponse with correct headers for Unity WebGL files.

    Unity compresses its build artefacts with Brotli (*.br) or gzip (*.gz).
    A plain static-file server will send these as raw bytes with a generic
    Content-Type, causing the Unity loader to fail.  This helper adds:
      * ``Content-Encoding: br / gzip`` so the browser decompresses them.
      * The correct ``Content-Type`` for the underlying asset (JS / wasm / data).
    """
    name = file_path.name.lower()
    for suffix, media_type in _WEBGL_BROTLI_TYPES.items():
        if name.endswith(suffix):
            return FileResponse(
                str(file_path),
                media_type=media_type,
                headers={"Content-Encoding": "br"},
            )
    for suffix, media_type in _WEBGL_GZIP_TYPES.items():
        if name.endswith(suffix):
            return FileResponse(
                str(file_path),
                media_type=media_type,
                headers={"Content-Encoding": "gzip"},
            )
    return FileResponse(str(file_path))


# Origins allowed by CORS.  Add your production domain here when deploying.
_CORS_ORIGINS = [
    "http://localhost:5173",   # Vite dev server
    "http://localhost:4173",   # Vite preview
    "http://localhost:8000",   # Same-origin (unlikely, but safe to include)
]


def create_app() -> FastAPI:
    """Construct and return the configured FastAPI application."""

    app = FastAPI(
        title="AgriTwin-GH API",
        description=(
            "FastAPI backend for the AgriTwin-GH Digital-Twin greenhouse "
            "control system.  Wraps the MPC solver, DT loop, and ML models."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── CORS ────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # ── Routers ─────────────────────────────────────────────────────────────
    # All routers publish under the /api prefix so the frontend api.js
    # BASE_URL can remain 'http://localhost:8000' without a path prefix.
    _prefix = "/api"
    app.include_router(dt.router,           prefix=_prefix)
    app.include_router(actuators.router,    prefix=_prefix)
    app.include_router(weather.router,      prefix=_prefix)
    app.include_router(intelligence.router, prefix=_prefix)
    app.include_router(resources.router,    prefix=_prefix)
    app.include_router(media.router,        prefix=_prefix)
    app.include_router(system.router,       prefix=_prefix)
    app.include_router(greenhouse_3d.router, prefix=_prefix)

    # ── Unity WebGL build — served at /greenhouse-3d ─────────────────────
    # Registered as explicit FastAPI routes (not StaticFiles) so that we can
    # inject the correct "Content-Encoding: br" header for Brotli-compressed
    # Unity artefacts.  StaticFiles has no hook for per-file response headers.
    if _BUILD_DIR.exists():
        _build_root = _BUILD_DIR.resolve()

        @app.get("/greenhouse-3d", include_in_schema=False)
        @app.get("/greenhouse-3d/", include_in_schema=False)
        async def _webgl_index() -> FileResponse:  # noqa: RUF029
            return FileResponse(str(_build_root / "index.html"))

        @app.get("/greenhouse-3d/{file_path:path}", include_in_schema=False)
        async def _webgl_static(file_path: str) -> Response:  # noqa: RUF029
            # Resolve and security-check the target path.
            target = (_build_root / file_path).resolve()
            if not str(target).startswith(str(_build_root)):
                return Response(status_code=404)
            if not target.exists() or not target.is_file():
                # SPA fallback — let Unity's loader handle unknown paths.
                return FileResponse(str(_build_root / "index.html"))
            return _webgl_response(target)

    return app
