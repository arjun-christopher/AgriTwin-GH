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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agritwin_gh.core.lifespan import lifespan
from agritwin_gh.api.routes import (
    dt,
    actuators,
    weather,
    intelligence,
    resources,
    media,
    system,
)

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

    return app
