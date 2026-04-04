"""
FastAPI dependency providers for AgriTwin-GH.

Responsibility
--------------
* ``get_db_session`` — yields a SQLAlchemy ``Session`` for endpoints that
  need database access (image streaming, realtime MPCRunner).
* ``get_runtime_store`` — yields the singleton ``RuntimeStore`` that holds
  the in-memory DT step state between HTTP requests.
* ``get_media_service`` — yields a ``MediaService`` bound to the current
  session and a MinIO client; injected into media endpoints.

All dependencies are FastAPI-compatible callables (generators or plain
functions) decorated with ``@lru_cache`` where appropriate for singletons.

Usage in a route
----------------
::

    from fastapi import Depends
    from agritwin_gh.api.dependencies import get_runtime_store, get_dashboard_service
    from agritwin_gh.services.dashboard_service import DashboardService

    @router.get("/dt/state")
    async def dt_state(
        svc: DashboardService = Depends(get_dashboard_service),
    ):
        ...
"""

from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from fastapi import Depends

from agritwin_gh.core.runtime_store import RuntimeStore, get_store
from agritwin_gh.services.control_service import ControlService
from agritwin_gh.services.dashboard_service import DashboardService
from agritwin_gh.services.media_service import MediaService
from agritwin_gh.services.system_service import SystemService
from agritwin_gh.utils.database import DatabaseManager

# ── Singleton instances ───────────────────────────────────────────────────────

_db_manager: DatabaseManager | None = None


def _get_db_manager() -> DatabaseManager:
    """Return the singleton DatabaseManager, creating it on first call."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency: yield a fresh SQLAlchemy session per request.

    The session is closed (and the connection returned to the pool) after
    the response is sent, even if an exception occurs.

    Inject with::

        session: Session = Depends(get_db_session)
    """
    mgr = _get_db_manager()
    session: Session = mgr.get_session()
    try:
        yield session
    finally:
        session.close()


def get_runtime_store() -> RuntimeStore:
    """FastAPI dependency: return the singleton in-memory DT state store.

    The store persists between requests within one server process.  It holds
    the latest ``GreenhouseState``, actuator levels, and derived scalars so
    that ``GET /api/dt/state`` does not need to replay the full DT loop on
    every HTTP request.

    Inject with::

        store: RuntimeStore = Depends(get_runtime_store)
    """
    return get_store()


def get_dashboard_service(
    store: RuntimeStore = Depends(get_runtime_store),
) -> DashboardService:
    """FastAPI dependency: return a ``DashboardService`` bound to the singleton store."""
    return DashboardService(store=store)


def get_control_service(
    store: RuntimeStore = Depends(get_runtime_store),
) -> ControlService:
    """FastAPI dependency: return a ``ControlService`` bound to the singleton store."""
    return ControlService(store=store)


def get_media_service(
    store: RuntimeStore = Depends(get_runtime_store),
    session: Session = Depends(get_db_session),
) -> MediaService:
    """FastAPI dependency: return a ``MediaService`` bound to the store and DB session."""
    return MediaService(store=store, session=session)


def get_system_service(
    store: RuntimeStore = Depends(get_runtime_store),
) -> SystemService:
    """FastAPI dependency: return a ``SystemService`` bound to the singleton store."""
    return SystemService(store=store)

