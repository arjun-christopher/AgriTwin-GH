"""
System health routes — ``/api/system/*``.

Endpoints
---------
GET /api/system/health → ``SystemHealthResponse``

Responsibility
--------------
Surfaces the operational health of each system component for the Dashboard's
health-panel widget.  Checked components:

  * Temperature / humidity / CO₂ / soil sensors — DB row staleness check.
  * MPC solver — reports last solve time and convergence status.
  * Disease classifier — last inference timestamp and model availability.
  * Growth-stage classifier — last inference timestamp.
  * MinIO connection — bucket reachability.
  * PostgreSQL connection — simple SELECT 1 heartbeat.
  * Weather forecast model — last update timestamp.

Health is derived from ``RuntimeStore`` metadata without making live DB
calls on every request (the store is updated asynchronously by the DT loop).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_system_service
from agritwin_gh.schemas.system_schemas import SystemHealthResponse
from agritwin_gh.services.system_service import SystemService

logger = logging.getLogger("agritwin.api.system")

router = APIRouter(tags=["System"])


@router.get("/system/health", response_model=SystemHealthResponse, status_code=200)
async def get_system_health(
    svc: SystemService = Depends(get_system_service),
) -> SystemHealthResponse:
    """Return sensor array and pipeline health status.

    Checks six subsystems (DT Loop, MPC Solver, Sensor Feed, Database,
    MinIO Storage, AI Models) without making live DB or solver calls.
    The overall ``status`` field is the worst-case across all rows.
    """
    try:
        return svc.get_health()
    except Exception as exc:
        logger.error("GET /system/health failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="System health check temporarily unavailable.")
