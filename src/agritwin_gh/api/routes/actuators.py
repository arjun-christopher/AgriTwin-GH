"""
Actuator routes — ``/api/actuators/*``.

Endpoints
---------
GET  /api/actuators/state   → ``ActuatorStateResponse``
POST /api/actuators/set     → ``ActuatorSetResponse``

Responsibility
--------------
* ``GET /api/actuators/state`` returns the current actuator levels and
  on/off states sourced from ``RuntimeStore`` via ``DashboardService``.

* ``POST /api/actuators/set`` accepts a batch of actuator-level overrides
  and applies them via ``ControlService``.  Overrides bypass the MPC solver
  until the next automatic re-solve.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_control_service, get_dashboard_service
from agritwin_gh.schemas.actuator_schemas import (
    ActuatorSetRequest,
    ActuatorSetResponse,
    ActuatorStateResponse,
)
from agritwin_gh.services.control_service import ControlService
from agritwin_gh.services.dashboard_service import DashboardService

logger = logging.getLogger("agritwin.api.actuators")

router = APIRouter(tags=["Actuators"])


@router.get("/actuators/state", response_model=ActuatorStateResponse, status_code=200)
async def get_actuator_state(
    svc: DashboardService = Depends(get_dashboard_service),
) -> ActuatorStateResponse:
    """Return current actuator states and levels.

    Sourced from the singleton ``RuntimeStore`` via ``DashboardService``.
    Returns a safe zero-state payload when the DT loop has not yet run.
    """
    try:
        return svc.get_actuator_state()
    except Exception as exc:
        logger.error("GET /actuators/state failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Actuator state temporarily unavailable.")


@router.post("/actuators/set", response_model=ActuatorSetResponse, status_code=200)
async def post_actuator_set(
    body: ActuatorSetRequest,
    svc: ControlService = Depends(get_control_service),
) -> ActuatorSetResponse:
    """Apply manual actuator level overrides.

    Overrides are merged into the active ``OverrideConfig`` immediately.
    The MPC solver sees them at the next re-solve cadence.

    Body: ``{"actuators": [{"id": "fan", "level": 75.0}, ...]}``
    """
    try:
        return svc.apply_actuator_set(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("POST /actuators/set failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Actuator override could not be applied.")
