"""
Digital-Twin routes — ``/api/dt/*``.

Endpoints
---------
GET    /api/dt/state            → ``DTStateResponse``
POST   /api/dt/override         → ``DtOverrideResponse``  (single-param variant)
POST   /api/dt/override/sim     → ``DtOverrideResponse``  (sim-params variant)
DELETE /api/dt/override         → ``DtOverrideResponse``  (clear all overrides)
POST   /api/dt/preset/{id}      → ``DtPresetResponse``

Responsibility
--------------
* ``GET /api/dt/state`` is the primary polling endpoint called by the
  React dashboard every few seconds.  It reads the latest DT step result
  from ``RuntimeStore`` via ``DashboardService`` and assembles the full
  ``DTStateResponse``, including the forward-compatible ``SceneContext``
  block for 3D integration.

* ``POST /api/dt/override`` accepts a manual parameter override
  (e.g. ``temperature_setpoint``) and forwards it to ``ControlService``.
  The override is applied at the next DT step.

* ``POST /api/dt/override/sim`` overrides the simulation stage, day-in-stage,
  start date, and start hour for time-travel / replay scenarios.

* ``DELETE /api/dt/override`` clears all active overrides (param, sim, preset).

* ``POST /api/dt/preset/{preset_id}`` applies a named quick-action preset
  (``day-cycle`` | ``night-cycle`` | ``emergency-flush``) via ``ControlService``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_control_service, get_dashboard_service
from agritwin_gh.schemas.dt_schemas import (
    DTStateResponse,
    DtOverrideResponse,
    DtParamOverrideRequest,
    DtPresetResponse,
    DtSimOverrideRequest,
)
from agritwin_gh.services.control_service import ControlService
from agritwin_gh.services.dashboard_service import DashboardService

logger = logging.getLogger("agritwin.api.dt")

router = APIRouter(tags=["Digital Twin"])


@router.get("/dt/state", response_model=DTStateResponse, status_code=200)
async def get_dt_state(
    svc: DashboardService = Depends(get_dashboard_service),
) -> DTStateResponse:
    """Return the current digital-twin snapshot.

    Reads the latest step result from ``RuntimeStore`` via ``DashboardService``.
    If the store is empty (server just started), returns a safe zero-state
    payload so the frontend renders without errors.

    The response includes the forward-compatible 3D-ready fields:
    ``timestamp``, ``time_of_day``, ``current_growth_stage``,
    ``next_growth_stage``, ``actuator_visual_state``, and ``scene_context``.
    """
    try:
        return svc.get_dt_state()
    except Exception as exc:
        logger.error("GET /dt/state failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="DT state temporarily unavailable.")


@router.post("/dt/override", response_model=DtOverrideResponse, status_code=200)
async def post_dt_override(
    body: DtParamOverrideRequest,
    svc: ControlService = Depends(get_control_service),
) -> DtOverrideResponse:
    """Inject a single-parameter manual override into the DT loop.

    The override is written to ``RuntimeStore`` immediately and picked up by
    the next DT step.  Merges with any existing param overrides so concurrent
    overrides are not clobbered.

    Body: ``{"param": "temperature_setpoint", "value": 26.0}``
    """
    try:
        return svc.apply_dt_param_override(body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("POST /dt/override failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Override could not be applied.")


@router.post("/dt/override/sim", response_model=DtOverrideResponse, status_code=200)
async def post_dt_sim_override(
    body: DtSimOverrideRequest,
    svc: ControlService = Depends(get_control_service),
) -> DtOverrideResponse:
    """Override simulation stage, day-in-stage, start date, and start hour.

    Enables time-travel / replay scenarios from the operator console.
    ``body.stage`` must be a valid canonical growth-stage label.

    Body: ``{"stage": "flowering", "day_in_stage": 5, "start_date": "2026-03-01", "start_hour": 6}``
    """
    import asyncio as _asyncio  # noqa: PLC0415
    try:
        # Run the sync service (which calls TF CNN) in the event-loop's default
        # thread executor so the event loop is never blocked during inference.
        _loop = _asyncio.get_running_loop()
        return await _loop.run_in_executor(None, svc.apply_dt_sim_override, body)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("POST /dt/override/sim failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Sim override could not be applied.")


@router.delete("/dt/override", response_model=DtOverrideResponse, status_code=200)
async def delete_dt_override(
    svc: ControlService = Depends(get_control_service),
) -> DtOverrideResponse:
    """Clear all active DT overrides (param, sim, and preset).

    After this call the DT loop resumes fully autonomous operation.
    """
    try:
        svc.clear_override()
        return DtOverrideResponse(
            ok=True,
            applied_param="",
            applied_value=0.0,
            applied_stage="",
            message="All overrides cleared. DT loop resuming autonomous operation.",
        )
    except Exception as exc:
        logger.error("DELETE /dt/override failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Could not clear overrides.")


@router.post("/dt/preset/{preset_id}", response_model=DtPresetResponse, status_code=200)
async def post_dt_preset(
    preset_id: str,
    svc: ControlService = Depends(get_control_service),
) -> DtPresetResponse:
    """Apply a named quick-action preset.

    Valid preset IDs: ``day-cycle`` | ``night-cycle`` | ``emergency-flush``.

    Preset actuator levels are written to ``RuntimeStore`` immediately and
    are visible in the next ``GET /api/actuators/state`` response.
    """
    try:
        return svc.apply_dt_preset(preset_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("POST /dt/preset/%s failed: %s", preset_id, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Preset could not be applied.")
