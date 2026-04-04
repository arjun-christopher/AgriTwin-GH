"""
Resource routes — ``/api/resources/*``.

Endpoints
---------
GET /api/resources/monthly → ``ResourcesResponse``

Responsibility
--------------
Aggregates energy (kWh) and water (L) consumption from the DT run log
stored in ``RuntimeStore``.  Cost conversion uses the Tamil Nadu tariff
constants from ``MPCConfig`` (see ``docs/MPC_COMPLETE_GUIDE.md`` §18).

The ``MonthlyCost`` schema reports costs in Indian Rupees (INR).

Wiring
------
``ResourceService`` reads the per-step energy/water accumulator from the
``RuntimeStore`` and groups it by calendar month.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_dashboard_service
from agritwin_gh.schemas.resource_schemas import ResourcesResponse
from agritwin_gh.services.dashboard_service import DashboardService

logger = logging.getLogger("agritwin.api.resources")

router = APIRouter(tags=["Resources"])


@router.get("/resources/monthly", response_model=ResourcesResponse, status_code=200)
async def get_monthly_resources(
    svc: DashboardService = Depends(get_dashboard_service),
) -> ResourcesResponse:
    """Return monthly resource consumption and cost breakdown.

    Applies Tamil Nadu electricity (₹7/kWh) and water (₹4/kL) tariff rates
    to the accumulated DT-loop totals.  Returns zero-cost defaults when the
    DT loop has not yet run.
    """
    try:
        return svc.get_resources()
    except Exception as exc:
        logger.error("GET /resources/monthly failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Resource data temporarily unavailable.")
