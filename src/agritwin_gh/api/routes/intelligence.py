"""
Intelligence routes — ``/api/intelligence/*``.

Endpoints
---------
GET /api/intelligence/disease → ``DiseaseRisksResponse``
GET /api/intelligence/growth  → ``GrowthIntelResponse``

Responsibility
--------------
* ``/intelligence/disease`` aggregates the composite disease risk scalar
  (from ``DiseaseRiskPenalty``) with per-pathogen probabilities from the
  disease classifier (``models/disease_inference.py``).

* ``/intelligence/growth`` returns current stage index, vpd, growth score,
  and a per-stage progress breakdown sourced from ``RuntimeStore`` and the
  growth-stage inference model (``models/growth_stage_inference.py``).

Wiring
------
``DashboardService`` is injected via FastAPI ``Depends``; it delegates to
``RuntimeStore`` projection methods for both classifiers' outputs.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from agritwin_gh.api.dependencies import get_dashboard_service
from agritwin_gh.schemas.intelligence_schemas import (
    DiseaseRisksResponse,
    GrowthIntelResponse,
)
from agritwin_gh.services.dashboard_service import DashboardService

logger = logging.getLogger("agritwin.api.intelligence")

router = APIRouter(tags=["Intelligence"])


@router.get("/intelligence/disease", response_model=DiseaseRisksResponse, status_code=200)
async def get_disease_risks(
    svc: DashboardService = Depends(get_dashboard_service),
) -> DiseaseRisksResponse:
    """Return composite and per-pathogen disease risk.

    Includes the composite risk scalar and a ``pathogens`` list with a
    ``DiseaseRiskEntry`` per known pathogen.
    """
    try:
        return svc.get_disease_risks()
    except Exception as exc:
        logger.error("GET /intelligence/disease failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Disease risk data temporarily unavailable.")


@router.get("/intelligence/growth", response_model=GrowthIntelResponse, status_code=200)
async def get_growth_intel(
    svc: DashboardService = Depends(get_dashboard_service),
) -> GrowthIntelResponse:
    """Return growth stage progress and transition predictions.

    Includes current/next stage labels, hours/days to transition, VPD,
    growth score, and a per-stage ``stage_history`` breakdown.
    """
    try:
        return svc.get_growth_intel()
    except Exception as exc:
        logger.error("GET /intelligence/growth failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Growth intelligence data temporarily unavailable.")
