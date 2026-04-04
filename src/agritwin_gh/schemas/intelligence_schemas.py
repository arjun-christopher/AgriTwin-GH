"""
Intelligence schemas — used by:
  ``GET /api/intelligence/disease``
  ``GET /api/intelligence/growth``

Same ``GrowthIntelResponse`` type is embedded in ``DTStateResponse.growth``
so that ``HomeDashboard`` can render the stage-history bars without a second
API call.  ``DiseaseRisksResponse`` is standalone.

Disease data sources
--------------------
  - Composite risk → ``agritwin_gh.mpc.disease_penalty.DiseaseRiskPenalty``
  - Per-pathogen probabilities → ``agritwin_gh.models.disease_inference``
  - Canonical labels → ``agritwin_gh.mpc.constants.DISEASE_CATEGORIES``

Growth data sources
-------------------
  - Stage progress / timing → ``agritwin_gh.mpc.realtime_core.STAGE_DURATION_HOURS``
  - Stage weights → ``agritwin_gh.mpc.growth_weights.GrowthStageWeights``
  - Classifier predictions → ``agritwin_gh.models.growth_stage_inference``
  - Canonical labels → ``agritwin_gh.mpc.constants.GROWTH_STAGES``

Frontend mock shapes
--------------------
``DISEASE_RISKS`` mock (``mockData.js``)::

    { name: 'Late Blight', pathogen: 'P. infestans', risk24h: 31, severity: 'High' }

``GROWTH_INTEL`` mock::

    {
      hoursToNextStage: 47,
      transitionProb24h: 18,
      stageHistory: [
        { stage: 'Seedling', daysUsed: 14, daysTarget: 14, complete: true },
        ...
      ]
    }
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from agritwin_gh.schemas.enums import (
    SeverityLevel,
    DISEASE_DISPLAY_NAME,
    DISEASE_PATHOGEN,
    GROWTH_STAGE_DISPLAY_NAME,
    STAGE_DURATION_DAYS,
    GROWTH_STAGE_ORDERED,
)


# ── Disease intelligence ───────────────────────────────────────────────────────


class DiseaseRiskEntry(BaseModel):
    """Risk score and metadata for one canonical disease category.

    Returned as an element of ``DiseaseRisksResponse.pathogens``.

    Fields are chosen to satisfy both the HomeDashboard risk badge and the
    DetailedInsights ``DiseaseBar`` component.

    Example::

        {
          "label": "late blight",
          "name": "Late Blight",
          "pathogen": "P. infestans",
          "risk_score": 0.31,
          "risk_24h": 31,
          "probability": 0.28,
          "severity": "High"
        }
    """

    label: str = Field(
        description="Canonical lowercase disease label from DISEASE_CATEGORIES, "
                    "e.g. 'late blight'.",
    )
    name: str = Field(
        default="",
        description="Title-cased display name for the React UI, e.g. 'Late Blight'. "
                    "Populated from DISEASE_DISPLAY_NAME in enums.py.",
    )
    pathogen: str = Field(
        default="",
        description="Scientific pathogen abbreviation, e.g. 'P. infestans'. "
                    "Populated from DISEASE_PATHOGEN in enums.py.  "
                    "Empty for 'healthy leaves'.",
    )
    risk_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Normalised risk score (0–1) from the disease-penalty model.",
    )
    risk_24h: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Risk percentage for 24-hour horizon (0–100). "
                    "Equals risk_score × 100 rounded to one decimal place. "
                    "Used by the DetailedInsights DiseaseBar width.",
    )
    probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Raw classifier softmax probability (0–1) for this disease class.",
    )
    severity: SeverityLevel = Field(
        default="Low",
        description="Human-readable severity tier: Low | Medium | High | Critical.",
    )


class DiseaseRisksResponse(BaseModel):
    """Response for ``GET /api/intelligence/disease``.

    ``composite_risk`` is the scalar forwarded to the MPC cost function
    (``agritwin_gh.mpc.disease_penalty``).  The individual entries in
    ``pathogens`` are the per-disease breakdown rendered as bars in the
    ``DetailedInsights`` Disease Risk panel.

    The list covers all five non-healthy disease categories.  The
    ``'healthy leaves'`` category is **not** included in ``pathogens`` — its
    probability is reflected in a high ``composite_risk == 0`` scenario
    (i.e. risk is near zero when healthy leaves is dominant).

    Example::

        {
          "composite_risk": 0.12,
          "dominant_label": "healthy leaves",
          "timestamp": "2026-04-04T09:15:00Z",
          "pathogens": [
            { "label": "late blight", "name": "Late Blight",
              "pathogen": "P. infestans", "risk_score": 0.31,
              "risk_24h": 31, "probability": 0.28, "severity": "High" },
            ...
          ]
        }
    """

    composite_risk: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Composite disease risk (0–1) forwarded to the MPC cost function.",
    )
    dominant_label: str = Field(
        default="healthy leaves",
        description="Canonical label of the dominant disease class (highest probability).",
    )
    timestamp: str = Field(
        default="",
        description="ISO-8601 timestamp of the last classifier inference.",
    )
    pathogens: list[DiseaseRiskEntry] = Field(
        default_factory=list,
        description="Per-disease breakdown, one entry per non-healthy DISEASE_CATEGORIES label.",
    )


# ── Growth intelligence ────────────────────────────────────────────────────────


class StageHistoryEntry(BaseModel):
    """Progress summary for one growth stage — used in the stage-history bars.

    Returned as an element of ``GrowthIntelResponse.stage_history``.

    Field names mirror the ``stageHistory`` array in ``mockData.js``
    (camelCase on the frontend, snake_case here):
      ``stage``       ↔ stage display name
      ``days_used``   ↔ daysUsed
      ``days_target`` ↔ daysTarget
      ``complete``    ↔ complete

    Example::

        { "stage": "Seedling", "days_used": 14.0, "days_target": 14.0,
          "complete": true, "status": "completed", "progress_pct": 100.0 }
    """

    stage: str = Field(
        description="Title-cased display name of the stage, e.g. 'Seedling'.",
    )
    stage_key: str = Field(
        default="",
        description="Canonical lowercase key matching GROWTH_STAGES, e.g. 'seedling'.",
    )
    days_used: float = Field(
        default=0.0,
        ge=0.0,
        description="Days elapsed in this stage (0 if not yet reached).",
    )
    days_target: float = Field(
        default=0.0,
        ge=0.0,
        description="Expected total days for this stage from STAGE_DURATION_DAYS.",
    )
    complete: bool = Field(
        default=False,
        description="True when the crop has passed through this stage.",
    )
    status: Literal["completed", "current", "pending"] = Field(
        default="pending",
        description="Stage lifecycle status for CropStageTrack rendering.",
    )
    progress_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Percentage completion of this stage (100 if complete, "
                    "partial if current, 0 if pending).",
    )


class GrowthIntelResponse(BaseModel):
    """Response for ``GET /api/intelligence/growth``.

    Also embedded as ``DTStateResponse.growth`` so that HomeDashboard can render
    the stage-history bars without a second API call.

    Mirrors the ``GROWTH_INTEL`` constant from ``mockData.js`` plus additional
    fields used by the ``DetailedInsights`` spotlight stats.

    Example::

        {
          "current_stage": "Flowering",
          "current_stage_index": 3,
          "next_stage": "Unripe",
          "hours_to_next_stage": 47.0,
          "days_to_next_stage": 1.96,
          "transition_prob_24h": 18.0,
          "vpd": 0.95,
          "growth_score": 0.78,
          "timestamp": "2026-04-04T09:15:00Z",
          "stage_history": [
            { "stage": "Seedling", "days_used": 14.0, "days_target": 14.0,
              "complete": true, "status": "completed", "progress_pct": 100.0 },
            ...
          ]
        }
    """

    current_stage: str = Field(
        default="",
        description="Title-cased display name of the current stage, e.g. 'Flowering'.",
    )
    current_stage_index: int = Field(
        default=0,
        ge=0,
        le=5,
        description="Zero-based index of the current stage (0 = Seedling, 5 = Ripe).",
    )
    next_stage: str | None = Field(
        default=None,
        description="Title-cased display name of the next stage, or null at 'Ripe'.",
    )
    hours_to_next_stage: float | None = Field(
        default=None,
        description="Estimated hours remaining until the next stage transition. "
                    "Null if at 'Ripe'. "
                    "Rendered as the primary spotlight stat in DetailedInsights.",
    )
    days_to_next_stage: float | None = Field(
        default=None,
        description="``hours_to_next_stage / 24`` for convenience.",
    )
    transition_prob_24h: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
        description="Estimated probability (%) of transitioning to the next stage "
                    "within 24 hours.  Rendered as the secondary spotlight stat.",
    )
    vpd: float = Field(
        default=0.0,
        description="Current vapour-pressure deficit in kPa (from FusedState).",
    )
    growth_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Synthetic yield-proxy score at the current DT step (0–1). "
                    "Computed by GrowthStageWeights.",
    )
    timestamp: str = Field(
        default="",
        description="ISO-8601 timestamp of the DT step this data was derived from.",
    )
    stage_history: list[StageHistoryEntry] = Field(
        default_factory=list,
        description="One entry per growth stage in GROWTH_STAGE_ORDERED order. "
                    "Drives CropStageTrack and the stage-history bars in both pages.",
    )
