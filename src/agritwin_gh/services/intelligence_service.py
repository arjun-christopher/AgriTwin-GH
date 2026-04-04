"""
IntelligenceService — aggregates disease and growth model outputs.

Responsibility
--------------
* ``get_disease_risks()`` — runs or caches inference from
  ``agritwin_gh.models.disease_inference`` and combines per-pathogen
  probabilities with the ``DiseaseRiskPenalty`` composite score to produce
  a ``DiseaseRisksResponse``.

* ``get_growth_intel()`` — reads the current growth stage from
  ``RuntimeStore`` and enriches it with per-stage progress entries,
  VPD, and yield-proxy score for the ``GrowthIntelResponse``.

Modules reused from the MPC layer
----------------------------------
* ``agritwin_gh.mpc.disease_penalty.DiseaseRiskPenalty``
* ``agritwin_gh.mpc.growth_weights.GrowthStageWeights``
* ``agritwin_gh.mpc.constants.GROWTH_STAGES``, ``DISEASE_CATEGORIES``
* ``agritwin_gh.models.disease_inference``          (optional — may be None)
* ``agritwin_gh.models.growth_stage_inference``     (optional — may be None)

Classifier availability
-----------------------
Both classifiers are loaded lazily and cached.  If a model file is absent,
``IntelligenceService`` falls back to the disease risk score and growth
stage stored in ``RuntimeStore`` without invoking the classifier.
"""

from __future__ import annotations

import logging

from agritwin_gh.core.runtime_store import RuntimeStore
from agritwin_gh.mpc.constants import DISEASE_CATEGORIES, GROWTH_STAGES

logger = logging.getLogger("agritwin.services.intelligence")


class IntelligenceService:
    """Aggregates ML model outputs for the intelligence endpoints."""

    def __init__(self, store: RuntimeStore) -> None:
        self._store = store

    def get_disease_risks(self):
        """Return composite and per-pathogen disease risk.

        TODO (implementation phase):
            1. Call ``agritwin_gh.models.disease_inference.predict_image(img)``
               with the latest leaf image from the store.
            2. Normalise probabilities → ``DiseaseRiskEntry`` list.
            3. Use ``DiseaseRiskPenalty.compute_risk_score()`` for composite risk.
        """
        return self._store.as_disease_risks_response()

    def get_growth_intel(self):
        """Return growth stage progress and transition predictions.

        Reads all required data from ``RuntimeStore.get_latest_state()``
        which returns a ``LatestState`` dataclass with ``GrowthSnapshot``
        and ``ClimateSnapshot`` fields — no direct attribute access on the
        store itself.
        """
        from agritwin_gh.mpc.realtime_core import STAGE_DURATION_HOURS

        state = self._store.get_latest_state()
        growth = state.growth
        current_idx = growth.current_stage_index
        days_in = growth.hours_to_next_stage / 24.0  # hours remaining → use inverse
        # days_elapsed_in_current_stage = total_stage_days - days_to_next
        current_stage = growth.current_stage or "seedling"
        total_stage_days = STAGE_DURATION_HOURS.get(current_stage, 336) / 24.0
        days_elapsed_in_stage = max(0.0, total_stage_days - growth.days_to_next_stage)

        return self._store.as_growth_intel_response()
