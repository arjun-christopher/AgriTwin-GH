"""
State fusion — assembles a ``FusedState`` from all AI model outputs,
DB queries, and derived values.

This is the central "glue" that combines:
  * Greenhouse indoor-climate snapshot (from DB)
  * Weather forecast (from ``WeatherDisturbanceForecast``)
  * Disease classification + progression
  * Growth-stage classification + progression
  * Derived risk score, setpoint, constraints

into the single ``FusedState`` consumed by the MPC solver.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

import numpy as np

from .config import MPCConfig
from .constants import (
    DISEASE_CATEGORIES,
    DISEASE_IMAGE_SUBCATEGORY,
    GROWTH_STAGE_FROM_DB,
    GROWTH_STAGE_IMAGE_SUBCATEGORY,
    compute_disease_risk_score,
    compute_dew_point,
    compute_leaf_wetness_proxy,
    compute_vpd,
    stage_label_to_index,
)
from .constraints import get_default_constraints, merge_constraints
from .disease_penalty import DiseaseRiskPenalty
from .disturbance import WeatherDisturbanceForecast
from .growth_weights import GrowthStageWeights
from .image_streamer import ImageStreamer
from .mpc_input_preparation import MPCInputPreparation
from .setpoints import get_setpoint
from .state import (
    DiseaseClassificationOutput,
    DiseaseProgressionOutput,
    FusedState,
    GreenhouseState,
    GrowthClassificationOutput,
    GrowthProgressionOutput,
    ImagePayload,
)

logger = logging.getLogger(__name__)

# ── Classifier label normalisation ────────────────────────────────────────────

# Disease classifier outputs "tomato_early_blight" → canonical "early blight"
_DISEASE_CLASSIFIER_TO_CANONICAL: dict[str, str] = {
    v: k for k, v in DISEASE_IMAGE_SUBCATEGORY.items()
}

# Growth-stage classifier outputs "Stage1_Seedling" → canonical "seedling"
# Derive the mapping from the canonical GROWTH_STAGE_IMAGE_SUBCATEGORY dict.
# Image subcategory values like "stage1_seedling" map to classifier labels like
# "Stage1_Seedling" via title-casing.
_GROWTH_CLASSIFIER_LABEL_MAP: dict[str, str] = {
    sub.replace("_", " ").title().replace(" ", "_"): canonical
    for canonical, sub in GROWTH_STAGE_IMAGE_SUBCATEGORY.items()
}


def _normalise_disease_label(raw: str) -> str:
    """Convert a classifier or image-subcategory label to canonical form."""
    if raw in _DISEASE_CLASSIFIER_TO_CANONICAL:
        return _DISEASE_CLASSIFIER_TO_CANONICAL[raw]
    # Fallback: strip "tomato_" prefix and replace underscores
    stripped = raw.replace("tomato_", "").replace("_", " ")
    return stripped


def _normalise_growth_label(raw: str) -> str:
    """Convert a classifier label (e.g. ``Stage2_Early_Vegetative``) to canonical."""
    if raw in _GROWTH_CLASSIFIER_LABEL_MAP:
        return _GROWTH_CLASSIFIER_LABEL_MAP[raw]
    # Fallback: try DB-form mapping
    if raw in GROWTH_STAGE_FROM_DB:
        return GROWTH_STAGE_FROM_DB[raw]
    return raw.lower().replace("_", " ")


class StateFusion:
    """Orchestrates all AI models + DB queries to produce a ``FusedState``.

    Parameters
    ----------
    config:
        Loaded ``MPCConfig``.
    input_prep:
        ``MPCInputPreparation`` instance (wraps a DB session).
    weather:
        Weather forecast wrapper.
    disease_penalty:
        Disease progression forecaster.
    growth_weights:
        Growth-stage weights + progression forecaster.
    image_streamer:
        Random image retriever.
    disease_classifier:
        Callable ``(image_bytes_or_path) -> dict`` from
        ``disease_inference.predict_image``.  *None* disables classification.
    growth_classifier:
        Callable ``(image_bytes_or_path) -> dict`` from
        ``growth_stage_inference.predict_growth_stage``.  *None* disables.
    """

    def __init__(
        self,
        config: MPCConfig,
        input_prep: MPCInputPreparation,
        weather: WeatherDisturbanceForecast,
        disease_penalty: DiseaseRiskPenalty,
        growth_weights: GrowthStageWeights,
        image_streamer: ImageStreamer,
        disease_classifier: Any | None = None,
        growth_classifier: Any | None = None,
    ) -> None:
        self._cfg = config
        self._prep = input_prep
        self._weather = weather
        self._disease = disease_penalty
        self._growth = growth_weights
        self._images = image_streamer
        self._disease_clf = disease_classifier
        self._growth_clf = growth_classifier

    # ── Main fusion pipeline ───────────────────────────────────────────

    def fuse(
        self,
        timestamp: _dt.datetime | None = None,
    ) -> FusedState:
        """Execute the full 9-step fusion pipeline and return ``FusedState``.

        Steps
        -----
        1. Fetch latest greenhouse row → ``GreenhouseState``
        2. Run weather forecast
        3. Run image classifiers (disease + growth stage)
        4. Run disease progression LSTM
        5. Run growth progression LSTM
        6. Compute derived disease risk score
        7. Determine authoritative growth stage
        8. Look up setpoint + constraints for that stage
        9. Assemble ``FusedState``
        """
        fused = FusedState(timestamp=timestamp or _dt.datetime.now())
        meta: dict[str, Any] = {}

        # ── Step 1: Greenhouse snapshot ─────────────────────────────────
        gh_row = self._prep.get_latest_greenhouse_row()
        if gh_row is None:
            logger.warning("No greenhouse data — using zero-initialised state.")
            gh = GreenhouseState(
                soil_moisture=self._cfg.initial_soil_moisture_pct,
                timestamp=fused.timestamp,
            )
        else:
            gh = GreenhouseState.from_db_row(gh_row)
            if gh.timestamp is None:
                gh.timestamp = fused.timestamp
        fused.greenhouse_state = gh
        meta["greenhouse_ts"] = str(gh.timestamp)

        # ── Step 2: Weather forecast ────────────────────────────────────
        try:
            df_weather = self._prep.get_weather_context(lookback_days=30)
            if not df_weather.empty:
                fused.weather_forecast = self._weather.get_forecast(
                    df_weather,
                    horizon_hours=self._cfg.prediction_horizon_hours,
                )
            else:
                logger.warning("Empty weather context — no forecast produced.")
        except Exception:
            logger.exception("Weather forecast failed — continuing without it.")

        # ── Step 3: Image classification ────────────────────────────────
        disease_clf_out = DiseaseClassificationOutput()
        growth_clf_out = GrowthClassificationOutput()

        if self._cfg.image_stream_enabled:
            disease_clf_out = self._run_disease_classifier()
            growth_clf_out = self._run_growth_classifier()

        fused.disease_classification = disease_clf_out.class_name
        fused.disease_confidence = disease_clf_out.confidence

        # ── Step 4: Disease progression ─────────────────────────────────
        disease_out = DiseaseProgressionOutput()
        try:
            df_disease = self._prep.get_disease_progression_context(
                sequence_length=self._disease.history_window * 2,
            )
            if not df_disease.empty:
                disease_out = self._disease.predict_all_diseases(df_disease)
        except Exception:
            logger.exception("Disease progression prediction failed.")

        fused.current_severity = disease_out.current_severity
        fused.severity_24h = disease_out.severity_24h
        fused.severity_48h = disease_out.severity_48h

        # ── Step 5: Growth progression ──────────────────────────────────
        growth_out = GrowthProgressionOutput()
        try:
            df_growth = self._prep.get_growth_progression_context(
                sequence_length=self._growth.seq_len * 2,
            )
            if not df_growth.empty:
                growth_out = self._growth.predict_from_dataframe(df_growth)
        except Exception:
            logger.exception("Growth progression prediction failed.")

        fused.next_stage = growth_out.next_stage
        fused.hours_to_transition = growth_out.hours_to_transition
        fused.transition_within_24h = growth_out.transition_within_24h
        fused.transition_within_48h = growth_out.transition_within_48h

        # ── Step 6: Derived disease risk score ──────────────────────────
        dew_pt = compute_dew_point(gh.indoor_temp, gh.indoor_humidity)
        lw = compute_leaf_wetness_proxy(gh.indoor_humidity, gh.indoor_temp, dew_pt)
        risk = compute_disease_risk_score(
            gh.indoor_temp,
            gh.indoor_humidity,
            lw,
            growth_out.current_stage or None,
        )
        fused.disease_risk_score = risk
        gh.disease_risk_score = risk
        gh.leaf_wetness_proxy = lw
        gh.vpd = compute_vpd(gh.indoor_temp, gh.indoor_humidity)

        # ── Step 7: Authoritative growth stage ──────────────────────────
        # Prefer progression model; fall back to classifier; then DB row.
        stage = self._resolve_growth_stage(growth_out, growth_clf_out, gh_row)
        fused.growth_stage = stage
        fused.growth_stage_index = stage_label_to_index(stage)
        gh.growth_stage_index = fused.growth_stage_index

        # ── Step 8: Setpoint + constraints ──────────────────────────────
        fused.setpoint = get_setpoint(stage)
        fused.constraints = get_default_constraints(stage)

        # Apply config-level constraint overrides
        if self._cfg.actuator_bounds:
            fused.constraints = merge_constraints(
                fused.constraints,
                type(fused.constraints)(actuator_bounds=self._cfg.actuator_bounds),
            )

        # ── Step 9: final assembly (already done inline) ────────────────
        logger.info(
            "State fusion complete | stage=%s risk=%.3f ts=%s",
            stage, risk, fused.timestamp,
        )
        return fused

    # ── Internal helpers ───────────────────────────────────────────────

    def _run_disease_classifier(self) -> DiseaseClassificationOutput:
        """Retrieve a random disease image and classify it."""
        if self._disease_clf is None:
            return DiseaseClassificationOutput()
        try:
            # Use first disease category that yields an image
            for disease in DISEASE_CATEGORIES:
                payload = self._images.get_random_disease_image(disease)
                if payload is not None:
                    result = self._disease_clf(payload.image_key)
                    raw_label = result.get("class_name", "")
                    return DiseaseClassificationOutput(
                        class_name=_normalise_disease_label(raw_label),
                        confidence=float(result.get("confidence", 0.0)),
                        top_k=result.get("topk", []),
                    )
        except Exception:
            logger.exception("Disease classification failed.")
        return DiseaseClassificationOutput()

    def _run_growth_classifier(self) -> GrowthClassificationOutput:
        """Retrieve a random growth-stage image and classify it."""
        if self._growth_clf is None:
            return GrowthClassificationOutput()
        try:
            from .constants import GROWTH_STAGES
            for stage in GROWTH_STAGES:
                payload = self._images.get_random_growth_stage_image(stage)
                if payload is not None:
                    result = self._growth_clf(payload.image_key)
                    raw_label = result.get("class_name", "")
                    return GrowthClassificationOutput(
                        class_name=_normalise_growth_label(raw_label),
                        confidence=float(result.get("confidence", 0.0)),
                        top_k=result.get("topk", []),
                    )
        except Exception:
            logger.exception("Growth-stage classification failed.")
        return GrowthClassificationOutput()

    @staticmethod
    def _resolve_growth_stage(
        growth_prog: GrowthProgressionOutput,
        growth_clf: GrowthClassificationOutput,
        gh_row: dict | None,
    ) -> str:
        """Pick the best available growth-stage label (canonical form).

        Priority: progression model → classifier → DB row → fallback.
        """
        # 1. Progression LSTM (most authoritative for temporal context)
        if growth_prog.current_stage:
            return growth_prog.current_stage

        # 2. Image classifier
        if growth_clf.class_name:
            return growth_clf.class_name

        # 3. DB row stage_name (if present)
        if gh_row and "stage_name" in gh_row:
            raw = gh_row["stage_name"]
            if raw in GROWTH_STAGE_FROM_DB:
                return GROWTH_STAGE_FROM_DB[raw]

        # 4. Default
        logger.warning("No growth-stage source available — defaulting to 'seedling'.")
        return "seedling"
