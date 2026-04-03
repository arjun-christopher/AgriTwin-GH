"""
Disease risk penalty calculator for the MPC cost function.

Responsibilities:
  1. Load the disease-progression LSTM (multi-disease wide-format forecaster).
  2. Predict 24 h / 48 h severity for all 5 disease categories in one pass.
  3. Compute an aggregate disease-risk penalty score.

The disease-progression LSTM expects ``(1, history_window, 92)`` — a wide-format
sequence with all diseases aggregated into one row per timestamp.
Feature engineering (long→wide pivot + one-hot encoding) happens in
``_engineer_features()`` before calling the model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .constants import (
    DISEASE_CATEGORIES,
    DISEASE_FROM_DB,
    DISEASE_TO_DB,
    compute_disease_risk_score,
)
from .state import DiseaseProgressionOutput
from .utils import _ARTIFACTS_DIR, _MODELS_DIR, discover_latest_artifact

logger = logging.getLogger(__name__)


class DiseaseRiskPenalty:
    """Wraps the per-disease progression LSTM and exposes an MPC-friendly API.

    Parameters
    ----------
    run_id:
        Explicit artifact run ID directory name.  *None* → auto-discover.
    """

    def __init__(self, run_id: str | None = None) -> None:
        if run_id:
            self._art_dir = _ARTIFACTS_DIR / run_id
        else:
            self._art_dir = discover_latest_artifact("disease_progression_")
        if self._art_dir is None or not self._art_dir.exists():
            raise FileNotFoundError(
                f"No disease-progression artifact found (run_id={run_id!r})."
            )
        self._model: Any = None
        self._scaler: Any = None
        self._config: dict[str, Any] = {}
        self._seq_feature_cols: list[str] = []
        self._loaded = False

    # ── Schema constants (mirror training notebook exactly) ────────────
    # Diseases in sorted order — matches the 5 model outputs
    _DISEASES: list[str] = [
        "early_blight", "late_blight", "leaf_mold",
        "powdery_mildew", "spider_mites",
    ]
    _STAGE_NAMES: list[str] = [
        "early_vegetative", "flowering", "flowering_initiation",
        "ripe", "seedling", "unripe",
    ]
    _CYCLE_LABELS: list[str] = [
        "kharif_2024", "kharif_2025", "rabi_2024", "summer_2025",
    ]
    _SEASON_LABELS: list[str] = [
        "dry_cool", "northeast_monsoon", "southwest_monsoon", "summer",
    ]
    _CTRL_ACTION_TYPES: dict[str, list[str]] = {
        "early_blight":   ["copper_treatment", "fungicide_spray", "none", "reduced_irrigation"],
        "late_blight":    ["fungicide_spray", "humidity_reduction", "none", "ventilation_increase"],
        "leaf_mold":      ["humidity_reduction", "leaf_pruning", "none", "ventilation_increase"],
        "powdery_mildew": ["improved_airflow", "none", "potassium_bicarbonate", "sulfur_treatment"],
        "spider_mites":   ["acaricide_spray", "biological_release", "humidity_increase", "none"],
    }

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import tensorflow as tf  # noqa: PLC0415

        # Config
        cfg_path = self._art_dir / "config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                self._config = json.load(f)

        # seq_feature_cols — saved by _extract_disease_features.py
        feat_cfg_path = self._art_dir / "seq_feature_cols.json"
        if feat_cfg_path.exists():
            with open(feat_cfg_path, encoding="utf-8") as f:
                feat_cfg = json.load(f)
            self._seq_feature_cols = feat_cfg["seq_feature_cols"]
        else:
            logger.warning(
                "seq_feature_cols.json not found in %s — disease predictions "
                "will fall back to zeros.", self._art_dir
            )

        # Model — prefer the LSTM checkpoint stored alongside artifacts
        keras_candidates = sorted(self._art_dir.glob("best_lstm_*.keras"))
        if keras_candidates:
            model_path = keras_candidates[0]
        else:
            # Fallback: model saved at models/ level
            model_glob = sorted(_MODELS_DIR.glob("disease_progression_*.keras"))
            if not model_glob:
                raise FileNotFoundError(
                    "Cannot locate disease-progression .keras model."
                )
            model_path = model_glob[-1]

        self._model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Loaded disease-progression LSTM from %s", model_path.name)

        # Scaler
        scaler_path = self._art_dir / "sequence_feature_scaler.joblib"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler from %s", scaler_path.name)
        else:
            logger.warning("No sequence_feature_scaler.joblib found; scaling disabled.")

        self._loaded = True

    # ── Helpers ────────────────────────────────────────────────────────

    @property
    def history_window(self) -> int:
        return int(self._config.get("history_window", 24))

    def _engineer_features(self, df_long: pd.DataFrame) -> pd.DataFrame:
        """Convert long-format disease DB context to wide-format with all 92 features.

        Mirrors the exact preprocessing in the training notebook:
        pivot per-disease columns → one-hot encode categoricals → select
        ``seq_feature_cols`` in the same order the scaler was fitted on.
        """
        # ── Shared (per-timestamp) base columns ───────────────────────
        shared_cols = [
            "timestamp", "cycle_id", "cycle_label", "season_label",
            "stage_name", "stage_index", "days_from_cycle_start", "day_of_year",
            "week_of_year", "hour", "hours_in_current_stage", "stage_progress_pct",
            "total_cycle_progress_pct", "is_stage_transition",
            "indoor_temp", "indoor_humidity", "indoor_air_velocity", "indoor_co2",
            "solarradiation", "day_night_flag", "vpd", "dew_point",
            "leaf_wetness_proxy", "temperature_rolling_mean_24h",
            "humidity_rolling_mean_24h", "vpd_proxy", "cumulative_gdd_like_index",
        ]
        available_shared = [c for c in shared_cols if c in df_long.columns]
        wide = (
            df_long[available_shared]
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        # DB uses indoor_co2 (lowercase); training used indoor_CO2
        if "indoor_co2" in wide.columns:
            wide = wide.rename(columns={"indoor_co2": "indoor_CO2"})

        # Coerce is_stage_transition from bool → float
        if "is_stage_transition" in wide.columns:
            wide["is_stage_transition"] = (
                wide["is_stage_transition"].fillna(False).astype(float)
            )

        # ── Pivot per-disease columns from long to wide ───────────────
        disease_val_cols = [
            "disease_present_flag", "current_infection_pct",
            "infection_growth_rate_hourly", "stage_susceptibility_score",
            "disease_risk_score", "control_action_flag",
            "outbreak_trigger_flag", "control_action_type",
        ]
        derived: dict[str, pd.Series] = {}
        for d in self._DISEASES:
            sub = df_long[df_long["disease_name"] == d].set_index("timestamp")
            for col in disease_val_cols:
                if col in sub.columns:
                    derived[f"{col}__{d}"] = wide["timestamp"].map(sub[col])

        # ── One-hot encode stage_name ─────────────────────────────────
        stage_raw = (
            wide["stage_name"].str.lower().str.replace(" ", "_").fillna("")
            if "stage_name" in wide.columns
            else pd.Series([""]*len(wide), index=wide.index)
        )
        for s in self._STAGE_NAMES:
            derived[f"stage_name_{s}"] = (stage_raw == s).astype(float)

        # ── One-hot encode cycle_label ────────────────────────────────
        cycle_raw = (
            wide["cycle_label"].fillna("")
            if "cycle_label" in wide.columns
            else pd.Series([""]*len(wide), index=wide.index)
        )
        for cl in self._CYCLE_LABELS:
            derived[f"cycle_label_{cl}"] = (cycle_raw == cl).astype(float)

        # ── One-hot encode season_label ───────────────────────────────
        season_raw = (
            wide["season_label"].fillna("")
            if "season_label" in wide.columns
            else pd.Series([""]*len(wide), index=wide.index)
        )
        for sl in self._SEASON_LABELS:
            derived[f"season_label_{sl}"] = (season_raw == sl).astype(float)

        # ── One-hot encode control_action_type per disease ────────────
        for d in self._DISEASES:
            cat_col_wide = f"control_action_type__{d}"
            cat_raw = (
                derived.get(cat_col_wide, wide.get(cat_col_wide,  # type: ignore[arg-type]
                    pd.Series(["none"]*len(wide), index=wide.index)))
                .fillna("none")
            )
            for action in self._CTRL_ACTION_TYPES.get(d, []):
                derived[f"control_action_type__{d}_{action}"] = (
                    (cat_raw == action).astype(float)
                )

        # ── Join all derived columns at once ──────────────────────────
        wide = pd.concat(
            [wide, pd.DataFrame(derived, index=wide.index)], axis=1
        )
        wide = wide.fillna(0.0)
        return wide

    # ── Public API ─────────────────────────────────────────────────────

    def predict_all_diseases(
        self,
        df_disease_context: pd.DataFrame,
    ) -> DiseaseProgressionOutput:
        """Predict severity for every disease category from DB context.

        The model is a **multi-disease** LSTM: one forward pass returns
        predictions for all 5 diseases simultaneously.  Long-format input
        is pivoted to wide format via ``_engineer_features()`` before
        passing to the scaler / model.

        Parameters
        ----------
        df_disease_context:
            Long-format DataFrame from
            ``MPCInputPreparation.get_disease_progression_context()``.

        Returns
        -------
        DiseaseProgressionOutput
            With ``current_severity``, ``severity_24h``, ``severity_48h``
            keyed by *canonical* disease labels.
        """
        self._ensure_loaded()

        # Zero defaults for all MPC disease categories
        current_sev: dict[str, float] = {d: 0.0 for d in DISEASE_CATEGORIES}
        sev_24h: dict[str, float] = {d: 0.0 for d in DISEASE_CATEGORIES}
        sev_48h: dict[str, float] = {d: 0.0 for d in DISEASE_CATEGORIES}

        if df_disease_context.empty:
            logger.warning("Empty disease context — returning zero severities.")
            return DiseaseProgressionOutput(
                current_severity=current_sev,
                severity_24h=sev_24h,
                severity_48h=sev_48h,
            )

        if not self._seq_feature_cols:
            logger.warning(
                "seq_feature_cols not loaded — disease predictions zeroed."
            )
            return DiseaseProgressionOutput(
                current_severity=current_sev,
                severity_24h=sev_24h,
                severity_48h=sev_48h,
            )

        # ── Feature engineering: long → wide, 92 features ─────────────
        try:
            wide = self._engineer_features(df_disease_context)
        except Exception as exc:
            logger.warning(
                "Disease feature engineering failed — zeroing. (%s)", exc
            )
            return DiseaseProgressionOutput(
                current_severity=current_sev,
                severity_24h=sev_24h,
                severity_48h=sev_48h,
            )

        if len(wide) < self.history_window:
            logger.debug(
                "Insufficient wide rows (%d/%d) — zeroing.",
                len(wide), self.history_window,
            )
            return DiseaseProgressionOutput(
                current_severity=current_sev,
                severity_24h=sev_24h,
                severity_48h=sev_48h,
            )

        # Fill any seq_feature_cols that are still missing
        missing_fc = [c for c in self._seq_feature_cols if c not in wide.columns]
        if missing_fc:
            logger.debug(
                "Filling %d missing seq_feature_cols with 0: %s",
                len(missing_fc), missing_fc[:5],
            )
            wide[missing_fc] = 0.0

        # ── Extract current severity from the last wide row ───────────
        for d_db in self._DISEASES:
            sev_col = f"current_infection_pct__{d_db}"
            if sev_col in wide.columns:
                last_val = wide[sev_col].iloc[-1]
                d_canon = DISEASE_FROM_DB.get(d_db, d_db)
                current_sev[d_canon] = float(last_val) if not pd.isna(last_val) else 0.0

        # ── Build (1, 24, 92) sequence and run model once ─────────────
        raw = wide[self._seq_feature_cols].values[-self.history_window:]  # (24, 92)
        if self._scaler is not None:
            raw = self._scaler.transform(raw)
        sequence = raw[np.newaxis, ...]  # (1, 24, 92)

        pres_prob_arr, fut_scaled_arr = self._model.predict(sequence, verbose=0)
        pres_prob = pres_prob_arr[0]   # (5,)
        fut_scaled = fut_scaled_arr[0] # (5,)

        # ── Map model outputs → canonical disease names ───────────────
        # Model output index = position in self._DISEASES (sorted order)
        for i, d_db in enumerate(self._DISEASES):
            d_canon = DISEASE_FROM_DB.get(d_db, d_db)
            severity_24 = float(np.clip(fut_scaled[i] * 100.0, 0.0, 100.0))
            curr = current_sev.get(d_canon, 0.0)
            trend = severity_24 - curr
            sev_24h[d_canon] = severity_24
            sev_48h[d_canon] = float(np.clip(severity_24 + trend, 0.0, 100.0))

        return DiseaseProgressionOutput(
            current_severity=current_sev,
            severity_24h=sev_24h,
            severity_48h=sev_48h,
        )

    @staticmethod
    def compute_risk_score(
        temp: float,
        humidity: float,
        leaf_wetness: float,
        growth_stage: str | None = None,
    ) -> float:
        """Compute the aggregate disease-risk score (0-1) using ``constants``."""
        return compute_disease_risk_score(temp, humidity, leaf_wetness, growth_stage)

    @property
    def artifact_dir(self) -> Path:
        return self._art_dir
