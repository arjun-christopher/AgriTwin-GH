"""
Disease risk penalty calculator for the MPC cost function.

Responsibilities:
  1. Load the disease-progression LSTM (per-disease severity forecaster).
  2. Predict 24 h / 48 h severity for each disease category.
  3. Compute an aggregate disease-risk penalty score.

The disease-progression LSTM expects a 3-D array ``(1, window, n_features)``
per disease and returns ``(presence_prob, future_scaled)`` where
``severity_pct = clip(future_scaled[0] * 100, 0, 100)``.
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
        self._loaded = False

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

    # ── Public API ─────────────────────────────────────────────────────

    def predict_severity(
        self,
        sequence: np.ndarray,
    ) -> tuple[float, float]:
        """Run a single ``(1, window, n_features)`` array through the LSTM.

        Returns
        -------
        (presence_probability, severity_pct)
            ``severity_pct`` is in [0, 100].
        """
        self._ensure_loaded()
        presence_prob, future_scaled = self._model.predict(sequence, verbose=0)
        severity_pct = float(np.clip(future_scaled[0] * 100.0, 0.0, 100.0))
        presence = float(presence_prob[0, 0]) if presence_prob.ndim > 1 else float(presence_prob[0])
        return presence, severity_pct

    def predict_all_diseases(
        self,
        df_disease_context: pd.DataFrame,
    ) -> DiseaseProgressionOutput:
        """Predict severity for every disease category from DB context.

        Parameters
        ----------
        df_disease_context:
            Long-format DataFrame from
            ``MPCInputPreparation.get_disease_progression_context()``.
            Must have columns: ``disease_name``, ``timestamp``, and the
            numeric feature columns expected by the scaler.

        Returns
        -------
        DiseaseProgressionOutput
            With ``current_severity``, ``severity_24h``, ``severity_48h``
            keyed by *canonical* disease labels.
        """
        self._ensure_loaded()

        current_sev: dict[str, float] = {}
        sev_24h: dict[str, float] = {}
        sev_48h: dict[str, float] = {}

        if df_disease_context.empty:
            logger.warning("Empty disease context — returning zero severities.")
            for d in DISEASE_CATEGORIES:
                current_sev[d] = 0.0
                sev_24h[d] = 0.0
                sev_48h[d] = 0.0
            return DiseaseProgressionOutput(
                current_severity=current_sev,
                severity_24h=sev_24h,
                severity_48h=sev_48h,
            )

        # Determine numeric feature columns (exclude metadata)
        meta_cols = {"timestamp", "cycle_id", "stage_name", "disease_name"}
        feature_cols = [
            c for c in df_disease_context.columns if c not in meta_cols
        ]

        for disease_canonical in DISEASE_CATEGORIES:
            db_name = DISEASE_TO_DB.get(disease_canonical, disease_canonical)
            df_d = df_disease_context[
                df_disease_context["disease_name"] == db_name
            ].sort_values("timestamp")

            if df_d.empty or len(df_d) < self.history_window:
                logger.debug(
                    "Insufficient rows for '%s' (%d/%d) — defaulting to 0.",
                    disease_canonical, len(df_d), self.history_window,
                )
                current_sev[disease_canonical] = 0.0
                sev_24h[disease_canonical] = 0.0
                sev_48h[disease_canonical] = 0.0
                continue

            # Current severity is the latest infection_pct value
            if "current_infection_pct" in df_d.columns:
                current_sev[disease_canonical] = float(
                    df_d["current_infection_pct"].iloc[-1]
                )
            else:
                current_sev[disease_canonical] = 0.0

            # Build sequence for LSTM
            raw = df_d[feature_cols].values[-self.history_window:]
            if self._scaler is not None:
                raw = self._scaler.transform(raw)
            sequence = raw[np.newaxis, ...]  # (1, window, F)

            presence, severity = self.predict_severity(sequence)

            # The model predicts "future severity" — we store that as 24 h.
            # For 48 h we apply a simple linear extrapolation from current trend.
            sev_24h[disease_canonical] = severity
            trend = severity - current_sev[disease_canonical]
            sev_48h[disease_canonical] = float(
                np.clip(severity + trend, 0.0, 100.0)
            )

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
