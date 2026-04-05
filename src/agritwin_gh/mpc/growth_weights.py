"""
Growth-stage adaptive weights and transition forecasting.

Responsibilities:
  1. Look up per-stage cost-weight multipliers from ``MPCConfig``.
  2. Load the growth-stage-progression LSTM and predict transition timing.

The growth-progression LSTM outputs a 5-tuple::

    (p_cur, p_nxt, p_hrs_sc, p_t24, p_t48) = model.predict(X)

where:
  * ``p_cur``    — softmax logits for current stage  (N, n_stages)
  * ``p_nxt``    — softmax logits for next stage      (N, n_stages)
  * ``p_hrs_sc`` — hours-to-transition (scaled)       (N, 1)
  * ``p_t24``    — sigmoid prob of transition ≤ 24 h  (N, 1)
  * ``p_t48``    — sigmoid prob of transition ≤ 48 h  (N, 1)

Hours are de-scaled via  ``hrs_raw * HRS_STD + HRS_MEAN``  using values
stored in ``inference_config.json``.
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
    GROWTH_STAGE_FROM_DB,
    GROWTH_STAGE_TO_DB,
    GROWTH_STAGES,
    stage_index_to_label,
)
from .state import GrowthProgressionOutput

logger = logging.getLogger(__name__)

# ── Path resolution ───────────────────────────────────────────────────────────

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
_ARTIFACTS_DIR = _MODELS_DIR / "artifacts"


def _discover_latest_artifact(prefix: str) -> Path | None:
    candidates = sorted(
        (d for d in _ARTIFACTS_DIR.iterdir() if d.is_dir() and d.name.startswith(prefix)),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


class GrowthStageWeights:
    """Adaptive MPC cost-weight multipliers and growth-transition forecasting.

    Parameters
    ----------
    stage_weight_multipliers:
        Dict loaded from ``MPCConfig.stage_weight_multipliers``.
        Keys are DB-form stage names, values are ``{weight_name: multiplier}``.
    run_id:
        Explicit growth-progression artifact directory name.
        *None* → auto-discover.
    """

    def __init__(
        self,
        stage_weight_multipliers: dict[str, dict[str, float]] | None = None,
        run_id: str | None = None,
    ) -> None:
        self._multipliers = stage_weight_multipliers or {}

        # Artifact discovery
        if run_id:
            self._art_dir: Path | None = _ARTIFACTS_DIR / run_id
        else:
            self._art_dir = _discover_latest_artifact("growth_stage_progression_")
        if self._art_dir is None or not self._art_dir.exists():
            raise FileNotFoundError(
                f"No growth-stage-progression artifact found (run_id={run_id!r})."
            )

        # Lazy-loaded state
        self._model: Any = None
        self._scaler: Any = None
        self._inference_cfg: dict[str, Any] = {}
        self._int_to_stage: dict[int, str] = {}
        self._loaded = False

    # ── Lazy loading ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        import tensorflow as tf  # noqa: PLC0415

        # Inference config
        cfg_path = self._art_dir / "inference_config.json"
        if cfg_path.exists():
            with open(cfg_path, encoding="utf-8") as f:
                self._inference_cfg = json.load(f)
        self._int_to_stage = {
            int(k): v
            for k, v in self._inference_cfg.get("int_to_stage", {}).items()
        }

        # Model — artifact-level .keras or top-level models/ .keras
        keras_in_art = sorted(self._art_dir.glob("*.keras"))
        if keras_in_art:
            model_path = keras_in_art[0]
        else:
            model_glob = sorted(_MODELS_DIR.glob("growth_stage_progression_*.keras"))
            if not model_glob:
                raise FileNotFoundError(
                    "Cannot locate growth-stage-progression .keras model."
                )
            model_path = model_glob[-1]

        from .utils import load_keras_model  # noqa: PLC0415
        self._model = load_keras_model(model_path)
        logger.info("Loaded growth-stage-progression LSTM from %s", model_path.name)

        # Feature scaler
        scaler_path = self._art_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            self._scaler = joblib.load(scaler_path)
            logger.info("Loaded feature scaler from %s", scaler_path.name)

        self._loaded = True

    # ── Weight multipliers ─────────────────────────────────────────────

    def get_weights(
        self,
        stage_name: str,
        base_weights: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Return cost weights for *stage_name* (canonical label).

        If *base_weights* is supplied the multipliers are applied on top;
        otherwise the multipliers alone are returned.
        """
        db_key = GROWTH_STAGE_TO_DB.get(stage_name, stage_name.replace(" ", "_"))
        mults = self._multipliers.get(db_key, {})

        if base_weights is None:
            return dict(mults)

        adjusted = dict(base_weights)
        for wname, mult in mults.items():
            if wname in adjusted:
                adjusted[wname] *= mult
        return adjusted

    # ── Transition prediction ──────────────────────────────────────────

    # Base columns that receive rolling/lag engineering (mirrors training notebook
    # Section 6). Order matches training — is_stage_transition gets rolling only,
    # no lag features.
    _RAW_COLS_FOR_ENGINEERING: list[str] = [
        "year", "month", "day_of_year", "week_of_year", "hour",
        "days_from_cycle_start", "stage_index", "hours_in_current_stage",
        "days_in_current_stage", "stage_duration_hours", "stage_duration_days",
        "stage_progress_pct", "total_cycle_progress_pct",
        "estimated_days_to_next_stage", "estimated_hours_to_next_stage",
        "is_stage_transition",
        "indoor_temp", "indoor_humidity", "indoor_air_velocity", "indoor_co2",
        "solarradiation", "day_night_flag", "vpd", "dew_point", "leaf_wetness_proxy",
        "temperature_rolling_mean_24h", "humidity_rolling_mean_24h",
        "vpd_proxy", "light_period_flag", "cumulative_gdd_like_index",
    ]
    _LAG_SKIP_COLS: frozenset[str] = frozenset({"is_stage_transition"})

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same feature engineering as the training notebook (Section 6).

        Input: raw DB context with base columns
        Output: DataFrame with all 358 model-ready features
        """
        df = df.copy().sort_values("timestamp").reset_index(drop=True)

        # ── Derived time features from timestamp ──────────────────────
        t0 = df["timestamp"].iloc[0]
        df["elapsed_hours"] = (df["timestamp"] - t0).dt.total_seconds() / 3600.0
        df["hour_of_day"]   = df["timestamp"].dt.hour
        df["day_of_cycle"]  = (df["elapsed_hours"] / 24).astype(int)

        # ── Coerce is_stage_transition to numeric ─────────────────────
        if "is_stage_transition" in df.columns:
            df["is_stage_transition"] = (
                df["is_stage_transition"].fillna(False).astype(float)
            )

        # ── Fill any missing base columns with 0 ──────────────────────
        for col in self._RAW_COLS_FOR_ENGINEERING:
            if col not in df.columns:
                df[col] = 0.0

        # ── Rolling means/stds and lags — batch via pd.concat to avoid
        # DataFrame fragmentation (matches notebook ROLLING_WINDOWS / LAG_STEPS)
        rolling_windows = [6, 12, 24]
        lag_steps       = [1, 2, 3, 6, 12]
        derived: dict[str, pd.Series] = {}

        for col in self._RAW_COLS_FOR_ENGINEERING:
            ser = df[col]
            for w in rolling_windows:
                derived[f"{col}_roll_mean_{w}h"] = (
                    ser.rolling(w, min_periods=1).mean()
                )
                derived[f"{col}_roll_std_{w}h"] = (
                    ser.rolling(w, min_periods=1).std().fillna(0)
                )
            if col not in self._LAG_SKIP_COLS:
                for lag in lag_steps:
                    derived[f"{col}_lag_{lag}"] = ser.shift(lag)

        df = pd.concat([df, pd.DataFrame(derived, index=df.index)], axis=1)
        return df

    @property
    def seq_len(self) -> int:
        return int(self._inference_cfg.get("seq_len", 24))

    @property
    def feature_cols(self) -> list[str]:
        return self._inference_cfg.get("feature_cols", [])

    @property
    def hrs_mean(self) -> float:
        return float(self._inference_cfg.get("hrs_mean", 270.33))

    @property
    def hrs_std(self) -> float:
        return float(self._inference_cfg.get("hrs_std", 192.35))

    def _decode_stage(self, idx: int) -> str:
        """Convert an integer index to a canonical (space-separated) stage label."""
        db_label = self._int_to_stage.get(idx, "unknown")
        return GROWTH_STAGE_FROM_DB.get(db_label, db_label)

    def predict_transition(
        self,
        sequence_3d: np.ndarray,
    ) -> GrowthProgressionOutput:
        """Run a ``(1, seq_len, n_features)`` array through the LSTM.

        Returns a populated ``GrowthProgressionOutput``.
        """
        self._ensure_loaded()
        preds = self._model.predict(sequence_3d, verbose=0)
        cur_logits, nxt_logits, hrs_pred, t24_prob, t48_prob = preds

        cur_idx = int(np.argmax(cur_logits[0]))
        nxt_idx = int(np.argmax(nxt_logits[0]))
        hrs_raw = float(hrs_pred[0, 0])
        hrs_actual = max(0.0, hrs_raw * self.hrs_std + self.hrs_mean)

        t24 = float(t24_prob[0, 0])
        t48 = float(t48_prob[0, 0])

        return GrowthProgressionOutput(
            current_stage=self._decode_stage(cur_idx),
            next_stage=self._decode_stage(nxt_idx),
            hours_to_transition=round(hrs_actual, 2),
            transition_within_24h=t24 >= 0.5,
            transition_within_48h=t48 >= 0.5,
        )

    def predict_from_dataframe(
        self,
        df_growth_context: pd.DataFrame,
    ) -> GrowthProgressionOutput:
        """Apply feature engineering then run the LSTM.

        Parameters
        ----------
        df_growth_context:
            DataFrame from ``MPCInputPreparation.get_growth_progression_context()``,
            sorted ascending by timestamp, with at least ``seq_len`` rows.
            The raw ~30 base columns are enough — rolling/lag features are
            computed here to exactly match the training notebook (Section 6).
        """
        self._ensure_loaded()

        if df_growth_context.empty:
            logger.warning("Empty growth context — returning defaults.")
            return GrowthProgressionOutput()

        # ── Apply inline feature engineering (mirrors training notebook) ──
        df_growth_context = self._engineer_features(df_growth_context)

        feat_cols = self.feature_cols
        if not feat_cols:
            # Fallback: use all numeric columns except metadata
            meta = {"timestamp", "cycle_id", "stage_name"}
            feat_cols = [c for c in df_growth_context.columns if c not in meta
                         and pd.api.types.is_numeric_dtype(df_growth_context[c])]

        # After engineering, any remaining missing column is a real problem
        missing = set(feat_cols) - set(df_growth_context.columns)
        if missing:
            logger.warning(
                "Growth model still missing %d features after engineering "
                "(e.g. %s). Returning defaults.",
                len(missing),
                ", ".join(sorted(missing)[:3]),
            )
            return GrowthProgressionOutput()

        if len(df_growth_context) < self.seq_len:
            logger.warning(
                "Insufficient growth-context rows (%d/%d) — returning defaults.",
                len(df_growth_context),
                self.seq_len,
            )
            return GrowthProgressionOutput()

        raw = df_growth_context[feat_cols].values[-self.seq_len:]

        # Replace NaN before scaling (lags at start of sequence will be NaN)
        raw = np.nan_to_num(raw, nan=0.0)

        if self._scaler is not None:
            raw = self._scaler.transform(raw)

        sequence_3d = raw[np.newaxis, ...]  # (1, seq_len, F)
        return self.predict_transition(sequence_3d)

    @property
    def artifact_dir(self) -> Path | None:
        return self._art_dir
