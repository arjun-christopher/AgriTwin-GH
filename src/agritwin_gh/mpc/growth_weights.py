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

        self._model = tf.keras.models.load_model(model_path, compile=False)
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
        """Build a sequence from a DB-sourced DataFrame and predict.

        Parameters
        ----------
        df_growth_context:
            DataFrame from ``MPCInputPreparation.get_growth_progression_context()``,
            sorted ascending by timestamp, with at least ``seq_len`` rows.
        """
        self._ensure_loaded()

        feat_cols = self.feature_cols
        if not feat_cols:
            # Fallback: use all numeric columns except metadata
            meta = {"timestamp", "cycle_id", "stage_name", "stage_index"}
            feat_cols = [c for c in df_growth_context.columns if c not in meta]

        if len(df_growth_context) < self.seq_len:
            logger.warning(
                "Insufficient growth-context rows (%d/%d) — returning defaults.",
                len(df_growth_context),
                self.seq_len,
            )
            return GrowthProgressionOutput()

        raw = df_growth_context[feat_cols].values[-self.seq_len:]

        # Handle missing columns gracefully
        if raw.shape[1] < len(feat_cols):
            logger.warning(
                "Feature count mismatch: expected %d, got %d — padding with zeros.",
                len(feat_cols), raw.shape[1],
            )
            pad = np.zeros((raw.shape[0], len(feat_cols) - raw.shape[1]))
            raw = np.hstack([raw, pad])

        # Replace NaN before scaling
        raw = np.nan_to_num(raw, nan=0.0)

        if self._scaler is not None:
            raw = self._scaler.transform(raw)

        sequence_3d = raw[np.newaxis, ...]  # (1, seq_len, F)
        return self.predict_transition(sequence_3d)

    @property
    def artifact_dir(self) -> Path | None:
        return self._art_dir
