"""
Weather disturbance forecast wrapper for the MPC control layer.

Wraps ``EnvironmentForecastModel`` (PyTorch ensemble) and interpolates its
24 h / 48 h point predictions to 5-minute MPC timesteps via linear
interpolation.
"""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import DT_MINUTES
from .utils import _ARTIFACTS_DIR, discover_latest_artifact

logger = logging.getLogger(__name__)


# ── Public class ──────────────────────────────────────────────────────────────


class WeatherDisturbanceForecast:
    """Adapter that turns the ensemble weather forecast into MPC-resolution
    disturbance vectors.

    Parameters
    ----------
    run_id:
        Explicit artifact run ID (e.g. ``"environment_forecast_20260319_114027"``).
        When *None* the latest ``environment_forecast_*`` directory is used.
    device:
        PyTorch device string.
    """

    def __init__(self, run_id: str | None = None, device: str = "cpu") -> None:
        if run_id:
            art_dir = _ARTIFACTS_DIR / run_id
        else:
            art_dir = discover_latest_artifact("environment_forecast_")
        if art_dir is None or not art_dir.exists():
            raise FileNotFoundError(
                f"No environment forecast artifact found (run_id={run_id!r})."
            )
        self._art_dir = art_dir
        self._device = device
        self._model: Any = None  # lazy-loaded

    # ── Lazy model loading ─────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        loader_path = self._art_dir / "environment_forecast_loader.py"
        if not loader_path.exists():
            raise FileNotFoundError(
                f"Missing loader script: {loader_path}"
            )

        # Dynamically import the loader module from the artifact directory.
        spec = importlib.util.spec_from_file_location(
            "environment_forecast_loader", loader_path
        )
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]

        # Locate the main .pt bundle
        pt_files = sorted(self._art_dir.glob("*.pt"))
        main_model_path = pt_files[0] if pt_files else None

        self._model = mod.EnvironmentForecastModel(
            artifacts_dir=str(self._art_dir),
            main_model_path=str(main_model_path) if main_model_path else None,
            device=self._device,
        )
        logger.info(
            "Loaded EnvironmentForecastModel from %s", self._art_dir.name
        )

    # ── Public interface ───────────────────────────────────────────────

    def get_forecast(
        self,
        df_context: pd.DataFrame,
        horizon_hours: int = 48,
    ) -> list[dict[str, Any]]:
        """Run the weather ensemble and interpolate to *DT_MINUTES* resolution.

        Parameters
        ----------
        df_context:
            Historical weather DataFrame suitable for
            ``EnvironmentForecastModel.predict()``.
        horizon_hours:
            How many hours ahead to produce (max 48).

        Returns
        -------
        list[dict]
            One dict per *DT_MINUTES* step over the requested horizon.
            Each dict maps variable names (e.g. ``"temp"``, ``"humidity"``)
            to interpolated float values.
        """
        self._ensure_model()

        raw: dict[str, dict[str, float]] = self._model.predict(df_context)
        # raw  →  { "temp": {"24h": 22.3, "48h": 21.1}, ... }

        horizon_hours = min(horizon_hours, 48)
        n_steps = horizon_hours * (60 // DT_MINUTES)
        timeline_hours = np.linspace(0.0, horizon_hours, n_steps, endpoint=False)

        forecast_steps: list[dict[str, Any]] = []
        for t_h in timeline_hours:
            step: dict[str, Any] = {"offset_hours": round(float(t_h), 4)}
            for var, preds in raw.items():
                v24 = preds.get("24h", 0.0)
                v48 = preds.get("48h", v24)
                # Linear interpolation: 0 h → v24, 24 h → v24, 48 h → v48
                if t_h <= 24.0:
                    step[var] = v24
                else:
                    frac = (t_h - 24.0) / 24.0
                    step[var] = v24 + frac * (v48 - v24)
            forecast_steps.append(step)

        logger.debug(
            "Weather forecast interpolated: %d steps over %d h", n_steps, horizon_hours
        )
        return forecast_steps

    # ── Convenience ────────────────────────────────────────────────────

    @property
    def artifact_dir(self) -> Path:
        return self._art_dir
