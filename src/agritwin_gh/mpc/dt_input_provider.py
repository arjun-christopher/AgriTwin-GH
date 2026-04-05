"""
Input provider abstraction for the DT closed-loop simulation.

Defines a ``DTInputProvider`` protocol that supplies all simulation inputs
— growth stage, initial state, weather, and (optionally) image observations.

Concrete implementations
------------------------
``SyntheticInputProvider``
    Assembles inputs from CLI arguments and synthetic weather generators.
    Use for offline / evaluation runs where no database is available.

``DatabaseInputProvider``
    Backed by PostgreSQL via ``MPCInputPreparation`` and the weather forecast
    AI model (``WeatherDisturbanceForecast``).  Pass to ``DTLoop`` to run a
    real-data closed loop:  DB → AI → MPC → DT → DB → …

    Example::

        from sqlalchemy.orm import Session
        from agritwin_gh.mpc import DTLoop, DatabaseInputProvider

        provider = DatabaseInputProvider(session, growth_stage="flowering")
        loop = DTLoop(growth_stage="flowering", input_provider=provider)
        for result in loop.run(n_steps=288):
            ...
"""

from __future__ import annotations

import datetime as _dt
import pathlib
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .constants import GROWTH_STAGES
from .state import GreenhouseState, WeatherState

from .dt_runtime_prep import (
    prepare_initial_state,
    prepare_weather_sequence,
)


# ── Dataclass returned by image observation ───────────────────────────────────


@dataclass
class ImageObservation:
    """Result of an image-refresh cycle.

    Contains metadata only — no raw bytes.  Mirrors the shape of
    ``ImagePayload`` from ``state.py`` but is a standalone value-object
    for the DT layer so it can be populated synthetically or from MinIO.
    """

    growth_stage_image_key: str = ""
    growth_stage_label: str = ""
    growth_stage_confidence: float = 0.0
    disease_image_key: str = ""
    disease_label: str = ""
    disease_confidence: float = 0.0
    timestamp: _dt.datetime | None = None
    source: str = "synthetic"  # "synthetic" | "minio"

    def to_dict(self) -> dict[str, Any]:
        return {
            "growth_stage_image_key": self.growth_stage_image_key,
            "growth_stage_label": self.growth_stage_label,
            "growth_stage_confidence": self.growth_stage_confidence,
            "disease_image_key": self.disease_image_key,
            "disease_label": self.disease_label,
            "disease_confidence": self.disease_confidence,
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "source": self.source,
        }


# ── Protocol (interface) ──────────────────────────────────────────────────────


@runtime_checkable
class DTInputProvider(Protocol):
    """Contract that any input-source must satisfy.

    Implementing classes decide *how* data is fetched — terminal input,
    synthetic generators, or live database queries.
    """

    @property
    def growth_stage(self) -> str:
        """Canonical growth-stage label."""
        ...

    @property
    def start_time(self) -> _dt.datetime:
        """Simulation anchor timestamp."""
        ...

    def get_initial_state(self) -> GreenhouseState:
        """Build or retrieve the starting greenhouse state."""
        ...

    def get_weather_sequence(self, n_steps: int, dt_minutes: int) -> list[WeatherState]:
        """Return a weather sequence covering *n_steps* intervals."""
        ...


# ── Concrete: CSV-backed provider ─────────────────────────────────────────────

# Project root is 3 parents above this file:
#   mpc/ → agritwin_gh/ → src/ → AgriTwin-GH/
_PROJECT_ROOT = pathlib.Path(__file__).parents[3]

_INDOOR_CSV = _PROJECT_ROOT / "data/processed/Greenhouse Indoor Conditions/dindigul_greenhouse_indoor_2025.csv"
_WEATHER_CSV = _PROJECT_ROOT / "data/external/Weather Data/dindigul_weather_2025.csv"


class CSVInputProvider:
    """Input provider seeded from the 2025 historical CSV files.

    * Initial greenhouse state   — ``dindigul_greenhouse_indoor_2025.csv``
      Row whose timestamp matches the **current hour** (month/day/hour) mapped
      into 2025.  Only the columns used by the DT loop are read.

    * Weather sequence           — ``dindigul_weather_2025.csv``
      Daily rows whose date matches the **current date** (month/day) mapped
      into 2025, repeated to fill the required number of 5-minute steps.

    Growth stage is forced to *seedling* (index 0), disease risk to 0.0,
    soil moisture to the CSV default of 65 % (not present in the CSV).
    """

    def __init__(
        self,
        start_time: _dt.datetime | None = None,
    ) -> None:
        self._start_time = start_time or _dt.datetime.now()
        self._anchor = self._to_2025(self._start_time)
        self._model_weather_24h: "dict | None" = None
        self._model_growth_result: "dict | None" = None
        self._model_disease_result: "dict | None" = None
        # Run all three startup AI models concurrently — they’re independent
        # and each writes to a separate instance attribute.
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ai_init") as _pool:
            _fw = _pool.submit(self._init_weather_model)
            _fg = _pool.submit(self._init_growth_model)
            _fd = _pool.submit(self._init_disease_model)
        # Re-raise any exception that occurred inside a worker thread.
        _fw.result()
        _fg.result()
        _fd.result()

    # ── Helpers ──────────────────────────────────────────────────

    def _init_weather_model(self) -> None:
        """Run WeatherDisturbanceForecast once on startup using 30-day CSV context.

        Caches the 24h-ahead point prediction in ``self._model_weather_24h``
        (a dict with the same keys as ``WeatherState.to_dict()``).  Falls back
        to ``None`` silently if the model artifacts are unavailable.
        """
        import logging as _log
        _logger = _log.getLogger(__name__)
        try:
            from .disturbance import WeatherDisturbanceForecast
            df = self._load_weather_csv()
            anchor_day = self._anchor.date()
            start_day  = anchor_day - _dt.timedelta(days=30)
            mask = (
                (df["datetime"].dt.date >= start_day)
                & (df["datetime"].dt.date <= anchor_day)
            )
            df_ctx = df[mask].copy()
            if len(df_ctx) < 7:
                _logger.warning(
                    "CSVInputProvider: only %d weather rows available — "
                    "weather forecast model skipped.", len(df_ctx)
                )
                return
            wdf   = WeatherDisturbanceForecast()
            steps = wdf.get_forecast(df_ctx, horizon_hours=24)
            if not steps:
                return
            last = steps[-1]  # last interpolated step ≈ 24 h ahead
            self._model_weather_24h = {
                "temp_external":     float(last.get("temp", 20.0)),
                "humidity_external": float(last.get("humidity", 65.0)),
                "solar_radiation":   float(last.get("solarradiation", 0.0)),
                "windspeed":         float(last.get("windspeed", 0.0)),
                "conditions":        "forecast",
                "timestamp":         None,
            }
            _logger.info(
                "CSVInputProvider: weather-forecast model 24h → "
                "T=%.1f°C  RH=%.1f%%  solar=%.0fW/m²  wind=%.1fkm/h",
                self._model_weather_24h["temp_external"],
                self._model_weather_24h["humidity_external"],
                self._model_weather_24h["solar_radiation"],
                self._model_weather_24h["windspeed"],
            )
        except Exception as exc:
            _logger.warning(
                "CSVInputProvider: weather forecast model unavailable — %s", exc
            )
            self._model_weather_24h = None

    def _init_growth_model(self) -> None:
        """Run the growth-stage-progression LSTM once on startup.

        Reads the last 24 hourly rows from the indoor CSV, synthesises the
        required biological columns for a seedling cycle, then calls
        ``GrowthStageWeights.predict_from_dataframe()`` (which applies the
        same rolling/lag feature engineering as the training notebook).

        Caches the result in ``self._model_growth_result`` as a plain dict.
        """
        import logging as _log
        _logger = _log.getLogger(__name__)
        try:
            import numpy as np
            import pandas as pd
            from .growth_weights import GrowthStageWeights

            df = self._load_indoor_csv()
            anchor = self._anchor.replace(minute=0, second=0, microsecond=0)

            # 24-hour history ending at anchor (needed for rolling features)
            df_hist = df[df["datetime"] <= anchor].tail(24).copy()
            if df_hist.empty:
                _logger.warning(
                    "CSVInputProvider: no indoor rows ≤ anchor — growth model skipped."
                )
                return

            # Rename to match growth model's expected column names
            df_hist = df_hist.rename(columns={
                "datetime":   "timestamp",
                "indoor_CO2": "indoor_co2",
            })

            # Time-derived features expected by _RAW_COLS_FOR_ENGINEERING
            df_hist["year"]        = df_hist["timestamp"].dt.year
            df_hist["month"]       = df_hist["timestamp"].dt.month
            df_hist["day_of_year"] = df_hist["timestamp"].dt.day_of_year
            df_hist["week_of_year"] = (
                df_hist["timestamp"].dt.isocalendar().week.astype(int)
            )
            df_hist["hour"] = df_hist["timestamp"].dt.hour

            # Biological / cycle columns (synthetic seedling cycle, day 0)
            n = len(df_hist)
            df_hist = df_hist.reset_index(drop=True)
            df_hist["stage_index"]                  = 0.0   # seedling
            df_hist["days_from_cycle_start"]        = np.arange(n) / 24.0
            df_hist["hours_in_current_stage"]       = np.arange(n, dtype=float)
            df_hist["days_in_current_stage"]        = np.arange(n) / 24.0
            df_hist["stage_duration_hours"]         = 240.0  # ~10 days for seedling
            df_hist["stage_duration_days"]          = 10.0
            df_hist["stage_progress_pct"]           = (
                df_hist["hours_in_current_stage"] / 240.0 * 100.0
            )
            df_hist["total_cycle_progress_pct"]     = (
                df_hist["days_from_cycle_start"] / 120.0 * 100.0
            )
            df_hist["estimated_hours_to_next_stage"] = (
                240.0 - df_hist["hours_in_current_stage"]
            )
            df_hist["estimated_days_to_next_stage"] = (
                df_hist["estimated_hours_to_next_stage"] / 24.0
            )
            df_hist["is_stage_transition"] = 0.0

            # Pre-computed rolling columns that are also base features
            df_hist["temperature_rolling_mean_24h"] = (
                df_hist["indoor_temp"].rolling(24, min_periods=1).mean()
            )
            df_hist["humidity_rolling_mean_24h"] = (
                df_hist["indoor_humidity"].rolling(24, min_periods=1).mean()
            )
            df_hist["vpd_proxy"]         = df_hist["vpd"]
            df_hist["light_period_flag"] = (df_hist["solarradiation"] > 0).astype(float)
            df_hist["cumulative_gdd_like_index"] = (
                np.maximum(0.0, df_hist["indoor_temp"] - 10.0).cumsum()
            )

            weights = GrowthStageWeights()
            out = weights.predict_from_dataframe(df_hist)

            self._model_growth_result = {
                "current_stage":      out.current_stage,
                "next_stage":         out.next_stage,
                "hours_to_transition": out.hours_to_transition,
                "within_24h":         out.transition_within_24h,
                "within_48h":         out.transition_within_48h,
            }
            _logger.info(
                "CSVInputProvider: growth-progression LSTM → "
                "current=%s  next=%s  h_to_transition=%.1f  within_24h=%s",
                out.current_stage, out.next_stage,
                out.hours_to_transition, out.transition_within_24h,
            )
        except Exception as exc:
            _logger.warning(
                "CSVInputProvider: growth-progression model unavailable — %s", exc
            )
            self._model_growth_result = None

    @staticmethod
    def _compute_disease_trend(
        current: float,
        future: float,
        delta: float = 3.0,
        floor: float = 0.5,
    ) -> str:
        """Rule-derived 24-h trend label — mirrors notebook compute_trend logic.

        Parameters match CONFIG from the disease_progression training notebook:
        ``severity_delta=3.0``, ``severity_floor=0.5`` (both in % units).
        """
        if current < floor and future < floor:
            return "absent"
        if current < floor and future >= floor:
            return "emerging"
        if future < current - delta:
            return "reducing"
        if future > current + delta:
            return "worsening"
        return "stable"

    def _init_disease_model(self) -> None:
        """Run the disease-progression LSTM once on startup.

        Reads the last 24 hourly rows from the indoor CSV, constructs the
        92-column wide-format input expected by the model (disease-specific
        columns zeroed — no real disease history for a fresh CSV run), runs
        the scaler and model, and caches per-disease severity_24h predictions
        in ``self._model_disease_result``.
        """
        import logging as _log
        _logger = _log.getLogger(__name__)
        try:
            import json
            import numpy as np
            import pandas as pd
            import joblib

            from .utils import discover_latest_artifact, load_keras_model

            art_dir = discover_latest_artifact("disease_progression_")
            if art_dir is None or not art_dir.exists():
                _logger.warning(
                    "CSVInputProvider: no disease-progression artifact found — skipped."
                )
                return

            # Load feature column order
            with open(art_dir / "seq_feature_cols.json", encoding="utf-8") as f:
                seq_feature_cols: list[str] = json.load(f)["seq_feature_cols"]

            # Load config
            with open(art_dir / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            history_window = int(cfg.get("history_window", 24))

            # Load LSTM model
            keras_candidates = sorted(art_dir.glob("best_lstm_*.keras"))
            if not keras_candidates:
                _logger.warning(
                    "CSVInputProvider: no disease LSTM .keras found — skipped."
                )
                return
            model = load_keras_model(keras_candidates[0])
            _logger.info(
                "CSVInputProvider: loaded disease-progression LSTM from %s",
                keras_candidates[0].name,
            )

            # Load scaler (optional)
            scaler_path = art_dir / "sequence_feature_scaler.joblib"
            scaler = joblib.load(scaler_path) if scaler_path.exists() else None

            # ── Build 24-row wide-format DataFrame ─────────────────────
            df = self._load_indoor_csv()
            anchor = self._anchor.replace(minute=0, second=0, microsecond=0)
            df_hist = df[df["datetime"] <= anchor].tail(history_window).copy()
            if df_hist.empty:
                _logger.warning(
                    "CSVInputProvider: no indoor rows ≤ anchor — disease model skipped."
                )
                return

            n = len(df_hist)
            df_hist = df_hist.reset_index(drop=True)

            # Shared sensor + biological columns
            rows_data: list[dict] = []
            for i in range(n):
                dt = df_hist.loc[i, "datetime"]
                rows_data.append({
                    "cycle_id":               1.0,
                    "stage_index":            0.0,   # seedling
                    "days_from_cycle_start":  i / 24.0,
                    "day_of_year":            float(dt.day_of_year),
                    "week_of_year":           float(dt.isocalendar().week),
                    "hour":                   float(dt.hour),
                    "hours_in_current_stage": float(i),
                    "stage_progress_pct":     (i / 240.0) * 100.0,
                    "total_cycle_progress_pct": (i / 24.0 / 120.0) * 100.0,
                    "is_stage_transition":    0.0,
                    "indoor_temp":            float(df_hist.loc[i, "indoor_temp"]),
                    "indoor_humidity":        float(df_hist.loc[i, "indoor_humidity"]),
                    "indoor_air_velocity":    float(df_hist.loc[i, "indoor_air_velocity"]),
                    "indoor_CO2":             float(df_hist.loc[i, "indoor_CO2"]),
                    "solarradiation":         float(df_hist.loc[i, "solarradiation"]),
                    "day_night_flag":         float(df_hist.loc[i, "day_night_flag"]),
                    "vpd":                    float(df_hist.loc[i, "vpd"]),
                    "dew_point":              float(df_hist.loc[i, "dew_point"]),
                    "leaf_wetness_proxy":     float(df_hist.loc[i, "leaf_wetness_proxy"]),
                })
            wide = pd.DataFrame(rows_data)

            # Rolling features
            wide["temperature_rolling_mean_24h"] = (
                wide["indoor_temp"].rolling(history_window, min_periods=1).mean()
            )
            wide["humidity_rolling_mean_24h"] = (
                wide["indoor_humidity"].rolling(history_window, min_periods=1).mean()
            )
            wide["vpd_proxy"] = wide["vpd"]
            wide["cumulative_gdd_like_index"] = (
                np.maximum(0.0, wide["indoor_temp"] - 10.0).cumsum()
            )

            # Disease-specific columns (zeroed — no real disease history)
            _DISEASES = [
                "early_blight", "late_blight", "leaf_mold",
                "powdery_mildew", "spider_mites",
            ]
            for d in _DISEASES:
                for col in [
                    "infection_growth_rate_hourly", "stage_susceptibility_score",
                    "disease_risk_score", "control_action_flag",
                    "outbreak_trigger_flag", "current_infection_pct",
                    "disease_present_flag",
                ]:
                    wide[f"{col}__{d}"] = 0.0

            # One-hot: cycle_label / season_label (unknown → all zeros)
            for cl in ["kharif_2024", "kharif_2025", "rabi_2024", "summer_2025"]:
                wide[f"cycle_label_{cl}"] = 0.0
            for sl in ["dry_cool", "northeast_monsoon", "southwest_monsoon", "summer"]:
                wide[f"season_label_{sl}"] = 0.0

            # One-hot: stage_name → seedling
            for sn in [
                "early_vegetative", "flowering", "flowering_initiation",
                "ripe", "seedling", "unripe",
            ]:
                wide[f"stage_name_{sn}"] = 1.0 if sn == "seedling" else 0.0

            # One-hot: control_action_type → "none" for every disease
            _CTRL = {
                "early_blight":   ["copper_treatment", "fungicide_spray", "none",
                                   "reduced_irrigation"],
                "late_blight":    ["fungicide_spray", "humidity_reduction", "none",
                                   "ventilation_increase"],
                "leaf_mold":      ["humidity_reduction", "leaf_pruning", "none",
                                   "ventilation_increase"],
                "powdery_mildew": ["improved_airflow", "none",
                                   "potassium_bicarbonate", "sulfur_treatment"],
                "spider_mites":   ["acaricide_spray", "biological_release",
                                   "humidity_increase", "none"],
            }
            for d, actions in _CTRL.items():
                for a in actions:
                    wide[f"control_action_type__{d}_{a}"] = (
                        1.0 if a == "none" else 0.0
                    )

            # Ensure every seq_feature_col exists (fill any gaps with 0)
            for c in seq_feature_cols:
                if c not in wide.columns:
                    wide[c] = 0.0

            # ── Scale and run model ─────────────────────────────────────
            raw = wide[seq_feature_cols].values[-history_window:]  # (24, 92)
            raw = np.nan_to_num(raw, nan=0.0)
            if scaler is not None:
                raw = scaler.transform(raw)
            sequence = raw[np.newaxis, ...]  # (1, 24, 92)

            pres_prob_arr, fut_scaled_arr = model.predict(sequence, verbose=0)
            pres_prob  = pres_prob_arr[0]   # (5,)
            fut_scaled = fut_scaled_arr[0]  # (5,)

            result: dict[str, dict] = {}
            for i, d in enumerate(_DISEASES):
                # current_infection_pct is zeroed for a fresh CSV run (no history)
                current_sev = 0.0
                future_sev  = float(np.clip(fut_scaled[i] * 100.0, 0.0, 100.0))
                result[d] = {
                    "present":      bool(float(pres_prob[i]) >= 0.5),
                    "severity_24h": future_sev,
                    "trend_24h":    self._compute_disease_trend(current_sev, future_sev),
                }

            self._model_disease_result = result
            _logger.info(
                "CSVInputProvider: disease-progression LSTM → %s",
                "  ".join(
                    f"{d}={v['severity_24h']:.1f}%"
                    for d, v in result.items()
                ),
            )
        except Exception as exc:
            _logger.warning(
                "CSVInputProvider: disease-progression model unavailable — %s", exc
            )
            self._model_disease_result = None

    # ── Per-cycle AI model refresh ─────────────────────────────────────────
    # Called by DTLoop at every MPC cadence step (every 15 simulated minutes).
    # Keeps the growth/disease LSTM forecasts coherent with the evolving DT
    # state instead of relying solely on the initial CSV conditions.

    def refresh_ai_models(
        self,
        hourly_states: list,
        stage: str = "seedling",
        timestamp: "_dt.datetime | None" = None,
        elapsed_hours: float = 0.0,
    ) -> None:
        """Re-run both progression LSTMs with the current DT state history.

        Parameters
        ----------
        hourly_states:
            Up to 24 ``GreenhouseState`` objects, oldest first, sub-sampled to
            approximately hourly frequency from the 5-min DT state buffer.
        stage:
            Canonical growth-stage label for the current step.
        timestamp:
            Simulation anchor time for time-derived features.  ``None`` → now.
        """
        ts = timestamp or _dt.datetime.now()
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai_refresh") as pool:
            _fg = pool.submit(self._refresh_growth_model, hourly_states, stage, ts, elapsed_hours)
            _fd = pool.submit(self._refresh_disease_model, hourly_states, stage, ts)
        _fg.result()
        _fd.result()

    def _refresh_growth_model(
        self,
        hourly_states: list,
        stage: str,
        ts: _dt.datetime,
        elapsed_hours: float = 0.0,
    ) -> None:
        """Rebuild the growth-progression LSTM forecast from DT state history."""
        import logging as _log
        _logger = _log.getLogger(__name__)
        try:
            import numpy as np
            import pandas as pd
            from .growth_weights import GrowthStageWeights
            from .constants import stage_label_to_index
            from .realtime_core import STAGE_DURATION_HOURS

            if not hourly_states:
                return

            states = list(hourly_states)
            while len(states) < 24:
                states = [states[0]] + states
            states = states[-24:]

            stage_idx    = float(stage_label_to_index(stage))
            stage_dur_h  = float(STAGE_DURATION_HOURS.get(stage, 240))
            stage_dur_d  = stage_dur_h / 24.0

            rows = []
            for i, st in enumerate(states):
                row_ts = ts - _dt.timedelta(hours=(24 - 1 - i))
                rh     = float(st.indoor_humidity)
                temp   = float(st.indoor_temp)
                dew    = temp - (100.0 - rh) / 5.0   # simplified Magnus approx.
                # hours_in_current_stage counts backward from the current
                # elapsed time so the LSTM sees the plant's real age in-stage
                # rather than always starting from 0.
                hours_in      = max(0.0, elapsed_hours - (24 - 1 - i))
                days_in       = hours_in / 24.0
                hours_to_next = max(0.0, stage_dur_h - hours_in)
                rows.append({
                    "timestamp":                     row_ts,
                    "indoor_temp":                   temp,
                    "indoor_humidity":               rh,
                    "indoor_co2":                    float(st.co2),
                    "solarradiation":                float(st.light_intensity),
                    "vpd":                           float(st.vpd),
                    "dew_point":                     dew,
                    "leaf_wetness_proxy":            float(st.leaf_wetness_proxy),
                    "indoor_air_velocity":           0.5,
                    "day_night_flag":                1.0 if st.light_intensity > 50 else 0.0,
                    "year":                          float(row_ts.year),
                    "month":                         float(row_ts.month),
                    "day_of_year":                   float(row_ts.timetuple().tm_yday),
                    "week_of_year":                  float(row_ts.isocalendar()[1]),
                    "hour":                          float(row_ts.hour),
                    "stage_index":                   stage_idx,
                    "days_from_cycle_start":         days_in,
                    "hours_in_current_stage":        hours_in,
                    "days_in_current_stage":         days_in,
                    "stage_duration_hours":          stage_dur_h,
                    "stage_duration_days":           stage_dur_d,
                    "stage_progress_pct":            (hours_in / stage_dur_h) * 100.0,
                    "total_cycle_progress_pct":      (days_in / 120.0) * 100.0,
                    "estimated_hours_to_next_stage": hours_to_next,
                    "estimated_days_to_next_stage":  hours_to_next / 24.0,
                    "is_stage_transition":           0.0,
                })

            df_hist = pd.DataFrame(rows)
            df_hist["temperature_rolling_mean_24h"] = (
                df_hist["indoor_temp"].rolling(24, min_periods=1).mean()
            )
            df_hist["humidity_rolling_mean_24h"] = (
                df_hist["indoor_humidity"].rolling(24, min_periods=1).mean()
            )
            df_hist["vpd_proxy"]               = df_hist["vpd"]
            df_hist["light_period_flag"]       = df_hist["day_night_flag"]
            df_hist["cumulative_gdd_like_index"] = (
                np.maximum(0.0, df_hist["indoor_temp"] - 10.0).cumsum()
            )

            wsw = GrowthStageWeights()
            out = wsw.predict_from_dataframe(df_hist)
            if out is None:
                return

            self._model_growth_result = {
                "current_stage":       out.current_stage,
                "next_stage":          out.next_stage,
                "hours_to_transition": out.hours_to_transition,
                "within_24h":          out.transition_within_24h,
                "within_48h":          out.transition_within_48h,
            }
            _logger.debug(
                "CSVInputProvider.refresh: growth → next=%s  h_to_transition=%.1f",
                out.next_stage, out.hours_to_transition,
            )
        except Exception as exc:
            _logger.warning(
                "CSVInputProvider: growth model refresh failed — %s", exc
            )

    def _refresh_disease_model(
        self,
        hourly_states: list,
        stage: str,
        ts: _dt.datetime,
    ) -> None:
        """Rebuild the disease-progression LSTM forecast from DT state history."""
        import logging as _log
        _logger = _log.getLogger(__name__)
        try:
            import json
            import numpy as np
            import pandas as pd
            import joblib
            from .utils import discover_latest_artifact, load_keras_model
            from .constants import stage_label_to_index

            art_dir = discover_latest_artifact("disease_progression_")
            if art_dir is None or not art_dir.exists():
                return

            with open(art_dir / "seq_feature_cols.json", encoding="utf-8") as f:
                seq_feature_cols: list = json.load(f)["seq_feature_cols"]
            with open(art_dir / "config.json", encoding="utf-8") as f:
                cfg = json.load(f)
            history_window = int(cfg.get("history_window", 24))

            keras_candidates = sorted(art_dir.glob("best_lstm_*.keras"))
            if not keras_candidates:
                return
            model = load_keras_model(keras_candidates[0])

            scaler_path = art_dir / "sequence_feature_scaler.joblib"
            scaler      = joblib.load(scaler_path) if scaler_path.exists() else None

            states = list(hourly_states)
            while len(states) < history_window:
                states = [states[0]] + states
            states = states[-history_window:]

            stage_idx   = float(stage_label_to_index(stage))
            stage_snake = stage.lower().replace(" ", "_")

            rows_data = []
            for i, st in enumerate(states):
                row_ts = ts - _dt.timedelta(hours=(history_window - 1 - i))
                rh     = float(st.indoor_humidity)
                temp   = float(st.indoor_temp)
                dew    = temp - (100.0 - rh) / 5.0
                rows_data.append({
                    "cycle_id":                 1.0,
                    "stage_index":              stage_idx,
                    "days_from_cycle_start":    i / 24.0,
                    "day_of_year":              float(row_ts.timetuple().tm_yday),
                    "week_of_year":             float(row_ts.isocalendar()[1]),
                    "hour":                     float(row_ts.hour),
                    "hours_in_current_stage":   float(i),
                    "stage_progress_pct":       (i / 240.0) * 100.0,
                    "total_cycle_progress_pct": (i / 24.0 / 120.0) * 100.0,
                    "is_stage_transition":      0.0,
                    "indoor_temp":              temp,
                    "indoor_humidity":          rh,
                    "indoor_air_velocity":      0.5,
                    "indoor_CO2":               float(st.co2),
                    "solarradiation":           float(st.light_intensity),
                    "day_night_flag":           1.0 if st.light_intensity > 50 else 0.0,
                    "vpd":                      float(st.vpd),
                    "dew_point":                dew,
                    "leaf_wetness_proxy":       float(st.leaf_wetness_proxy),
                })
            wide = pd.DataFrame(rows_data)

            wide["temperature_rolling_mean_24h"] = (
                wide["indoor_temp"].rolling(history_window, min_periods=1).mean()
            )
            wide["humidity_rolling_mean_24h"] = (
                wide["indoor_humidity"].rolling(history_window, min_periods=1).mean()
            )
            wide["vpd_proxy"] = wide["vpd"]
            wide["cumulative_gdd_like_index"] = (
                np.maximum(0.0, wide["indoor_temp"] - 10.0).cumsum()
            )

            _DISEASES = [
                "early_blight", "late_blight", "leaf_mold",
                "powdery_mildew", "spider_mites",
            ]
            for d in _DISEASES:
                for col in [
                    "infection_growth_rate_hourly", "stage_susceptibility_score",
                    "disease_risk_score", "control_action_flag",
                    "outbreak_trigger_flag", "current_infection_pct",
                    "disease_present_flag",
                ]:
                    wide[f"{col}__{d}"] = 0.0

            for cl in ["kharif_2024", "kharif_2025", "rabi_2024", "summer_2025"]:
                wide[f"cycle_label_{cl}"] = 0.0
            for sl in ["dry_cool", "northeast_monsoon", "southwest_monsoon", "summer"]:
                wide[f"season_label_{sl}"] = 0.0

            for sn in [
                "early_vegetative", "flowering", "flowering_initiation",
                "ripe", "seedling", "unripe",
            ]:
                wide[f"stage_name_{sn}"] = 1.0 if sn == stage_snake else 0.0

            _CTRL = {
                "early_blight":   ["copper_treatment", "fungicide_spray", "none",
                                   "reduced_irrigation"],
                "late_blight":    ["fungicide_spray", "humidity_reduction", "none",
                                   "ventilation_increase"],
                "leaf_mold":      ["humidity_reduction", "leaf_pruning", "none",
                                   "ventilation_increase"],
                "powdery_mildew": ["improved_airflow", "none",
                                   "potassium_bicarbonate", "sulfur_treatment"],
                "spider_mites":   ["acaricide_spray", "biological_release",
                                   "humidity_increase", "none"],
            }
            for d, actions in _CTRL.items():
                for a in actions:
                    wide[f"control_action_type__{d}_{a}"] = (
                        1.0 if a == "none" else 0.0
                    )

            for c in seq_feature_cols:
                if c not in wide.columns:
                    wide[c] = 0.0

            raw = wide[seq_feature_cols].values[-history_window:]
            raw = np.nan_to_num(raw, nan=0.0)
            if scaler is not None:
                raw = scaler.transform(raw)
            sequence = raw[np.newaxis, ...]

            pres_prob_arr, fut_scaled_arr = model.predict(sequence, verbose=0)
            pres_prob  = pres_prob_arr[0]
            fut_scaled = fut_scaled_arr[0]

            result: dict = {}
            for i, d in enumerate(_DISEASES):
                future_sev = float(np.clip(fut_scaled[i] * 100.0, 0.0, 100.0))
                result[d] = {
                    "present":      bool(float(pres_prob[i]) >= 0.5),
                    "severity_24h": future_sev,
                    "trend_24h":    self._compute_disease_trend(0.0, future_sev),
                }

            self._model_disease_result = result
            _logger.debug(
                "CSVInputProvider.refresh: disease → %s",
                "  ".join(f"{d}={v['severity_24h']:.1f}%" for d, v in result.items()),
            )
        except Exception as exc:
            _logger.warning(
                "CSVInputProvider: disease model refresh failed — %s", exc
            )

    @staticmethod
    def _to_2025(dt: _dt.datetime) -> _dt.datetime:
        """Return dt with the year replaced by 2025 (handles Feb 29 edge)."""
        try:
            return dt.replace(year=2025)
        except ValueError:
            return dt.replace(year=2025, day=28)

    @staticmethod
    def _load_indoor_csv() -> "Any":
        import pandas as pd
        return pd.read_csv(_INDOOR_CSV, parse_dates=["datetime"])

    @staticmethod
    def _load_weather_csv() -> "Any":
        import pandas as pd
        return pd.read_csv(_WEATHER_CSV, parse_dates=["datetime"])

    # ── Protocol implementation ───────────────────────────────────

    @property
    def growth_stage(self) -> str:
        return "seedling"

    @property
    def start_time(self) -> _dt.datetime:
        return self._start_time

    def get_initial_state(self) -> GreenhouseState:
        """Return the greenhouse state from the matching 2025 CSV hour.

        Columns used (indoor CSV):
            ``indoor_temp``, ``indoor_humidity``, ``indoor_CO2``,
            ``solarradiation``, ``vpd``, ``leaf_wetness_proxy``

        Fixed overrides: growth_stage_index=0 (seedling), disease_risk_score=0.0,
        soil_moisture=65.0 (not in CSV), timestamp=now.
        """
        df = self._load_indoor_csv()
        anchor = self._anchor.replace(minute=0, second=0, microsecond=0)

        mask = (
            (df["datetime"].dt.month == anchor.month)
            & (df["datetime"].dt.day == anchor.day)
            & (df["datetime"].dt.hour == anchor.hour)
        )
        row = df[mask]
        if row.empty:
            # Nearest hour fallback
            df["_diff"] = (df["datetime"] - anchor).abs()
            row = df.nsmallest(1, "_diff")
        if row.empty:
            return GreenhouseState(timestamp=self._start_time)

        r = row.iloc[0]
        rh = float(r["indoor_humidity"])
        leaf = float(r["leaf_wetness_proxy"])
        vpd = float(r["vpd"])
        # Estimate soil moisture from available sensor columns.
        # Higher humidity and leaf wetness → more available water.
        # Higher VPD drives evaporation → dries the soil.
        soil = 55.0 + (rh - 65.0) * 0.30 + leaf * 8.0 - vpd * 2.0
        soil = max(40.0, min(90.0, round(soil, 1)))
        return GreenhouseState(
            indoor_temp=float(r["indoor_temp"]),
            indoor_humidity=rh,
            co2=float(r["indoor_CO2"]),
            light_intensity=float(r["solarradiation"]),
            vpd=vpd,
            leaf_wetness_proxy=leaf,
            soil_moisture=soil,
            growth_stage_index=0,        # always seedling at start
            disease_risk_score=0.0,      # always 0 at start
            timestamp=self._start_time,
        )

    def get_weather_sequence(
        self, n_steps: int, dt_minutes: int = 5,
    ) -> list[WeatherState]:
        """Return per-step weather from the 2025 daily external-weather CSV.

        Each daily row is repeated for ``24 * 60 // dt_minutes`` steps.
        When the required span exceeds the CSV coverage, the last known row
        is carried forward.

        Columns used (weather CSV):
            ``temp``, ``humidity``, ``windspeed``, ``solarradiation``,
            ``conditions``
        """
        df = self._load_weather_csv()
        steps_per_day = max(1, 24 * 60 // dt_minutes)
        anchor_date = self._anchor.date()

        n_days_needed = (n_steps // steps_per_day) + 2  # +2 for partial days
        states: list[WeatherState] = []
        last_ws = WeatherState()

        for i in range(n_days_needed):
            day = anchor_date + _dt.timedelta(days=i)
            mask = (
                (df["datetime"].dt.month == day.month)
                & (df["datetime"].dt.day == day.day)
            )
            row = df[mask]
            if row.empty:
                ws = last_ws
            else:
                r = row.iloc[0]
                ws = WeatherState(
                    temp_external=float(r["temp"]),
                    humidity_external=float(r["humidity"]),
                    solar_radiation=float(r["solarradiation"]),
                    windspeed=float(r["windspeed"]),
                    conditions=str(r["conditions"]),
                )
            last_ws = ws
            for _ in range(steps_per_day):
                states.append(ws)
                if len(states) >= n_steps:
                    break
            if len(states) >= n_steps:
                break

        # Pad if CSV coverage was insufficient
        while len(states) < n_steps:
            states.append(last_ws)

        return states[:n_steps]


# ── Concrete: synthetic / terminal-driven provider ────────────────────────────


class SyntheticInputProvider:
    """Input provider backed by synthetic generators and CLI arguments.

    Parameters
    ----------
    growth_stage:
        Canonical label (e.g. ``"flowering"``).
    start_time:
        Simulation anchor.  ``None`` → ``datetime.now()``.
    base_temp:
        Mean outdoor temperature for the synthetic weather model.
    diurnal_amp:
        Half-range of the diurnal sinusoid (°C).
    """

    def __init__(
        self,
        growth_stage: str,
        start_time: _dt.datetime | None = None,
        base_temp: float = 20.0,
        diurnal_amp: float = 8.0,
    ) -> None:
        if growth_stage not in GROWTH_STAGES:
            raise ValueError(
                f"Unknown growth stage '{growth_stage}'. "
                f"Valid: {GROWTH_STAGES}"
            )
        self._growth_stage = growth_stage
        self._start_time = start_time or _dt.datetime.now()
        self._base_temp = base_temp
        self._diurnal_amp = diurnal_amp

    # ── Protocol implementation ───────────────────────────────────

    @property
    def growth_stage(self) -> str:
        return self._growth_stage

    @property
    def start_time(self) -> _dt.datetime:
        return self._start_time

    def get_initial_state(self) -> GreenhouseState:
        return prepare_initial_state(
            growth_stage=self._growth_stage,
            timestamp=self._start_time,
            base_temp=self._base_temp,
        )

    def get_weather_sequence(
        self, n_steps: int, dt_minutes: int = 5,
    ) -> list[WeatherState]:
        return prepare_weather_sequence(
            n_steps=n_steps,
            start_time=self._start_time,
            dt_minutes=dt_minutes,
            base_temp=self._base_temp,
            diurnal_amp=self._diurnal_amp,
        )


# ── Concrete: database-backed provider ───────────────────────────────────────


class DatabaseInputProvider:
    """Input provider backed by PostgreSQL + AI weather forecast.

    Reads the most recent greenhouse sensor row and real weather history
    from the database to seed the DT simulation with live data.

    Intended use: pass to ``DTLoop`` so the synthetic loop runs with a real
    starting state and real-forecast weather instead of diurnal sinusoids.

    Parameters
    ----------
    session:
        A live SQLAlchemy ``Session``.
    growth_stage:
        Canonical growth-stage label (validated against ``GROWTH_STAGES``).
    weather_run_id:
        Artifact run-ID for ``WeatherDisturbanceForecast``.  ``None`` →
        loaded from the default MPC config.
    device:
        PyTorch device string for the weather-forecast ensemble (``"cpu"``
        by default).
    """

    def __init__(
        self,
        session: object,
        growth_stage: str,
        weather_run_id: str | None = None,
        device: str = "cpu",
    ) -> None:
        from .config import load_mpc_config
        from .disturbance import WeatherDisturbanceForecast
        from .mpc_input_preparation import MPCInputPreparation

        if growth_stage not in GROWTH_STAGES:
            raise ValueError(
                f"Unknown growth stage '{growth_stage}'. Valid: {GROWTH_STAGES}"
            )
        self._growth_stage = growth_stage
        self._input_prep = MPCInputPreparation(session)  # type: ignore[arg-type]
        cfg = load_mpc_config()
        run_id = weather_run_id or cfg.environment_forecast_run_id
        self._weather_forecast = WeatherDisturbanceForecast(run_id=run_id, device=device)
        self._cfg = cfg

    # ── Protocol implementation ───────────────────────────────────────

    @staticmethod
    def _prior_year_anchor() -> _dt.datetime:
        """Return the same calendar date/hour as 'now' but one year ago.

        The previous year's data is guaranteed to exist in the DB whereas the
        current-year tables may still be empty.  Using this anchor lets both
        the initial greenhouse state and the weather-forecast context pull from
        real historical records.
        """
        now = _dt.datetime.now()
        try:
            return now.replace(year=now.year - 1)
        except ValueError:
            # Edge case: Feb 29 on a leap year — fall back to Feb 28
            return now.replace(year=now.year - 1, day=28)

    @property
    def growth_stage(self) -> str:
        return self._growth_stage

    @property
    def start_time(self) -> _dt.datetime:
        return self._prior_year_anchor()

    def get_initial_state(self) -> GreenhouseState:
        """Return historical greenhouse state from the same date/hour last year.

        Looks up the ``greenhouse_data`` row closest to (today's month/day/hour)
        from exactly one year ago so the DT loop starts from a realistic,
        data-backed initial state even when the current-year table is empty.
        Falls back to a synthetic state when no matching DB row is found.

        Regardless of which year the DB row comes from, the returned state
        always carries today's timestamp, growth_stage_index=0 (seedling) and
        disease_risk_score=0.0 so the UI shows the correct initial conditions.
        """
        anchor = self._prior_year_anchor()
        row = self._input_prep.get_greenhouse_row_at_datetime(anchor)
        if row is None:
            return prepare_initial_state(
                growth_stage=self._growth_stage,
                timestamp=_dt.datetime.now(),
            )
        state = GreenhouseState.from_db_row(row)
        state.timestamp = _dt.datetime.now()   # always stamp with today's time
        return state

    def get_weather_sequence(
        self, n_steps: int, dt_minutes: int = 5,
    ) -> list[WeatherState]:
        """Return a per-step weather sequence for *n_steps* DT steps.

        Strategy
        --------
        1. Query the 30 days of ``weather_data`` ending at the same
           date/hour **one year ago** (not the current date), so the AI
           forecast always has real historical context rather than an empty
           current-year window.
        2. Run the ``WeatherDisturbanceForecast`` AI model to project
           48 h ahead at hourly resolution.
        3. Expand hourly dicts to per-step ``WeatherState`` objects by
           repeating each hourly value for ``60 // dt_minutes`` steps.
        4. If the database is empty or the model fails, fall back to a
           synthetic sinusoidal sequence (``prepare_weather_sequence``).
        """
        import math

        try:
            anchor = self._prior_year_anchor()
            df_weather = self._input_prep.get_weather_context_before(
                before_dt=anchor, lookback_days=30
            )
            if df_weather.empty:
                raise ValueError("Empty weather context — falling back to synthetic.")

            horizon_hours = max(48, math.ceil(n_steps * dt_minutes / 60) + 2)
            forecast_dicts = self._weather_forecast.get_forecast(
                df_weather, horizon_hours=min(horizon_hours, 48)
            )
            steps_per_hour = max(1, 60 // dt_minutes)

            # Expand hourly forecast to per-step WeatherState objects
            states: list[WeatherState] = []
            for d in forecast_dicts:
                ws = WeatherState(
                    temp_external=float(d.get("temp", 20.0)),
                    humidity_external=float(d.get("humidity", 65.0)),
                    solar_radiation=float(d.get("solarradiation", 0.0)),
                    windspeed=float(d.get("windspeed", 0.0)),
                    conditions="forecast",
                )
                for _ in range(steps_per_hour):
                    states.append(ws)
                    if len(states) >= n_steps:
                        break
                if len(states) >= n_steps:
                    break

            # Pad to exactly n_steps if the forecast was shorter
            while len(states) < n_steps:
                states.append(states[-1] if states else WeatherState())

            return states[:n_steps]

        except Exception:
            # Graceful fallback: synthetic sinusoidal weather
            return prepare_weather_sequence(
                n_steps=n_steps,
                start_time=self.start_time,
                dt_minutes=dt_minutes,
            )
