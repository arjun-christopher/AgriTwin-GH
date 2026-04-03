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
    disease_image_key: str = ""
    disease_label: str = ""
    timestamp: _dt.datetime | None = None
    source: str = "synthetic"  # "synthetic" | "minio"

    def to_dict(self) -> dict[str, Any]:
        return {
            "growth_stage_image_key": self.growth_stage_image_key,
            "growth_stage_label": self.growth_stage_label,
            "disease_image_key": self.disease_image_key,
            "disease_label": self.disease_label,
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

    @property
    def growth_stage(self) -> str:
        return self._growth_stage

    @property
    def start_time(self) -> _dt.datetime:
        row = self._input_prep.get_latest_greenhouse_row()
        if row and row.get("datetime"):
            return row["datetime"]
        return _dt.datetime.now()

    def get_initial_state(self) -> GreenhouseState:
        """Return most recent ``greenhouse_data`` row as a ``GreenhouseState``.

        Falls back to a synthetic initial state when the table is empty.
        """
        row = self._input_prep.get_latest_greenhouse_row()
        if row is None:
            return prepare_initial_state(
                growth_stage=self._growth_stage,
                timestamp=_dt.datetime.now(),
            )
        return GreenhouseState.from_db_row(row)

    def get_weather_sequence(
        self, n_steps: int, dt_minutes: int = 5,
    ) -> list[WeatherState]:
        """Return a per-step weather sequence for *n_steps* DT steps.

        Strategy
        --------
        1. Query the last 30 days of ``weather_data`` from PostgreSQL.
        2. Run the ``WeatherDisturbanceForecast`` AI model to project
           48 h ahead at hourly resolution.
        3. Expand hourly dicts to per-step ``WeatherState`` objects by
           repeating each hourly value for ``60 // dt_minutes`` steps.
        4. If the database is empty or the model fails, fall back to a
           synthetic sinusoidal sequence (``prepare_weather_sequence``).
        """
        import math

        try:
            df_weather = self._input_prep.get_weather_context(lookback_days=30)
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
