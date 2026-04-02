"""
Input provider abstraction for the DT closed-loop simulation.

Defines a ``DTInputProvider`` protocol that supplies all simulation inputs
— growth stage, initial state, weather, and (optionally) image observations.

The concrete ``SyntheticInputProvider`` assembles inputs from terminal /
CLI arguments and synthetic weather.  Swap it for a ``DatabaseInputProvider``
to read from PostgreSQL + MinIO without changing the loop.

Future DB migration path
------------------------
1. Create ``DatabaseInputProvider(DTInputProvider)`` backed by
   ``MPCInputPreparation`` and ``StateFusion``.
2. Growth stage: ``StateFusion.fuse()`` step 5 already resolves the
   authoritative stage from DB + progression model + classifier.
3. Initial state: ``MPCInputPreparation.get_latest_greenhouse_row()``
   → ``GreenhouseState.from_db_row()``.
4. Weather: ``WeatherDisturbanceForecast.get_forecast()`` on the
   ``MPCInputPreparation.get_weather_context()`` DataFrame.
5. Image observation: ``ImageStreamer.get_random_growth_stage_image()``
   / ``get_random_disease_image()`` — already cache-backed via TTL.
6. Instantiate the ``DatabaseInputProvider`` with a SQLAlchemy session
   and pass it to ``DTLoop`` — zero changes to loop logic.
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
