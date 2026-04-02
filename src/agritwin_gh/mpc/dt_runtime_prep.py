"""
Runtime preparation for the DT closed-loop simulation.

Builds the initial state, weather sequence, and MPC solver from runtime
context (current wall-clock time, user-supplied growth stage) **without**
requiring a database connection.

This module bridges the gap between the purely-synthetic helpers in
``experiment_runner`` and a future DB-backed ``StateFusion`` path.
The weather sequence is anchored to the *actual* hour-of-day so the
diurnal cycle lines up with reality.

Functions
---------
``prepare_initial_state``
    Builds the starting ``GreenhouseState`` anchored to runtime time.

``prepare_weather_sequence``
    Generates a diurnal weather sequence starting from *now*.

``prepare_growth_stages``
    Constant-stage sequence (wraps ``generate_default_growth_stages``).

``build_mpc_solver``
    Creates a configured ``MPCSolver`` with sensible closed-loop defaults.

``build_fused_state``
    Assembles a minimal ``FusedState`` from components for the solver.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

import numpy as np

from .config import MPCConfig, load_mpc_config
from .constants import (
    DT_MINUTES,
    compute_disease_risk_score,
    compute_dew_point,
    compute_leaf_wetness_proxy,
    compute_vpd,
    stage_label_to_index,
)
from .constraints import get_default_constraints
from .mpc_solver import MPCSolver
from .setpoints import get_setpoint
from .state import ActuatorState, FusedState, GreenhouseState, WeatherState


# ── Initial state ─────────────────────────────────────────────────────────────


def prepare_initial_state(
    growth_stage: str,
    timestamp: _dt.datetime | None = None,
    base_temp: float = 23.0,
) -> GreenhouseState:
    """Build a physically reasonable initial ``GreenhouseState``.

    Temperature and light are adjusted for the hour-of-day so that a
    midnight start does not begin with full solar light.

    Parameters
    ----------
    growth_stage:
        Canonical growth-stage label (e.g. ``"flowering"``).
    timestamp:
        Anchor time.  ``None`` → ``datetime.now()``.
    base_temp:
        Daytime indoor baseline temperature (°C).
    """
    ts = timestamp or _dt.datetime.now()
    hour = ts.hour + ts.minute / 60.0

    # Mild diurnal indoor temperature variation (±2 °C).
    phase = 2.0 * math.pi * (hour - 6.0) / 24.0
    temp = base_temp + 2.0 * math.sin(phase)

    # Indoor light follows a daytime bell curve (0 at night).
    if 6.0 <= hour <= 20.0:
        light = 350.0 * math.sin(math.pi * (hour - 6.0) / 14.0)
    else:
        light = 0.0

    humidity = 65.0
    vpd = compute_vpd(temp, humidity)
    dew_pt = compute_dew_point(temp, humidity)
    leaf_wet = compute_leaf_wetness_proxy(humidity, temp, dew_pt)

    disease_risk = compute_disease_risk_score(
        temp=temp, humidity=humidity,
        leaf_wetness=leaf_wet, growth_stage=growth_stage,
    )

    return GreenhouseState(
        indoor_temp=round(temp, 2),
        indoor_humidity=round(humidity, 2),
        soil_moisture=65.0,
        co2=650.0,
        light_intensity=round(light, 1),
        disease_risk_score=round(disease_risk, 4),
        growth_stage_index=stage_label_to_index(growth_stage),
        vpd=round(vpd, 4),
        leaf_wetness_proxy=round(leaf_wet, 4),
        timestamp=ts,
    )


# ── Weather sequence ──────────────────────────────────────────────────────────


def prepare_weather_sequence(
    n_steps: int,
    start_time: _dt.datetime | None = None,
    dt_minutes: int = DT_MINUTES,
    base_temp: float = 20.0,
    diurnal_amp: float = 8.0,
) -> list[WeatherState]:
    """Generate a synthetic weather sequence anchored to *start_time*.

    Unlike ``generate_default_weather`` (which starts at hour 0), this
    function aligns the sinusoidal diurnal profile to the real wall-clock
    hour so the simulation begins with weather that matches the time of day.

    Parameters
    ----------
    n_steps:
        Number of 5-minute steps to generate.
    start_time:
        ``None`` → ``datetime.now()``.
    dt_minutes:
        Step duration.
    base_temp:
        Mean outdoor temperature (°C).
    diurnal_amp:
        Half-range of the daily temperature swing.
    """
    ts = start_time or _dt.datetime.now()
    start_hour_frac = ts.hour + ts.minute / 60.0

    weather: list[WeatherState] = []
    for i in range(n_steps):
        hour_frac = (start_hour_frac + i * dt_minutes / 60.0) % 24.0
        phase = 2.0 * math.pi * (hour_frac - 6.0) / 24.0  # peak ~15:00

        temp = base_temp + diurnal_amp * math.sin(phase)
        hum = 60.0 + 15.0 * math.cos(phase)  # anti-correlated
        hum = max(30.0, min(95.0, hum))

        solar = (
            max(0.0, 800.0 * math.sin(math.pi * (hour_frac - 6.0) / 14.0))
            if 6.0 <= hour_frac <= 20.0
            else 0.0
        )

        step_ts = ts + _dt.timedelta(minutes=i * dt_minutes)

        weather.append(WeatherState(
            temp_external=round(temp, 1),
            humidity_external=round(hum, 1),
            solar_radiation=round(solar, 1),
            windspeed=round(2.0 + 1.5 * abs(math.sin(phase)), 1),
            conditions="clear" if solar > 200 else "cloudy",
            timestamp=step_ts,
        ))
    return weather


# ── Growth stages ─────────────────────────────────────────────────────────────


def prepare_growth_stages(
    n_steps: int,
    stage: str,
) -> list[str]:
    """Constant growth-stage sequence for the simulation window."""
    return [stage] * n_steps


# ── MPC solver ────────────────────────────────────────────────────────────────


def build_mpc_solver(
    config: MPCConfig | None = None,
) -> MPCSolver:
    """Create an ``MPCSolver`` with sensible closed-loop defaults.

    If no config is provided, loads from YAML and overrides horizon /
    iteration settings for tractable closed-loop execution.
    """
    if config is None:
        try:
            config = load_mpc_config()
        except Exception:
            config = MPCConfig()

        # Tighter horizons for fast closed-loop steps.
        config.prediction_horizon_hours = 1
        config.control_horizon_hours = 1
        config.solver_max_iter = 300
        config.solver_ftol = 1e-5

    return MPCSolver(config)


# ── Minimal FusedState builder ────────────────────────────────────────────────


def build_fused_state(
    state: GreenhouseState,
    growth_stage: str,
    disease_risk: float,
    weather_forecast: list[dict[str, Any]] | None = None,
    timestamp: _dt.datetime | None = None,
) -> FusedState:
    """Assemble a minimal ``FusedState`` for the MPC solver.

    This is the DB-free equivalent of ``StateFusion.fuse()``.  It populates
    the fields the solver actually reads: greenhouse state, stage, risk,
    setpoint, constraints, and weather forecast.
    """
    try:
        setpoint = get_setpoint(growth_stage)
    except (KeyError, ValueError):
        setpoint = None

    constraints = get_default_constraints(growth_stage)

    return FusedState(
        greenhouse_state=state,
        growth_stage=growth_stage,
        growth_stage_index=stage_label_to_index(growth_stage),
        disease_risk_score=disease_risk,
        disease_classification=(
            "healthy leaves" if disease_risk < 0.3 else "early blight"
        ),
        disease_confidence=0.8,
        setpoint=setpoint,
        constraints=constraints,
        weather_forecast=weather_forecast or [],
        timestamp=timestamp or _dt.datetime.now(),
    )
