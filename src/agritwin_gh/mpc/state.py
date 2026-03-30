"""
Dataclasses representing the greenhouse state vector, actuator vector,
and all intermediate / output structures used throughout the MPC package.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .constants import (
    CONTROL_VARIABLES,
    STATE_VARIABLES,
    stage_index_to_label,
    stage_label_to_index,
)

if TYPE_CHECKING:
    from .constraints import ConstraintSet
    from .setpoints import StageSetpoint

# ── Core state dataclasses ────────────────────────────────────────────────────


@dataclass
class GreenhouseState:
    """Indoor-climate state vector consumed by the MPC solver.

    Field order mirrors ``constants.STATE_VARIABLES``.
    """

    indoor_temp: float = 0.0
    indoor_humidity: float = 0.0
    soil_moisture: float = 0.0
    co2: float = 0.0
    light_intensity: float = 0.0
    disease_risk_score: float = 0.0
    growth_stage_index: int = 0
    vpd: float = 0.0
    leaf_wetness_proxy: float = 0.0
    timestamp: _dt.datetime | None = None

    # ── conversion helpers ─────────────────────────────────────────────

    def to_numpy(self) -> np.ndarray:
        """Return a 1-D array matching ``STATE_VARIABLES`` order."""
        return np.array(
            [
                self.indoor_temp,
                self.indoor_humidity,
                self.soil_moisture,
                self.co2,
                self.light_intensity,
                self.disease_risk_score,
                float(self.growth_stage_index),
                self.vpd,
                self.leaf_wetness_proxy,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray, timestamp: _dt.datetime | None = None) -> GreenhouseState:
        """Construct from a 1-D array (``STATE_VARIABLES`` order)."""
        if arr.shape != (len(STATE_VARIABLES),):
            raise ValueError(
                f"Expected array of length {len(STATE_VARIABLES)}, got {arr.shape}"
            )
        return cls(
            indoor_temp=float(arr[0]),
            indoor_humidity=float(arr[1]),
            soil_moisture=float(arr[2]),
            co2=float(arr[3]),
            light_intensity=float(arr[4]),
            disease_risk_score=float(arr[5]),
            growth_stage_index=int(round(arr[6])),
            vpd=float(arr[7]),
            leaf_wetness_proxy=float(arr[8]),
            timestamp=timestamp,
        )

    @classmethod
    def from_db_row(cls, row: dict) -> GreenhouseState:
        """Build from a ``greenhouse_data`` DB row dict.

        The DB schema has no ``soil_moisture`` or ``light_intensity`` columns
        — those are initialised to defaults and must be supplied externally.
        """
        return cls(
            indoor_temp=float(row.get("indoor_temp", 0.0)),
            indoor_humidity=float(row.get("indoor_humidity", 0.0)),
            soil_moisture=float(row.get("soil_moisture", 60.0)),
            co2=float(row.get("indoor_co2", 0.0)),
            light_intensity=float(row.get("solarradiation", 0.0)),
            disease_risk_score=0.0,
            growth_stage_index=0,
            vpd=float(row.get("vpd", 0.0)),
            leaf_wetness_proxy=float(row.get("leaf_wetness_proxy", 0.0)),
            timestamp=row.get("datetime"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def growth_stage_label(self) -> str:
        return stage_index_to_label(self.growth_stage_index)


@dataclass
class ActuatorState:
    """Control-action vector produced by the MPC solver or baseline controller.

    Field order mirrors ``constants.CONTROL_VARIABLES``.
    """

    fan_speed: float = 0.0
    vent_opening: float = 0.0
    irrigation_qty: float = 0.0
    heater_output: float = 0.0
    led_intensity: float = 0.0
    co2_valve_pct: float = 0.0
    fogger_duty: float = 0.0

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.fan_speed,
                self.vent_opening,
                self.irrigation_qty,
                self.heater_output,
                self.led_intensity,
                self.co2_valve_pct,
                self.fogger_duty,
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_numpy(cls, arr: np.ndarray) -> ActuatorState:
        if arr.shape != (len(CONTROL_VARIABLES),):
            raise ValueError(
                f"Expected array of length {len(CONTROL_VARIABLES)}, got {arr.shape}"
            )
        return cls(
            fan_speed=float(arr[0]),
            vent_opening=float(arr[1]),
            irrigation_qty=float(arr[2]),
            heater_output=float(arr[3]),
            led_intensity=float(arr[4]),
            co2_valve_pct=float(arr[5]),
            fogger_duty=float(arr[6]),
        )

    def clip(self, bounds: dict[str, tuple[float, float]]) -> ActuatorState:
        """Return a new ``ActuatorState`` clipped to *bounds*.

        *bounds* maps actuator names to ``(low, high)`` tuples.
        """
        vals: dict[str, float] = {}
        for name in CONTROL_VARIABLES:
            v = getattr(self, name)
            lo, hi = bounds.get(name, (-float("inf"), float("inf")))
            vals[name] = max(lo, min(hi, v))
        return ActuatorState(**vals)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MPCState:
    """Composite snapshot: greenhouse state + actuator state + metadata."""

    greenhouse: GreenhouseState
    actuators: ActuatorState
    timestamp: _dt.datetime | None = None
    step_index: int = 0
    run_id: str = ""


# ── Weather ───────────────────────────────────────────────────────────────────


@dataclass
class WeatherState:
    """Single-timestep external weather observation / forecast."""

    temp_external: float = 0.0
    humidity_external: float = 0.0
    solar_radiation: float = 0.0
    windspeed: float = 0.0
    conditions: str = "unknown"
    timestamp: _dt.datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Classification / progression outputs ──────────────────────────────────────


@dataclass
class GrowthClassificationOutput:
    """Output of the growth-stage image classifier."""

    class_name: str = ""
    confidence: float = 0.0
    top_k: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DiseaseClassificationOutput:
    """Output of the disease image classifier."""

    class_name: str = ""
    confidence: float = 0.0
    top_k: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DiseaseProgressionOutput:
    """Summarised disease-progression forecast."""

    current_severity: dict[str, float] = field(default_factory=dict)
    severity_24h: dict[str, float] = field(default_factory=dict)
    severity_48h: dict[str, float] = field(default_factory=dict)


@dataclass
class GrowthProgressionOutput:
    """Summarised growth-stage transition forecast."""

    current_stage: str = ""
    next_stage: str = ""
    hours_to_transition: float = 0.0
    transition_within_24h: bool = False
    transition_within_48h: bool = False


# ── Fused state (MPC input) ──────────────────────────────────────────────────


@dataclass
class FusedState:
    """Comprehensive state assembled by ``StateFusion`` — the single input to the MPC solver."""

    # Current indoor climate
    greenhouse_state: GreenhouseState = field(default_factory=GreenhouseState)

    # Weather forecast over prediction horizon (one dict per horizon step)
    weather_forecast: list[dict[str, Any]] = field(default_factory=list)

    # Growth stage classification
    growth_stage: str = ""
    growth_stage_index: int = 0
    next_stage: str = ""
    hours_to_transition: float = 0.0
    transition_within_24h: bool = False
    transition_within_48h: bool = False

    # Disease classification (from latest image)
    disease_classification: str = ""
    disease_confidence: float = 0.0

    # Current disease severity
    current_severity: dict[str, float] = field(default_factory=dict)

    # Disease progression forecasts
    severity_24h: dict[str, float] = field(default_factory=dict)
    severity_48h: dict[str, float] = field(default_factory=dict)

    # Derived targets (populated by fusion pipeline)
    setpoint: StageSetpoint | None = None
    disease_risk_score: float = 0.0

    # Constraints and metadata
    constraints: ConstraintSet | None = None
    timestamp: _dt.datetime | None = None
    cycle_id: int = 0


# ── Controller decision context ───────────────────────────────────────────────


@dataclass
class ControllerDecisionContext:
    """Versionable snapshot of everything that informed one MPC decision.

    Designed to be serialised alongside ``DigitalTwinStepPayload`` so that
    every control action is fully traceable.
    """

    run_id: str = ""
    timestamp: _dt.datetime | None = None
    step_index: int = 0

    # Solver meta
    solver_config: dict[str, Any] = field(default_factory=dict)
    cost_weights: dict[str, float] = field(default_factory=dict)

    # Context summaries (dicts for JSON serialisability)
    weather_stress_summary: dict[str, float] = field(default_factory=dict)
    disease_context_summary: dict[str, Any] = field(default_factory=dict)
    constraint_tightening: dict[str, Any] = field(default_factory=dict)

    # Solver performance
    solver_performance: dict[str, Any] = field(default_factory=dict)

    # Provenance — model run IDs used by this decision
    model_ids: dict[str, str] = field(default_factory=dict)

    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.timestamp is not None:
            d["timestamp"] = str(self.timestamp)
        return d


# ── Digital-twin output payloads ──────────────────────────────────────────────


@dataclass
class DigitalTwinStepPayload:
    """Single MPC control-step output sent to the digital-twin / dashboard."""

    timestamp: _dt.datetime | None = None
    run_id: str = ""
    step_index: int = 0

    # Current observations
    observed_state: dict[str, Any] = field(default_factory=dict)
    growth_stage: str = ""
    disease_classification: str = ""
    disease_risk_score: float = 0.0

    # MPC decision
    applied_actuators: dict[str, Any] = field(default_factory=dict)
    predicted_next_state: dict[str, Any] = field(default_factory=dict)

    # Forecast summaries
    weather_forecast_24h: dict[str, Any] = field(default_factory=dict)
    severity_forecast_24h: dict[str, float] = field(default_factory=dict)
    severity_forecast_48h: dict[str, float] = field(default_factory=dict)
    hours_to_stage_transition: float = 0.0

    # Cost & resource tracking
    step_cost: float = 0.0
    cumulative_energy_kwh: float = 0.0
    cumulative_water_litres: float = 0.0

    # Image keys (for visualisation, resolved by dashboard via MinIO)
    disease_image_key: str | None = None
    growth_stage_image_key: str | None = None

    # Alert
    alert_level: str = "GREEN"
    alert_icons: list[str] = field(default_factory=list)

    # Explanation & decision context (Subprompt 7)
    explanation: dict[str, Any] = field(default_factory=dict)
    decision_context: dict[str, Any] = field(default_factory=dict)
    solver_performance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DigitalTwinTrajectoryPayload:
    """Multi-step trajectory payload for comparison visualisation."""

    run_id: str = ""
    steps: list[DigitalTwinStepPayload] = field(default_factory=list)
    total_cost: float = 0.0
    total_energy_kwh: float = 0.0
    total_water_litres: float = 0.0


# ── Evaluation ────────────────────────────────────────────────────────────────


@dataclass
class ComparisonMetrics:
    """Aggregate comparison between baseline controller and MPC."""

    # Tracking error (RMSE)
    temp_rmse_baseline: float = 0.0
    temp_rmse_mpc: float = 0.0
    humidity_rmse_baseline: float = 0.0
    humidity_rmse_mpc: float = 0.0
    co2_rmse_baseline: float = 0.0
    co2_rmse_mpc: float = 0.0
    vpd_rmse_baseline: float = 0.0
    vpd_rmse_mpc: float = 0.0
    soil_moisture_rmse_baseline: float = 0.0
    soil_moisture_rmse_mpc: float = 0.0

    # Disease risk
    disease_risk_mean_baseline: float = 0.0
    disease_risk_mean_mpc: float = 0.0
    disease_risk_max_baseline: float = 0.0
    disease_risk_max_mpc: float = 0.0

    # Resource consumption
    water_total_baseline: float = 0.0
    water_total_mpc: float = 0.0
    energy_total_baseline: float = 0.0
    energy_total_mpc: float = 0.0

    # Actuator switching frequency per actuator
    switching_freq_baseline: dict[str, float] = field(default_factory=dict)
    switching_freq_mpc: dict[str, float] = field(default_factory=dict)

    # Growth quality (fraction of steps within crop-safety bounds)
    growth_quality_baseline: float = 0.0
    growth_quality_mpc: float = 0.0

    # Solver performance
    mean_solve_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationSummary:
    """Wrapper around comparison metrics with run metadata."""

    run_id: str = ""
    simulation_start: _dt.datetime | None = None
    simulation_end: _dt.datetime | None = None
    total_steps: int = 0
    metrics: ComparisonMetrics = field(default_factory=ComparisonMetrics)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Image payload ─────────────────────────────────────────────────────────────


@dataclass
class ImagePayload:
    """Metadata returned by ``ImageStreamer`` — no binary data."""

    image_key: str = ""
    bucket_name: str = ""
    file_name: str = ""
    label: str = ""
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
