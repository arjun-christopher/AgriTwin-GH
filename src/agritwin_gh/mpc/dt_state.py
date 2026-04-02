"""
Digital Twin state and payload dataclasses.

Defines the input, output, and diagnostic structures that flow through the
DT simulation layer.  All structures **reuse** the core MPC dataclasses
(``GreenhouseState``, ``ActuatorState``, ``WeatherState``) rather than
duplicating them.

Design
------
* ``DTSnapshot``    — full DT view at one timestep (state + crop + actuator).
* ``DTStepInput``   — everything needed to simulate one step forward.
* ``DTStepOutput``  — predicted next state + diagnostics + snapshot.
* ``DTDiagnostics`` — resource accounting, state deltas, safety info.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any

from .state import ActuatorState, GreenhouseState, WeatherState


# ── DT snapshot ───────────────────────────────────────────────────────────────


@dataclass
class DTSnapshot:
    """Complete Digital Twin view at a single timestep.

    Combines the indoor climate state with crop context and the actuator
    action that was applied, giving a full picture of the greenhouse at
    one moment in time.
    """

    greenhouse_state: GreenhouseState = field(default_factory=GreenhouseState)
    growth_stage: str = ""
    disease_classification: str = ""
    disease_severity: dict[str, float] = field(default_factory=dict)
    actuators_applied: ActuatorState | None = None
    weather: WeatherState | None = None
    timestamp: _dt.datetime | None = None
    step_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict representation."""
        return {
            "greenhouse_state": self.greenhouse_state.to_dict(),
            "growth_stage": self.growth_stage,
            "disease_classification": self.disease_classification,
            "disease_severity": dict(self.disease_severity),
            "actuators_applied": (
                self.actuators_applied.to_dict()
                if self.actuators_applied is not None
                else None
            ),
            "weather": (
                self.weather.to_dict() if self.weather is not None else None
            ),
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "step_index": self.step_index,
            "metadata": dict(self.metadata),
        }


# ── DT step input ────────────────────────────────────────────────────────────


@dataclass
class DTStepInput:
    """Everything the Digital Twin needs to simulate one step forward.

    Parameters
    ----------
    current_state:
        Indoor-climate state at the start of this step.
    action:
        Actuator commands to apply during this step.
    weather:
        External weather conditions during this step.
    growth_stage:
        Current growth stage label (canonical form from ``GROWTH_STAGES``).
    disease_risk_score:
        Carried-forward disease risk score [0, 1].
    disease_classification:
        Current disease classification label (canonical).
    disease_severity:
        Per-disease severity dict, if available.
    dt_minutes:
        Step duration in minutes (default 5, matching ``DT_MINUTES``).
    step_index:
        Ordinal position within the simulation run.
    timestamp:
        Logical timestamp for this step.
    """

    current_state: GreenhouseState = field(default_factory=GreenhouseState)
    action: ActuatorState = field(default_factory=ActuatorState)
    weather: WeatherState = field(default_factory=WeatherState)
    growth_stage: str = ""
    disease_risk_score: float = 0.0
    disease_classification: str = ""
    disease_severity: dict[str, float] = field(default_factory=dict)
    dt_minutes: int = 5
    step_index: int = 0
    timestamp: _dt.datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict representation."""
        return {
            "current_state": self.current_state.to_dict(),
            "action": self.action.to_dict(),
            "weather": self.weather.to_dict(),
            "growth_stage": self.growth_stage,
            "disease_risk_score": self.disease_risk_score,
            "disease_classification": self.disease_classification,
            "disease_severity": dict(self.disease_severity),
            "dt_minutes": self.dt_minutes,
            "step_index": self.step_index,
            "timestamp": str(self.timestamp) if self.timestamp else None,
        }


# ── DT diagnostics ───────────────────────────────────────────────────────────


@dataclass
class DTDiagnostics:
    """Per-step diagnostics produced alongside the DT state transition.

    Provides resource accounting, state-change tracking, and safety
    information for logging and evaluation.
    """

    # Resource accounting
    energy_kwh: float = 0.0
    water_litres: float = 0.0

    # Disease risk recomputed from the new environment state
    disease_risk_recomputed: float = 0.0

    # State delta: {variable_name: new − old}
    state_delta: dict[str, float] = field(default_factory=dict)

    # Variables that hit physical bounds after clamping
    bounds_clamped: list[str] = field(default_factory=list)

    # Tracking: signed error (actual − setpoint) for key variables
    setpoint_error: dict[str, float] = field(default_factory=dict)

    # Per-actuator / disturbance contribution breakdown for each state var.
    # Outer key = state variable, inner key = effect source (actuator or
    # disturbance name), value = estimated linear contribution to the delta.
    effect_attribution: dict[str, dict[str, float]] = field(
        default_factory=dict,
    )

    # Boolean flags for disease-favorable environmental conditions.
    # Keys: "high_humidity_risk", "high_leaf_wetness", "disease_temp_band",
    #        "fogger_disease_concern".
    disease_environment_flags: dict[str, bool] = field(default_factory=dict)

    # Step timing
    step_compute_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict representation."""
        return {
            "energy_kwh": self.energy_kwh,
            "water_litres": self.water_litres,
            "disease_risk_recomputed": self.disease_risk_recomputed,
            "state_delta": dict(self.state_delta),
            "bounds_clamped": list(self.bounds_clamped),
            "setpoint_error": dict(self.setpoint_error),
            "effect_attribution": {
                k: dict(v) for k, v in self.effect_attribution.items()
            },
            "disease_environment_flags": dict(self.disease_environment_flags),
            "step_compute_ms": self.step_compute_ms,
        }


# ── DT step output ───────────────────────────────────────────────────────────


@dataclass
class DTStepOutput:
    """Result of a single Digital Twin simulation step.

    Contains the predicted next state, diagnostic information, and a
    complete snapshot of the DT after the step.
    """

    next_state: GreenhouseState = field(default_factory=GreenhouseState)
    diagnostics: DTDiagnostics = field(default_factory=DTDiagnostics)
    snapshot: DTSnapshot = field(default_factory=DTSnapshot)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict representation."""
        return {
            "next_state": self.next_state.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "snapshot": self.snapshot.to_dict(),
        }
