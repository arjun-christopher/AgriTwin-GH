"""
Evaluation metric calculators for MPC controller comparison.

Provides granular, per-variable and aggregate metrics for comparing
greenhouse controller strategies.  Every calculator works on aligned
sequences of ``GreenhouseState`` / ``ActuatorState`` objects.

Metrics are grouped into:

* **Tracking error** — RMSE, MAE, max-absolute-error vs setpoint
* **Disease burden** — mean/max risk, RH exposure, disease-favorable time
* **Resource consumption** — total water, total energy, efficiency ratios
* **Control quality** — switching frequency, smoothness (L2 of Δu)
* **Safety** — violation counts (time outside crop-safety bounds)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .constants import CONTROL_VARIABLES, STATE_VARIABLES
from .setpoints import StageSetpoint, get_setpoint
from .state import ActuatorState, GreenhouseState


# ── Tracking error metrics ────────────────────────────────────────────────────


@dataclass
class TrackingMetrics:
    """Per-variable setpoint tracking error statistics."""

    variable: str = ""
    rmse: float = 0.0
    mae: float = 0.0
    max_abs_error: float = 0.0
    mean_signed_error: float = 0.0
    within_tolerance_pct: float = 0.0   # % of steps within setpoint tolerance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_tracking_metrics(
    states: Sequence[GreenhouseState],
    growth_stages: Sequence[str],
    variable: str,
) -> TrackingMetrics:
    """Compute tracking error of *variable* against stage-specific setpoints.

    Parameters
    ----------
    states : sequence of GreenhouseState
        Observed or simulated state trajectory.
    growth_stages : sequence of str
        Canonical growth stage label per step (same length as *states*).
    variable : str
        One of ``"indoor_temp"``, ``"indoor_humidity"``, ``"soil_moisture"``,
        ``"co2"``, ``"vpd"``.
    """
    if len(states) != len(growth_stages):
        raise ValueError("states and growth_stages must have the same length")

    _sp_field, _tol_field = _VARIABLE_SP_MAP.get(variable, (variable, None))
    errors: list[float] = []
    within = 0

    for gs, stage in zip(states, growth_stages):
        sp = get_setpoint(stage)
        target = getattr(sp, _sp_field, 0.0)
        actual = getattr(gs, variable, 0.0)
        err = actual - target
        errors.append(err)
        tol = getattr(sp, _tol_field, 0.0) if _tol_field else 0.0
        if abs(err) <= tol:
            within += 1

    errors_np = np.array(errors)
    n = len(errors) or 1
    return TrackingMetrics(
        variable=variable,
        rmse=float(np.sqrt(np.mean(errors_np ** 2))),
        mae=float(np.mean(np.abs(errors_np))),
        max_abs_error=float(np.max(np.abs(errors_np))),
        mean_signed_error=float(np.mean(errors_np)),
        within_tolerance_pct=round(100.0 * within / n, 2),
    )


# Mapping: variable_name → (setpoint_field, tolerance_field)
_VARIABLE_SP_MAP: dict[str, tuple[str, str | None]] = {
    "indoor_temp": ("temp", "temp_tol"),
    "indoor_humidity": ("humidity", "hum_tol"),
    "soil_moisture": ("soil_moisture", None),
    "co2": ("co2", None),
    "vpd": ("vpd", None),
}


# ── Disease burden metrics ────────────────────────────────────────────────────


@dataclass
class DiseaseBurdenMetrics:
    """Disease-related metrics over a simulation window."""

    mean_risk: float = 0.0
    max_risk: float = 0.0
    cumulative_rh_exposure: float = 0.0      # Σ max(0, RH − safe_ceiling) * dt
    disease_favorable_steps: int = 0          # steps where risk > threshold
    disease_favorable_duration_hours: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_disease_burden(
    states: Sequence[GreenhouseState],
    growth_stages: Sequence[str],
    dt_minutes: int = 5,
    risk_threshold: float = 0.35,
) -> DiseaseBurdenMetrics:
    """Compute disease burden metrics from state trajectory.

    Parameters
    ----------
    risk_threshold : float
        Disease risk above which a step is counted as "disease-favorable".
    """
    risks: list[float] = []
    rh_exposure = 0.0
    fav_steps = 0

    for gs, stage in zip(states, growth_stages):
        sp = get_setpoint(stage)
        risks.append(gs.disease_risk_score)
        rh_excess = max(0.0, gs.indoor_humidity - (sp.humidity + sp.hum_tol))
        rh_exposure += rh_excess * (dt_minutes / 60.0)
        if gs.disease_risk_score > risk_threshold:
            fav_steps += 1

    risks_np = np.array(risks) if risks else np.zeros(1)
    return DiseaseBurdenMetrics(
        mean_risk=float(np.mean(risks_np)),
        max_risk=float(np.max(risks_np)),
        cumulative_rh_exposure=round(rh_exposure, 2),
        disease_favorable_steps=fav_steps,
        disease_favorable_duration_hours=round(fav_steps * dt_minutes / 60.0, 2),
    )


# ── Resource consumption metrics ──────────────────────────────────────────────


@dataclass
class ResourceMetrics:
    """Water and energy consumption statistics."""

    total_water_litres: float = 0.0
    total_energy_kwh: float = 0.0
    water_per_hour: float = 0.0
    energy_per_hour: float = 0.0
    water_efficiency: float = 0.0    # 1 / total_water (or inf if 0)
    energy_efficiency: float = 0.0   # 1 / total_energy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_resource_metrics(
    actuators: Sequence[ActuatorState],
    dt_minutes: int = 5,
    energy_costs: dict[str, float] | None = None,
) -> ResourceMetrics:
    """Compute resource consumption from actuator trajectory.

    Parameters
    ----------
    energy_costs : dict, optional
        Per-actuator energy cost in kWh per step at full output.
        Defaults to conservative estimates.
    """
    _costs = energy_costs or {
        "fan_speed": 0.15, "vent_opening": 0.02, "heater_output": 0.80,
        "led_intensity": 0.30, "co2_valve_pct": 0.05, "fogger_duty": 0.10,
        "irrigation_qty": 0.01,
    }
    total_water = 0.0
    total_energy = 0.0
    for act in actuators:
        total_water += act.irrigation_qty + act.fogger_duty * 2.0
        energy = sum(
            getattr(act, name) * _costs.get(name, 0.0)
            for name in CONTROL_VARIABLES
        )
        total_energy += energy

    n_steps = len(actuators) or 1
    total_hours = n_steps * dt_minutes / 60.0

    return ResourceMetrics(
        total_water_litres=round(total_water, 2),
        total_energy_kwh=round(total_energy, 4),
        water_per_hour=round(total_water / max(total_hours, 1e-6), 2),
        energy_per_hour=round(total_energy / max(total_hours, 1e-6), 4),
        water_efficiency=round(1.0 / max(total_water, 1e-6), 4),
        energy_efficiency=round(1.0 / max(total_energy, 1e-6), 4),
    )


# ── Control quality metrics ───────────────────────────────────────────────────


@dataclass
class ControlQualityMetrics:
    """Actuator switching frequency and control smoothness."""

    switching_frequency: dict[str, float] = field(default_factory=dict)
    total_switches: dict[str, int] = field(default_factory=dict)
    smoothness_l2: dict[str, float] = field(default_factory=dict)
    mean_smoothness_l2: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_control_quality(
    actuators: Sequence[ActuatorState],
    switch_threshold: float = 0.05,
) -> ControlQualityMetrics:
    """Compute switching frequency and smoothness from actuator trajectory.

    A "switch" is defined as a change exceeding *switch_threshold* for a
    given actuator channel.
    """
    if len(actuators) < 2:
        return ControlQualityMetrics()

    n = len(actuators)
    switches: dict[str, int] = {name: 0 for name in CONTROL_VARIABLES}
    l2_accum: dict[str, float] = {name: 0.0 for name in CONTROL_VARIABLES}

    for i in range(1, n):
        for name in CONTROL_VARIABLES:
            prev = getattr(actuators[i - 1], name)
            curr = getattr(actuators[i], name)
            delta = abs(curr - prev)
            if delta > switch_threshold:
                switches[name] += 1
            l2_accum[name] += delta ** 2

    freq = {name: round(cnt / (n - 1), 4) for name, cnt in switches.items()}
    smoothness = {name: round(math.sqrt(v / (n - 1)), 4) for name, v in l2_accum.items()}
    mean_l2 = sum(smoothness.values()) / max(len(smoothness), 1)

    return ControlQualityMetrics(
        switching_frequency=freq,
        total_switches=switches,
        smoothness_l2=smoothness,
        mean_smoothness_l2=round(mean_l2, 4),
    )


# ── Safety violation metrics ──────────────────────────────────────────────────


@dataclass
class SafetyMetrics:
    """Count and duration of safety-critical bound violations."""

    violation_count: dict[str, int] = field(default_factory=dict)
    violation_duration_hours: dict[str, float] = field(default_factory=dict)
    total_violations: int = 0
    violation_rate_pct: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Default safety bounds (conservative, stage-independent)
_SAFETY_BOUNDS: dict[str, tuple[float, float]] = {
    "indoor_temp": (10.0, 38.0),
    "indoor_humidity": (25.0, 95.0),
    "vpd": (0.3, 2.0),
    "co2": (250.0, 2500.0),
    "soil_moisture": (15.0, 95.0),
}


def compute_safety_metrics(
    states: Sequence[GreenhouseState],
    dt_minutes: int = 5,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> SafetyMetrics:
    """Count safety-critical bound violations across the trajectory."""
    _bounds = bounds or _SAFETY_BOUNDS
    counts: dict[str, int] = {k: 0 for k in _bounds}
    n = len(states)

    for gs in states:
        for var, (lo, hi) in _bounds.items():
            val = getattr(gs, var, 0.0)
            if val < lo or val > hi:
                counts[var] += 1

    total = sum(counts.values())
    duration = {
        k: round(cnt * dt_minutes / 60.0, 2) for k, cnt in counts.items()
    }
    n_checks = n * len(_bounds) or 1

    return SafetyMetrics(
        violation_count=counts,
        violation_duration_hours=duration,
        total_violations=total,
        violation_rate_pct=round(100.0 * total / n_checks, 2),
    )


# ── Aggregate all metrics ────────────────────────────────────────────────────


@dataclass
class ControllerMetricsBundle:
    """Complete metrics bundle for one controller run."""

    controller_id: str = ""
    controller_type: str = ""   # "baseline", "mpc", "mpc_disease_aware", etc.
    n_steps: int = 0

    tracking: dict[str, TrackingMetrics] = field(default_factory=dict)
    disease_burden: DiseaseBurdenMetrics = field(default_factory=DiseaseBurdenMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    control_quality: ControlQualityMetrics = field(default_factory=ControlQualityMetrics)
    safety: SafetyMetrics = field(default_factory=SafetyMetrics)

    # Populated by yield_proxy module
    yield_quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "controller_type": self.controller_type,
            "n_steps": self.n_steps,
            "tracking": {k: v.to_dict() for k, v in self.tracking.items()},
            "disease_burden": self.disease_burden.to_dict(),
            "resources": self.resources.to_dict(),
            "control_quality": self.control_quality.to_dict(),
            "safety": self.safety.to_dict(),
            "yield_quality_score": self.yield_quality_score,
        }


def compute_all_metrics(
    states: Sequence[GreenhouseState],
    actuators: Sequence[ActuatorState],
    growth_stages: Sequence[str],
    controller_id: str = "",
    controller_type: str = "",
    dt_minutes: int = 5,
) -> ControllerMetricsBundle:
    """Compute the full metrics suite for one controller trajectory.

    This is the primary entry point for evaluation — it runs every
    calculator and bundles the results.
    """
    tracked_vars = ["indoor_temp", "indoor_humidity", "soil_moisture", "co2", "vpd"]
    tracking = {}
    for var in tracked_vars:
        tracking[var] = compute_tracking_metrics(states, growth_stages, var)

    disease = compute_disease_burden(states, growth_stages, dt_minutes=dt_minutes)
    resources = compute_resource_metrics(actuators, dt_minutes=dt_minutes)
    control_q = compute_control_quality(actuators)
    safety = compute_safety_metrics(states, dt_minutes=dt_minutes)

    return ControllerMetricsBundle(
        controller_id=controller_id,
        controller_type=controller_type,
        n_steps=len(states),
        tracking=tracking,
        disease_burden=disease,
        resources=resources,
        control_quality=control_q,
        safety=safety,
    )
