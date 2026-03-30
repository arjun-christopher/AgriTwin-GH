"""
Constraint sets for the MPC solver.

Four categories of constraints:
  • Environmental  — hard physical limits on indoor climate.
  • Resource       — daily water / energy budgets.
  • Crop-safety    — VPD, disease-risk ceilings (stage-dependent overrides).
  • Actuator       — box bounds, rate-of-change limits, cooldown steps.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .constants import CONTROL_VARIABLES, GROWTH_STAGES

# Type alias: variable name → (low, high)
Bounds = dict[str, tuple[float, float]]


@dataclass
class ConstraintSet:
    """Grouped constraints consumed by the MPC solver."""

    environmental: Bounds = field(default_factory=dict)
    resource: Bounds = field(default_factory=dict)
    crop_safety: Bounds = field(default_factory=dict)
    actuator_bounds: Bounds = field(default_factory=dict)
    actuator_rate_limits: Bounds = field(default_factory=dict)   # per-step Δ
    actuator_cooldown_steps: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "environmental": dict(self.environmental),
            "resource": dict(self.resource),
            "crop_safety": dict(self.crop_safety),
            "actuator_bounds": dict(self.actuator_bounds),
            "actuator_rate_limits": dict(self.actuator_rate_limits),
            "actuator_cooldown_steps": dict(self.actuator_cooldown_steps),
        }


# ── Default constraints (stage-independent base) ─────────────────────────────

_BASE_ENVIRONMENTAL: Bounds = {
    "indoor_temp": (10.0, 40.0),
    "indoor_humidity": (30.0, 95.0),
    "co2": (300.0, 2000.0),
    "soil_moisture": (20.0, 90.0),
    "light_intensity": (0.0, 1200.0),
}

_BASE_RESOURCE: Bounds = {
    "daily_water_budget_litres": (0.0, 500.0),
    "daily_energy_budget_kwh": (0.0, 100.0),
}

_BASE_CROP_SAFETY: Bounds = {
    "vpd": (0.4, 1.6),
    "disease_risk_score": (0.0, 0.8),
    "leaf_wetness_proxy": (0.0, 0.85),
}

_BASE_ACTUATOR_BOUNDS: Bounds = {
    "fan_speed": (0.0, 1.0),
    "vent_opening": (0.0, 1.0),
    "irrigation_qty": (0.0, 50.0),   # litres per step
    "heater_output": (0.0, 1.0),
    "led_intensity": (0.0, 1.0),
    "co2_valve_pct": (0.0, 1.0),
    "fogger_duty": (0.0, 1.0),
}

_BASE_ACTUATOR_RATE_LIMITS: Bounds = {
    "fan_speed": (-0.20, 0.20),
    "vent_opening": (-0.15, 0.15),
    "irrigation_qty": (-10.0, 10.0),
    "heater_output": (-0.25, 0.25),
    "led_intensity": (-0.20, 0.20),
    "co2_valve_pct": (-0.20, 0.20),
    "fogger_duty": (-0.20, 0.20),
}

_BASE_COOLDOWN_STEPS: dict[str, int] = {
    "heater_output": 3,
    "co2_valve_pct": 2,
    "irrigation_qty": 6,
}

# ── Stage-dependent overrides ─────────────────────────────────────────────────

_STAGE_CROP_SAFETY_OVERRIDES: dict[str, Bounds] = {
    "seedling": {
        "vpd": (0.4, 1.2),
        "disease_risk_score": (0.0, 0.35),
    },
    "flowering initiation": {
        "vpd": (0.5, 1.4),
        "disease_risk_score": (0.0, 0.30),
    },
    "flowering": {
        "vpd": (0.6, 1.5),
        "disease_risk_score": (0.0, 0.25),
    },
    "unripe": {
        "disease_risk_score": (0.0, 0.25),
    },
}


def get_default_constraints(stage_name: str | None = None) -> ConstraintSet:
    """Return the default ``ConstraintSet``, optionally tightened for *stage_name*.

    Parameters
    ----------
    stage_name:
        Canonical growth-stage label.  If ``None``, the base (stage-
        independent) constraints are returned.
    """
    cs = ConstraintSet(
        environmental=dict(_BASE_ENVIRONMENTAL),
        resource=dict(_BASE_RESOURCE),
        crop_safety=dict(_BASE_CROP_SAFETY),
        actuator_bounds=dict(_BASE_ACTUATOR_BOUNDS),
        actuator_rate_limits=dict(_BASE_ACTUATOR_RATE_LIMITS),
        actuator_cooldown_steps=dict(_BASE_COOLDOWN_STEPS),
    )

    if stage_name is not None:
        if stage_name not in GROWTH_STAGES:
            raise ValueError(
                f"Unknown growth stage '{stage_name}'. "
                f"Valid stages: {GROWTH_STAGES}"
            )
        overrides = _STAGE_CROP_SAFETY_OVERRIDES.get(stage_name, {})
        cs.crop_safety.update(overrides)

    return cs


def merge_constraints(base: ConstraintSet, override: ConstraintSet) -> ConstraintSet:
    """Return a new ``ConstraintSet`` where *override* values replace *base*."""
    merged = ConstraintSet(
        environmental={**base.environmental, **override.environmental},
        resource={**base.resource, **override.resource},
        crop_safety={**base.crop_safety, **override.crop_safety},
        actuator_bounds={**base.actuator_bounds, **override.actuator_bounds},
        actuator_rate_limits={**base.actuator_rate_limits, **override.actuator_rate_limits},
        actuator_cooldown_steps={**base.actuator_cooldown_steps, **override.actuator_cooldown_steps},
    )
    return merged


# ── Disease-sensitive constraint tightening ───────────────────────────────────


def tighten_constraints_for_disease(
    base: ConstraintSet,
    disease_risk: float,
    *,
    rh_tightening_threshold: float = 0.4,
    rh_tightened_ceiling: float = 80.0,
    fogger_suppress_threshold: float = 0.5,
    fogger_suppressed_max_duty: float = 0.3,
    severity_24h: dict[str, float] | None = None,
) -> ConstraintSet:
    """Return a copy of *base* with constraints tightened by disease pressure.

    When the disease risk score exceeds configurable thresholds, the
    following adjustments are applied:

    **RH ceiling lowering** — High humidity is the primary environmental
    driver of fungal diseases (powdery mildew, leaf mold, late blight).
    When ``disease_risk ≥ rh_tightening_threshold`` the indoor humidity
    upper bound is linearly reduced from its base value toward
    ``rh_tightened_ceiling``.  The reduction is proportional to
    ``(risk − threshold) / (1 − threshold)`` so that risk = 1.0 yields the
    full tightening.

    **Fogger suppression** — Foggers add moisture directly.  When
    ``disease_risk ≥ fogger_suppress_threshold`` the fogger duty-cycle
    upper bound is reduced to ``fogger_suppressed_max_duty``.

    **Severity forecast amplification** — If the 24 h disease severity
    forecast suggests a worsening trend (any disease > 50 % predicted
    severity), the thresholds are effectively lowered by 0.1 to trigger
    tightening earlier.

    Parameters
    ----------
    base : ConstraintSet
        Starting constraint set (typically from ``get_default_constraints``).
    disease_risk : float
        Composite disease risk score (0–1).
    rh_tightening_threshold : float
        Risk above which the RH ceiling begins to drop.
    rh_tightened_ceiling : float
        Lowest RH ceiling (%) that tightening can reach.
    fogger_suppress_threshold : float
        Risk above which the fogger is suppressed.
    fogger_suppressed_max_duty : float
        Max fogger duty cycle under disease pressure.
    severity_24h : dict, optional
        Per-disease 24 h severity forecast.  If any value > 50 the
        thresholds are lowered by 0.1 for earlier intervention.
    """
    cs = ConstraintSet(
        environmental=dict(base.environmental),
        resource=dict(base.resource),
        crop_safety=dict(base.crop_safety),
        actuator_bounds=dict(base.actuator_bounds),
        actuator_rate_limits=dict(base.actuator_rate_limits),
        actuator_cooldown_steps=dict(base.actuator_cooldown_steps),
    )

    # Severity forecast amplification — lower thresholds if trend is bad
    threshold_offset = 0.0
    if severity_24h:
        max_sev = max(severity_24h.values(), default=0.0)
        if max_sev > 50.0:
            threshold_offset = 0.1

    effective_rh_thresh = max(0.05, rh_tightening_threshold - threshold_offset)
    effective_fog_thresh = max(0.05, fogger_suppress_threshold - threshold_offset)

    # ── RH ceiling tightening ─────────────────────────────────────────
    if disease_risk >= effective_rh_thresh:
        base_hi = cs.environmental.get("indoor_humidity", (30.0, 95.0))[1]
        lo = cs.environmental.get("indoor_humidity", (30.0, 95.0))[0]
        alpha = min(1.0, (disease_risk - effective_rh_thresh) / (1.0 - effective_rh_thresh))
        new_hi = base_hi - alpha * (base_hi - rh_tightened_ceiling)
        cs.environmental["indoor_humidity"] = (lo, round(new_hi, 1))

    # ── Fogger suppression ────────────────────────────────────────────
    if disease_risk >= effective_fog_thresh:
        lo = cs.actuator_bounds.get("fogger_duty", (0.0, 1.0))[0]
        cs.actuator_bounds["fogger_duty"] = (lo, fogger_suppressed_max_duty)

    return cs
