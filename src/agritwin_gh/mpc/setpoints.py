"""
Stage-specific target setpoints and control profiles.

Each canonical growth stage maps to:
  • ``StageSetpoint``       — ideal indoor-climate targets.
  • ``StageControlProfile`` — per-variable control weights, disease-
    sensitivity modifier, and resource-priority hints.

The two are kept separate so that the cost function receives weights from
the profile while the baseline controller uses the setpoint directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import GROWTH_STAGES


# ── Setpoints ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageSetpoint:
    """Target climate profile for a single growth stage."""

    temp: float               # °C
    temp_tol: float           # ± band
    humidity: float           # %RH
    hum_tol: float            # ± band
    soil_moisture: float      # %
    co2: float                # ppm
    light: float              # W/m² (or µmol — units consistent with sensor)
    vpd: float                # kPa
    disease_risk_max: float   # soft ceiling (0-1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Control profiles ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StageControlProfile:
    """Per-stage control weights, disease sensitivity, and resource priority.

    *control_weights* maps state-variable name → relative importance
    multiplier applied to base cost weights during this growth stage.
    Values >1 penalise deviations more; <1 relax them.

    *disease_sensitivity* is a scalar multiplier (typically 0.8–1.5) that
    amplifies the disease-risk cost term during stages vulnerable to
    pathogen pressure.

    *resource_priority* maps ``"water"`` / ``"energy"`` → priority float
    (higher = conserve more).  Used by the baseline controller and cost
    function to bias actuator selection.

    *night_temp_offset* lowers the target temperature during night-time
    steps (negative °C offset, e.g. -2.0).
    """

    control_weights: dict[str, float] = field(default_factory=dict)
    disease_sensitivity: float = 1.0
    resource_priority: dict[str, float] = field(default_factory=dict)
    night_temp_offset: float = -2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ── Default setpoint table (values match Notebook-03 / Notebook-04) ──────────

_DEFAULT_SETPOINTS: dict[str, StageSetpoint] = {
    "seedling": StageSetpoint(
        temp=23.0, temp_tol=2.0,
        humidity=75.0, hum_tol=5.0,
        soil_moisture=70.0,
        co2=600.0,
        light=250.0,
        vpd=0.6,
        disease_risk_max=0.35,
    ),
    "early vegetative": StageSetpoint(
        temp=24.0, temp_tol=2.0,
        humidity=70.0, hum_tol=5.0,
        soil_moisture=65.0,
        co2=700.0,
        light=400.0,
        vpd=0.8,
        disease_risk_max=0.35,
    ),
    "flowering initiation": StageSetpoint(
        temp=22.0, temp_tol=1.5,
        humidity=65.0, hum_tol=5.0,
        soil_moisture=60.0,
        co2=800.0,
        light=500.0,
        vpd=0.9,
        disease_risk_max=0.30,
    ),
    "flowering": StageSetpoint(
        temp=21.0, temp_tol=1.5,
        humidity=60.0, hum_tol=5.0,
        soil_moisture=60.0,
        co2=900.0,
        light=550.0,
        vpd=1.0,
        disease_risk_max=0.25,
    ),
    "unripe": StageSetpoint(
        temp=22.0, temp_tol=2.0,
        humidity=65.0, hum_tol=5.0,
        soil_moisture=65.0,
        co2=800.0,
        light=450.0,
        vpd=0.8,
        disease_risk_max=0.25,
    ),
    "ripe": StageSetpoint(
        temp=20.0, temp_tol=2.5,
        humidity=60.0, hum_tol=5.0,
        soil_moisture=55.0,
        co2=600.0,
        light=350.0,
        vpd=0.7,
        disease_risk_max=0.40,
    ),
}


# ── Default control-profile table ─────────────────────────────────────────────

_DEFAULT_PROFILES: dict[str, StageControlProfile] = {
    "seedling": StageControlProfile(
        control_weights={
            "temp": 1.2, "humidity": 1.0, "soil_moisture": 1.3,
            "co2": 0.6, "vpd": 0.8, "light": 0.7, "disease_risk": 1.0,
        },
        disease_sensitivity=1.0,
        resource_priority={"water": 1.2, "energy": 0.8},
        night_temp_offset=-1.5,
    ),
    "early vegetative": StageControlProfile(
        control_weights={
            "temp": 1.0, "humidity": 1.0, "soil_moisture": 1.0,
            "co2": 0.8, "vpd": 0.9, "light": 1.0, "disease_risk": 1.0,
        },
        disease_sensitivity=1.0,
        resource_priority={"water": 1.0, "energy": 1.0},
        night_temp_offset=-2.0,
    ),
    "flowering initiation": StageControlProfile(
        control_weights={
            "temp": 1.3, "humidity": 1.2, "soil_moisture": 0.9,
            "co2": 1.0, "vpd": 1.2, "light": 1.1, "disease_risk": 1.3,
        },
        disease_sensitivity=1.3,
        resource_priority={"water": 0.9, "energy": 1.1},
        night_temp_offset=-2.0,
    ),
    "flowering": StageControlProfile(
        control_weights={
            "temp": 1.4, "humidity": 1.3, "soil_moisture": 1.0,
            "co2": 1.2, "vpd": 1.3, "light": 1.2, "disease_risk": 1.5,
        },
        disease_sensitivity=1.5,
        resource_priority={"water": 1.0, "energy": 1.2},
        night_temp_offset=-2.5,
    ),
    "unripe": StageControlProfile(
        control_weights={
            "temp": 1.1, "humidity": 1.2, "soil_moisture": 1.1,
            "co2": 1.0, "vpd": 1.1, "light": 1.0, "disease_risk": 1.4,
        },
        disease_sensitivity=1.4,
        resource_priority={"water": 1.1, "energy": 1.0},
        night_temp_offset=-2.0,
    ),
    "ripe": StageControlProfile(
        control_weights={
            "temp": 0.9, "humidity": 0.8, "soil_moisture": 0.8,
            "co2": 0.5, "vpd": 0.7, "light": 0.6, "disease_risk": 0.8,
        },
        disease_sensitivity=0.8,
        resource_priority={"water": 0.7, "energy": 0.7},
        night_temp_offset=-3.0,
    ),
}


# ── Public API ────────────────────────────────────────────────────────────────


def _validate_stage(stage_name: str) -> None:
    if stage_name not in GROWTH_STAGES:
        raise ValueError(
            f"Unknown growth stage '{stage_name}'. "
            f"Valid stages: {GROWTH_STAGES}"
        )


def get_setpoint(stage_name: str) -> StageSetpoint:
    """Return the setpoint for *stage_name* (canonical label).

    Raises ``ValueError`` if the stage is unknown.
    """
    sp = _DEFAULT_SETPOINTS.get(stage_name)
    if sp is None:
        _validate_stage(stage_name)
    return sp  # type: ignore[return-value]


def get_control_profile(stage_name: str) -> StageControlProfile:
    """Return the ``StageControlProfile`` for *stage_name*.

    Raises ``ValueError`` if the stage is unknown.
    """
    profile = _DEFAULT_PROFILES.get(stage_name)
    if profile is None:
        _validate_stage(stage_name)
    return profile  # type: ignore[return-value]


def get_all_setpoints() -> dict[str, StageSetpoint]:
    """Return a copy of the full setpoints table."""
    return dict(_DEFAULT_SETPOINTS)


def get_all_control_profiles() -> dict[str, StageControlProfile]:
    """Return a copy of the full control-profile table."""
    return dict(_DEFAULT_PROFILES)


def override_setpoints(overrides: dict[str, dict[str, float]]) -> None:
    """Merge runtime overrides (e.g. from YAML config) into the default table.

    *overrides* maps stage name → field → value.  Only fields present in the
    override dict are changed; all others keep their defaults.
    """
    for stage, fields in overrides.items():
        if stage not in _DEFAULT_SETPOINTS:
            raise ValueError(
                f"Cannot override unknown stage '{stage}'. "
                f"Valid stages: {GROWTH_STAGES}"
            )
        base = asdict(_DEFAULT_SETPOINTS[stage])
        base.update(fields)
        _DEFAULT_SETPOINTS[stage] = StageSetpoint(**base)
