"""
MPC-specific configuration dataclass and YAML loader.

Convention follows ``config/minio_config.py`` (dataclass with ``__post_init__``
hook) and ``utils/database.py`` (``Path``-based config discovery).
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# ── Path resolution ───────────────────────────────────────────────────────────

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_DEFAULT_MPC_CONFIG = _PROJECT_ROOT / "config" / "mpc_config.yaml"


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class MPCConfig:
    """Comprehensive MPC configuration loaded from ``config/mpc_config.yaml``."""

    # Timing
    dt_minutes: int = 5
    prediction_horizon_hours: int = 12
    control_horizon_hours: int = 6

    # Solver
    solver_method: str = "SLSQP"
    solver_max_iter: int = 200
    solver_ftol: float = 1e-6

    # Cost weights (defaults — overridden per stage by multiplier table)
    w_temp: float = 2.0
    w_humidity: float = 2.0
    w_soil_moisture: float = 1.5
    w_co2: float = 1.0
    w_vpd: float = 1.0
    w_light: float = 0.4
    w_disease: float = 0.8
    w_energy: float = 0.10
    w_water: float = 0.10
    w_switch: float = 0.30

    # Disease-aware cost tuning
    w_humidity_exposure: float = 0.1       # penalty for RH above setpoint when predicted disease risk is elevated
    w_fogger_suppression: float = 0.1      # extra fogger cost when predicted disease risk exceeds threshold
    w_irrigation_caution: float = 0.05     # extra irrigation cost when RH is high AND disease risk is elevated
    w_severity_amplification: float = 1.0  # how much severity forecast amplifies the base disease weight

    # Disease-sensitive constraint thresholds
    disease_rh_tightening_risk_threshold: float = 0.4  # risk above which RH ceiling is lowered
    disease_rh_tightened_ceiling: float = 80.0          # tightened RH upper bound (%RH)
    disease_fogger_suppress_risk_threshold: float = 0.5 # risk above which fogger upper bound is reduced
    disease_fogger_suppressed_max_duty: float = 0.3     # max fogger duty cycle under disease pressure

    # Weather adaptation
    weather_temp_anticipation: float = 0.3    # weight for pre-cooling when temp spike is forecast
    weather_rh_anticipation: float = 0.2      # weight for ventilation prep when RH shift is forecast
    weather_solar_anticipation: float = 0.15  # weight for heat-load prep when solar gain is forecast
    weather_lookahead_steps: int = 12         # how many steps ahead to scan for weather stress

    # Stage transition blending
    stage_transition_blend_steps: int = 12    # number of steps over which cost weights ramp to next stage

    # Stage weight multipliers (stage_db_key → {weight_name → multiplier})
    stage_weight_multipliers: dict[str, dict[str, float]] = field(
        default_factory=dict
    )

    # Setpoint overrides (stage → {field → value})
    setpoint_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    # Actuator constraints
    actuator_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    actuator_rate_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    actuator_cooldowns: dict[str, int] = field(default_factory=dict)

    # Environmental bounds
    env_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Resource budgets
    daily_water_budget_litres: float = 500.0
    daily_energy_budget_kwh: float = 100.0

    # Energy cost coefficients
    energy_costs: dict[str, float] = field(default_factory=dict)

    # Model artifact run IDs (None → auto-discover latest)
    environment_forecast_run_id: str | None = None
    disease_progression_run_id: str | None = None
    growth_progression_run_id: str | None = None
    disease_classifier_run_id: str | None = None
    growth_classifier_run_id: str | None = None

    # Image streaming
    image_stream_interval_minutes: int = 5
    image_stream_enabled: bool = True

    # Initial conditions
    initial_soil_moisture_pct: float = 60.0

    # Run identity (auto-generated if empty)
    run_id: str = ""

    # ── post-init ──────────────────────────────────────────────────────

    def __post_init__(self) -> None:
        if not self.run_id:
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.run_id = f"mpc_{ts}"

    # ── Derived helpers ────────────────────────────────────────────────

    @property
    def prediction_horizon_steps(self) -> int:
        return self.prediction_horizon_hours * (60 // self.dt_minutes)

    @property
    def control_horizon_steps(self) -> int:
        return self.control_horizon_hours * (60 // self.dt_minutes)

    @property
    def cost_weight_vector(self) -> dict[str, float]:
        """Return the base cost weights as a dict keyed by weight name."""
        return {
            "w_temp": self.w_temp,
            "w_humidity": self.w_humidity,
            "w_soil_moisture": self.w_soil_moisture,
            "w_co2": self.w_co2,
            "w_vpd": self.w_vpd,
            "w_light": self.w_light,
            "w_disease": self.w_disease,
            "w_energy": self.w_energy,
            "w_water": self.w_water,
            "w_switch": self.w_switch,
        }

    @property
    def disease_cost_weights(self) -> dict[str, float]:
        """Return disease-aware cost weights as a dict."""
        return {
            "w_humidity_exposure": self.w_humidity_exposure,
            "w_fogger_suppression": self.w_fogger_suppression,
            "w_irrigation_caution": self.w_irrigation_caution,
            "w_severity_amplification": self.w_severity_amplification,
        }


# ── Loader ────────────────────────────────────────────────────────────────────


def _parse_bounds(raw: dict[str, list]) -> dict[str, tuple[float, float]]:
    """Convert ``{name: [lo, hi]}`` mapping from YAML to tuple pairs."""
    return {k: (float(v[0]), float(v[1])) for k, v in raw.items()}


def load_mpc_config(config_path: str | Path | None = None) -> MPCConfig:
    """Load and return an ``MPCConfig`` from YAML.

    Parameters
    ----------
    config_path:
        Path to ``mpc_config.yaml``.  Defaults to
        ``<project_root>/config/mpc_config.yaml``.
    """
    path = Path(config_path) if config_path else _DEFAULT_MPC_CONFIG
    raw = _load_yaml(path)

    timing = raw.get("timing", {})
    solver = raw.get("solver", {})
    weights = raw.get("cost_weights", {})
    disease = raw.get("disease_cost", {})
    weather_adapt = raw.get("weather_adaptation", {})
    transition = raw.get("stage_transition", {})
    resources = raw.get("resources", {})
    img = raw.get("image_stream", {})
    artifacts = raw.get("model_artifacts", {})
    initial = raw.get("initial_conditions", {})

    return MPCConfig(
        # Timing
        dt_minutes=timing.get("dt_minutes", 5),
        prediction_horizon_hours=timing.get("prediction_horizon_hours", 12),
        control_horizon_hours=timing.get("control_horizon_hours", 6),
        # Solver
        solver_method=solver.get("method", "SLSQP"),
        solver_max_iter=solver.get("max_iter", 200),
        solver_ftol=solver.get("ftol", 1e-6),
        # Cost weights
        w_temp=weights.get("w_temp", 1.0),
        w_humidity=weights.get("w_humidity", 1.0),
        w_soil_moisture=weights.get("w_soil_moisture", 0.8),
        w_co2=weights.get("w_co2", 0.5),
        w_vpd=weights.get("w_vpd", 0.6),
        w_light=weights.get("w_light", 0.4),
        w_disease=weights.get("w_disease", 2.0),
        w_energy=weights.get("w_energy", 0.3),
        w_water=weights.get("w_water", 0.3),
        w_switch=weights.get("w_switch", 0.1),
        # Disease-aware cost
        w_humidity_exposure=disease.get("w_humidity_exposure", 0.5),
        w_fogger_suppression=disease.get("w_fogger_suppression", 0.3),
        w_irrigation_caution=disease.get("w_irrigation_caution", 0.2),
        w_severity_amplification=disease.get("w_severity_amplification", 1.0),
        disease_rh_tightening_risk_threshold=disease.get("rh_tightening_risk_threshold", 0.4),
        disease_rh_tightened_ceiling=disease.get("rh_tightened_ceiling", 80.0),
        disease_fogger_suppress_risk_threshold=disease.get("fogger_suppress_risk_threshold", 0.5),
        disease_fogger_suppressed_max_duty=disease.get("fogger_suppressed_max_duty", 0.3),
        # Weather adaptation
        weather_temp_anticipation=weather_adapt.get("temp_anticipation", 0.3),
        weather_rh_anticipation=weather_adapt.get("rh_anticipation", 0.2),
        weather_solar_anticipation=weather_adapt.get("solar_anticipation", 0.15),
        weather_lookahead_steps=weather_adapt.get("lookahead_steps", 12),
        # Stage transition
        stage_transition_blend_steps=transition.get("blend_steps", 12),
        # Stage weight multipliers
        stage_weight_multipliers=raw.get("stage_weight_multipliers", {}),
        # Setpoint overrides
        setpoint_overrides=raw.get("setpoint_overrides", {}),
        # Actuator constraints
        actuator_bounds=_parse_bounds(raw.get("actuator_bounds", {})),
        actuator_rate_limits=_parse_bounds(raw.get("actuator_rate_limits", {})),
        actuator_cooldowns=raw.get("actuator_cooldowns", {}),
        # Environmental bounds
        env_bounds=_parse_bounds(raw.get("env_bounds", {})),
        # Resources
        daily_water_budget_litres=resources.get("daily_water_budget_litres", 500.0),
        daily_energy_budget_kwh=resources.get("daily_energy_budget_kwh", 100.0),
        # Energy costs
        energy_costs=raw.get("energy_costs", {}),
        # Model artifacts
        environment_forecast_run_id=artifacts.get("environment_forecast_run_id"),
        disease_progression_run_id=artifacts.get("disease_progression_run_id"),
        growth_progression_run_id=artifacts.get("growth_progression_run_id"),
        disease_classifier_run_id=artifacts.get("disease_classifier_run_id"),
        growth_classifier_run_id=artifacts.get("growth_classifier_run_id"),
        # Image streaming
        image_stream_interval_minutes=img.get("interval_minutes", 5),
        image_stream_enabled=img.get("enabled", True),
        # Initial conditions
        initial_soil_moisture_pct=initial.get("soil_moisture_pct", 60.0),
    )
