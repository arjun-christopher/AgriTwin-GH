"""
Offline experiment runner for comparative MPC evaluation.

Provides:
  • ``ExperimentConfig``          — declarative experiment specification.
  • ``ExperimentRunner``          — orchestrates multi-controller runs.
  • ``ComparisonReport``          — consolidated results with pairwise
    improvements and yield-proxy scoring.
  • Adapter factories             — ``make_baseline_adapter()``,
    ``make_mpc_adapter()``.
  • Scenario helpers              — ``generate_default_weather()``,
    ``generate_default_growth_stages()``, ``make_default_initial_state()``.

The runner is intentionally decoupled from the real-time ``MPCRunner`` so
it can be used in offline evaluation, ablation studies, and CI checks
without touching the database.
"""

from __future__ import annotations

import datetime as _dt
import logging
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .baseline_controller import RuleBasedController
from .config import MPCConfig
from .constants import (
    CONTROL_VARIABLES,
    compute_disease_risk_score,
    stage_label_to_index,
)
from .evaluation_metrics import (
    ControllerMetricsBundle,
    compute_all_metrics,
)
from .greenhouse_model import GreenhouseTransitionModel
from .mpc_solver import MPCSolver, MPCSolution
from .setpoints import get_setpoint
from .state import (
    ActuatorState,
    FusedState,
    GreenhouseState,
    WeatherState,
)
from .yield_proxy import YieldProxyResult, YieldProxyWeights, compute_yield_proxy

logger = logging.getLogger(__name__)

# ── Type alias for a controller adapter ───────────────────────────────────────
# Callable(state, weather, growth_stage, disease_risk, step_index) → ActuatorState

_ControllerAdapter = Callable[
    [GreenhouseState, WeatherState, str, float, int],
    ActuatorState,
]


# ── Experiment configuration ──────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Declarative specification for one comparative experiment run.

    Parameters
    ----------
    n_steps : int
        Total simulation steps.
    dt_minutes : int
        Step duration in minutes.
    initial_state : GreenhouseState
        Starting indoor-climate state.
    weather_sequence : list[WeatherState]
        External disturbance sequence (length must equal ``n_steps``).
    growth_stage_sequence : list[str]
        Canonical growth-stage label per step.
    random_seed : int
        Seed for reproducibility.
    yield_proxy_weights : YieldProxyWeights, optional
        Weights for the yield-proxy scorer.
    experiment_name : str
        Human-readable identifier stored in the report.
    """

    # Required
    n_steps: int = 288
    dt_minutes: int = 5
    initial_state: GreenhouseState = field(default_factory=GreenhouseState)
    weather_sequence: list[WeatherState] = field(default_factory=list)
    growth_stage_sequence: list[str] = field(default_factory=list)

    # Optional
    random_seed: int = 42
    yield_proxy_weights: YieldProxyWeights | None = None
    experiment_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "n_steps": self.n_steps,
            "dt_minutes": self.dt_minutes,
            "random_seed": self.random_seed,
            "initial_state": asdict(self.initial_state),
            "growth_stages_used": sorted(set(self.growth_stage_sequence)),
            "weather_steps": len(self.weather_sequence),
        }


# ── Comparison report ─────────────────────────────────────────────────────────


@dataclass
class ComparisonReport:
    """Consolidated multi-controller experiment results.

    Attributes
    ----------
    controller_metrics : dict[str, ControllerMetricsBundle]
        Full metric bundle per registered controller ID.
    yield_results : dict[str, YieldProxyResult]
        Yield-proxy score breakdown per controller.
    improvements : dict[str, dict[str, float]]
        Pairwise percentage improvements, keyed by
        ``"<ref_id> vs <cmp_id>"``.
    experiment_config : ExperimentConfig
        The configuration that produced this report.
    generated_at : datetime
        Report generation timestamp.
    """

    controller_metrics: dict[str, ControllerMetricsBundle] = field(default_factory=dict)
    yield_results: dict[str, YieldProxyResult] = field(default_factory=dict)
    improvements: dict[str, dict[str, float]] = field(default_factory=dict)
    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    generated_at: _dt.datetime = field(default_factory=_dt.datetime.now)

    # ── convenience accessors ─────────────────────────────────────────

    def summary_table(self) -> dict[str, dict[str, Any]]:
        """Return a nested dict suitable for tabular display.

        Outer key = controller_id.
        Inner keys = selected scalar metrics.
        """
        table: dict[str, dict[str, Any]] = {}
        for cid, m in self.controller_metrics.items():
            yr = self.yield_results.get(cid)
            table[cid] = {
                "n_steps": m.n_steps,
                "temp_rmse": round(m.tracking.get("indoor_temp", _stub_tm()).rmse, 3),
                "hum_rmse": round(m.tracking.get("indoor_humidity", _stub_tm()).rmse, 3),
                "sm_rmse": round(m.tracking.get("soil_moisture", _stub_tm()).rmse, 3),
                "mean_disease_risk": round(m.disease_burden.mean_risk, 4),
                "max_disease_risk": round(m.disease_burden.max_risk, 4),
                "water_litres": round(m.resources.total_water_litres, 2),
                "energy_kwh": round(m.resources.total_energy_kwh, 4),
                "safety_violations": m.safety.total_violations,
                "yield_score": round(yr.overall_score, 2) if yr else 0.0,
            }
        return table

    def to_json_dict(self) -> dict[str, Any]:
        """Full JSON-serialisable representation."""
        return {
            "generated_at": self.generated_at.isoformat(),
            "experiment_config": self.experiment_config.to_dict(),
            "summary_table": self.summary_table(),
            "improvements": self.improvements,
            "controller_metrics": {
                cid: m.to_dict() for cid, m in self.controller_metrics.items()
            },
            "yield_results": {
                cid: yr.to_dict() for cid, yr in self.yield_results.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias for ``to_json_dict()`` — kept for script compatibility."""
        return self.to_json_dict()


def _stub_tm():
    """Return a zeroed TrackingMetrics instance (avoids circular import)."""
    from .evaluation_metrics import TrackingMetrics
    return TrackingMetrics()


# ── Experiment runner ─────────────────────────────────────────────────────────


class ExperimentRunner:
    """Run multiple controllers over the same scenario and compare results.

    Usage
    -----
    ::

        cfg = ExperimentConfig(n_steps=288, ...)
        runner = ExperimentRunner(cfg)
        runner.register_controller("baseline", make_baseline_adapter())
        runner.register_controller("mpc", make_mpc_adapter())
        report = runner.run()
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._cfg = config
        self._controllers: list[tuple[str, _ControllerAdapter, str]] = []
        self._model = GreenhouseTransitionModel()

    def register_controller(
        self,
        controller_id: str,
        adapter: _ControllerAdapter,
        controller_type: str = "unknown",
    ) -> None:
        """Register a controller adapter for evaluation.

        Parameters
        ----------
        controller_id : str
            Unique identifier (e.g. "baseline", "mpc").
        adapter : callable
            Function ``(state, weather, growth_stage, disease_risk, step) → ActuatorState``.
        controller_type : str
            Descriptive type label stored in the metrics bundle.
        """
        self._controllers.append((controller_id, adapter, controller_type))

    def run(self) -> ComparisonReport:
        """Execute all registered controllers and produce a ``ComparisonReport``.

        Each controller is run independently over the same scenario
        (same initial state, weather, growth-stage sequence).  Results
        are compared pairwise (first registered controller is the reference).

        Returns
        -------
        ComparisonReport
        """
        cfg = self._cfg
        n = cfg.n_steps
        weather = _pad_weather(cfg.weather_sequence, n)
        stages = _pad_stages(cfg.growth_stage_sequence, n)

        # Validate inputs
        if n <= 0:
            raise ValueError("ExperimentConfig.n_steps must be > 0")

        rng = random.Random(cfg.random_seed)
        np.random.seed(cfg.random_seed)

        all_metrics: dict[str, ControllerMetricsBundle] = {}
        all_trajectories: dict[str, tuple[list[GreenhouseState], list[ActuatorState]]] = {}

        for cid, adapter, ctype in self._controllers:
            logger.info("Running controller: %s (%s), steps=%d", cid, ctype, n)
            states, actuators = self._simulate(adapter, weather, stages, rng, n, cfg)
            bundle = compute_all_metrics(
                states, actuators, stages,
                controller_id=cid,
                controller_type=ctype,
                dt_minutes=cfg.dt_minutes,
            )
            all_metrics[cid] = bundle
            all_trajectories[cid] = (states, actuators)
            logger.info("  %s: temp_rmse=%.3f  disease=%.4f",
                        cid,
                        bundle.tracking.get("indoor_temp", _stub_tm()).rmse,
                        bundle.disease_burden.mean_risk)

        # Yield proxy scores
        yield_results: dict[str, YieldProxyResult] = {}
        for cid, (states, actuators) in all_trajectories.items():
            yr = compute_yield_proxy(states, actuators, stages, cfg.yield_proxy_weights)
            yr.controller_id = cid  # backfill
            all_metrics[cid].yield_quality_score = yr.overall_score
            yield_results[cid] = yr

        # Pairwise improvements (first controller is baseline reference)
        improvements = _compute_improvements(all_metrics, yield_results)

        return ComparisonReport(
            controller_metrics=all_metrics,
            yield_results=yield_results,
            improvements=improvements,
            experiment_config=cfg,
            generated_at=_dt.datetime.now(),
        )

    # ── Internal simulation ───────────────────────────────────────────

    def _simulate(
        self,
        adapter: _ControllerAdapter,
        weather: list[WeatherState],
        stages: list[str],
        rng: random.Random,
        n: int,
        cfg: ExperimentConfig,
    ) -> tuple[list[GreenhouseState], list[ActuatorState]]:
        """Roll out a single controller over ``n`` steps."""
        state = _copy_state(cfg.initial_state)
        states: list[GreenhouseState] = []
        actuators: list[ActuatorState] = []

        for step in range(n):
            w = weather[step]
            stage = stages[step]

            # Estimate disease risk from current state
            sp = get_setpoint(stage)
            disease_risk = compute_disease_risk_score(
                state.indoor_temp, state.indoor_humidity, state.leaf_wetness_proxy, stage
            )
            state.disease_risk_score = disease_risk
            state.growth_stage_index = stage_label_to_index(stage)

            # Call adapter
            try:
                action = adapter(state, w, stage, disease_risk, step)
            except Exception as exc:
                logger.warning("Adapter error at step %d: %s — using zero action", step, exc)
                action = ActuatorState()

            # Clamp action to [0, 1]
            action = _clamp_action(action)

            states.append(_copy_state(state))
            actuators.append(action)

            # Advance physics
            state = self._model.step(state, action, w)

        return states, actuators


# ── Pairwise improvement computation ─────────────────────────────────────────


def _compute_improvements(
    all_metrics: dict[str, ControllerMetricsBundle],
    yield_results: dict[str, YieldProxyResult],
) -> dict[str, dict[str, float]]:
    """Compute pairwise percentage improvements relative to the first controller."""
    cids = list(all_metrics.keys())
    if len(cids) < 2:
        return {}

    ref_id = cids[0]
    ref_m = all_metrics[ref_id]
    ref_yr = yield_results.get(ref_id)
    improvements: dict[str, dict[str, float]] = {}

    for cid in cids[1:]:
        m = all_metrics[cid]
        yr = yield_results.get(cid)
        pair_key = f"{ref_id} vs {cid}"

        def pct_improve(ref_val: float, new_val: float, lower_is_better: bool = True) -> float:
            """% improvement: positive means new_val is better than ref_val."""
            if abs(ref_val) < 1e-12:
                return 0.0
            if lower_is_better:
                return round(100.0 * (ref_val - new_val) / abs(ref_val), 2)
            else:
                return round(100.0 * (new_val - ref_val) / abs(ref_val), 2)

        imps: dict[str, float] = {}

        # Tracking improvements (lower RMSE is better)
        for var in ["indoor_temp", "indoor_humidity", "soil_moisture", "co2", "vpd"]:
            ref_rmse = ref_m.tracking.get(var, _stub_tm()).rmse
            new_rmse = m.tracking.get(var, _stub_tm()).rmse
            imps[f"{var}_rmse"] = pct_improve(ref_rmse, new_rmse, lower_is_better=True)

        # Disease burden (lower is better)
        imps["mean_disease_risk"] = pct_improve(
            ref_m.disease_burden.mean_risk, m.disease_burden.mean_risk, lower_is_better=True
        )
        imps["rh_exposure"] = pct_improve(
            ref_m.disease_burden.cumulative_rh_exposure,
            m.disease_burden.cumulative_rh_exposure,
            lower_is_better=True,
        )

        # Resources (lower is better)
        imps["water_litres"] = pct_improve(
            ref_m.resources.total_water_litres, m.resources.total_water_litres, lower_is_better=True
        )
        imps["energy_kwh"] = pct_improve(
            ref_m.resources.total_energy_kwh, m.resources.total_energy_kwh, lower_is_better=True
        )

        # Safety (fewer violations is better)
        imps["safety_violations"] = pct_improve(
            float(ref_m.safety.total_violations), float(m.safety.total_violations), lower_is_better=True
        )

        # Yield score (higher is better)
        if ref_yr and yr:
            imps["yield_score"] = pct_improve(
                ref_yr.overall_score, yr.overall_score, lower_is_better=False
            )

        improvements[pair_key] = imps

    return improvements


# ── Controller adapter factories ──────────────────────────────────────────────


def make_baseline_adapter() -> _ControllerAdapter:
    """Return a controller adapter wrapping ``RuleBasedController``.

    The adapter signature is:
    ``(state, weather, growth_stage, disease_risk, step_index) → ActuatorState``
    """
    controller = RuleBasedController()

    def _adapter(
        state: GreenhouseState,
        weather: WeatherState,
        growth_stage: str,
        disease_risk: float,
        step_index: int,
    ) -> ActuatorState:
        payload = controller.compute_action(
            state=state,
            growth_stage=growth_stage,
            disease_risk=disease_risk,
            weather=weather,
        )
        return payload.actuators

    return _adapter


def make_mpc_adapter(config: MPCConfig | None = None) -> _ControllerAdapter:
    """Return a controller adapter wrapping ``MPCSolver``.

    The adapter signature is:
    ``(state, weather, growth_stage, disease_risk, step_index) → ActuatorState``

    The MPC solver optimises over a short horizon at each step
    (receding-horizon, single-shooting formulation).
    """
    from .cost_function import DiseaseContext

    _cfg = config or MPCConfig()
    solver = MPCSolver(_cfg)
    _prev_action: list[ActuatorState] = [ActuatorState()]

    def _adapter(
        state: GreenhouseState,
        weather: WeatherState,
        growth_stage: str,
        disease_risk: float,
        step_index: int,
    ) -> ActuatorState:
        fused = FusedState(
            greenhouse_state=state,
            growth_stage=growth_stage,
            growth_stage_index=stage_label_to_index(growth_stage),
            disease_risk_score=disease_risk,
            disease_classification="healthy leaves",
            disease_confidence=0.9,
            weather_forecast=[weather.to_dict()] * max(1, _cfg.control_horizon_steps),
        )
        try:
            solution: MPCSolution = solver.solve(
                fused,
                previous_control=_prev_action[0],
            )
            action = solution.first_action
        except Exception as exc:
            logger.warning("MPC solve failed at step %d: %s — falling back", step_index, exc)
            from .baseline_controller import RuleBasedController
            fb = RuleBasedController()
            payload = fb.compute_action(
                state=state,
                growth_stage=growth_stage,
                disease_risk=disease_risk,
                weather=weather,
            )
            action = payload.actuators

        _prev_action[0] = action
        return action

    return _adapter


# ── Scenario generation helpers ───────────────────────────────────────────────


def generate_default_weather(
    n_steps: int,
    dt_minutes: int = 5,
    base_temp: float = 22.0,
    base_humidity: float = 60.0,
) -> list[WeatherState]:
    """Generate a synthetic diurnal weather sequence.

    Temperature and humidity follow sinusoidal day/night cycles;
    solar radiation follows a standard sunrise-14h-sunset pattern.

    Parameters
    ----------
    n_steps : int
        Number of timesteps to generate.
    dt_minutes : int
        Step duration in minutes.
    base_temp : float
        Mean outdoor temperature (°C).
    base_humidity : float
        Mean outdoor relative humidity (%RH).

    Returns
    -------
    list[WeatherState]
    """
    seq: list[WeatherState] = []
    for i in range(n_steps):
        hour = (i * dt_minutes / 60.0) % 24.0
        phase = 2.0 * math.pi * (hour - 6.0) / 24.0
        temp = base_temp + 4.0 * math.sin(phase)
        hum = base_humidity - 10.0 * math.sin(phase)
        hum = max(30.0, min(95.0, hum))
        if 6.0 <= hour <= 20.0:
            solar = max(0.0, 500.0 * math.sin(math.pi * (hour - 6.0) / 14.0))
        else:
            solar = 0.0
        windspeed = 1.5 + 0.5 * abs(math.cos(phase))
        seq.append(WeatherState(
            temp_external=round(temp, 1),
            humidity_external=round(hum, 1),
            solar_radiation=round(solar, 1),
            windspeed=round(windspeed, 1),
            conditions="clear" if solar > 100 else "overcast",
        ))
    return seq


def generate_default_growth_stages(
    n_steps: int,
    stage: str = "flowering",
) -> list[str]:
    """Return a constant growth-stage sequence.

    Parameters
    ----------
    n_steps : int
        Length of the sequence.
    stage : str
        Growth stage label to repeat.

    Returns
    -------
    list[str]
    """
    return [stage] * n_steps


def make_default_initial_state() -> GreenhouseState:
    """Return a sensible default initial indoor-climate state.

    Represents a healthy flowering-stage greenhouse at mid-morning.
    """
    return GreenhouseState(
        indoor_temp=24.0,
        indoor_humidity=65.0,
        soil_moisture=60.0,
        co2=600.0,
        light_intensity=300.0,
        disease_risk_score=0.1,
        growth_stage_index=stage_label_to_index("flowering"),
        vpd=0.8,
        leaf_wetness_proxy=0.2,
    )


# ── Internal helpers ──────────────────────────────────────────────────────────


def _pad_weather(seq: list[WeatherState], n: int) -> list[WeatherState]:
    """Ensure weather sequence has exactly ``n`` entries."""
    if not seq:
        return generate_default_weather(n)
    if len(seq) >= n:
        return seq[:n]
    # Tile
    out = list(seq)
    while len(out) < n:
        out.extend(seq)
    return out[:n]


def _pad_stages(seq: list[str], n: int) -> list[str]:
    """Ensure growth-stage sequence has exactly ``n`` entries."""
    if not seq:
        return ["flowering"] * n
    if len(seq) >= n:
        return seq[:n]
    out = list(seq)
    while len(out) < n:
        out.append(seq[-1])
    return out[:n]


def _copy_state(state: GreenhouseState) -> GreenhouseState:
    """Shallow-copy a GreenhouseState (avoids mutation across controllers)."""
    from dataclasses import replace
    return replace(state)


def _clamp_action(action: ActuatorState) -> ActuatorState:
    """Clamp all actuator values to [0, 1]."""
    from dataclasses import replace
    kwargs = {
        name: max(0.0, min(1.0, getattr(action, name)))
        for name in CONTROL_VARIABLES
    }
    return replace(action, **kwargs)
