"""
Experiment runner for controller comparison in AgriTwin-GH.

Drives one or more controller strategies through a shared synthetic or
replayed weather / disturbance scenario using ``GreenhouseTransitionModel``,
then evaluates them via the metrics in ``evaluation_metrics`` and yields
``ComparisonReport`` artefacts.

Design goals
============
* **Reproducible** — same ``ExperimentConfig`` → same results (seeded noise).
* **Controller-agnostic** — any callable that maps
  ``(GreenhouseState, str, float, WeatherState) → ActuatorState`` works.
* **Artifacts** — everything JSON-serialisable, timestamped, with run IDs.

Usage
-----
>>> cfg = ExperimentConfig(n_steps=100, initial_state=..., weather_sequence=..., growth_stage_sequence=...)
>>> runner = ExperimentRunner(cfg)
>>> runner.register_controller("baseline", baseline_adapter)
>>> runner.register_controller("mpc_aware", mpc_adapter)
>>> report = runner.run()
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, Sequence

import numpy as np

from .constants import (
    GROWTH_STAGES,
    compute_disease_risk_score,
    stage_label_to_index,
)
from .greenhouse_model import GreenhouseModelParams, GreenhouseTransitionModel
from .evaluation_metrics import (
    ControllerMetricsBundle,
    compute_all_metrics,
)
from .state import ActuatorState, GreenhouseState, WeatherState
from .yield_proxy import YieldProxyResult, YieldProxyWeights, compute_yield_proxy

logger = logging.getLogger(__name__)


# ── Controller protocol ───────────────────────────────────────────────────────


class ControllerCallable(Protocol):
    """Minimal callable interface any controller adapter must satisfy."""

    def __call__(
        self,
        state: GreenhouseState,
        growth_stage: str,
        disease_risk: float,
        weather: WeatherState,
        prev_actuators: ActuatorState | None,
        step_index: int,
    ) -> ActuatorState: ...


# ── Experiment configuration ──────────────────────────────────────────────────


@dataclass
class ExperimentConfig:
    """Full specification of a reproducible evaluation experiment.

    All sequences must have length >= ``n_steps``.
    """

    n_steps: int = 288                                # default: 1 day at 5-min dt
    dt_minutes: int = 5

    # Initial greenhouse state
    initial_state: GreenhouseState = field(default_factory=GreenhouseState)

    # External disturbances (same for all controllers)
    weather_sequence: list[WeatherState] = field(default_factory=list)

    # Growth-stage label per step (constant or stage-transitioning)
    growth_stage_sequence: list[str] = field(default_factory=list)

    # Transition model configuration
    model_params: GreenhouseModelParams | None = None

    # Random seed for stochastic noise (reproducibility)
    random_seed: int = 42

    # Yield proxy weights
    yield_proxy_weights: YieldProxyWeights | None = None

    # Metadata
    experiment_name: str = ""
    description: str = ""
    start_time: _dt.datetime = field(
        default_factory=lambda: _dt.datetime(2025, 6, 1, 0, 0),
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_steps": self.n_steps,
            "dt_minutes": self.dt_minutes,
            "initial_state": self.initial_state.to_dict(),
            "random_seed": self.random_seed,
            "experiment_name": self.experiment_name,
            "description": self.description,
            "start_time": self.start_time.isoformat(),
            "n_weather_steps": len(self.weather_sequence),
            "n_growth_stages": len(self.growth_stage_sequence),
        }


# ── Controller trajectory container ──────────────────────────────────────────


@dataclass
class ControllerTrajectory:
    """Raw simulation trajectory for one controller run."""

    controller_id: str = ""
    controller_type: str = ""
    states: list[GreenhouseState] = field(default_factory=list)
    actuators: list[ActuatorState] = field(default_factory=list)
    growth_stages: list[str] = field(default_factory=list)
    timestamps: list[_dt.datetime] = field(default_factory=list)
    disease_risks: list[float] = field(default_factory=list)

    # Computed after simulation
    metrics: ControllerMetricsBundle | None = None
    yield_proxy: YieldProxyResult | None = None


# ── Experiment runner ─────────────────────────────────────────────────────────


class ExperimentRunner:
    """Orchestrates controller comparison experiments.

    Usage::

        runner = ExperimentRunner(config)
        runner.register_controller("baseline", adapter_fn)
        runner.register_controller("mpc", mpc_adapter_fn)
        report = runner.run()
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._cfg = config
        self._controllers: dict[str, tuple[str, ControllerCallable]] = {}
        self._trajectories: dict[str, ControllerTrajectory] = {}
        self._validate_config()

    def _validate_config(self) -> None:
        n = self._cfg.n_steps
        if len(self._cfg.weather_sequence) < n:
            raise ValueError(
                f"weather_sequence length {len(self._cfg.weather_sequence)} "
                f"< n_steps {n}"
            )
        if len(self._cfg.growth_stage_sequence) < n:
            raise ValueError(
                f"growth_stage_sequence length {len(self._cfg.growth_stage_sequence)} "
                f"< n_steps {n}"
            )

    def register_controller(
        self,
        controller_id: str,
        controller_fn: ControllerCallable,  # type: ignore[override]
        controller_type: str = "custom",
    ) -> None:
        """Register a controller for evaluation.

        Parameters
        ----------
        controller_id : str
            Unique identifier (e.g. ``"baseline"``, ``"mpc_v2"``).
        controller_fn : callable
            Must conform to ``ControllerCallable`` protocol.
        controller_type : str
            Descriptive label (``"baseline"``, ``"mpc"``, ``"mpc_disease_aware"``).
        """
        self._controllers[controller_id] = (controller_type, controller_fn)

    def run(self) -> ComparisonReport:
        """Execute all registered controllers and produce a comparison report."""
        if not self._controllers:
            raise RuntimeError("No controllers registered")

        self._trajectories.clear()

        for cid, (ctype, cfn) in self._controllers.items():
            logger.info("Running controller '%s' (%s) for %d steps", cid, ctype, self._cfg.n_steps)
            traj = self._simulate_controller(cid, ctype, cfn)
            self._trajectories[cid] = traj

        return self._build_report()

    # ── Simulation ────────────────────────────────────────────────────

    def _simulate_controller(
        self,
        controller_id: str,
        controller_type: str,
        controller_fn: ControllerCallable,
    ) -> ControllerTrajectory:
        """Simulate one controller over the full experiment window."""
        cfg = self._cfg
        model = GreenhouseTransitionModel(
            params=cfg.model_params,
            dt_minutes=cfg.dt_minutes,
        )

        # Seed the model's noise for reproducibility
        np.random.seed(cfg.random_seed)

        state = GreenhouseState(**asdict(cfg.initial_state))
        prev_act: ActuatorState | None = None

        traj = ControllerTrajectory(
            controller_id=controller_id,
            controller_type=controller_type,
        )

        for step in range(cfg.n_steps):
            weather = cfg.weather_sequence[step]
            stage = cfg.growth_stage_sequence[step]

            # Update disease risk from current climate
            disease_risk = compute_disease_risk_score(
                temp=state.indoor_temp,
                humidity=state.indoor_humidity,
                leaf_wetness=state.leaf_wetness_proxy,
                growth_stage=stage,
            )
            state.disease_risk_score = disease_risk

            timestamp = cfg.start_time + _dt.timedelta(minutes=step * cfg.dt_minutes)
            state.timestamp = timestamp

            # Record pre-action state
            traj.states.append(GreenhouseState(**asdict(state)))
            traj.growth_stages.append(stage)
            traj.timestamps.append(timestamp)
            traj.disease_risks.append(disease_risk)

            # Controller decides
            actuators = controller_fn(
                state=state,
                growth_stage=stage,
                disease_risk=disease_risk,
                weather=weather,
                prev_actuators=prev_act,
                step_index=step,
            )

            traj.actuators.append(actuators)
            prev_act = actuators

            # Transition model advances the plant
            next_state = model.step(state, actuators, weather)
            state = next_state

        # ── Post-simulation metric computation ─────────────────────
        metrics = compute_all_metrics(
            states=traj.states,
            actuators=traj.actuators,
            growth_stages=traj.growth_stages,
            controller_id=controller_id,
            controller_type=controller_type,
            dt_minutes=cfg.dt_minutes,
        )

        yield_result = compute_yield_proxy(
            states=traj.states,
            actuators=traj.actuators,
            growth_stages=traj.growth_stages,
            weights=cfg.yield_proxy_weights,
        )

        metrics.yield_quality_score = yield_result.overall_score
        traj.metrics = metrics
        traj.yield_proxy = yield_result

        return traj

    # ── Report assembly ───────────────────────────────────────────────

    def _build_report(self) -> ComparisonReport:
        """Assemble a ComparisonReport from all controller trajectories."""
        bundles: dict[str, ControllerMetricsBundle] = {}
        yield_results: dict[str, YieldProxyResult] = {}

        for cid, traj in self._trajectories.items():
            if traj.metrics:
                bundles[cid] = traj.metrics
            if traj.yield_proxy:
                yield_results[cid] = traj.yield_proxy

        # Compute pair-wise improvements
        improvements: dict[str, dict[str, float]] = {}
        controller_ids = list(bundles.keys())
        if len(controller_ids) >= 2:
            ref_id = controller_ids[0]
            for cid in controller_ids[1:]:
                improvements[f"{cid}_vs_{ref_id}"] = _compute_improvements(
                    reference=bundles[ref_id],
                    candidate=bundles[cid],
                )

        return ComparisonReport(
            experiment_config=self._cfg,
            controller_metrics=bundles,
            yield_results=yield_results,
            improvements=improvements,
            trajectories=dict(self._trajectories),
            generated_at=_dt.datetime.now(),
        )


# ── Comparison report ─────────────────────────────────────────────────────────


@dataclass
class ComparisonReport:
    """Aggregated comparison of all controllers in an experiment."""

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    controller_metrics: dict[str, ControllerMetricsBundle] = field(default_factory=dict)
    yield_results: dict[str, YieldProxyResult] = field(default_factory=dict)
    improvements: dict[str, dict[str, float]] = field(default_factory=dict)
    trajectories: dict[str, ControllerTrajectory] = field(default_factory=dict)
    generated_at: _dt.datetime = field(default_factory=_dt.datetime.now)

    def summary_table(self) -> dict[str, dict[str, Any]]:
        """Return a controller-keyed dict suitable for tabular display."""
        table: dict[str, dict[str, Any]] = {}
        for cid, m in self.controller_metrics.items():
            row: dict[str, Any] = {
                "type": m.controller_type,
                "n_steps": m.n_steps,
                "temp_rmse": round(m.tracking.get("indoor_temp", _empty_tracking()).rmse, 3),
                "humidity_rmse": round(m.tracking.get("indoor_humidity", _empty_tracking()).rmse, 3),
                "soil_moisture_rmse": round(m.tracking.get("soil_moisture", _empty_tracking()).rmse, 3),
                "disease_risk_mean": round(m.disease_burden.mean_risk, 4),
                "disease_risk_max": round(m.disease_burden.max_risk, 4),
                "total_water_l": m.resources.total_water_litres,
                "total_energy_kwh": m.resources.total_energy_kwh,
                "mean_smoothness": m.control_quality.mean_smoothness_l2,
                "safety_violations": m.safety.total_violations,
                "yield_score": m.yield_quality_score,
            }
            table[cid] = row
        return table

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_config": self.experiment_config.to_dict(),
            "controller_metrics": {
                cid: m.to_dict() for cid, m in self.controller_metrics.items()
            },
            "yield_results": {
                cid: yr.to_dict() for cid, yr in self.yield_results.items()
            },
            "improvements": dict(self.improvements),
            "generated_at": self.generated_at.isoformat(),
            "summary_table": self.summary_table(),
        }


def _empty_tracking():
    from .evaluation_metrics import TrackingMetrics
    return TrackingMetrics()


# ── Improvement computation ───────────────────────────────────────────────────


def _compute_improvements(
    reference: ControllerMetricsBundle,
    candidate: ControllerMetricsBundle,
) -> dict[str, float]:
    """Compute percentage improvement of *candidate* over *reference*.

    Positive = candidate is better, negative = worse.
    Convention:
      - For RMSE / risk / consumption: improvement = (ref - cand) / ref * 100
      - For yield score: improvement = (cand - ref) / max(ref, 1) * 100
    """
    imp: dict[str, float] = {}

    # Tracking RMSE improvements (lower is better)
    for var in ("indoor_temp", "indoor_humidity", "soil_moisture", "co2", "vpd"):
        ref_v = reference.tracking.get(var, _empty_tracking()).rmse
        cand_v = candidate.tracking.get(var, _empty_tracking()).rmse
        if ref_v > 1e-9:
            imp[f"{var}_rmse_improvement_pct"] = round((ref_v - cand_v) / ref_v * 100, 2)

    # Disease risk (lower is better)
    ref_dr = reference.disease_burden.mean_risk
    cand_dr = candidate.disease_burden.mean_risk
    if ref_dr > 1e-9:
        imp["disease_risk_mean_improvement_pct"] = round((ref_dr - cand_dr) / ref_dr * 100, 2)

    # Resource consumption (lower is better)
    ref_w = reference.resources.total_water_litres
    cand_w = candidate.resources.total_water_litres
    if ref_w > 1e-9:
        imp["water_savings_pct"] = round((ref_w - cand_w) / ref_w * 100, 2)

    ref_e = reference.resources.total_energy_kwh
    cand_e = candidate.resources.total_energy_kwh
    if ref_e > 1e-9:
        imp["energy_savings_pct"] = round((ref_e - cand_e) / ref_e * 100, 2)

    # Control smoothness (lower L2 is better)
    ref_sm = reference.control_quality.mean_smoothness_l2
    cand_sm = candidate.control_quality.mean_smoothness_l2
    if ref_sm > 1e-9:
        imp["smoothness_improvement_pct"] = round((ref_sm - cand_sm) / ref_sm * 100, 2)

    # Safety (fewer violations is better)
    ref_v = reference.safety.total_violations
    cand_v = candidate.safety.total_violations
    if ref_v > 0:
        imp["safety_improvement_pct"] = round((ref_v - cand_v) / ref_v * 100, 2)

    # Yield score (higher is better)
    ref_y = reference.yield_quality_score
    cand_y = candidate.yield_quality_score
    imp["yield_score_improvement_pct"] = round((cand_y - ref_y) / max(ref_y, 1.0) * 100, 2)

    return imp


# ── Convenience: controller adapters ──────────────────────────────────────────


def make_baseline_adapter(
    controller: Any | None = None,
) -> ControllerCallable:
    """Wrap a ``RuleBasedController`` into the ``ControllerCallable`` protocol.

    If *controller* is ``None`` a default ``RuleBasedController`` is created.
    """
    if controller is None:
        from .baseline_controller import RuleBasedController
        controller = RuleBasedController()

    def _adapter(
        state: GreenhouseState,
        growth_stage: str,
        disease_risk: float,
        weather: WeatherState,
        prev_actuators: ActuatorState | None,
        step_index: int,
    ) -> ActuatorState:
        payload = controller.compute_action(
            state=state,
            growth_stage=growth_stage,
            disease_risk=disease_risk,
            weather=weather,
        )
        return payload.actuators

    return _adapter  # type: ignore[return-value]


def make_mpc_adapter(
    solver: Any | None = None,
    config: Any | None = None,
    weather_lookahead: int = 12,
    weather_sequence: list[WeatherState] | None = None,
) -> ControllerCallable:
    """Wrap an ``MPCSolver`` into the ``ControllerCallable`` protocol.

    The adapter constructs a minimal ``FusedState`` from the arguments and
    passes the weather sequence as a look-ahead window.

    Parameters
    ----------
    solver : MPCSolver or None
        If ``None`` a default solver is created using *config*.
    config : MPCConfig or None
        Used only when *solver* is ``None``.
    weather_lookahead : int
        Number of weather steps passed to the solver per call.
    weather_sequence : list[WeatherState] or None
        Full weather forecast.  When provided, the adapter slices a
        lookahead window starting at the current *step_index* so the
        solver sees upcoming weather transitions.
    """
    if solver is None:
        from .config import MPCConfig
        from .mpc_solver import MPCSolver
        _cfg = config or MPCConfig()
        solver = MPCSolver(_cfg)

    # ── Safety filter: predict-then-check with baseline fallback ──────
    from .greenhouse_model import GreenhouseTransitionModel
    from .baseline_controller import RuleBasedController
    from .constraints import get_default_constraints

    _safety_model = GreenhouseTransitionModel()
    _baseline = RuleBasedController(constraints=get_default_constraints())

    # Tighter bounds with ~5 % margin inside the evaluation safety bounds
    _filter_bounds: dict[str, tuple[float, float]] = {
        "indoor_temp": (11.4, 36.6),
        "indoor_humidity": (28.5, 91.2),
        "vpd": (0.385, 1.9),
        "co2": (287.5, 2375.0),
        "soil_moisture": (19.0, 91.2),
    }

    # Convert weather_sequence to dicts once for fast slicing
    _weather_dicts: list[dict[str, Any]] | None = None
    if weather_sequence:
        _weather_dicts = [w.to_dict() for w in weather_sequence]

    def _adapter(
        state: GreenhouseState,
        growth_stage: str,
        disease_risk: float,
        weather: WeatherState,
        prev_actuators: ActuatorState | None,
        step_index: int,
    ) -> ActuatorState:
        from .state import FusedState

        # Build a minimal FusedState for the solver
        fused = FusedState(
            greenhouse_state=state,
            growth_stage=growth_stage,
            growth_stage_index=state.growth_stage_index,
            disease_risk_score=disease_risk,
            disease_classification="healthy leaves" if disease_risk < 0.3 else "early blight",
            disease_confidence=0.8,
        )

        # Build weather forecast: lookahead window when full sequence available
        if _weather_dicts is not None:
            wf = _weather_dicts[step_index: step_index + weather_lookahead]
            if not wf:
                wf = [weather.to_dict()]
        else:
            wf = [weather.to_dict()]

        solution = solver.solve(
            fused=fused,
            weather_forecast=wf,
            previous_control=prev_actuators,
        )
        action = solution.first_action

        # ── Safety filter: simulate one step, check bounds ────────────
        _w_d = weather.to_dict()
        predicted = _safety_model.step(state, action, _w_d)
        safe = True
        for var, (lo, hi) in _filter_bounds.items():
            val = getattr(predicted, var, 0.0)
            if val < lo or val > hi:
                safe = False
                break

        if not safe:
            bl_result = _baseline.compute_action(
                state, growth_stage, disease_risk=disease_risk,
            )
            action = bl_result.actuators

        # ── Dead-band filter: snap small changes to previous action ───
        # Reduces unnecessary switching → improves yield proxy stability
        # score without harming tracking for meaningful adjustments.
        # CO2 valve excluded — frequent small adjustments needed for tracking.
        if prev_actuators is not None:
            _db = {
                'fan_speed': 0.08, 'vent_opening': 0.08,
                'irrigation_qty': 2.0, 'heater_output': 0.08,
                'led_intensity': 0.08,
                'fogger_duty': 0.08,
            }
            for attr, threshold in _db.items():
                cur = getattr(action, attr, 0.0)
                prev = getattr(prev_actuators, attr, 0.0)
                if abs(cur - prev) < threshold:
                    setattr(action, attr, prev)

        return action

    return _adapter  # type: ignore[return-value]


# ── Convenience: synthetic scenario generators ────────────────────────────────


def generate_default_weather(
    n_steps: int,
    dt_minutes: int = 5,
    base_temp: float = 20.0,
    diurnal_amp: float = 8.0,
) -> list[WeatherState]:
    """Generate a synthetic diurnal weather sequence.

    Produces a temperature sinusoid (24h period) with mild humidity
    variation, suitable for quick experiments.
    """
    weather: list[WeatherState] = []
    for i in range(n_steps):
        hour_frac = (i * dt_minutes / 60.0) % 24.0
        phase = 2.0 * np.pi * (hour_frac - 6.0) / 24.0  # peak at 15:00
        temp = base_temp + diurnal_amp * np.sin(phase)
        hum = 60.0 + 15.0 * np.cos(phase)  # anti-correlated with temp
        solar = max(0.0, 800.0 * np.sin(np.pi * hour_frac / 14.0)) if 6 <= hour_frac <= 20 else 0.0
        weather.append(WeatherState(
            temp_external=round(temp, 1),
            humidity_external=round(max(30.0, min(95.0, hum)), 1),
            solar_radiation=round(solar, 1),
            windspeed=round(2.0 + 1.5 * abs(np.sin(phase)), 1),
            conditions="clear" if solar > 200 else "cloudy",
        ))
    return weather


def generate_default_growth_stages(
    n_steps: int,
    stage: str = "flowering",
) -> list[str]:
    """Generate a constant growth stage sequence."""
    return [stage] * n_steps


def make_default_initial_state() -> GreenhouseState:
    """Return a reasonable initial greenhouse state for experiments."""
    return GreenhouseState(
        indoor_temp=23.0,
        indoor_humidity=68.0,
        soil_moisture=65.0,
        co2=650.0,
        light_intensity=300.0,
        disease_risk_score=0.15,
        growth_stage_index=stage_label_to_index("flowering"),
        vpd=0.8,
        leaf_wetness_proxy=0.3,
    )
