"""
MPC streaming engine — the main orchestration loop.

``MPCRunner`` drives the entire real-time (or replay) MPC workflow:

  1. Acquire a DB session.
  2. Instantiate all model wrappers and the state-fusion pipeline.
  3. At each timestep (``DT_MINUTES``):
     a. Call ``StateFusion.fuse()`` to build the ``FusedState``.
     b. Solve the MPC optimisation via ``MPCSolver``.
     c. Format the result via ``DigitalTwinOutput``.
     d. Yield / collect the step payload.
  4. Aggregate into a ``DigitalTwinTrajectoryPayload``.
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import asdict
from typing import Any, Generator

from sqlalchemy.orm import Session

from .config import MPCConfig, load_mpc_config
from .constants import DT_MINUTES
from .digital_twin_output import DigitalTwinOutput
from .disease_penalty import DiseaseRiskPenalty
from .disturbance import WeatherDisturbanceForecast
from .greenhouse_model import GreenhouseTransitionModel
from .growth_weights import GrowthStageWeights
from .image_streamer import ImageStreamer
from .mpc_input_preparation import MPCInputPreparation
from .mpc_solver import MPCSolver, MPCSolution
from .state import (
    ActuatorState,
    ControllerDecisionContext,
    DigitalTwinStepPayload,
    DigitalTwinTrajectoryPayload,
    FusedState,
)
from .state_fusion import StateFusion

logger = logging.getLogger(__name__)


class MPCRunner:
    """Top-level orchestrator that ties together every MPC sub-system.

    Parameters
    ----------
    session:
        A live SQLAlchemy ``Session``.
    config:
        Loaded ``MPCConfig``.  If *None*, the default YAML is loaded.
    disease_classifier:
        Optional callable wrapping ``disease_inference.predict_image``.
    growth_classifier:
        Optional callable wrapping ``growth_stage_inference.predict_growth_stage``.
    device:
        PyTorch device string for the weather-forecast ensemble.
    """

    def __init__(
        self,
        session: Session,
        config: MPCConfig | None = None,
        disease_classifier: Any | None = None,
        growth_classifier: Any | None = None,
        device: str = "cpu",
    ) -> None:
        self._session = session
        self._cfg = config or load_mpc_config()

        # ── Sub-system instantiation ───────────────────────────────────
        self._input_prep = MPCInputPreparation(session)
        self._image_streamer = ImageStreamer(session)

        self._weather = WeatherDisturbanceForecast(
            run_id=self._cfg.environment_forecast_run_id,
            device=device,
        )
        self._disease_penalty = DiseaseRiskPenalty(
            run_id=self._cfg.disease_progression_run_id,
        )
        self._growth_weights = GrowthStageWeights(
            stage_weight_multipliers=self._cfg.stage_weight_multipliers,
            run_id=self._cfg.growth_progression_run_id,
        )

        self._fusion = StateFusion(
            config=self._cfg,
            input_prep=self._input_prep,
            weather=self._weather,
            disease_penalty=self._disease_penalty,
            growth_weights=self._growth_weights,
            image_streamer=self._image_streamer,
            disease_classifier=disease_classifier,
            growth_classifier=growth_classifier,
        )

        self._output = DigitalTwinOutput(run_id=self._cfg.run_id)

        # ── MPC solver ─────────────────────────────────────────────────
        self._model = GreenhouseTransitionModel()
        self._solver = MPCSolver(
            config=self._cfg,
            model=self._model,
        )
        self._prev_control: ActuatorState | None = None

        logger.info(
            "MPCRunner initialised | run_id=%s dt=%d min horizon=%d h",
            self._cfg.run_id, self._cfg.dt_minutes,
            self._cfg.prediction_horizon_hours,
        )

    # ── Single-step execution ──────────────────────────────────────────

    def run_single_step(
        self,
        timestamp: _dt.datetime | None = None,
    ) -> DigitalTwinStepPayload:
        """Execute one full MPC control step.

        Parameters
        ----------
        timestamp:
            The logical timestamp for this control step.
            *None* → ``datetime.now()``.

        Returns
        -------
        DigitalTwinStepPayload
            Formatted result ready for the dashboard / logging.
        """
        ts = timestamp or _dt.datetime.now()
        step_meta = {"run_id": self._cfg.run_id, "timestamp": str(ts)}

        # 1. Fuse state
        logger.debug("Step %s — fusing state …", ts)
        fused = self._fusion.fuse(timestamp=ts)

        # 2. Compute adaptive cost weights for current growth stage
        weights = self._growth_weights.get_weights(
            fused.growth_stage,
            base_weights=self._cfg.cost_weight_vector,
        )
        step_meta["cost_weights"] = weights

        # 3. MPC solve
        solution: MPCSolution = self._solver.solve(
            fused=fused,
            previous_control=self._prev_control,
        )
        actuators = solution.first_action
        self._prev_control = actuators

        # 4. Resource accounting from solution
        energy_kwh = self._estimate_energy(actuators)
        water_litres = actuators.irrigation_qty
        step_cost = solution.total_cost

        # Predicted next state (second element of trajectory, if available)
        predicted_next = (
            solution.predicted_states[1]
            if len(solution.predicted_states) > 1
            else None
        )

        # 5. Solver performance snapshot
        solver_perf = {
            "converged": solution.converged,
            "fallback_used": solution.fallback_used,
            "solve_time_ms": solution.solve_time_ms,
            "n_iterations": solution.n_iterations,
            "n_function_evals": solution.n_function_evals,
            "solver_status": solution.solver_status,
        }

        # 6. Weather stress summary (from solver's last adaptation, if any)
        weather_stress = self._solver.last_weather_stress_summary

        # 7. Constraint tightening info
        tightened_constraints = self._solver.last_constraint_tightening

        # 8. Build decision context for traceability
        decision_ctx = ControllerDecisionContext(
            run_id=self._cfg.run_id,
            timestamp=ts,
            step_index=self._output._step_index,
            solver_config={
                "method": self._cfg.solver_method,
                "horizon_hours": self._cfg.prediction_horizon_hours,
                "dt_minutes": self._cfg.dt_minutes,
                "max_iter": self._cfg.solver_max_iter,
            },
            cost_weights=weights,
            weather_stress_summary=weather_stress or {},
            disease_context_summary={
                "risk_score": fused.disease_risk_score,
                "classification": fused.disease_classification,
                "severity_24h": dict(fused.severity_24h),
            },
            constraint_tightening=tightened_constraints or {},
            solver_performance=solver_perf,
            model_ids={
                "environment_forecast": self._cfg.environment_forecast_run_id,
                "disease_progression": self._cfg.disease_progression_run_id,
                "growth_progression": self._cfg.growth_progression_run_id,
            },
        )

        # 9. Format output payload
        payload = self._output.format_step(
            fused=fused,
            actuators=actuators,
            predicted_next=predicted_next,
            step_cost=step_cost,
            energy_kwh=energy_kwh,
            water_litres=water_litres,
            cost_breakdown=solution.cost_breakdown,
            solver_converged=solution.converged,
            weather_stress=weather_stress,
            tightened_constraints=tightened_constraints,
            decision_context=decision_ctx,
            solver_performance=solver_perf,
        )

        logger.info(
            "Step %d complete | ts=%s stage=%s risk=%.3f alert=%s",
            payload.step_index, ts, fused.growth_stage,
            fused.disease_risk_score, payload.alert_level,
        )
        return payload

    # ── Simulation loop ────────────────────────────────────────────────

    def run_simulation(
        self,
        start_time: _dt.datetime,
        end_time: _dt.datetime,
    ) -> DigitalTwinTrajectoryPayload:
        """Iterate over [start_time, end_time) in ``DT_MINUTES`` steps.

        Returns
        -------
        DigitalTwinTrajectoryPayload
            Complete trajectory containing every step payload.
        """
        delta = _dt.timedelta(minutes=DT_MINUTES)
        steps: list[DigitalTwinStepPayload] = []
        current = start_time

        total_steps = int((end_time - start_time).total_seconds() / (DT_MINUTES * 60))
        logger.info(
            "Starting simulation: %s → %s (%d steps, dt=%d min)",
            start_time, end_time, total_steps, DT_MINUTES,
        )

        while current < end_time:
            payload = self.run_single_step(timestamp=current)
            steps.append(payload)
            current += delta

        trajectory = self._output.format_trajectory(steps)
        logger.info(
            "Simulation complete | %d steps | total_cost=%.2f "
            "energy=%.2f kWh water=%.2f L",
            len(steps), trajectory.total_cost,
            trajectory.total_energy_kwh, trajectory.total_water_litres,
        )
        return trajectory

    def run_simulation_iter(
        self,
        start_time: _dt.datetime,
        end_time: _dt.datetime,
    ) -> Generator[DigitalTwinStepPayload, None, None]:
        """Streaming variant of ``run_simulation`` — yields step-by-step.

        Useful for live dashboards that consume payloads incrementally.
        """
        delta = _dt.timedelta(minutes=DT_MINUTES)
        current = start_time
        while current < end_time:
            yield self.run_single_step(timestamp=current)
            current += delta

    # ── Resource estimation ──────────────────────────────────────────

    @staticmethod
    def _estimate_energy(actuators: ActuatorState) -> float:
        """Estimate energy consumption (kWh) for one control step.

        Simple linear model: each actuator channel has a rated power (kW)
        multiplied by its duty fraction and the step duration.
        """
        dt_hours = DT_MINUTES / 60.0
        # Rated power per channel (kW) — conservative estimates
        fan_kw = 0.75
        vent_kw = 0.05     # motor only
        heater_kw = 5.0
        led_kw = 1.2
        fogger_kw = 0.3
        co2_kw = 0.1       # solenoid only
        return (
            actuators.fan_speed * fan_kw
            + actuators.vent_opening * vent_kw
            + actuators.heater_output * heater_kw
            + actuators.led_intensity * led_kw
            + actuators.fogger_duty * fogger_kw
            + actuators.co2_valve_pct * co2_kw
        ) * dt_hours

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def config(self) -> MPCConfig:
        return self._cfg

    @property
    def run_id(self) -> str:
        return self._cfg.run_id

    @property
    def fusion(self) -> StateFusion:
        return self._fusion

    @property
    def output(self) -> DigitalTwinOutput:
        return self._output
