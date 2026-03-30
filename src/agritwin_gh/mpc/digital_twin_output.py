"""
Digital-twin output formatting.

Converts raw MPC / fusion results into the structured payload dataclasses
(``DigitalTwinStepPayload``, ``DigitalTwinTrajectoryPayload``) consumed by
the dashboard and logging subsystems.
"""

from __future__ import annotations

import datetime as _dt
import logging
from typing import Any

from .constants import ALERT_GREEN, ALERT_RED, ALERT_YELLOW
from .explanation import ExplanationBuilder
from .state import (
    ActuatorState,
    ControllerDecisionContext,
    DigitalTwinStepPayload,
    DigitalTwinTrajectoryPayload,
    FusedState,
    GreenhouseState,
)

logger = logging.getLogger(__name__)


class DigitalTwinOutput:
    """Formats MPC fusion + control outputs into dashboard-ready payloads.

    Parameters
    ----------
    run_id:
        Unique identifier for the current control run.
    """

    def __init__(self, run_id: str = "") -> None:
        self._run_id = run_id
        self._step_index = 0
        self._cumulative_energy_kwh = 0.0
        self._cumulative_water_litres = 0.0
        self._explainer = ExplanationBuilder()

    # ── Step formatting ────────────────────────────────────────────────

    def format_step(
        self,
        fused: FusedState,
        actuators: ActuatorState | None = None,
        predicted_next: GreenhouseState | None = None,
        step_cost: float = 0.0,
        energy_kwh: float = 0.0,
        water_litres: float = 0.0,
        disease_image_key: str | None = None,
        growth_image_key: str | None = None,
        *,
        cost_breakdown: dict[str, float] | None = None,
        solver_converged: bool = True,
        weather_stress: dict[str, float] | None = None,
        tightened_constraints: dict[str, Any] | None = None,
        decision_context: ControllerDecisionContext | None = None,
        solver_performance: dict[str, Any] | None = None,
    ) -> DigitalTwinStepPayload:
        """Build a single-step payload from the fused state and MPC output.

        Parameters
        ----------
        fused:
            The ``FusedState`` assembled by ``StateFusion.fuse()``.
        actuators:
            MPC-computed actuator commands.  *None* → zero actuators.
        predicted_next:
            Predicted greenhouse state at the next timestep.
        step_cost:
            Scalar cost for this MPC step.
        energy_kwh / water_litres:
            Resource consumption for this step.
        disease_image_key / growth_image_key:
            MinIO image keys for dashboard visualisation.
        cost_breakdown:
            Per-component cost contributions from the solver.
        solver_converged:
            Whether the MPC solver converged.
        weather_stress:
            Weather stress summary from ``WeatherAdaptiveModifiers.to_summary()``.
        tightened_constraints:
            Constraint tightening details (e.g. reduced RH ceiling).
        decision_context:
            Full ``ControllerDecisionContext`` snapshot for traceability.
        solver_performance:
            Solver timing / iteration metadata.
        """
        self._cumulative_energy_kwh += energy_kwh
        self._cumulative_water_litres += water_litres

        alert_level, alert_icons = self._compute_alert(fused)
        act = actuators or ActuatorState()

        # Build structured explanation
        explanation_obj = self._explainer.build(
            fused=fused,
            actuators=act,
            cost_breakdown=cost_breakdown,
            solver_converged=solver_converged,
            weather_stress=weather_stress,
            tightened_constraints=tightened_constraints,
        )

        payload = DigitalTwinStepPayload(
            timestamp=fused.timestamp,
            run_id=self._run_id,
            step_index=self._step_index,
            # Observations
            observed_state=fused.greenhouse_state.to_dict(),
            growth_stage=fused.growth_stage,
            disease_classification=fused.disease_classification,
            disease_risk_score=fused.disease_risk_score,
            # MPC decision
            applied_actuators=act.to_dict(),
            predicted_next_state=(predicted_next.to_dict() if predicted_next else {}),
            # Forecasts
            weather_forecast_24h=(
                fused.weather_forecast[0] if fused.weather_forecast else {}
            ),
            severity_forecast_24h=fused.severity_24h,
            severity_forecast_48h=fused.severity_48h,
            hours_to_stage_transition=fused.hours_to_transition,
            # Cost / resources
            step_cost=step_cost,
            cumulative_energy_kwh=self._cumulative_energy_kwh,
            cumulative_water_litres=self._cumulative_water_litres,
            # Images
            disease_image_key=disease_image_key,
            growth_stage_image_key=growth_image_key,
            # Alert
            alert_level=alert_level,
            alert_icons=alert_icons,
            # Explanation & context
            explanation=explanation_obj.to_dict(),
            decision_context=(
                decision_context.to_dict() if decision_context else {}
            ),
            solver_performance=solver_performance or {},
        )

        self._step_index += 1
        return payload

    # ── Trajectory formatting ──────────────────────────────────────────

    def format_trajectory(
        self,
        steps: list[DigitalTwinStepPayload],
    ) -> DigitalTwinTrajectoryPayload:
        """Wrap a list of step payloads into a trajectory payload."""
        total_cost = sum(s.step_cost for s in steps)
        return DigitalTwinTrajectoryPayload(
            run_id=self._run_id,
            steps=steps,
            total_cost=total_cost,
            total_energy_kwh=self._cumulative_energy_kwh,
            total_water_litres=self._cumulative_water_litres,
        )

    # ── Alert logic ────────────────────────────────────────────────────

    @staticmethod
    def _compute_alert(fused: FusedState) -> tuple[str, list[str]]:
        """Determine alert level and icon list from fused state."""
        icons: list[str] = []
        level = ALERT_GREEN

        risk = fused.disease_risk_score
        if risk > 0.6:
            level = ALERT_RED
            icons.append("disease_high")
        elif risk > 0.35:
            level = ALERT_YELLOW
            icons.append("disease_moderate")

        # Check if any disease severity > 40 %
        for disease, sev in fused.severity_24h.items():
            if sev > 40.0:
                if level != ALERT_RED:
                    level = ALERT_RED
                icons.append(f"sev_{disease.replace(' ', '_')}")
                break

        # Imminent transition warning
        if fused.transition_within_24h:
            icons.append("transition_24h")
            if level == ALERT_GREEN:
                level = ALERT_YELLOW

        # Temperature bounds check
        gh = fused.greenhouse_state
        if gh.indoor_temp > 35.0 or gh.indoor_temp < 12.0:
            level = ALERT_RED
            icons.append("temp_extreme")
        elif gh.indoor_temp > 32.0 or gh.indoor_temp < 15.0:
            if level == ALERT_GREEN:
                level = ALERT_YELLOW
            icons.append("temp_warning")

        return level, icons

    # ── State ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset step counter and cumulative resource trackers."""
        self._step_index = 0
        self._cumulative_energy_kwh = 0.0
        self._cumulative_water_litres = 0.0
