"""
Structured controller explanation generator.

Produces human-readable, machine-parseable explanations for *why* the MPC
controller selected a particular control action.  Each explanation is a
list of ``ExplanationEntry`` items, each representing one identified
condition and the controller's response.

Explanation categories
----------------------
``climate``
    Temperature, humidity, VPD, CO2, light deviations from setpoint.
``disease``
    Disease classification, severity trend, risk-score driven actions.
``weather``
    Upcoming weather stress that triggers proactive actuator changes.
``growth``
    Growth-stage adaptive behaviour, imminent stage transitions.
``resource``
    Energy / water budgeting decisions, fogger suppression.
``constraint``
    Active constraint tightening (e.g. RH ceiling lowered for disease).

Usage::

    from agritwin_gh.mpc.explanation import ExplanationBuilder

    builder = ExplanationBuilder()
    entries = builder.build(fused, actuators, solution, weather_mods, disease_ctx)
    for e in entries:
        print(f"[{e.category}] {e.summary}")  # structured, not free text
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import STATE_VARIABLES
from .state import ActuatorState, FusedState, GreenhouseState


# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class ExplanationEntry:
    """Single explanation item describing one identified condition + response.

    Attributes
    ----------
    category : str
        One of: climate, disease, weather, growth, resource, constraint.
    trigger : str
        Machine-readable trigger key (e.g. ``"humidity_above_setpoint"``).
    summary : str
        Human-readable one-liner.
    details : dict
        Numeric context for dashboard display and analytics.
    severity : str
        ``"info"`` / ``"warning"`` / ``"critical"``
    """

    category: str = ""
    trigger: str = ""
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ControllerExplanation:
    """Complete explanation payload for one MPC step.

    Attributes
    ----------
    timestamp : datetime | None
        When this explanation was generated.
    entries : list[ExplanationEntry]
        Ordered list of identified conditions and responses.
    dominant_factor : str
        Category of the highest-severity entry.
    action_summary : str
        One-sentence summary of the overall control action.
    """

    timestamp: _dt.datetime | None = None
    entries: list[ExplanationEntry] = field(default_factory=list)
    dominant_factor: str = ""
    action_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "entries": [e.to_dict() for e in self.entries],
            "dominant_factor": self.dominant_factor,
            "action_summary": self.action_summary,
        }


# ── Builder ───────────────────────────────────────────────────────────────────

_SEVERITY_RANK = {"info": 0, "warning": 1, "critical": 2}


class ExplanationBuilder:
    """Builds structured explanations from controller context.

    Stateless — call ``build()`` once per control step.
    """

    def build(
        self,
        fused: FusedState,
        actuators: ActuatorState,
        cost_breakdown: dict[str, float] | None = None,
        solver_converged: bool = True,
        weather_stress: dict[str, float] | None = None,
        tightened_constraints: dict[str, Any] | None = None,
    ) -> ControllerExplanation:
        """Generate a complete explanation for the current control decision.

        Parameters
        ----------
        fused : FusedState
            Comprehensive fused state at this timestep.
        actuators : ActuatorState
            The control action that was chosen.
        cost_breakdown : dict, optional
            Cost decomposition from solver (keys such as
            ``"stage_cost_total"``, ``"disease_risk_score"``,
            ``"weather_temp_stress"``).
        solver_converged : bool
            Whether the optimiser converged (if False, fallback was used).
        weather_stress : dict, optional
            Summary weather stress indicators from
            ``WeatherAdaptiveModifiers.to_summary()``.
        tightened_constraints : dict, optional
            Any constraint tightening that was applied (e.g. reduced RH
            ceiling, fogger suppression).
        """
        entries: list[ExplanationEntry] = []
        gs = fused.greenhouse_state
        sp = fused.setpoint

        # ── Climate deviations ────────────────────────────────────────
        entries.extend(self._check_climate(gs, sp, actuators))

        # ── Disease-related ───────────────────────────────────────────
        entries.extend(self._check_disease(fused, actuators, cost_breakdown))

        # ── Weather stress ────────────────────────────────────────────
        entries.extend(self._check_weather(weather_stress, actuators))

        # ── Growth stage ──────────────────────────────────────────────
        entries.extend(self._check_growth(fused))

        # ── Constraint tightening ─────────────────────────────────────
        entries.extend(self._check_constraints(tightened_constraints))

        # ── Solver fallback ───────────────────────────────────────────
        if not solver_converged:
            entries.append(ExplanationEntry(
                category="resource",
                trigger="solver_fallback",
                summary="MPC solver did not converge; rule-based fallback was applied",
                severity="warning",
            ))

        # Determine dominant factor and action summary
        dominant = self._dominant_category(entries)
        action_summary = self._build_action_summary(actuators, entries)

        return ControllerExplanation(
            timestamp=fused.timestamp,
            entries=entries,
            dominant_factor=dominant,
            action_summary=action_summary,
        )

    # ── Climate checks ────────────────────────────────────────────────

    @staticmethod
    def _check_climate(
        gs: GreenhouseState,
        sp: Any,
        act: ActuatorState,
    ) -> list[ExplanationEntry]:
        entries: list[ExplanationEntry] = []
        if sp is None:
            return entries

        # Temperature
        temp_dev = gs.indoor_temp - sp.temp
        if abs(temp_dev) > sp.temp_tol:
            if temp_dev > 0:
                sev = "critical" if temp_dev > sp.temp_tol * 2 else "warning"
                entries.append(ExplanationEntry(
                    category="climate",
                    trigger="temperature_above_setpoint",
                    summary=(
                        f"Temperature {gs.indoor_temp:.1f}°C is {temp_dev:.1f}°C "
                        f"above target {sp.temp:.1f}°C"
                    ),
                    details={
                        "current": gs.indoor_temp, "target": sp.temp,
                        "deviation": round(temp_dev, 2), "tolerance": sp.temp_tol,
                        "fan_response": round(act.fan_speed, 2),
                        "vent_response": round(act.vent_opening, 2),
                    },
                    severity=sev,
                ))
            else:
                sev = "critical" if abs(temp_dev) > sp.temp_tol * 2 else "warning"
                entries.append(ExplanationEntry(
                    category="climate",
                    trigger="temperature_below_setpoint",
                    summary=(
                        f"Temperature {gs.indoor_temp:.1f}°C is {abs(temp_dev):.1f}°C "
                        f"below target {sp.temp:.1f}°C"
                    ),
                    details={
                        "current": gs.indoor_temp, "target": sp.temp,
                        "deviation": round(temp_dev, 2), "tolerance": sp.temp_tol,
                        "heater_response": round(act.heater_output, 2),
                    },
                    severity=sev,
                ))

        # Humidity
        hum_dev = gs.indoor_humidity - sp.humidity
        if abs(hum_dev) > sp.hum_tol:
            if hum_dev > 0:
                sev = "critical" if hum_dev > sp.hum_tol * 2 else "warning"
                entries.append(ExplanationEntry(
                    category="climate",
                    trigger="humidity_above_setpoint",
                    summary=(
                        f"Humidity {gs.indoor_humidity:.1f}% is {hum_dev:.1f}% "
                        f"above {fused_stage_label(sp)} safe range"
                    ),
                    details={
                        "current": gs.indoor_humidity, "target": sp.humidity,
                        "deviation": round(hum_dev, 2), "tolerance": sp.hum_tol,
                        "fan_response": round(act.fan_speed, 2),
                        "vent_response": round(act.vent_opening, 2),
                        "fogger_duty": round(act.fogger_duty, 2),
                    },
                    severity=sev,
                ))
            else:
                entries.append(ExplanationEntry(
                    category="climate",
                    trigger="humidity_below_setpoint",
                    summary=(
                        f"Humidity {gs.indoor_humidity:.1f}% is {abs(hum_dev):.1f}% "
                        f"below target {sp.humidity:.1f}%"
                    ),
                    details={
                        "current": gs.indoor_humidity, "target": sp.humidity,
                        "deviation": round(hum_dev, 2),
                        "fogger_response": round(act.fogger_duty, 2),
                    },
                    severity="warning",
                ))

        # VPD
        if hasattr(sp, "vpd") and sp.vpd > 0:
            vpd_dev = gs.vpd - sp.vpd
            if abs(vpd_dev) > 0.3:
                entries.append(ExplanationEntry(
                    category="climate",
                    trigger="vpd_deviation",
                    summary=f"VPD {gs.vpd:.2f} kPa deviates {vpd_dev:+.2f} from target {sp.vpd:.2f}",
                    details={"current": gs.vpd, "target": sp.vpd, "deviation": round(vpd_dev, 2)},
                    severity="warning" if abs(vpd_dev) > 0.5 else "info",
                ))

        return entries

    # ── Disease checks ────────────────────────────────────────────────

    @staticmethod
    def _check_disease(
        fused: FusedState,
        act: ActuatorState,
        cost_breakdown: dict[str, float] | None,
    ) -> list[ExplanationEntry]:
        entries: list[ExplanationEntry] = []
        risk = fused.disease_risk_score

        if risk > 0.6:
            entries.append(ExplanationEntry(
                category="disease",
                trigger="high_disease_risk",
                summary=f"Disease risk {risk:.2f} is critically elevated",
                details={
                    "risk_score": risk,
                    "classification": fused.disease_classification,
                    "confidence": fused.disease_confidence,
                    "fan_response": round(act.fan_speed, 2),
                    "fogger_suppressed": act.fogger_duty < 0.1,
                },
                severity="critical",
            ))
        elif risk > 0.35:
            entries.append(ExplanationEntry(
                category="disease",
                trigger="moderate_disease_risk",
                summary=f"Disease risk {risk:.2f} is moderately elevated",
                details={"risk_score": risk, "classification": fused.disease_classification},
                severity="warning",
            ))

        # Severity worsening trend
        for disease, sev_24h in fused.severity_24h.items():
            current = fused.current_severity.get(disease, 0.0)
            if sev_24h > current + 5.0:
                entries.append(ExplanationEntry(
                    category="disease",
                    trigger="severity_worsening",
                    summary=(
                        f"{disease.replace('_', ' ').title()} severity predicted "
                        f"to worsen: {current:.0f}% → {sev_24h:.0f}% in 24h"
                    ),
                    details={
                        "disease": disease,
                        "current_severity": current,
                        "predicted_24h": sev_24h,
                        "predicted_48h": fused.severity_48h.get(disease, 0.0),
                    },
                    severity="critical" if sev_24h > 50 else "warning",
                ))

        # Fogger / irrigation caution
        if risk > 0.4 and act.fogger_duty < 0.15:
            entries.append(ExplanationEntry(
                category="disease",
                trigger="fogger_suppressed_for_disease",
                summary="Fogger suppressed to reduce moisture and disease-favorable conditions",
                details={"risk_score": risk, "fogger_duty": round(act.fogger_duty, 2)},
                severity="info",
            ))

        if risk > 0.4 and act.irrigation_qty < 5.0:
            entries.append(ExplanationEntry(
                category="disease",
                trigger="irrigation_cautious_for_disease",
                summary="Irrigation kept low to avoid high humidity persistence",
                details={"risk_score": risk, "irrigation_qty": round(act.irrigation_qty, 2)},
                severity="info",
            ))

        return entries

    # ── Weather checks ────────────────────────────────────────────────

    @staticmethod
    def _check_weather(
        weather_stress: dict[str, float] | None,
        act: ActuatorState,
    ) -> list[ExplanationEntry]:
        entries: list[ExplanationEntry] = []
        if not weather_stress:
            return entries

        temp_stress = weather_stress.get("avg_temp_stress", 0.0)
        rh_stress = weather_stress.get("avg_rh_stress", 0.0)

        if temp_stress > 0.3:
            entries.append(ExplanationEntry(
                category="weather",
                trigger="heat_stress_anticipated",
                summary=(
                    f"External temperature stress anticipated "
                    f"(stress={temp_stress:.2f}); ventilation increased proactively"
                ),
                details={
                    "temp_stress": temp_stress,
                    "fan_response": round(act.fan_speed, 2),
                    "vent_response": round(act.vent_opening, 2),
                },
                severity="warning" if temp_stress > 0.6 else "info",
            ))

        if rh_stress > 0.3:
            entries.append(ExplanationEntry(
                category="weather",
                trigger="humidity_stress_anticipated",
                summary=(
                    f"External humidity stress anticipated "
                    f"(stress={rh_stress:.2f}); ventilation adjusted"
                ),
                details={
                    "rh_stress": rh_stress,
                    "fan_response": round(act.fan_speed, 2),
                },
                severity="warning" if rh_stress > 0.6 else "info",
            ))

        return entries

    # ── Growth stage checks ───────────────────────────────────────────

    @staticmethod
    def _check_growth(fused: FusedState) -> list[ExplanationEntry]:
        entries: list[ExplanationEntry] = []

        if fused.transition_within_24h:
            entries.append(ExplanationEntry(
                category="growth",
                trigger="stage_transition_imminent",
                summary=(
                    f"Growth stage transition {fused.growth_stage} → "
                    f"{fused.next_stage} expected in ~{fused.hours_to_transition:.0f}h; "
                    f"control weights blending toward next-stage profile"
                ),
                details={
                    "current_stage": fused.growth_stage,
                    "next_stage": fused.next_stage,
                    "hours_to_transition": fused.hours_to_transition,
                },
                severity="info",
            ))
        elif fused.transition_within_48h:
            entries.append(ExplanationEntry(
                category="growth",
                trigger="stage_transition_approaching",
                summary=(
                    f"Growth stage transition to {fused.next_stage} "
                    f"expected in ~{fused.hours_to_transition:.0f}h"
                ),
                details={
                    "current_stage": fused.growth_stage,
                    "next_stage": fused.next_stage,
                    "hours_to_transition": fused.hours_to_transition,
                },
                severity="info",
            ))

        return entries

    # ── Constraint checks ─────────────────────────────────────────────

    @staticmethod
    def _check_constraints(
        tightened: dict[str, Any] | None,
    ) -> list[ExplanationEntry]:
        entries: list[ExplanationEntry] = []
        if not tightened:
            return entries

        if "rh_ceiling" in tightened:
            entries.append(ExplanationEntry(
                category="constraint",
                trigger="rh_ceiling_tightened",
                summary=(
                    f"RH ceiling tightened to {tightened['rh_ceiling']:.1f}% "
                    f"due to disease risk"
                ),
                details=tightened,
                severity="warning",
            ))

        if "fogger_suppressed" in tightened:
            entries.append(ExplanationEntry(
                category="constraint",
                trigger="fogger_constraint_active",
                summary=(
                    f"Fogger max duty limited to "
                    f"{tightened['fogger_suppressed']:.0%} due to disease risk"
                ),
                details=tightened,
                severity="warning",
            ))

        return entries

    # ── Helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _dominant_category(entries: list[ExplanationEntry]) -> str:
        if not entries:
            return "climate"
        best = max(entries, key=lambda e: _SEVERITY_RANK.get(e.severity, 0))
        return best.category

    @staticmethod
    def _build_action_summary(
        act: ActuatorState,
        entries: list[ExplanationEntry],
    ) -> str:
        parts: list[str] = []
        if act.fan_speed > 0.5:
            parts.append(f"fan at {act.fan_speed:.0%}")
        if act.vent_opening > 0.3:
            parts.append(f"vents at {act.vent_opening:.0%}")
        if act.heater_output > 0.1:
            parts.append(f"heater at {act.heater_output:.0%}")
        if act.irrigation_qty > 1.0:
            parts.append(f"irrigating {act.irrigation_qty:.1f}L")
        if act.fogger_duty > 0.1:
            parts.append(f"fogger at {act.fogger_duty:.0%}")
        if act.led_intensity > 0.3:
            parts.append(f"LEDs at {act.led_intensity:.0%}")

        action_str = ", ".join(parts) if parts else "minimal actuator activity"

        # Find dominant reason
        critical = [e for e in entries if e.severity == "critical"]
        if critical:
            reason = critical[0].summary.split(";")[0]
            return f"{action_str} — driven by: {reason}"
        warnings = [e for e in entries if e.severity == "warning"]
        if warnings:
            reason = warnings[0].summary.split(";")[0]
            return f"{action_str} — responding to: {reason}"
        return action_str


def fused_stage_label(sp: Any) -> str:
    """Helper to get a display label for the setpoint's context."""
    # sp is a StageSetpoint — we just use a safe generic label
    return "stage-safe"
