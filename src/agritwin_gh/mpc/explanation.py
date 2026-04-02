"""
explanation.py — lightweight controller-explanation builder.

Provides rule-based natural-language summaries of MPC decisions without
any SHAP / heavy ML dependency.  Suitable for real-time logging and
dashboard display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExplanationEntry:
    """A single explanation bullet describing one aspect of the MPC decision."""

    factor: str = ""
    description: str = ""
    impact: str = ""       # "high" | "medium" | "low"
    sentiment: str = ""    # "positive" | "negative" | "neutral"

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor":      self.factor,
            "description": self.description,
            "impact":      self.impact,
            "sentiment":   self.sentiment,
        }


@dataclass
class ControllerExplanation:
    """Structured explanation of one MPC control-step decision."""

    summary: str = ""
    entries: list[ExplanationEntry] = field(default_factory=list)
    solver_note: str = ""
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary":        self.summary,
            "entries":        [e.to_dict() for e in self.entries],
            "solver_note":    self.solver_note,
            "schema_version": self.schema_version,
        }


class ExplanationBuilder:
    """Builds a ``ControllerExplanation`` from fused state + MPC outputs.

    Uses rule-based heuristics — no external ML library required.
    """

    def build(
        self,
        fused: Any,
        actuators: Any,
        cost_breakdown: dict[str, float] | None = None,
        solver_converged: bool = True,
        weather_stress: dict[str, float] | None = None,
        tightened_constraints: dict[str, Any] | None = None,
    ) -> ControllerExplanation:
        """Generate a ``ControllerExplanation`` for one control step.

        Parameters
        ----------
        fused:
            ``FusedState`` assembled by ``StateFusion.fuse()``.
        actuators:
            ``ActuatorState`` with the MPC-computed commands.
        cost_breakdown:
            Per-component cost contributions.
        solver_converged:
            Whether the SLSQP solver converged.
        weather_stress:
            Weather-stress summary from ``WeatherAdaptiveModifiers``.
        tightened_constraints:
            Constraint-tightening details (disease-driven RH ceiling, etc.).
        """
        entries: list[ExplanationEntry] = []

        # ── Disease risk ────────────────────────────────────────────────
        risk = getattr(fused, "disease_risk_score", 0.0)
        if risk > 0.6:
            entries.append(ExplanationEntry(
                factor="Disease risk",
                description=f"High risk ({risk:.2f}) — MPC is suppressing fogger and irrigaton.",
                impact="high",
                sentiment="negative",
            ))
        elif risk > 0.35:
            entries.append(ExplanationEntry(
                factor="Disease risk",
                description=f"Moderate risk ({risk:.2f}) — RH ceiling tightened.",
                impact="medium",
                sentiment="negative",
            ))

        # ── Temperature control ──────────────────────────────────────────
        gh = getattr(fused, "greenhouse_state", None)
        if gh is not None:
            temp = getattr(gh, "indoor_temp", 25.0)
            if temp > 30.0:
                entries.append(ExplanationEntry(
                    factor="Temperature",
                    description=f"High temp ({temp:.1f}°C) — fan and vents increased.",
                    impact="medium",
                    sentiment="negative",
                ))
            elif temp < 16.0:
                entries.append(ExplanationEntry(
                    factor="Temperature",
                    description=f"Low temp ({temp:.1f}°C) — heater activated.",
                    impact="medium",
                    sentiment="negative",
                ))

        # ── Stage transition warning ─────────────────────────────────────
        if getattr(fused, "transition_within_24h", False):
            nxt = getattr(fused, "next_stage", "next stage")
            hrs = getattr(fused, "hours_to_transition", 0.0)
            entries.append(ExplanationEntry(
                factor="Stage transition",
                description=f"Transition to '{nxt}' expected in ~{hrs:.0f} h.",
                impact="medium",
                sentiment="neutral",
            ))

        # ── Weather stress ───────────────────────────────────────────────
        if weather_stress:
            temp_shift = weather_stress.get("temp_shift", 0.0)
            if abs(temp_shift) > 2.0:
                direction = "rise" if temp_shift > 0 else "drop"
                entries.append(ExplanationEntry(
                    factor="Weather forecast",
                    description=f"External temp forecast to {direction} "
                                f"{abs(temp_shift):.1f}°C — pre-emptive action taken.",
                    impact="medium",
                    sentiment="neutral",
                ))

        # ── Solver convergence ───────────────────────────────────────────
        solver_note = (
            "Solver converged — optimal trajectory found."
            if solver_converged
            else "Solver did not converge — baseline fallback applied."
        )

        # ── Summary ──────────────────────────────────────────────────────
        stage = getattr(fused, "growth_stage", "unknown")
        summary = (
            f"MPC control step for stage '{stage}'. "
            f"Disease risk: {risk:.2f}. "
            + (f"{len(entries)} active intervention(s) applied." if entries else
               "No critical conditions — operating within normal bounds.")
        )

        return ControllerExplanation(
            summary=summary,
            entries=entries,
            solver_note=solver_note,
        )
