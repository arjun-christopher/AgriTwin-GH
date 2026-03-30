"""
Rule-based baseline greenhouse controller.

Priority-ordered logic that produces an ``ActuatorState`` from the current
climate observation, growth stage, weather forecast, and disease risk.

The controller is intentionally simple so it can serve as:
  1. A **fallback** when the MPC solver fails.
  2. A **comparison benchmark** for evaluating MPC performance.
  3. A **starting-point** for warm-starting the optimiser.

Rule priority (highest → lowest):
  1. Disease risk  → reduce humidity (fan + vent), cut fogger.
  2. Temperature   → heater (cold) or fan + vent (hot).
  3. Humidity      → fan / vent / fogger.
  4. Soil moisture → irrigation.
  5. CO2           → CO2 valve (+ vent interaction).
  6. Light         → LED supplement.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from .constants import CONTROL_VARIABLES, DT_MINUTES
from .constraints import ConstraintSet, get_default_constraints
from .setpoints import (
    StageControlProfile,
    StageSetpoint,
    get_control_profile,
    get_setpoint,
)
from .state import ActuatorState, GreenhouseState, WeatherState


# ── Controller output payload ─────────────────────────────────────────────────


@dataclass
class BaselineControlPayload:
    """Structured output from the rule-based controller.

    Carries the decided actuator values together with the reasoning trace
    and resource accounting so the dashboard can display *why* the
    controller acted.
    """

    actuators: ActuatorState = field(default_factory=ActuatorState)
    reasoning: list[str] = field(default_factory=list)
    resource_energy_kwh: float = 0.0
    resource_water_litres: float = 0.0
    priority_triggered: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return {
            "actuators": asdict(self.actuators),
            "reasoning": list(self.reasoning),
            "resource_energy_kwh": self.resource_energy_kwh,
            "resource_water_litres": self.resource_water_litres,
            "priority_triggered": self.priority_triggered,
        }


# ── Energy / water coefficients ───────────────────────────────────────────────

_DEFAULT_ENERGY_COSTS: dict[str, float] = {
    "fan_speed": 0.15,          # kWh per step at full speed
    "vent_opening": 0.02,       # kWh per step (motor)
    "heater_output": 0.80,      # kWh per step at full power
    "led_intensity": 0.30,      # kWh per step at full intensity
    "co2_valve_pct": 0.05,      # kWh per step (solenoid + compressor)
    "fogger_duty": 0.10,        # kWh per step at full duty
    "irrigation_qty": 0.01,     # kWh per step (pump)
}

_WATER_PER_IRRIGATION_LITRE = 1.0   # litres per unit of irrigation_qty
_WATER_PER_FOGGER_STEP = 2.0        # litres per step at full duty


# ── Baseline controller ──────────────────────────────────────────────────────


class RuleBasedController:
    """Priority-based rule controller producing ``ActuatorState`` actions.

    Parameters
    ----------
    constraints : ConstraintSet or None
        Actuator box constraints and cooldowns.  Defaults are loaded
        automatically if ``None``.
    energy_costs : dict or None
        Per-actuator energy cost (kWh per step at full output).
    dt_minutes : int
        Timestep in minutes (used only for bookkeeping).
    """

    def __init__(
        self,
        constraints: ConstraintSet | None = None,
        energy_costs: dict[str, float] | None = None,
        dt_minutes: int = DT_MINUTES,
    ) -> None:
        self._constraints = constraints or get_default_constraints()
        self._energy_costs = energy_costs or dict(_DEFAULT_ENERGY_COSTS)
        self._dt = dt_minutes

        # Cooldown tracking: actuator_name → remaining cooldown steps
        self._cooldown_remaining: dict[str, int] = {
            name: 0 for name in CONTROL_VARIABLES
        }
        self._prev_actuators: ActuatorState | None = None

    # ── Public API ────────────────────────────────────────────────────

    def compute_action(
        self,
        state: GreenhouseState,
        growth_stage: str,
        disease_risk: float = 0.0,
        weather: WeatherState | dict[str, Any] | None = None,
    ) -> BaselineControlPayload:
        """Decide actuator settings for the current timestep.

        Parameters
        ----------
        state : GreenhouseState
            Current indoor-climate observation.
        growth_stage : str
            Canonical growth-stage label.
        disease_risk : float
            Overall disease risk score (0-1).
        weather : WeatherState or dict, optional
            External weather (used for anticipatory ventilation).
        """
        sp = get_setpoint(growth_stage)
        profile = get_control_profile(growth_stage)
        constraints = self._constraints

        # Start from zero (additive rule stacking)
        fan = 0.0
        vent = 0.0
        irrigation = 0.0
        heater = 0.0
        led = 0.0
        co2_valve = 0.0
        fogger = 0.0
        reasoning: list[str] = []
        priority = ""

        # ── 1. Disease risk response (highest priority) ──────────────
        if disease_risk > sp.disease_risk_max:
            excess = disease_risk - sp.disease_risk_max
            # Aggressiveness proportional to excess, scaled by sensitivity.
            severity_factor = min(1.0, excess * 3.0) * profile.disease_sensitivity
            fan = max(fan, 0.3 + 0.5 * severity_factor)
            vent = max(vent, 0.2 + 0.4 * severity_factor)
            fogger = 0.0   # suppress fogger to reduce humidity
            reasoning.append(
                f"Disease risk {disease_risk:.2f} > max {sp.disease_risk_max:.2f}: "
                f"fan={fan:.2f}, vent={vent:.2f}, fogger off"
            )
            if not priority:
                priority = "disease_risk"

        # ── 2. Temperature regulation ─────────────────────────────────
        temp_error = state.indoor_temp - sp.temp
        if temp_error > sp.temp_tol:
            # Too hot → fan + vent
            intensity = min(1.0, (temp_error - sp.temp_tol) / 5.0)
            fan = max(fan, 0.3 + 0.5 * intensity)
            vent = max(vent, 0.2 + 0.5 * intensity)
            reasoning.append(
                f"Temp {state.indoor_temp:.1f}°C > target {sp.temp:.1f}±{sp.temp_tol}: "
                f"fan={fan:.2f}, vent={vent:.2f}"
            )
            if not priority:
                priority = "temperature_high"
        elif temp_error < -sp.temp_tol:
            # Too cold → heater
            intensity = min(1.0, (-temp_error - sp.temp_tol) / 5.0)
            heater = max(heater, 0.2 + 0.6 * intensity)
            reasoning.append(
                f"Temp {state.indoor_temp:.1f}°C < target {sp.temp:.1f}±{sp.temp_tol}: "
                f"heater={heater:.2f}"
            )
            if not priority:
                priority = "temperature_low"

        # ── 3. Humidity regulation ────────────────────────────────────
        hum_error = state.indoor_humidity - sp.humidity
        if hum_error > sp.hum_tol:
            # Too humid → fan + vent
            intensity = min(1.0, (hum_error - sp.hum_tol) / 15.0)
            fan = max(fan, 0.2 + 0.4 * intensity)
            vent = max(vent, 0.15 + 0.35 * intensity)
            reasoning.append(
                f"RH {state.indoor_humidity:.1f}% > target {sp.humidity:.1f}±{sp.hum_tol}: "
                f"fan={fan:.2f}, vent={vent:.2f}"
            )
            if not priority:
                priority = "humidity_high"
        elif hum_error < -sp.hum_tol:
            # Too dry → fogger
            intensity = min(1.0, (-hum_error - sp.hum_tol) / 15.0)
            fogger = max(fogger, 0.2 + 0.5 * intensity)
            reasoning.append(
                f"RH {state.indoor_humidity:.1f}% < target {sp.humidity:.1f}±{sp.hum_tol}: "
                f"fogger={fogger:.2f}"
            )
            if not priority:
                priority = "humidity_low"

        # ── 4. Soil moisture regulation ───────────────────────────────
        sm_error = sp.soil_moisture - state.soil_moisture
        if sm_error > 5.0:
            # Under-watered → irrigate
            irrigation = min(20.0, sm_error * 0.5)
            reasoning.append(
                f"SM {state.soil_moisture:.1f}% < target {sp.soil_moisture:.1f}%: "
                f"irrigation={irrigation:.1f}"
            )
            if not priority:
                priority = "soil_moisture_low"

        # ── 5. CO2 regulation ─────────────────────────────────────────
        co2_error = sp.co2 - state.co2
        if co2_error > 50.0:
            # CO2 too low → inject
            intensity = min(1.0, co2_error / 300.0)
            co2_valve = max(co2_valve, 0.1 + 0.6 * intensity)
            reasoning.append(
                f"CO2 {state.co2:.0f} ppm < target {sp.co2:.0f}: "
                f"co2_valve={co2_valve:.2f}"
            )
            if not priority:
                priority = "co2_low"
        elif co2_error < -100.0:
            # CO2 too high → vent
            vent = max(vent, 0.15)
            reasoning.append(
                f"CO2 {state.co2:.0f} ppm > target {sp.co2:.0f}: "
                f"vent={vent:.2f}"
            )
            if not priority:
                priority = "co2_high"

        # ── 6. Light supplement ───────────────────────────────────────
        if state.light_intensity < sp.light * 0.6:
            gap_fraction = 1.0 - state.light_intensity / max(sp.light, 1.0)
            led = min(1.0, 0.2 + 0.6 * gap_fraction)
            reasoning.append(
                f"Light {state.light_intensity:.0f} W/m² < 60% of {sp.light:.0f}: "
                f"led={led:.2f}"
            )
            if not priority:
                priority = "light_low"

        if not reasoning:
            reasoning.append("All variables within tolerance — no action.")
            priority = "none"

        # ── Build ActuatorState and apply constraints ─────────────────
        raw = ActuatorState(
            fan_speed=fan,
            vent_opening=vent,
            irrigation_qty=irrigation,
            heater_output=heater,
            led_intensity=led,
            co2_valve_pct=co2_valve,
            fogger_duty=fogger,
        )
        clipped = raw.clip(constraints.actuator_bounds)
        clipped = self._apply_rate_limits(clipped)
        clipped = self._apply_cooldowns(clipped)
        self._prev_actuators = clipped

        # ── Resource accounting ───────────────────────────────────────
        energy = self._compute_energy(clipped)
        water = self._compute_water(clipped)

        return BaselineControlPayload(
            actuators=clipped,
            reasoning=reasoning,
            resource_energy_kwh=round(energy, 4),
            resource_water_litres=round(water, 2),
            priority_triggered=priority,
        )

    # ── Rate limiting ─────────────────────────────────────────────────

    def _apply_rate_limits(self, target: ActuatorState) -> ActuatorState:
        """Clamp per-step change to configured rate limits."""
        if self._prev_actuators is None:
            return target
        prev = self._prev_actuators
        limits = self._constraints.actuator_rate_limits
        vals: dict[str, float] = {}
        for name in CONTROL_VARIABLES:
            cur = getattr(target, name)
            prv = getattr(prev, name)
            delta = cur - prv
            lo, hi = limits.get(name, (-float("inf"), float("inf")))
            clamped_delta = max(lo, min(hi, delta))
            vals[name] = prv + clamped_delta
        return ActuatorState(**vals)

    def _apply_cooldowns(self, target: ActuatorState) -> ActuatorState:
        """Zero out actuators that are still in cooldown."""
        cooldown_cfg = self._constraints.actuator_cooldown_steps
        vals = {name: getattr(target, name) for name in CONTROL_VARIABLES}

        for name, steps in cooldown_cfg.items():
            if self._cooldown_remaining.get(name, 0) > 0:
                vals[name] = 0.0
                self._cooldown_remaining[name] -= 1
            elif self._prev_actuators is not None:
                # Start cooldown if actuator was just turned off
                prev_val = getattr(self._prev_actuators, name)
                if prev_val > 0.0 and vals[name] == 0.0:
                    self._cooldown_remaining[name] = steps

        return ActuatorState(**vals)

    # ── Resource tracking ─────────────────────────────────────────────

    def _compute_energy(self, act: ActuatorState) -> float:
        total = 0.0
        for name in CONTROL_VARIABLES:
            total += getattr(act, name) * self._energy_costs.get(name, 0.0)
        return total

    def _compute_water(self, act: ActuatorState) -> float:
        return (
            act.irrigation_qty * _WATER_PER_IRRIGATION_LITRE
            + act.fogger_duty * _WATER_PER_FOGGER_STEP
        )

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset internal state (cooldowns, previous actuators)."""
        self._cooldown_remaining = {name: 0 for name in CONTROL_VARIABLES}
        self._prev_actuators = None
