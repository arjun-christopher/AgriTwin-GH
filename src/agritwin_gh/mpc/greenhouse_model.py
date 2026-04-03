"""
Simple first-order greenhouse transition model.

Estimates the *next* indoor-climate state from (current state, actuator
action, external weather disturbance) using an ARX-like discrete update.

The model is intentionally transparent: every coefficient has a physical
interpretation so it can be validated against domain knowledge and swapped
out for a learned model later.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .constants import (
    DT_MINUTES,
    compute_dew_point,
    compute_leaf_wetness_proxy,
    compute_vpd,
)
from .state import ActuatorState, GreenhouseState, WeatherState

# ── Physical state bounds (clamp after each step) ────────────────────────────
TEMP_BOUNDS = (5.0, 45.0)
HUM_BOUNDS = (20.0, 99.0)
SM_BOUNDS = (0.0, 100.0)
CO2_BOUNDS = (200.0, 3000.0)
LIGHT_BOUNDS = (0.0, 1500.0)


# ── Model parameters ──────────────────────────────────────────────────────────


@dataclass
class GreenhouseModelParams:
    """Tuneable coefficients for the first-order transition model.

    Defaults are physically sensible for a ~200 m² passive greenhouse
    (aligned with the AgriTwin indoor-greenhouse dataset).
    """

    # Temperature dynamics (°C per step)
    temp_external_gain: float = 0.08      # heat exchange with outside
    temp_solar_gain: float = 0.005        # solar radiation heating
    temp_heater_gain: float = 2.0         # heater at full power
    temp_fan_cool: float = -1.5           # evaporative + convective cooling
    temp_vent_cool: float = -1.2          # ventilation cooling
    temp_decay: float = 0.92              # self-decay (thermal mass)

    # Humidity dynamics (%RH per step)
    hum_external_gain: float = 0.05       # leakage toward outdoor RH
    hum_fogger_gain: float = 8.0          # fogger at full duty
    hum_fan_loss: float = -3.0            # fan-driven evaporation / exhaust
    hum_vent_loss: float = -2.5           # ventilation
    hum_evapotranspiration: float = 0.3   # plant transpiration (slow)
    hum_decay: float = 0.95

    # Soil moisture dynamics (% per step)
    sm_irrigation_gain: float = 0.5       # litres → % (depends on bed volume)
    sm_evapotranspiration_loss: float = 0.02
    sm_decay: float = 0.998               # very slow natural drying

    # CO2 dynamics (ppm per step)
    co2_injection_gain: float = 300.0     # valve at full open
    co2_plant_uptake: float = -5.0        # photosynthetic uptake (light dep.)
    co2_vent_loss: float = -40.0          # CO2 flush via ventilation
    co2_decay: float = 0.995
    co2_ambient: float = 420.0            # outdoor baseline

    # Light dynamics (W/m² per step)
    light_solar_fraction: float = 0.6     # transmittance of glazing
    light_led_gain: float = 400.0         # LED at full intensity

    # Process noise std-dev (optional, for stochastic rollout)
    noise_temp: float = 0.0
    noise_humidity: float = 0.0
    noise_soil_moisture: float = 0.0
    noise_co2: float = 0.0

    # Actuator response lag — first-order time constants (minutes).
    # 0 = instantaneous (legacy behaviour).  Positive values introduce a
    # first-order lag: effective_output approaches commanded value with
    # alpha = 1 - exp(-dt / tau).
    lag_fan_minutes: float = 0.0          # fan spool-up / spool-down
    lag_vent_minutes: float = 0.0         # vent motor opening/closing
    lag_heater_minutes: float = 0.0       # heating element warm-up
    lag_fogger_minutes: float = 0.0       # fogger pressure build-up
    lag_co2_valve_minutes: float = 0.0    # CO2 valve actuator travel
    lag_led_minutes: float = 0.0          # LED driver ramp (usually fast)
    lag_irrigation_minutes: float = 0.0   # pump start / valve open

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)


# ── Transition model ──────────────────────────────────────────────────────────


class GreenhouseTransitionModel:
    """Discrete-time first-order transition model.

    Usage
    -----
    >>> model = GreenhouseTransitionModel()
    >>> next_state = model.step(current_state, actuators, weather)
    >>> trajectory = model.simulate(state, actuator_sequence, weather_sequence)

    The model is **deterministic** by default (noise std = 0).  Set the
    ``noise_*`` parameters in ``GreenhouseModelParams`` for stochastic
    roll-outs (e.g. during MPC scenario evaluation).

    Actuator response lag
    ---------------------
    When any ``lag_*_minutes`` parameter is positive, the model applies a
    first-order lag filter to the corresponding actuator output before using
    it in the physics equations.  The effective output approaches the
    commanded value with ``alpha = 1 - exp(-dt / tau)`` each 5-minute step.

    Call ``reset_actuator_state()`` between independent simulations to clear
    the internal effective-actuator memory.
    """

    def __init__(
        self,
        params: GreenhouseModelParams | None = None,
        dt_minutes: int = DT_MINUTES,
    ) -> None:
        self.params = params or GreenhouseModelParams()
        self.dt_minutes = dt_minutes
        self._rng = np.random.default_rng()
        # Effective actuator outputs (for lag filter).  None = uninitialised.
        self._effective_actuators: dict[str, float] | None = None

    def reset_actuator_state(self) -> None:
        """Clear stored effective-actuator memory (call between runs)."""
        self._effective_actuators = None

    def _apply_actuator_lag(self, actuators: ActuatorState) -> ActuatorState:
        """Apply first-order response lag to each actuator channel.

        Returns a *new* ``ActuatorState`` with effective (lagged) values.
        If all lag parameters are zero, returns the original unchanged.
        """
        p = self.params
        lag_map = {
            "fan_speed":      p.lag_fan_minutes,
            "vent_opening":   p.lag_vent_minutes,
            "heater_output":  p.lag_heater_minutes,
            "fogger_duty":    p.lag_fogger_minutes,
            "co2_valve_pct":  p.lag_co2_valve_minutes,
            "led_intensity":  p.lag_led_minutes,
            "irrigation_qty": p.lag_irrigation_minutes,
        }

        # Fast path: no lag configured at all.
        if all(tau <= 0.0 for tau in lag_map.values()):
            return actuators

        # Initialise effective state on first call.
        if self._effective_actuators is None:
            self._effective_actuators = {
                fld: getattr(actuators, fld, 0.0) or 0.0
                for fld in lag_map
            }

        dt = float(self.dt_minutes)
        effective = {}
        for fld, tau in lag_map.items():
            cmd = getattr(actuators, fld, 0.0) or 0.0
            prev = self._effective_actuators.get(fld, cmd)
            if tau > 0.0:
                alpha = 1.0 - math.exp(-dt / tau)
                eff = prev + alpha * (cmd - prev)
            else:
                eff = cmd
            effective[fld] = eff

        self._effective_actuators = effective

        return ActuatorState(**effective)

    # ── Single-step update ────────────────────────────────────────────

    def step(
        self,
        state: GreenhouseState,
        actuators: ActuatorState,
        weather: WeatherState | dict[str, Any],
    ) -> GreenhouseState:
        """Compute the next ``GreenhouseState`` from current inputs.

        Parameters
        ----------
        state : GreenhouseState
            Current indoor-climate state.
        actuators : ActuatorState
            Control action to apply during this step.
        weather : WeatherState or dict
            External weather at this timestep.  If a ``dict``, expects keys
            ``temp_external``, ``humidity_external``, ``solar_radiation``.
        """
        p = self.params

        # ── Apply actuator response lag (if configured) ───────────────
        actuators = self._apply_actuator_lag(actuators)

        # Unpack weather (accept both WeatherState and dict)
        if isinstance(weather, dict):
            t_ext = float(weather.get("temp_external", state.indoor_temp))
            rh_ext = float(weather.get("humidity_external", state.indoor_humidity))
            solar = float(weather.get("solar_radiation", 0.0))
        else:
            t_ext = weather.temp_external
            rh_ext = weather.humidity_external
            solar = weather.solar_radiation

        # ── Temperature ───────────────────────────────────────────────
        temp_next = (
            p.temp_decay * state.indoor_temp
            + p.temp_external_gain * (t_ext - state.indoor_temp)
            + p.temp_solar_gain * solar
            + p.temp_heater_gain * actuators.heater_output
            + p.temp_fan_cool * actuators.fan_speed
            + p.temp_vent_cool * actuators.vent_opening
        )
        temp_next += self._noise(p.noise_temp)

        # ── Humidity ──────────────────────────────────────────────────
        hum_next = (
            p.hum_decay * state.indoor_humidity
            + p.hum_external_gain * (rh_ext - state.indoor_humidity)
            + p.hum_fogger_gain * actuators.fogger_duty
            + p.hum_fan_loss * actuators.fan_speed
            + p.hum_vent_loss * actuators.vent_opening
            + p.hum_evapotranspiration
        )
        hum_next += self._noise(p.noise_humidity)

        # ── Soil moisture ─────────────────────────────────────────────
        sm_next = (
            p.sm_decay * state.soil_moisture
            + p.sm_irrigation_gain * actuators.irrigation_qty
            - p.sm_evapotranspiration_loss * max(temp_next - 15.0, 0.0)
        )
        sm_next += self._noise(p.noise_soil_moisture)

        # ── CO2 ───────────────────────────────────────────────────────
        # Plant uptake scales with light (photosynthesis).
        light_factor = max(0.0, min(1.0, state.light_intensity / 500.0))
        co2_next = (
            p.co2_decay * state.co2
            + p.co2_injection_gain * actuators.co2_valve_pct
            + p.co2_plant_uptake * light_factor
            + p.co2_vent_loss * actuators.vent_opening
        )
        # Drift toward ambient when ventilating
        co2_next += p.temp_external_gain * actuators.vent_opening * (
            p.co2_ambient - state.co2
        )
        co2_next += self._noise(p.noise_co2)

        # ── Light ─────────────────────────────────────────────────────
        light_next = (
            p.light_solar_fraction * solar
            + p.light_led_gain * actuators.led_intensity
        )

        # ── Derived quantities ────────────────────────────────────────
        # Clamp to physical bounds before deriving VPD / leaf wetness.
        temp_next = max(TEMP_BOUNDS[0], min(TEMP_BOUNDS[1], temp_next))
        hum_next = max(HUM_BOUNDS[0], min(HUM_BOUNDS[1], hum_next))
        sm_next = max(SM_BOUNDS[0], min(SM_BOUNDS[1], sm_next))
        co2_next = max(CO2_BOUNDS[0], min(CO2_BOUNDS[1], co2_next))
        light_next = max(LIGHT_BOUNDS[0], min(LIGHT_BOUNDS[1], light_next))

        vpd_next = compute_vpd(temp_next, hum_next)
        dew_point = compute_dew_point(temp_next, hum_next)
        leaf_wet_next = compute_leaf_wetness_proxy(hum_next, temp_next, dew_point)

        return GreenhouseState(
            indoor_temp=round(temp_next, 2),
            indoor_humidity=round(hum_next, 2),
            soil_moisture=round(sm_next, 2),
            co2=round(co2_next, 1),
            light_intensity=round(light_next, 1),
            disease_risk_score=state.disease_risk_score,   # updated externally
            growth_stage_index=state.growth_stage_index,   # updated externally
            vpd=round(vpd_next, 4),
            leaf_wetness_proxy=round(leaf_wet_next, 4),
            timestamp=state.timestamp,
        )

    # ── Multi-step rollout ────────────────────────────────────────────

    def simulate(
        self,
        initial_state: GreenhouseState,
        actuator_sequence: list[ActuatorState],
        weather_sequence: list[WeatherState | dict[str, Any]],
    ) -> list[GreenhouseState]:
        """Roll the model forward for ``len(actuator_sequence)`` steps.

        Returns a list of predicted states (one per step, **not** including
        the initial state).
        """
        n = len(actuator_sequence)
        if len(weather_sequence) < n:
            raise ValueError(
                f"Weather sequence length ({len(weather_sequence)}) must be "
                f">= actuator sequence length ({n})."
            )
        # Reset lag state so each rollout starts from commanded values.
        saved = self._effective_actuators
        self._effective_actuators = None

        trajectory: list[GreenhouseState] = []
        state = initial_state
        for k in range(n):
            state = self.step(state, actuator_sequence[k], weather_sequence[k])
            trajectory.append(state)

        # Restore previous effective-actuator memory.
        self._effective_actuators = saved
        return trajectory

    # ── Placeholder for future system-identification ──────────────────

    def calibrate(self, df: Any) -> None:
        """Fit model parameters from historical data.

        Currently a no-op placeholder.  Will use least-squares regression
        on (state[k+1] − state[k]) given (actuators[k], weather[k]).
        """

    # ── Private ───────────────────────────────────────────────────────

    def _noise(self, std: float) -> float:
        if std <= 0.0:
            return 0.0
        return float(self._rng.normal(0.0, std))
