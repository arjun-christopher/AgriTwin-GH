"""
Digital Twin transition engine.

``DigitalTwinEngine`` is the detailed, commentary-rich wrapper around
``GreenhouseTransitionModel`` that provides:

*   Explicit **per-actuator effect tracing** — for any state variable you can
    see exactly how much the heater, fan, vent, fogger, etc. contributed.
*   **Disease-aware environmental diagnostics** — boolean flags that signal
    when the post-step environment is favourable for pathogen growth.
*   **Resource accounting** — energy (kWh) and water (litres) per step.
*   **Setpoint-error tracking** — signed deviation from growth-stage targets.
*   **Physical-bounds detection** — which variables hit the clamp limits.

The engine does **NOT** replace or duplicate the ARX model in
``greenhouse_model.py``.  It delegates the core physics to
``GreenhouseTransitionModel.step()`` and then adds DT-specific diagnostics
on top.

Usage
-----
>>> engine = DigitalTwinEngine()
>>> result = engine.step(step_input)
>>> print(result.diagnostics.effect_attribution["indoor_temp"])
{'natural_decay': -2.0, 'weather_exchange': 0.16, 'solar_heating': 1.5, ...}
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .constants import DT_MINUTES, compute_disease_risk_score
from .dt_state import DTDiagnostics, DTSnapshot, DTStepInput, DTStepOutput
from .greenhouse_model import (
    CO2_BOUNDS,
    HUM_BOUNDS,
    LIGHT_BOUNDS,
    SM_BOUNDS,
    TEMP_BOUNDS,
    GreenhouseModelParams,
    GreenhouseTransitionModel,
)
from .setpoints import StageSetpoint, get_setpoint
from .state import ActuatorState, GreenhouseState, WeatherState

logger = logging.getLogger(__name__)


# ── Rated power per actuator channel (kW) ────────────────────────────────────
# Mirrors ``MPCRunner._estimate_energy`` in runner.py.  These are *peak*
# power draws — energy is ``power × duty × step_hours``.

_RATED_POWER_KW: dict[str, float] = {
    "fan_speed": 0.75,         # circulation / extraction fan
    "vent_opening": 0.05,      # motorised vent actuator
    "heater_output": 5.0,      # gas / electric heater
    "led_intensity": 1.2,      # supplemental LED bars
    "fogger_duty": 0.3,        # high-pressure fogger
    "co2_valve_pct": 0.1,      # CO₂ dosing valve
    "irrigation_qty": 0.01,    # drip-line pump (marginal)
}

# Water consumption coefficients
_WATER_PER_IRRIGATION_UNIT: float = 1.0   # litres per unit of irrigation_qty
_WATER_PER_FOGGER_STEP: float = 2.0       # litres per step at full duty

# ── Setpoint field mapping ───────────────────────────────────────────────────
# (GreenhouseState field, StageSetpoint field)

_SETPOINT_FIELD_MAP: list[tuple[str, str]] = [
    ("indoor_temp", "temp"),
    ("indoor_humidity", "humidity"),
    ("soil_moisture", "soil_moisture"),
    ("co2", "co2"),
    ("light_intensity", "light"),
    ("vpd", "vpd"),
]

# ── Disease-environment thresholds ───────────────────────────────────────────
# Thresholds for the boolean disease-environment flags.  These come from
# agronomic literature for Solanum lycopersicum foliar pathogens.

_HIGH_HUMIDITY_THRESH: float = 80.0       # %RH above which fungal risk rises
_HIGH_LEAF_WETNESS_THRESH: float = 0.5    # leaf-wetness proxy [0-1]
_DISEASE_TEMP_LO: float = 18.0            # °C — lower bound of fungal band
_DISEASE_TEMP_HI: float = 25.0            # °C — upper bound of fungal band
_FOGGER_RISK_THRESH: float = 0.45         # disease risk score threshold


# ── Digital Twin engine ──────────────────────────────────────────────────────


class DigitalTwinEngine:
    """Core DT transition engine with structured helpers and effect tracing.

    Wraps ``GreenhouseTransitionModel`` to provide a one-call ``step()``
    that returns:

    * the predicted next ``GreenhouseState``,
    * a ``DTDiagnostics`` bundle containing:
        - per-actuator **effect attribution** (which actuator contributed
          how much to each state-variable change),
        - **disease-environment flags** (boolean conditions favourable
          for pathogen growth),
        - resource usage (energy, water),
        - state delta, setpoint error, bounds clamped, step timing.

    Parameters
    ----------
    model_params:
        Tuneable ARX coefficients.  ``None`` → ``GreenhouseModelParams()``
        defaults.
    dt_minutes:
        Step duration in minutes (default ``DT_MINUTES`` = 5).
    """

    def __init__(
        self,
        model_params: GreenhouseModelParams | None = None,
        dt_minutes: int = DT_MINUTES,
    ) -> None:
        self._model = GreenhouseTransitionModel(
            params=model_params,
            dt_minutes=dt_minutes,
        )
        self._dt_minutes = dt_minutes

    # ── Public API ────────────────────────────────────────────────────

    def step(self, step_input: DTStepInput) -> DTStepOutput:
        """Simulate one timestep forward.

        Workflow
        -------
        1. Delegate climate update to the ARX model
           (``GreenhouseTransitionModel.step``).
        2. Recompute disease-risk score from the *new* environment.
        3. Decompose the state change into per-actuator effect attributions.
        4. Assess disease-favourable environmental conditions.
        5. Compute resource usage, state delta, setpoint error,
           bounds clamping.
        6. Pack everything into ``DTStepOutput``.

        Parameters
        ----------
        step_input:
            Complete input payload for this step.

        Returns
        -------
        DTStepOutput
        """
        t0 = time.perf_counter()

        current = step_input.current_state
        action = step_input.action
        weather = step_input.weather

        # ── 1. Climate update (core ARX physics) ────────────────────
        # The ARX model computes: temp, humidity, soil moisture, CO₂,
        # light, VPD, leaf-wetness proxy — then clamps to physical
        # bounds.  We do NOT duplicate that logic here.
        next_state = self._update_climate(current, action, weather)

        # Carry forward growth-stage index (updated by the image
        # classification pipeline, not by the plant model).
        next_state.growth_stage_index = current.growth_stage_index
        next_state.timestamp = step_input.timestamp

        # ── 2. Recompute disease risk from post-step environment ────
        # Uses humidity, leaf wetness, temperature, and growth stage
        # through sigmoid sub-scores.
        disease_risk = self._recompute_disease_risk(
            next_state, step_input.growth_stage,
        )
        next_state.disease_risk_score = disease_risk

        # ── 3. Per-actuator effect attribution ──────────────────────
        # Decomposes the delta for each state variable into named
        # contributions (actuators + natural processes + disturbances).
        effect_attribution = self._compute_effect_attribution(
            current, action, weather,
        )

        # ── 4. Disease-environment flags ────────────────────────────
        # Boolean flags for conditions known to favour tomato foliar
        # pathogens (powdery mildew, late blight, leaf mold, etc.).
        disease_flags = self._assess_disease_environment(
            next_state, action, step_input.growth_stage, disease_risk,
        )

        # ── 5. Resource usage ───────────────────────────────────────
        energy_kwh, water_litres = self._compute_resource_usage(action)

        # ── 6. State delta ──────────────────────────────────────────
        state_delta = self._compute_state_delta(current, next_state)

        # ── 7. Bounds clamping detection ────────────────────────────
        bounds_clamped = self._detect_bounds_clamped(next_state)

        # ── 8. Setpoint error ───────────────────────────────────────
        setpoint_error = self._compute_setpoint_error(
            next_state, step_input.growth_stage,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # ── Pack diagnostics ────────────────────────────────────────
        diagnostics = DTDiagnostics(
            energy_kwh=round(energy_kwh, 6),
            water_litres=round(water_litres, 4),
            disease_risk_recomputed=round(disease_risk, 4),
            state_delta=state_delta,
            bounds_clamped=bounds_clamped,
            setpoint_error=setpoint_error,
            effect_attribution=effect_attribution,
            disease_environment_flags=disease_flags,
            step_compute_ms=round(elapsed_ms, 3),
        )

        # ── Pack snapshot ───────────────────────────────────────────
        snapshot = DTSnapshot(
            greenhouse_state=next_state,
            growth_stage=step_input.growth_stage,
            disease_classification=step_input.disease_classification,
            disease_severity=dict(step_input.disease_severity),
            actuators_applied=action,
            weather=weather,
            timestamp=step_input.timestamp,
            step_index=step_input.step_index,
        )

        return DTStepOutput(
            next_state=next_state,
            diagnostics=diagnostics,
            snapshot=snapshot,
        )

    def simulate(
        self,
        step_inputs: list[DTStepInput],
    ) -> list[DTStepOutput]:
        """Run multiple steps sequentially, chaining state forward.

        ``step_inputs[0].current_state`` is the initial state.  Each
        subsequent input's ``current_state`` is **overwritten** with the
        previous step's ``next_state``.

        Parameters
        ----------
        step_inputs:
            Ordered step inputs.  Only the first needs a populated
            ``current_state``; the rest are filled automatically.

        Returns
        -------
        list[DTStepOutput]
            One output per input step.
        """
        if not step_inputs:
            return []

        outputs: list[DTStepOutput] = []
        state = step_inputs[0].current_state

        for inp in step_inputs:
            inp.current_state = state
            out = self.step(inp)
            outputs.append(out)
            state = out.next_state

        return outputs

    # ── Properties ────────────────────────────────────────────────────

    @property
    def model(self) -> GreenhouseTransitionModel:
        """Access the underlying ARX transition model."""
        return self._model

    @property
    def dt_minutes(self) -> int:
        """Step duration in minutes."""
        return self._dt_minutes

    # ══════════════════════════════════════════════════════════════════
    #  Private helper methods
    # ══════════════════════════════════════════════════════════════════

    # ── 1. Climate update ─────────────────────────────────────────────

    def _update_climate(
        self,
        state: GreenhouseState,
        action: ActuatorState,
        weather: WeatherState,
    ) -> GreenhouseState:
        """Delegate the core physics step to ``GreenhouseTransitionModel``.

        This is intentionally a thin wrapper — the ARX model already
        handles temperature, humidity, soil moisture, CO₂, light, VPD,
        and leaf-wetness computation with physical clamping.
        """
        return self._model.step(state, action, weather)

    # ── 2. Disease risk recomputation ─────────────────────────────────

    def _recompute_disease_risk(
        self,
        state: GreenhouseState,
        growth_stage: str,
    ) -> float:
        """Recompute disease risk from the *post-step* environment.

        Uses the sigmoid-based scorer from ``constants.py`` which combines:

        * Humidity risk   — sigmoid(RH, 75%, k=0.2)       weight 0.40
        * Leaf-wetness    — sigmoid(LW, 0.5, k=8.0)       weight 0.35
        * Temperature     — sigmoid(T, 22°C, k=0.15)      weight 0.25
        * Stage modifier  — ×1.2 for flowering/unripe,
                            ×1.1 for ripe

        Returns a score in [0, 1].
        """
        return compute_disease_risk_score(
            temp=state.indoor_temp,
            humidity=state.indoor_humidity,
            leaf_wetness=state.leaf_wetness_proxy,
            growth_stage=growth_stage,
        )

    # ── 3. Per-actuator effect attribution ────────────────────────────

    def _compute_effect_attribution(
        self,
        state: GreenhouseState,
        action: ActuatorState,
        weather: WeatherState,
    ) -> dict[str, dict[str, float]]:
        """Decompose the state change into per-actuator / disturbance terms.

        For each state variable we compute the **linear contribution** of
        every actuator and natural process using the identical coefficients
        as the ARX model.  These contributions sum to the *pre-clamping*
        delta; the actual (post-clamping) delta may differ where bounds
        are active.

        Returns
        -------
        dict[str, dict[str, float]]
            Outer key = state variable name (``indoor_temp``, …).
            Inner key = effect source (``heater``, ``natural_decay``, …).
            Value     = estimated contribution to Δ (signed, rounded).

        Notes
        -----
        The decomposition follows the ARX model equations in
        ``greenhouse_model.py`` **exactly**.  If the model coefficients
        change, these attributions stay in sync because we read from
        ``self._model.params``.
        """
        p = self._model.params
        t_ext = weather.temp_external
        rh_ext = weather.humidity_external
        solar = weather.solar_radiation

        # ── Temperature contributions ─────────────────────────────
        # next_T = decay·T + ext_gain·(T_ext - T) + solar_gain·solar
        #        + heater_gain·heater + fan_cool·fan + vent_cool·vent
        # ΔT    = (decay - 1)·T + ext_gain·(T_ext − T) + …
        temp_attr: dict[str, float] = {
            # Natural thermal-mass decay toward ambient.
            # (decay < 1) ⇒ always a slight cooling toward zero.
            "natural_decay": (p.temp_decay - 1.0) * state.indoor_temp,
            # Heat exchange with outdoor air through the envelope.
            "weather_exchange": p.temp_external_gain * (
                t_ext - state.indoor_temp
            ),
            # Solar radiation entering through glazing.
            "solar_heating": p.temp_solar_gain * solar,
            # Active heater: major warming actuator.
            "heater": p.temp_heater_gain * action.heater_output,
            # Fan: evaporative + convective cooling (negative coeff).
            "fan_cooling": p.temp_fan_cool * action.fan_speed,
            # Vent: passive cooling via outside air (negative coeff).
            "vent_cooling": p.temp_vent_cool * action.vent_opening,
        }

        # ── Humidity contributions ────────────────────────────────
        # next_RH = decay·RH + ext_gain·(RH_ext − RH) + fogger_gain·fog
        #         + fan_loss·fan + vent_loss·vent + evapotranspiration
        # ΔRH    = (decay − 1)·RH + …
        hum_attr: dict[str, float] = {
            # Natural RH drift toward equilibrium.
            "natural_decay": (p.hum_decay - 1.0) * state.indoor_humidity,
            # Moisture exchange with outdoor air.
            "weather_exchange": p.hum_external_gain * (
                rh_ext - state.indoor_humidity
            ),
            # Fogger: primary humidification actuator (positive).
            "fogger": p.hum_fogger_gain * action.fogger_duty,
            # Fan: removes moisture via exhaust / enhanced evap
            # (negative coeff → drying).
            "fan_drying": p.hum_fan_loss * action.fan_speed,
            # Vent: drying via ventilation (negative coeff).
            "vent_drying": p.hum_vent_loss * action.vent_opening,
            # Plant evapotranspiration: slow constant humidification.
            "evapotranspiration": p.hum_evapotranspiration,
        }

        # ── Soil-moisture contributions ───────────────────────────
        # next_SM = decay·SM + irrig_gain·irrig
        #         − evapotranspiration_loss·max(T_next − 15, 0)
        # For the temperature used in evapotranspiration we approximate
        # with the pre-clamped next_temp from the temp attribution.
        approx_next_temp = state.indoor_temp + sum(temp_attr.values())
        sm_attr: dict[str, float] = {
            # Very slow natural drying (decay ≈ 0.998).
            "natural_drying": (p.sm_decay - 1.0) * state.soil_moisture,
            # Drip irrigation: direct water addition.
            "irrigation": p.sm_irrigation_gain * action.irrigation_qty,
            # Evapotranspiration loss: temperature-dependent.
            "evapotranspiration": (
                -p.sm_evapotranspiration_loss
                * max(approx_next_temp - 15.0, 0.0)
            ),
        }

        # ── CO₂ contributions ─────────────────────────────────────
        # next_CO2 = decay·CO2 + inj_gain·valve + uptake·light_factor
        #          + vent_loss·vent + ext_gain·vent·(ambient − CO2)
        # light_factor ∈ [0, 1] scales photosynthetic uptake with PAR.
        light_factor = max(0.0, min(1.0, state.light_intensity / 500.0))
        co2_attr: dict[str, float] = {
            # Natural CO₂ drift (slow leak / respiration balance).
            "natural_decay": (p.co2_decay - 1.0) * state.co2,
            # Supplemental CO₂ dosing.
            "injection": p.co2_injection_gain * action.co2_valve_pct,
            # Photosynthetic uptake (negative: plants consume CO₂).
            # Scales linearly with light intensity (PAR proxy).
            "plant_uptake": p.co2_plant_uptake * light_factor,
            # Ventilation flushes CO₂ to outside (negative coeff).
            "vent_loss": p.co2_vent_loss * action.vent_opening,
            # Drift toward outdoor 420 ppm when vent is open.
            "ambient_drift": (
                p.temp_external_gain
                * action.vent_opening
                * (p.co2_ambient - state.co2)
            ),
        }

        # ── Light contributions ───────────────────────────────────
        # Light is *memoryless* — it depends only on current inputs,
        # not on previous light.  The "delta" framing is less meaningful
        # here, but we still decompose for consistency.
        light_attr: dict[str, float] = {
            # Natural sunlight through the glazing.
            "solar": p.light_solar_fraction * solar,
            # Supplemental LED lighting.
            "led": p.light_led_gain * action.led_intensity,
            # Subtract current light to express as a delta.
            "previous_offset": -state.light_intensity,
        }

        # Round everything for readability.
        def _round_dict(d: dict[str, float]) -> dict[str, float]:
            return {k: round(v, 4) for k, v in d.items()}

        return {
            "indoor_temp": _round_dict(temp_attr),
            "indoor_humidity": _round_dict(hum_attr),
            "soil_moisture": _round_dict(sm_attr),
            "co2": _round_dict(co2_attr),
            "light_intensity": _round_dict(light_attr),
        }

    # ── 4. Disease-environment assessment ─────────────────────────────

    def _assess_disease_environment(
        self,
        state: GreenhouseState,
        action: ActuatorState,
        growth_stage: str,
        disease_risk: float,
    ) -> dict[str, bool]:
        """Flag disease-favourable environmental conditions.

        Each flag is an independent boolean derived from the *post-step*
        state.  They are **diagnostic only** — they do not feed back into
        the model.  Downstream code (MPC cost function, dashboard alerts)
        can use them to trigger warnings or constraint tightening.

        Flags
        -----
        high_humidity_risk
            Relative humidity exceeds {_HIGH_HUMIDITY_THRESH}% RH.
            Most foliar fungal pathogens (powdery mildew, leaf mold)
            require sustained high humidity.

        high_leaf_wetness
            Leaf-wetness proxy exceeds {_HIGH_LEAF_WETNESS_THRESH}.
            Free moisture on leaf surfaces is the primary infection
            requirement for late blight (*Phytophthora infestans*).

        disease_temp_band
            Temperature is in the [{_DISEASE_TEMP_LO}, {_DISEASE_TEMP_HI}]°C
            band AND humidity is high.  This combination is the classic
            "disease triangle" for tomato fungal pathogens.

        fogger_disease_concern
            The fogger was active during this step AND the disease risk
            score exceeds the threshold.  Fogging raises humidity and
            leaf wetness, which can accelerate disease progression when
            risk is already elevated.
        """
        return {
            # High humidity alone raises spore germination probability.
            "high_humidity_risk": (
                state.indoor_humidity > _HIGH_HUMIDITY_THRESH
            ),
            # Leaf wetness → direct infection pathway.
            "high_leaf_wetness": (
                state.leaf_wetness_proxy > _HIGH_LEAF_WETNESS_THRESH
            ),
            # Joint temperature + humidity condition ("disease triangle").
            "disease_temp_band": (
                _DISEASE_TEMP_LO <= state.indoor_temp <= _DISEASE_TEMP_HI
                and state.indoor_humidity > _HIGH_HUMIDITY_THRESH
            ),
            # Fogger is actively making the disease environment worse.
            "fogger_disease_concern": (
                action.fogger_duty > 0.0
                and disease_risk > _FOGGER_RISK_THRESH
            ),
        }

    # ── 5. Resource usage ─────────────────────────────────────────────

    def _compute_resource_usage(
        self,
        action: ActuatorState,
    ) -> tuple[float, float]:
        """Compute energy (kWh) and water (litres) for this step.

        Energy is estimated as ``Σ (rated_power × duty) × step_hours``.
        This is the same linear model used by ``MPCRunner._estimate_energy``.

        Water accounts for irrigation and fogging:
        * Drip irrigation contributes ``irrigation_qty × 1.0 L/unit``.
        * Fogging contributes ``fogger_duty × 2.0 L/step``.
        """
        dt_hours = self._dt_minutes / 60.0

        # Energy: sum of (duty × rated_power) over all actuator channels.
        energy_kwh = sum(
            getattr(action, channel, 0.0) * power_kw
            for channel, power_kw in _RATED_POWER_KW.items()
        ) * dt_hours

        # Water: irrigation + fogging.
        water_litres = (
            action.irrigation_qty * _WATER_PER_IRRIGATION_UNIT
            + action.fogger_duty * _WATER_PER_FOGGER_STEP
        )

        return energy_kwh, water_litres

    # ── 6. State delta ────────────────────────────────────────────────

    @staticmethod
    def _compute_state_delta(
        current: GreenhouseState,
        next_state: GreenhouseState,
    ) -> dict[str, float]:
        """Per-variable state change (next − current).

        Covers the five primary controlled variables plus the two derived
        quantities (VPD, leaf wetness).
        """
        return {
            "indoor_temp": round(
                next_state.indoor_temp - current.indoor_temp, 4,
            ),
            "indoor_humidity": round(
                next_state.indoor_humidity - current.indoor_humidity, 4,
            ),
            "soil_moisture": round(
                next_state.soil_moisture - current.soil_moisture, 4,
            ),
            "co2": round(next_state.co2 - current.co2, 4),
            "light_intensity": round(
                next_state.light_intensity - current.light_intensity, 4,
            ),
            "vpd": round(next_state.vpd - current.vpd, 4),
            "leaf_wetness_proxy": round(
                next_state.leaf_wetness_proxy
                - current.leaf_wetness_proxy,
                4,
            ),
        }

    # ── 7. Setpoint error ─────────────────────────────────────────────

    @staticmethod
    def _compute_setpoint_error(
        state: GreenhouseState,
        growth_stage: str,
    ) -> dict[str, float]:
        """Signed error (actual − setpoint) for key variables.

        A positive error means the variable is *above* its target;
        negative means *below*.  Returns an empty dict if the growth
        stage has no registered setpoint.
        """
        try:
            sp: StageSetpoint = get_setpoint(growth_stage)
        except (KeyError, ValueError):
            return {}

        errors: dict[str, float] = {}
        for state_field, sp_field in _SETPOINT_FIELD_MAP:
            sp_val = getattr(sp, sp_field, None)
            if sp_val is not None:
                state_val = getattr(state, state_field, 0.0)
                errors[state_field] = round(
                    float(state_val) - float(sp_val), 4,
                )
        return errors

    # ── 8. Bounds clamping detection ──────────────────────────────────

    @staticmethod
    def _detect_bounds_clamped(state: GreenhouseState) -> list[str]:
        """Return names of variables sitting at (or beyond) physical bounds.

        Hitting a bound means the model tried to push the variable further
        but it was clamped — useful for diagnosing control saturation.
        """
        clamped: list[str] = []
        checks: list[tuple[str, float, tuple[float, float]]] = [
            ("indoor_temp", state.indoor_temp, TEMP_BOUNDS),
            ("indoor_humidity", state.indoor_humidity, HUM_BOUNDS),
            ("soil_moisture", state.soil_moisture, SM_BOUNDS),
            ("co2", state.co2, CO2_BOUNDS),
            ("light_intensity", state.light_intensity, LIGHT_BOUNDS),
        ]
        for name, val, (lo, hi) in checks:
            if val <= lo or val >= hi:
                clamped.append(name)
        return clamped
