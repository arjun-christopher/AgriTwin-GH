"""
MPC cost-function components.

Five building blocks:

  • ``DiseaseContext``  — disease information from ``FusedState`` that shapes
    the objective (severity amplification, exposure penalties).
  • ``StageCost``       — per-timestep weighted quadratic tracking error with
    disease-aware, weather-adaptive terms.
  • ``TerminalCost``    — penalty on the final predicted state.
  • ``CostBuilder``     — assembles a composite :math:`J_{total}` with
    growth-stage transition blending and per-step weather modifiers.

All classes operate on NumPy arrays so they can be plugged directly into
a ``scipy.optimize``-based solver or a future CasADi formulation.

**Disease-aware terms** (new in Subprompt 6):

  • *Environment disease risk*: instead of penalising the (static)
    ``disease_risk_score`` state field, we **recompute** disease risk from
    the predicted humidity / temperature / leaf wetness at each step.
    This means the solver sees the cost of letting humidity climb.
  • *Humidity exposure*: extra penalty when RH exceeds the setpoint
    AND the predicted disease risk is elevated — discourages lingering
    above the setpoint.
  • *Fogger suppression*: extra cost on fogger duty when predicted
    disease risk exceeds a configurable threshold.
  • *Irrigation caution*: extra irrigation cost when humidity is
    already high and disease risk is elevated.
  • *Severity amplification*: the 24 h / 48 h disease severity forecast
    amplifies the base disease weight so the controller becomes more
    aggressive when an outbreak is predicted.

**Weather-adaptive weights** (new in Subprompt 6):

  ``StageCost.evaluate`` accepts an optional ``step_modifiers`` array
  (from ``weather_adaptation.compute_weather_adaptation``) that scales
  the tracking weights element-wise, making the controller pre-act
  against upcoming weather stress.

**Growth-stage transition blending** (new in Subprompt 6):

  ``CostBuilder`` can hold a second ``StageCost`` for the *next* growth
  stage.  When the transition is predicted within the horizon, the
  weight vector linearly ramps from the current to the next stage over
  a configurable number of steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MPCConfig
from .constraints import get_default_constraints
from .constants import CONTROL_VARIABLES, STATE_VARIABLES, sigmoid
from .setpoints import (
    StageControlProfile,
    StageSetpoint,
    get_control_profile,
    get_setpoint,
)
from .state import ActuatorState, GreenhouseState


# ── Helpers ───────────────────────────────────────────────────────────────────

# Indices into the state vector (mirrors STATE_VARIABLES order).
_STATE_IDX: dict[str, int] = {name: i for i, name in enumerate(STATE_VARIABLES)}
_CTRL_IDX: dict[str, int] = {name: i for i, name in enumerate(CONTROL_VARIABLES)}

# Mapping from cost-weight name → state variable for tracking terms.
_WEIGHT_TO_STATE: dict[str, str] = {
    "w_temp": "indoor_temp",
    "w_humidity": "indoor_humidity",
    "w_soil_moisture": "soil_moisture",
    "w_co2": "co2",
    "w_vpd": "vpd",
    "w_light": "light_intensity",
}


def _setpoint_vector(sp: StageSetpoint) -> np.ndarray:
    """Build a reference vector aligned with ``STATE_VARIABLES``."""
    ref = np.zeros(len(STATE_VARIABLES), dtype=np.float64)
    ref[_STATE_IDX["indoor_temp"]] = sp.temp
    ref[_STATE_IDX["indoor_humidity"]] = sp.humidity
    ref[_STATE_IDX["soil_moisture"]] = sp.soil_moisture
    ref[_STATE_IDX["co2"]] = sp.co2
    ref[_STATE_IDX["light_intensity"]] = sp.light
    ref[_STATE_IDX["vpd"]] = sp.vpd
    ref[_STATE_IDX["disease_risk_score"]] = 0.0  # target = zero risk
    return ref


def _normalisation_scales() -> np.ndarray:
    """Per-variable normalisation denominators to make quadratic terms
    dimensionally comparable (order ``STATE_VARIABLES``)."""
    scales = np.ones(len(STATE_VARIABLES), dtype=np.float64)
    scales[_STATE_IDX["indoor_temp"]] = 5.0         # °C
    scales[_STATE_IDX["indoor_humidity"]] = 10.0     # %RH
    scales[_STATE_IDX["soil_moisture"]] = 10.0       # %
    scales[_STATE_IDX["co2"]] = 150.0                # ppm
    scales[_STATE_IDX["light_intensity"]] = 200.0    # W/m²
    scales[_STATE_IDX["vpd"]] = 0.3                  # kPa
    scales[_STATE_IDX["disease_risk_score"]] = 0.3   # unitless
    scales[_STATE_IDX["leaf_wetness_proxy"]] = 0.3   # unitless
    scales[_STATE_IDX["growth_stage_index"]] = 1.0   # not penalised
    return scales


# ── Disease context ───────────────────────────────────────────────────────────


@dataclass
class DiseaseContext:
    """Disease information extracted from ``FusedState`` for cost tuning.

    This is a lightweight snapshot that the solver passes to the cost
    function at the start of each optimisation.  It does **not** change
    during the horizon rollout — its role is to *modulate* how
    aggressively the cost function penalises disease-related terms.
    """

    risk_score: float = 0.0
    classification: str = ""
    confidence: float = 0.0
    current_severity: dict[str, float] = field(default_factory=dict)
    severity_24h: dict[str, float] = field(default_factory=dict)
    severity_48h: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_fused(cls, fused: Any) -> "DiseaseContext":
        """Construct from a ``FusedState`` instance."""
        return cls(
            risk_score=getattr(fused, "disease_risk_score", 0.0),
            classification=getattr(fused, "disease_classification", ""),
            confidence=getattr(fused, "disease_confidence", 0.0),
            current_severity=getattr(fused, "current_severity", {}),
            severity_24h=getattr(fused, "severity_24h", {}),
            severity_48h=getattr(fused, "severity_48h", {}),
        )

    @property
    def max_severity_24h(self) -> float:
        return max(self.severity_24h.values(), default=0.0)

    @property
    def max_severity_48h(self) -> float:
        return max(self.severity_48h.values(), default=0.0)

    @property
    def severity_amplifier(self) -> float:
        """Multiplicative factor (≥1) by which severity forecasts amplify
        the base disease penalty weight.

        Formula: ``1 + config.w_severity_amplification * max_severity / 100``
        """
        max_sev = max(self.max_severity_24h, self.max_severity_48h)
        return 1.0 + max_sev / 100.0  # config multiplier applied externally

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "classification": self.classification,
            "confidence": self.confidence,
            "max_severity_24h": self.max_severity_24h,
            "max_severity_48h": self.max_severity_48h,
            "severity_amplifier": round(self.severity_amplifier, 3),
        }


# ── Environment-based disease risk ────────────────────────────────────────────


def _compute_env_disease_risk(state: np.ndarray) -> float:
    """Compute disease risk from *predicted* environmental state variables.

    Unlike the static ``disease_risk_score`` field (which stays at its
    initial value during the horizon), this function re-evaluates risk
    from the humidity, temperature, leaf wetness, and VPD that the model
    actually predicts.  This makes the disease penalty responsive to the
    solver's actuator choices.

    Uses the same sigmoid sub-score approach as
    ``constants.compute_disease_risk_score`` for consistency.
    """
    hum = state[_STATE_IDX["indoor_humidity"]]
    temp = state[_STATE_IDX["indoor_temp"]]
    lw = state[_STATE_IDX["leaf_wetness_proxy"]]
    vpd = state[_STATE_IDX["vpd"]]

    hum_risk = sigmoid(hum, 75.0, 0.2)
    wet_risk = sigmoid(lw, 0.5, 8.0)
    temp_risk = sigmoid(temp, 22.0, 0.15)
    # Low VPD → stagnant humid air → higher risk
    vpd_risk = 1.0 - sigmoid(vpd, 0.8, 5.0) if vpd > 0 else 0.5

    return 0.35 * hum_risk + 0.25 * wet_risk + 0.20 * temp_risk + 0.20 * vpd_risk


# ── Stage cost ────────────────────────────────────────────────────────────────


# Variables penalised by the environmental-bounds barrier.
_ENV_BOUND_VARS: list[str] = [
    "indoor_temp",
    "indoor_humidity",
    "co2",
    "soil_moisture",
    "light_intensity",
]


def _env_bounds_penalty(
    state: np.ndarray,
    env_bounds: dict[str, tuple[float, float]],
    scales: np.ndarray,
) -> float:
    """Quadratic penalty for predicted state outside stage environmental bounds.

    Returns 0 when all tracked variables are within ``[lo, hi]``.
    When a variable exceeds a bound, the penalty grows as
    ``((violation) / normalisation_scale)²``.
    """
    penalty = 0.0
    for var in _ENV_BOUND_VARS:
        if var not in env_bounds or var not in _STATE_IDX:
            continue
        lo, hi = env_bounds[var]
        idx = _STATE_IDX[var]
        val = state[idx]
        s = scales[idx]
        if val < lo:
            penalty += ((lo - val) / s) ** 2
        elif val > hi:
            penalty += ((val - hi) / s) ** 2
    return penalty


class StageCost:
    """Per-timestep quadratic tracking cost with disease-aware extensions.

    .. math::

        \\ell(x_k, u_k) = \\sum_i (m_i \\cdot w_i) \\left(\\frac{x_{k,i} - r_i}{s_i}\\right)^2
                          + w^{\\prime}_{\\text{dis}} \\cdot \\hat{d}(x_k)^2
                          + J_{\\text{hum\\_exp}} + J_{\\text{fog\\_sup}} + J_{\\text{irr\\_caut}}
                          + w_{\\text{energy}} \\cdot E(u_k)
                          + w_{\\text{water}} \\cdot W(u_k)
                          + w_{\\text{switch}} \\cdot \\|\\Delta u_k\\|^2

    where :math:`m_i` is a per-step weather modifier (default 1),
    :math:`\\hat{d}(x_k)` is the *predicted* disease risk computed from the
    environmental state, and :math:`w^{\\prime}_{\\text{dis}}` includes
    severity amplification.
    """

    def __init__(
        self,
        config: MPCConfig,
        growth_stage: str,
        energy_costs: dict[str, float] | None = None,
        disease_context: DiseaseContext | None = None,
    ) -> None:
        self._config = config
        self._stage = growth_stage

        sp = get_setpoint(growth_stage)
        profile = get_control_profile(growth_stage)

        self._ref = _setpoint_vector(sp)
        self._scales = _normalisation_scales()

        # Build weight vector from base config * profile multipliers.
        base_weights = config.cost_weight_vector
        self._w = np.zeros(len(STATE_VARIABLES), dtype=np.float64)
        for wname, svar in _WEIGHT_TO_STATE.items():
            idx = _STATE_IDX[svar]
            self._w[idx] = base_weights[wname] * profile.control_weights.get(
                svar.replace("indoor_", "").replace("_intensity", ""),
                1.0,
            )

        # ── Disease weight with severity amplification ────────────────
        dc = disease_context or DiseaseContext()
        sev_amp = 1.0 + config.w_severity_amplification * (dc.severity_amplifier - 1.0)
        self._w_disease = base_weights["w_disease"] * profile.disease_sensitivity * sev_amp

        # ── Disease-aware penalty weights ─────────────────────────────
        self._w_hum_exposure = config.w_humidity_exposure * profile.disease_sensitivity
        self._w_fogger_suppress = config.w_fogger_suppression
        self._fogger_suppress_threshold = config.disease_fogger_suppress_risk_threshold
        self._w_irr_caution = config.w_irrigation_caution

        self._w_energy = base_weights["w_energy"] * profile.resource_priority.get("energy", 1.0)
        self._w_water = base_weights["w_water"] * profile.resource_priority.get("water", 1.0)
        self._w_switch = base_weights["w_switch"]

        # ── Stress penalty (matches yield-proxy stress score) ─────────
        self._sp_temp = sp.temp
        self._sp_temp_tol = max(sp.temp_tol, 0.5)
        self._sp_hum = sp.humidity
        self._sp_hum_tol = max(sp.hum_tol, 1.0)
        self._sp_vpd = sp.vpd
        self._w_stress = 1.5  # weight for stress-excursion penalty

        # ── Environmental bounds penalty ──────────────────────────────
        cs = get_default_constraints(growth_stage)
        self._env_bounds = cs.environmental
        self._w_env_bounds = 0.5  # weight for env-bounds barrier

        self._energy_costs = energy_costs or {
            "fan_speed": 0.15,
            "vent_opening": 0.02,
            "heater_output": 0.80,
            "led_intensity": 0.30,
            "co2_valve_pct": 0.05,
            "fogger_duty": 0.10,
            "irrigation_qty": 0.01,
        }

    def evaluate(
        self,
        state: np.ndarray,
        control: np.ndarray,
        prev_control: np.ndarray | None = None,
        step_modifiers: np.ndarray | None = None,
    ) -> float:
        """Compute scalar stage cost.

        Parameters
        ----------
        state : (N_STATE,) array
            Predicted state at step *k*.
        control : (N_CTRL,) array
            Applied control at step *k*.
        prev_control : (N_CTRL,) array, optional
            Control at step *k-1* (for switching penalty).
        step_modifiers : (N_STATE,) array, optional
            Per-variable weather-adaptive weight multipliers.
            Values of 1.0 = no change.  Provided by
            ``WeatherAdaptiveModifiers.modifiers[k]``.
        """
        # Tracking error — with optional weather-adaptive scaling
        err = (state - self._ref) / self._scales
        w_active = self._w * step_modifiers if step_modifiers is not None else self._w
        tracking = float(np.sum(w_active * err ** 2))

        # ── Disease penalty (environment-based) ───────────────────────
        # Recompute disease risk from predicted state so the penalty
        # responds to the solver's actuator choices.
        predicted_risk = _compute_env_disease_risk(state)
        disease_penalty = self._w_disease * predicted_risk ** 2

        # ── Humidity exposure penalty ─────────────────────────────────
        # Extra cost when RH exceeds setpoint AND disease risk is elevated.
        # This discourages the controller from tolerating high humidity
        # when disease conditions are present.
        hum_idx = _STATE_IDX["indoor_humidity"]
        hum_excess = max(0.0, state[hum_idx] - self._ref[hum_idx])
        hum_exposure = self._w_hum_exposure * (hum_excess / 20.0) ** 2 * predicted_risk

        # ── Fogger suppression penalty ────────────────────────────────
        # Extra cost on fogger duty when predicted risk exceeds threshold.
        fogger_duty = control[_CTRL_IDX["fogger_duty"]]
        risk_excess = max(0.0, predicted_risk - self._fogger_suppress_threshold)
        fogger_suppress = self._w_fogger_suppress * fogger_duty * risk_excess

        # ── Irrigation caution penalty ────────────────────────────────
        # Extra irrigation cost when RH is above setpoint AND disease risk
        # is elevated — prevents adding moisture to an already-wet environment.
        irr = control[_CTRL_IDX["irrigation_qty"]]
        irr_caution = self._w_irr_caution * (irr / 50.0) * (hum_excess / 20.0) * predicted_risk

        # ── Energy cost ───────────────────────────────────────────────
        energy = 0.0
        for name, idx in _CTRL_IDX.items():
            energy += control[idx] * self._energy_costs.get(name, 0.0)
        energy_cost = self._w_energy * energy

        # ── Water cost ────────────────────────────────────────────────
        water = (
            control[_CTRL_IDX["irrigation_qty"]]
            + control[_CTRL_IDX["fogger_duty"]] * 2.0
        )
        water_cost = self._w_water * water

        # ── Stress-excursion penalty ──────────────────────────────────
        # Penalise states outside tolerance bands (matches yield proxy
        # _stress_step_score).  Activates only when error exceeds 1×tol.
        stress_penalty = 0.0
        temp_val = state[_STATE_IDX["indoor_temp"]]
        temp_err = abs(temp_val - self._sp_temp) / self._sp_temp_tol
        if temp_err > 1.0:
            stress_penalty += min(1.0, (temp_err - 1.0) / 3.0)

        hum_val = state[_STATE_IDX["indoor_humidity"]]
        hum_excess = max(0.0, hum_val - (self._sp_hum + self._sp_hum_tol))
        stress_penalty += min(1.0, hum_excess / 15.0)

        vpd_val = state[_STATE_IDX["vpd"]]
        vpd_err = abs(vpd_val - self._sp_vpd) / max(self._sp_vpd, 0.3)
        if vpd_err > 0.5:
            stress_penalty += min(1.0, (vpd_err - 0.5) / 2.0)

        stress_cost = self._w_stress * stress_penalty

        # ── Environmental bounds barrier ─────────────────────────────
        env_cost = self._w_env_bounds * _env_bounds_penalty(
            state, self._env_bounds, self._scales,
        )

        # ── Switching cost ───────────────────────────────────────────
        switch_cost = 0.0
        if prev_control is not None:
            delta = control - prev_control
            switch_cost = self._w_switch * float(np.sum(delta ** 2))

        return (
            tracking
            + disease_penalty
            + hum_exposure
            + fogger_suppress
            + irr_caution
            + energy_cost
            + water_cost
            + stress_cost
            + env_cost
            + switch_cost
        )


# ── Terminal cost ─────────────────────────────────────────────────────────────


class TerminalCost:
    """Terminal penalty applied to the final predicted state.

    Same quadratic form as ``StageCost`` but with a configurable
    multiplier (default 2×) to incentivise the solver to end in a
    desirable state.  Uses environment-based disease risk.
    """

    def __init__(
        self,
        config: MPCConfig,
        growth_stage: str,
        terminal_multiplier: float = 2.0,
        disease_context: DiseaseContext | None = None,
    ) -> None:
        sp = get_setpoint(growth_stage)
        profile = get_control_profile(growth_stage)

        self._ref = _setpoint_vector(sp)
        self._scales = _normalisation_scales()
        self._multiplier = terminal_multiplier

        base_weights = config.cost_weight_vector
        self._w = np.zeros(len(STATE_VARIABLES), dtype=np.float64)
        for wname, svar in _WEIGHT_TO_STATE.items():
            idx = _STATE_IDX[svar]
            self._w[idx] = base_weights[wname] * profile.control_weights.get(
                svar.replace("indoor_", "").replace("_intensity", ""),
                1.0,
            )

        dc = disease_context or DiseaseContext()
        sev_amp = 1.0 + config.w_severity_amplification * (dc.severity_amplifier - 1.0)
        self._w_disease = base_weights["w_disease"] * profile.disease_sensitivity * sev_amp

        # Environmental bounds penalty (same as StageCost)
        cs = get_default_constraints(growth_stage)
        self._env_bounds = cs.environmental
        self._w_env_bounds = 0.5

    def evaluate(self, state: np.ndarray) -> float:
        """Compute scalar terminal cost for final predicted state."""
        err = (state - self._ref) / self._scales
        tracking = float(np.sum(self._w * err ** 2))

        predicted_risk = _compute_env_disease_risk(state)
        disease_penalty = self._w_disease * predicted_risk ** 2

        env_penalty = self._w_env_bounds * _env_bounds_penalty(
            state, self._env_bounds, self._scales,
        )

        return self._multiplier * (tracking + disease_penalty + env_penalty)


# ── Cost builder ──────────────────────────────────────────────────────────────


class CostBuilder:
    """Factory that assembles ``StageCost`` + ``TerminalCost`` for a given
    growth stage and evaluates the total horizon cost.

    Stage-transition blending
    -------------------------
    When *next_stage* is provided and the transition is predicted to occur
    within the horizon, the builder creates a secondary ``StageCost`` for
    the upcoming stage.  During horizon steps near the transition boundary,
    a linear blend ramps from the current-stage cost to the next-stage
    cost over ``config.stage_transition_blend_steps`` steps, preventing
    abrupt weight jumps.

    Weather-adaptive modifiers
    --------------------------
    ``total_cost`` accepts an optional list of per-step modifier arrays
    (from ``compute_weather_adaptation``).  Each modifier is passed to
    the corresponding ``StageCost.evaluate`` call so that tracking
    weights are proactively scaled by upcoming weather stress.

    Usage
    -----
    >>> builder = CostBuilder(config, "flowering", disease_context=dc,
    ...                       next_stage="fruiting", steps_to_transition=60)
    >>> J = builder.total_cost(states, controls, weather_modifiers=wm.modifiers)
    """

    def __init__(
        self,
        config: MPCConfig,
        growth_stage: str,
        energy_costs: dict[str, float] | None = None,
        terminal_multiplier: float = 2.0,
        disease_context: DiseaseContext | None = None,
        next_stage: str | None = None,
        steps_to_transition: int | None = None,
    ) -> None:
        dc = disease_context
        self.stage_cost = StageCost(config, growth_stage, energy_costs, dc)
        self.terminal_cost = TerminalCost(
            config, growth_stage, terminal_multiplier, dc,
        )
        self._config = config

        # --- stage-transition blending ---
        self._next_stage_cost: StageCost | None = None
        self._blend_start: int | None = None
        self._blend_steps: int = config.stage_transition_blend_steps
        if next_stage and steps_to_transition is not None and steps_to_transition > 0:
            horizon = config.prediction_horizon_steps
            if steps_to_transition <= horizon + self._blend_steps:
                self._next_stage_cost = StageCost(
                    config, next_stage, energy_costs, dc,
                )
                # blending starts `blend_steps / 2` before the transition
                self._blend_start = max(
                    0, steps_to_transition - self._blend_steps // 2,
                )

    def _blend_alpha(self, k: int) -> float:
        """Return blending weight for next-stage cost at horizon step *k*.

        Returns 0.0 (current stage only) when no blending is active,
        ramps linearly to 1.0 (next stage only) over ``blend_steps``.
        """
        if self._next_stage_cost is None or self._blend_start is None:
            return 0.0
        if k < self._blend_start:
            return 0.0
        progress = (k - self._blend_start) / max(self._blend_steps, 1)
        return min(progress, 1.0)

    def total_cost(
        self,
        states: list[np.ndarray] | np.ndarray,
        controls: list[np.ndarray] | np.ndarray,
        weather_modifiers: list[np.ndarray] | None = None,
    ) -> float:
        """Evaluate :math:`J = \\sum_{k=0}^{N-1} \\ell(x_k, u_k) + V_f(x_N)`.

        Parameters
        ----------
        states : list of (N_STATE,) arrays or (N+1, N_STATE) matrix
            Predicted states *including* the initial state (length N+1).
        controls : list of (N_CTRL,) arrays or (N, N_CTRL) matrix
            Control actions over the horizon (length N).
        weather_modifiers : list of (N_STATE,) arrays, optional
            Per-step weight multipliers from weather adaptation.
        """
        if isinstance(states, np.ndarray) and states.ndim == 2:
            states = [states[i] for i in range(states.shape[0])]
        if isinstance(controls, np.ndarray) and controls.ndim == 2:
            controls = [controls[i] for i in range(controls.shape[0])]

        N = len(controls)
        J = 0.0
        for k in range(N):
            prev_u = controls[k - 1] if k > 0 else None
            step_mod = weather_modifiers[k] if weather_modifiers and k < len(weather_modifiers) else None

            alpha = self._blend_alpha(k)
            if alpha <= 0.0:
                J += self.stage_cost.evaluate(states[k], controls[k], prev_u, step_mod)
            elif alpha >= 1.0:
                J += self._next_stage_cost.evaluate(states[k], controls[k], prev_u, step_mod)
            else:
                cost_cur = self.stage_cost.evaluate(states[k], controls[k], prev_u, step_mod)
                cost_nxt = self._next_stage_cost.evaluate(states[k], controls[k], prev_u, step_mod)
                J += (1.0 - alpha) * cost_cur + alpha * cost_nxt

        # Terminal cost on final predicted state
        if len(states) > N:
            J += self.terminal_cost.evaluate(states[N])

        return J
