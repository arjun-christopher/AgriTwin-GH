"""
Weather-adaptive cost-weight modifiers for the MPC solver.

The greenhouse controller should respond **proactively** to upcoming weather
changes rather than reacting only after they arrive.  This module analyses
the weather forecast over the prediction horizon and returns per-step
weight multipliers that adjust the MPC objective so the solver:

  • Pre-cools before an external temperature spike.
  • Adjusts ventilation strategy before an external humidity swing.
  • Anticipates increased solar heat load.

**How it works**

For each step *k* in the horizon the module computes a "stress" measure
for temperature, humidity, and solar radiation by comparing the forecast
to comfortable conditions (derived from the current setpoint).  It then
applies a **lookahead window** so that stress arriving at step *k+L* also
increases the weight at step *k*, incentivising the solver to act early.

The result is a list of per-step weight multiplier arrays (aligned with
``STATE_VARIABLES``) that the cost function applies element-wise on top
of the base cost weights.  A multiplier of 1.0 means "no change"; values
>1 tighten tracking; values <1 relax it.

**Tuning knobs** (all in ``MPCConfig``):

  ``weather_temp_anticipation``    — how aggressively to pre-cool (0 = off)
  ``weather_rh_anticipation``      — humidity anticipation weight
  ``weather_solar_anticipation``   — solar heat-load anticipation weight
  ``weather_lookahead_steps``      — how far ahead to scan (default 12 = 1 h)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import MPCConfig
from .constants import STATE_VARIABLES, STEPS_PER_HOUR
from .setpoints import StageSetpoint

logger = logging.getLogger(__name__)

_STATE_IDX: dict[str, int] = {name: i for i, name in enumerate(STATE_VARIABLES)}
_N_X = len(STATE_VARIABLES)


# ── Public data structure ─────────────────────────────────────────────────────


@dataclass
class WeatherAdaptiveModifiers:
    """Per-step weight multipliers computed from the weather forecast.

    ``modifiers[k]`` is an ``(N_X,)`` array of multipliers applied to the
    cost weight vector at horizon step *k*.  Values of 1.0 mean no change.
    """

    modifiers: list[np.ndarray] = field(default_factory=list)
    temp_stress: list[float] = field(default_factory=list)
    rh_stress: list[float] = field(default_factory=list)
    solar_stress: list[float] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary for logging / inspection."""
        return {
            "n_steps": len(self.modifiers),
            "avg_temp_stress": round(float(np.mean(self.temp_stress)), 4) if self.temp_stress else 0.0,
            "avg_rh_stress": round(float(np.mean(self.rh_stress)), 4) if self.rh_stress else 0.0,
            "avg_solar_stress": round(float(np.mean(self.solar_stress)), 4) if self.solar_stress else 0.0,
            "max_temp_modifier": round(float(max(
                m[_STATE_IDX["indoor_temp"]] for m in self.modifiers
            )), 4) if self.modifiers else 1.0,
        }


# ── Core computation ──────────────────────────────────────────────────────────


def compute_weather_adaptation(
    weather_seq: list[dict[str, Any]],
    setpoint: StageSetpoint,
    horizon_steps: int,
    config: MPCConfig,
) -> WeatherAdaptiveModifiers:
    """Compute per-step weight multipliers based on the weather forecast.

    Parameters
    ----------
    weather_seq : list of dicts
        Per-step weather with keys ``temp_external``, ``humidity_external``,
        ``solar_radiation``.  Length ≥ *horizon_steps*.
    setpoint : StageSetpoint
        Current growth-stage setpoint (provides target indoor temp / humidity).
    horizon_steps : int
        Number of control-horizon steps.
    config : MPCConfig
        Contains anticipation weights and lookahead depth.

    Returns
    -------
    WeatherAdaptiveModifiers
    """
    N = min(horizon_steps, len(weather_seq))
    lookahead = config.weather_lookahead_steps
    w_temp = config.weather_temp_anticipation
    w_rh = config.weather_rh_anticipation
    w_solar = config.weather_solar_anticipation

    # ── Compute raw stress at each step ───────────────────────────────
    temp_stress = np.zeros(N, dtype=np.float64)
    rh_stress = np.zeros(N, dtype=np.float64)
    solar_stress = np.zeros(N, dtype=np.float64)

    for k in range(N):
        w = weather_seq[k]
        t_ext = w.get("temp_external", setpoint.temp)
        rh_ext = w.get("humidity_external", setpoint.humidity)
        solar = w.get("solar_radiation", 300.0)

        # Temperature stress: how much hotter external is vs setpoint
        # A 10 °C excess maps to stress = 1.0
        temp_stress[k] = max(0.0, (t_ext - setpoint.temp - 3.0)) / 10.0

        # RH stress: external humidity different from setpoint
        # High external RH makes dehumidification harder
        rh_stress[k] = max(0.0, (rh_ext - setpoint.humidity - 5.0)) / 20.0

        # Solar stress: high solar → increased heat load
        # 800 W/m² above baseline is stress = 1.0
        solar_stress[k] = max(0.0, (solar - 300.0)) / 800.0

    # ── Apply lookahead (anticipation) ────────────────────────────────
    # For each step k, average the stress over [k, k+lookahead) to create
    # an anticipation signal.  This means future stress propagates backward.
    temp_antic = _apply_lookahead(temp_stress, lookahead)
    rh_antic = _apply_lookahead(rh_stress, lookahead)
    solar_antic = _apply_lookahead(solar_stress, lookahead)

    # ── Build per-step multiplier arrays ──────────────────────────────
    modifiers: list[np.ndarray] = []
    for k in range(N):
        m = np.ones(_N_X, dtype=np.float64)

        # Temperature: increase tracking weight when stress is high
        m[_STATE_IDX["indoor_temp"]] = 1.0 + w_temp * temp_antic[k]

        # Solar also increases effective temperature stress
        m[_STATE_IDX["indoor_temp"]] += w_solar * solar_antic[k]

        # Humidity: increase tracking weight when external RH is elevated
        m[_STATE_IDX["indoor_humidity"]] = 1.0 + w_rh * rh_antic[k]

        # VPD tracks with humidity stress (VPD and RH are correlated)
        m[_STATE_IDX["vpd"]] = 1.0 + 0.5 * w_rh * rh_antic[k]

        modifiers.append(m)

    result = WeatherAdaptiveModifiers(
        modifiers=modifiers,
        temp_stress=temp_stress.tolist(),
        rh_stress=rh_stress.tolist(),
        solar_stress=solar_stress.tolist(),
    )

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Weather adaptation: %s", result.to_summary())

    return result


# ── Helpers ───────────────────────────────────────────────────────────────────


def _apply_lookahead(stress: np.ndarray, lookahead: int) -> np.ndarray:
    """Forward-looking moving average: step k sees avg stress over [k, k+L).

    This propagates future stress backward so the controller pre-acts.
    """
    N = len(stress)
    if lookahead <= 1:
        return stress.copy()

    result = np.zeros(N, dtype=np.float64)
    # Cumulative sum for O(N) sliding window
    cumsum = np.concatenate([[0.0], np.cumsum(stress)])
    for k in range(N):
        end = min(k + lookahead, N)
        window_len = end - k
        result[k] = (cumsum[end] - cumsum[k]) / max(window_len, 1)
    return result
