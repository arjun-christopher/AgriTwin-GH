"""
Yield / growth quality proxy for greenhouse controller evaluation.

Produces a transparent, configurable scalar score (0–100) that estimates
the likely impact of controller behaviour on tomato crop yield and quality.

The proxy is **not** a biophysical crop model — it aggregates normalised
penalty and reward signals that correlate with yield outcomes based on
agronomic heuristics:

1. **Climate tracking fidelity** — closeness to stage-specific climate
   targets, weighted by per-stage ``StageControlProfile.control_weights``.
2. **Disease burden** — integrated disease risk over the window.
3. **Stress exposure** — cumulative excursions of temperature and
   humidity outside safe envelopes.
4. **Resource stability** — penalises erratic actuator usage that
   stresses the plant (rapid temperature or humidity swings).

All components are individually logged so the score is fully auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

import numpy as np

from .setpoints import StageSetpoint, get_control_profile, get_setpoint
from .state import ActuatorState, GreenhouseState


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class YieldProxyWeights:
    """Relative importance of each component (they are normalised internally)."""

    climate_tracking: float = 0.40
    disease_burden: float = 0.25
    stress_exposure: float = 0.20
    resource_stability: float = 0.15

    def total(self) -> float:
        return (
            self.climate_tracking
            + self.disease_burden
            + self.stress_exposure
            + self.resource_stability
        )


# ── Per-step scores ───────────────────────────────────────────────────────────


@dataclass
class YieldProxyStepScore:
    """Per-step component scores (each 0–1, higher = better)."""

    climate_score: float = 1.0
    disease_score: float = 1.0
    stress_score: float = 1.0
    stability_score: float = 1.0
    composite: float = 1.0


# ── Aggregate result ──────────────────────────────────────────────────────────


@dataclass
class YieldProxyResult:
    """Aggregate yield/quality proxy for a full simulation window."""

    overall_score: float = 0.0           # 0-100 scale
    climate_tracking_score: float = 0.0  # 0-100
    disease_burden_score: float = 0.0    # 0-100
    stress_exposure_score: float = 0.0   # 0-100
    resource_stability_score: float = 0.0  # 0-100
    weights_used: dict[str, float] = field(default_factory=dict)
    per_step_scores: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # per_step_scores can be large; keep summary statistics only
        if self.per_step_scores:
            d["per_step_summary"] = {
                "min": round(min(self.per_step_scores), 2),
                "max": round(max(self.per_step_scores), 2),
                "mean": round(float(np.mean(self.per_step_scores)), 2),
                "std": round(float(np.std(self.per_step_scores)), 2),
            }
        d.pop("per_step_scores", None)
        return d


# ── Core computation ──────────────────────────────────────────────────────────


def compute_yield_proxy(
    states: Sequence[GreenhouseState],
    actuators: Sequence[ActuatorState],
    growth_stages: Sequence[str],
    weights: YieldProxyWeights | None = None,
) -> YieldProxyResult:
    """Compute the yield / growth quality proxy.

    Parameters
    ----------
    states : sequence of GreenhouseState
    actuators : sequence of ActuatorState
    growth_stages : sequence of str
        Canonical growth stage label per step.
    weights : YieldProxyWeights, optional
        Component weights; defaults to balanced config.

    Returns
    -------
    YieldProxyResult
        Auditable score breakdown.
    """
    w = weights or YieldProxyWeights()
    n = len(states)
    if n == 0:
        return YieldProxyResult()

    climate_scores: list[float] = []
    disease_scores: list[float] = []
    stress_scores: list[float] = []
    stability_scores: list[float] = []
    per_step: list[float] = []

    for i in range(n):
        gs = states[i]
        stage = growth_stages[i]
        sp = get_setpoint(stage)
        profile = get_control_profile(stage)

        # ── 1. Climate tracking (weighted by stage profile) ──────────
        c_score = _climate_step_score(gs, sp, profile.control_weights)
        climate_scores.append(c_score)

        # ── 2. Disease burden ────────────────────────────────────────
        d_score = max(0.0, 1.0 - gs.disease_risk_score / max(sp.disease_risk_max, 0.01))
        d_score = min(1.0, d_score)
        disease_scores.append(d_score)

        # ── 3. Stress exposure ───────────────────────────────────────
        s_score = _stress_step_score(gs, sp)
        stress_scores.append(s_score)

        # ── 4. Resource stability ────────────────────────────────────
        if i > 0:
            r_score = _stability_step_score(actuators[i], actuators[i - 1])
        else:
            r_score = 1.0
        stability_scores.append(r_score)

        # ── Composite per-step ───────────────────────────────────────
        w_total = w.total() or 1.0
        step_val = (
            w.climate_tracking * c_score
            + w.disease_burden * d_score
            + w.stress_exposure * s_score
            + w.resource_stability * r_score
        ) / w_total
        per_step.append(step_val * 100.0)

    # Aggregate (mean over steps)
    agg_climate = float(np.mean(climate_scores)) * 100.0
    agg_disease = float(np.mean(disease_scores)) * 100.0
    agg_stress = float(np.mean(stress_scores)) * 100.0
    agg_stability = float(np.mean(stability_scores)) * 100.0

    w_total = w.total() or 1.0
    overall = (
        w.climate_tracking * agg_climate
        + w.disease_burden * agg_disease
        + w.stress_exposure * agg_stress
        + w.resource_stability * agg_stability
    ) / w_total

    return YieldProxyResult(
        overall_score=round(overall, 2),
        climate_tracking_score=round(agg_climate, 2),
        disease_burden_score=round(agg_disease, 2),
        stress_exposure_score=round(agg_stress, 2),
        resource_stability_score=round(agg_stability, 2),
        weights_used={
            "climate_tracking": w.climate_tracking,
            "disease_burden": w.disease_burden,
            "stress_exposure": w.stress_exposure,
            "resource_stability": w.resource_stability,
        },
        per_step_scores=per_step,
    )


# ── Sub-scores ────────────────────────────────────────────────────────────────


def _climate_step_score(
    gs: GreenhouseState,
    sp: StageSetpoint,
    control_weights: dict[str, float],
) -> float:
    """Weighted normalised proximity to stage setpoints (0-1)."""
    checks = [
        ("temp", gs.indoor_temp, sp.temp, sp.temp_tol),
        ("humidity", gs.indoor_humidity, sp.humidity, sp.hum_tol),
        ("soil_moisture", gs.soil_moisture, sp.soil_moisture, 5.0),
        ("co2", gs.co2, sp.co2, 100.0),
        ("vpd", gs.vpd, sp.vpd, 0.2),
    ]
    weighted_sum = 0.0
    weight_total = 0.0
    for name, actual, target, tolerance in checks:
        w = control_weights.get(name, 1.0)
        # Score = 1 inside tolerance, linear decay to 0 at 3× tolerance
        error = abs(actual - target)
        if tolerance <= 0:
            tolerance = 1.0
        ratio = error / tolerance
        score = max(0.0, 1.0 - ratio / 3.0)
        weighted_sum += w * score
        weight_total += w

    return weighted_sum / max(weight_total, 1e-6)


def _stress_step_score(gs: GreenhouseState, sp: StageSetpoint) -> float:
    """Penalty for temperature, humidity, and VPD excursions (0-1)."""
    penalties = 0.0

    # Temperature — severe penalty outside ±3× tolerance
    temp_err = abs(gs.indoor_temp - sp.temp) / max(sp.temp_tol, 0.5)
    if temp_err > 1.0:
        penalties += min(1.0, (temp_err - 1.0) / 3.0)

    # Humidity — above tolerance promotes disease
    hum_err = max(0.0, gs.indoor_humidity - (sp.humidity + sp.hum_tol))
    penalties += min(1.0, hum_err / 15.0)

    # VPD — both extremes stress the plant
    vpd_err = abs(gs.vpd - sp.vpd) / max(sp.vpd, 0.3)
    if vpd_err > 0.5:
        penalties += min(1.0, (vpd_err - 0.5) / 2.0)

    return max(0.0, 1.0 - penalties / 3.0)


def _stability_step_score(
    curr: ActuatorState, prev: ActuatorState
) -> float:
    """Penalise large actuator swings (0-1, 1 = perfectly smooth)."""
    curr_arr = curr.to_numpy()
    prev_arr = prev.to_numpy()
    delta = np.abs(curr_arr - prev_arr)
    # Normalise by typical actuator range (0–1 for most, 0–several for irrigation)
    norms = np.array([1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0])
    normed = delta / np.maximum(norms, 1e-6)
    mean_change = float(np.mean(normed))
    return max(0.0, 1.0 - mean_change)
