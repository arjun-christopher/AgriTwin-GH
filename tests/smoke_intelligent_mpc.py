"""Smoke test for the intelligent MPC upgrade (disease + weather + stage blending)."""

import sys
sys.path.insert(0, "src")

import numpy as np
from agritwin_gh.mpc import (
    DiseaseContext,
    WeatherAdaptiveModifiers,
    compute_weather_adaptation,
    tighten_constraints_for_disease,
    CostBuilder,
    StageCost,
    TerminalCost,
    MPCSolver,
    MPCSolution,
    MPCConfig,
    FusedState,
    GreenhouseState,
    ActuatorState,
    get_setpoint,
    get_default_constraints,
)

print("1. All imports OK")

# ── DiseaseContext ────────────────────────────────────────────────────────────
dc = DiseaseContext(
    risk_score=0.6,
    classification="early_blight",
    confidence=0.85,
    current_severity={"early_blight": 30.0},
    severity_24h={"early_blight": 45.0},
    severity_48h={"early_blight": 60.0},
)
assert dc.severity_amplifier > 1.0, "severity_amplifier should be > 1"
assert dc.max_severity_24h == 45.0
assert dc.max_severity_48h == 60.0
print(f"2. DiseaseContext OK — sev_amp={dc.severity_amplifier:.2f}")

# ── DiseaseContext.from_fused ─────────────────────────────────────────────────
fused = FusedState(
    greenhouse_state=GreenhouseState(
        indoor_temp=32.0, indoor_humidity=85.0, soil_moisture=60.0,
        co2=450.0, light_intensity=300.0, vpd=1.2,
        leaf_wetness_proxy=70.0, disease_risk_score=0.6, growth_stage_index=3,
    ),
    growth_stage="flowering",
    next_stage="unripe",
    hours_to_transition=4.0,
    disease_classification="early_blight",
    disease_confidence=0.85,
    disease_risk_score=0.6,
    current_severity={"early_blight": 30.0},
    severity_24h={"early_blight": 45.0},
    severity_48h={"early_blight": 60.0},
    weather_forecast=[
        {"temp_external": 38.0, "humidity_external": 90.0, "solar_radiation": 800.0}
    ] * 20,
)
dc2 = DiseaseContext.from_fused(fused)
assert dc2.risk_score == 0.6
assert dc2.classification == "early_blight"
print("3. DiseaseContext.from_fused OK")

# ── Weather adaptation ────────────────────────────────────────────────────────
config = MPCConfig()
sp = get_setpoint("flowering")
weather = [{"temp_external": 38.0, "humidity_external": 90.0, "solar_radiation": 800.0}] * 10
wm = compute_weather_adaptation(weather, sp, 10, config)
assert len(wm.modifiers) == 10
assert max(wm.temp_stress) > 0, "Should detect temp stress (38 > setpoint)"
assert max(wm.rh_stress) > 0, "Should detect RH stress (90% > setpoint)"
print(f"4. Weather adaptation OK — temp_stress_max={max(wm.temp_stress):.3f}, rh_stress_max={max(wm.rh_stress):.3f}")

# ── Constraint tightening ─────────────────────────────────────────────────────
base = get_default_constraints("flowering")
tight = tighten_constraints_for_disease(
    base, 0.6,
    rh_tightening_threshold=0.4,
    rh_tightened_ceiling=80.0,
    fogger_suppress_threshold=0.5,
    fogger_suppressed_max_duty=0.3,
    severity_24h={"early_blight": 55.0},
)
rh_orig = base.environmental.get("indoor_humidity", (0, 100))[1]
rh_tight = tight.environmental.get("indoor_humidity", (0, 100))[1]
assert rh_tight < rh_orig, f"RH ceiling should be tightened: {rh_tight} < {rh_orig}"
fog_tight = tight.actuator_bounds.get("fogger_duty", (0, 1.0))[1]
assert fog_tight <= 0.3, f"Fogger should be suppressed: {fog_tight}"
print(f"5. Constraint tightening OK — RH: {rh_orig} -> {rh_tight:.1f}, fogger hi={fog_tight:.2f}")

# ── CostBuilder with disease + blending ───────────────────────────────────────
builder = CostBuilder(
    config, "flowering",
    disease_context=dc,
    next_stage="unripe",
    steps_to_transition=60,
)
assert builder._next_stage_cost is not None, "Should have next-stage cost for blending"
assert builder._blend_start is not None
print(f"6. CostBuilder with blending OK — blend_start={builder._blend_start}")

# ── Cost eval with weather modifiers ──────────────────────────────────────────
x = fused.greenhouse_state.to_numpy()
u = np.zeros(7)
u[0] = 0.8  # fan
u[1] = 0.6  # vent
J_with = builder.total_cost([x, x], [u], weather_modifiers=wm.modifiers[:1])
J_without = builder.total_cost([x, x], [u], weather_modifiers=None)
print(f"7. Cost eval: with_weather={J_with:.4f}, without={J_without:.4f}")

# ── Stage cost blending test ──────────────────────────────────────────────────
alpha_0 = builder._blend_alpha(0)
alpha_mid = builder._blend_alpha(builder._blend_start + builder._blend_steps // 2)
alpha_end = builder._blend_alpha(builder._blend_start + builder._blend_steps)
print(f"8. Blending alpha: step_0={alpha_0:.2f}, mid={alpha_mid:.2f}, end={alpha_end:.2f}")
assert alpha_0 <= alpha_end, "Alpha should increase toward transition"

# ── Full MPC solve with disease context ───────────────────────────────────────
print("9. Running full MPC solve (short horizon=8)...")
solver = MPCSolver(config)
solution = solver.solve(fused, horizon_override=8)
print(f"   Converged: {solution.converged}")
print(f"   Total cost: {solution.total_cost:.4f}")
print(f"   Solve time: {solution.solve_time_ms:.1f} ms")
print(f"   Iterations: {solution.n_iterations}")
print(f"   Breakdown: {solution.cost_breakdown}")
fan = solution.first_action.fan_speed
vent = solution.first_action.vent_opening
print(f"   First action: fan={fan:.2f}, vent={vent:.2f}")
assert solution.converged or solution.fallback_used, "Should converge or fallback"

# Check disease/weather info in breakdown
if solution.converged:
    assert "disease_risk_score" in solution.cost_breakdown
    assert "weather_temp_stress" in solution.cost_breakdown
    print(f"   Disease risk in breakdown: {solution.cost_breakdown['disease_risk_score']}")
    print(f"   Weather stress in breakdown: {solution.cost_breakdown['weather_temp_stress']}")

print()
print("=" * 50)
print("ALL INTELLIGENT MPC SMOKE TESTS PASSED")
print("=" * 50)
