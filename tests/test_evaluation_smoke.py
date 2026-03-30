"""Smoke test for the evaluation framework (Subprompt 8)."""
import json
import os
import sys
import tempfile

# ── 1. Import all evaluation symbols ──
from agritwin_gh.mpc import (
    TrackingMetrics, DiseaseBurdenMetrics, ResourceMetrics,
    ControlQualityMetrics, SafetyMetrics, ControllerMetricsBundle,
    compute_all_metrics,
    YieldProxyWeights, YieldProxyResult, compute_yield_proxy,
    ExperimentConfig, ExperimentRunner, ComparisonReport,
    make_baseline_adapter, generate_default_weather,
    generate_default_growth_stages, make_default_initial_state,
    run_evaluation, save_evaluation_artifacts, load_evaluation_report,
)
print("[OK] All evaluation symbols imported")

# ── 2. Metrics test on synthetic data ──
from agritwin_gh.mpc.state import GreenhouseState, ActuatorState
import random
random.seed(42)

states = []
actuators = []
stages = []
for i in range(50):
    states.append(GreenhouseState(
        indoor_temp=22.0 + 3.0 * (i % 10) / 10,
        indoor_humidity=65.0 + 10.0 * ((i * 7) % 13) / 13,
        soil_moisture=60.0 + 5.0 * (i % 5) / 5,
        co2=700 + 50 * (i % 4),
        vpd=0.8 + 0.1 * (i % 3),
        disease_risk_score=0.1 + 0.02 * (i % 8),
        leaf_wetness_proxy=0.3 + 0.05 * (i % 6),
    ))
    actuators.append(ActuatorState(
        fan_speed=0.3 + 0.05 * (i % 7),
        vent_opening=0.2 + 0.03 * (i % 5),
        irrigation_qty=0.5 if i % 10 == 0 else 0.0,
        heater_output=0.1 * ((i + 3) % 4),
        led_intensity=0.4 if i > 25 else 0.0,
        co2_valve_pct=0.15,
        fogger_duty=0.1 if i % 15 == 0 else 0.0,
    ))
    stages.append("flowering")

bundle = compute_all_metrics(states, actuators, stages, "test_ctrl", "test")
t_rmse = bundle.tracking["indoor_temp"].rmse
print(f"[OK] Metrics bundle: {bundle.n_steps} steps, temp RMSE={t_rmse:.3f}")
print(f"     Disease mean_risk={bundle.disease_burden.mean_risk:.4f}")
print(f"     Energy={bundle.resources.total_energy_kwh:.4f} kWh, Water={bundle.resources.total_water_litres:.2f} L")
print(f"     Safety violations={bundle.safety.total_violations}")
print(f"     Smoothness L2={bundle.control_quality.mean_smoothness_l2:.4f}")

# ── 3. Yield proxy ──
yp = compute_yield_proxy(states, actuators, stages)
avg_step = sum(yp.per_step_scores) / len(yp.per_step_scores)
print(f"[OK] Yield proxy: overall={yp.overall_score:.2f}/100, climate={yp.climate_tracking_score:.2f}, disease={yp.disease_burden_score:.2f}")
print(f"     Per-step scores: {len(yp.per_step_scores)} entries, mean={avg_step:.2f}")

# ── 4. Experiment runner with baseline ──
weather = generate_default_weather(50)
stages_seq = generate_default_growth_stages(50)
init = make_default_initial_state()

exp_cfg = ExperimentConfig(
    n_steps=50,
    initial_state=init,
    weather_sequence=weather,
    growth_stage_sequence=stages_seq,
    experiment_name="smoke_test",
)
runner = ExperimentRunner(exp_cfg)
runner.register_controller("baseline", make_baseline_adapter(), controller_type="rule_based")
report = runner.run()

table = report.summary_table()
bl = table["baseline"]
print(f"[OK] Experiment: baseline temp_rmse={bl['temp_rmse']:.3f}, yield={bl['yield_score']:.2f}")
print(f"     Water={bl['total_water_l']:.2f} L, Energy={bl['total_energy_kwh']:.4f} kWh")

# ── 5. Artifact save/load ──
with tempfile.TemporaryDirectory() as tmpdir:
    out = save_evaluation_artifacts(report, workspace_root=tmpdir, run_id="smoke_001")
    files = os.listdir(out)
    print(f"[OK] Artifacts saved: {sorted(files)}")
    loaded = load_evaluation_report(out)
    print(f"[OK] Report loaded back: keys={sorted(loaded.keys())}")

# ── 6. Report serialisation ──
d = report.to_dict()
js = json.dumps(d, indent=2, default=str)
print(f"[OK] Report JSON serialisation: {len(js)} chars")

# ── 7. to_dict round-trip for all dataclasses ──
for cid, m in report.controller_metrics.items():
    md = m.to_dict()
    assert "tracking" in md, "Missing tracking in metrics dict"
    assert "disease_burden" in md, "Missing disease_burden"
    assert "resources" in md, "Missing resources"
    assert "control_quality" in md, "Missing control_quality"
    assert "safety" in md, "Missing safety"
    print(f"[OK] {cid} metrics dict has all 5 sections")

for cid, yr in report.yield_results.items():
    yd = yr.to_dict()
    assert "overall_score" in yd, "Missing overall_score"
    assert "per_step_summary" in yd, "Missing per_step_summary"
    print(f"[OK] {cid} yield proxy dict has summary stats")

print()
print("=== ALL SMOKE TESTS PASSED ===")
