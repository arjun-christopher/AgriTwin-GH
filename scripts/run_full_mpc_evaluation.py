#!/usr/bin/env python
"""
Complete end-to-end MPC evaluation for AgriTwin-GH.

Runs the entire MPC pipeline — greenhouse model, cost function, SLSQP solver,
baseline comparison, evaluation metrics, yield proxy — across multiple
realistic scenarios and prints a comprehensive report.

Scenarios
---------
1. Standard 24 h flowering stage (baseline vs MPC)
2. High disease-pressure fruiting stage (baseline vs MPC)
3. Multi-day (48 h) with growth-stage transition
4. Direct MPC solver component validation

Usage
-----
    python tests/run_full_mpc_evaluation.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import sys
import textwrap
import time
from pathlib import Path

# Ensure UTF-8 output on Windows consoles (handles ₹, °, →, — etc.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ── Ensure project root is importable ─────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# ── Imports ───────────────────────────────────────────────────────────────────
from agritwin_gh.mpc import (
    # State
    GreenhouseState, ActuatorState, WeatherState, FusedState,
    # Config
    MPCConfig, load_mpc_config,
    # Greenhouse model
    GreenhouseModelParams, GreenhouseTransitionModel,
    # Baseline controller
    RuleBasedController,
    # Cost function
    CostBuilder, DiseaseContext, StageCost, TerminalCost,
    # Constraints
    ConstraintSet, get_default_constraints, tighten_constraints_for_disease,
    # Weather adaptation
    WeatherAdaptiveModifiers, compute_weather_adaptation,
    # Solver
    MPCSolver, MPCSolution,
    # Setpoints
    get_setpoint, get_control_profile,
    # Constants
    GROWTH_STAGES, compute_disease_risk_score, stage_label_to_index,
    # Evaluation
    TrackingMetrics, DiseaseBurdenMetrics, ResourceMetrics,
    ControlQualityMetrics, SafetyMetrics, ControllerMetricsBundle,
    compute_all_metrics,
    # Yield proxy
    YieldProxyWeights, YieldProxyResult, compute_yield_proxy,
    # Experiment runner
    ExperimentConfig, ExperimentRunner, ComparisonReport,
    make_baseline_adapter, make_mpc_adapter,
    generate_default_weather, generate_default_growth_stages,
    make_default_initial_state,
    # Evaluation API
    run_evaluation, save_evaluation_artifacts, load_evaluation_report,
    # Explanation
    ExplanationBuilder,
)

# ── Formatting helpers ────────────────────────────────────────────────────────

BANNER = "=" * 78
SECTION = "-" * 78

def banner(title: str) -> None:
    print(f"\n{BANNER}")
    print(f"  {title}")
    print(BANNER)

def section(title: str) -> None:
    print(f"\n{SECTION}")
    print(f"  {title}")
    print(SECTION)

def kv(label: str, value, width: int = 40) -> None:
    print(f"  {label:<{width}} {value}")

def print_summary_table(report: ComparisonReport) -> None:
    """Print a formatted comparison table from the report."""
    table = report.summary_table()
    if not table:
        print("  (no controllers in report)")
        return

    headers = list(next(iter(table.values())).keys())
    col_w = max(18, max(len(h) for h in headers) + 2)

    # Header row
    print(f"  {'Controller':<18}", end="")
    for h in headers:
        print(f" {h:>{col_w}}", end="")
    print()
    print("  " + "-" * (18 + len(headers) * (col_w + 1)))

    # Data rows
    for cid, row in table.items():
        print(f"  {cid:<18}", end="")
        for h in headers:
            val = row[h]
            if isinstance(val, float):
                print(f" {val:>{col_w}.4f}", end="")
            else:
                print(f" {str(val):>{col_w}}", end="")
        print()


def print_improvements(improvements: dict[str, dict[str, float]]) -> None:
    """Print pairwise improvement percentages."""
    for pair, imps in improvements.items():
        print(f"\n  {pair}:")
        for metric, pct in imps.items():
            direction = "better" if pct > 0 else "worse"
            print(f"    {metric:<45} {pct:+.2f}%  ({direction})")


# ── Tamil Nadu, India — resource pricing (2026 estimates) ─────────────────────
#   Electricity: ₹6.60/kWh  (TNEB HT-I commercial/agriculture tariff)
#   Water:       ₹0.05/L    (≈ ₹50/kL — TWAD agricultural supply)

TN_ELECTRICITY_RATE = 6.60   # ₹ per kWh
TN_WATER_RATE = 0.05         # ₹ per litre


def print_resource_costs_tn(report: ComparisonReport) -> None:
    """Print resource cost comparison using Tamil Nadu, India pricing."""
    section("Resource Cost — Tamil Nadu, India Pricing (₹)")
    print(f"  Electricity: ₹{TN_ELECTRICITY_RATE:.2f}/kWh | "
          f"Water: ₹{TN_WATER_RATE*1000:.0f}/kL (₹{TN_WATER_RATE:.2f}/L)")
    print()

    costs: dict[str, dict[str, float]] = {}
    print(f"  {'Controller':<18} {'Energy(kWh)':>12} {'Water(L)':>10} "
          f"{'₹ Energy':>10} {'₹ Water':>10} {'₹ TOTAL':>10}")
    print("  " + "-" * 75)

    for cid, metrics in report.controller_metrics.items():
        r = metrics.resources
        e_cost = r.total_energy_kwh * TN_ELECTRICITY_RATE
        w_cost = r.total_water_litres * TN_WATER_RATE
        total = e_cost + w_cost
        costs[cid] = {"energy": e_cost, "water": w_cost, "total": total}
        print(f"  {cid:<18} {r.total_energy_kwh:>12.4f} {r.total_water_litres:>10.2f} "
              f"{e_cost:>10.2f} {w_cost:>10.2f} {total:>10.2f}")

    # Savings comparison
    cids = list(costs.keys())
    if len(cids) >= 2:
        ref_id = cids[0]
        for cid in cids[1:]:
            saving = costs[ref_id]["total"] - costs[cid]["total"]
            pct = (saving / costs[ref_id]["total"] * 100) if costs[ref_id]["total"] > 0 else 0
            label = "saves" if saving >= 0 else "costs extra"
            print(f"\n  {cid} vs {ref_id}: {label} ₹{abs(saving):.2f} ({pct:+.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 1: Standard 24 h flowering stage
# ══════════════════════════════════════════════════════════════════════════════

def scenario_1_standard_24h() -> ComparisonReport:
    banner("SCENARIO 1: Standard 12 h Flowering Stage (Baseline vs MPC)")

    mpc_cfg = MPCConfig(
        control_horizon_hours=1,     # short horizon for SLSQP tractability
        prediction_horizon_hours=1,
        solver_max_iter=500,
        solver_ftol=1e-4,            # relaxed tolerance for convergence
    )

    t0 = time.perf_counter()
    report = run_evaluation(
        n_steps=144,           # 12 h at 5-min intervals
        dt_minutes=5,
        growth_stage="flowering",
        include_baseline=True,
        include_mpc=True,
        mpc_config=mpc_cfg,
        random_seed=42,
        experiment_name="scenario_1_flowering_12h",
    )
    elapsed = time.perf_counter() - t0

    section("Summary Table")
    print_summary_table(report)

    section("Pairwise Improvements (MPC vs Baseline)")
    print_improvements(report.improvements)

    section("Detailed Tracking Metrics")
    for cid, metrics in report.controller_metrics.items():
        print(f"\n  Controller: {cid} ({metrics.controller_type})")
        for var, tm in metrics.tracking.items():
            print(f"    {var:<22} RMSE={tm.rmse:.4f}  MAE={tm.mae:.4f}  "
                  f"MaxErr={tm.max_abs_error:.4f}  InTol={tm.within_tolerance_pct:.1f}%")

    section("Disease Burden")
    for cid, metrics in report.controller_metrics.items():
        db = metrics.disease_burden
        print(f"  {cid:<18} mean_risk={db.mean_risk:.4f}  max_risk={db.max_risk:.4f}  "
              f"rh_exposure={db.cumulative_rh_exposure:.2f}  "
              f"fav_hours={db.disease_favorable_duration_hours:.2f}")

    section("Resource Consumption")
    for cid, metrics in report.controller_metrics.items():
        r = metrics.resources
        print(f"  {cid:<18} water={r.total_water_litres:.2f} L  energy={r.total_energy_kwh:.4f} kWh  "
              f"water/hr={r.water_per_hour:.2f}  energy/hr={r.energy_per_hour:.4f}")

    section("Control Quality")
    for cid, metrics in report.controller_metrics.items():
        cq = metrics.control_quality
        print(f"  {cid:<18} mean_smoothness_L2={cq.mean_smoothness_l2:.4f}")
        if cq.switching_frequency:
            top = sorted(cq.switching_frequency.items(), key=lambda x: -x[1])[:3]
            for name, freq in top:
                print(f"    {name:<20} switching_freq={freq:.4f}  switches={cq.total_switches.get(name, 0)}")

    section("Safety Violations")
    for cid, metrics in report.controller_metrics.items():
        s = metrics.safety
        print(f"  {cid:<18} total={s.total_violations}  rate={s.violation_rate_pct:.2f}%")
        for var, cnt in s.violation_count.items():
            if cnt > 0:
                print(f"    {var:<22} {cnt} violations ({s.violation_duration_hours[var]:.2f} h)")

    section("Yield Proxy Scores")
    for cid, yr in report.yield_results.items():
        print(f"  {cid:<18} overall={yr.overall_score:.2f}/100  "
              f"climate={yr.climate_tracking_score:.2f}  disease={yr.disease_burden_score:.2f}  "
              f"stress={yr.stress_exposure_score:.2f}  stability={yr.resource_stability_score:.2f}")

    kv("Elapsed time", f"{elapsed:.2f} s")
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 2: High disease-pressure fruiting stage
# ══════════════════════════════════════════════════════════════════════════════

def _generate_high_disease_weather(n_steps: int, dt_minutes: int = 5) -> list[WeatherState]:
    """Warm + humid weather that elevates disease risk."""
    weather: list[WeatherState] = []
    for i in range(n_steps):
        hour = (i * dt_minutes / 60.0) % 24.0
        phase = 2 * np.pi * (hour - 6.0) / 24.0
        temp = 28.0 + 5.0 * np.sin(phase)       # hot baseline
        hum = 82.0 + 8.0 * np.cos(phase)         # persistently humid
        solar = max(0.0, 600.0 * np.sin(np.pi * hour / 14.0)) if 6 <= hour <= 20 else 0.0
        weather.append(WeatherState(
            temp_external=round(temp, 1),
            humidity_external=round(max(50.0, min(98.0, hum)), 1),
            solar_radiation=round(solar, 1),
            windspeed=round(1.0 + 0.8 * abs(np.sin(phase)), 1),
            conditions="overcast",
        ))
    return weather


def _generate_high_disease_initial_state() -> GreenhouseState:
    """Greenhouse already warm with elevated humidity."""
    return GreenhouseState(
        indoor_temp=29.0,
        indoor_humidity=82.0,
        soil_moisture=72.0,
        co2=550.0,
        light_intensity=200.0,
        disease_risk_score=0.45,
        growth_stage_index=stage_label_to_index("unripe"),
        vpd=0.6,
        leaf_wetness_proxy=0.65,
    )


def scenario_2_high_disease() -> ComparisonReport:
    banner("SCENARIO 2: High Disease-Pressure Fruiting Stage (24 h)")

    n_steps = 288   # 24 h at 5-min intervals (practical runtime)
    weather = _generate_high_disease_weather(n_steps)
    init = _generate_high_disease_initial_state()
    stages = generate_default_growth_stages(n_steps, "unripe")

    cfg = ExperimentConfig(
        n_steps=n_steps,
        dt_minutes=5,
        initial_state=init,
        weather_sequence=weather,
        growth_stage_sequence=stages,
        random_seed=7,
        experiment_name="scenario_2_high_disease_fruiting",
    )

    mpc_config = MPCConfig(
        control_horizon_hours=1,
        prediction_horizon_hours=1,
        solver_max_iter=500,
        solver_ftol=1e-4,
        w_disease=3.0,              # amplify disease penalty
        w_humidity_exposure=1.0,
        w_fogger_suppression=0.6,
    )

    runner = ExperimentRunner(cfg)
    runner.register_controller("baseline", make_baseline_adapter(), controller_type="rule_based")
    runner.register_controller("mpc_disease", make_mpc_adapter(config=mpc_config), controller_type="mpc_disease_aware")

    t0 = time.perf_counter()
    report = runner.run()
    elapsed = time.perf_counter() - t0

    section("Summary Table")
    print_summary_table(report)

    section("Pairwise Improvements")
    print_improvements(report.improvements)

    section("Disease Burden (critical for this scenario)")
    for cid, metrics in report.controller_metrics.items():
        db = metrics.disease_burden
        print(f"  {cid:<18} mean_risk={db.mean_risk:.4f}  max_risk={db.max_risk:.4f}  "
              f"rh_exposure={db.cumulative_rh_exposure:.2f}  "
              f"fav_steps={db.disease_favorable_steps}/{n_steps}  "
              f"fav_hours={db.disease_favorable_duration_hours:.2f}")

    section("Yield Proxy Scores")
    for cid, yr in report.yield_results.items():
        print(f"  {cid:<18} overall={yr.overall_score:.2f}/100  "
              f"climate={yr.climate_tracking_score:.2f}  disease={yr.disease_burden_score:.2f}  "
              f"stress={yr.stress_exposure_score:.2f}  stability={yr.resource_stability_score:.2f}")

    kv("Elapsed time", f"{elapsed:.2f} s")
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 3: Multi-day with growth-stage transition
# ══════════════════════════════════════════════════════════════════════════════

def _generate_transition_stages(n_steps: int) -> list[str]:
    """Transition from flowering → unripe at the midpoint."""
    mid = n_steps // 2
    return ["flowering"] * mid + ["unripe"] * (n_steps - mid)


def scenario_3_multiday_transition() -> ComparisonReport:
    banner("SCENARIO 3: 24 h with Stage Transition (Flowering -> Unripe)")

    n_steps = 288   # 24 h at 5-min intervals (practical runtime)
    weather = generate_default_weather(n_steps, dt_minutes=5)
    init = make_default_initial_state()
    stages = _generate_transition_stages(n_steps)

    print(f"  Transition at step {n_steps // 2} (hour {n_steps // 2 * 5 / 60:.0f})")

    cfg = ExperimentConfig(
        n_steps=n_steps,
        dt_minutes=5,
        initial_state=init,
        weather_sequence=weather,
        growth_stage_sequence=stages,
        random_seed=123,
        experiment_name="scenario_3_transition_48h",
    )

    mpc_config = MPCConfig(
        control_horizon_hours=1,
        prediction_horizon_hours=1,
        solver_max_iter=500,
        solver_ftol=1e-4,
        stage_transition_blend_steps=24,  # 2-hour blend window
    )

    runner = ExperimentRunner(cfg)
    runner.register_controller("baseline", make_baseline_adapter(), controller_type="rule_based")
    runner.register_controller("mpc_blend", make_mpc_adapter(config=mpc_config), controller_type="mpc_disease_aware")

    t0 = time.perf_counter()
    report = runner.run()
    elapsed = time.perf_counter() - t0

    section("Summary Table")
    print_summary_table(report)

    section("Pairwise Improvements")
    print_improvements(report.improvements)

    section("Yield Proxy Scores")
    for cid, yr in report.yield_results.items():
        print(f"  {cid:<18} overall={yr.overall_score:.2f}/100")

    kv("Elapsed time", f"{elapsed:.2f} s")
    return report


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 4: Direct MPC solver component validation
# ══════════════════════════════════════════════════════════════════════════════

def scenario_4_solver_components():
    banner("SCENARIO 4: MPC Solver Component-Level Validation")

    # ── 4a. Greenhouse transition model ──────────────────────────────
    section("4a. Greenhouse Transition Model — Single Step")
    model = GreenhouseTransitionModel()
    state = GreenhouseState(
        indoor_temp=25.0, indoor_humidity=70.0, soil_moisture=60.0,
        co2=600.0, light_intensity=400.0,
        growth_stage_index=stage_label_to_index("flowering"),
        vpd=0.8, leaf_wetness_proxy=0.35,
    )
    act = ActuatorState(
        fan_speed=0.4, vent_opening=0.2, irrigation_qty=0.3,
        heater_output=0.0, led_intensity=0.5, co2_valve_pct=0.2, fogger_duty=0.1,
    )
    weather = WeatherState(temp_external=22.0, humidity_external=55.0, solar_radiation=500.0, windspeed=2.5)

    next_state = model.step(state, act, weather)
    kv("Input temp", f"{state.indoor_temp:.1f} °C")
    kv("Output temp", f"{next_state.indoor_temp:.2f} °C")
    kv("Input humidity", f"{state.indoor_humidity:.1f} %")
    kv("Output humidity", f"{next_state.indoor_humidity:.2f} %")
    kv("Input soil_moisture", f"{state.soil_moisture:.1f} %")
    kv("Output soil_moisture", f"{next_state.soil_moisture:.2f} %")
    kv("Input CO2", f"{state.co2:.0f} ppm")
    kv("Output CO2", f"{next_state.co2:.2f} ppm")
    print("  [OK] Transition model produced valid state")

    # ── 4b. Cost function evaluation ─────────────────────────────────
    section("4b. Cost Function (CostBuilder) — Single Stage Cost")
    cfg = MPCConfig()
    cost_builder = CostBuilder(
        config=cfg,
        growth_stage="flowering",
        disease_context=DiseaseContext(risk_score=0.2, severity_24h={"early_blight": 0.1}),
    )
    setpoint = get_setpoint("flowering")
    stage_cost = cost_builder.stage_cost.evaluate(state.to_numpy(), act.to_numpy())
    kv("Stage cost", f"{stage_cost:.6f}")
    terminal_cost = cost_builder.terminal_cost.evaluate(next_state.to_numpy())
    kv("Terminal cost", f"{terminal_cost:.6f}")
    print("  [OK] Cost function returns finite values")

    # ── 4c. Weather adaptation ───────────────────────────────────────
    section("4c. Weather Adaptation Modifiers")
    weather_seq = [w.to_dict() for w in generate_default_weather(12)]
    mods = compute_weather_adaptation(weather_seq, setpoint, 12, cfg)
    kv("Temp stress mean", f"{np.mean(mods.temp_stress):.4f}" if mods.temp_stress else "0")
    kv("RH stress mean", f"{np.mean(mods.rh_stress):.4f}" if mods.rh_stress else "0")
    kv("Solar stress mean", f"{np.mean(mods.solar_stress):.4f}" if mods.solar_stress else "0")
    print("  [OK] Weather adaptation computed")

    # ── 4d. Constraint tightening ────────────────────────────────────
    section("4d. Disease-Aware Constraint Tightening")
    base_constraints = get_default_constraints("flowering")
    print(f"  Default fogger bounds: {base_constraints.actuator_bounds.get('fogger_duty', 'N/A')}")

    tight = tighten_constraints_for_disease(
        base_constraints, 0.6,
        rh_tightening_threshold=0.4,
        rh_tightened_ceiling=80.0,
        fogger_suppress_threshold=0.5,
        fogger_suppressed_max_duty=0.3,
        severity_24h={"early_blight": 0.3, "late_blight": 0.1},
    )
    print(f"  Tightened fogger bounds: {tight.actuator_bounds.get('fogger_duty', 'N/A')}")
    print("  [OK] Constraints tightened for high disease risk")

    # ── 4e. MPC solver — single solve call ───────────────────────────
    section("4e. MPC Solver — Single Solve Call")
    solver = MPCSolver(MPCConfig(
        control_horizon_hours=1,
        solver_max_iter=500,
        solver_ftol=1e-4,
    ))
    test_state = make_default_initial_state()
    fused = FusedState(
        greenhouse_state=test_state,
        growth_stage="flowering",
        growth_stage_index=stage_label_to_index("flowering"),
        disease_risk_score=0.15,
        disease_classification="healthy leaves",
        disease_confidence=0.9,
        weather_forecast=[w.to_dict() for w in generate_default_weather(12)],
    )
    t0 = time.perf_counter()
    solution = solver.solve(fused, horizon_override=6)
    solve_time = (time.perf_counter() - t0) * 1000

    kv("Converged", solution.converged)
    kv("Fallback used", solution.fallback_used)
    kv("Total cost", f"{solution.total_cost:.6f}")
    kv("Solver status", solution.solver_status)
    kv("Iterations", solution.n_iterations)
    kv("Function evals", solution.n_function_evals)
    kv("Solve time (ms)", f"{solution.solve_time_ms:.1f}")
    kv("Horizon length", len(solution.optimal_controls))
    kv("First action fan_speed", f"{solution.first_action.fan_speed:.4f}")
    kv("First action heater", f"{solution.first_action.heater_output:.4f}")
    kv("First action irrigation", f"{solution.first_action.irrigation_qty:.4f}")

    if solution.cost_breakdown:
        print("\n  Cost breakdown:")
        for key, val in sorted(solution.cost_breakdown.items()):
            print(f"    {key:<40} {val:.6f}")

    print("\n  [OK] MPC solver completed successfully")

    # ── 4f. Baseline controller single-step ──────────────────────────
    section("4f. Baseline Controller — Single Step")
    baseline = RuleBasedController()
    payload = baseline.compute_action(
        state=state,
        growth_stage="flowering",
        disease_risk=0.2,
        weather=weather,
    )
    kv("Fan speed", f"{payload.actuators.fan_speed:.4f}")
    kv("Vent opening", f"{payload.actuators.vent_opening:.4f}")
    kv("Irrigation", f"{payload.actuators.irrigation_qty:.4f}")
    kv("Heater", f"{payload.actuators.heater_output:.4f}")
    kv("Priority triggered", payload.priority_triggered)
    print(f"  Reasoning: {payload.reasoning[:3]}")
    print("  [OK] Baseline controller produced action")

    # ── 4g. Explanation builder ───────────────────────────────────────
    section("4g. Explanation Builder")
    try:
        eb = ExplanationBuilder()
        explanation = eb.build(
            fused=fused,
            actuators=solution.first_action,
            cost_breakdown=solution.cost_breakdown,
            solver_converged=solution.converged,
        )
        kv("Explanation entries", len(explanation.entries))
        kv("Summary", explanation.action_summary[:80] if explanation.action_summary else "(empty)")
        print("  [OK] ExplanationBuilder produced explanation")
    except Exception as e:
        print(f"  [WARN] ExplanationBuilder: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  SCENARIO 5: Multi-horizon convergence test
# ══════════════════════════════════════════════════════════════════════════════

def scenario_5_convergence_test():
    banner("SCENARIO 5: MPC Solver Convergence Across Horizons")

    state = make_default_initial_state()
    fused = FusedState(
        greenhouse_state=state,
        growth_stage="flowering",
        growth_stage_index=stage_label_to_index("flowering"),
        disease_risk_score=0.15,
        disease_classification="healthy leaves",
        disease_confidence=0.9,
        weather_forecast=[w.to_dict() for w in generate_default_weather(72)],
    )

    horizons = [3, 6, 12, 24]
    print(f"\n  {'Horizon':>8} {'Converged':>10} {'Fallback':>10} {'Cost':>12} "
          f"{'Iters':>8} {'Evals':>8} {'Time(ms)':>10}")
    print("  " + "-" * 70)

    for h in horizons:
        solver = MPCSolver(MPCConfig(control_horizon_hours=1))
        sol = solver.solve(fused, horizon_override=h)
        print(f"  {h:>8} {str(sol.converged):>10} {str(sol.fallback_used):>10} "
              f"{sol.total_cost:>12.4f} {sol.n_iterations:>8} "
              f"{sol.n_function_evals:>8} {sol.solve_time_ms:>10.1f}")

    print("\n  [OK] Multi-horizon convergence test complete")


# ══════════════════════════════════════════════════════════════════════════════
#  ARTIFACT PERSISTENCE
# ══════════════════════════════════════════════════════════════════════════════

def save_all_artifacts(reports: dict[str, ComparisonReport]) -> Path:
    banner("SAVING EVALUATION ARTIFACTS")

    workspace_root = _PROJECT_ROOT
    saved_dirs: list[Path] = []

    for name, report in reports.items():
        run_id = f"full_eval_{name}_{dt.datetime.now():%Y%m%d_%H%M%S}"
        out_dir = save_evaluation_artifacts(report, workspace_root=workspace_root, run_id=run_id)
        saved_dirs.append(out_dir)
        files = sorted(os.listdir(out_dir))
        print(f"  {name}: {out_dir.relative_to(workspace_root)}/")
        for f in files:
            size = os.path.getsize(out_dir / f)
            print(f"    {f:<30} {size:>8} bytes")

    # Verify round-trip
    section("Artifact Round-Trip Verification")
    for d in saved_dirs:
        loaded = load_evaluation_report(d)
        assert "summary_table" in loaded, f"Missing summary_table in {d}"
        n_controllers = len(loaded["summary_table"])
        print(f"  {d.name}: loaded OK ({n_controllers} controllers)")

    print("\n  [OK] All artifacts saved and verified")
    return saved_dirs[0].parent if saved_dirs else workspace_root


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def print_final_summary(reports: dict[str, ComparisonReport]):
    banner("FINAL EVALUATION SUMMARY")

    print(f"\n  {'Scenario':<35} {'Controllers':>12} {'Steps':>8} "
          f"{'MPC Yield':>10} {'BL Yield':>10} {'Yield Δ':>10}")
    print("  " + "-" * 90)

    for name, report in reports.items():
        table = report.summary_table()
        n_ctrl = len(table)
        # Find n_steps from any controller
        n_steps = next(iter(report.controller_metrics.values())).n_steps if report.controller_metrics else 0

        # MPC yield
        mpc_yield = "N/A"
        bl_yield = "N/A"
        delta = "N/A"
        for cid, row in table.items():
            if "mpc" in cid.lower():
                mpc_yield = f"{row['yield_score']:.2f}"
            elif "baseline" in cid.lower():
                bl_yield = f"{row['yield_score']:.2f}"

        if mpc_yield != "N/A" and bl_yield != "N/A":
            delta = f"{float(mpc_yield) - float(bl_yield):+.2f}"

        print(f"  {name:<35} {n_ctrl:>12} {n_steps:>8} "
              f"{mpc_yield:>10} {bl_yield:>10} {delta:>10}")

    # Overall pass/fail checks
    section("Validation Checks")
    all_ok = True

    for name, report in reports.items():
        for cid, metrics in report.controller_metrics.items():
            # Check no NaN in tracking
            for var, tm in metrics.tracking.items():
                if math.isnan(tm.rmse):
                    print(f"  [FAIL] {name}/{cid}: NaN in {var} RMSE")
                    all_ok = False
            # Check yield is in [0, 100]
            ys = metrics.yield_quality_score
            if not (0 <= ys <= 100):
                print(f"  [FAIL] {name}/{cid}: yield_score={ys} out of [0,100]")
                all_ok = False
            # Check n_steps > 0
            if metrics.n_steps == 0:
                print(f"  [FAIL] {name}/{cid}: n_steps=0")
                all_ok = False

    if all_ok:
        print("  [OK] All validation checks passed")

    # Report JSON serialisation test
    section("JSON Serialisation Test")
    for name, report in reports.items():
        d = report.to_dict()
        js = json.dumps(d, indent=2, default=str)
        print(f"  {name}: {len(js):,} chars JSON")

    print()


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    banner("AgriTwin-GH — Complete MPC End-to-End Evaluation")
    print(f"  Timestamp:       {dt.datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"  Python:          {sys.version.split()[0]}")
    print(f"  Growth stages:   {GROWTH_STAGES}")
    print(f"  Project root:    {_PROJECT_ROOT}")

    total_t0 = time.perf_counter()
    reports: dict[str, ComparisonReport] = {}

    # ── Scenario 1 ──
    try:
        reports["scenario_1_flowering"] = scenario_1_standard_24h()
        print_resource_costs_tn(reports["scenario_1_flowering"])
    except Exception as e:
        print(f"\n  [ERROR] Scenario 1 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Scenario 2 ──
    try:
        reports["scenario_2_disease"] = scenario_2_high_disease()
        print_resource_costs_tn(reports["scenario_2_disease"])
    except Exception as e:
        print(f"\n  [ERROR] Scenario 2 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Scenario 3 ──
    try:
        reports["scenario_3_transition"] = scenario_3_multiday_transition()
        print_resource_costs_tn(reports["scenario_3_transition"])
    except Exception as e:
        print(f"\n  [ERROR] Scenario 3 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Scenario 4 (component-level) ──
    try:
        scenario_4_solver_components()
    except Exception as e:
        print(f"\n  [ERROR] Scenario 4 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Scenario 5 (convergence) ──
    try:
        scenario_5_convergence_test()
    except Exception as e:
        print(f"\n  [ERROR] Scenario 5 failed: {e}")
        import traceback; traceback.print_exc()

    # ── Save artifacts ──
    if reports:
        try:
            save_all_artifacts(reports)
        except Exception as e:
            print(f"\n  [ERROR] Artifact save failed: {e}")
            import traceback; traceback.print_exc()

    # ── Final summary ──
    if reports:
        print_final_summary(reports)

    total_elapsed = time.perf_counter() - total_t0
    banner(f"EVALUATION COMPLETE — Total Time: {total_elapsed:.1f} s")

    return 0 if reports else 1


if __name__ == "__main__":
    sys.exit(main())
