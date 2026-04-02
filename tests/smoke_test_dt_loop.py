"""
Smoke test for the Digital Twin closed-loop layer (Prompts 2–7).

Validates the full DT stack end-to-end without a database:
  1. Imports                 — all DT modules load cleanly.
  2. Input provider          — SyntheticInputProvider accepts stage & generates state/weather.
  3. DT engine               — single step produces valid diagnostics + attribution.
  4. DTLoop construction     — builds with all defaults and validates parameters.
  5. Multi-step execution    — runs 12 steps, MPC fires, state evolves.
  6. Image observer          — SyntheticImageObserver returns observations on cadence.
  7. Output writer           — fanout_step_to_writer populates all streams.
  8. Artifact manager        — run folder created, metadata + summary persisted.
  9. Logger / summary        — DTLoopRunSummary has enriched min/max/tracking fields.
 10. Error handling          — invalid stage / negative steps raise ValueError.
 11. Schema compatibility    — DTLoopStepResult.to_dict() round-trips cleanly.
 12. Integration cross-check — FusedState ↔ MPCSolver interaction.

Usage:
    python tests/smoke_test_dt_loop.py
"""

import datetime as dt
import json
import sys
import tempfile

sys.path.insert(0, "src")

PASS = 0
FAIL = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        msg = f"  [FAIL] {label}"
        if detail:
            msg += f"  ({detail})"
        print(msg)


# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  DT Loop — Comprehensive Smoke Test")
print("=" * 60)

# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n1. Imports")
try:
    from agritwin_gh.mpc import (
        # Data structures
        GreenhouseState, ActuatorState, WeatherState, FusedState,
        GreenhouseModelParams, GreenhouseTransitionModel,
        # DT state
        DTSnapshot, DTStepInput, DTStepOutput, DTDiagnostics,
        DigitalTwinEngine, DigitalTwinPlant,
        # DT runtime
        prepare_initial_state, prepare_weather_sequence,
        prepare_growth_stages, build_mpc_solver, build_fused_state,
        # DT loop
        DTLoop, DTLoopStepResult, MPC_CADENCE_STEPS, IMAGE_CADENCE_STEPS,
        should_force_mpc_update,
        # DT logger
        DTLoopLogger, DTLoopRunSummary,
        # DT I/O
        DTInputProvider, ImageObservation, SyntheticInputProvider,
        ImageObserver, SyntheticImageObserver,
        DTOutputWriter, JsonFileOutputWriter, fanout_step_to_writer,
        # DT artifact manager
        DTArtifactManager,
        # MPC solver
        MPCSolver, MPCSolution,
        # Constants
        GROWTH_STAGES, DT_MINUTES,
    )
    check("All DT imports OK", True)
except ImportError as exc:
    check("All DT imports OK", False, str(exc))
    sys.exit(1)

# ── 2. Input provider ────────────────────────────────────────────────────────
print("\n2. SyntheticInputProvider")
start = dt.datetime(2026, 4, 2, 10, 0)
provider = SyntheticInputProvider(
    growth_stage="flowering",
    start_time=start,
    base_temp=22.0,
)
check("growth_stage property", provider.growth_stage == "flowering")
check("start_time property", provider.start_time == start)

init_state = provider.get_initial_state()
check(
    "initial state valid",
    5 < init_state.indoor_temp < 45 and 20 < init_state.indoor_humidity < 99,
    f"T={init_state.indoor_temp:.1f}, RH={init_state.indoor_humidity:.1f}",
)

weather = provider.get_weather_sequence(n_steps=24, dt_minutes=5)
check("weather sequence length", len(weather) == 24)
check(
    "weather has temp_external",
    hasattr(weather[0], "temp_external") and weather[0].temp_external > -50,
)

# ── 3. DT engine — single step ───────────────────────────────────────────────
print("\n3. DT Engine single step")
engine = DigitalTwinEngine()
step_in = DTStepInput(
    current_state=init_state,
    action=ActuatorState(heater_output=0.5, fan_speed=0.3),
    weather=weather[0],
    growth_stage="flowering",
    disease_risk_score=0.2,
    disease_classification="healthy leaves",
    dt_minutes=5,
    step_index=0,
    timestamp=start,
)
step_out = engine.step(step_in)
check("step returns DTStepOutput", isinstance(step_out, DTStepOutput))
check("next_state is GreenhouseState", isinstance(step_out.next_state, GreenhouseState))
check(
    "diagnostics has energy",
    step_out.diagnostics.energy_kwh >= 0,
    f"energy={step_out.diagnostics.energy_kwh:.6f}",
)
check(
    "effect_attribution populated",
    len(step_out.diagnostics.effect_attribution) > 0,
    f"vars={list(step_out.diagnostics.effect_attribution.keys())}",
)
check(
    "setpoint_error populated",
    "indoor_temp" in step_out.diagnostics.setpoint_error,
)

# ── 4. DTLoop construction ───────────────────────────────────────────────────
print("\n4. DTLoop construction & validation")
loop = DTLoop(
    growth_stage="flowering",
    start_time=start,
    n_steps=12,
    weather_base_temp=22.0,
    input_provider=provider,
)
check("DTLoop created", loop is not None)
check("engine accessible", loop.engine is not None)
check("initial_state accessible", loop.initial_state.indoor_temp > 0)
check("growth_stage correct", loop.growth_stage == "flowering")

# Invalid stage
try:
    DTLoop(growth_stage="invalid_stage_xyz", n_steps=1)
    check("invalid stage raises ValueError", False)
except ValueError:
    check("invalid stage raises ValueError", True)

# Negative n_steps
try:
    DTLoop(growth_stage="flowering", n_steps=-1)
    check("negative n_steps raises ValueError", False)
except ValueError:
    check("negative n_steps raises ValueError", True)

# ── 5. Multi-step execution ──────────────────────────────────────────────────
print("\n5. Multi-step execution (12 steps)")
results = list(loop.run())
check("got 12 results", len(results) == 12)

mpc_steps = [r for r in results if r.mpc_ran_this_step]
check("MPC fired at least once", len(mpc_steps) >= 1)
check(
    f"MPC cadence ~{MPC_CADENCE_STEPS} steps",
    len(mpc_steps) == 12 // MPC_CADENCE_STEPS,
    f"expected {12 // MPC_CADENCE_STEPS}, got {len(mpc_steps)}",
)

first = results[0]
last = results[-1]
check(
    "state evolved over 12 steps",
    abs(first.next_state.indoor_temp - last.next_state.indoor_temp) > 0.01
    or abs(first.next_state.indoor_humidity - last.next_state.indoor_humidity) > 0.01,
)

# current_state and weather_used fields (Prompt 6 additions)
check(
    "current_state populated",
    first.current_state.indoor_temp > 0,
    f"T={first.current_state.indoor_temp:.1f}",
)
check(
    "weather_used populated",
    first.weather_used.temp_external != 0.0 or first.weather_used.solar_radiation != 0.0,
)
check(
    "current_state != next_state",
    abs(first.current_state.indoor_temp - first.next_state.indoor_temp) > 0.001
    or abs(first.current_state.indoor_humidity - first.next_state.indoor_humidity) > 0.001,
)

# ── 6. Image observer ────────────────────────────────────────────────────────
print("\n6. Image observer")
img_steps = [r for r in results if r.image_refresh_this_step]
check(
    "image refreshes on cadence",
    len(img_steps) == 12 // IMAGE_CADENCE_STEPS,
    f"expected {12 // IMAGE_CADENCE_STEPS}, got {len(img_steps)}",
)
if img_steps:
    obs = img_steps[0].image_observation
    check("ImageObservation has growth_stage_label", obs.growth_stage_label != "")
    check("ImageObservation has disease_label", obs.disease_label != "")
    check("ImageObservation source is synthetic", obs.source == "synthetic")

# ── 7. Output writer ─────────────────────────────────────────────────────────
print("\n7. Output writer + fanout")
with tempfile.TemporaryDirectory() as td:
    writer = JsonFileOutputWriter(output_dir=td, run_tag="smoke")
    for r in results:
        fanout_step_to_writer(r, writer)

    logger_obj = DTLoopLogger(growth_stage="flowering", console_every=0)
    for r in results:
        logger_obj.log_step(r)
    summary = logger_obj.summary()
    writer.write_summary(summary.to_dict())

    paths = writer.flush()
    check("4 artifact files created", len(paths) == 4, f"got {len(paths)}")

    # Verify state_log has the enriched structure
    import pathlib
    state_file = pathlib.Path(td) / "dt_state_smoke.json"
    with open(state_file) as f:
        state_data = json.load(f)
    first_rec = state_data[0]
    check(
        "state record has current_state",
        "current_state" in first_rec,
    )
    check(
        "state record has weather",
        "weather" in first_rec,
    )
    check(
        "state record has next_state",
        "next_state" in first_rec,
    )

# ── 8. Artifact manager ──────────────────────────────────────────────────────
print("\n8. DTArtifactManager")
with tempfile.TemporaryDirectory() as td:
    mgr = DTArtifactManager(base_dir=td, run_id="smoke_test_run")
    check("run_id set", mgr.run_id == "smoke_test_run")
    check("run_dir created", mgr.run_dir.exists())

    # Metadata save
    meta_path = mgr.save_run_metadata({
        "growth_stage": "flowering",
        "hours": 1.0,
        "dt_minutes": 5,
    })
    check("run_metadata.json saved", meta_path.exists())
    with open(meta_path) as f:
        meta = json.load(f)
    check("metadata has run_id", meta.get("run_id") == "smoke_test_run")

    # Writer creation
    w = mgr.create_output_writer()
    check("create_output_writer returns JsonFileOutputWriter", isinstance(w, JsonFileOutputWriter))

    # Summary save
    sum_path = mgr.save_summary(summary.to_dict())
    check("summary.json saved", sum_path.exists())

# ── 9. Logger enriched summary ───────────────────────────────────────────────
print("\n9. DTLoopRunSummary enriched fields")
check("has min_temp", hasattr(summary, "min_temp") and summary.min_temp > 0)
check("has max_temp", hasattr(summary, "max_temp") and summary.max_temp > 0)
check("min_temp <= max_temp", summary.min_temp <= summary.max_temp)
check("has min_humidity", summary.min_humidity > 0)
check("has max_humidity", summary.max_humidity > 0)
check("has min_soil_moisture", summary.min_soil_moisture > 0)
check("has max_soil_moisture", summary.max_soil_moisture > 0)
check(
    "has mean_setpoint_error_temp",
    hasattr(summary, "mean_setpoint_error_temp") and summary.mean_setpoint_error_temp >= 0,
)
check(
    "has mean_setpoint_error_humidity",
    summary.mean_setpoint_error_humidity >= 0,
)
summary_d = summary.to_dict()
check("to_dict has min/max keys", "min_temp" in summary_d and "max_soil_moisture" in summary_d)

# ── 10. Error handling ────────────────────────────────────────────────────────
print("\n10. Error handling")

# Invalid stage in SyntheticInputProvider
try:
    SyntheticInputProvider(growth_stage="nonexistent")
    check("SyntheticInputProvider rejects bad stage", False)
except ValueError:
    check("SyntheticInputProvider rejects bad stage", True)

# should_force_mpc_update with extreme state
extreme_state = GreenhouseState(indoor_temp=5.0, indoor_humidity=95.0)
check("force MPC on extreme state", should_force_mpc_update(extreme_state) is True)

safe_state = GreenhouseState(indoor_temp=22.0, indoor_humidity=55.0, disease_risk_score=0.1)
check("no force MPC on safe state", should_force_mpc_update(safe_state) is False)

# ── 11. Schema round-trip ─────────────────────────────────────────────────────
print("\n11. Schema round-trip")
r0 = results[0]
d = r0.to_dict()
check("to_dict returns dict", isinstance(d, dict))
check("to_dict has current_state", "current_state" in d)
check("to_dict has weather_used", "weather_used" in d)
check("to_dict has next_state", "next_state" in d)
check("to_dict has diagnostics", "diagnostics" in d)
check("to_dict has cadence_info", "cadence_info" in d)

# Verify JSON-serializable
try:
    json_str = json.dumps(d, default=str)
    check("to_dict is JSON-serializable", len(json_str) > 100)
except (TypeError, ValueError) as exc:
    check("to_dict is JSON-serializable", False, str(exc))

# ── 12. Integration: FusedState ↔ MPCSolver ───────────────────────────────────
print("\n12. Integration: FusedState ↔ MPCSolver")
solver = build_mpc_solver()
fused = build_fused_state(
    state=init_state,
    growth_stage="flowering",
    disease_risk=0.25,
    weather_forecast=[w.to_dict() for w in weather[:12]],
    timestamp=start,
)
check("FusedState built", isinstance(fused, FusedState))
check("FusedState has setpoint", fused.setpoint is not None)
check("FusedState has constraints", fused.constraints is not None)

solution = solver.solve(fused)
check("MPCSolver.solve() returns MPCSolution", isinstance(solution, MPCSolution))
check("first_action is ActuatorState", isinstance(solution.first_action, ActuatorState))
check(
    "solver converged or fallback used",
    solution.converged or solution.fallback_used,
)

# ═══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
total = PASS + FAIL
print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
if FAIL == 0:
    print("  STATUS: ALL CHECKS PASSED")
else:
    print("  STATUS: SOME CHECKS FAILED")
print("=" * 60)

sys.exit(0 if FAIL == 0 else 1)
