"""Smoke test for Subprompt 7: output layer, explanations, decision context, hybrid replay."""

import sys
import datetime as dt
from dataclasses import fields as dc_fields

print("=== Import test ===")
from agritwin_gh.mpc import (
    ExplanationEntry, ControllerExplanation, ExplanationBuilder,
    ControllerDecisionContext,
    ReplayConfig, ReplayEngine, ReplayStep, ReplaySummary,
    DigitalTwinStepPayload, DigitalTwinOutput,
    FusedState, GreenhouseState, ActuatorState, WeatherState,
    MPCSolver, MPCSolution,
)
print("All imports OK")

print()
print("=== ExplanationBuilder test ===")
from agritwin_gh.mpc.setpoints import get_setpoint

sp = get_setpoint("flowering")
gs = GreenhouseState(
    indoor_temp=30.5, indoor_humidity=88.0, soil_moisture=55.0,
    co2=420.0, light_intensity=350.0, disease_risk_score=0.55,
    growth_stage_index=3, vpd=1.8, leaf_wetness_proxy=0.7,
)
fused = FusedState(
    greenhouse_state=gs,
    growth_stage="flowering",
    growth_stage_index=3,
    next_stage="unripe",
    hours_to_transition=18.0,
    transition_within_24h=True,
    disease_classification="leaf_mold",
    disease_confidence=0.82,
    disease_risk_score=0.55,
    current_severity={"leaf_mold": 25.0},
    severity_24h={"leaf_mold": 38.0},
    severity_48h={"leaf_mold": 52.0},
    setpoint=sp,
    timestamp=dt.datetime(2025, 7, 15, 14, 0),
)
act = ActuatorState(
    fan_speed=0.8, vent_opening=0.6, irrigation_qty=3.0,
    heater_output=0.0, led_intensity=0.4, co2_valve_pct=0.1,
    fogger_duty=0.05,
)

builder = ExplanationBuilder()
explanation = builder.build(
    fused=fused,
    actuators=act,
    cost_breakdown={"stage_cost_total": 12.5, "disease_risk_score": 0.55},
    solver_converged=True,
    weather_stress={"avg_temp_stress": 0.45, "avg_rh_stress": 0.35, "avg_solar_stress": 0.1},
    tightened_constraints={"rh_ceiling": 88.0, "fogger_suppressed": 0.15},
)
print(f"Dominant factor: {explanation.dominant_factor}")
print(f"Action summary: {explanation.action_summary}")
print(f"Entries ({len(explanation.entries)}):")
for e in explanation.entries:
    print(f"  [{e.severity:8s}] [{e.category:10s}] {e.trigger}: {e.summary}")

d = explanation.to_dict()
assert "entries" in d
assert "dominant_factor" in d
print(f"to_dict: {len(d['entries'])} entries OK")

print()
print("=== ControllerDecisionContext test ===")
ctx = ControllerDecisionContext(
    run_id="test-run-001",
    timestamp=dt.datetime(2025, 7, 15, 14, 0),
    step_index=0,
    solver_config={"method": "SLSQP", "horizon_hours": 6},
    cost_weights={"temp": 10.0, "humidity": 8.0},
    weather_stress_summary={"avg_temp_stress": 0.45},
    solver_performance={"converged": True, "solve_time_ms": 350.0},
    model_ids={"environment_forecast": "wf-run-abc"},
)
d = ctx.to_dict()
assert d["schema_version"] == "1.0"
assert d["run_id"] == "test-run-001"
print(f"DecisionContext to_dict OK, schema={d['schema_version']}")

print()
print("=== DigitalTwinOutput with explanation test ===")
output = DigitalTwinOutput(run_id="test-run-001")
payload = output.format_step(
    fused=fused,
    actuators=act,
    step_cost=12.5,
    energy_kwh=1.2,
    water_litres=3.0,
    cost_breakdown={"stage_cost_total": 12.5},
    solver_converged=True,
    weather_stress={"avg_temp_stress": 0.45, "avg_rh_stress": 0.35},
    tightened_constraints={"rh_ceiling": 88.0},
    decision_context=ctx,
    solver_performance={"converged": True, "solve_time_ms": 350.0},
)
assert payload.explanation != {}
assert payload.decision_context != {}
assert payload.solver_performance != {}
assert payload.explanation["dominant_factor"] != ""
print(f"Payload explanation entries: {len(payload.explanation['entries'])}")
print(f"Payload decision_context run_id: {payload.decision_context['run_id']}")
print(f"Payload solver_performance converged: {payload.solver_performance['converged']}")
print(f"Payload alert: {payload.alert_level} {payload.alert_icons}")

print()
print("=== DigitalTwinStepPayload fields check ===")
names = [f.name for f in dc_fields(DigitalTwinStepPayload)]
for needed in ("explanation", "decision_context", "solver_performance"):
    assert needed in names, f"Missing field: {needed}"
print("All new fields present in DigitalTwinStepPayload")

print()
print("=== ReplayEngine test ===")
rc = ReplayConfig(use_model_predicted_start=False)
print(f"ReplayConfig: replay_id={rc.replay_id}")
print(f"ReplayStep fields: {[f.name for f in dc_fields(ReplayStep)]}")
print(f"ReplaySummary fields: {[f.name for f in dc_fields(ReplaySummary)]}")

print()
print("=== ALL SMOKE TESTS PASSED ===")
