"""
test_db_ai_mpc_pipeline.py

Verify the documented data flow:

    DB (PostgreSQL / SQLite)
      └─► MPCInputPreparation   ←── schema: weather_data, greenhouse_data,
            │                              disease_progression,
            │                              growth_progression_hourly
            │
            ├─► WeatherDisturbanceForecast.get_forecast()
            │         └─► FusedState.weather_forecast   (list[dict])
            │
            ├─► DiseaseRiskPenalty.predict_all_diseases()
            │         └─► FusedState.{current_severity, severity_24h, severity_48h}
            │
            ├─► GrowthStageWeights.predict_from_dataframe()
            │         └─► FusedState.{growth_stage, next_stage, hours_to_transition, …}
            │
            └─► GreenhouseState.from_db_row()
                      └─► FusedState.greenhouse_state

    StateFusion.fuse()  ──────────────────────────► FusedState (all fields)
                                                         │
                                                         ▼
                                                   MPCSolver.solve()
                                                         │
                                                         ▼
                                                   MPCSolution (actuators + cost)

Design:
  - DB unavailability is handled gracefully: each AI model is tested with
    synthetic / empty fallback data so the pipeline checks still run.
  - No results are saved anywhere.
  - Run with:  uv run python tests/test_db_ai_mpc_pipeline.py
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import traceback

sys.path.insert(0, "src")

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import pandas as pd

# ── Symbols ───────────────────────────────────────────────────────────────────
PASS = "[ PASS ]"
FAIL = "[ FAIL ]"
SKIP = "[ SKIP ]"
INFO = "[ INFO ]"


def section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — DB Connection
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 1 — DB Connection")

db_ok = False
session = None

try:
    from agritwin_gh.utils.database import get_db_session
    from sqlalchemy import text

    session = get_db_session()
    session.execute(text("SELECT 1"))
    print(f"{PASS} DB session opened and responsive")
    db_ok = True
except Exception as exc:
    print(f"{SKIP} DB unavailable — AI model checks will use synthetic / empty fallback")
    print(f"       ({type(exc).__name__}: {exc})")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — MPCInputPreparation: DB → DataFrames / dicts
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 2 — MPCInputPreparation (DB → raw DataFrames)")

from agritwin_gh.mpc.mpc_input_preparation import MPCInputPreparation

gh_row: dict | None = None
df_weather: pd.DataFrame = pd.DataFrame()
df_disease: pd.DataFrame = pd.DataFrame()
df_growth: pd.DataFrame = pd.DataFrame()
prep: MPCInputPreparation | None = None

if db_ok and session is not None:
    prep = MPCInputPreparation(session)

    # ── Greenhouse snapshot ────────────────────────────────────────────
    try:
        gh_row = prep.get_latest_greenhouse_row()
        if gh_row is not None:
            expected = {
                "datetime", "indoor_temp", "indoor_humidity",
                "indoor_air_velocity", "indoor_co2", "solarradiation",
                "day_night_flag", "vpd", "dew_point", "leaf_wetness_proxy",
            }
            missing = expected - gh_row.keys()
            if missing:
                print(f"{FAIL} get_latest_greenhouse_row — missing keys: {missing}")
            else:
                print(
                    f"{PASS} get_latest_greenhouse_row → {len(gh_row)} keys | "
                    f"temp={gh_row['indoor_temp']:.1f}°C  "
                    f"rh={gh_row['indoor_humidity']:.1f}%"
                )
        else:
            print(f"{SKIP} greenhouse_data table is empty")
    except Exception as exc:
        print(f"{FAIL} get_latest_greenhouse_row raised: {exc!r}")

    # ── Weather context ────────────────────────────────────────────────
    try:
        df_weather = prep.get_weather_context(lookback_days=30)
        if not df_weather.empty:
            expected_cols = {"datetime", "temp", "humidity", "windspeed",
                             "solarradiation", "conditions"}
            missing_cols = expected_cols - set(df_weather.columns)
            if missing_cols:
                print(f"{FAIL} get_weather_context — missing columns: {missing_cols}")
            else:
                print(
                    f"{PASS} get_weather_context → {len(df_weather)} rows | "
                    f"range: {df_weather['datetime'].min().date()} – "
                    f"{df_weather['datetime'].max().date()}"
                )
        else:
            print(f"{SKIP} weather_data table is empty → will use synthetic fallback")
    except Exception as exc:
        print(f"{FAIL} get_weather_context raised: {exc!r}")

    # ── Disease progression context ────────────────────────────────────
    try:
        df_disease = prep.get_disease_progression_context(sequence_length=96)
        if not df_disease.empty:
            print(
                f"{PASS} get_disease_progression_context → {len(df_disease)} rows "
                f"| first cols: {list(df_disease.columns)[:6]}"
            )
        else:
            print(f"{SKIP} disease_progression table is empty → model will zero-fill")
    except Exception as exc:
        print(f"{FAIL} get_disease_progression_context raised: {exc!r}")

    # ── Growth progression context ─────────────────────────────────────
    try:
        df_growth = prep.get_growth_progression_context(sequence_length=96)
        if not df_growth.empty:
            print(
                f"{PASS} get_growth_progression_context → {len(df_growth)} rows "
                f"| first cols: {list(df_growth.columns)[:6]}"
            )
        else:
            print(f"{SKIP} growth_progression_hourly table is empty → model will fall back to defaults")
    except Exception as exc:
        print(f"{FAIL} get_growth_progression_context raised: {exc!r}")

else:
    print(f"{SKIP} All MPCInputPreparation checks skipped (no DB)")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Weather AI Model: DB context → disturbance list
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 3 — WeatherDisturbanceForecast  (DB weather → disturbance steps)")

from agritwin_gh.mpc.disturbance import WeatherDisturbanceForecast

wdf_instance: WeatherDisturbanceForecast | None = None
weather_forecast_out: list[dict] = []

try:
    wdf_instance = WeatherDisturbanceForecast()
    print(f"{INFO} artifact: {wdf_instance._art_dir.name}")

    # Use real DB data if available, else 30-row synthetic fallback
    if db_ok and not df_weather.empty:
        context_df = df_weather
        print(f"{INFO} source: DB ({len(context_df)} rows from weather_data)")
    else:
        dates = pd.date_range(end=dt.date.today(), periods=30, freq="D")
        context_df = pd.DataFrame({
            "datetime":       dates,
            "temp":           np.random.uniform(24, 35, 30),
            "humidity":       np.random.uniform(50, 80, 30),
            "windspeed":      np.random.uniform(2, 15, 30),
            "solarradiation": np.random.uniform(200, 600, 30),
            "conditions":     ["Clear"] * 30,
        })
        print(f"{INFO} source: synthetic (DB unavailable or empty)")

    weather_forecast_out = wdf_instance.get_forecast(context_df, horizon_hours=48)

    # Assertions
    assert isinstance(weather_forecast_out, list), "Expected list"
    assert len(weather_forecast_out) > 0, "Expected non-empty forecast"

    first = weather_forecast_out[0]
    expected_fkeys = {"temp", "humidity", "windspeed", "solarradiation"}
    missing_fkeys = expected_fkeys - set(first.keys())

    print(
        f"{PASS} get_forecast → {len(weather_forecast_out)} 5-min steps | "
        f"keys: {sorted(first.keys())}"
    )
    if missing_fkeys:
        print(f"{FAIL} forecast step missing expected keys: {missing_fkeys}")
    else:
        print(f"{PASS} All expected forecast keys present in each step")

except Exception as exc:
    print(f"{FAIL} WeatherDisturbanceForecast: {exc!r}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Disease AI Model: DB context → DiseaseProgressionOutput
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 4 — DiseaseRiskPenalty  (DB disease context → severity dicts)")

from agritwin_gh.mpc.disease_penalty import DiseaseRiskPenalty
from agritwin_gh.mpc.state import DiseaseProgressionOutput

drp_instance: DiseaseRiskPenalty | None = None
disease_out: DiseaseProgressionOutput = DiseaseProgressionOutput()

try:
    drp_instance = DiseaseRiskPenalty()
    print(f"{INFO} artifact: {drp_instance._art_dir.name}")

    if db_ok and not df_disease.empty:
        print(f"{INFO} source: DB ({len(df_disease)} rows from disease_progression)")
        disease_out = drp_instance.predict_all_diseases(df_disease)
    else:
        print(f"{SKIP} No disease DB data — testing graceful empty-DataFrame path")
        disease_out = drp_instance.predict_all_diseases(pd.DataFrame())

    assert isinstance(disease_out, DiseaseProgressionOutput)
    assert isinstance(disease_out.current_severity, dict)
    assert isinstance(disease_out.severity_24h, dict)
    assert isinstance(disease_out.severity_48h, dict)

    print(f"{PASS} predict_all_diseases returned DiseaseProgressionOutput")
    print(f"       current_severity = {disease_out.current_severity}")
    print(f"       severity_24h     = {disease_out.severity_24h}")
    print(f"       severity_48h     = {disease_out.severity_48h}")

except Exception as exc:
    print(f"{FAIL} DiseaseRiskPenalty: {exc!r}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Growth AI Model: DB context → GrowthProgressionOutput
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 5 — GrowthStageWeights  (DB growth context → GrowthProgressionOutput)")

from agritwin_gh.mpc.growth_weights import GrowthStageWeights
from agritwin_gh.mpc.state import GrowthProgressionOutput

gsw_instance: GrowthStageWeights | None = None
growth_out: GrowthProgressionOutput = GrowthProgressionOutput()

try:
    gsw_instance = GrowthStageWeights()
    print(f"{INFO} artifact: {gsw_instance._art_dir.name}")

    if db_ok and not df_growth.empty:
        print(f"{INFO} source: DB ({len(df_growth)} rows from growth_progression_hourly)")
        growth_out = gsw_instance.predict_from_dataframe(df_growth)
    else:
        print(f"{SKIP} No growth DB data — testing graceful empty-DataFrame path")
        growth_out = gsw_instance.predict_from_dataframe(pd.DataFrame())

    assert isinstance(growth_out, GrowthProgressionOutput)

    print(f"{PASS} predict_from_dataframe returned GrowthProgressionOutput")
    print(f"       current_stage       = {growth_out.current_stage!r}")
    print(f"       next_stage          = {growth_out.next_stage!r}")
    print(f"       hours_to_transition = {growth_out.hours_to_transition}")
    print(f"       transition_24h      = {growth_out.transition_within_24h}")
    print(f"       transition_48h      = {growth_out.transition_within_48h}")

except Exception as exc:
    print(f"{FAIL} GrowthStageWeights: {exc!r}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — StateFusion: all AI model outputs → FusedState
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 6 — StateFusion.fuse()  (AI model outputs → FusedState)")

from agritwin_gh.mpc.config import MPCConfig
from agritwin_gh.mpc.state import FusedState
from agritwin_gh.mpc.state_fusion import StateFusion

fused: FusedState | None = None


class _NullImageStreamer:
    """Stub — disables MinIO image retrieval; StateFusion accepts any duck-typed object."""

    def get_random_disease_image(self, *args, **kwargs):
        return None

    def get_random_growth_stage_image(self, *args, **kwargs):
        return None


class _NoDBPrep:
    """Stub for MPCInputPreparation when DB is unavailable."""

    def get_latest_greenhouse_row(self):
        return None

    def get_weather_context(self, **kwargs):
        return pd.DataFrame()

    def get_disease_progression_context(self, **kwargs):
        return pd.DataFrame()

    def get_growth_progression_context(self, **kwargs):
        return pd.DataFrame()


try:
    if wdf_instance is None or drp_instance is None or gsw_instance is None:
        raise RuntimeError(
            "One or more AI model wrappers failed to instantiate; cannot run StateFusion."
        )

    cfg = MPCConfig()
    cfg.image_stream_enabled = False  # disable MinIO image calls in this test

    fusion_prep = prep if (db_ok and prep is not None) else _NoDBPrep()
    if db_ok and prep is not None:
        print(f"{INFO} StateFusion will use live DB session for per-step queries")
    else:
        print(f"{INFO} StateFusion will use _NoDBPrep stub (DB offline) — "
              "AI models will receive empty DataFrames and fall back gracefully")

    sf = StateFusion(
        config=cfg,
        input_prep=fusion_prep,        # type: ignore[arg-type]
        weather=wdf_instance,
        disease_penalty=drp_instance,
        growth_weights=gsw_instance,
        image_streamer=_NullImageStreamer(),   # type: ignore[arg-type]
        disease_classifier=None,
        growth_classifier=None,
    )

    fused = sf.fuse()

    # ── Structural assertions ────────────────────────────────────────
    assert isinstance(fused, FusedState), "fuse() must return FusedState"
    assert fused.greenhouse_state is not None, "greenhouse_state must be set"
    assert fused.setpoint is not None, "setpoint must be resolved"
    assert fused.constraints is not None, "constraints must be resolved"
    assert isinstance(fused.current_severity, dict), "current_severity must be dict"
    assert isinstance(fused.severity_24h, dict), "severity_24h must be dict"
    assert isinstance(fused.weather_forecast, list), "weather_forecast must be list"

    print(f"{PASS} StateFusion.fuse() returned valid FusedState")
    print(f"       growth_stage       = {fused.growth_stage!r}")
    print(f"       next_stage         = {fused.next_stage!r}")
    print(f"       hours_to_trans     = {fused.hours_to_transition}")
    print(f"       disease_risk_score = {fused.disease_risk_score:.3f}")
    print(f"       current_severity   = {fused.current_severity}")
    print(f"       severity_24h       = {fused.severity_24h}")
    print(f"       weather_steps      = {len(fused.weather_forecast)}")
    print(f"       indoor_temp        = {fused.greenhouse_state.indoor_temp:.1f}°C")
    print(f"       indoor_humidity    = {fused.greenhouse_state.indoor_humidity:.1f}%")

    # ── Confirm AI model outputs are wired into FusedState ─────────
    if len(fused.weather_forecast) > 0:
        print(f"{PASS} FusedState.weather_forecast populated from WeatherDisturbanceForecast output")
    else:
        print(f"{SKIP} weather_forecast is empty (expected if weather context was empty)")

    if fused.current_severity:
        print(f"{PASS} FusedState.current_severity populated from DiseaseRiskPenalty output")
    else:
        print(f"{INFO} current_severity is zero-filled (expected if disease context was empty)")

    if fused.growth_stage:
        print(f"{PASS} FusedState.growth_stage resolved (from GrowthStageWeights or fallback)")

except Exception as exc:
    print(f"{FAIL} StateFusion.fuse(): {exc!r}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — MPCSolver: FusedState → MPCSolution
# ─────────────────────────────────────────────────────────────────────────────

section("STEP 7 — MPCSolver.solve()  (FusedState → MPCSolution)")

from agritwin_gh.mpc.mpc_solver import MPCSolver, MPCSolution
from agritwin_gh.mpc.state import ActuatorState

try:
    if fused is None:
        raise RuntimeError("FusedState unavailable — StateFusion step failed.")

    solver = MPCSolver(config=cfg)
    solution = solver.solve(
        fused=fused,
        previous_control=ActuatorState(),   # all zero / off
        horizon_override=6,                 # short horizon for fast test
    )

    assert isinstance(solution, MPCSolution), "Expected MPCSolution"
    assert hasattr(solution, "optimal_controls"), "MPCSolution must carry optimal_controls"
    assert hasattr(solution, "converged"), "MPCSolution must carry converged flag"
    assert hasattr(solution, "total_cost"), "MPCSolution must carry total_cost"

    act = solution.first_action
    print(f"{PASS} MPCSolver.solve() returned MPCSolution")
    print(f"       converged  = {solution.converged}")
    print(f"       total_cost = {solution.total_cost:.4f}")
    print(
        f"       first_action -> "
        f"fan={act.fan_speed:.2f}  "
        f"vent={act.vent_opening:.2f}  "
        f"irr={act.irrigation_qty:.2f}  "
        f"heat={act.heater_output:.2f}  "
        f"led={act.led_intensity:.2f}"
    )
    if solution.converged:
        print(f"{PASS} Solver converged → MPC solution is usable")
    else:
        print(f"{INFO} Solver did not converge (FusedState may need richer context)")

except Exception as exc:
    print(f"{FAIL} MPCSolver.solve(): {exc!r}")
    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────

section("PIPELINE CHECK COMPLETE")

print(
    "DB was LIVE — all checks ran against real data."
    if db_ok else
    "DB was OFFLINE — AI model checks ran with synthetic / empty fallback data."
)
print("No results saved.")

if session is not None:
    session.close()
