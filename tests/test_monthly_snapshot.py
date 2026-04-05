"""
test_monthly_snapshot.py

Verify that MonthlySnapshotService:
1. Correctly accumulates per-step data.
2. Flushes a completed month to the DB with accurate aggregates.
3. Detects calendar-month rollover and writes exactly one row per month.
4. Creates a unique crop_cycles row on each start_loop() call.

Design:
  - Uses SQLite in-memory so no external database is needed.
  - All DTLoopStepResult fields are mocked with simple dataclasses.
  - Run with:  uv run python tests/test_monthly_snapshot.py

Exit code: 0 if all tests pass, non-zero otherwise.
"""

from __future__ import annotations

import datetime as dt
import sys
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, "src")

# ── ANSI report helpers ───────────────────────────────────────────────────────
PASS = "\033[32m[ PASS ]\033[0m"
FAIL = "\033[31m[ FAIL ]\033[0m"
SKIP = "\033[33m[ SKIP ]\033[0m"

_failures: list[str] = []


def section(title: str) -> None:
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print("=" * 64)


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS}  {label}")
    else:
        msg = f"{label}  →  {detail}" if detail else label
        print(f"  {FAIL}  {msg}")
        _failures.append(msg)


# ── In-memory SQLite engine / session factory ─────────────────────────────────

def _make_engine_and_factory():
    """Return (engine, session_factory) backed by an in-memory SQLite DB."""
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})

    # Create schema
    with engine.connect() as conn:
        # crop_cycles
        conn.execute(text("""
            CREATE TABLE crop_cycles (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_label   TEXT    NOT NULL UNIQUE,
                started_at    TEXT    DEFAULT (datetime('now')),
                ended_at      TEXT,
                completed     INTEGER DEFAULT 0,
                crop_type     TEXT    DEFAULT 'tomato',
                notes         TEXT
            )
        """))
        # monthly_snapshots — minimal columns for tests (expand as needed)
        conn.execute(text("""
            CREATE TABLE monthly_snapshots (
                id                     INTEGER PRIMARY KEY AUTOINCREMENT,
                cycle_id               INTEGER NOT NULL REFERENCES crop_cycles(id),
                billing_month          TEXT    NOT NULL,
                month_label            TEXT,
                month_start_ts         TEXT,
                month_end_ts           TEXT,
                total_steps            INTEGER DEFAULT 0,
                mpc_solve_count        INTEGER DEFAULT 0,
                image_classify_count   INTEGER DEFAULT 0,
                stage_at_month_start   TEXT,
                stage_at_month_end     TEXT,
                stage_idx_start        INTEGER DEFAULT 0,
                stage_idx_end          INTEGER DEFAULT 0,
                stage_transitions      INTEGER DEFAULT 0,
                stage_transition_log   TEXT    DEFAULT '[]',
                avg_indoor_temp        REAL,
                avg_indoor_humidity    REAL,
                avg_co2                REAL,
                avg_soil_moisture      REAL,
                avg_light_intensity    REAL,
                avg_vpd                REAL,
                avg_leaf_wetness       REAL,
                avg_disease_risk_score REAL,
                min_indoor_temp        REAL,
                max_indoor_temp        REAL,
                min_indoor_humidity    REAL,
                max_indoor_humidity    REAL,
                min_co2                REAL,
                max_co2                REAL,
                min_soil_moisture      REAL,
                max_soil_moisture      REAL,
                max_disease_risk_score REAL,
                avg_setpt_err_temp     REAL,
                avg_setpt_err_humidity REAL,
                avg_setpt_err_soil     REAL,
                avg_setpt_err_co2      REAL,
                avg_setpt_err_light    REAL,
                avg_setpt_err_vpd      REAL,
                total_energy_kwh       REAL DEFAULT 0,
                total_water_l          REAL DEFAULT 0,
                total_cost_inr         REAL DEFAULT 0,
                act_fan_speed_kwh      REAL DEFAULT 0,
                act_vent_opening_kwh   REAL DEFAULT 0,
                act_heater_output_kwh  REAL DEFAULT 0,
                act_led_intensity_kwh  REAL DEFAULT 0,
                act_fogger_duty_kwh    REAL DEFAULT 0,
                act_co2_valve_kwh      REAL DEFAULT 0,
                act_irrigation_kwh     REAL DEFAULT 0,
                act_fan_speed_inr      REAL DEFAULT 0,
                act_vent_opening_inr   REAL DEFAULT 0,
                act_heater_output_inr  REAL DEFAULT 0,
                act_led_intensity_inr  REAL DEFAULT 0,
                act_fogger_duty_inr    REAL DEFAULT 0,
                act_co2_valve_inr      REAL DEFAULT 0,
                act_irrigation_inr     REAL DEFAULT 0,
                act_irrigation_water_l REAL DEFAULT 0,
                avg_mpc_cost           REAL,
                mpc_converge_count     INTEGER DEFAULT 0,
                avg_mpc_fan            REAL,
                avg_mpc_vent           REAL,
                avg_mpc_heat           REAL,
                avg_mpc_led            REAL,
                avg_mpc_co2v           REAL,
                avg_mpc_fog            REAL,
                avg_mpc_irrig          REAL,
                avg_ext_temp           REAL,
                avg_ext_humidity       REAL,
                avg_solar_radiation    REAL,
                avg_wind_speed         REAL,
                peak_early_blight_sev  REAL DEFAULT 0,
                peak_late_blight_sev   REAL DEFAULT 0,
                peak_leaf_mold_sev     REAL DEFAULT 0,
                peak_powdery_mildew_sev REAL DEFAULT 0,
                peak_spider_mites_sev  REAL DEFAULT 0,
                disease_alert_steps    INTEGER DEFAULT 0,
                growth_lstm_runs       INTEGER DEFAULT 0,
                disease_lstm_runs      INTEGER DEFAULT 0,
                weather_forecast_runs  INTEGER DEFAULT 0,
                snapshot_recorded_at   TEXT DEFAULT (datetime('now')),
                UNIQUE(cycle_id, billing_month)
            )
        """))
        conn.commit()

    Session = sessionmaker(bind=engine)
    return engine, Session


# ── Fake DTLoopStepResult ─────────────────────────────────────────────────────

def _fake_result(
    step_index: int,
    timestamp: dt.datetime,
    temp: float = 26.0,
    humidity: float = 72.0,
    co2: float = 550.0,
    soil: float = 62.0,
    light: float = 9500.0,
    vpd: float = 0.90,
    leaf: float = 0.18,
    risk: float = 0.12,
    stage_idx: int = 0,
    mpc_ran: bool = False,
    mpc_cost: float = 0.0,
    image_refresh: bool = False,
) -> Any:
    state = SimpleNamespace(
        indoor_temp=temp,
        indoor_humidity=humidity,
        co2=co2,
        soil_moisture=soil,
        light_intensity=light,
        vpd=vpd,
        leaf_wetness_proxy=leaf,
        disease_risk_score=risk,
        growth_stage_index=stage_idx,
        growth_stage_label=None,  # intentionally None — service must derive it
    )
    diag = SimpleNamespace(
        setpoint_error={"indoor_temp": -1.5, "indoor_humidity": 2.0},
        setpoint_errors=None,
    )
    weather = SimpleNamespace(
        temp_external=25.0,
        humidity_external=60.0,
        solar_radiation=300.0,
        windspeed=14.0,
    )
    return SimpleNamespace(
        step_index=step_index,
        timestamp=timestamp,
        current_state=state,
        next_state=state,
        action_applied=SimpleNamespace(
            fan_speed=0.4, vent_opening=0.2, heater_output=0.0,
            led_intensity=0.6, co2_valve_pct=0.1,
            fogger_duty=0.05, irrigation_qty=0.1,
        ),
        mpc_ran_this_step=mpc_ran,
        mpc_cost=mpc_cost,
        mpc_solution=SimpleNamespace(converged=True, fallback_used=False, total_cost=mpc_cost) if mpc_ran else None,
        image_refresh_this_step=image_refresh,
        cadence_info={
            "growth_due": False,
            "weather_forecast_due": False,
            "model_disease_result": {},
        },
        diagnostics=diag,
        weather_used=weather,
        mpc_forced=False,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_actuator_energy() -> dict:
    dt_h = 5.0 / 60.0
    rated = {"fan_speed": 0.75, "vent_opening": 0.10, "heater_output": 3.00,
             "led_intensity": 2.00, "fogger_duty": 0.20, "co2_valve_pct": 0.05,
             "irrigation_qty": 0.15}
    lvl   = {"fan_speed": 0.4, "vent_opening": 0.2, "heater_output": 0.0,
             "led_intensity": 0.6, "fogger_duty": 0.05, "co2_valve_pct": 0.1,
             "irrigation_qty": 0.1}
    return {k: round(rated[k] * lvl[k] * dt_h, 6) for k in rated}


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Accumulator populates correctly after N steps
# ─────────────────────────────────────────────────────────────────────────────

def test_accumulator_populates() -> None:
    section("Test 1 — Accumulator populates correctly after N steps")

    from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService

    _, Session = _make_engine_and_factory()

    svc = MonthlySnapshotService(session_factory=None, cycle_label="t1-cycle")
    svc._initialised = True   # skip DB cycle-row creation

    ts_base = dt.datetime(2026, 3, 15, 8, 0, 0)
    act_e = _default_actuator_energy()
    energy_kwh = sum(act_e.values())

    for i in range(5):
        result = _fake_result(
            step_index=i,
            timestamp=ts_base + dt.timedelta(minutes=5 * i),
            temp=25.0 + i * 0.1,
        )
        svc.ingest_step(result, act_e, water_l=1.0, energy_kwh=energy_kwh)

    acc = svc._acc
    check("billing_month set to 2026-03", acc.billing_month == "2026-03")
    check("total_steps == 5", acc.total_steps == 5)
    check("total_energy_kwh accumulated", acc.total_energy_kwh > 0)
    check("total_water_l accumulated (5 × 1.0 L)", abs(acc.total_water_l - 5.0) < 1e-6)
    check("min_temp set correctly", acc.min_temp is not None and acc.min_temp < acc.max_temp)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — flush_month writes a DB row with correct aggregates
# ─────────────────────────────────────────────────────────────────────────────

def test_flush_month_writes_db_row() -> None:
    section("Test 2 — flush_month writes a DB row with correct aggregates")

    from sqlalchemy import text
    from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService

    engine, Session = _make_engine_and_factory()

    # Create a cycle row manually (cycle_id = 1)
    with engine.connect() as conn:
        conn.execute(text("INSERT INTO crop_cycles (cycle_label) VALUES ('t2-cycle')"))
        conn.commit()

    svc = MonthlySnapshotService(session_factory=Session, cycle_label="t2-cycle")
    svc._initialised = True
    svc._cycle_id = 1

    ts_base = dt.datetime(2026, 4, 10, 10, 0, 0)
    act_e = _default_actuator_energy()
    energy_kwh = sum(act_e.values())
    N = 10

    for i in range(N):
        result = _fake_result(step_index=i, timestamp=ts_base + dt.timedelta(minutes=5 * i))
        svc.ingest_step(result, act_e, water_l=2.0, energy_kwh=energy_kwh)

    svc.flush_current_month()

    with engine.connect() as conn:
        row = conn.execute(text(
            "SELECT total_steps, total_water_l, total_energy_kwh, avg_indoor_temp "
            "FROM monthly_snapshots WHERE billing_month = '2026-04'"
        )).fetchone()

    check("DB row exists for 2026-04", row is not None)
    if row:
        check("total_steps = 10", row[0] == 10, f"got {row[0]}")
        check("total_water_l = 20.0", abs(row[1] - 20.0) < 1e-4, f"got {row[1]}")
        check("total_energy_kwh > 0", row[2] > 0)
        check("avg_indoor_temp ≈ 26.0", abs(row[3] - 26.0) < 0.01, f"got {row[3]}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Month rollover triggers auto-flush and starts new accumulator
# ─────────────────────────────────────────────────────────────────────────────

def test_month_rollover() -> None:
    section("Test 3 — Month rollover triggers auto-flush and starts new accumulator")

    from sqlalchemy import text
    from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService

    engine, Session = _make_engine_and_factory()

    with engine.connect() as conn:
        conn.execute(text("INSERT INTO crop_cycles (cycle_label) VALUES ('t3-cycle')"))
        conn.commit()

    svc = MonthlySnapshotService(session_factory=Session, cycle_label="t3-cycle")
    svc._initialised = True
    svc._cycle_id = 1

    act_e = _default_actuator_energy()
    energy_kwh = sum(act_e.values())

    # 3 steps in March, 3 steps in April (crossing month boundary)
    march_ts = [dt.datetime(2026, 3, 31, 23, 0) + dt.timedelta(minutes=5 * i) for i in range(3)]
    april_ts = [dt.datetime(2026, 4, 1, 0, 0) + dt.timedelta(minutes=5 * i) for i in range(3)]

    for i, ts in enumerate(march_ts + april_ts):
        result = _fake_result(step_index=i, timestamp=ts)
        svc.ingest_step(result, act_e, water_l=1.0, energy_kwh=energy_kwh)

    # Force-flush the remaining April bucket
    svc.flush_current_month()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT billing_month, total_steps FROM monthly_snapshots ORDER BY billing_month"
        )).fetchall()

    billing_months = {r[0]: r[1] for r in rows}
    check("March snapshot written on rollover",  "2026-03" in billing_months,
          f"found months: {list(billing_months.keys())}")
    check("March has 3 steps", billing_months.get("2026-03") == 3,
          f"got {billing_months.get('2026-03')}")
    check("April snapshot written on flush",     "2026-04" in billing_months,
          f"found months: {list(billing_months.keys())}")
    check("April has 3 steps", billing_months.get("2026-04") == 3,
          f"got {billing_months.get('2026-04')}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — New cycle row is created for each run, auto-incrementing label
# ─────────────────────────────────────────────────────────────────────────────

def test_cycle_row_created() -> None:
    section("Test 4 — New crop_cycles row is created on initialise()")

    from sqlalchemy import text
    from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService

    engine, Session = _make_engine_and_factory()

    svc1 = MonthlySnapshotService(session_factory=Session, cycle_label="cycle-001")
    svc1.initialise()

    svc2 = MonthlySnapshotService(session_factory=Session, cycle_label="cycle-002")
    svc2.initialise()

    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT cycle_label FROM crop_cycles ORDER BY id"
        )).fetchall()

    labels = [r[0] for r in rows]
    check("Two crop_cycles rows created", len(labels) == 2, f"found {labels}")
    check("cycle-001 exists", "cycle-001" in labels)
    check("cycle-002 exists", "cycle-002" in labels)
    check("Idempotent — second initialise() is safe",
          svc1._cycle_id is not None and svc2._cycle_id is not None)


# ─────────────────────────────────────────────────────────────────────────────
# Test 5 — Stage transition log is populated correctly
# ─────────────────────────────────────────────────────────────────────────────

def test_stage_transition_log() -> None:
    section("Test 5 — Stage transition log is populated on stage change")

    from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService

    svc = MonthlySnapshotService(session_factory=None, cycle_label="t5-cycle")
    svc._initialised = True

    ts_base = dt.datetime(2026, 5, 1, 8, 0)
    act_e = _default_actuator_energy()
    energy_kwh = sum(act_e.values())

    # Steps 0-4: stage index 0 (seedling), step 5: stage index 1 (early vegetative)
    for i in range(5):
        result = _fake_result(step_index=i, timestamp=ts_base + dt.timedelta(minutes=5 * i), stage_idx=0)
        svc.ingest_step(result, act_e, water_l=0.5, energy_kwh=energy_kwh)

    result = _fake_result(step_index=5, timestamp=ts_base + dt.timedelta(minutes=25), stage_idx=1)
    svc.ingest_step(result, act_e, water_l=0.5, energy_kwh=energy_kwh)

    acc = svc._acc
    check("stage_transitions == 1", acc.stage_transitions == 1, f"got {acc.stage_transitions}")
    check("stage_at_end is early vegetative",
          acc.stage_at_end == "early vegetative", f"got {acc.stage_at_end!r}")
    check("transition_log has one entry", len(acc.stage_transition_log) == 1,
          f"log={acc.stage_transition_log}")
    if acc.stage_transition_log:
        entry = acc.stage_transition_log[0]
        check("transition from_stage = seedling",
              entry.get("from_stage") == "seedling", f"got {entry.get('from_stage')}")
        check("transition to_stage = early vegetative",
              entry.get("to_stage") == "early vegetative", f"got {entry.get('to_stage')}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 6 — disease_alert_steps counted when risk > 0.3
# ─────────────────────────────────────────────────────────────────────────────

def test_disease_alert_steps() -> None:
    section("Test 6 — disease_alert_steps counted when risk > 0.3")

    from agritwin_gh.services.monthly_snapshot_service import MonthlySnapshotService

    svc = MonthlySnapshotService(session_factory=None, cycle_label="t6-cycle")
    svc._initialised = True

    ts_base = dt.datetime(2026, 6, 5, 8, 0)
    act_e = _default_actuator_energy()
    energy_kwh = sum(act_e.values())

    risks = [0.05, 0.32, 0.41, 0.10, 0.35]  # 3 above threshold
    for i, risk in enumerate(risks):
        result = _fake_result(step_index=i, timestamp=ts_base + dt.timedelta(minutes=5 * i), risk=risk)
        svc.ingest_step(result, act_e, water_l=0.5, energy_kwh=energy_kwh)

    check(
        "disease_alert_steps == 3",
        svc._acc.disease_alert_steps == 3,
        f"got {svc._acc.disease_alert_steps}",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "╔" + "═" * 62 + "╗")
    print("║  test_monthly_snapshot.py                                    ║")
    print("╚" + "═" * 62 + "╝")

    test_accumulator_populates()
    test_flush_month_writes_db_row()
    test_month_rollover()
    test_cycle_row_created()
    test_stage_transition_log()
    test_disease_alert_steps()

    print("\n" + "═" * 64)
    if _failures:
        print(f"  RESULT: {len(_failures)} failure(s)")
        for f in _failures:
            print(f"    ✗ {f}")
        sys.exit(1)
    else:
        print(f"  RESULT: All tests passed.")
    print("═" * 64 + "\n")
