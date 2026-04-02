#!/usr/bin/env python3
"""
run_realtime_loop.py — Real-time closed-loop Digital Twin + MPC controller.

Integrates all AgriTwin-GH subsystems in a single live loop:

  PostgreSQL sensor stream  →  StateFusion (AI models)  →  MPCSolver
       ↑                                                          |
  realtime_greenhouse_stream  ←  GreenhouseTransitionModel  ←  action

Usage
-----
    python scripts/run_realtime_loop.py
    python scripts/run_realtime_loop.py --steps 288        # 24 h at 5 min dt
    python scripts/run_realtime_loop.py --device cuda      # GPU for weather model
    python scripts/run_realtime_loop.py --no-images        # skip image classifiers
    python scripts/run_realtime_loop.py --mpc-every 3      # solve MPC every 3 steps (15 min)
    python scripts/run_realtime_loop.py --dry-run          # print plan, do not write to DB

Press Ctrl-C at any time to stop gracefully.

After a run, delete accumulated test rows with:
    python scripts/run_realtime_loop.py --show-delete-query
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# ── PYTHONPATH bootstrap ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

# ── Core imports ──────────────────────────────────────────────────────────────
from sqlalchemy import text
from sqlalchemy.orm import Session

from agritwin_gh.models.timeseries import Base, RealtimeGreenhouseStream
from agritwin_gh.mpc.config import load_mpc_config
from agritwin_gh.mpc.constants import (
    GROWTH_STAGE_TO_DB,
    GROWTH_STAGES,
    DT_MINUTES,
    compute_dew_point,
    compute_leaf_wetness_proxy,
    compute_vpd,
    stage_label_to_index,
)
from agritwin_gh.mpc.digital_twin_output import DigitalTwinOutput
from agritwin_gh.mpc.disease_penalty import DiseaseRiskPenalty
from agritwin_gh.mpc.disturbance import WeatherDisturbanceForecast
from agritwin_gh.mpc.greenhouse_model import GreenhouseTransitionModel
from agritwin_gh.mpc.growth_weights import GrowthStageWeights
from agritwin_gh.mpc.image_streamer import ImageStreamer
from agritwin_gh.mpc.mpc_input_preparation import MPCInputPreparation
from agritwin_gh.mpc.mpc_solver import MPCSolver, MPCSolution
from agritwin_gh.mpc.state import (
    ActuatorState,
    ControllerDecisionContext,
    DigitalTwinStepPayload,
    FusedState,
    GreenhouseState,
)
from agritwin_gh.mpc.state_fusion import StateFusion
from agritwin_gh.utils.database import get_db_manager

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("agritwin.realtime")

# ── Stage constants ───────────────────────────────────────────────────────────

# Canonical stage durations (hours) — from growth_progression_cycle_summary averages
STAGE_DURATION_HOURS: dict[str, int] = {
    "seedling":              336,   # 14 days
    "early vegetative":      480,   # 20 days
    "flowering initiation":  240,   # 10 days
    "flowering":             360,   # 15 days
    "unripe":                480,   # 20 days
    "ripe":                  240,   # 10 days
}

# Cumulative prior-stage hours (for days_from_cycle_start computation)
_STAGE_ORDER = list(GROWTH_STAGES)
_PRIOR_STAGE_HOURS: dict[str, int] = {}
_acc = 0
for _s in _STAGE_ORDER:
    _PRIOR_STAGE_HOURS[_s] = _acc
    _acc += STAGE_DURATION_HOURS[_s]

# ── ANSI colours (degrade gracefully on Windows without VT100) ────────────────
try:
    import ctypes
    ctypes.windll.kernel32.SetConsoleMode(
        ctypes.windll.kernel32.GetStdHandle(-11), 7
    )
    _ANSI = True
except Exception:
    _ANSI = sys.platform != "win32"

_R = "\033[91m" if _ANSI else ""
_G = "\033[92m" if _ANSI else ""
_Y = "\033[93m" if _ANSI else ""
_B = "\033[94m" if _ANSI else ""
_C = "\033[96m" if _ANSI else ""
_W = "\033[97m" if _ANSI else ""
_DIM = "\033[2m" if _ANSI else ""
_RST = "\033[0m" if _ANSI else ""
_BOLD = "\033[1m" if _ANSI else ""

# ── Delete-query helper ───────────────────────────────────────────────────────

_DELETE_QUERIES = """\
-- ── Delete queries for realtime_greenhouse_stream ──────────────────────────

-- Delete all rows from a SPECIFIC run (replace the run_id value):
DELETE FROM realtime_greenhouse_stream
WHERE run_id = 'rt_20260402_185434';

-- Delete ALL realtime stream data (full table reset):
DELETE FROM realtime_greenhouse_stream;

-- Truncate for faster full reset (no transaction log):
TRUNCATE TABLE realtime_greenhouse_stream RESTART IDENTITY;

-- Delete only old runs (keep the last 7 days):
DELETE FROM realtime_greenhouse_stream
WHERE created_at < NOW() - INTERVAL '7 days';

-- Delete only bootstrap rows (keep DT-simulated rows):
DELETE FROM realtime_greenhouse_stream
WHERE source = 'bootstrap';

-- Review all runs before deleting:
SELECT run_id,
       MIN(datetime) AS started,
       MAX(datetime) AS ended,
       COUNT(*)      AS rows
FROM realtime_greenhouse_stream
GROUP BY run_id
ORDER BY started DESC;
"""


# ─────────────────────────────────────────────────────────────────────────────
# Realtime-aware MPCInputPreparation override
# ─────────────────────────────────────────────────────────────────────────────


class RealtimeMPCInputPreparation(MPCInputPreparation):
    """Overrides ``get_latest_greenhouse_row()`` to read from the real-time
    stream table instead of the historical ``greenhouse_data`` table.

    All other methods (weather, growth, disease context) continue to query
    their original tables so AI models receive real historical context.

    Parameters
    ----------
    session:
        Active SQLAlchemy Session.
    run_id:
        The current loop run ID — used to filter only rows from this run
        if ``filter_by_run`` is *True* (default).
    filter_by_run:
        When *True*, ``get_latest_greenhouse_row()`` returns only the most
        recent row written by the current run.  Set *False* to allow reading
        rows from prior runs (useful for warm-start scenarios).
    """

    def __init__(
        self,
        session: Session,
        run_id: str,
        filter_by_run: bool = True,
    ) -> None:
        super().__init__(session)
        self._run_id = run_id
        self._filter_by_run = filter_by_run

    def get_latest_greenhouse_row(self) -> dict[str, Any] | None:
        """Return the most recent row from ``realtime_greenhouse_stream``.

        The dict matches the exact schema expected by
        ``GreenhouseState.from_db_row()``.
        """
        q = self._session.query(RealtimeGreenhouseStream)
        if self._filter_by_run:
            q = q.filter(RealtimeGreenhouseStream.run_id == self._run_id)
        row = q.order_by(RealtimeGreenhouseStream.datetime.desc()).first()

        if row is None:
            return None

        return {
            "datetime": row.datetime,
            "indoor_temp": row.indoor_temp,
            "indoor_humidity": row.indoor_humidity,
            "indoor_air_velocity": row.indoor_air_velocity or 0.5,
            "indoor_co2": row.indoor_co2,
            "solarradiation": row.solarradiation,
            "day_night_flag": row.day_night_flag,
            "vpd": row.vpd,
            "dew_point": row.dew_point,
            "leaf_wetness_proxy": row.leaf_wetness_proxy,
            # Pass-through extras (used by state_fusion helpers)
            "soil_moisture": row.soil_moisture,
            "stage_name": GROWTH_STAGE_TO_DB.get(row.growth_stage or "seedling", "seedling"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────


def ensure_stream_table(session: Session) -> None:
    """Create ``realtime_greenhouse_stream`` if it does not already exist."""
    engine = session.get_bind()
    Base.metadata.create_all(engine, tables=[RealtimeGreenhouseStream.__table__])
    logger.info("realtime_greenhouse_stream table ensured.")


def seed_initial_state(
    session: Session,
    run_id: str,
    growth_stage: str,
    hours_in_stage: float,
    stage_progress_pct: float,
    days_from_cycle_start: float,
    start_ts: _dt.datetime,
) -> GreenhouseState:
    """Seed the stream with an initial greenhouse state.

    Priority:
    1. Most recent row from ``greenhouse_data`` (real historical readings).
    2. Diurnal-aligned synthetic defaults if the historical table is empty.

    Returns the initial ``GreenhouseState`` for the first DT step.
    """
    from agritwin_gh.models.timeseries import GreenhouseData

    # Attempt to read the most recent real sensor row
    latest = (
        session.query(GreenhouseData)
        .order_by(GreenhouseData.datetime.desc())
        .first()
    )

    hour = start_ts.hour

    if latest is not None:
        indoor_temp = latest.indoor_temp or _diurnal_temp(hour)
        indoor_humidity = latest.indoor_humidity or _diurnal_humidity(hour)
        indoor_co2 = latest.indoor_co2 or 800.0
        solar = latest.solarradiation or _diurnal_solar(hour)
        vpd = latest.vpd or compute_vpd(indoor_temp, indoor_humidity)
        dew_pt = latest.dew_point or compute_dew_point(indoor_temp, indoor_humidity)
        lw = latest.leaf_wetness_proxy or compute_leaf_wetness_proxy(
            indoor_humidity, indoor_temp, dew_pt
        )
        air_vel = latest.indoor_air_velocity or 0.5
        dnf = latest.day_night_flag if latest.day_night_flag is not None else int(6 <= hour < 20)
    else:
        print(f"  {_Y}[WARN]{_RST} greenhouse_data is empty — using diurnal defaults.")
        indoor_temp = _diurnal_temp(hour)
        indoor_humidity = _diurnal_humidity(hour)
        indoor_co2 = 800.0
        solar = _diurnal_solar(hour)
        vpd = compute_vpd(indoor_temp, indoor_humidity)
        dew_pt = compute_dew_point(indoor_temp, indoor_humidity)
        lw = compute_leaf_wetness_proxy(indoor_humidity, indoor_temp, dew_pt)
        air_vel = 0.5
        dnf = int(6 <= hour < 20)

    # Soil moisture default by stage
    soil_moisture = _default_soil_moisture(growth_stage)

    row = RealtimeGreenhouseStream(
        run_id=run_id,
        step_index=0,
        source="bootstrap",
        datetime=start_ts,
        indoor_temp=indoor_temp,
        indoor_humidity=indoor_humidity,
        indoor_air_velocity=air_vel,
        indoor_co2=indoor_co2,
        solarradiation=solar,
        day_night_flag=dnf,
        vpd=vpd,
        dew_point=dew_pt,
        leaf_wetness_proxy=lw,
        soil_moisture=soil_moisture,
        disease_risk_score=0.0,
        growth_stage=growth_stage,
        growth_stage_index=stage_label_to_index(growth_stage),
        disease_classification="healthy leaves",
        fan_speed=0.0,
        vent_opening=0.0,
        heater_output=0.0,
        led_intensity=0.0,
        fogger_duty=0.0,
        co2_valve_pct=0.0,
        irrigation_qty=0.0,
        mpc_ran=False,
        mpc_converged=False,
        mpc_fallback_used=False,
        step_cost=0.0,
        solve_time_ms=0.0,
        hours_in_current_stage=hours_in_stage,
        stage_progress_pct=stage_progress_pct,
        hours_to_stage_transition=float(
            STAGE_DURATION_HOURS[growth_stage] - hours_in_stage
        ),
        step_energy_kwh=0.0,
        cumulative_energy_kwh=0.0,
        cumulative_water_litres=0.0,
        alert_level="GREEN",
    )
    session.add(row)
    session.commit()

    return GreenhouseState(
        indoor_temp=indoor_temp,
        indoor_humidity=indoor_humidity,
        soil_moisture=soil_moisture,
        co2=indoor_co2,
        light_intensity=solar,
        disease_risk_score=0.0,
        growth_stage_index=stage_label_to_index(growth_stage),
        vpd=vpd,
        leaf_wetness_proxy=lw,
        timestamp=start_ts,
    )


def write_step_to_stream(
    session: Session,
    *,
    run_id: str,
    step_index: int,
    payload: DigitalTwinStepPayload,
    hours_in_stage: float,
    stage_progress_pct: float,
    cumulative_energy_kwh: float,
    cumulative_water_litres: float,
) -> None:
    """Persist one MPC output step to ``realtime_greenhouse_stream``."""
    obs = payload.observed_state
    act = payload.applied_actuators
    ts = payload.timestamp or _dt.datetime.utcnow()

    dew_pt = compute_dew_point(
        obs.get("indoor_temp", 25.0),
        obs.get("indoor_humidity", 65.0),
    )

    row = RealtimeGreenhouseStream(
        run_id=run_id,
        step_index=step_index,
        source="dt_sim",
        datetime=ts,
        indoor_temp=obs.get("indoor_temp"),
        indoor_humidity=obs.get("indoor_humidity"),
        indoor_air_velocity=0.5,
        indoor_co2=obs.get("co2"),
        solarradiation=obs.get("light_intensity"),
        day_night_flag=int(6 <= ts.hour < 20) if ts else 1,
        vpd=obs.get("vpd"),
        dew_point=dew_pt,
        leaf_wetness_proxy=obs.get("leaf_wetness_proxy"),
        soil_moisture=obs.get("soil_moisture"),
        disease_risk_score=payload.disease_risk_score,
        growth_stage=payload.growth_stage,
        growth_stage_index=obs.get("growth_stage_index"),
        disease_classification=payload.disease_classification,
        fan_speed=act.get("fan_speed"),
        vent_opening=act.get("vent_opening"),
        heater_output=act.get("heater_output"),
        led_intensity=act.get("led_intensity"),
        fogger_duty=act.get("fogger_duty"),
        co2_valve_pct=act.get("co2_valve_pct"),
        irrigation_qty=act.get("irrigation_qty"),
        mpc_ran=True,
        mpc_converged=payload.solver_performance.get("converged", False),
        mpc_fallback_used=payload.solver_performance.get("fallback_used", False),
        step_cost=payload.step_cost,
        solve_time_ms=payload.solver_performance.get("solve_time_ms", 0.0),
        hours_in_current_stage=hours_in_stage,
        stage_progress_pct=stage_progress_pct,
        hours_to_stage_transition=payload.hours_to_stage_transition,
        step_energy_kwh=payload.cumulative_energy_kwh - cumulative_energy_kwh,
        cumulative_energy_kwh=payload.cumulative_energy_kwh,
        cumulative_water_litres=payload.cumulative_water_litres,
        alert_level=payload.alert_level,
    )
    session.add(row)
    session.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Diurnal defaults
# ─────────────────────────────────────────────────────────────────────────────


def _diurnal_temp(hour: int) -> float:
    """Sinusoidal indoor temp (°C) as a function of hour-of-day."""
    return round(22.0 + 5.0 * math.sin(math.pi * (hour - 6) / 12), 2)


def _diurnal_humidity(hour: int) -> float:
    """Sinusoidal indoor RH (%) peaking at night."""
    return round(65.0 - 10.0 * math.sin(math.pi * (hour - 6) / 12), 2)


def _diurnal_solar(hour: int) -> float:
    """Simple daytime solar radiation estimate (W/m²)."""
    if not (6 <= hour < 20):
        return 0.0
    return round(600.0 * math.sin(math.pi * (hour - 6) / 14), 2)


def _default_soil_moisture(stage: str) -> float:
    """Stage-aware default soil moisture (%)."""
    return {
        "seedling": 70.0,
        "early vegetative": 65.0,
        "flowering initiation": 60.0,
        "flowering": 58.0,
        "unripe": 55.0,
        "ripe": 50.0,
    }.get(stage, 60.0)


# ─────────────────────────────────────────────────────────────────────────────
# CLI prompts
# ─────────────────────────────────────────────────────────────────────────────


def prompt_growth_stage() -> str:
    """Interactive menu to select the current tomato growth stage."""
    print(f"\n{_BOLD}{_C}── Current Growth Stage ───────────────────────────────{_RST}")
    for i, s in enumerate(GROWTH_STAGES, start=1):
        print(f"  {_W}{i}{_RST}. {s.title()}")
    while True:
        raw = input(f"\n  {_B}Enter stage number (1–{len(GROWTH_STAGES)}){_RST}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(GROWTH_STAGES):
            return GROWTH_STAGES[int(raw) - 1]
        print(f"  {_R}Invalid choice. Please enter a number between 1 and {len(GROWTH_STAGES)}.{_RST}")


def prompt_days_elapsed(stage: str) -> float:
    """Ask how many days have elapsed in the current stage."""
    max_days = STAGE_DURATION_HOURS[stage] / 24
    print(f"\n  Stage max duration: {_W}{max_days:.0f} days{_RST}")
    while True:
        raw = input(
            f"  {_B}Days already elapsed in '{stage}' stage{_RST} (0–{max_days:.0f}): "
        ).strip()
        try:
            val = float(raw)
            if 0.0 <= val <= max_days:
                return val
            print(f"  {_R}Value must be between 0 and {max_days:.0f}.{_RST}")
        except ValueError:
            print(f"  {_R}Please enter a numeric value.{_RST}")


def prompt_steps(default: int = 288) -> int:
    """Ask how many 5-minute steps to run (interactive fallback)."""
    while True:
        raw = input(
            f"  {_B}Number of simulation steps{_RST} (5 min each, default {default}): "
        ).strip()
        if raw == "":
            return default
        if raw.isdigit() and int(raw) > 0:
            return int(raw)
        print(f"  {_R}Please enter a positive integer.{_RST}")


# ─────────────────────────────────────────────────────────────────────────────
# Console display
# ─────────────────────────────────────────────────────────────────────────────


def _alert_colour(level: str) -> str:
    return {"GREEN": _G, "YELLOW": _Y, "RED": _R}.get(level, _W)


def print_banner(run_id: str, stage: str, days_elapsed: float, total_steps: int) -> None:
    """Print the startup banner."""
    hours = days_elapsed * 24
    hours_remaining = STAGE_DURATION_HOURS[stage] - hours
    print(
        f"\n{_BOLD}{_C}"
        f"╔══════════════════════════════════════════════════════════╗\n"
        f"║          AgriTwin-GH  Real-Time Closed-Loop MPC          ║\n"
        f"╚══════════════════════════════════════════════════════════╝"
        f"{_RST}"
    )
    print(f"  {_DIM}Run ID   :{_RST} {_W}{run_id}{_RST}")
    print(f"  {_DIM}Stage    :{_RST} {_W}{stage.title()}{_RST}")
    print(
        f"  {_DIM}Progress :{_RST} {_W}{days_elapsed:.1f} d elapsed "
        f"/ ~{hours_remaining/24:.1f} d remaining{_RST}"
    )
    print(
        f"  {_DIM}Steps    :{_RST} {_W}{total_steps}{_RST} "
        f"{_DIM}({total_steps * DT_MINUTES // 60:.1f} h simulated time){_RST}"
    )
    print(f"  {_DIM}DB table :{_RST} realtime_greenhouse_stream")
    print(f"\n  Press {_Y}Ctrl-C{_RST} to stop gracefully.\n")
    print("─" * 62)


def print_step(step: int, total: int, payload: DigitalTwinStepPayload,
               hours_in_stage: float, wall_dt_ms: float) -> None:
    """Print a single-line step summary to the console."""
    obs = payload.observed_state
    act = payload.applied_actuators
    alert_col = _alert_colour(payload.alert_level)

    temp  = obs.get("indoor_temp", 0.0)
    rh    = obs.get("indoor_humidity", 0.0)
    co2   = obs.get("co2", 0.0)
    risk  = payload.disease_risk_score
    stage = payload.growth_stage or "—"

    fan   = act.get("fan_speed", 0.0)
    vent  = act.get("vent_opening", 0.0)
    heat  = act.get("heater_output", 0.0)
    led   = act.get("led_intensity", 0.0)

    ts_str = payload.timestamp.strftime("%H:%M") if payload.timestamp else "??:??"
    pct_done = 100.0 * step / max(total, 1)

    converged = payload.solver_performance.get("converged", False)
    conv_sym = f"{_G}✓{_RST}" if converged else f"{_Y}⚠{_RST}"

    print(
        f"  [{_DIM}{ts_str}{_RST}] "
        f"step {_W}{step:>4d}/{total}{_RST} ({pct_done:5.1f}%)  "
        f"|  T={_W}{temp:5.1f}°C{_RST}  RH={_W}{rh:5.1f}%{_RST}  "
        f"CO₂={_W}{co2:>5.0f}ppm{_RST}  "
        f"risk={alert_col}{risk:.3f}{_RST}  "
        f"stage={_C}{stage}{_RST}  "
        f"fan={fan:.2f}  vent={vent:.2f}  heat={heat:.2f}  led={led:.2f}  "
        f"MPC{conv_sym}  "
        f"{_DIM}{wall_dt_ms:.0f}ms{_RST}"
    )


def print_summary(
    run_id: str,
    steps_run: int,
    total_energy: float,
    total_water: float,
    total_cost: float,
    artifact_dir: Path | None,
) -> None:
    """Print the end-of-run summary."""
    print("\n" + "─" * 62)
    print(f"  {_BOLD}{_G}Run complete{_RST}")
    print(f"  {_DIM}Run ID         :{_RST} {_W}{run_id}{_RST}")
    print(f"  {_DIM}Steps executed :{_RST} {_W}{steps_run}{_RST}")
    print(f"  {_DIM}Simulated time :{_RST} {_W}{steps_run * DT_MINUTES / 60:.2f} h{_RST}")
    print(f"  {_DIM}Total energy   :{_RST} {_W}{total_energy:.3f} kWh{_RST}")
    print(f"  {_DIM}Total water    :{_RST} {_W}{total_water:.2f} L{_RST}")
    print(f"  {_DIM}Total MPC cost :{_RST} {_W}{total_cost:.4f}{_RST}")
    if artifact_dir:
        print(f"  {_DIM}Artifacts saved:{_RST} {_W}{artifact_dir}{_RST}")
    print()
    print(f"  To delete this run's data from the DB:")
    print(
        f"    {_Y}DELETE FROM realtime_greenhouse_stream "
        f"WHERE run_id = '{run_id}';{_RST}"
    )
    print("─" * 62)


# ─────────────────────────────────────────────────────────────────────────────
# Artifact saving
# ─────────────────────────────────────────────────────────────────────────────


def setup_artifact_dir(run_id: str) -> Path:
    """Create and return the run artifact directory under logs/."""
    art_dir = _ROOT / "logs" / "realtime" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir


def save_step_artifact(art_dir: Path, step: int, payload: DigitalTwinStepPayload) -> None:
    """Append a compact step dict to the NDJSON log file."""
    log_file = art_dir / "steps.ndjson"
    record = {
        "step": step,
        "ts": str(payload.timestamp),
        "stage": payload.growth_stage,
        "risk": payload.disease_risk_score,
        "alert": payload.alert_level,
        "cost": payload.step_cost,
        "energy_kwh": payload.cumulative_energy_kwh,
        "water_l": payload.cumulative_water_litres,
        "obs": payload.observed_state,
        "act": payload.applied_actuators,
    }
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def save_run_manifest(
    art_dir: Path,
    *,
    run_id: str,
    growth_stage: str,
    days_elapsed: float,
    total_steps: int,
    steps_run: int,
    start_ts: _dt.datetime,
    end_ts: _dt.datetime,
    total_energy: float,
    total_water: float,
    total_cost: float,
    args: argparse.Namespace,
) -> None:
    """Write a JSON manifest summarising the run."""
    manifest = {
        "run_id": run_id,
        "growth_stage": growth_stage,
        "days_elapsed_at_start": days_elapsed,
        "planned_steps": total_steps,
        "steps_run": steps_run,
        "start_ts": str(start_ts),
        "end_ts": str(end_ts),
        "simulated_hours": steps_run * DT_MINUTES / 60,
        "total_energy_kwh": total_energy,
        "total_water_litres": total_water,
        "total_mpc_cost": total_cost,
        "mpc_every_steps": args.mpc_every,
        "images_enabled": not args.no_images,
        "device": args.device,
        "db_table": "realtime_greenhouse_stream",
        "delete_query": (
            f"DELETE FROM realtime_greenhouse_stream WHERE run_id = '{run_id}';"
        ),
    }
    with open(art_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Graceful shutdown
# ─────────────────────────────────────────────────────────────────────────────

_stop_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _stop_requested
    _stop_requested = True
    print(f"\n  {_Y}[INFO]{_RST} Stop signal received — finishing current step …")


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────


def run_realtime_loop(args: argparse.Namespace) -> None:
    """Bootstrap and execute the real-time closed-loop."""
    global _stop_requested

    # ── Show delete queries and exit if requested ──────────────────────
    if args.show_delete_query:
        print(_DELETE_QUERIES)
        return

    # ── Interactive prompts (only if not provided via CLI) ─────────────
    if args.stage:
        stage_lower = args.stage.lower()
        if stage_lower not in GROWTH_STAGES:
            # Try prefix match
            matches = [s for s in GROWTH_STAGES if s.startswith(stage_lower)]
            if len(matches) == 1:
                growth_stage = matches[0]
            else:
                print(f"{_R}Unknown stage '{args.stage}'. Valid: {GROWTH_STAGES}{_RST}")
                sys.exit(1)
        else:
            growth_stage = stage_lower
    else:
        growth_stage = prompt_growth_stage()

    if args.days_elapsed is not None:
        days_elapsed = float(args.days_elapsed)
    else:
        days_elapsed = prompt_days_elapsed(growth_stage)

    total_steps = args.steps if args.steps > 0 else prompt_steps()
    mpc_every   = max(1, args.mpc_every)   # solve MPC every N steps (≥1)

    # ── Derived temporal quantities ────────────────────────────────────
    hours_in_stage_0   = days_elapsed * 24.0
    stage_progress_0   = min(100.0, 100.0 * hours_in_stage_0 / STAGE_DURATION_HOURS[growth_stage])
    days_from_cycle_0  = _PRIOR_STAGE_HOURS[growth_stage] / 24.0 + days_elapsed

    # ── Run identity + timestamps ──────────────────────────────────────
    now_wall   = _dt.datetime.now()
    run_id     = f"rt_{now_wall:%Y%m%d_%H%M%S}"
    # Simulation clock starts at current wall-clock time (truncated to minute)
    start_ts   = now_wall.replace(second=0, microsecond=0)
    dt_delta   = _dt.timedelta(minutes=DT_MINUTES)

    # ── Artifact directory ─────────────────────────────────────────────
    art_dir = setup_artifact_dir(run_id) if not args.dry_run else None

    # ── DB session ─────────────────────────────────────────────────────
    db_manager = get_db_manager()
    session    = db_manager.get_session()

    try:
        if not args.dry_run:
            ensure_stream_table(session)

        # ── Load image classifiers (optional) ─────────────────────────
        disease_classifier  = None
        growth_classifier   = None

        if not args.no_images:
            try:
                sys.path.insert(0, str(_ROOT / "src"))
                from agritwin_gh.models.disease_inference import predict_image
                from agritwin_gh.models.growth_stage_inference import (
                    predict_growth_stage,
                )
                disease_classifier = predict_image
                growth_classifier  = predict_growth_stage
                logger.info("Image classifiers loaded.")
            except Exception as exc:
                print(f"  {_Y}[WARN]{_RST} Image classifiers not available ({exc}). "
                      "Continuing without them.")

        # ── Configuration ──────────────────────────────────────────────
        cfg = load_mpc_config()

        # ── Sub-system wiring ──────────────────────────────────────────
        rt_input_prep = RealtimeMPCInputPreparation(
            session=session,
            run_id=run_id,
            filter_by_run=True,
        )
        image_streamer   = ImageStreamer(session)
        weather_forecast = WeatherDisturbanceForecast(
            run_id=cfg.environment_forecast_run_id,
            device=args.device,
        )
        disease_penalty  = DiseaseRiskPenalty(run_id=cfg.disease_progression_run_id)
        growth_weights   = GrowthStageWeights(
            stage_weight_multipliers=cfg.stage_weight_multipliers,
            run_id=cfg.growth_progression_run_id,
        )
        fusion = StateFusion(
            config=cfg,
            input_prep=rt_input_prep,
            weather=weather_forecast,
            disease_penalty=disease_penalty,
            growth_weights=growth_weights,
            image_streamer=image_streamer,
            disease_classifier=disease_classifier,
            growth_classifier=growth_classifier,
        )
        dt_model = GreenhouseTransitionModel()
        solver   = MPCSolver(config=cfg, model=dt_model)
        output   = DigitalTwinOutput(run_id=run_id)

        # ── Seed initial state ─────────────────────────────────────────
        print(f"\n  {_DIM}Seeding initial state from DB …{_RST}")
        if not args.dry_run:
            current_state = seed_initial_state(
                session=session,
                run_id=run_id,
                growth_stage=growth_stage,
                hours_in_stage=hours_in_stage_0,
                stage_progress_pct=stage_progress_0,
                days_from_cycle_start=days_from_cycle_0,
                start_ts=start_ts,
            )
        else:
            # Dry-run: synthesise state without writing
            hour = start_ts.hour
            current_state = GreenhouseState(
                indoor_temp=_diurnal_temp(hour),
                indoor_humidity=_diurnal_humidity(hour),
                soil_moisture=_default_soil_moisture(growth_stage),
                co2=800.0,
                light_intensity=_diurnal_solar(hour),
                disease_risk_score=0.0,
                growth_stage_index=stage_label_to_index(growth_stage),
                vpd=compute_vpd(_diurnal_temp(hour), _diurnal_humidity(hour)),
                leaf_wetness_proxy=0.0,
                timestamp=start_ts,
            )

        # ── Banner ─────────────────────────────────────────────────────
        print_banner(run_id, growth_stage, days_elapsed, total_steps)

        # ── Signal handling ─────────────────────────────────────────────
        signal.signal(signal.SIGINT,  _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # ── Loop state ──────────────────────────────────────────────────
        current_ts          = start_ts + dt_delta  # first active step timestamp
        prev_action: ActuatorState | None = None
        hours_in_stage      = hours_in_stage_0
        stage_progress      = stage_progress_0
        cumulative_energy   = 0.0
        cumulative_water    = 0.0
        total_cost          = 0.0
        steps_run           = 0

        # ── Main simulation loop ────────────────────────────────────────
        for step_i in range(1, total_steps + 1):
            if _stop_requested:
                break

            wall_start = time.perf_counter()

            # ── Advance stage accounting ─────────────────────────────────
            hours_in_stage  += DT_MINUTES / 60.0
            stage_progress   = min(
                100.0,
                100.0 * hours_in_stage / STAGE_DURATION_HOURS[growth_stage],
            )

            # ── Decide whether to solve MPC this step ────────────────────
            # MPC runs on startup (step 1) and every mpc_every steps thereafter.
            run_mpc = (step_i % mpc_every == 1) or (step_i == 1)

            if run_mpc:
                # ── 1. Fuse state (reads from realtime_greenhouse_stream) ─
                fused: FusedState = fusion.fuse(timestamp=current_ts)

                # ── 2. Adaptive cost weights for current growth stage ─────
                weights = growth_weights.get_weights(
                    fused.growth_stage,
                    base_weights=cfg.cost_weight_vector,
                )

                # ── 3. MPC solve ─────────────────────────────────────────
                solution: MPCSolution = solver.solve(
                    fused=fused,
                    previous_control=prev_action,
                )
                actuators   = solution.first_action
                prev_action = actuators

                # ── 4. Resource accounting ───────────────────────────────
                step_energy = _estimate_energy(actuators)
                step_water  = actuators.irrigation_qty
                cumulative_energy += step_energy
                cumulative_water  += step_water
                total_cost        += solution.total_cost

                # ── 5. Predicted next state (from MPC trajectory) ────────
                predicted_next: GreenhouseState | None = (
                    solution.predicted_states[1]
                    if len(solution.predicted_states) > 1
                    else None
                )

                # ── 6. Build solver-performance snapshot ─────────────────
                solver_perf = {
                    "converged":      solution.converged,
                    "fallback_used":  solution.fallback_used,
                    "solve_time_ms":  solution.solve_time_ms,
                    "n_iterations":   solution.n_iterations,
                    "n_function_evals": solution.n_function_evals,
                    "solver_status":  solution.solver_status,
                }

                # ── 7. Decision context ──────────────────────────────────
                decision_ctx = ControllerDecisionContext(
                    run_id=run_id,
                    timestamp=current_ts,
                    step_index=step_i,
                    solver_config={
                        "method":        cfg.solver_method,
                        "horizon_hours": cfg.prediction_horizon_hours,
                        "dt_minutes":    cfg.dt_minutes,
                        "max_iter":      cfg.solver_max_iter,
                    },
                    cost_weights=weights,
                    weather_stress_summary=solver.last_weather_stress_summary or {},
                    disease_context_summary={
                        "risk_score":       fused.disease_risk_score,
                        "classification":   fused.disease_classification,
                        "severity_24h":     dict(fused.severity_24h),
                    },
                    constraint_tightening=solver.last_constraint_tightening or {},
                    solver_performance=solver_perf,
                    model_ids={
                        "environment_forecast": cfg.environment_forecast_run_id,
                        "disease_progression":  cfg.disease_progression_run_id,
                        "growth_progression":   cfg.growth_progression_run_id,
                    },
                )

                # ── 8. Format payload ────────────────────────────────────
                payload: DigitalTwinStepPayload = output.format_step(
                    fused=fused,
                    actuators=actuators,
                    predicted_next=predicted_next,
                    step_cost=solution.total_cost,
                    energy_kwh=step_energy,
                    water_litres=step_water,
                    cost_breakdown=solution.cost_breakdown,
                    solver_converged=solution.converged,
                    weather_stress=solver.last_weather_stress_summary,
                    tightened_constraints=solver.last_constraint_tightening,
                    decision_context=decision_ctx,
                    solver_performance=solver_perf,
                )

                # Use predicted next state for next DB write
                # (propagates DT physics state forward)
                if predicted_next is not None:
                    predicted_next.timestamp = current_ts + dt_delta
                    current_state = predicted_next

            else:
                # Off-MPC step — reuse last payload with updated timestamp
                # and carry forward actuators unchanged.
                payload = DigitalTwinStepPayload(
                    timestamp=current_ts,
                    run_id=run_id,
                    step_index=step_i,
                    observed_state=(current_state.to_dict() if hasattr(current_state, "to_dict")
                                    else {}),
                    growth_stage=growth_stage,
                    disease_risk_score=current_state.disease_risk_score,
                    applied_actuators=prev_action.to_dict() if prev_action else {},
                    step_cost=0.0,
                    cumulative_energy_kwh=cumulative_energy,
                    cumulative_water_litres=cumulative_water,
                    alert_level="GREEN",
                    solver_performance={"converged": False, "fallback_used": False,
                                        "solve_time_ms": 0.0},
                    hours_to_stage_transition=float(
                        STAGE_DURATION_HOURS[growth_stage] - hours_in_stage
                    ),
                )

                # Advance DT physics with held actuators
                if prev_action is not None and hasattr(dt_model, "step"):
                    try:
                        nxt = dt_model.step(current_state, prev_action,
                                            weather=None, dt_minutes=DT_MINUTES)
                        if nxt is not None:
                            nxt.timestamp = current_ts + dt_delta
                            current_state = nxt
                    except Exception:
                        pass

            # ── 9. Write step to PostgreSQL stream ───────────────────────
            if not args.dry_run:
                write_step_to_stream(
                    session,
                    run_id=run_id,
                    step_index=step_i,
                    payload=payload,
                    hours_in_stage=hours_in_stage,
                    stage_progress_pct=stage_progress,
                    cumulative_energy_kwh=cumulative_energy,
                    cumulative_water_litres=cumulative_water,
                )

                # ── 10. Save per-step artifact ────────────────────────────
                if art_dir is not None:
                    save_step_artifact(art_dir, step_i, payload)

            # ── Console output ───────────────────────────────────────────
            wall_dt_ms = (time.perf_counter() - wall_start) * 1000.0
            print_step(step_i, total_steps, payload, hours_in_stage, wall_dt_ms)

            steps_run  += 1
            current_ts += dt_delta

        # ── End of loop ─────────────────────────────────────────────────
        end_ts = current_ts - dt_delta

        if art_dir is not None:
            save_run_manifest(
                art_dir,
                run_id=run_id,
                growth_stage=growth_stage,
                days_elapsed=days_elapsed,
                total_steps=total_steps,
                steps_run=steps_run,
                start_ts=start_ts,
                end_ts=end_ts,
                total_energy=cumulative_energy,
                total_water=cumulative_water,
                total_cost=total_cost,
                args=args,
            )

        print_summary(
            run_id=run_id,
            steps_run=steps_run,
            total_energy=cumulative_energy,
            total_water=cumulative_water,
            total_cost=total_cost,
            artifact_dir=art_dir,
        )

    finally:
        session.close()
        db_manager.close()


# ─────────────────────────────────────────────────────────────────────────────
# Energy estimation (mirrors runner.py _estimate_energy)
# ─────────────────────────────────────────────────────────────────────────────


def _estimate_energy(actuators: ActuatorState) -> float:
    """Estimate kWh for one 5-minute DT step."""
    dt_hours = DT_MINUTES / 60.0
    rated_kw = {
        "fan_speed":    0.75,
        "vent_opening": 0.10,
        "heater_output": 3.00,
        "led_intensity": 2.00,
        "fogger_duty":   0.20,
        "co2_valve_pct": 0.05,
        "irrigation_qty": 0.15,  # pump
    }
    total = 0.0
    for field, kw in rated_kw.items():
        duty = getattr(actuators, field, 0.0) or 0.0
        total += kw * duty * dt_hours
    return round(total, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Argument parser
# ─────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AgriTwin-GH Real-Time Closed-Loop MPC Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--steps", type=int, default=0,
        help="Number of 5-minute simulation steps (0 = ask interactively). "
             "288 = 24 h, 2016 = 7 days.",
    )
    p.add_argument(
        "--stage", type=str, default="",
        help="Starting growth stage (e.g. 'flowering'). "
             "Omit to select interactively.",
    )
    p.add_argument(
        "--days-elapsed", type=float, default=None,
        metavar="DAYS",
        help="Days already elapsed in the current stage. "
             "Omit to enter interactively.",
    )
    p.add_argument(
        "--mpc-every", type=int, default=3, metavar="N",
        help="Solve MPC every N steps (default 3 = every 15 min). "
             "Set 1 to solve every 5-minute step.",
    )
    p.add_argument(
        "--device", choices=["cpu", "cuda", "mps"], default="cpu",
        help="PyTorch device for the weather-forecast ensemble (default: cpu).",
    )
    p.add_argument(
        "--no-images", action="store_true",
        help="Disable image-classification step (faster, no MinIO needed).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Execute the loop without writing to the database or saving artifacts.",
    )
    p.add_argument(
        "--show-delete-query", action="store_true",
        help="Print SQL DELETE queries for the stream table and exit.",
    )
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = build_parser().parse_args()
    run_realtime_loop(args)


if __name__ == "__main__":
    main()
