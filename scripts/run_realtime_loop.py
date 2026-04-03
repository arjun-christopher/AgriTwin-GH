#!/usr/bin/env python3
"""
run_realtime_loop.py -- Test / evaluation CLI for the real-time closed-loop.

This is a thin CLI wrapper around :class:`agritwin_gh.mpc.RealtimeLoop`.
All core logic lives in ``src/agritwin_gh/mpc/realtime_core.py`` so it can
be imported by production integrations, web services, or notebooks without
any CLI / console dependencies.

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
import json
import signal
import sys
from pathlib import Path
from typing import Any

# -- PYTHONPATH bootstrap ------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

# -- Core imports (all domain logic lives here) --------------------------------
from agritwin_gh.mpc.constants import GROWTH_STAGES, DT_MINUTES
from agritwin_gh.mpc.realtime_core import (
    RealtimeLoop,
    RealtimeLoopConfig,
    RealtimeStepResult,
    RealtimeRunSummary,
    RunRegistry,
    STAGE_DURATION_HOURS,
)
from agritwin_gh.mpc.state import DigitalTwinStepPayload
from agritwin_gh.utils.database import get_db_manager


# ==============================================================================
# ANSI colours (degrade gracefully on Windows without VT100)
# ==============================================================================

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


# ==============================================================================
# Delete-query helper
# ==============================================================================

_DELETE_QUERIES = """\
-- -- Delete queries for realtime_greenhouse_stream --------------------------

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


# ==============================================================================
# Interactive CLI prompts
# ==============================================================================

def prompt_growth_stage() -> str:
    """Interactive menu to select the current tomato growth stage."""
    print(f"\n{_BOLD}{_C}\u2500\u2500 Current Growth Stage \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500{_RST}")
    for i, s in enumerate(GROWTH_STAGES, start=1):
        print(f"  {_W}{i}{_RST}. {s.title()}")
    while True:
        raw = input(f"\n  {_B}Enter stage number (1\u2013{len(GROWTH_STAGES)}){_RST}: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(GROWTH_STAGES):
            return GROWTH_STAGES[int(raw) - 1]
        print(f"  {_R}Invalid choice. Please enter a number between 1 and {len(GROWTH_STAGES)}.{_RST}")


def prompt_days_elapsed(stage: str) -> float:
    """Ask how many days have elapsed in the current stage."""
    max_days = STAGE_DURATION_HOURS[stage] / 24
    print(f"\n  Stage max duration: {_W}{max_days:.0f} days{_RST}")
    while True:
        raw = input(
            f"  {_B}Days already elapsed in '{stage}' stage{_RST} (0\u2013{max_days:.0f}): "
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


# ==============================================================================
# Console display helpers
# ==============================================================================

def _alert_colour(level: str) -> str:
    return {"GREEN": _G, "YELLOW": _Y, "RED": _R}.get(level, _W)


def print_banner(run_id: str, stage: str, days_elapsed: float, total_steps: int) -> None:
    """Print the startup banner."""
    hours = days_elapsed * 24
    hours_remaining = STAGE_DURATION_HOURS[stage] - hours
    print(
        f"\n{_BOLD}{_C}"
        f"\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557\n"
        f"\u2551          AgriTwin-GH  Real-Time Closed-Loop MPC          \u2551\n"
        f"\u255a\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255d"
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
    print("\u2500" * 62)


def print_step_line(result: RealtimeStepResult, total: int) -> None:
    """Print a single-line step summary to the console."""
    payload = result.payload
    obs = payload.observed_state
    act = payload.applied_actuators
    alert_col = _alert_colour(payload.alert_level)

    temp  = obs.get("indoor_temp", 0.0)
    rh    = obs.get("indoor_humidity", 0.0)
    co2   = obs.get("co2", 0.0)
    risk  = payload.disease_risk_score
    stage = payload.growth_stage or "\u2014"

    fan   = act.get("fan_speed", 0.0)
    vent  = act.get("vent_opening", 0.0)
    heat  = act.get("heater_output", 0.0)
    led   = act.get("led_intensity", 0.0)

    ts_str = payload.timestamp.strftime("%H:%M") if payload.timestamp else "??:??"
    step = result.step_index
    pct_done = 100.0 * step / max(total, 1)

    converged = payload.solver_performance.get("converged", False)
    conv_sym = f"{_G}\u2713{_RST}" if converged else f"{_Y}\u26a0{_RST}"

    print(
        f"  [{_DIM}{ts_str}{_RST}] "
        f"step {_W}{step:>4d}/{total}{_RST} ({pct_done:5.1f}%)  "
        f"|  T={_W}{temp:5.1f}\u00b0C{_RST}  RH={_W}{rh:5.1f}%{_RST}  "
        f"CO\u2082={_W}{co2:>5.0f}ppm{_RST}  "
        f"risk={alert_col}{risk:.3f}{_RST}  "
        f"stage={_C}{stage}{_RST}  "
        f"fan={fan:.2f}  vent={vent:.2f}  heat={heat:.2f}  led={led:.2f}  "
        f"MPC{conv_sym}  "
        f"{_DIM}{result.wall_time_ms:.0f}ms{_RST}"
    )


def print_run_summary(summary: RealtimeRunSummary, artifact_dir: Path | None) -> None:
    """Print the end-of-run summary."""
    print("\n" + "\u2500" * 62)
    print(f"  {_BOLD}{_G}Run complete{_RST}")
    print(f"  {_DIM}Run ID         :{_RST} {_W}{summary.run_id}{_RST}")
    print(f"  {_DIM}Steps executed :{_RST} {_W}{summary.steps_run}{_RST}")
    print(f"  {_DIM}Simulated time :{_RST} {_W}{summary.steps_run * DT_MINUTES / 60:.2f} h{_RST}")
    print(f"  {_DIM}Total energy   :{_RST} {_W}{summary.total_energy_kwh:.3f} kWh{_RST}")
    print(f"  {_DIM}Total water    :{_RST} {_W}{summary.total_water_litres:.2f} L{_RST}")
    print(f"  {_DIM}Total MPC cost :{_RST} {_W}{summary.total_cost:.4f}{_RST}")
    if artifact_dir:
        print(f"  {_DIM}Artifacts saved:{_RST} {_W}{artifact_dir}{_RST}")
    print()
    print(f"  To delete this run's data from the DB:")
    print(
        f"    {_Y}DELETE FROM realtime_greenhouse_stream "
        f"WHERE run_id = '{summary.run_id}';{_RST}"
    )
    print("\u2500" * 62)


# ==============================================================================
# Artifact saving
# ==============================================================================

def setup_artifact_dir(run_id: str) -> Path:
    """Create and return the run artifact directory under logs/."""
    art_dir = _ROOT / "logs" / "realtime" / run_id
    art_dir.mkdir(parents=True, exist_ok=True)
    return art_dir


def save_step_artifact(art_dir: Path, result: RealtimeStepResult) -> None:
    """Append a compact step dict to the NDJSON log file."""
    payload = result.payload
    log_file = art_dir / "steps.ndjson"
    record = {
        "step": result.step_index,
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
        fh.write(json.dumps(record, default=str) + "\n")


def save_run_manifest(
    art_dir: Path,
    summary: RealtimeRunSummary,
    config: RealtimeLoopConfig,
) -> None:
    """Write a JSON manifest summarising the run."""
    manifest = {
        "run_id": summary.run_id,
        "growth_stage": summary.growth_stage,
        "days_elapsed_at_start": config.days_elapsed,
        "planned_steps": config.total_steps,
        "steps_run": summary.steps_run,
        "start_ts": str(summary.start_ts),
        "end_ts": str(summary.end_ts),
        "simulated_hours": summary.steps_run * DT_MINUTES / 60,
        "total_energy_kwh": summary.total_energy_kwh,
        "total_water_litres": summary.total_water_litres,
        "total_mpc_cost": summary.total_cost,
        "mpc_every_steps": config.mpc_every,
        "images_enabled": not config.no_images,
        "device": config.device,
        "db_table": "realtime_greenhouse_stream",
        "delete_query": (
            f"DELETE FROM realtime_greenhouse_stream "
            f"WHERE run_id = '{summary.run_id}';"
        ),
    }
    with open(art_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    # ── Update persistent run registry ────────────────────────────
    RunRegistry().register(summary, config, artifact_dir=art_dir)


# ==============================================================================
# Graceful shutdown
# ==============================================================================

_stop_requested = False


def _handle_signal(signum: int, frame: Any) -> None:
    global _stop_requested
    _stop_requested = True
    print(f"\n  {_Y}[INFO]{_RST} Stop signal received \u2014 finishing current step \u2026")


# ==============================================================================
# Main entry: CLI -> RealtimeLoopConfig -> RealtimeLoop
# ==============================================================================

def run_from_cli(args: argparse.Namespace) -> None:
    """Bootstrap and execute the real-time closed-loop from parsed CLI args."""
    global _stop_requested

    # -- Show delete queries and exit if requested ---------------------
    if args.show_delete_query:
        print(_DELETE_QUERIES)
        return

    # -- Interactive prompts (only if not provided via CLI) ------------
    if args.stage:
        stage_lower = args.stage.lower()
        if stage_lower not in GROWTH_STAGES:
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

    # -- Build config --------------------------------------------------
    config = RealtimeLoopConfig(
        growth_stage=growth_stage,
        days_elapsed=days_elapsed,
        total_steps=total_steps,
        mpc_every=max(1, args.mpc_every),
        device=args.device,
        no_images=args.no_images,
        dry_run=args.dry_run,
    )

    # -- Load image classifiers (optional) ----------------------------
    disease_classifier = None
    growth_classifier = None

    if not config.no_images:
        try:
            from agritwin_gh.models.disease_inference import predict_image
            from agritwin_gh.models.growth_stage_inference import predict_growth_stage

            disease_classifier = predict_image
            growth_classifier = predict_growth_stage
        except Exception as exc:
            print(
                f"  {_Y}[WARN]{_RST} Image classifiers not available ({exc}). "
                "Continuing without them."
            )

    # -- DB session ----------------------------------------------------
    db_manager = get_db_manager()
    session = db_manager.get_session()

    try:
        # -- Create loop -----------------------------------------------
        loop = RealtimeLoop(
            config,
            session,
            disease_classifier=disease_classifier,
            growth_classifier=growth_classifier,
        )

        # -- Artifact directory ----------------------------------------
        art_dir = setup_artifact_dir(loop.run_id) if not config.dry_run else None

        # -- Setup (seed DB) -------------------------------------------
        print(f"\n  {_DIM}Seeding initial state from DB ...{_RST}")
        loop.setup()

        # -- Banner ----------------------------------------------------
        print_banner(loop.run_id, growth_stage, days_elapsed, total_steps)

        # -- Signal handling -------------------------------------------
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        # -- Callbacks -------------------------------------------------
        def on_step(result: RealtimeStepResult) -> None:
            print_step_line(result, total_steps)
            if art_dir is not None:
                save_step_artifact(art_dir, result)

        def should_stop() -> bool:
            return _stop_requested

        # -- Run the loop ----------------------------------------------
        summary = loop.run(on_step=on_step, should_stop=should_stop)

        # -- Save manifest + print summary ----------------------------
        if art_dir is not None:
            save_run_manifest(art_dir, summary, config)

        print_run_summary(summary, art_dir)

    finally:
        session.close()
        db_manager.close()


# ==============================================================================
# Argument parser
# ==============================================================================

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


# ==============================================================================
# Entry point
# ==============================================================================

def main() -> None:
    args = build_parser().parse_args()
    run_from_cli(args)


if __name__ == "__main__":
    main()