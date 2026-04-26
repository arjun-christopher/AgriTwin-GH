#!/usr/bin/env python3
"""
AgriTwin-GH — Greenhouse Scenario Runner
=========================================

Drives the 3-D Unity greenhouse through a curated set of real-world
growing conditions by writing the canonical ``greenhouse_state.json``
file that ``GreenhouseStateApplier.cs`` polls every second.

Run order
---------
  Phase 1  Start ``main.py`` as a background process, wait for the
           backend health-check, then open the WebGL greenhouse in
           the default browser.

  Phase 2  Walk through every scenario one at a time.
           Press **Enter** to apply the next scenario.
           The 3-D scene reflects each change within ≤ 1 second.

Usage
-----
    python scripts/greenhouse_scenarios.py

Environment overrides
---------------------
    AGRITWIN_NO_SERVER=1   Skip launching main.py (use an already-running backend)
    AGRITWIN_API_URL=...   Base URL  (default: http://localhost:8000)
    AGRITWIN_JSON_PATH=... Absolute path to greenhouse_state.json
                           (default: auto-resolved from repo root)

Exit codes
----------
    0  All scenarios applied successfully.
    1  Backend failed to start, or JSON file could not be written.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import threading
import webbrowser
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


# ── env / path resolution ─────────────────────────────────────────────────────

BASE_URL        = os.environ.get("AGRITWIN_API_URL",  "http://localhost:8000").rstrip("/")
NO_SERVER       = os.environ.get("AGRITWIN_NO_SERVER", "0").strip() == "1"
GREENHOUSE_URL  = f"{BASE_URL}/greenhouse-3d/"
STATE_URL       = f"{BASE_URL}/api/greenhouse-3d/state"

SCRIPT_DIR      = Path(__file__).resolve().parent
REPO_ROOT       = SCRIPT_DIR.parent
MAIN_PY         = REPO_ROOT / "main.py"

# The Unity scene polls this file every second.
_DEFAULT_JSON   = REPO_ROOT / "unity_module" / "Assets" / "StreamingAssets" / "greenhouse_state.json"
JSON_PATH       = Path(os.environ.get("AGRITWIN_JSON_PATH", str(_DEFAULT_JSON)))


# ── ANSI helpers ──────────────────────────────────────────────────────────────

def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m"

def _green(s: str)  -> str: return _c("32", s)
def _red(s: str)    -> str: return _c("31", s)
def _yellow(s: str) -> str: return _c("33", s)
def _cyan(s: str)   -> str: return _c("36", s)
def _bold(s: str)   -> str: return _c("1",  s)
def _dim(s: str)    -> str: return _c("2",  s)


# ── scenario catalogue ────────────────────────────────────────────────────────
#
# Every scenario is a dict with two mandatory keys:
#   "name"        Short display title shown in the terminal header
#   "description" One-sentence explanation of what the scenario represents
#
# All remaining keys map 1-to-1 onto the greenhouse_state.json schema
# documented in GREENHOUSE_3D_MODEL_REFERENCE.md:
#
#   fluorescentLight  { isOn: bool }
#   heater            { isOn: bool }
#   energyCanister    { isOn: bool }
#   humidifier        { isOn: bool }
#   windowFan         { isOn: bool }
#   vent              { isOn: bool }
#   waterTankFloor    { isOn: bool }
#   cropStage         { stage: "Seedling|Vegetative|FloweringInitiation|Flowering|Unripe|Ripe" }
#   timeOfDay         { time:  "Morning|Afternoon|Evening|Night" }
#   cropHealth        { state: "Green|Yellow|Red", blinkGreen: bool }
#
# blinkGreen is kept for schema compatibility but is ALWAYS auto-computed
# by GreenhouseStateApplier: it activates only when ALL crops are Ripe.

SCENARIOS: list[dict[str, Any]] = [

    # ── 1 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Germination — Pre-Dawn Start",
        "description": "Seeds have just been planted. Minimal systems on; "
                       "gentle humidity to encourage sprouting before sunrise.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": False},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Seedling"},
        "timeOfDay":        {"time":  "Night"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 2 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Seedling — Morning Warm-Up",
        "description": "Young seedlings in their first week. Lights on, "
                       "heater off (seedlings are heat-sensitive), "
                       "humidifier running, morning sky.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": False},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Seedling"},
        "timeOfDay":        {"time":  "Morning"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 3 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Vegetative — Peak Growth Afternoon",
        "description": "Plants in rapid leaf/stem expansion. Full light, "
                       "gentle airflow from the fan, water tank active, "
                       "all energy online. Bright afternoon sun.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Vegetative"},
        "timeOfDay":        {"time":  "Afternoon"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 4 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Vegetative — Mild Heat Stress Warning",
        "description": "Temperature crept up mid-afternoon. Vent opened, "
                       "fan on, humidifier off to reduce moisture load. "
                       "Yellow health indicator — operator should monitor.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": True},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Vegetative"},
        "timeOfDay":        {"time":  "Afternoon"},
        "cropHealth":       {"state": "Yellow", "blinkGreen": False},
    },

    # ── 5 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Flowering Initiation — Evening Transition",
        "description": "Buds beginning to form as daylight shortens. "
                       "Heater brought online for the cooler evening, "
                       "fan reduced, warm orange sky.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "FloweringInitiation"},
        "timeOfDay":        {"time":  "Evening"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 6 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Full Bloom — Optimal Conditions",
        "description": "All flowers open. Every system at optimal settings: "
                       "lights, heater, humidity, gentle airflow. "
                       "Afternoon peak-production window.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Flowering"},
        "timeOfDay":        {"time":  "Afternoon"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 7 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Full Bloom — Disease Alert (Critical)",
        "description": "Fungal pathogen detected mid-bloom. Fan and vent "
                       "both on to purge spores, humidifier off, red health "
                       "light active — immediate operator intervention needed.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": True},
        "waterTankFloor":   {"isOn": False},
        "cropStage":        {"stage": "Flowering"},
        "timeOfDay":        {"time":  "Night"},
        "cropHealth":       {"state": "Red", "blinkGreen": False},
    },

    # ── 8 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Unripe Fruit — Night Ripening Mode",
        "description": "Fruit is set but needs warm nights to accumulate "
                       "sugars. Heater on, lights dimmed via energy saver, "
                       "no airflow to retain warmth.",
        "fluorescentLight": {"isOn": False},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Unripe"},
        "timeOfDay":        {"time":  "Night"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 9 ─────────────────────────────────────────────────────────────────────
    {
        "name":        "Unripe Fruit — Water Deficit Stress",
        "description": "Water tank ran low overnight. Yellow health warning; "
                       "humidifier compensating, fan off to reduce "
                       "transpiration until tank is refilled.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": False},
        "cropStage":        {"stage": "Unripe"},
        "timeOfDay":        {"time":  "Morning"},
        "cropHealth":       {"state": "Yellow", "blinkGreen": False},
    },

    # ── 10 ────────────────────────────────────────────────────────────────────
    {
        "name":        "Ripe — Harvest-Ready (Full Blink)",
        "description": "All crops fully mature. Every system running at "
                       "peak. Green health indicator automatically blinking "
                       "(Unity auto-override: all crops Ripe). "
                       "Ready for harvest pickup.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": True},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Ripe"},
        "timeOfDay":        {"time":  "Afternoon"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 11 ────────────────────────────────────────────────────────────────────
    {
        "name":        "Post-Harvest — Greenhouse Reset (All Off)",
        "description": "Harvest complete. All actuators off, lights off, "
                       "stage reset to Seedling ready for the next cycle. "
                       "Night mode for dormancy.",
        "fluorescentLight": {"isOn": False},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": False},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": False},
        "cropStage":        {"stage": "Seedling"},
        "timeOfDay":        {"time":  "Night"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 12 ────────────────────────────────────────────────────────────────────
    {
        "name":        "Emergency — Total System Failure",
        "description": "Power fault detected. All actuators off except "
                       "emergency lighting. Crops stuck at Flowering. "
                       "Red health alert — critical system failure.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": False},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": False},
        "cropStage":        {"stage": "Flowering"},
        "timeOfDay":        {"time":  "Night"},
        "cropHealth":       {"state": "Red", "blinkGreen": False},
    },

    # ── 13 ────────────────────────────────────────────────────────────────────
    {
        "name":        "Cold Snap — Winter Night Protocol",
        "description": "Unexpected cold front overnight. Heater maxed, "
                       "all vents and fans closed to retain heat, "
                       "humidifier off. Crops mid-vegetative, health green.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": False},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Vegetative"},
        "timeOfDay":        {"time":  "Night"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },

    # ── 14 ────────────────────────────────────────────────────────────────────
    {
        "name":        "Heatwave — Maximum Ventilation",
        "description": "External temperature spike. Fan + vent both open, "
                       "heater off, humidifier off. Lights dimmed. "
                       "Yellow warning — operator watching closely.",
        "fluorescentLight": {"isOn": False},
        "heater":           {"isOn": False},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": False},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": True},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Flowering"},
        "timeOfDay":        {"time":  "Afternoon"},
        "cropHealth":       {"state": "Yellow", "blinkGreen": False},
    },

    # ── 15 ────────────────────────────────────────────────────────────────────
    {
        "name":        "Ideal Cycle — End-to-End Showcase",
        "description": "Demonstration state: every system on, full bloom, "
                       "bright afternoon, green health. "
                       "Perfect reference for investor or visitor walkthroughs.",
        "fluorescentLight": {"isOn": True},
        "heater":           {"isOn": True},
        "energyCanister":   {"isOn": True},
        "humidifier":       {"isOn": True},
        "windowFan":        {"isOn": True},
        "vent":             {"isOn": False},
        "waterTankFloor":   {"isOn": True},
        "cropStage":        {"stage": "Flowering"},
        "timeOfDay":        {"time":  "Afternoon"},
        "cropHealth":       {"state": "Green", "blinkGreen": False},
    },
]


# ── API writer ────────────────────────────────────────────────────────────────

def _build_state(scenario: dict[str, Any]) -> dict[str, Any]:
    """Return a clean payload (schema keys only)."""
    keys = [
        "fluorescentLight", "heater", "energyCanister", "humidifier",
        "windowFan", "vent", "waterTankFloor",
        "cropStage", "timeOfDay", "cropHealth",
    ]
    return {k: scenario[k] for k in keys if k in scenario}


def write_state(scenario: dict[str, Any]) -> bool:
    """POST the scenario state to /greenhouse-3d/state. Returns True on success."""
    payload = json.dumps(_build_state(scenario)).encode("utf-8")
    req = urllib.request.Request(
        STATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except urllib.error.HTTPError as exc:
        print(_red(f"  HTTP {exc.code}: {exc.reason}"))
        return False
    except Exception as exc:
        print(_red(f"  ERROR posting state: {exc}"))
        return False


# ── display helpers ───────────────────────────────────────────────────────────

_HEALTH_COLOR = {"Green": _green, "Yellow": _yellow, "Red": _red}

def _bool_icon(v: bool) -> str:
    return _green("●  ON ") if v else _dim("○  off")

def _stage_bar(stage: str) -> str:
    order = ["Seedling", "Vegetative", "FloweringInitiation",
             "Flowering", "Unripe", "Ripe"]
    idx   = next((i for i, s in enumerate(order)
                  if s.lower() == stage.lower()), -1)
    parts = []
    for i, s in enumerate(order):
        label = s if s != "FloweringInitiation" else "FlwrInit"
        parts.append(_cyan(f"[{label}]") if i == idx else _dim(f" {label} "))
    return "  ".join(parts)

def _time_icon(t: str) -> str:
    icons = {"Morning": "🌅", "Afternoon": "☀️ ", "Evening": "🌇", "Night": "🌙"}
    return icons.get(t, t)

def print_scenario(index: int, total: int, scenario: dict[str, Any]) -> None:
    name   = scenario["name"]
    desc   = scenario["description"]
    stage  = scenario["cropStage"]["stage"]
    tod    = scenario["timeOfDay"]["time"]
    health = scenario["cropHealth"]["state"]
    hfn    = _HEALTH_COLOR.get(health, str)

    width = 70
    print()
    print("╔" + "═" * (width - 2) + "╗")
    print(f"║  {_bold(f'Scenario [{index}/{total}]'):<{width + 7}}║")
    print(f"║  {_cyan(name):<{width + 9}}║")
    print("╠" + "═" * (width - 2) + "╣")
    print(f"║  {desc:<{width - 2}}║")
    print("╠" + "═" * (width - 2) + "╣")

    # Actuators grid — 2 columns
    actuator_fields = [
        ("fluorescentLight", "Fluorescent Light"),
        ("heater",           "Heater           "),
        ("energyCanister",   "Energy Canister  "),
        ("humidifier",       "Humidifier       "),
        ("windowFan",        "Window Fan       "),
        ("vent",             "Vent             "),
        ("waterTankFloor",   "Water Tank Floor "),
    ]
    print(f"║  {'ACTUATORS':<{width - 4}}  ║")
    for i in range(0, len(actuator_fields), 2):
        left_key,  left_label  = actuator_fields[i]
        left_val  = scenario.get(left_key, {}).get("isOn", False)
        left_cell = f"{left_label}: {_bool_icon(left_val)}"
        if i + 1 < len(actuator_fields):
            right_key, right_label = actuator_fields[i + 1]
            right_val  = scenario.get(right_key, {}).get("isOn", False)
            right_cell = f"{right_label}: {_bool_icon(right_val)}"
        else:
            right_cell = ""
        row = f"  {left_cell:<32}  {right_cell}"
        print(f"║{row:<{width - 1}}║")

    print("╠" + "═" * (width - 2) + "╣")
    print(f"║  {'ENVIRONMENT':<{width - 4}}  ║")
    print(f"║  Time of Day : {_time_icon(tod)} {tod:<{width - 22}}║")
    print(f"║  Crop Health : {hfn(f'■ {health}'):<{width - 3 + 9}}║")
    print("╠" + "═" * (width - 2) + "╣")
    print(f"║  {'GROWTH STAGE':<{width - 4}}  ║")
    bar_line = _stage_bar(stage)
    # strip ANSI for length calculation
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    bar_plain   = ansi_escape.sub("", bar_line)
    pad         = max(0, width - 4 - len(bar_plain))
    print(f"║  {bar_line}{' ' * pad}  ║")
    print("╚" + "═" * (width - 2) + "╝")


# ── Phase 1: backend ──────────────────────────────────────────────────────────

def _wait_for_server(timeout: int = 120) -> bool:
    health_url = f"{BASE_URL}/api/system/health"
    deadline   = time.monotonic() + timeout
    attempt    = 0
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(health_url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        if attempt % 5 == 0:
            remaining = int(deadline - time.monotonic())
            print(f"  … still waiting  ({remaining}s left)", flush=True)
        time.sleep(2)
    return False


def start_backend() -> subprocess.Popen | None:
    if NO_SERVER:
        print(_yellow("  AGRITWIN_NO_SERVER=1 — skipping server launch."))
        return None

    if not MAIN_PY.exists():
        print(_red(f"  ERROR: {MAIN_PY} not found."))
        sys.exit(1)

    print(_bold("\n── Phase 1: Starting backend + 3-D greenhouse ──"))
    print(f"  Entry point  : {MAIN_PY}")
    print(f"  3-D view     : {GREENHOUSE_URL}")
    print(f"  JSON target  : {JSON_PATH}")
    print("=" * 70)

    env = os.environ.copy()
    env["AGRITWIN_NO_FRONTEND"] = "1"

    proc = subprocess.Popen(
        [sys.executable, str(MAIN_PY)],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )

    def _pipe() -> None:
        assert proc.stdout
        for line in proc.stdout:
            print(f"  [server] {line}", end="", flush=True)

    threading.Thread(target=_pipe, daemon=True).start()

    print("  Waiting for backend to become healthy (up to 120s) …", flush=True)
    if not _wait_for_server():
        print(_red("  ERROR: Backend did not become healthy within 120s."))
        proc.terminate()
        sys.exit(1)

    print(_green("  Backend is healthy ✓"))
    print(f"  Opening 3-D greenhouse → {GREENHOUSE_URL}")
    webbrowser.open(GREENHOUSE_URL)
    print("  (Browser opened — the scene updates within 1 s of each scenario write)")
    print("=" * 70)
    return proc


# ── Phase 2: scenario runner ──────────────────────────────────────────────────

def run_scenarios() -> bool:
    total   = len(SCENARIOS)
    passed  = 0
    failed  = 0

    print(_bold(f"\n── Phase 2: Greenhouse Scenario Runner  ({total} scenarios) ──"))
    print("  Press  Enter  to apply the next scenario.")
    print("  Press  Ctrl-C  at any time to stop.")
    print("=" * 70)

    for idx, scenario in enumerate(SCENARIOS, start=1):
        # Prompt before every scenario (no prompt before the first one so the
        # banner is readable before anything happens)
        if idx == 1:
            try:
                input(_yellow(f"\n  ↵  Press Enter to apply Scenario 1/{total} …"))
            except (EOFError, KeyboardInterrupt):
                print("\n  Stopped before first scenario.")
                return passed == total
        else:
            try:
                input(_yellow(f"\n  ↵  Press Enter for Scenario {idx}/{total} …"))
            except (EOFError, KeyboardInterrupt):
                print(f"\n  Stopped at scenario {idx - 1}/{total}.")
                break

        print_scenario(idx, total, scenario)

        t0 = time.monotonic()
        ok = write_state(scenario)
        elapsed_ms = (time.monotonic() - t0) * 1000

        if ok:
            passed += 1
            print(_green(f"  ✓  POSTed to API  ({elapsed_ms:.0f} ms)  "
                         f"— Unity scene will update within 1 s"))
        else:
            failed += 1
            print(_red(f"  ✗  Failed to POST state to API"))

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    result_str = f"{passed}/{total} scenarios written"
    if failed == 0:
        print(_green(f"  Scenarios : {result_str}"))
    else:
        print(_red(f"  Scenarios : {result_str}  ({failed} failed)"))

    return failed == 0


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(_bold(f"\nAgriTwin-GH Scenario Runner"))
    print(f"  Backend URL  : {BASE_URL}")
    print(f"  JSON path    : {JSON_PATH}")
    print(f"  Total scenes : {len(SCENARIOS)}")

    server_proc = start_backend()

    success = run_scenarios()

    if server_proc:
        print(_yellow(f"\n  Backend still running — 3-D view live at {GREENHOUSE_URL}"))
        print(_yellow("  Press Ctrl-C to shut down the backend when done."))
        try:
            server_proc.wait()
        except KeyboardInterrupt:
            print("\n[runner] Shutting down backend …")
            server_proc.terminate()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()