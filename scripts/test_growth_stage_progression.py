"""
test_growth_stage_progression.py
──────────────────────────────────
Standalone test script for the Tomato Growth Stage Progression model
(Multi-task LSTM: current stage, next stage, hours-to-transition, t24/t48 probs).

10 scenarios covering each growth stage, transition boundaries, stress conditions,
and day/night comparison.

Feature layout (first 33 primary features out of 358):
  0  year              1  month             2  day_of_year
  3  week_of_year      4  hour              5  days_from_cycle_start
  6  stage_index       7  hours_in_current_stage  8  days_in_current_stage
  9  stage_duration_hours   10 stage_duration_days  11 stage_progress_pct
  12 total_cycle_progress_pct  13 estimated_days_to_next_stage
  14 estimated_hours_to_next_stage  15 is_stage_transition
  16 indoor_temp       17 indoor_humidity    18 indoor_air_velocity
  19 indoor_co2        20 solarradiation     21 day_night_flag
  22 vpd               23 dew_point          24 leaf_wetness_proxy
  25 temperature_rolling_mean_24h  26 humidity_rolling_mean_24h
  27 vpd_proxy         28 light_period_flag  29 cumulative_gdd_like_index
  30 elapsed_hours     31 hour_of_day        32 day_of_cycle
  (remaining 325 features are rolling/lag derivatives — set to primary values)

Usage:
    python scripts/test_growth_stage_progression.py            # run all 10
    python scripts/test_growth_stage_progression.py --scenario 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

# ── Resolve repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Model / artifact paths ────────────────────────────────────────────────────
MODEL_PATH   = REPO_ROOT / "src/agritwin_gh/models/growth_stage_progression_20260310_192038.keras"
ARTIFACTS_DIR = REPO_ROOT / "src/agritwin_gh/models/artifacts/growth_stage_progression_20260310_192038"
SCALER_PATH  = ARTIFACTS_DIR / "feature_scaler.pkl"
INF_CFG_PATH = ARTIFACTS_DIR / "inference_config.json"

# ── Constants (from inference_config.json) ───────────────────────────────────
SEQ_LEN    = 24
N_FEATURES = 358
HRS_MEAN   = 270.3345031738281
HRS_STD    = 192.34524536132812

STAGE_TO_INT = {
    "seedling"              : 0,
    "early_vegetative"      : 1,
    "flowering_initiation"  : 2,
    "flowering"             : 3,
    "unripe"                : 4,
    "ripe"                  : 5,
}
INT_TO_STAGE = {v: k for k, v in STAGE_TO_INT.items()}

STAGE_DISPLAY = {
    "seedling"              : "Stage 1 – Seedling",
    "early_vegetative"      : "Stage 2 – Early Vegetative",
    "flowering_initiation"  : "Stage 3 – Flowering Initiation",
    "flowering"             : "Stage 4 – Flowering",
    "unripe"                : "Stage 5 – Unripe",
    "ripe"                  : "Stage 6 – Ripe",
    "harvest"               : "End of Cycle (Harvest)",
}

# Typical total cycle hours per stage (approximate, for sequence generation)
STAGE_DURATION_HRS = {
    "seedling"             : 240,
    "early_vegetative"     : 360,
    "flowering_initiation" : 240,
    "flowering"            : 480,
    "unripe"               : 720,
    "ripe"                 : 480,
}

# ── Sequence builder helpers ──────────────────────────────────────────────────

def _blank_sequence() -> np.ndarray:
    """Return a zeroed raw sequence (24, 358)."""
    return np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)


def _fill_sequence(
    seq: np.ndarray,
    stage: str,
    days_from_cycle_start: float,
    hours_in_stage: float,
    stage_progress_pct: float,
    total_cycle_progress_pct: float,
    estimated_hrs_to_next: float,
    temp: float,
    humidity: float,
    air_vel: float = 1.2,
    co2: float = 800.0,
    solar: float = 400.0,
    day_night: float = 1.0,
    vpd: float = 1.0,
    gdd: float = 50.0,
) -> np.ndarray:
    """
    Broadcast the supplied parameter values to all 24 timesteps.
    Primary features (indices 0–32) are set explicitly.
    The remaining 325 rolling/lag features are set equal to the primary values
    for the matching channel so the scaler can normalise them consistently.
    """
    stage_idx = float(STAGE_TO_INT[stage])
    dur_hrs    = float(STAGE_DURATION_HRS[stage])
    dur_days   = dur_hrs / 24.0
    dew_point  = temp - (100.0 - humidity) / 5.0   # Magnus approximation

    for t in range(SEQ_LEN):
        s = seq[t]
        s[0]  = 2026.0                               # year
        s[1]  = float(int(days_from_cycle_start / 30) + 1)  # month
        s[2]  = float(days_from_cycle_start % 365)   # day_of_year
        s[3]  = float(days_from_cycle_start / 7)     # week_of_year
        s[4]  = float((t * 1) % 24)                  # hour (increments per step)
        s[5]  = days_from_cycle_start
        s[6]  = stage_idx
        s[7]  = hours_in_stage
        s[8]  = hours_in_stage / 24.0
        s[9]  = dur_hrs
        s[10] = dur_days
        s[11] = stage_progress_pct
        s[12] = total_cycle_progress_pct
        s[13] = estimated_hrs_to_next / 24.0         # days to next
        s[14] = estimated_hrs_to_next
        s[15] = 0.0                                  # is_stage_transition
        s[16] = temp
        s[17] = humidity
        s[18] = air_vel
        s[19] = co2
        s[20] = solar
        s[21] = day_night
        s[22] = vpd
        s[23] = dew_point
        s[24] = humidity * 0.4                       # leaf_wetness_proxy
        s[25] = temp                                 # temperature_rolling_mean_24h
        s[26] = humidity                             # humidity_rolling_mean_24h
        s[27] = vpd                                  # vpd_proxy
        s[28] = day_night                            # light_period_flag
        s[29] = gdd                                  # cumulative_gdd_like_index
        s[30] = days_from_cycle_start * 24.0         # elapsed_hours
        s[31] = float((t * 1) % 24)                  # hour_of_day
        s[32] = days_from_cycle_start               # day_of_cycle
        # Fill rolling/lag features with the same primary values (columns 33–357)
        # Group repeating pattern: many rolling means for each primary feature
        for col in range(33, N_FEATURES):
            # Cycle the primary feature values across the rolling slots
            src = col % 33
            s[col] = s[src]

    return seq


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    print(f"Loading model   : {MODEL_PATH.name}")
    model  = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    scaler = joblib.load(str(SCALER_PATH))
    print(f"  Model loaded.  Input shape: {model.input_shape}")
    print(f"  Scaler loaded. n_features_in_: {scaler.n_features_in_}\n")
    return model, scaler


# ── Inference helper ──────────────────────────────────────────────────────────

def _predict(raw_seq: np.ndarray, model, scaler) -> dict:
    """
    raw_seq : (24, 358) float array.
    Returns current_stage, next_stage, hrs_to_next, t24_prob, t48_prob.
    """
    scaled  = scaler.transform(raw_seq)              # (24, 358)
    X_input = scaled[np.newaxis, :, :]               # (1, 24, 358)
    preds   = model.predict(X_input, verbose=0)

    cur_idx  = int(np.argmax(preds[0][0]))
    nxt_idx  = int(np.argmax(preds[1][0]))
    hrs_raw  = float(preds[2][0, 0]) * HRS_STD + HRS_MEAN
    t24_prob = float(preds[3][0, 0])
    t48_prob = float(preds[4][0, 0])

    cur_stage = INT_TO_STAGE.get(cur_idx, str(cur_idx))
    nxt_stage = INT_TO_STAGE.get(nxt_idx, str(nxt_idx))

    # Ripe is the final stage — no meaningful next stage or upcoming transition
    if cur_stage == "ripe":
        nxt_stage = "harvest"
        t24_prob  = 0.0
        t48_prob  = 0.0

    return {
        "current_stage"  : cur_stage,
        "current_display": STAGE_DISPLAY.get(cur_stage, cur_stage),
        "next_stage"     : nxt_stage,
        "next_display"   : STAGE_DISPLAY.get(nxt_stage, nxt_stage),
        "hrs_to_next"    : max(hrs_raw, 0.0),
        "t24_prob"       : np.clip(t24_prob, 0.0, 1.0),
        "t48_prob"       : np.clip(t48_prob, 0.0, 1.0),
        "cur_logits"     : preds[0][0].tolist(),
        "nxt_logits"     : preds[1][0].tolist(),
    }


# ── Result printer ────────────────────────────────────────────────────────────

def _print_result(result: dict, scenario_num: int, scenario_name: str) -> None:
    t24 = result["t24_prob"]
    t48 = result["t48_prob"]
    hrs = result["hrs_to_next"]
    print(f"{'─'*70}")
    print(f"Scenario {scenario_num:>2}: {scenario_name}")
    print(f"  Current stage      : {result['current_display']}")
    print(f"  Next stage         : {result['next_display']}")
    print(f"  Hrs to transition  : {hrs:.1f} h  (~{hrs/24:.1f} days)")
    t24_bar = "█" * int(t24 * 30)
    t48_bar = "█" * int(t48 * 30)
    print(f"  Transition in 24h  : {t24:.3f}  {t24_bar}")
    print(f"  Transition in 48h  : {t48:.3f}  {t48_bar}")


# ── Scenario functions ────────────────────────────────────────────────────────

def run_scenario_1(model, scaler):
    """Seedling Day 1 — fresh transplant, low progress, expect Stage 1 current."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="seedling", days_from_cycle_start=1.0,
                   hours_in_stage=5.0, stage_progress_pct=2.0,
                   total_cycle_progress_pct=1.0, estimated_hrs_to_next=235.0,
                   temp=22.0, humidity=65.0, solar=200.0, vpd=0.7)
    result = _predict(seq, model, scaler)
    _print_result(result, 1, "Seedling Day 1 — freshly transplanted")
    return result


def run_scenario_2(model, scaler):
    """Seedling near transition — stage_progress_pct=92%, high t24 expected."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="seedling", days_from_cycle_start=9.0,
                   hours_in_stage=215.0, stage_progress_pct=90.0,
                   total_cycle_progress_pct=8.5, estimated_hrs_to_next=20.0,
                   temp=23.0, humidity=60.0, solar=250.0, vpd=0.8)
    result = _predict(seq, model, scaler)
    _print_result(result, 2, "Seedling near transition — progress 90%, ~20h to next stage")
    print(f"  Expected: high t24 probability")
    return result


def run_scenario_3(model, scaler):
    """Early Vegetative — stable mid-stage, low transition probability."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="early_vegetative", days_from_cycle_start=18.0,
                   hours_in_stage=180.0, stage_progress_pct=50.0,
                   total_cycle_progress_pct=15.0, estimated_hrs_to_next=180.0,
                   temp=24.0, humidity=62.0, solar=350.0, vpd=0.9, co2=850.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 3, "Early Vegetative — mid-stage stable (50% progress)")
    print(f"  Expected: low transition probabilities")
    return result


def run_scenario_4(model, scaler):
    """Flowering Initiation — onset of budding, expect Stage 3 current."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="flowering_initiation", days_from_cycle_start=32.0,
                   hours_in_stage=60.0, stage_progress_pct=25.0,
                   total_cycle_progress_pct=28.0, estimated_hrs_to_next=180.0,
                   temp=24.5, humidity=58.0, solar=500.0, vpd=1.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 4, "Flowering Initiation — Stage 3, early in stage")
    return result


def run_scenario_5(model, scaler):
    """Full Flowering — optimal conditions, high solar, anthesis peak."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="flowering", days_from_cycle_start=52.0,
                   hours_in_stage=240.0, stage_progress_pct=50.0,
                   total_cycle_progress_pct=42.0, estimated_hrs_to_next=240.0,
                   temp=24.0, humidity=55.0, solar=650.0, vpd=1.2, co2=900.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 5, "Full Flowering — mid-stage optimal conditions")
    return result


def run_scenario_6(model, scaler):
    """Unripe approaching Ripe — stage_progress 85%, t24/t48 should be high."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="unripe", days_from_cycle_start=78.0,
                   hours_in_stage=612.0, stage_progress_pct=85.0,
                   total_cycle_progress_pct=72.0, estimated_hrs_to_next=15.0,
                   temp=25.0, humidity=60.0, solar=500.0, vpd=1.1)
    result = _predict(seq, model, scaler)
    _print_result(result, 6, "Unripe → Ripe transition (85% progress, ~15h remaining)")
    print(f"  Expected: high t24 + t48 probability")
    return result


def run_scenario_7(model, scaler):
    """Ripe final stage — fruit ready, very high cycle progress."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="ripe", days_from_cycle_start=108.0,
                   hours_in_stage=380.0, stage_progress_pct=80.0,
                   total_cycle_progress_pct=95.0, estimated_hrs_to_next=0.0,
                   temp=25.0, humidity=58.0, solar=450.0, vpd=1.1)
    result = _predict(seq, model, scaler)
    _print_result(result, 7, "Ripe — Stage 6, final harvest phase (95% cycle progress)")
    return result


def run_scenario_8(model, scaler):
    """Cold stress during Early Vegetative — temp 10°C, reduced light."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="early_vegetative", days_from_cycle_start=20.0,
                   hours_in_stage=200.0, stage_progress_pct=55.0,
                   total_cycle_progress_pct=17.0, estimated_hrs_to_next=160.0,
                   temp=10.0, humidity=70.0, solar=80.0, vpd=0.3, gdd=10.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 8, "Cold stress during Early Vegetative — 10°C + low light")
    print(f"  Expected: delayed transition (more hrs_to_next) vs normal")
    return result


def run_scenario_9(model, scaler):
    """Heat stress during Flowering — temp 38°C, affects fruit set."""
    seq = _blank_sequence()
    _fill_sequence(seq, stage="flowering", days_from_cycle_start=48.0,
                   hours_in_stage=200.0, stage_progress_pct=42.0,
                   total_cycle_progress_pct=38.0, estimated_hrs_to_next=280.0,
                   temp=38.0, humidity=35.0, solar=800.0, vpd=3.5, gdd=120.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 9, "Heat stress during Flowering — 38°C (above pollen viability limit)")
    return result


def run_scenario_10(model, scaler):
    """Day vs Night comparison — same Flowering stage, toggle day_night_flag."""
    seq_day   = _blank_sequence()
    seq_night = _blank_sequence()
    common_kwargs = dict(
        stage="flowering", days_from_cycle_start=55.0,
        hours_in_stage=300.0, stage_progress_pct=62.0,
        total_cycle_progress_pct=46.0, estimated_hrs_to_next=180.0,
        temp=24.0, humidity=58.0, solar=550.0, vpd=1.1
    )
    _fill_sequence(seq_day,   day_night=1.0, **common_kwargs)
    _fill_sequence(seq_night, day_night=0.0, **common_kwargs)
    # Night: no solar radiation
    seq_night[:, 20] = 0.0

    res_day   = _predict(seq_day,   model, scaler)
    res_night = _predict(seq_night, model, scaler)
    print(f"{'─'*70}")
    print(f"Scenario 10: Day vs Night — Flowering Stage")
    print(f"  Daytime   → cur: {res_day['current_display']}  | t24: {res_day['t24_prob']:.3f}  | hrs: {res_day['hrs_to_next']:.1f}h")
    print(f"  Nighttime → cur: {res_night['current_display']}  | t24: {res_night['t24_prob']:.3f}  | hrs: {res_night['hrs_to_next']:.1f}h")
    return res_day


# ── Scenario registry ─────────────────────────────────────────────────────────
SCENARIOS = {
    1:  ("Seedling Day 1 — freshly transplanted",                   run_scenario_1),
    2:  ("Seedling near transition (90% progress)",                 run_scenario_2),
    3:  ("Early Vegetative — stable mid-stage",                     run_scenario_3),
    4:  ("Flowering Initiation — Stage 3 onset",                    run_scenario_4),
    5:  ("Full Flowering — optimal conditions",                     run_scenario_5),
    6:  ("Unripe → Ripe transition (85% progress)",                 run_scenario_6),
    7:  ("Ripe — Stage 6 final phase (95% cycle)",                  run_scenario_7),
    8:  ("Cold stress during Early Vegetative (10°C)",              run_scenario_8),
    9:  ("Heat stress during Flowering (38°C)",                     run_scenario_9),
    10: ("Day vs Night comparison — Flowering stage",               run_scenario_10),
}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone test suite for the Tomato Growth Stage Progression model"
    )
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Run a specific scenario number (1–10). Default 0 runs all."
    )
    args = parser.parse_args()

    model, scaler = load_model()

    if args.scenario == 0:
        print(f"Running all {len(SCENARIOS)} scenarios ...\n")
        passed = failed = 0
        for idx, (name, fn) in SCENARIOS.items():
            try:
                fn(model, scaler)
                passed += 1
            except Exception as exc:
                print(f"\n  [FAIL] Scenario {idx} — {exc}")
                failed += 1
        print(f"\n{'═'*70}")
        print(f"Results: {passed} passed / {failed} failed  (total {len(SCENARIOS)})")
    else:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario {args.scenario}. Choose 1–{len(SCENARIOS)}.")
            sys.exit(1)
        name, fn = SCENARIOS[args.scenario]
        fn(model, scaler)
