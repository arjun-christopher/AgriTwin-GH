"""
test_disease_progression.py
────────────────────────────
Standalone test script for the Tomato Disease Progression model (LSTM/GRU).

10 scenarios covering healthy baselines, single-disease outbreaks, environmental
stress conditions, treatment effects, and trend verification.

Feature layout (92 features):
  10  indoor_temp               11  indoor_humidity
  12  indoor_air_velocity       15  day_night_flag
  16  vpd                       18  leaf_wetness_proxy
  19  temperature_rolling_mean_24h   20  humidity_rolling_mean_24h
  26  control_action_flag__early_blight
  31  control_action_flag__late_blight
  36  control_action_flag__leaf_mold
  41  control_action_flag__powdery_mildew
  46  control_action_flag__spider_mites
  56  stage_name_early_vegetative    57  stage_name_flowering
  58  stage_name_flowering_initiation 59 stage_name_ripe
  60  stage_name_seedling            61  stage_name_unripe
  82  current_infection_pct__early_blight
  83  current_infection_pct__late_blight
  84  current_infection_pct__leaf_mold
  85  current_infection_pct__powdery_mildew
  86  current_infection_pct__spider_mites
  87  disease_present_flag__early_blight
  88  disease_present_flag__late_blight
  89  disease_present_flag__leaf_mold
  90  disease_present_flag__powdery_mildew
  91  disease_present_flag__spider_mites

Usage:
    python scripts/test_disease_progression.py             # run all 10 scenarios
    python scripts/test_disease_progression.py --scenario 2   # run one scenario
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
MODEL_PATH      = REPO_ROOT / "src/agritwin_gh/models/disease_progression_20260325_145100.keras"
ARTIFACTS_DIR   = REPO_ROOT / "src/agritwin_gh/models/artifacts/disease_progression_20260325_145100"
SCALER_PATH     = ARTIFACTS_DIR / "sequence_feature_scaler.joblib"
FEAT_COLS_PATH  = ARTIFACTS_DIR / "seq_feature_cols.json"
CONFIG_PATH     = ARTIFACTS_DIR / "config.json"

# ── Constants (from config.json) ──────────────────────────────────────────────
HISTORY_WINDOW      = 24
N_FEATURES          = 92
PRESENCE_THRESHOLD  = 0.5
SEVERITY_DELTA      = 3.0   # percentage points

DISEASES       = ["early_blight", "late_blight", "leaf_mold", "powdery_mildew", "spider_mites"]
DISEASE_LABELS = {
    "early_blight"   : "Early Blight",
    "late_blight"    : "Late Blight",
    "leaf_mold"      : "Leaf Mold",
    "powdery_mildew" : "Powdery Mildew",
    "spider_mites"   : "Spider Mites",
}

TREND_MAP = {0: "absent", 1: "emerging", 2: "reducing", 3: "stable", 4: "worsening"}

# Feature index shortcuts
IDX = {
    "indoor_temp"                        : 10,
    "indoor_humidity"                    : 11,
    "indoor_air_velocity"                : 12,
    "day_night_flag"                     : 15,
    "vpd"                                : 16,
    "leaf_wetness_proxy"                 : 18,
    "temperature_rolling_mean_24h"       : 19,
    "humidity_rolling_mean_24h"          : 20,
    "control_action_flag__early_blight"  : 26,
    "control_action_flag__late_blight"   : 31,
    "control_action_flag__leaf_mold"     : 36,
    "control_action_flag__powdery_mildew": 41,
    "control_action_flag__spider_mites"  : 46,
    "stage_name_early_vegetative"        : 56,
    "stage_name_flowering"               : 57,
    "stage_name_flowering_initiation"    : 58,
    "stage_name_ripe"                    : 59,
    "stage_name_seedling"                : 60,
    "stage_name_unripe"                  : 61,
    "sev__early_blight"                  : 82,
    "sev__late_blight"                   : 83,
    "sev__leaf_mold"                     : 84,
    "sev__powdery_mildew"                : 85,
    "sev__spider_mites"                  : 86,
    "flag__early_blight"                 : 87,
    "flag__late_blight"                  : 88,
    "flag__leaf_mold"                    : 89,
    "flag__powdery_mildew"               : 90,
    "flag__spider_mites"                 : 91,
}

SEV_IDX  = [IDX[f"sev__{d}"]  for d in DISEASES]
FLAG_IDX = [IDX[f"flag__{d}"] for d in DISEASES]
STAGE_IDXS = [
    IDX["stage_name_seedling"],
    IDX["stage_name_early_vegetative"],
    IDX["stage_name_flowering_initiation"],
    IDX["stage_name_flowering"],
    IDX["stage_name_unripe"],
    IDX["stage_name_ripe"],
]


# ── Sequence builder helpers ──────────────────────────────────────────────────

def _blank_sequence() -> np.ndarray:
    """Return a zeroed raw sequence of shape (24, 92)."""
    return np.zeros((HISTORY_WINDOW, N_FEATURES), dtype=np.float32)


def _set_stage(seq: np.ndarray, stage_name: str) -> np.ndarray:
    """Set the one-hot stage columns in all timesteps."""
    stage_map = {
        "seedling"              : IDX["stage_name_seedling"],
        "early_vegetative"      : IDX["stage_name_early_vegetative"],
        "flowering_initiation"  : IDX["stage_name_flowering_initiation"],
        "flowering"             : IDX["stage_name_flowering"],
        "unripe"                : IDX["stage_name_unripe"],
        "ripe"                  : IDX["stage_name_ripe"],
    }
    for idx in STAGE_IDXS:
        seq[:, idx] = 0.0
    if stage_name in stage_map:
        seq[:, stage_map[stage_name]] = 1.0
    return seq


def _set_env(seq: np.ndarray, temp: float, humidity: float,
             air_vel: float, vpd: float, leaf_wet: float,
             day_night: float = 1.0) -> np.ndarray:
    """Broadcast constant environmental readings across all 24 timesteps."""
    seq[:, IDX["indoor_temp"]]                  = temp
    seq[:, IDX["indoor_humidity"]]              = humidity
    seq[:, IDX["indoor_air_velocity"]]          = air_vel
    seq[:, IDX["vpd"]]                          = vpd
    seq[:, IDX["leaf_wetness_proxy"]]           = leaf_wet
    seq[:, IDX["day_night_flag"]]               = day_night
    seq[:, IDX["temperature_rolling_mean_24h"]] = temp
    seq[:, IDX["humidity_rolling_mean_24h"]]    = humidity
    return seq


def _set_disease(seq: np.ndarray, disease: str, sev: float,
                 present: float = 1.0) -> np.ndarray:
    """Set severity and presence flag for one disease in all timesteps."""
    sev_key  = f"sev__{disease}"
    flag_key = f"flag__{disease}"
    if sev_key in IDX:
        seq[:, IDX[sev_key]]  = sev
        seq[:, IDX[flag_key]] = present
    return seq


def _apply_treatment(seq: np.ndarray, disease: str) -> np.ndarray:
    """Activate control_action_flag for a disease across all timesteps."""
    ctl_key = f"control_action_flag__{disease}"
    if ctl_key in IDX:
        seq[:, IDX[ctl_key]] = 1.0
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
    raw_seq : (24, 92) float array of un-scaled feature values.
    Returns a dict with per-disease presence_prob, future_sev, current_sev, trend.
    """
    scaled   = scaler.transform(raw_seq)                  # (24, 92)
    X_input  = scaled[np.newaxis, :, :]                   # (1, 24, 92)
    preds    = model.predict(X_input, verbose=0)

    presence_prob = preds[0][0]                           # shape (5,)
    future_sev    = np.clip(preds[1][0] * 100.0, 0, 100) # shape (5,)
    presence_bin  = (presence_prob >= PRESENCE_THRESHOLD).astype(int)
    current_sev   = raw_seq[-1, SEV_IDX]                  # last timestep severities

    trends = []
    for i, d in enumerate(DISEASES):
        delta = float(future_sev[i] - current_sev[i])
        if presence_bin[i] == 0 and future_sev[i] < 2.0:
            trend = 0  # absent
        elif presence_bin[i] == 1 and current_sev[i] < 5.0:
            trend = 1  # emerging
        elif delta <= -SEVERITY_DELTA:
            trend = 2  # reducing
        elif delta >= SEVERITY_DELTA:
            trend = 4  # worsening
        else:
            trend = 3  # stable
        trends.append(trend)

    return {
        "diseases"      : DISEASES,
        "presence_prob" : presence_prob.tolist(),
        "presence_bin"  : presence_bin.tolist(),
        "current_sev"   : current_sev.tolist(),
        "future_sev"    : future_sev.tolist(),
        "trend"         : [TREND_MAP[t] for t in trends],
    }


# ── Result printer ────────────────────────────────────────────────────────────

def _print_result(result: dict, scenario_num: int, scenario_name: str) -> None:
    print(f"{'─'*70}")
    print(f"Scenario {scenario_num:>2}: {scenario_name}")
    print(f"  {'Disease':<20}  {'Prob':>6}  {'Present':>7}  {'CurSev':>6}  {'FutSev':>6}  Trend")
    print(f"  {'─'*20}  {'─'*6}  {'─'*7}  {'─'*6}  {'─'*6}  {'─'*10}")
    for i, d in enumerate(result["diseases"]):
        label   = DISEASE_LABELS[d]
        prob    = result["presence_prob"][i]
        present = "Yes" if result["presence_bin"][i] else "No"
        c_sev   = result["current_sev"][i]
        f_sev   = result["future_sev"][i]
        trend   = result["trend"][i]
        print(f"  {label:<20}  {prob:>6.3f}  {present:>7}  {c_sev:>5.1f}%  {f_sev:>5.1f}%  {trend}")


# ── Scenario functions ────────────────────────────────────────────────────────

def run_scenario_1(model, scaler):
    """Healthy greenhouse — optimal conditions, zero disease severity."""
    seq = _blank_sequence()
    _set_env(seq, temp=25.0, humidity=58.0, air_vel=1.5, vpd=0.9, leaf_wet=10.0)
    _set_stage(seq, "flowering")
    result = _predict(seq, model, scaler)
    _print_result(result, 1, "Healthy greenhouse — optimal conditions")
    return result


def run_scenario_2(model, scaler):
    """High humidity / poor ventilation — Leaf Mold + Late Blight already emerging."""
    seq = _blank_sequence()
    _set_env(seq, temp=24.0, humidity=88.0, air_vel=0.2, vpd=0.4, leaf_wet=75.0)
    _set_stage(seq, "flowering")
    # Both diseases already present at low severity — model should predict worsening
    _set_disease(seq, "leaf_mold",   sev=12.0, present=1.0)
    _set_disease(seq, "late_blight", sev=8.0,  present=1.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 2, "High humidity + poor ventilation (leaf mold 12% / late blight 8%)")
    print(f"  Expected: Leaf Mold + Late Blight worsening/stable in humid conditions")
    return result


def run_scenario_3(model, scaler):
    """Hot dry stress — Spider Mites + Powdery Mildew already emerging."""
    seq = _blank_sequence()
    _set_env(seq, temp=36.0, humidity=30.0, air_vel=0.8, vpd=2.5, leaf_wet=5.0)
    _set_stage(seq, "unripe")
    # Both dry-stress diseases present at low severity — expect model to flag worsening
    _set_disease(seq, "spider_mites",   sev=10.0, present=1.0)
    _set_disease(seq, "powdery_mildew", sev=8.0,  present=1.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 3, "Hot dry stress — spider mites 10% / powdery mildew 8%")
    print(f"  Expected: Spider Mites + Powdery Mildew worsening in hot/dry conditions")
    return result


def run_scenario_4(model, scaler):
    """Seedling stage — baseline reference with moderate conditions."""
    seq = _blank_sequence()
    _set_env(seq, temp=23.0, humidity=62.0, air_vel=1.2, vpd=0.8, leaf_wet=15.0)
    _set_stage(seq, "seedling")
    result = _predict(seq, model, scaler)
    _print_result(result, 4, "Seedling stage — moderate conditions baseline")
    return result


def run_scenario_5(model, scaler):
    """Ripe stage, damp environment — late-season disease pressure already active."""
    seq = _blank_sequence()
    _set_env(seq, temp=22.0, humidity=82.0, air_vel=0.5, vpd=0.5, leaf_wet=60.0)
    _set_stage(seq, "ripe")
    # Late-season damp: leaf mold and early blight both present
    _set_disease(seq, "leaf_mold",    sev=15.0, present=1.0)
    _set_disease(seq, "early_blight", sev=10.0, present=1.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 5, "Ripe stage — damp late-season (leaf mold 15%, early blight 10%)")
    print(f"  Expected: both diseases stable or worsening under continued damp")
    return result


def run_scenario_6(model, scaler):
    """Progressive Early Blight worsening — severity ramps from 5 → 40%."""
    seq = _blank_sequence()
    _set_env(seq, temp=26.0, humidity=70.0, air_vel=0.8, vpd=1.1, leaf_wet=35.0)
    _set_stage(seq, "flowering")
    # Build increasing severity over the 24 steps
    for t in range(HISTORY_WINDOW):
        sev_t = 5.0 + (35.0 / HISTORY_WINDOW) * t
        seq[t, IDX["sev__early_blight"]]  = sev_t
        seq[t, IDX["flag__early_blight"]] = 1.0
    result = _predict(seq, model, scaler)
    _print_result(result, 6, "Progressive worsening — Early Blight ramps 5→40%")
    print(f"  Expected Early Blight trend: worsening/stable")
    return result


def run_scenario_7(model, scaler):
    """Recovery scenario — Leaf Mold severity decreases from 45 → 5%."""
    seq = _blank_sequence()
    _set_env(seq, temp=24.5, humidity=65.0, air_vel=1.4, vpd=0.9, leaf_wet=20.0)
    _set_stage(seq, "early_vegetative")
    _apply_treatment(seq, "leaf_mold")   # treatment active
    for t in range(HISTORY_WINDOW):
        sev_t = max(45.0 - (40.0 / HISTORY_WINDOW) * t, 2.0)
        seq[t, IDX["sev__leaf_mold"]]  = sev_t
        seq[t, IDX["flag__leaf_mold"]] = 1.0
    result = _predict(seq, model, scaler)
    _print_result(result, 7, "Recovery — Leaf Mold severity drops 45→5% (treatment active)")
    print(f"  Expected Leaf Mold trend: reducing")
    return result


def run_scenario_8(model, scaler):
    """All diseases at high severity (60%) — verify all predicted present."""
    seq = _blank_sequence()
    _set_env(seq, temp=27.0, humidity=80.0, air_vel=0.3, vpd=0.7, leaf_wet=65.0)
    _set_stage(seq, "flowering")
    for d in DISEASES:
        _set_disease(seq, d, sev=60.0, present=1.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 8, "All diseases at 60% severity — all should be present")
    present_count = sum(result["presence_bin"])
    print(f"  Diseases predicted present: {present_count}/{len(DISEASES)}")
    return result


def run_scenario_9(model, scaler):
    """Nocturnal damp spell — leaf mold emerging under night humidity peak."""
    seq = _blank_sequence()
    _set_env(seq, temp=20.0, humidity=92.0, air_vel=0.1, vpd=0.3,
             leaf_wet=90.0, day_night=0.0)
    _set_stage(seq, "flowering")
    # Leaf mold is the primary night-damp pathogen; low initial sev to show emergence
    _set_disease(seq, "leaf_mold",   sev=8.0,  present=1.0)
    _set_disease(seq, "late_blight", sev=5.0,  present=1.0)
    result = _predict(seq, model, scaler)
    _print_result(result, 9, "Nocturnal damp — leaf mold 8% + late blight 5% at night")
    print(f"  Expected: Leaf Mold worsening under 92% humidity + high leaf wetness")
    return result


def run_scenario_10(model, scaler):
    """Post-treatment recovery — Spider Mites mid-severity, treatment flags on."""
    seq = _blank_sequence()
    _set_env(seq, temp=30.0, humidity=40.0, air_vel=1.0, vpd=1.8, leaf_wet=8.0)
    _set_stage(seq, "unripe")
    _set_disease(seq, "spider_mites", sev=30.0, present=1.0)
    _apply_treatment(seq, "spider_mites")
    result = _predict(seq, model, scaler)
    _print_result(result, 10, "Post-treatment recovery — Spider Mites 30% + treatment active")
    print(f"  Expected Spider Mites trend: reducing or stable")
    return result


# ── Scenario registry ─────────────────────────────────────────────────────────
SCENARIOS = {
    1:  ("Healthy greenhouse — optimal conditions",                   run_scenario_1),
    2:  ("High humidity / poor ventilation — Leaf Mold risk",        run_scenario_2),
    3:  ("Hot dry stress — Spider Mites + Powdery Mildew risk",      run_scenario_3),
    4:  ("Seedling stage — moderate conditions baseline",             run_scenario_4),
    5:  ("Ripe stage — damp late-season conditions",                  run_scenario_5),
    6:  ("Worsening — Early Blight severity ramps 5→40%",            run_scenario_6),
    7:  ("Recovery — Leaf Mold drops 45→5% with treatment",          run_scenario_7),
    8:  ("All diseases at 60% severity",                              run_scenario_8),
    9:  ("Nocturnal damp spell — night humidity peak",                run_scenario_9),
    10: ("Post-treatment — Spider Mites 30% + treatment active",     run_scenario_10),
}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone test suite for the Tomato Disease Progression model"
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
