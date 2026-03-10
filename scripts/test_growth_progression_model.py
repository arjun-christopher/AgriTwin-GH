#!/usr/bin/env python
"""
AgriTwin-GH :: TFT Growth Progression Model — Integration Test Suite
=====================================================================
Evaluates the trained Temporal Fusion Transformer against held-out test
cycles (7, 12, 16) using real-time-style greenhouse scenario windows.

Parts
-----
  A — Artifact loading and validation
  B — Test data preparation (chronological real-data windows)
  C — Functional tests (inference correctness & output structure)
  D — Scenario-based real-time tests (8 greenhouse scenario categories)
  E — Biological consistency checks
  F — Accuracy and regression metrics
  G — Forecast behavior and transition analysis
  H — Save all outputs, metrics, and plots

Usage
-----
    python tests/integration/test_growth_progression_model.py
    python tests/integration/test_growth_progression_model.py --max-windows 200
    python tests/integration/test_growth_progression_model.py \\
        --run-id growth_progression_20260308_141446 --max-windows 300
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
import uuid
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message="Found.*unknown classes",           category=UserWarning)
warnings.filterwarnings("ignore", message="The behavior of array concatenation", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
warnings.filterwarnings("ignore", message="Attribute.*is an instance of.*nn.Module", category=UserWarning)

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score,
        explained_variance_score, median_absolute_error,
    )
except ImportError:
    sys.exit("[ERROR] scikit-learn is required: pip install scikit-learn")

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
except ImportError:
    sys.exit("[ERROR] pytorch_forecasting is required: pip install pytorch-forecasting")

# ============================================================================
# CONSTANTS
# ============================================================================

ROOT          = Path(__file__).resolve().parent.parent.parent
ARTIFACT_BASE = ROOT / "src" / "agritwin_gh" / "models" / "artifacts"
MODEL_DIR     = ROOT / "src" / "agritwin_gh" / "models"

STAGE_ORDER  = ["seedling", "early_vegetative", "flowering_initiation",
                "flowering", "unripe", "ripe"]
STAGE_LABELS = ["Seedling", "Early Vegetative", "Flowering Init.",
                "Flowering", "Unripe", "Ripe"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}
N_STAGES     = 6

# Stage approximate durations used for hours-to-next-stage extrapolation
STAGE_DURATION_H: dict[str, int] = {
    "seedling"             : 168,
    "early_vegetative"     : 336,
    "flowering_initiation" : 120,
    "flowering"            : 240,
    "unripe"               : 480,
    "ripe"                 : 240,
}

TEST_CYCLES = [7, 12, 16]

ENC_LEN    = 72
PRED_LEN   = 48
H24        = 23
H48        = 47
QUANTILES  = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MEDIAN_IDX = QUANTILES.index(0.5)

# ============================================================================
# PART A — ARTIFACT LOADING
# ============================================================================

def _require(path: Path, label: str) -> Path:
    if not path.exists():
        sys.exit(f"\n[ERROR] Missing required file: {label}\n  Expected at: {path}")
    return path


def locate_artifact_dir(run_id: Optional[str]) -> Path:
    candidates = [
        d for d in ARTIFACT_BASE.iterdir()
        if d.is_dir() and d.name.startswith("growth_progression_")
    ]
    if not candidates:
        sys.exit(f"[ERROR] No growth_progression artifact directories in {ARTIFACT_BASE}")
    if run_id:
        matched = [d for d in candidates if run_id in d.name]
        if not matched:
            sys.exit(f"[ERROR] No artifact directory matching run_id '{run_id}'")
        return matched[0]
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


def load_artifacts(art_dir: Path) -> dict:
    """Load all model artifacts. Exits with a clear error message on any failure."""
    print(f"\n[A] Loading artifacts from: {art_dir.name}")

    feature_roles = json.loads(_require(art_dir / "feature_roles.json",      "feature_roles").read_text())
    seq_meta      = json.loads(_require(art_dir / "sequence_metadata.json", "sequence_metadata").read_text())
    tgt_def       = json.loads(_require(art_dir / "target_definition.json", "target_definition").read_text())
    trn_cfg       = json.loads(_require(art_dir / "training_config.json",   "training_config").read_text())
    split_sum     = json.loads(_require(art_dir / "split_summary.json",     "split_summary").read_text())

    run_id     = trn_cfg["run_id"]
    enc_len    = int(trn_cfg.get("encoder_length",    ENC_LEN))
    pred_len   = int(trn_cfg.get("prediction_length", PRED_LEN))
    quantiles  = trn_cfg.get("quantiles", QUANTILES)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else MEDIAN_IDX
    h24        = min(23, pred_len - 1)
    h48        = min(47, pred_len - 1)

    with open(_require(art_dir / "robust_scaler.pkl",        "scaler"), "rb") as fh:
        scaler = pickle.load(fh)
    with open(_require(art_dir / "tft_training_dataset.pkl", "training_dataset"), "rb") as fh:
        _saved = pickle.load(fh)
    training_dataset = _saved["training"]

    scaled_features: list[str] = (
        seq_meta.get("scaled_features") or
        feature_roles.get("time_varying_known_reals", []) +
        feature_roles.get("time_varying_unknown_reals", [])
    )

    ckpt_dir  = art_dir / "checkpoints"
    ckpt_arts = sorted(ckpt_dir.glob("*.ckpt")) if ckpt_dir.exists() else []
    ckpt_root = MODEL_DIR / f"growth_progression_{run_id}.ckpt"

    if ckpt_root.exists():
        print(f"  Loading checkpoint : {ckpt_root.name}")
        tft_model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_root))
    elif ckpt_arts:
        print(f"  Loading checkpoint : {ckpt_arts[-1].name}")
        tft_model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_arts[-1]))
    else:
        sys.exit("[ERROR] No model checkpoint found in artifact dir or model dir.")

    tft_model.eval()
    print(f"  Run ID            : {run_id}")
    print(f"  Encoder / pred    : {enc_len} h / {pred_len} h")
    print(f"  Scaler            : {type(scaler).__name__}  ({len(scaler.feature_names_in_)} features)")
    print(f"  Model             : {type(tft_model).__name__}")
    print(f"  Test cycles       : {split_sum.get('test', {}).get('cycles', TEST_CYCLES)}")

    return dict(
        feature_roles=feature_roles, seq_meta=seq_meta, tgt_def=tgt_def,
        trn_cfg=trn_cfg, split_sum=split_sum,
        scaled_features=scaled_features, scaler=scaler,
        training_dataset=training_dataset, tft_model=tft_model,
        art_dir=art_dir, run_id=run_id,
        enc_len=enc_len, pred_len=pred_len,
        quantiles=quantiles, median_idx=median_idx,
        h24=h24, h48=h48,
    )


# ============================================================================
# PART B — DATASET LOADING AND WINDOW EXTRACTION
# ============================================================================

def load_dataset(art_dir: Path) -> pd.DataFrame:
    """Load sequence-ready dataset. Prefers parquet; falls back to CSV."""
    seq_csv  = art_dir / "sequence_ready_dataset.csv"
    exp_parq = art_dir / "expanded_hourly_growth_dataset.parquet"

    if seq_csv.exists():
        print(f"\n[B] Dataset : {seq_csv.name}  (CSV)")
        df = pd.read_csv(seq_csv, parse_dates=["timestamp"], low_memory=False)
    elif exp_parq.exists():
        print(f"\n[B] Dataset : {exp_parq.name}  (parquet)")
        df = pd.read_parquet(exp_parq)
    else:
        sys.exit(f"[ERROR] No dataset found in {art_dir}")

    if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    req_target = "target_stage_index_24h"
    if req_target not in df.columns:
        sys.exit(f"[ERROR] Column '{req_target}' not found in dataset. Need sequence_ready_dataset.csv.")

    print(f"  Rows      : {len(df):,}   cols={len(df.columns)}")
    print(f"  Cycles    : {sorted(df['cycle_id'].unique().tolist())}")
    return df


def extract_windows(
    df: pd.DataFrame,
    enc_len: int,
    cycles: list[int],
    max_per_cycle: int = 50,
) -> list[dict]:
    """
    Extract evenly-spaced encoder windows from specified cycles.
    Each window: enc_len encoder rows + pre-computed target cols from last row.
    Windows that lack valid 48h ground-truth targets are skipped.
    """
    windows = []
    for cycle_id in cycles:
        cyc = (df[df["cycle_id"] == cycle_id]
               .sort_values("time_idx")
               .reset_index(drop=True))

        # All rows in sequence_ready_dataset have non-NaN targets, but guard anyway
        cyc = cyc.dropna(subset=["target_stage_index_24h", "target_stage_index_48h"])

        if len(cyc) < enc_len + 1:
            print(f"  [WARN] Cycle {cycle_id} too short ({len(cyc)} rows). Skipping.")
            continue

        max_start = len(cyc) - enc_len
        step      = max(1, max_start // max_per_cycle)
        starts    = list(range(0, max_start, step))[:max_per_cycle]

        for start in starts:
            enc_rows = cyc.iloc[start: start + enc_len].reset_index(drop=True)
            gt_row   = cyc.iloc[start + enc_len - 1]

            windows.append({
                "cycle_id" : int(cycle_id),
                "start_idx": int(start),
                "enc_rows" : enc_rows,
                "gt"       : {
                    "stage_now"          : int(gt_row["stage_index"]),
                    "stage_now_name"     : str(gt_row.get("stage_name", "")),
                    "stage_24h"          : int(round(float(gt_row["target_stage_index_24h"]))),
                    "stage_48h"          : int(round(float(gt_row["target_stage_index_48h"]))),
                    "progress_24h"       : float(gt_row["target_stage_progress_24h"]),
                    "progress_48h"       : float(gt_row["target_stage_progress_48h"]),
                    "hours_to_next"      : float(gt_row["target_hours_to_next_stage"]),
                    "stage_progress_now" : float(gt_row["stage_progress_pct"]),
                    "indoor_temp"        : float(gt_row["indoor_temp"]),
                    "indoor_humidity"    : float(gt_row["indoor_humidity"]),
                    "solarradiation"     : float(gt_row["solarradiation"]),
                    "vpd"                : float(gt_row["vpd"]),
                    "cycle_origin_type"  : str(gt_row.get("cycle_origin_type", "generated")),
                    "season_label"       : str(gt_row.get("season_label", "summer")),
                    "timestamp"          : str(gt_row.get("timestamp", "")),
                },
            })

    print(f"  Windows   : {len(windows)} total from cycles {cycles}")
    return windows


# ============================================================================
# INFERENCE ENGINE
# ============================================================================

def _build_window_df(enc_rows: pd.DataFrame) -> pd.DataFrame:
    """
    Format a real-data 72-row window for TFT inference.
    Re-indexes time_idx from 0; uses dummy cycle_id '999'.
    """
    df = enc_rows.copy()
    df["time_idx"]       = np.arange(len(df), dtype=int)
    df["cycle_id"]       = "999"
    df["cycle_origin_type"] = df["cycle_origin_type"].fillna("original").astype(str)
    df["season_label"]   = df["season_label"].fillna("summer").astype(str)
    df["target_stage_index_24h"] = 0.0   # dummy — not used during inference
    return df


def _append_decoder_rows(enc_df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
    """Append pred_len decoder rows by propagating and updating the last encoder row."""
    last    = enc_df.iloc[-1].to_dict()
    last_ti = int(last["time_idx"])
    last_hr = int(last.get("hour", 12))
    last_doy = int(last.get("day_of_year", 180))

    dec_rows = []
    for k in range(1, pred_len + 1):
        r = dict(last)
        fh = (last_hr + k) % 24
        ed = (last_hr + k) // 24
        r["time_idx"]          = last_ti + k
        r["hour"]              = float(fh)
        r["day_of_year"]       = float(min(last_doy + ed, 366))
        r["day_night_flag"]    = float(int(6 <= fh < 20))
        r["light_period_flag"] = float(int(8 <= fh < 18))
        r["target_stage_index_24h"] = 0.0
        dec_rows.append(r)

    full = pd.concat([enc_df, pd.DataFrame(dec_rows)], ignore_index=True)
    full["time_idx"] = full["time_idx"].astype(int)
    return full


def _scale_df(df: pd.DataFrame, scaler, scaled_features: list[str]) -> pd.DataFrame:
    cols = [c for c in scaled_features if c in df.columns]
    df   = df.copy()
    df[cols] = scaler.transform(df[cols])
    return df


def run_inference(enc_rows: pd.DataFrame, arts: dict) -> Optional[dict]:
    """
    Run a single TFT forward pass on a real-data 72-row window.
    Returns a prediction dict or None on failure.
    """
    try:
        win_df  = _build_window_df(enc_rows)
        full_df = _append_decoder_rows(win_df, arts["pred_len"])
        full_df = _scale_df(full_df, arts["scaler"], arts["scaled_features"])

        infer_ds = TimeSeriesDataSet.from_dataset(
            arts["training_dataset"], full_df, predict=True, stop_randomization=True
        )
        infer_dl = infer_ds.to_dataloader(train=False, batch_size=1, num_workers=0)

        arts["tft_model"].eval()
        with torch.no_grad():
            x, _y   = next(iter(infer_dl))
            out     = arts["tft_model"](x)
            pred_dn = arts["tft_model"].transform_output(out.prediction, x["target_scale"])

        preds = pred_dn[0].cpu().numpy()   # (pred_len, n_quantiles)
        q50   = preds[:, arts["median_idx"]]

        p24 = float(q50[arts["h24"]])
        p48 = float(q50[arts["h48"]])
        q24 = preds[arts["h24"]]
        q48 = preds[arts["h48"]]

        si24 = int(np.clip(round(p24), 0, 5))
        si48 = int(np.clip(round(p48), 0, 5))

        prog24 = float(np.clip((p24 - math.floor(p24)) * 100.0, 0.0, 100.0))
        prog48 = float(np.clip((p48 - math.floor(p48)) * 100.0, 0.0, 100.0))

        # Hours-to-next: scan q50 decoder steps for a stage transition; extrapolate otherwise
        cur_si   = int(enc_rows.iloc[-1].get("stage_index", 0))
        cur_base = int(math.floor(q50[0]))
        transition_step = None
        for step, val in enumerate(q50):
            if int(math.floor(val)) > cur_base:
                transition_step = step
                break
        if transition_step is not None:
            hours_nxt = float(transition_step + 1)
        else:
            frac_now  = float(q50[0] - cur_base)
            stage_key = STAGE_ORDER[cur_si] if 0 <= cur_si < N_STAGES else "early_vegetative"
            hours_nxt = round((1.0 - frac_now) * STAGE_DURATION_H.get(stage_key, 240), 1)

        def _probs(q_arr: np.ndarray) -> list[float]:
            mid   = arts["median_idx"]
            mu    = q_arr[mid]
            sigma = max((q_arr[-1] - q_arr[0]) / 4.0, 0.3)
            raw   = np.exp(-0.5 * ((np.arange(float(N_STAGES)) - mu) / sigma) ** 2)
            return (raw / raw.sum()).tolist()

        has_nan = bool(np.any(np.isnan(preds)))
        has_inf = bool(np.any(np.isinf(preds)))

        return {
            "stage_index_24h" : si24,
            "stage_index_48h" : si48,
            "stage_cont_24h"  : round(p24, 4),
            "stage_cont_48h"  : round(p48, 4),
            "progress_24h"    : round(prog24, 2),
            "progress_48h"    : round(prog48, 2),
            "hours_to_next"   : round(hours_nxt, 1),
            "class_probs_24h" : _probs(q24),
            "class_probs_48h" : _probs(q48),
            "iqr_24h"         : [round(float(q24[2]), 3), round(float(q24[4]), 3)],
            "iqr_48h"         : [round(float(q48[2]), 3), round(float(q48[4]), 3)],
            "has_nan"         : has_nan,
            "has_inf"         : has_inf,
        }

    except Exception as exc:
        return None


def run_all_inference(
    windows: list[dict], arts: dict, tag: str = "test", verbose: bool = True
) -> list[dict]:
    """Batch-run inference on all windows. Attaches 'pred' key (or None on failure)."""
    results  = []
    n_ok     = 0
    n_err    = 0
    t_start  = time.time()
    for i, win in enumerate(windows):
        pred = run_inference(win["enc_rows"], arts)
        rec  = {k: v for k, v in win.items() if k != "enc_rows"}
        rec["enc_rows"] = win["enc_rows"]   # keep for S4 radiation check
        rec["pred"]     = pred
        results.append(rec)
        if pred is None:
            n_err += 1
        else:
            n_ok += 1
        if verbose and (i + 1) % 20 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            print(f"  [{tag}] {i+1}/{len(windows)}  ok={n_ok}  err={n_err}"
                  f"  ({rate:.1f} win/s)", end="\r", flush=True)

    elapsed = time.time() - t_start
    print(f"  [{tag}] Done: {n_ok} ok, {n_err} failed — {elapsed:.1f}s            ")
    return results


# ============================================================================
# PART C — FUNCTIONAL TESTS
# ============================================================================

def functional_tests(arts: dict, windows: list[dict]) -> dict:
    """Ten structural/functional correctness checks."""
    print("\n[C] Functional tests ...")
    res      = {}
    failures = []

    # C1 — Model type
    res["C1_model_type_correct"] = isinstance(arts["tft_model"], TemporalFusionTransformer)
    if not res["C1_model_type_correct"]:
        failures.append("C1: model is not TemporalFusionTransformer")

    # C2 — Scaler feature count consistency
    n_scaler  = len(arts["scaler"].feature_names_in_)
    n_listed  = len(arts["scaled_features"])
    res["C2_scaler_feature_count"] = n_scaler
    res["C2_scaler_matches_config"] = (n_scaler == n_listed)
    if n_scaler != n_listed:
        failures.append(f"C2: scaler has {n_scaler} features but config lists {n_listed}")

    # C3 — Sample window available
    sample = windows[0] if windows else None
    res["C3_sample_window_available"] = sample is not None
    if not sample:
        failures.append("C3: No sample windows available")
        _summarise_c(res, failures)
        return res

    # Checks C4–C10 require a real inference call on the sample window
    pred = run_inference(sample["enc_rows"], arts)
    res["C4_inference_executes"]   = pred is not None
    if pred is None:
        failures.append("C4: Inference crashed on sample window")
        _summarise_c(res, failures)
        return res

    required_keys = ["stage_index_24h", "stage_index_48h", "progress_24h",
                     "progress_48h", "hours_to_next", "class_probs_24h"]
    missing       = [k for k in required_keys if k not in pred]
    res["C5_output_keys_present"] = len(missing) == 0
    if missing:
        failures.append(f"C5: Missing output keys: {missing}")

    res["C6_no_nan_in_output"] = not pred.get("has_nan", True)
    res["C6_no_inf_in_output"] = not pred.get("has_inf", True)
    if pred.get("has_nan"):
        failures.append("C6: NaN values found in raw predictions")
    if pred.get("has_inf"):
        failures.append("C6: Inf values found in raw predictions")

    si24_ok = 0 <= pred["stage_index_24h"] <= 5
    si48_ok = 0 <= pred["stage_index_48h"] <= 5
    res["C7_stage_indices_valid"] = si24_ok and si48_ok
    if not (si24_ok and si48_ok):
        failures.append(f"C7: Stage indices out of range: {pred['stage_index_24h']}, {pred['stage_index_48h']}")

    p24_ok = 0.0 <= pred["progress_24h"] <= 100.0
    p48_ok = 0.0 <= pred["progress_48h"] <= 100.0
    res["C8_progress_in_valid_range"] = p24_ok and p48_ok
    if not (p24_ok and p48_ok):
        failures.append(f"C8: Progress out of [0,100]: {pred['progress_24h']}, {pred['progress_48h']}")

    res["C9_hours_to_next_nonneg"] = pred["hours_to_next"] >= 0.0
    if pred["hours_to_next"] < 0:
        failures.append(f"C9: Negative hours_to_next: {pred['hours_to_next']}")

    s24 = sum(pred.get("class_probs_24h", [0]))
    s48 = sum(pred.get("class_probs_48h", [0]))
    res["C10_class_probs_sum_to_1"] = abs(s24 - 1.0) < 0.01 and abs(s48 - 1.0) < 0.01
    if not res["C10_class_probs_sum_to_1"]:
        failures.append(f"C10: Class probs don't sum to 1: +24h={s24:.3f}, +48h={s48:.3f}")

    _summarise_c(res, failures)
    return res


def _summarise_c(res: dict, failures: list[str]) -> None:
    n_bool   = sum(1 for v in res.values() if isinstance(v, bool))
    n_passed = sum(1 for v in res.values() if v is True)
    res["checks_passed"] = n_passed
    res["total_checks"]  = n_bool
    res["failures"]      = failures
    res["all_passed"]    = len(failures) == 0
    status = "PASS" if res["all_passed"] else f"PARTIAL ({len(failures)} failed)"
    print(f"  Result : {status}  ({n_passed}/{n_bool} checks)")
    for f in failures:
        print(f"  FAIL   : {f}")


# ============================================================================
# PART D — SCENARIO DEFINITIONS AND ASSIGNMENT
# ============================================================================

SCENARIOS: dict[str, dict] = {
    "S1_stable_early_growth": {
        "desc"   : "Stable early growth — seedling/early_veg, <60% stage progress",
        "filter" : lambda gt, enc: gt["stage_now"] in (0, 1) and gt["stage_progress_now"] < 60,
    },
    "S2_approaching_transition": {
        "desc"   : "Approaching stage transition — >85% stage progress remaining",
        "filter" : lambda gt, enc: gt["stage_progress_now"] > 85,
    },
    "S3_mid_stage_development": {
        "desc"   : "Mid-stage steady development — 30–70% stage progress",
        "filter" : lambda gt, enc: 30 < gt["stage_progress_now"] < 70,
    },
    "S4_diurnal_variation": {
        "desc"   : "Clear day/night cycle — radiation range > 100 W/m² in window",
        "filter" : lambda gt, enc: float(enc["solarradiation"].max() - enc["solarradiation"].min()) > 100,
    },
    "S5_high_humidity": {
        "desc"   : "High humidity / lower ventilation — humidity > 82%",
        "filter" : lambda gt, enc: gt["indoor_humidity"] > 82,
    },
    "S6_high_growth_conditions": {
        "desc"   : "High-growth conditions — T > 24°C and radiation > 150 W/m²",
        "filter" : lambda gt, enc: gt["indoor_temp"] > 24 and gt["solarradiation"] > 150,
    },
    "S7_late_fruit_development": {
        "desc"   : "Late fruit development — unripe/ripe stage, >75% progress",
        "filter" : lambda gt, enc: gt["stage_now"] in (4, 5) and gt["stage_progress_now"] > 75,
    },
    "S8_original_cycles": {
        "desc"   : "Original (non-synthetic) crop cycle data",
        "filter" : lambda gt, enc: gt["cycle_origin_type"] == "original",
    },
    "S8_generated_cycles": {
        "desc"   : "Augmentation-generated cycle data",
        "filter" : lambda gt, enc: gt["cycle_origin_type"] == "generated",
    },
}


def assign_scenarios(results: list[dict]) -> dict[str, list[dict]]:
    """Assign each valid result into matching scenario buckets."""
    buckets = {k: [] for k in SCENARIOS}
    for rec in results:
        if rec["pred"] is None:
            continue
        gt  = rec["gt"]
        enc = rec["enc_rows"]
        for sc_key, sc_def in SCENARIOS.items():
            try:
                if sc_def["filter"](gt, enc):
                    buckets[sc_key].append(rec)
            except Exception:
                pass
    return buckets


def _quick_metrics(bucket: list[dict]) -> dict:
    if not bucket:
        return {"n": 0}
    y_si24_t  = [r["gt"]["stage_24h"]    for r in bucket]
    y_si24_p  = [r["pred"]["stage_index_24h"] for r in bucket]
    y_pr24_t  = [r["gt"]["progress_24h"] for r in bucket]
    y_pr24_p  = [r["pred"]["progress_24h"] for r in bucket]
    y_h2n_t   = [r["gt"]["hours_to_next"] for r in bucket]
    y_h2n_p   = [r["pred"]["hours_to_next"] for r in bucket]
    return {
        "n"              : len(bucket),
        "stage_acc_24h"  : round(accuracy_score(y_si24_t, y_si24_p), 4),
        "ord_dist_24h"   : round(float(np.mean(np.abs(np.array(y_si24_t) - np.array(y_si24_p)))), 4),
        "progress_mae_24h": round(mean_absolute_error(y_pr24_t, y_pr24_p), 2),
        "h2n_mae"        : round(mean_absolute_error(y_h2n_t, y_h2n_p), 2),
    }


def compute_scenario_metrics(buckets: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for sc_key, bucket in buckets.items():
        m = _quick_metrics(bucket)
        rows.append({
            "scenario"       : sc_key,
            "description"    : SCENARIOS.get(sc_key, {}).get("desc", ""),
            **m,
        })
    return pd.DataFrame(rows)


# ============================================================================
# PART E — BIOLOGICAL CONSISTENCY TESTS
# ============================================================================

def biological_consistency_tests(results: list[dict]) -> dict:
    """Check biological plausibility of all predictions."""
    print("\n[E] Biological consistency checks ...")
    valid = [r for r in results if r["pred"] is not None]
    n     = len(valid)
    if n == 0:
        return {"error": "No valid predictions to check"}

    back_24:    list[dict] = []
    back_48:    list[dict] = []
    skip_24:    list[dict] = []
    skip_48:    list[dict] = []
    prog_back:  list[dict] = []
    h2n_incon:  list[dict] = []
    h2n_neg:    list[dict] = []
    h2n_ext:    list[dict] = []

    for rec in valid:
        pred   = rec["pred"]
        gt     = rec["gt"]
        si_now = int(gt["stage_now"])
        si24   = int(pred["stage_index_24h"])
        si48   = int(pred["stage_index_48h"])
        h2n    = float(pred["hours_to_next"])
        pr24   = float(pred["progress_24h"])
        pr48   = float(pred["progress_48h"])
        base   = {"cycle": rec["cycle_id"], "stage_now": si_now}

        # E1 — No backward stage jump
        if si24 < si_now:
            back_24.append({**base, "si24": si24})
        if si48 < si_now:
            back_48.append({**base, "si48": si48})

        # E2 — No impossibly large jump in a short horizon
        if si24 > si_now + 1:
            skip_24.append({**base, "si24": si24, "skip": si24 - si_now})
        if si48 > si_now + 2:
            skip_48.append({**base, "si48": si48, "skip": si48 - si_now})

        # E3 — Progress shouldn't go backwards within the same stage
        if si24 == si48 and pr48 < pr24 - 5.0:
            prog_back.append({**base, "pr24": pr24, "pr48": pr48})

        # E4 — Transition consistency: if +24h=same and +48h=next, h2n should be ≤ 48h
        if si24 == si_now and si48 == si_now + 1 and h2n > 48:
            h2n_incon.append({**base, "si48": si48, "h2n": h2n})

        # E5 — h2n sanity bounds
        if h2n < 0:
            h2n_neg.append({**base, "h2n": h2n})
        if h2n > 2000:
            h2n_ext.append({**base, "h2n": h2n})

    total = (len(back_24) + len(back_48) + len(skip_24) + len(skip_48) +
             len(prog_back) + len(h2n_incon) + len(h2n_neg) + len(h2n_ext))
    pass_rate = round(100 * (1 - total / max(n, 1)), 2)

    summary = {
        "n_evaluated"             : n,
        "E1_backward_jumps_24h"   : len(back_24),
        "E1_backward_jumps_48h"   : len(back_48),
        "E2_multi_stage_skip_24h" : len(skip_24),
        "E2_multi_stage_skip_48h" : len(skip_48),
        "E3_progress_backward"    : len(prog_back),
        "E4_h2n_inconsistent"     : len(h2n_incon),
        "E5_h2n_negative"         : len(h2n_neg),
        "E5_h2n_extreme"          : len(h2n_ext),
        "total_violations"        : total,
        "biological_pass_rate_pct": pass_rate,
        "violation_examples"      : {
            "backward_jumps_24h"   : back_24[:5],
            "multi_stage_skip_24h" : skip_24[:5],
            "progress_backward"    : prog_back[:5],
            "h2n_inconsistent"     : h2n_incon[:5],
        },
    }
    print(f"  Pass rate : {pass_rate}%  ({total} violations in {n} samples)")
    for tag, lst in [("E1 backward 24h", back_24), ("E2 skip 24h", skip_24),
                     ("E3 prog backward", prog_back), ("E4 h2n inconsist.", h2n_incon),
                     ("E5 h2n extreme", h2n_ext)]:
        if lst:
            print(f"    {tag}: {len(lst)}")
    return summary


# ============================================================================
# PART F — ACCURACY AND REGRESSION METRICS
# ============================================================================

def _ordinal_distance(y_true: list[int], y_pred: list[int]) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))


def _transition_acc(y_true: list[int], y_pred: list[int], si_now: list[int]) -> float:
    """Whether transition (predicted stage > current) was correctly detected."""
    gt   = [int(t > n) for t, n in zip(y_true, si_now)]
    pred = [int(p > n) for p, n in zip(y_pred, si_now)]
    return float(accuracy_score(gt, pred)) if gt else 0.0


def compute_stage_metrics(
    y_true_si: list[int], y_pred_si: list[int],
    si_now: list[int], horizon: str
) -> dict:
    labels   = list(range(N_STAGES))
    present  = sorted(set(y_true_si) | set(y_pred_si))
    cm       = confusion_matrix(y_true_si, y_pred_si, labels=labels)
    return {
        "horizon"               : horizon,
        "n_samples"             : len(y_true_si),
        "accuracy"              : round(accuracy_score(y_true_si, y_pred_si), 4),
        "balanced_accuracy"     : round(balanced_accuracy_score(y_true_si, y_pred_si), 4),
        "precision_macro"       : round(precision_score(y_true_si, y_pred_si, average="macro",    zero_division=0, labels=present), 4),
        "recall_macro"          : round(recall_score(   y_true_si, y_pred_si, average="macro",    zero_division=0, labels=present), 4),
        "f1_macro"              : round(f1_score(       y_true_si, y_pred_si, average="macro",    zero_division=0, labels=present), 4),
        "f1_weighted"           : round(f1_score(       y_true_si, y_pred_si, average="weighted", zero_division=0), 4),
        "ordinal_distance"      : round(_ordinal_distance(y_true_si, y_pred_si), 4),
        "transition_detect_acc" : round(_transition_acc(y_true_si, y_pred_si, si_now), 4),
        "confusion_matrix"      : cm.tolist(),
    }


def compute_regression_metrics(
    y_true: list[float], y_pred: list[float], name: str
) -> dict:
    t = np.array(y_true, dtype=float)
    p = np.array(y_pred, dtype=float)
    abs_err = np.abs(t - p)
    try:
        r2 = float(r2_score(t, p))
        ev = float(explained_variance_score(t, p))
    except Exception:
        r2, ev = float("nan"), float("nan")
    safe_mape = float(np.nanmean(np.where(t != 0, np.abs((t - p) / t), np.nan))) * 100
    return {
        "metric"            : name,
        "n_samples"         : len(y_true),
        "mae"               : round(float(np.mean(abs_err)), 4),
        "rmse"              : round(float(np.sqrt(np.mean((t - p) ** 2))), 4),
        "r2"                : round(r2, 4) if not math.isnan(r2) else None,
        "explained_var"     : round(ev, 4) if not math.isnan(ev) else None,
        "median_abs_error"  : round(float(np.median(abs_err)), 4),
        "mape_pct"          : round(safe_mape, 2) if not math.isnan(safe_mape) else None,
        "mean_bias"         : round(float(np.mean(p - t)), 4),
    }


def compute_h2n_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    t       = np.array(y_true, dtype=float)
    p       = np.array(y_pred, dtype=float)
    abs_err = np.abs(t - p)
    return {
        "n_samples"        : len(y_true),
        "mae"              : round(float(np.mean(abs_err)), 2),
        "rmse"             : round(float(np.sqrt(np.mean((t - p) ** 2))), 2),
        "median_abs_error" : round(float(np.median(abs_err)), 2),
        "pct_within_6h"    : round(float(np.mean(abs_err <= 6))  * 100, 1),
        "pct_within_12h"   : round(float(np.mean(abs_err <= 12)) * 100, 1),
        "pct_within_24h"   : round(float(np.mean(abs_err <= 24)) * 100, 1),
        "mean_bias_h"      : round(float(np.mean(p - t)), 2),
    }


def compute_all_metrics(results: list[dict]) -> dict:
    """Compute the full metric suite over all valid inference results."""
    valid = [r for r in results if r["pred"] is not None]
    n     = len(valid)
    print(f"\n[F] Computing metrics over {n} valid windows ...")
    if n == 0:
        return {"error": "No valid inference results"}

    si_now   = [r["gt"]["stage_now"]     for r in valid]
    si24_t   = [r["gt"]["stage_24h"]     for r in valid]
    si48_t   = [r["gt"]["stage_48h"]     for r in valid]
    prog24_t = [r["gt"]["progress_24h"]  for r in valid]
    prog48_t = [r["gt"]["progress_48h"]  for r in valid]
    h2n_t    = [r["gt"]["hours_to_next"] for r in valid]

    si24_p   = [r["pred"]["stage_index_24h"] for r in valid]
    si48_p   = [r["pred"]["stage_index_48h"] for r in valid]
    prog24_p = [r["pred"]["progress_24h"]    for r in valid]
    prog48_p = [r["pred"]["progress_48h"]    for r in valid]
    h2n_p    = [r["pred"]["hours_to_next"]   for r in valid]

    return {
        "n_valid"      : n,
        "stage_24h"    : compute_stage_metrics(si24_t,   si24_p,   si_now, "24h"),
        "stage_48h"    : compute_stage_metrics(si48_t,   si48_p,   si_now, "48h"),
        "progress_24h" : compute_regression_metrics(prog24_t, prog24_p, "stage_progress_24h"),
        "progress_48h" : compute_regression_metrics(prog48_t, prog48_p, "stage_progress_48h"),
        "hours_to_next": compute_h2n_metrics(h2n_t, h2n_p),
    }


def compute_per_stage_metrics(results: list[dict]) -> pd.DataFrame:
    valid = [r for r in results if r["pred"] is not None]
    rows  = []
    for si, stage in enumerate(STAGE_ORDER):
        sub = [r for r in valid if r["gt"]["stage_now"] == si]
        if not sub:
            rows.append({"stage": stage, "n": 0}); continue
        si24_t  = [r["gt"]["stage_24h"]    for r in sub]
        si24_p  = [r["pred"]["stage_index_24h"] for r in sub]
        pr24_t  = [r["gt"]["progress_24h"] for r in sub]
        pr24_p  = [r["pred"]["progress_24h"] for r in sub]
        h2n_t   = [r["gt"]["hours_to_next"] for r in sub]
        h2n_p   = [r["pred"]["hours_to_next"] for r in sub]
        rows.append({
            "stage"         : stage,
            "n"             : len(sub),
            "acc_24h"       : round(accuracy_score(si24_t, si24_p), 4),
            "f1_weighted_24h": round(f1_score(si24_t, si24_p, average="weighted", zero_division=0), 4),
            "ord_dist_24h"  : round(_ordinal_distance(si24_t, si24_p), 4),
            "progress_mae_24h": round(mean_absolute_error(pr24_t, pr24_p), 2),
            "h2n_mae"       : round(mean_absolute_error(h2n_t, h2n_p), 2),
        })
    return pd.DataFrame(rows)


def compute_per_cycle_metrics(results: list[dict]) -> pd.DataFrame:
    valid  = [r for r in results if r["pred"] is not None]
    cycles = sorted(set(r["cycle_id"] for r in valid))
    rows   = []
    for cyc in cycles:
        sub = [r for r in valid if r["cycle_id"] == cyc]
        si24_t  = [r["gt"]["stage_24h"]    for r in sub]
        si24_p  = [r["pred"]["stage_index_24h"] for r in sub]
        pr24_t  = [r["gt"]["progress_24h"] for r in sub]
        pr24_p  = [r["pred"]["progress_24h"] for r in sub]
        h2n_t   = [r["gt"]["hours_to_next"] for r in sub]
        h2n_p   = [r["pred"]["hours_to_next"] for r in sub]
        origins = set(r["gt"]["cycle_origin_type"] for r in sub)
        rows.append({
            "cycle_id"      : cyc,
            "origin_type"   : "/".join(sorted(origins)),
            "n"             : len(sub),
            "acc_24h"       : round(accuracy_score(si24_t, si24_p), 4),
            "f1_weighted_24h": round(f1_score(si24_t, si24_p, average="weighted", zero_division=0), 4),
            "ord_dist_24h"  : round(_ordinal_distance(si24_t, si24_p), 4),
            "progress_mae_24h": round(mean_absolute_error(pr24_t, pr24_p), 2),
            "h2n_mae"       : round(mean_absolute_error(h2n_t, h2n_p), 2),
        })
    return pd.DataFrame(rows)


# ============================================================================
# PART G — FORECAST BEHAVIOR AND TRANSITION ANALYSIS
# ============================================================================

def transition_analysis(results: list[dict]) -> pd.DataFrame:
    """Detailed analysis of windows where a real stage transition is expected within 48h."""
    valid = [r for r in results if r["pred"] is not None]
    rows  = []
    for rec in valid:
        gt    = rec["gt"]
        pred  = rec["pred"]
        si_n  = int(gt["stage_now"])
        si24t = int(gt["stage_24h"])
        si48t = int(gt["stage_48h"])
        si24p = int(pred["stage_index_24h"])
        si48p = int(pred["stage_index_48h"])

        trans_24h = si24t > si_n
        trans_48h = si48t > si_n and not trans_24h
        if not (trans_24h or trans_48h):
            continue

        rows.append({
            "cycle_id"           : rec["cycle_id"],
            "timestamp"          : gt["timestamp"],
            "stage_now"          : STAGE_ORDER[si_n] if 0 <= si_n < N_STAGES else "?",
            "actual_trans_24h"   : trans_24h,
            "actual_trans_48h"   : trans_48h,
            "pred_trans_24h"     : si24p > si_n,
            "pred_trans_48h"     : si48p > si_n,
            "true_hours_to_next" : round(float(gt["hours_to_next"]), 1),
            "pred_hours_to_next" : round(float(pred["hours_to_next"]), 1),
            "h2n_error_h"        : round(float(pred["hours_to_next"]) - float(gt["hours_to_next"]), 1),
            "correct_24h"        : (si24p > si_n) == trans_24h,
        })

    if rows:
        df       = pd.DataFrame(rows)
        df_trans = df[df["actual_trans_24h"]]
        if len(df_trans) > 0:
            detect_rate = df_trans["correct_24h"].mean()
            early_bias  = float(df_trans["h2n_error_h"].mean())
            print(f"  Transitions within 24h : {len(df_trans)}  "
                  f"detect_rate={detect_rate:.2%}  "
                  f"avg_timing_bias={early_bias:+.1f}h")
        return df

    return pd.DataFrame(columns=["cycle_id", "timestamp", "stage_now",
                                  "actual_trans_24h", "actual_trans_48h",
                                  "pred_trans_24h", "pred_trans_48h",
                                  "true_hours_to_next", "pred_hours_to_next",
                                  "h2n_error_h", "correct_24h"])


def build_prediction_samples(results: list[dict], n: int = 200) -> pd.DataFrame:
    """Flat CSV of actual vs predicted values for a sample of results."""
    valid = [r for r in results if r["pred"] is not None]
    step  = max(1, len(valid) // n)
    rows  = []
    for rec in valid[::step]:
        gt   = rec["gt"]
        pred = rec["pred"]
        rows.append({
            "cycle_id"           : rec["cycle_id"],
            "timestamp"          : gt["timestamp"],
            "stage_now"          : gt["stage_now_name"],
            "stage_progress_now_pct" : round(gt["stage_progress_now"], 1),
            "actual_stage_24h"   : STAGE_ORDER[gt["stage_24h"]] if 0 <= gt["stage_24h"] < N_STAGES else "?",
            "pred_stage_24h"     : STAGE_ORDER[pred["stage_index_24h"]] if 0 <= pred["stage_index_24h"] < N_STAGES else "?",
            "actual_stage_48h"   : STAGE_ORDER[gt["stage_48h"]] if 0 <= gt["stage_48h"] < N_STAGES else "?",
            "pred_stage_48h"     : STAGE_ORDER[pred["stage_index_48h"]] if 0 <= pred["stage_index_48h"] < N_STAGES else "?",
            "actual_progress_24h": round(gt["progress_24h"], 1),
            "pred_progress_24h"  : round(pred["progress_24h"], 1),
            "actual_progress_48h": round(gt["progress_48h"], 1),
            "pred_progress_48h"  : round(pred["progress_48h"], 1),
            "actual_h2n"         : round(gt["hours_to_next"], 1),
            "pred_h2n"           : round(pred["hours_to_next"], 1),
            "stage_acc_24h"      : int(gt["stage_24h"] == pred["stage_index_24h"]),
            "stage_acc_48h"      : int(gt["stage_48h"] == pred["stage_index_48h"]),
            "indoor_temp"        : round(gt["indoor_temp"], 1),
            "indoor_humidity"    : round(gt["indoor_humidity"], 1),
            "solarradiation"     : round(gt["solarradiation"], 1),
        })
    return pd.DataFrame(rows)


# ============================================================================
# PART H — SAVE OUTPUTS AND PLOTS
# ============================================================================

def _plot_confusion_matrix(cm_data: list[list], title: str, path: Path) -> None:
    cm  = np.array(cm_data, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_STAGES)); ax.set_xticklabels(STAGE_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(N_STAGES)); ax.set_yticklabels(STAGE_LABELS, fontsize=8)
    max_val = cm.max() if cm.max() > 0 else 1
    for i in range(N_STAGES):
        for j in range(N_STAGES):
            color = "white" if cm[i, j] > max_val * 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    ax.set_xlabel("Predicted stage",  fontsize=9)
    ax.set_ylabel("Actual stage",     fontsize=9)
    ax.set_title(title,               fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def _plot_error_dist(errors: list[float], title: str, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", linewidth=1.2, linestyle="--", label="Zero error")
    ax.axvline(float(np.mean(errors)), color="orange", linewidth=1.2,
               linestyle=":", label=f"Mean {np.mean(errors):.1f}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def _plot_actual_vs_pred(y_true: list, y_pred: list, title: str,
                          xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="#2196F3")
    lim = [min(min(y_true), min(y_pred)) * 0.95, max(max(y_true), max(y_pred)) * 1.05]
    ax.plot(lim, lim, "r--", linewidth=1, label="Perfect")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Actual {xlabel}", fontsize=9)
    ax.set_ylabel(f"Predicted {xlabel}", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def _plot_scenario_comparison(sc_metrics: pd.DataFrame, path: Path) -> None:
    df = sc_metrics[sc_metrics["n"] > 0].copy()
    if df.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    cols   = ["stage_acc_24h", "progress_mae_24h", "h2n_mae"]
    titles = ["Stage Accuracy +24h", "Progress MAE +24h (%)", "Hours-to-Next MAE"]
    colors = ["#4CAF50", "#FF9800", "#2196F3"]
    labels = [s.replace("S8_", "").replace("_", "\n") for s in df["scenario"]]
    for ax, col, title, color in zip(axes, cols, titles, colors):
        if col not in df.columns:
            ax.set_visible(False); continue
        ax.barh(labels, df[col], color=color, edgecolor="white")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Score", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("Scenario Comparison — TFT Growth Progression", fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()


def _plot_stage_trajectory(results: list[dict], out_dir: Path, n_examples: int = 4) -> None:
    """Plot actual vs predicted stage index trajectories for sample windows."""
    valid = [r for r in results if r["pred"] is not None]
    sample = valid[::max(1, len(valid) // n_examples)][:n_examples]
    if not sample:
        return

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4), sharey=True)
    if len(sample) == 1:
        axes = [axes]

    for ax, rec in zip(axes, sample):
        gt   = rec["gt"]
        pred = rec["pred"]
        # Encoder stage progression from enc_rows
        enc_si = rec["enc_rows"]["stage_index"].values.tolist()
        ax.plot(range(len(enc_si)), enc_si, color="gray", linewidth=1, alpha=0.6, label="Observed")
        ax.scatter([len(enc_si) + 23], [gt["stage_24h"]],      marker="o", s=50, color="green",  label="Actual +24h")
        ax.scatter([len(enc_si) + 47], [gt["stage_48h"]],      marker="o", s=50, color="darkgreen")
        ax.scatter([len(enc_si) + 23], [pred["stage_index_24h"]], marker="x", s=60, color="red",     label="Pred +24h")
        ax.scatter([len(enc_si) + 47], [pred["stage_index_48h"]], marker="x", s=60, color="darkred")
        ax.set_yticks(range(N_STAGES)); ax.set_yticklabels(STAGE_LABELS, fontsize=6)
        ax.set_title(f"Cycle {rec['cycle_id']}", fontsize=8)
        ax.set_xlabel("Hour idx", fontsize=7)
        ax.legend(fontsize=6)

    plt.suptitle("Stage Trajectory: Observed + Forecast", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "stage_prediction_examples.png", dpi=130, bbox_inches="tight")
    plt.close()


def save_all_outputs(
    out_dir: Path,
    test_cfg: dict,
    functional: dict,
    bio: dict,
    metrics: dict,
    per_stage: pd.DataFrame,
    per_cycle: pd.DataFrame,
    sc_metrics: pd.DataFrame,
    trans_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    results: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[H] Saving outputs to: {out_dir}")

    # JSON outputs
    def _jdump(obj, name: str) -> None:
        (out_dir / name).write_text(json.dumps(obj, indent=2, default=str))

    _jdump(test_cfg,   "test_config.json")
    _jdump(functional, "functional_test_results.json")
    _jdump(bio,        "logical_consistency_results.json")

    if "error" not in metrics:
        _jdump(metrics.get("stage_24h", {}),    "stage_metrics_24h.json")
        _jdump(metrics.get("stage_48h", {}),    "stage_metrics_48h.json")
        _jdump({
            "progress_24h" : metrics.get("progress_24h", {}),
            "progress_48h" : metrics.get("progress_48h", {}),
        },                                       "regression_metrics.json")
        _jdump(metrics.get("hours_to_next", {}), "hours_to_next_metrics.json")

        summary = {
            "test_id"          : test_cfg["test_id"],
            "run_id"           : test_cfg["run_id"],
            "n_windows"        : metrics["n_valid"],
            "functional_passed": functional.get("all_passed", False),
            "bio_pass_rate_pct": bio.get("biological_pass_rate_pct", 0),
            "stage_acc_24h"    : metrics["stage_24h"].get("accuracy"),
            "stage_acc_48h"    : metrics["stage_48h"].get("accuracy"),
            "f1_macro_24h"     : metrics["stage_24h"].get("f1_macro"),
            "f1_macro_48h"     : metrics["stage_48h"].get("f1_macro"),
            "ordinal_dist_24h" : metrics["stage_24h"].get("ordinal_distance"),
            "progress_mae_24h" : metrics["progress_24h"].get("mae"),
            "progress_mae_48h" : metrics["progress_48h"].get("mae"),
            "h2n_mae"          : metrics["hours_to_next"].get("mae"),
            "h2n_pct_within_12h": metrics["hours_to_next"].get("pct_within_12h"),
            "violations_total" : bio.get("total_violations", 0),
        }
        _jdump(summary, "test_summary.json")

        # Confusion matrices
        cm24 = np.array(metrics["stage_24h"]["confusion_matrix"])
        cm48 = np.array(metrics["stage_48h"]["confusion_matrix"])
        pd.DataFrame(cm24, index=STAGE_LABELS, columns=STAGE_LABELS).to_csv(
            out_dir / "confusion_matrix_24h.csv")
        pd.DataFrame(cm48, index=STAGE_LABELS, columns=STAGE_LABELS).to_csv(
            out_dir / "confusion_matrix_48h.csv")

        # Plots — confusion matrices
        _plot_confusion_matrix(cm24.tolist(), "Stage Prediction Confusion — +24h",
                               out_dir / "confusion_matrix_24h.png")
        _plot_confusion_matrix(cm48.tolist(), "Stage Prediction Confusion — +48h",
                               out_dir / "confusion_matrix_48h.png")

        # Plots — error distributions
        valid      = [r for r in results if r["pred"] is not None]
        prog_errs  = [r["pred"]["progress_24h"] - r["gt"]["progress_24h"]  for r in valid]
        h2n_errs   = [r["pred"]["hours_to_next"] - r["gt"]["hours_to_next"] for r in valid]
        prog24_t   = [r["gt"]["progress_24h"]  for r in valid]
        prog24_p   = [r["pred"]["progress_24h"] for r in valid]
        h2n_t      = [r["gt"]["hours_to_next"]  for r in valid]
        h2n_p      = [r["pred"]["hours_to_next"] for r in valid]

        if prog_errs:
            _plot_error_dist(prog_errs, "Stage Progress Error Distribution (+24h)",
                             "Predicted – Actual (%)", out_dir / "error_distribution_progress.png")
        if h2n_errs:
            _plot_error_dist(h2n_errs, "Hours-to-Next-Stage Error Distribution",
                             "Predicted – Actual (h)", out_dir / "error_distribution_time_to_next_stage.png")
        if prog24_t:
            _plot_actual_vs_pred(prog24_t, prog24_p, "Stage Progress Actual vs Predicted (+24h)",
                                 "progress (%)", out_dir / "progress_prediction_examples.png")
        if h2n_t:
            _plot_actual_vs_pred(h2n_t, h2n_p, "Hours-to-Next Actual vs Predicted",
                                 "hours", out_dir / "time_to_next_stage_examples.png")

    # Stage trajectory examples
    _plot_stage_trajectory(results, out_dir)

    # Scenario comparison plot
    if not sc_metrics.empty:
        _plot_scenario_comparison(sc_metrics, out_dir / "scenario_comparison.png")

    # CSVs
    if not per_stage.empty:
        per_stage.to_csv(out_dir / "per_stage_metrics.csv", index=False)
    if not per_cycle.empty:
        per_cycle.to_csv(out_dir / "per_cycle_metrics.csv", index=False)
    if not sc_metrics.empty:
        sc_metrics.to_csv(out_dir / "per_scenario_metrics.csv", index=False)
    if not trans_df.empty:
        trans_df.to_csv(out_dir / "transition_analysis.csv", index=False)
    if not samples_df.empty:
        samples_df.to_csv(out_dir / "actual_vs_predicted.csv", index=False)
        samples_df.head(50).to_csv(out_dir / "prediction_samples.csv", index=False)

    # Violation examples CSV
    viol = bio.get("violation_examples", {})
    viol_rows = []
    for vtype, items in viol.items():
        for item in items:
            viol_rows.append({"type": vtype, **item})
    if viol_rows:
        pd.DataFrame(viol_rows).to_csv(out_dir / "violation_examples.csv", index=False)

    # Original vs generated comparison
    valid = [r for r in results if r["pred"] is not None]
    orig_r = [r for r in valid if r["gt"]["cycle_origin_type"] == "original"]
    gen_r  = [r for r in valid if r["gt"]["cycle_origin_type"] == "generated"]
    if orig_r or gen_r:
        ov_rows = []
        for label, subset in [("original", orig_r), ("generated", gen_r)]:
            if not subset:
                continue
            si24_t = [r["gt"]["stage_24h"]        for r in subset]
            si24_p = [r["pred"]["stage_index_24h"] for r in subset]
            si_now = [r["gt"]["stage_now"]          for r in subset]
            ov_rows.append({
                "origin_type"  : label,
                "n"            : len(subset),
                "acc_24h"      : round(accuracy_score(si24_t, si24_p), 4),
                "f1_24h"       : round(f1_score(si24_t, si24_p, average="weighted", zero_division=0), 4),
                "ord_dist_24h" : round(_ordinal_distance(si24_t, si24_p), 4),
            })
        pd.DataFrame(ov_rows).to_csv(out_dir / "original_vs_generated_metrics.csv", index=False)

    saved = sorted(f.name for f in out_dir.iterdir())
    print(f"  Saved {len(saved)} files: {', '.join(saved[:8])}" +
          (f" ... (+{len(saved)-8} more)" if len(saved) > 8 else ""))


# ============================================================================
# FINAL SUMMARY PRINT — PART J
# ============================================================================

def print_summary(
    art_dir: Path, out_dir: Path, test_cfg: dict,
    functional: dict, bio: dict, metrics: dict,
    n_windows: int, n_scenarios: int,
) -> None:
    bar = "=" * 66
    print(f"\n{bar}")
    print(f"  AgriTwin-GH :: TFT Growth Progression — Test Results")
    print(bar)
    print(f"  Model tested  : {test_cfg['run_id']}")
    print(f"  Test cycles   : {test_cfg['test_cycles']}")
    print(f"  Total windows : {n_windows}")
    print(f"  Scenarios     : {n_scenarios}")
    print(f"  Functional    : {'PASS' if functional.get('all_passed') else 'PARTIAL'}"
          f"  ({functional.get('checks_passed', '?')}/{functional.get('total_checks', '?')} checks)")
    print(f"  Bio sanity    : {bio.get('biological_pass_rate_pct', '?')}% pass rate"
          f"  ({bio.get('total_violations', '?')} violations)")
    if "error" not in metrics:
        m24  = metrics["stage_24h"]
        m48  = metrics["stage_48h"]
        mpr  = metrics["progress_24h"]
        mh2n = metrics["hours_to_next"]
        print(f"\n  Stage +24h    : acc={m24.get('accuracy'):.4f}"
              f"  F1w={m24.get('f1_weighted'):.4f}"
              f"  ord_dist={m24.get('ordinal_distance'):.4f}"
              f"  trans_detect={m24.get('transition_detect_acc'):.4f}")
        print(f"  Stage +48h    : acc={m48.get('accuracy'):.4f}"
              f"  F1w={m48.get('f1_weighted'):.4f}"
              f"  ord_dist={m48.get('ordinal_distance'):.4f}")
        print(f"  Progress +24h : MAE={mpr.get('mae'):.2f}%"
              f"  RMSE={mpr.get('rmse'):.2f}%"
              f"  R2={mpr.get('r2')}")
        print(f"  Hours-to-next : MAE={mh2n.get('mae'):.1f}h"
              f"  within12h={mh2n.get('pct_within_12h')}%"
              f"  within24h={mh2n.get('pct_within_24h')}%"
              f"  bias={mh2n.get('mean_bias_h'):+.1f}h")
        overall_ok = (functional.get("all_passed", False) and
                      bio.get("total_violations", 999) < n_windows * 0.05)
        verdict = "DEPLOY-READY" if overall_ok else "NEEDS REVIEW"
        print(f"\n  Verdict       : {verdict}")
    print(f"  Artifacts     : {out_dir}")
    print(bar)


# ============================================================================
# ARGUMENT PARSING AND MAIN
# ============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgriTwin-GH TFT Growth Progression — Integration Test Suite"
    )
    p.add_argument("--run-id",      default=None,
                   help="Specific run_id to test (default: latest artifact)")
    p.add_argument("--max-windows", type=int, default=60,
                   help="Max windows per test cycle (default: 60, i.e. ~180 total)")
    p.add_argument("--test-cycles", type=int, nargs="+", default=TEST_CYCLES,
                   help="Cycle IDs to use for accuracy evaluation (default: 7 12 16)")
    p.add_argument("--test-id",     default=None,
                   help="Test run ID (default: auto-generated UUID8)")
    p.add_argument("--no-plots",    action="store_true",
                   help="Skip plot generation (faster in CI environments)")
    return p.parse_args()


def main() -> None:
    args    = parse_args()
    test_id = args.test_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    print("=" * 66)
    print("  AgriTwin-GH :: TFT Growth Progression — Integration Tests")
    print(f"  Test ID  : {test_id}")
    print(f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 66)

    # ── A: Load artifacts ────────────────────────────────────────────────────
    art_dir = locate_artifact_dir(args.run_id)
    arts    = load_artifacts(art_dir)
    run_id  = arts["run_id"]
    enc_len = arts["enc_len"]

    # Output dir for this test run
    out_dir = art_dir / f"test_run_{test_id}"

    # ── B: Load dataset and extract test windows ─────────────────────────────
    df      = load_dataset(art_dir)
    windows = extract_windows(df, enc_len, cycles=args.test_cycles,
                              max_per_cycle=args.max_windows)

    if not windows:
        sys.exit("[ERROR] No valid test windows extracted. Check test_cycles and dataset.")

    test_cfg = {
        "test_id"      : test_id,
        "run_id"       : run_id,
        "test_cycles"  : args.test_cycles,
        "max_per_cycle": args.max_windows,
        "n_windows"    : len(windows),
        "enc_len"      : enc_len,
        "pred_len"     : arts["pred_len"],
        "artifact_dir" : str(art_dir),
        "output_dir"   : str(out_dir),
        "started_at"   : datetime.now().isoformat(),
    }

    # ── C: Functional tests (on sample window before full inference) ──────────
    functional = functional_tests(arts, windows)

    # ── Full inference on all test windows ───────────────────────────────────
    print(f"\n  Running inference on {len(windows)} windows ...")
    results = run_all_inference(windows, arts, tag="accuracy")

    n_valid = sum(1 for r in results if r["pred"] is not None)
    print(f"  Valid results: {n_valid}/{len(results)}")

    if n_valid == 0:
        sys.exit("[ERROR] All inference calls failed. Cannot compute metrics.")

    # ── D: Scenario assignment and metrics ───────────────────────────────────
    print("\n[D] Assigning scenario buckets ...")
    buckets    = assign_scenarios(results)
    sc_metrics = compute_scenario_metrics(buckets)
    n_scenarios = sum(1 for _, b in buckets.items() if len(b) > 0)
    for sc_key, bucket in buckets.items():
        if bucket:
            m = _quick_metrics(bucket)
            print(f"  {sc_key:<35} n={m['n']:>4}  "
                  f"acc={m.get('stage_acc_24h', 'N/A')}  "
                  f"prog_mae={m.get('progress_mae_24h', 'N/A')}%")

    # ── E: Biological consistency ─────────────────────────────────────────────
    bio = biological_consistency_tests(results)

    # ── F: Full metrics ───────────────────────────────────────────────────────
    metrics    = compute_all_metrics(results)
    per_stage  = compute_per_stage_metrics(results)
    per_cycle  = compute_per_cycle_metrics(results)

    # ── G: Transition and behavior analysis ──────────────────────────────────
    print("\n[G] Transition analysis ...")
    trans_df   = transition_analysis(results)
    samples_df = build_prediction_samples(results, n=300)
    print(f"  Transition-near windows : {len(trans_df)}")

    # ── H: Save all outputs ───────────────────────────────────────────────────
    save_all_outputs(
        out_dir, test_cfg, functional, bio, metrics,
        per_stage, per_cycle, sc_metrics, trans_df, samples_df, results,
    )

    # ── J: Final summary ──────────────────────────────────────────────────────
    print_summary(art_dir, out_dir, test_cfg, functional, bio, metrics,
                  n_valid, n_scenarios)


if __name__ == "__main__":
    main()
