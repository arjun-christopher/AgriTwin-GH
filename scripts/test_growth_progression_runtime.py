#!/usr/bin/env python
"""
AgriTwin-GH :: TFT Growth Progression Model — Interactive Runtime Test Script
==============================================================================
Interactive script for evaluating the trained Temporal Fusion Transformer (TFT)
under realistic greenhouse scenarios.  Supports both fully interactive
(menu-driven) and non-interactive (CLI-args only) execution.

Usage — interactive:
    python scripts/test_growth_progression_runtime.py

Usage — CLI (non-interactive):
    python scripts/test_growth_progression_runtime.py \\
        --run-id growth_progression_20260308_202542 \\
        --mode 4 --scenario 8 --n-windows 50 --save-plots

Parts
-----
  A  Interactive menu helpers
  B  Model selection and artifact loading
  C  Test data source selection and window extraction
  D  Scenario definitions and filtering
  E  Inference engine
  F  Test mode runners  (modes 1-6)
  G  Functional tests
  H  Biological consistency checks
  I  Metrics computation (stage classification, regression, h2n)
  J  Output display (single prediction, table)
  K  Plots
  L  Save outputs
  M  Final summary + argparse + main
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

warnings.filterwarnings("ignore", message="Found.*unknown classes",            category=UserWarning)
warnings.filterwarnings("ignore", message="The behavior of array concatenation", category=FutureWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true", category=UserWarning)
warnings.filterwarnings("ignore", message="Attribute.*is an instance of.*nn.Module", category=UserWarning)
warnings.filterwarnings("ignore", message=".*does not have many workers.*",     category=UserWarning)

try:
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score,
        precision_score, recall_score, f1_score,
        confusion_matrix,
        mean_absolute_error, mean_squared_error, r2_score,
        explained_variance_score,
    )
except ImportError:
    sys.exit("[ERROR] scikit-learn is required: pip install scikit-learn")

try:
    from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
    from pytorch_forecasting.metrics import QuantileLoss
except ImportError:
    sys.exit("[ERROR] pytorch_forecasting is required: pip install pytorch-forecasting")


class OrdinalQuantileLoss(QuantileLoss):
    """QuantileLoss augmented with an ordinal skip penalty.

    Adds a quadratic penalty whenever |median_pred − truth| > 1 stage,
    making multi-stage jumps disproportionately costly.
    """

    def __init__(self, quantiles, ordinal_weight: float = 0.4, **kwargs):
        super().__init__(quantiles=quantiles, **kwargs)
        self.ordinal_weight = ordinal_weight

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        q_loss  = super().loss(y_pred, y_actual)
        med_idx = len(self.quantiles) // 2
        q_med   = y_pred[..., med_idx]
        gap     = (q_med - y_actual).abs()
        skip    = torch.clamp(gap - 1.0, min=0.0) ** 2
        return q_loss + self.ordinal_weight * skip.mean()

# ============================================================================
# CONSTANTS
# ============================================================================

ROOT          = Path(__file__).resolve().parent.parent
ARTIFACT_BASE = ROOT / "src" / "agritwin_gh" / "models" / "artifacts"
MODEL_DIR     = ROOT / "src" / "agritwin_gh" / "models"

STAGE_ORDER  = ["seedling", "early_vegetative", "flowering_initiation",
                "flowering", "unripe", "ripe"]
STAGE_LABELS = ["Seedling", "Early Vegetative", "Flowering Init.",
                "Flowering", "Unripe", "Ripe"]
STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}
N_STAGES     = 6

STAGE_DURATION_H: dict[str, int] = {
    "seedling"             : 168,
    "early_vegetative"     : 336,
    "flowering_initiation" : 120,
    "flowering"            : 240,
    "unripe"               : 480,
    "ripe"                 : 240,
}

DEFAULT_TEST_CYCLES = [7, 12, 16]
DEFAULT_ENC_LEN     = 72
DEFAULT_PRED_LEN    = 48
QUANTILES           = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
MEDIAN_IDX          = QUANTILES.index(0.5)

REQUIRED_COLUMNS = [
    "cycle_id", "timestamp", "time_idx", "stage_index", "stage_progress_pct",
    "indoor_temp", "indoor_humidity", "indoor_air_velocity", "indoor_CO2",
    "solarradiation", "vpd", "hour", "day_of_year", "week_of_year", "month",
    "target_stage_index_24h", "target_stage_index_48h",
    "target_stage_progress_24h", "target_stage_progress_48h",
    "target_hours_to_next_stage",
]

DIV  = "=" * 68
SDIV = "-" * 68

TEST_MODE_LABELS = [
    "Functional sanity test     (model load + inference structure checks)",
    "Single window prediction   (choose one window, print full output)",
    "Batch scenario testing     (multiple windows, per-scenario metrics)",
    "Full evaluation            (all held-out windows, complete metrics + plots)",
    "Biological consistency     (focus on logical realism violations)",
    "Stage transition focused   (near-stage-boundary windows only)",
]

# ============================================================================
# PART A — INTERACTIVE MENU HELPERS
# ============================================================================

def _hdr(title: str) -> None:
    print(f"\n{DIV}\n  {title}\n{DIV}")

def _sec(title: str) -> None:
    print(f"\n{SDIV}\n  {title}\n{SDIV}")

def _prompt(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {prompt}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        sys.exit("\n[Aborted by user]")
    return val if val else default

def _prompt_int(prompt: str, default: int, lo: int = 1, hi: int = 9999) -> int:
    while True:
        raw = _prompt(prompt, str(default))
        try:
            val = int(raw)
            if lo <= val <= hi:
                return val
            print(f"  [!] Enter a number between {lo} and {hi}.")
        except ValueError:
            print("  [!] Invalid input — please enter a whole number.")

def _prompt_choice(options: list[str], title: str, default: int = 1) -> int:
    print(f"\n  {title}")
    for i, opt in enumerate(options, 1):
        marker = "  (default)" if i == default else ""
        print(f"    {i:>2}. {opt}{marker}")
    return _prompt_int("Enter choice", default, lo=1, hi=len(options))

def _prompt_yn(prompt: str, default: bool = True) -> bool:
    def_str = "y" if default else "n"
    while True:
        raw = _prompt(f"{prompt} (y/n)", def_str).lower()
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  [!] Please enter y or n.")

def _prompt_path(prompt: str, must_exist: bool = True,
                 extensions: Optional[list[str]] = None) -> Path:
    while True:
        raw = _prompt(prompt)
        p   = Path(raw)
        if must_exist and not p.exists():
            print(f"  [!] Path not found: {p}")
            continue
        if extensions and p.suffix.lower() not in extensions:
            print(f"  [!] Expected file type: {extensions}")
            continue
        return p

# ============================================================================
# PART B — MODEL SELECTION AND ARTIFACT LOADING
# ============================================================================

def _find_artifact_dirs() -> list[Path]:
    if not ARTIFACT_BASE.exists():
        return []
    return sorted(
        [d for d in ARTIFACT_BASE.iterdir()
         if d.is_dir() and d.name.startswith("growth_progression_")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

def select_model_interactive() -> Path:
    _sec("STEP 1 — Model Selection")
    mode = _prompt_choice(
        ["Auto-detect models in src/agritwin_gh/models/ (recommended)",
         "Enter model artifact path manually"],
        "How would you like to select the model?",
        default=1,
    )

    if mode == 1:
        candidates = _find_artifact_dirs()
        if not candidates:
            print(f"  [!] No growth_progression artifact folders found in:\n      {ARTIFACT_BASE}")
            print("       Switching to manual entry ...")
            mode = 2
        else:
            print(f"\n  Found {len(candidates)} artifact folder(s):")
            for i, c in enumerate(candidates, 1):
                mtime = datetime.fromtimestamp(c.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
                ckpts = list((c / "checkpoints").glob("*.ckpt")) if (c / "checkpoints").exists() else []
                trn   = "training_config.json" in [f.name for f in c.iterdir()]
                print(f"    {i:>2}. {c.name:<52} mod={mtime}  ckpts={len(ckpts)}"
                      f"  {'[valid]' if trn else '[!missing config]'}")
            idx = _prompt_int("Select artifact number", default=1, lo=1, hi=len(candidates))
            return candidates[idx - 1]

    if mode == 2:
        print(f"\n  Enter the path to an artifact folder.")
        print(f"  Example: {ARTIFACT_BASE / 'growth_progression_20260308_202542'}")
        while True:
            raw = _prompt("Artifact folder path")
            p   = Path(raw)
            if not p.exists():
                print(f"  [!] Path does not exist: {p}")
                continue
            if p.is_file():
                p = p.parent
            if not (p / "training_config.json").exists():
                print(f"  [!] Missing training_config.json — not a valid artifact folder.")
                continue
            return p

    sys.exit("[ERROR] Model selection failed.")

def load_artifacts(art_dir: Path) -> dict:
    """Load all model artifacts: configs, scaler, training dataset, TFT checkpoint."""
    print(f"\n  Loading artifacts from: {art_dir.name} ...")

    def _req(fname: str) -> Path:
        p = art_dir / fname
        if not p.exists():
            sys.exit(f"\n[ERROR] Required file not found: {fname}\n  Expected at: {p}")
        return p

    feature_roles = json.loads(_req("feature_roles.json").read_text())
    seq_meta      = json.loads(_req("sequence_metadata.json").read_text())
    tgt_def       = json.loads(_req("target_definition.json").read_text())
    trn_cfg       = json.loads(_req("training_config.json").read_text())
    split_sum     = json.loads(_req("split_summary.json").read_text())

    run_id     = trn_cfg["run_id"]
    enc_len    = int(trn_cfg.get("encoder_length",    DEFAULT_ENC_LEN))
    pred_len   = int(trn_cfg.get("prediction_length", DEFAULT_PRED_LEN))
    quantiles  = trn_cfg.get("quantiles", QUANTILES)
    median_idx = quantiles.index(0.5) if 0.5 in quantiles else MEDIAN_IDX
    h24        = min(23, pred_len - 1)
    h48        = min(47, pred_len - 1)

    with open(_req("robust_scaler.pkl"), "rb") as fh:
        scaler = pickle.load(fh)

    with open(_req("tft_training_dataset.pkl"), "rb") as fh:
        saved = pickle.load(fh)
    training_dataset = saved["training"]

    scaled_features: list[str] = (
        seq_meta.get("scaled_features") or
        feature_roles.get("time_varying_known_reals", []) +
        feature_roles.get("time_varying_unknown_reals", [])
    )

    # Locate checkpoint
    ckpt_root = MODEL_DIR / f"growth_progression_{run_id}.ckpt"
    ckpt_arts = sorted((art_dir / "checkpoints").glob("*.ckpt")) \
                if (art_dir / "checkpoints").exists() else []

    if ckpt_root.exists():
        ckpt_path = ckpt_root
    elif ckpt_arts:
        ckpt_path = ckpt_arts[-1]
    else:
        sys.exit("[ERROR] No model checkpoint (.ckpt) found in artifact dir or models dir.")

    print(f"    Checkpoint      : {ckpt_path.name}")
    tft_model = TemporalFusionTransformer.load_from_checkpoint(str(ckpt_path))
    tft_model.eval()

    test_cycles = split_sum.get("test", {}).get("cycles", DEFAULT_TEST_CYCLES)
    all_cycles  = sorted(
        split_sum.get("train", {}).get("cycles", []) +
        split_sum.get("val",   {}).get("cycles", []) +
        split_sum.get("test",  {}).get("cycles", [])
    )

    print(f"    Run ID          : {run_id}")
    print(f"    Encoder / pred  : {enc_len}h / {pred_len}h")
    print(f"    Scaler          : {type(scaler).__name__}  ({len(scaler.feature_names_in_)} features)")
    print(f"    Model type      : {type(tft_model).__name__}")
    print(f"    Held-out cycles : {test_cycles}")
    print(f"    All cycles      : {all_cycles}")

    return dict(
        feature_roles=feature_roles, seq_meta=seq_meta, tgt_def=tgt_def,
        trn_cfg=trn_cfg, split_sum=split_sum,
        scaled_features=scaled_features, scaler=scaler,
        training_dataset=training_dataset, tft_model=tft_model,
        art_dir=art_dir, run_id=run_id,
        enc_len=enc_len, pred_len=pred_len,
        quantiles=quantiles, median_idx=median_idx, h24=h24, h48=h48,
        test_cycles=test_cycles, all_cycles=all_cycles,
    )

# ============================================================================
# PART C — TEST DATA SOURCE SELECTION
# ============================================================================

def _load_artifact_df(art_dir: Path) -> pd.DataFrame:
    csv_path  = art_dir / "sequence_ready_dataset.csv"
    parq_path = art_dir / "expanded_hourly_growth_dataset.parquet"
    if csv_path.exists():
        df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)
        print(f"    Dataset : {csv_path.name}  ({len(df):,} rows, {df['cycle_id'].nunique()} cycles)")
    elif parq_path.exists():
        df = pd.read_parquet(parq_path)
        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        print(f"    Dataset : {parq_path.name}  ({len(df):,} rows, {df['cycle_id'].nunique()} cycles)")
    else:
        sys.exit(f"[ERROR] No dataset found in artifact folder: {art_dir}")
    return df

def _validate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def select_data_source_interactive(arts: dict) -> pd.DataFrame:
    _sec("STEP 2 — Test Data Source")
    opts = [
        f"Held-out processed dataset  (test cycles: {arts['test_cycles']})",
        "User-provided CSV file",
        "Extract sequence from processed dataset (choose cycle + range)",
        "Scenario-driven windows from processed dataset",
    ]
    mode = _prompt_choice(opts, "Select test data source:", default=1)

    if mode in (1, 3, 4):
        df      = _load_artifact_df(arts["art_dir"])
        missing = _validate_columns(df)
        if missing:
            sys.exit(f"[ERROR] Dataset missing required columns: {missing}")
        return df

    # mode == 2 — user CSV
    print("\n  Required columns include: cycle_id, timestamp, time_idx, stage_index,")
    print("  stage_progress_pct, indoor_temp, indoor_humidity, solarradiation,")
    print("  target_stage_index_24h, target_stage_index_48h, target_stage_progress_24h,")
    print("  target_stage_progress_48h, target_hours_to_next_stage, and others.")
    csv_path = _prompt_path("CSV file path", must_exist=True, extensions=[".csv"])
    df = pd.read_csv(csv_path, parse_dates=["timestamp"], low_memory=False)
    missing = _validate_columns(df)
    if missing:
        print(f"\n  [!] Missing columns: {missing}")
        print("       Ground-truth metrics will be unavailable for these columns.")
        if not _prompt_yn("Continue anyway?", default=False):
            sys.exit("[Aborted] Supply a CSV with required columns.")
    print(f"  Loaded {len(df):,} rows from {csv_path.name}")
    return df

# ============================================================================
# PART D — SCENARIO DEFINITIONS
# ============================================================================

SCENARIO_DEFS: dict[int, dict] = {
    1:  {"name": "Stable early growth",
         "desc": "Seedling or early vegetative, stage progress < 60%",
         "fn":   lambda gt, enc: gt["stage_now"] in (0, 1) and gt["stage_progress_now"] < 60},
    2:  {"name": "Approaching stage transition",
         "desc": "Any stage, progress > 85% — near-boundary windows",
         "fn":   lambda gt, enc: gt["stage_progress_now"] > 85},
    3:  {"name": "Mid-stage development",
         "desc": "30–70% stage progress — steady midpoint windows",
         "fn":   lambda gt, enc: 30 < gt["stage_progress_now"] < 70},
    4:  {"name": "Day/night environmental variation",
         "desc": "Radiation swing > 100 W/m² in the encoder window",
         "fn":   lambda gt, enc: float(enc["solarradiation"].max() - enc["solarradiation"].min()) > 100},
    5:  {"name": "High humidity greenhouse",
         "desc": "Indoor humidity > 82%",
         "fn":   lambda gt, enc: gt["indoor_humidity"] > 82},
    6:  {"name": "Warm fast-growth conditions",
         "desc": "Temperature > 24°C and solar radiation > 150 W/m²",
         "fn":   lambda gt, enc: gt["indoor_temp"] > 24 and gt["solarradiation"] > 150},
    7:  {"name": "Late fruit development (unripe → ripe)",
         "desc": "Unripe or ripe stage with progress > 75%",
         "fn":   lambda gt, enc: gt["stage_now"] in (4, 5) and gt["stage_progress_now"] > 75},
    8:  {"name": "Mixed scenarios (all windows)",
         "desc": "Sample across all scenario categories without filtering",
         "fn":   lambda gt, enc: True},
    9:  {"name": "Original cycles only",
         "desc": "Non-synthetic, original crop cycle data",
         "fn":   lambda gt, enc: gt.get("cycle_origin_type", "original") == "original"},
    10: {"name": "Generated (synthetic) cycles only",
         "desc": "Augmentation-generated synthetic cycles",
         "fn":   lambda gt, enc: gt.get("cycle_origin_type", "generated") == "generated"},
    11: {"name": "Combined — original + generated",
         "desc": "All cycles together for comparative evaluation",
         "fn":   lambda gt, enc: True},
}

def select_scenario_interactive() -> int:
    _sec("STEP 4 — Scenario Type")
    items = [f"{v['name']} — {v['desc']}" for v in SCENARIO_DEFS.values()]
    return _prompt_choice(items, "Select greenhouse scenario:", default=8)

def select_test_mode_interactive() -> int:
    _sec("STEP 3 — Test Mode")
    return _prompt_choice(TEST_MODE_LABELS, "Select test mode:", default=4)

def select_runtime_config_interactive(arts: dict, test_mode: int) -> dict:
    _sec("STEP 5 — Runtime Configuration")

    enc_choice = _prompt_choice(
        ["48h  — shorter history window",
         "72h  — standard (matches training, recommended)",
         "96h  — extended context"],
        "Encoder / history length:", default=2,
    )
    enc_map = {1: 48, 2: 72, 3: 96}
    enc_len = enc_map[enc_choice]
    if enc_len > arts["enc_len"]:
        print(f"  [!] Model trained with enc_len={arts['enc_len']}. Clamping.")
        enc_len = arts["enc_len"]

    hor_choice = _prompt_choice(
        ["24h only", "48h only", "Both 24h and 48h (recommended)"],
        "Forecast horizon:", default=3,
    )
    horizon = {1: "24h", 2: "48h", 3: "both"}[hor_choice]

    if test_mode == 2:
        n_windows = 1
    elif test_mode == 1:
        n_windows = 5
    else:
        win_choice = _prompt_choice(
            ["1", "5", "10", "25", "50", "100", "All available"],
            "Number of windows to test:", default=5,
        )
        n_map     = {1: 1, 2: 5, 3: 10, 4: 25, 5: 50, 6: 100, 7: 99999}
        n_windows = n_map[win_choice]

    show_plots  = _prompt_yn("Show plots interactively?",         default=False)
    save_plots  = _prompt_yn("Save plots as PNG?",                default=True)
    print_table = _prompt_yn("Print detailed prediction table?",  default=(test_mode in (2, 3)))
    bio_strict  = _prompt_yn("Run strict biological validation?", default=True)

    return dict(
        enc_len=enc_len, horizon=horizon, n_windows=n_windows,
        show_plots=show_plots, save_plots=save_plots,
        print_table=print_table, bio_strict=bio_strict,
    )

# ============================================================================
# PART E — WINDOW EXTRACTION
# ============================================================================

def _gt_from_row(row: "pd.Series") -> dict:
    si = int(row.get("stage_index", 0))
    return {
        "stage_now"         : si,
        "stage_now_name"    : str(row.get("stage_name", STAGE_ORDER[min(si, 5)])),
        "stage_24h"         : int(round(float(row["target_stage_index_24h"]))),
        "stage_48h"         : int(round(float(row["target_stage_index_48h"]))),
        "progress_24h"      : float(row["target_stage_progress_24h"]),
        "progress_48h"      : float(row["target_stage_progress_48h"]),
        "hours_to_next"     : float(row["target_hours_to_next_stage"]),
        "stage_progress_now": float(row["stage_progress_pct"]),
        "indoor_temp"       : float(row["indoor_temp"]),
        "indoor_humidity"   : float(row["indoor_humidity"]),
        "solarradiation"    : float(row.get("solarradiation", 0.0)),
        "vpd"               : float(row.get("vpd", 0.0)),
        "cycle_origin_type" : str(row.get("cycle_origin_type", "generated")),
        "season_label"      : str(row.get("season_label", "summer")),
        "timestamp"         : str(row.get("timestamp", "")),
    }

def _apply_scenario_filter(windows: list[dict], sc_id: int) -> list[dict]:
    sc = SCENARIO_DEFS.get(sc_id)
    if sc is None:
        return windows
    fn = sc["fn"]
    out = []
    for w in windows:
        try:
            if fn(w["gt"], w["enc_rows"]):
                out.append(w)
        except Exception:
            pass
    return out

def extract_windows(
    df: pd.DataFrame,
    enc_len: int,
    cycles: list[int],
    max_windows: int = 50,
    scenario_id: int = 8,
) -> list[dict]:
    """
    Extract chronological encoder windows from the dataset.
    Each window: enc_len consecutive rows + ground-truth targets from last row.
    Windows are evenly spaced, then filtered by scenario, then capped at max_windows.
    """
    all_wins: list[dict] = []
    for cid in cycles:
        cyc = (df[df["cycle_id"] == cid]
               .sort_values("time_idx")
               .reset_index(drop=True))
        cyc = cyc.dropna(subset=["target_stage_index_24h", "target_stage_index_48h"])
        if len(cyc) < enc_len + 1:
            print(f"  [WARN] Cycle {cid}: {len(cyc)} rows — too short, skipping.")
            continue
        max_start = len(cyc) - enc_len
        # Evenly space start positions within cycle
        per_cycle = max(1, max_windows)
        step      = max(1, max_start // per_cycle)
        starts    = list(range(0, max_start, step))[:per_cycle]
        for start in starts:
            enc_rows = cyc.iloc[start: start + enc_len].reset_index(drop=True)
            gt_row   = cyc.iloc[start + enc_len - 1]
            all_wins.append({
                "cycle_id" : int(cid),
                "start_idx": int(start),
                "enc_rows" : enc_rows,
                "gt"       : _gt_from_row(gt_row),
            })

    # Apply scenario filter
    filtered = _apply_scenario_filter(all_wins, scenario_id)
    if not filtered and all_wins:
        sc_name = SCENARIO_DEFS.get(scenario_id, {}).get("name", f"Scenario {scenario_id}")
        print(f"  [WARN] Scenario '{sc_name}' matched 0 windows — using all {len(all_wins)}.")
        filtered = all_wins

    # Global cap
    if max_windows < len(filtered):
        step     = max(1, len(filtered) // max_windows)
        filtered = filtered[::step][:max_windows]

    sc_name = SCENARIO_DEFS.get(scenario_id, {}).get("name", "?")
    print(f"  Windows : {len(filtered)}  (from {len(all_wins)} in {len(cycles)} cycles"
          f"  |  scenario: {sc_name})")
    return filtered

def extract_transition_windows(
    df: pd.DataFrame, enc_len: int, cycles: list[int], max_windows: int = 50
) -> list[dict]:
    """Extract only windows where a real stage transition occurs within 48h."""
    wins: list[dict] = []
    for cid in cycles:
        cyc = (df[df["cycle_id"] == cid]
               .sort_values("time_idx")
               .reset_index(drop=True))
        cyc = cyc.dropna(subset=["target_stage_index_24h", "target_stage_index_48h"])
        if len(cyc) < enc_len + 1:
            continue
        for start in range(len(cyc) - enc_len):
            gt_row = cyc.iloc[start + enc_len - 1]
            si_now = int(gt_row["stage_index"])
            si24   = int(round(float(gt_row["target_stage_index_24h"])))
            si48   = int(round(float(gt_row["target_stage_index_48h"])))
            if si24 > si_now or si48 > si_now:
                enc_rows = cyc.iloc[start: start + enc_len].reset_index(drop=True)
                wins.append({
                    "cycle_id" : int(cid),
                    "start_idx": int(start),
                    "enc_rows" : enc_rows,
                    "gt"       : _gt_from_row(gt_row),
                })

    if len(wins) > max_windows:
        step = max(1, len(wins) // max_windows)
        wins = wins[::step][:max_windows]

    print(f"  Transition windows : {len(wins)}")
    return wins

# ============================================================================
# PART F — INFERENCE ENGINE
# ============================================================================

def _build_window_df(enc_rows: pd.DataFrame) -> pd.DataFrame:
    df = enc_rows.copy()
    df["time_idx"] = np.arange(len(df), dtype=int)
    df["cycle_id"] = "999"
    if "cycle_origin_type" in df.columns:
        df["cycle_origin_type"] = df["cycle_origin_type"].fillna("original").astype(str)
    else:
        df["cycle_origin_type"] = "original"
    if "season_label" in df.columns:
        df["season_label"] = df["season_label"].fillna("summer").astype(str)
    else:
        df["season_label"] = "summer"
    # NOTE: Do NOT zero out target_stage_index_24h here.
    # EncoderNormalizer computes output scale from encoder target values;
    # zeroing them collapses target_scale to ~0 and destroys all predictions.
    return df

def _append_decoder_rows(enc_df: pd.DataFrame, pred_len: int) -> pd.DataFrame:
    last     = enc_df.iloc[-1].to_dict()
    last_ti  = int(last["time_idx"])
    last_hr  = int(last.get("hour", 12))
    last_doy = int(last.get("day_of_year", 180))
    dec_rows = []
    for k in range(1, pred_len + 1):
        r   = dict(last)
        fh  = (last_hr + k) % 24
        ed  = (last_hr + k) // 24
        r["time_idx"]               = last_ti + k
        r["hour"]                   = float(fh)
        r["day_of_year"]            = float(min(last_doy + ed, 366))
        r["day_night_flag"]         = float(int(6 <= fh < 20))
        r["light_period_flag"]      = float(int(8 <= fh < 18))
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
    Run one TFT forward pass on a real-data encoder window.
    Returns a prediction dict, or None on any failure.
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

        si24   = int(np.clip(round(p24), 0, 5))
        si48   = int(np.clip(round(p48), 0, 5))
        prog24 = float(np.clip((p24 - math.floor(p24)) * 100.0, 0.0, 100.0))
        prog48 = float(np.clip((p48 - math.floor(p48)) * 100.0, 0.0, 100.0))

        # Hours-to-next: first decoder step where stage index increases
        cur_si    = int(enc_rows.iloc[-1].get("stage_index", 0))
        cur_base  = int(math.floor(q50[0]))
        trans_step = next(
            (i for i, v in enumerate(q50) if int(math.floor(v)) > cur_base), None
        )
        if trans_step is not None:
            hours_nxt = float(trans_step + 1)
        else:
            frac_now  = float(q50[0] - cur_base)
            stage_key = STAGE_ORDER[cur_si] if 0 <= cur_si < N_STAGES else "early_vegetative"
            hours_nxt = round((1.0 - frac_now) * STAGE_DURATION_H.get(stage_key, 240), 1)

        def _probs(q_arr: np.ndarray) -> list[float]:
            mu    = q_arr[arts["median_idx"]]
            sigma = max((q_arr[-1] - q_arr[0]) / 4.0, 0.3)
            raw   = np.exp(-0.5 * ((np.arange(float(N_STAGES)) - mu) / sigma) ** 2)
            return (raw / raw.sum()).tolist()

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
            "has_nan"         : bool(np.any(np.isnan(preds))),
            "has_inf"         : bool(np.any(np.isinf(preds))),
        }
    except Exception:
        return None

def run_batch_inference(windows: list[dict], arts: dict, tag: str = "") -> list[dict]:
    """Batch inference with progress bar. Returns records with 'pred' key attached."""
    results = []
    n_ok = n_err = 0
    t0   = time.time()
    bw   = 38
    for i, win in enumerate(windows):
        pred = run_inference(win["enc_rows"], arts)
        rec  = {k: v for k, v in win.items()}
        rec["pred"] = pred
        results.append(rec)
        n_ok  += pred is not None
        n_err += pred is None
        frac   = (i + 1) / len(windows)
        bar    = "█" * int(bw * frac) + "░" * (bw - int(bw * frac))
        rate   = (i + 1) / max(time.time() - t0, 0.001)
        label  = f" [{tag}]" if tag else ""
        print(f"\r  [{bar}]{label} {i+1}/{len(windows)}  ok={n_ok} err={n_err}"
              f"  {rate:.1f} w/s", end="", flush=True)
    print(f"\n  Inference: {n_ok} ok, {n_err} failed — {time.time()-t0:.1f}s")
    return results

# ============================================================================
# PART G — FUNCTIONAL TESTS
# ============================================================================

def run_functional_tests(arts: dict, windows: list[dict]) -> dict:
    """Structural and inference correctness checks."""
    print("\n  [Functional tests]")
    res: dict = {}
    failures:  list[str] = []

    def _chk(key: str, ok: bool, msg: str) -> None:
        res[key] = ok
        if not ok:
            failures.append(msg)

    _chk("F01_model_type",
         isinstance(arts["tft_model"], TemporalFusionTransformer),
         "F01: model is not TemporalFusionTransformer")
    _chk("F02_scaler_loaded",
         arts["scaler"] is not None, "F02: scaler is None")
    n_sc, n_ls = len(arts["scaler"].feature_names_in_), len(arts["scaled_features"])
    res["F03_scaler_feature_count"] = n_sc
    _chk("F03_scaler_matches_config", n_sc == n_ls,
         f"F03: scaler feature count ({n_sc}) ≠ config ({n_ls})")
    _chk("F04_training_dataset_loaded",
         arts["training_dataset"] is not None, "F04: training_dataset is None")
    _chk("F05_windows_available", len(windows) > 0, "F05: no test windows available")

    if not windows:
        return _finalize_fn(res, failures)

    pred = run_inference(windows[0]["enc_rows"], arts)
    _chk("F06_inference_executes", pred is not None,
         "F06: inference raised an exception on sample window")
    if pred is None:
        return _finalize_fn(res, failures)

    req = ["stage_index_24h", "stage_index_48h", "progress_24h",
           "progress_48h", "hours_to_next", "class_probs_24h"]
    miss = [k for k in req if k not in pred]
    _chk("F07_output_keys_present", not miss, f"F07: missing output keys: {miss}")
    _chk("F08_no_nan", not pred.get("has_nan", True), "F08: NaN detected in predictions")
    _chk("F09_no_inf", not pred.get("has_inf", True), "F09: Inf detected in predictions")
    _chk("F10_stage_24h_valid",
         0 <= pred["stage_index_24h"] <= 5,
         f"F10: stage_index_24h out of range: {pred['stage_index_24h']}")
    _chk("F11_stage_48h_valid",
         0 <= pred["stage_index_48h"] <= 5,
         f"F11: stage_index_48h out of range: {pred['stage_index_48h']}")
    _chk("F12_progress_24h_valid",
         0.0 <= pred["progress_24h"] <= 100.0,
         f"F12: progress_24h out of [0,100]: {pred['progress_24h']}")
    _chk("F13_progress_48h_valid",
         0.0 <= pred["progress_48h"] <= 100.0,
         f"F13: progress_48h out of [0,100]: {pred['progress_48h']}")
    _chk("F14_hours_to_next_nonneg",
         pred["hours_to_next"] >= 0.0,
         f"F14: hours_to_next negative: {pred['hours_to_next']}")
    ps24 = sum(pred.get("class_probs_24h", [0]))
    _chk("F15_class_probs_sum_to_1",
         abs(ps24 - 1.0) < 0.02, f"F15: class_probs_24h sum = {ps24:.4f}")
    return _finalize_fn(res, failures)

def _finalize_fn(res: dict, failures: list[str]) -> dict:
    bools    = [k for k, v in res.items() if isinstance(v, bool)]
    n_passed = sum(1 for k in bools if res[k])
    res["checks_passed"] = n_passed
    res["total_checks"]  = len(bools)
    res["failures"]      = failures
    res["all_passed"]    = len(failures) == 0
    status = "PASS" if res["all_passed"] else f"PARTIAL  ({len(failures)} check(s) failed)"
    print(f"  Functional : {status}  ({n_passed}/{len(bools)} passed)")
    for f in failures:
        print(f"    FAIL: {f}")
    return res

# ============================================================================
# PART H — BIOLOGICAL CONSISTENCY CHECKS
# ============================================================================

def run_biological_checks(results: list[dict]) -> dict:
    """Check all predictions for biological and logical plausibility."""
    valid = [r for r in results if r["pred"] is not None]
    n     = len(valid)
    if n == 0:
        return {"error": "No valid predictions to check"}

    viols: dict[str, list] = {
        "backward_jump_24h"   : [],
        "backward_jump_48h"   : [],
        "multi_stage_skip_24h": [],
        "multi_stage_skip_48h": [],
        "progress_backward"   : [],
        "h2n_inconsistent"    : [],
        "h2n_negative"        : [],
        "h2n_extreme"         : [],
    }

    for rec in valid:
        pred   = rec["pred"]
        gt     = rec["gt"]
        si_now = int(gt["stage_now"])
        si24   = int(pred["stage_index_24h"])
        si48   = int(pred["stage_index_48h"])
        h2n    = float(pred["hours_to_next"])
        pr24   = float(pred["progress_24h"])
        pr48   = float(pred["progress_48h"])
        b      = {"cycle": rec["cycle_id"], "timestamp": gt["timestamp"],
                  "stage_now": si_now, "stage_now_name": gt.get("stage_now_name", "")}

        # E1 — No backward stage jump
        if si24 < si_now:
            viols["backward_jump_24h"].append({**b, "si24": si24})
        if si48 < si_now:
            viols["backward_jump_48h"].append({**b, "si48": si48})

        # E2 — No impossible multi-stage jump within 24h / 48h
        if si24 > si_now + 1:
            viols["multi_stage_skip_24h"].append({**b, "si24": si24, "skip": si24 - si_now})
        if si48 > si_now + 2:
            viols["multi_stage_skip_48h"].append({**b, "si48": si48, "skip": si48 - si_now})

        # E3 — Progress should not decrease if stage stays the same
        if si24 == si48 and pr48 < pr24 - 5.0:
            viols["progress_backward"].append({**b, "pr24": round(pr24, 1), "pr48": round(pr48, 1)})

        # E4 — If +24h same stage but +48h next stage, h2n must be ≤ 48h
        if si24 == si_now and si48 == si_now + 1 and h2n > 48:
            viols["h2n_inconsistent"].append({**b, "si48": si48, "h2n": round(h2n, 1)})

        # E5 — Basic h2n sanity
        if h2n < 0:
            viols["h2n_negative"].append({**b, "h2n": round(h2n, 1)})
        if h2n > 2000:
            viols["h2n_extreme"].append({**b, "h2n": round(h2n, 1)})

    total_v   = sum(len(v) for v in viols.values())
    pass_rate = round(100.0 * (1 - total_v / max(n, 1)), 2)

    summary = {
        "n_evaluated"             : n,
        "total_violations"        : total_v,
        "biological_pass_rate_pct": pass_rate,
        "violation_counts"        : {k: len(v) for k, v in viols.items()},
        "violation_examples"      : {k: v[:5] for k, v in viols.items() if v},
    }

    print(f"  Biological : pass_rate={pass_rate}%  ({total_v} violations / {n} samples)")
    for k, lst in viols.items():
        if lst:
            print(f"    {k:<30}: {len(lst)}")
    return summary

# ============================================================================
# PART I — METRICS COMPUTATION
# ============================================================================

def _ord_dist(y_true: list[int], y_pred: list[int]) -> float:
    return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))

def compute_stage_metrics(
    y_true: list[int], y_pred: list[int], si_now: list[int], horizon: str
) -> dict:
    labels   = list(range(N_STAGES))
    present  = sorted(set(y_true) | set(y_pred))
    cm       = confusion_matrix(y_true, y_pred, labels=labels)
    trans_gt = [int(t > n) for t, n in zip(y_true, si_now)]
    trans_pr = [int(p > n) for p, n in zip(y_pred, si_now)]
    t_acc    = float(accuracy_score(trans_gt, trans_pr)) if trans_gt else 0.0
    return {
        "horizon"              : horizon,
        "n_samples"            : len(y_true),
        "accuracy"             : round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy"    : round(balanced_accuracy_score(y_true, y_pred), 4),
        "precision_macro"      : round(precision_score(y_true, y_pred, average="macro",
                                       zero_division=0, labels=present), 4),
        "recall_macro"         : round(recall_score(y_true, y_pred, average="macro",
                                       zero_division=0, labels=present), 4),
        "f1_macro"             : round(f1_score(y_true, y_pred, average="macro",
                                       zero_division=0, labels=present), 4),
        "f1_weighted"          : round(f1_score(y_true, y_pred, average="weighted",
                                       zero_division=0), 4),
        "ordinal_distance"     : round(_ord_dist(y_true, y_pred), 4),
        "transition_detect_acc": round(t_acc, 4),
        "confusion_matrix"     : cm.tolist(),
    }

def compute_regression_metrics(
    y_true: list[float], y_pred: list[float], name: str
) -> dict:
    t, p    = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    abs_err = np.abs(t - p)
    try:
        r2 = float(r2_score(t, p))
        ev = float(explained_variance_score(t, p))
    except Exception:
        r2 = ev = float("nan")
    mape = float(np.nanmean(np.where(t != 0, np.abs((t - p) / t), np.nan))) * 100
    return {
        "metric"           : name,
        "n_samples"        : len(y_true),
        "mae"              : round(float(np.mean(abs_err)), 4),
        "rmse"             : round(float(np.sqrt(np.mean((t - p) ** 2))), 4),
        "r2"               : round(r2, 4) if not math.isnan(r2) else None,
        "explained_var"    : round(ev, 4) if not math.isnan(ev) else None,
        "median_abs_error" : round(float(np.median(abs_err)), 4),
        "mape_pct"         : round(mape, 2) if not math.isnan(mape) else None,
        "mean_bias"        : round(float(np.mean(p - t)), 4),
    }

def compute_h2n_metrics(y_true: list[float], y_pred: list[float]) -> dict:
    t, p    = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
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

def compute_all_metrics(results: list[dict], horizon: str = "both") -> dict:
    valid = [r for r in results if r["pred"] is not None]
    n     = len(valid)
    if n == 0:
        return {"error": "No valid predictions"}

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

    out: dict = {"n_valid": n}
    if horizon in ("24h", "both"):
        out["stage_24h"]    = compute_stage_metrics(si24_t, si24_p, si_now, "24h")
        out["progress_24h"] = compute_regression_metrics(prog24_t, prog24_p, "stage_progress_24h")
    if horizon in ("48h", "both"):
        out["stage_48h"]    = compute_stage_metrics(si48_t, si48_p, si_now, "48h")
        out["progress_48h"] = compute_regression_metrics(prog48_t, prog48_p, "stage_progress_48h")
    out["hours_to_next"] = compute_h2n_metrics(h2n_t, h2n_p)
    return out

def compute_per_stage_metrics(results: list[dict]) -> pd.DataFrame:
    valid = [r for r in results if r["pred"] is not None]
    rows  = []
    for si, stage in enumerate(STAGE_ORDER):
        sub = [r for r in valid if r["gt"]["stage_now"] == si]
        if not sub:
            rows.append({"stage": stage, "n": 0}); continue
        si24_t = [r["gt"]["stage_24h"]        for r in sub]
        si24_p = [r["pred"]["stage_index_24h"] for r in sub]
        pr24_t = [r["gt"]["progress_24h"]      for r in sub]
        pr24_p = [r["pred"]["progress_24h"]    for r in sub]
        h2n_t  = [r["gt"]["hours_to_next"]     for r in sub]
        h2n_p  = [r["pred"]["hours_to_next"]   for r in sub]
        rows.append({
            "stage"            : stage,
            "n"                : len(sub),
            "acc_24h"          : round(accuracy_score(si24_t, si24_p), 4),
            "f1_weighted_24h"  : round(f1_score(si24_t, si24_p, average="weighted",
                                       zero_division=0), 4),
            "ord_dist_24h"     : round(_ord_dist(si24_t, si24_p), 4),
            "progress_mae_24h" : round(mean_absolute_error(pr24_t, pr24_p), 2),
            "h2n_mae"          : round(mean_absolute_error(h2n_t, h2n_p), 2),
        })
    return pd.DataFrame(rows)

def compute_per_cycle_metrics(results: list[dict]) -> pd.DataFrame:
    valid  = [r for r in results if r["pred"] is not None]
    cycles = sorted(set(r["cycle_id"] for r in valid))
    rows   = []
    for cyc in cycles:
        sub    = [r for r in valid if r["cycle_id"] == cyc]
        si24_t = [r["gt"]["stage_24h"]          for r in sub]
        si24_p = [r["pred"]["stage_index_24h"]   for r in sub]
        pr24_t = [r["gt"]["progress_24h"]        for r in sub]
        pr24_p = [r["pred"]["progress_24h"]      for r in sub]
        h2n_t  = [r["gt"]["hours_to_next"]       for r in sub]
        h2n_p  = [r["pred"]["hours_to_next"]     for r in sub]
        rows.append({
            "cycle_id"         : cyc,
            "origin_type"      : "/".join(sorted(set(r["gt"]["cycle_origin_type"] for r in sub))),
            "n"                : len(sub),
            "acc_24h"          : round(accuracy_score(si24_t, si24_p), 4),
            "f1_weighted_24h"  : round(f1_score(si24_t, si24_p, average="weighted",
                                       zero_division=0), 4),
            "ord_dist_24h"     : round(_ord_dist(si24_t, si24_p), 4),
            "progress_mae_24h" : round(mean_absolute_error(pr24_t, pr24_p), 2),
            "h2n_mae"          : round(mean_absolute_error(h2n_t, h2n_p), 2),
        })
    return pd.DataFrame(rows)

def compute_per_scenario_metrics(results: list[dict]) -> pd.DataFrame:
    valid = [r for r in results if r["pred"] is not None]
    rows  = []
    for sc_id, sc in SCENARIO_DEFS.items():
        sub = []
        for rec in valid:
            try:
                if sc["fn"](rec["gt"], rec["enc_rows"]):
                    sub.append(rec)
            except Exception:
                pass
        if not sub:
            rows.append({"scenario_id": sc_id, "name": sc["name"], "n": 0}); continue
        si24_t = [r["gt"]["stage_24h"]          for r in sub]
        si24_p = [r["pred"]["stage_index_24h"]   for r in sub]
        pr24_t = [r["gt"]["progress_24h"]        for r in sub]
        pr24_p = [r["pred"]["progress_24h"]      for r in sub]
        h2n_t  = [r["gt"]["hours_to_next"]       for r in sub]
        h2n_p  = [r["pred"]["hours_to_next"]     for r in sub]
        rows.append({
            "scenario_id"      : sc_id,
            "name"             : sc["name"],
            "n"                : len(sub),
            "acc_24h"          : round(accuracy_score(si24_t, si24_p), 4),
            "f1_weighted_24h"  : round(f1_score(si24_t, si24_p, average="weighted",
                                       zero_division=0), 4),
            "ord_dist_24h"     : round(_ord_dist(si24_t, si24_p), 4),
            "progress_mae_24h" : round(mean_absolute_error(pr24_t, pr24_p), 2),
            "h2n_mae"          : round(mean_absolute_error(h2n_t, h2n_p), 2),
        })
    return pd.DataFrame(rows)

def compute_transition_analysis(results: list[dict]) -> pd.DataFrame:
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
        if si24t <= si_n and si48t <= si_n:
            continue
        rows.append({
            "cycle_id"        : rec["cycle_id"],
            "timestamp"       : gt["timestamp"],
            "stage_now"       : STAGE_ORDER[si_n] if 0 <= si_n < N_STAGES else "?",
            "actual_trans_24h": si24t > si_n,
            "actual_trans_48h": si48t > si_n and si24t == si_n,
            "pred_trans_24h"  : si24p > si_n,
            "pred_trans_48h"  : si48p > si_n,
            "true_h2n"        : round(float(gt["hours_to_next"]), 1),
            "pred_h2n"        : round(float(pred["hours_to_next"]), 1),
            "h2n_error_h"     : round(float(pred["hours_to_next"]) - float(gt["hours_to_next"]), 1),
            "correct_24h"     : (si24p > si_n) == (si24t > si_n),
            "stage_skip"      : max(0, si24p - si24t),
        })
    cols = ["cycle_id", "timestamp", "stage_now", "actual_trans_24h", "actual_trans_48h",
            "pred_trans_24h", "pred_trans_48h", "true_h2n", "pred_h2n",
            "h2n_error_h", "correct_24h", "stage_skip"]
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)

def build_prediction_samples(results: list[dict], n: int = 300) -> pd.DataFrame:
    valid = [r for r in results if r["pred"] is not None]
    step  = max(1, len(valid) // max(1, n))
    rows  = []
    for rec in valid[::step]:
        gt   = rec["gt"]
        pred = rec["pred"]
        si24 = pred["stage_index_24h"]
        si48 = pred["stage_index_48h"]
        rows.append({
            "cycle_id"              : rec["cycle_id"],
            "timestamp"             : gt["timestamp"],
            "stage_now"             : gt["stage_now_name"],
            "stage_progress_now_pct": round(gt["stage_progress_now"], 1),
            "actual_stage_24h"      : STAGE_ORDER[gt["stage_24h"]] if 0 <= gt["stage_24h"] < N_STAGES else "?",
            "pred_stage_24h"        : STAGE_ORDER[si24] if 0 <= si24 < N_STAGES else "?",
            "actual_stage_48h"      : STAGE_ORDER[gt["stage_48h"]] if 0 <= gt["stage_48h"] < N_STAGES else "?",
            "pred_stage_48h"        : STAGE_ORDER[si48] if 0 <= si48 < N_STAGES else "?",
            "actual_progress_24h"   : round(gt["progress_24h"], 1),
            "pred_progress_24h"     : round(pred["progress_24h"], 1),
            "actual_progress_48h"   : round(gt["progress_48h"], 1),
            "pred_progress_48h"     : round(pred["progress_48h"], 1),
            "actual_h2n"            : round(gt["hours_to_next"], 1),
            "pred_h2n"              : round(pred["hours_to_next"], 1),
            "stage_correct_24h"     : int(gt["stage_24h"] == si24),
            "stage_correct_48h"     : int(gt["stage_48h"] == si48),
            "indoor_temp"           : round(gt["indoor_temp"], 1),
            "indoor_humidity"       : round(gt["indoor_humidity"], 1),
            "solarradiation"        : round(gt["solarradiation"], 1),
        })
    return pd.DataFrame(rows)

# ============================================================================
# PART J — OUTPUT DISPLAY
# ============================================================================

def print_single_prediction(rec: dict, idx: int = 1) -> None:
    gt   = rec["gt"]
    pred = rec["pred"]
    if pred is None:
        print(f"\n  Window {idx}: [INFERENCE FAILED]")
        return

    si24 = pred["stage_index_24h"]
    si48 = pred["stage_index_48h"]
    g24  = gt["stage_24h"]
    g48  = gt["stage_48h"]
    si_n = gt["stage_now"]

    ck24 = "✓" if si24 == g24 else "✗"
    ck48 = "✓" if si48 == g48 else "✗"
    bio  = (si24 >= si_n and si48 >= si24 and
            not (si24 > si_n + 1) and not (si48 > si_n + 2))

    prn24 = STAGE_ORDER[si24] if 0 <= si24 < N_STAGES else "?"
    prn48 = STAGE_ORDER[si48] if 0 <= si48 < N_STAGES else "?"
    act24 = STAGE_ORDER[g24]  if 0 <= g24  < N_STAGES else "?"
    act48 = STAGE_ORDER[g48]  if 0 <= g48  < N_STAGES else "?"
    ts    = str(gt["timestamp"])[:19]

    print(f"\n  {'─'*62}")
    print(f"  Window {idx}  |  Cycle {rec['cycle_id']}  |  {ts}")
    print(f"  {'─'*62}")
    print(f"  Current state")
    print(f"    Stage       : {gt['stage_now_name']}  (index {si_n})")
    print(f"    Progress    : {gt['stage_progress_now']:.1f}%")
    print(f"    Temp        : {gt['indoor_temp']:.1f}°C  "
          f"| Humidity: {gt['indoor_humidity']:.1f}%  "
          f"| Solar: {gt['solarradiation']:.1f} W/m²")
    print(f"  Predictions")
    print(f"    +24h stage  : {prn24:<25}   actual: {act24}  {ck24}")
    print(f"    +48h stage  : {prn48:<25}   actual: {act48}  {ck48}")
    print(f"    +24h prog   : {pred['progress_24h']:>6.1f}%          actual: {gt['progress_24h']:.1f}%")
    print(f"    +48h prog   : {pred['progress_48h']:>6.1f}%          actual: {gt['progress_48h']:.1f}%")
    print(f"    Hrs to next : {pred['hours_to_next']:>6.1f}h         actual: {gt['hours_to_next']:.1f}h")
    print(f"    IQR +24h    : [{pred['iqr_24h'][0]:.2f}, {pred['iqr_24h'][1]:.2f}]")
    print(f"    IQR +48h    : [{pred['iqr_48h'][0]:.2f}, {pred['iqr_48h'][1]:.2f}]")
    print(f"  Biological    : {'PASS' if bio else 'WARNING — check consistency'}")

def print_prediction_table(results: list[dict], max_rows: int = 30) -> None:
    valid = [r for r in results if r["pred"] is not None]
    if not valid:
        return
    sample = valid[:max_rows]
    hdr = (f"  {'Cycle':>5}  {'Timestamp':<19}  {'NowStage':<22}"
           f"  {'Pred+24h':<22}  {'Act+24h':<22}  OK  {'H2N-P':>7}  {'H2N-A':>7}")
    sep = f"  {'─'*110}"
    print(f"\n{sep}\n{hdr}\n{sep}")
    for rec in sample:
        gt   = rec["gt"]
        pred = rec["pred"]
        si24 = pred["stage_index_24h"]
        g24  = gt["stage_24h"]
        ok   = "✓" if si24 == g24 else "✗"
        ps24 = STAGE_ORDER[si24] if 0 <= si24 < N_STAGES else "?"
        as24 = STAGE_ORDER[g24]  if 0 <= g24  < N_STAGES else "?"
        ts   = str(gt["timestamp"])[:19]
        print(f"  {rec['cycle_id']:>5}  {ts:<19}  {gt['stage_now_name']:<22}"
              f"  {ps24:<22}  {as24:<22}  {ok}   "
              f"{pred['hours_to_next']:>7.1f}  {gt['hours_to_next']:>7.1f}")
    if len(valid) > max_rows:
        print(f"  ... ({len(valid) - max_rows} more rows not shown)")
    print(sep)

def print_metrics_block(metrics: dict, bio: dict, horizon: str) -> None:
    if "error" in metrics:
        print(f"  [!] Metrics: {metrics['error']}")
        return
    print()
    if horizon in ("24h", "both") and "stage_24h" in metrics:
        m = metrics["stage_24h"]
        print(f"  Stage classification  +24h")
        print(f"    Accuracy           : {m.get('accuracy')}")
        print(f"    Balanced accuracy  : {m.get('balanced_accuracy')}")
        print(f"    F1 macro           : {m.get('f1_macro')}")
        print(f"    F1 weighted        : {m.get('f1_weighted')}")
        print(f"    Ordinal distance   : {m.get('ordinal_distance')}")
        print(f"    Transition detect  : {m.get('transition_detect_acc')}")
    if horizon in ("48h", "both") and "stage_48h" in metrics:
        m = metrics["stage_48h"]
        print(f"  Stage classification  +48h")
        print(f"    Accuracy           : {m.get('accuracy')}")
        print(f"    F1 weighted        : {m.get('f1_weighted')}")
        print(f"    Ordinal distance   : {m.get('ordinal_distance')}")
    if "progress_24h" in metrics:
        m = metrics["progress_24h"]
        print(f"  Stage progress  +24h")
        print(f"    MAE                : {m.get('mae')}%")
        print(f"    RMSE               : {m.get('rmse')}%")
        print(f"    R²                 : {m.get('r2')}")
    if "hours_to_next" in metrics:
        m = metrics["hours_to_next"]
        print(f"  Hours to next stage")
        print(f"    MAE                : {m.get('mae')}h")
        print(f"    RMSE               : {m.get('rmse')}h")
        print(f"    % within  6h       : {m.get('pct_within_6h')}%")
        print(f"    % within 12h       : {m.get('pct_within_12h')}%")
        print(f"    % within 24h       : {m.get('pct_within_24h')}%")
        print(f"    Mean bias          : {m.get('mean_bias_h')}h")
    print(f"  Biological consistency")
    print(f"    Pass rate          : {bio.get('biological_pass_rate_pct', 'N/A')}%")
    print(f"    Total violations   : {bio.get('total_violations', 'N/A')}")

# ============================================================================
# PART K — PLOTS
# ============================================================================

def _save_show(path: Path, show: bool) -> None:
    plt.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        try:
            plt.show()
        except Exception:
            pass
    plt.close()

def plot_confusion_matrix(cm_data: list[list], title: str,
                          path: Path, show: bool = False) -> None:
    cm  = np.array(cm_data, dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    im  = ax.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(N_STAGES)); ax.set_xticklabels(STAGE_LABELS, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(N_STAGES)); ax.set_yticklabels(STAGE_LABELS, fontsize=8)
    mx = cm.max() if cm.max() > 0 else 1
    for i in range(N_STAGES):
        for j in range(N_STAGES):
            col = "white" if cm[i, j] > mx * 0.5 else "black"
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center",
                    fontsize=8, color=col, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("Actual", fontsize=9)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    _save_show(path, show)

def plot_error_distribution(errors: list[float], title: str, xlabel: str,
                            path: Path, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=40, color="#2196F3", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red",    lw=1.2, linestyle="--", label="Zero error")
    ax.axvline(float(np.mean(errors)), color="orange", lw=1.2,
               linestyle=":", label=f"Mean={np.mean(errors):.1f}")
    ax.set_title(title, fontsize=10); ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("Count", fontsize=9); ax.legend(fontsize=8)
    plt.tight_layout()
    _save_show(path, show)

def plot_actual_vs_predicted(y_true: list, y_pred: list, title: str,
                             label: str, path: Path, show: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y_true, y_pred, alpha=0.3, s=12, color="#2196F3")
    lo = min(min(y_true), min(y_pred)) * 0.95
    hi = max(max(y_true), max(y_pred)) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1, label="Perfect")
    ax.set_xlim([lo, hi]); ax.set_ylim([lo, hi])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(f"Actual {label}", fontsize=9)
    ax.set_ylabel(f"Predicted {label}", fontsize=9)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_show(path, show)

def plot_scenario_comparison(sc_df: pd.DataFrame, path: Path,
                             show: bool = False) -> None:
    df = sc_df[sc_df["n"] > 0].copy()
    if df.empty:
        return
    cols   = ["acc_24h", "progress_mae_24h", "h2n_mae"]
    titles = ["Stage Accuracy +24h", "Progress MAE +24h (%)", "Hours-to-Next MAE"]
    colors = ["#4CAF50", "#FF9800", "#2196F3"]
    labels = [str(r)[:28] for r in df["name"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, col, t, c in zip(axes, cols, titles, colors):
        if col not in df.columns:
            ax.set_visible(False); continue
        ax.barh(labels, df[col], color=c, edgecolor="white")
        ax.set_title(t, fontsize=9); ax.set_xlabel("Score", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    plt.suptitle("Scenario Comparison — TFT Growth Progression", fontsize=10)
    plt.tight_layout()
    _save_show(path, show)

def plot_stage_trajectories(results: list[dict], out_dir: Path,
                            n: int = 4, show: bool = False) -> None:
    valid  = [r for r in results if r["pred"] is not None]
    sample = valid[::max(1, len(valid) // n)][:n]
    if not sample:
        return
    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4), sharey=True)
    if len(sample) == 1:
        axes = [axes]
    for ax, rec in zip(axes, sample):
        gt   = rec["gt"]
        pred = rec["pred"]
        enc_si = rec["enc_rows"]["stage_index"].values.tolist()
        L = len(enc_si)
        ax.plot(range(L), enc_si, color="gray", lw=1, alpha=0.6, label="Observed")
        ax.scatter([L + 23], [gt["stage_24h"]],         marker="o", s=50, color="green",   label="Actual +24h")
        ax.scatter([L + 47], [gt["stage_48h"]],         marker="o", s=50, color="darkgreen")
        ax.scatter([L + 23], [pred["stage_index_24h"]], marker="x", s=60, color="red",     label="Pred +24h")
        ax.scatter([L + 47], [pred["stage_index_48h"]], marker="x", s=60, color="darkred")
        ax.set_yticks(range(N_STAGES)); ax.set_yticklabels(STAGE_LABELS, fontsize=6)
        ax.set_title(f"Cycle {rec['cycle_id']}", fontsize=8)
        ax.set_xlabel("Hour index", fontsize=7); ax.legend(fontsize=6)
    plt.suptitle("Stage Trajectory: Observed + Forecast", fontsize=9)
    plt.tight_layout()
    _save_show(out_dir / "stage_prediction_examples.png", show)

def plot_transition_timing(trans_df: pd.DataFrame, path: Path,
                           show: bool = False) -> None:
    errors = trans_df["h2n_error_h"].dropna().tolist()
    if not errors:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors, bins=30, color="#9C27B0", edgecolor="white", linewidth=0.4)
    ax.axvline(0, color="red", lw=1.2, linestyle="--", label="Zero error")
    ax.axvline(float(np.mean(errors)), color="orange", lw=1.2,
               linestyle=":", label=f"Mean={np.mean(errors):.1f}h")
    ax.set_title("Transition Timing Error Distribution", fontsize=10)
    ax.set_xlabel("Predicted − Actual hours_to_next (h)", fontsize=9)
    ax.set_ylabel("Count", fontsize=9); ax.legend(fontsize=8)
    plt.tight_layout()
    _save_show(path, show)

# ============================================================================
# PART L — SAVE OUTPUTS
# ============================================================================

def save_outputs(
    out_dir: Path,
    test_cfg: dict,
    runtime_choices: dict,
    functional: dict,
    bio: dict,
    metrics: dict,
    per_stage: pd.DataFrame,
    per_cycle: pd.DataFrame,
    per_scenario: pd.DataFrame,
    trans_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    results: list[dict],
    save_plots: bool,
    show_plots: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  Saving outputs → {out_dir}")

    def _jdump(obj: dict, fname: str) -> None:
        (out_dir / fname).write_text(json.dumps(obj, indent=2, default=str))

    _jdump(test_cfg,       "test_config.json")
    _jdump(runtime_choices,"runtime_choices.json")
    _jdump(functional,     "functional_test_results.json")
    _jdump(bio,            "logical_consistency_results.json")

    valid = [r for r in results if r["pred"] is not None]

    if "error" not in metrics and valid:
        _jdump(metrics, "test_summary.json")

        si_now   = [r["gt"]["stage_now"]     for r in valid]
        si24_t   = [r["gt"]["stage_24h"]     for r in valid]
        si48_t   = [r["gt"]["stage_48h"]     for r in valid]
        si24_p   = [r["pred"]["stage_index_24h"] for r in valid]
        si48_p   = [r["pred"]["stage_index_48h"] for r in valid]
        prog24_t = [r["gt"]["progress_24h"]  for r in valid]
        prog24_p = [r["pred"]["progress_24h"] for r in valid]
        prog48_t = [r["gt"]["progress_48h"]  for r in valid]
        prog48_p = [r["pred"]["progress_48h"] for r in valid]
        h2n_t    = [r["gt"]["hours_to_next"] for r in valid]
        h2n_p    = [r["pred"]["hours_to_next"] for r in valid]

        if "stage_24h" in metrics:
            cm24 = np.array(metrics["stage_24h"]["confusion_matrix"])
            pd.DataFrame(cm24, index=STAGE_LABELS, columns=STAGE_LABELS).to_csv(
                out_dir / "confusion_matrix_24h.csv")
            if save_plots:
                plot_confusion_matrix(cm24.tolist(), "Confusion Matrix — Stage +24h",
                                      out_dir / "confusion_matrix_24h.png", show_plots)

        if "stage_48h" in metrics:
            cm48 = np.array(metrics["stage_48h"]["confusion_matrix"])
            pd.DataFrame(cm48, index=STAGE_LABELS, columns=STAGE_LABELS).to_csv(
                out_dir / "confusion_matrix_48h.csv")
            if save_plots:
                plot_confusion_matrix(cm48.tolist(), "Confusion Matrix — Stage +48h",
                                      out_dir / "confusion_matrix_48h.png", show_plots)

        # Regression metrics CSV
        reg_rows = [m for k, m in metrics.items()
                    if k.startswith("progress_") and isinstance(m, dict)]
        if reg_rows:
            pd.DataFrame(reg_rows).to_csv(out_dir / "regression_metrics.csv", index=False)

        if save_plots:
            prog_errs = [p - a for p, a in zip(prog24_p, prog24_t)]
            h2n_errs  = [p - a for p, a in zip(h2n_p, h2n_t)]
            if prog_errs:
                plot_error_distribution(prog_errs, "Stage Progress Error +24h",
                    "Predicted − Actual (%)",
                    out_dir / "error_distribution_progress.png", show_plots)
            if h2n_errs:
                plot_error_distribution(h2n_errs, "Hours-to-Next Error",
                    "Predicted − Actual (h)",
                    out_dir / "error_distribution_time_to_next_stage.png", show_plots)
            if prog24_t and prog24_p:
                plot_actual_vs_predicted(prog24_t, prog24_p,
                    "Stage Progress: Actual vs Predicted (+24h)", "progress (%)",
                    out_dir / "progress_prediction_examples.png", show_plots)
            if h2n_t and h2n_p:
                plot_actual_vs_predicted(h2n_t, h2n_p,
                    "Hours-to-Next: Actual vs Predicted", "hours",
                    out_dir / "time_to_next_stage_examples.png", show_plots)

    if save_plots:
        plot_stage_trajectories(results, out_dir, n=4, show=show_plots)
        if not trans_df.empty:
            plot_transition_timing(trans_df,
                                   out_dir / "transition_timing.png", show_plots)
        if not per_scenario.empty:
            plot_scenario_comparison(per_scenario,
                                     out_dir / "scenario_comparison.png", show_plots)

    # CSV outputs
    for df_out, fname in [
        (per_stage,    "per_stage_metrics.csv"),
        (per_cycle,    "per_cycle_metrics.csv"),
        (per_scenario, "per_scenario_metrics.csv"),
        (trans_df,     "transition_analysis.csv"),
        (samples_df,   "actual_vs_predicted.csv"),
    ]:
        if not df_out.empty:
            df_out.to_csv(out_dir / fname, index=False)

    if not samples_df.empty:
        samples_df.head(50).to_csv(out_dir / "prediction_samples.csv", index=False)

    # Violation examples
    viol_rows = []
    for vtype, items in bio.get("violation_examples", {}).items():
        for item in items:
            viol_rows.append({"violation_type": vtype, **item})
    if viol_rows:
        pd.DataFrame(viol_rows).to_csv(out_dir / "violation_examples.csv", index=False)

    # Original vs generated comparison
    ov_rows = []
    for label in ("original", "generated"):
        sub = [r for r in valid if r["gt"].get("cycle_origin_type") == label]
        if not sub:
            continue
        si24_t = [r["gt"]["stage_24h"]          for r in sub]
        si24_p = [r["pred"]["stage_index_24h"]   for r in sub]
        ov_rows.append({
            "origin_type"  : label,
            "n"            : len(sub),
            "acc_24h"      : round(accuracy_score(si24_t, si24_p), 4),
            "f1_24h"       : round(f1_score(si24_t, si24_p, average="weighted",
                                   zero_division=0), 4),
            "ord_dist_24h" : round(_ord_dist(si24_t, si24_p), 4),
        })
    if ov_rows:
        pd.DataFrame(ov_rows).to_csv(
            out_dir / "original_vs_generated_metrics.csv", index=False)

    n_files = sum(1 for _ in out_dir.iterdir())
    print(f"  Saved {n_files} file(s) → {out_dir.name}/")

# ============================================================================
# PART M — FINAL SUMMARY, ARGPARSE, MAIN
# ============================================================================

def print_final_summary(
    out_dir: Path,
    test_cfg: dict,
    runtime_choices: dict,
    functional: dict,
    bio: dict,
    metrics: dict,
    n_valid: int,
) -> None:
    print(f"\n{DIV}")
    print(f"  AgriTwin-GH :: TFT Growth Progression — Test Complete")
    print(DIV)
    print(f"  Model          : {test_cfg['run_id']}")
    print(f"  Artifact folder: {test_cfg['artifact_dir']}")
    print(f"  Test mode      : {test_cfg['mode_name']}")
    print(f"  Scenario       : {test_cfg['scenario_name']}")
    print(f"  Forecast horiz : {runtime_choices.get('horizon', 'both')}")
    print(f"  Encoder length : {runtime_choices.get('enc_len', '?')}h")
    print(f"  Cycles tested  : {test_cfg.get('cycles')}")
    print(f"  Windows tested : {n_valid}")
    print()
    print(f"  Functional     : {'PASS' if functional.get('all_passed') else 'PARTIAL'}"
          f"  ({functional.get('checks_passed', '?')}/{functional.get('total_checks', '?')} checks)")
    print(f"  Biological     : {bio.get('biological_pass_rate_pct', 'N/A')}% pass"
          f"  ({bio.get('total_violations', 'N/A')} violations)")
    if "error" not in metrics:
        if "stage_24h" in metrics:
            m = metrics["stage_24h"]
            print(f"  Stage +24h     : acc={m.get('accuracy')}  "
                  f"F1w={m.get('f1_weighted')}  "
                  f"ord={m.get('ordinal_distance')}")
        if "stage_48h" in metrics:
            m = metrics["stage_48h"]
            print(f"  Stage +48h     : acc={m.get('accuracy')}  "
                  f"F1w={m.get('f1_weighted')}  "
                  f"ord={m.get('ordinal_distance')}")
        if "progress_24h" in metrics:
            m = metrics["progress_24h"]
            print(f"  Progress +24h  : MAE={m.get('mae')}%  "
                  f"RMSE={m.get('rmse')}%  R²={m.get('r2')}")
        if "hours_to_next" in metrics:
            m = metrics["hours_to_next"]
            print(f"  Hrs-to-next    : MAE={m.get('mae')}h  "
                  f"within12h={m.get('pct_within_12h')}%  "
                  f"bias={m.get('mean_bias_h'):+}h")
        acc24     = float(metrics.get("stage_24h", {}).get("accuracy") or 0)
        pass_rate = float(bio.get("biological_pass_rate_pct") or 0)
        deploy_ok = (functional.get("all_passed", False)
                     and acc24 >= 0.5 and pass_rate >= 80)
        verdict   = "DEPLOY-READY" if deploy_ok else "NEEDS REVIEW"
        print(f"\n  Verdict        : {verdict}")
    print(f"  Output folder  : {out_dir}")
    print(DIV)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AgriTwin-GH — Interactive TFT Growth Progression Runtime Test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
  python scripts/test_growth_progression_runtime.py
  python scripts/test_growth_progression_runtime.py --mode 4 --scenario 8 --n-windows 50
  python scripts/test_growth_progression_runtime.py \\
      --run-id growth_progression_20260308_202542 \\
      --mode 3 --scenario 2 --n-windows 25 --horizon both --save-plots
        """,
    )
    p.add_argument("--run-id",    default=None,
                   help="Artifact run_id to use (default: auto-discover latest)")
    p.add_argument("--mode",      type=int, default=None, choices=range(1, 7),
                   help="Test mode 1-6 (default: interactive)")
    p.add_argument("--scenario",  type=int, default=None, choices=range(1, 12),
                   help="Scenario ID 1-11 (default: interactive)")
    p.add_argument("--n-windows", type=int, default=None,
                   help="Windows to test (default: interactive)")
    p.add_argument("--enc-len",   type=int, default=None, choices=[48, 72, 96],
                   help="Encoder length in hours (default: 72)")
    p.add_argument("--horizon",   default=None, choices=["24h", "48h", "both"],
                   help="Forecast horizon (default: both)")
    p.add_argument("--cycles",    type=int, nargs="+", default=None,
                   help="Cycle IDs to test (default: held-out test cycles)")
    p.add_argument("--save-plots",action="store_true", help="Save plots as PNG")
    p.add_argument("--show-plots",action="store_true", help="Show plots interactively")
    p.add_argument("--no-plots",  action="store_true", help="Disable all plots")
    p.add_argument("--test-id",   default=None,
                   help="Test run ID (default: timestamp + uuid)")
    p.add_argument("--non-interactive", action="store_true",
                   help="Skip all interactive prompts (requires --mode, etc.)")
    return p.parse_args()


def _resolve_cfg_from_args(args: argparse.Namespace, arts: dict, test_mode: int) -> dict:
    """Build runtime config entirely from CLI args (non-interactive path)."""
    enc_len = min(args.enc_len or arts["enc_len"], arts["enc_len"])
    return dict(
        enc_len     = enc_len,
        horizon     = args.horizon  or "both",
        n_windows   = args.n_windows or 50,
        show_plots  = args.show_plots and not args.no_plots,
        save_plots  = (args.save_plots or True) and not args.no_plots,
        print_table = False,
        bio_strict  = True,
    )


def main() -> None:
    args    = parse_args()
    test_id = args.test_id or datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]

    _hdr(f"AgriTwin-GH :: TFT Growth Progression — Runtime Test\n"
         f"  Test ID  : {test_id}\n"
         f"  Started  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ── B: Model selection ───────────────────────────────────────────────────
    if args.non_interactive or args.run_id:
        candidates = _find_artifact_dirs()
        if args.run_id:
            matched = [d for d in candidates if args.run_id in d.name]
            art_dir = matched[0] if matched else None
        else:
            art_dir = candidates[0] if candidates else None
        if art_dir is None:
            sys.exit(f"[ERROR] No artifact directories found in {ARTIFACT_BASE}")
        print(f"\n  Artifact: {art_dir.name}")
    else:
        art_dir = select_model_interactive()

    arts   = load_artifacts(art_dir)
    run_id = arts["run_id"]

    # ── C: Test data source ──────────────────────────────────────────────────
    if args.non_interactive:
        print("\n  [Data] Loading artifact dataset ...")
        df      = _load_artifact_df(art_dir)
        missing = _validate_columns(df)
        if missing:
            print(f"  [!] Missing columns: {missing}")
    else:
        df = select_data_source_interactive(arts)

    # ── D: Test mode ─────────────────────────────────────────────────────────
    if args.mode is not None:
        test_mode = args.mode
    elif args.non_interactive:
        test_mode = 4
    else:
        test_mode = select_test_mode_interactive()
    mode_name = TEST_MODE_LABELS[test_mode - 1]

    # ── E: Scenario ──────────────────────────────────────────────────────────
    if args.scenario is not None:
        scenario_id = args.scenario
    elif args.non_interactive or test_mode in (1, 2, 5):
        scenario_id = 8
    else:
        scenario_id = select_scenario_interactive()
    scenario_name = SCENARIO_DEFS[scenario_id]["name"]

    # ── F: Runtime config ────────────────────────────────────────────────────
    if args.non_interactive:
        cfg = _resolve_cfg_from_args(args, arts, test_mode)
    else:
        cfg = select_runtime_config_interactive(arts, test_mode)
        # CLI overrides for hybrid usage
        if args.enc_len:
            cfg["enc_len"] = min(args.enc_len, arts["enc_len"])
        if args.horizon:
            cfg["horizon"] = args.horizon
        if args.n_windows:
            cfg["n_windows"] = args.n_windows
        if args.save_plots:
            cfg["save_plots"] = True
        if args.show_plots:
            cfg["show_plots"] = True
        if args.no_plots:
            cfg["save_plots"] = cfg["show_plots"] = False

    # ── Cycles ────────────────────────────────────────────────────────────────
    available = sorted(df["cycle_id"].unique().tolist())
    if args.cycles:
        cycles = [c for c in args.cycles if c in available]
    else:
        cycles = [c for c in arts["test_cycles"] if c in available]
    if not cycles:
        print(f"  [!] Requested cycles not in dataset — using first 3 available: {available[:3]}")
        cycles = available[:3]

    # ── Output dir ────────────────────────────────────────────────────────────
    out_dir = art_dir / f"test_run_{test_id}"

    # ── Assemble config records ───────────────────────────────────────────────
    test_cfg: dict = {
        "test_id"      : test_id,
        "run_id"       : run_id,
        "mode"         : test_mode,
        "mode_name"    : mode_name,
        "scenario_id"  : scenario_id,
        "scenario_name": scenario_name,
        "cycles"       : cycles,
        "enc_len"      : cfg["enc_len"],
        "pred_len"     : arts["pred_len"],
        "horizon"      : cfg["horizon"],
        "n_windows_req": cfg["n_windows"],
        "artifact_dir" : str(art_dir),
        "output_dir"   : str(out_dir),
        "started_at"   : datetime.now().isoformat(),
    }
    runtime_choices: dict = {
        "enc_len"    : cfg["enc_len"],
        "horizon"    : cfg["horizon"],
        "n_windows"  : cfg["n_windows"],
        "show_plots" : cfg["show_plots"],
        "save_plots" : cfg["save_plots"],
        "print_table": cfg["print_table"],
        "bio_strict" : cfg["bio_strict"],
        "scenario_id": scenario_id,
        "cycles"     : cycles,
    }

    # ============================================================
    # MODE RUNNERS
    # ============================================================
    _sec(f"RUNNING: {mode_name}")
    print(f"  Scenario : {scenario_name}")
    print(f"  Cycles   : {cycles}")
    print(f"  Enc len  : {cfg['enc_len']}h  |  Horizon: {cfg['horizon']}")
    print(f"  Max wins : {cfg['n_windows']}")

    # Shared empties for modes that skip some outputs
    _empty_df = pd.DataFrame()

    # ── MODE 1: Functional sanity test ───────────────────────────────────────
    if test_mode == 1:
        windows    = extract_windows(df, cfg["enc_len"], cycles,
                                     max_windows=5, scenario_id=scenario_id)
        functional = run_functional_tests(arts, windows)
        results    = run_batch_inference(windows[:3], arts, "functional") if windows else []
        n_valid    = sum(1 for r in results if r["pred"] is not None)
        bio        = run_biological_checks(results)
        metrics    = {}
        per_stage = per_cycle = per_scenario = trans_df = samples_df = _empty_df

    # ── MODE 2: Single window prediction ─────────────────────────────────────
    elif test_mode == 2:
        windows = extract_windows(df, cfg["enc_len"], cycles,
                                  max_windows=50, scenario_id=scenario_id)
        if not windows:
            sys.exit("[ERROR] No windows matched the selected scenario and cycles.")

        print(f"\n  {len(windows)} matching windows available.")
        if args.non_interactive:
            win_idx = 1
        else:
            win_idx = _prompt_int(f"Select a window (1–{len(windows)})",
                                  default=1, lo=1, hi=len(windows))

        selected   = [windows[win_idx - 1]]
        functional = run_functional_tests(arts, selected)
        results    = run_batch_inference(selected, arts, "single")
        n_valid    = sum(1 for r in results if r["pred"] is not None)
        print_single_prediction(results[0], idx=1)
        bio        = run_biological_checks(results)
        metrics    = {"error": "Single-window run — full metrics require multiple samples"}
        per_stage = per_cycle = per_scenario = trans_df = _empty_df
        samples_df = build_prediction_samples(results, n=1)

    # ── MODE 3: Batch scenario testing ───────────────────────────────────────
    elif test_mode == 3:
        windows = extract_windows(df, cfg["enc_len"], cycles,
                                  max_windows=cfg["n_windows"], scenario_id=scenario_id)
        if not windows:
            sys.exit("[ERROR] No windows for selected scenario.")
        functional   = run_functional_tests(arts, windows)
        results      = run_batch_inference(windows, arts, "batch")
        n_valid      = sum(1 for r in results if r["pred"] is not None)
        if cfg["print_table"]:
            print_prediction_table(results)
        bio          = run_biological_checks(results)
        metrics      = compute_all_metrics(results, cfg["horizon"])
        per_stage    = compute_per_stage_metrics(results)
        per_cycle    = compute_per_cycle_metrics(results)
        per_scenario = compute_per_scenario_metrics(results)
        trans_df     = compute_transition_analysis(results)
        samples_df   = build_prediction_samples(results)
        _sec("Batch Scenario Metrics")
        print_metrics_block(metrics, bio, cfg["horizon"])

    # ── MODE 4: Full evaluation ──────────────────────────────────────────────
    elif test_mode == 4:
        windows = extract_windows(df, cfg["enc_len"], cycles,
                                  max_windows=cfg["n_windows"], scenario_id=scenario_id)
        if not windows:
            sys.exit("[ERROR] No windows available.")
        functional   = run_functional_tests(arts, windows)
        results      = run_batch_inference(windows, arts, "full")
        n_valid      = sum(1 for r in results if r["pred"] is not None)
        if cfg["print_table"]:
            print_prediction_table(results)
        bio          = run_biological_checks(results)
        metrics      = compute_all_metrics(results, cfg["horizon"])
        per_stage    = compute_per_stage_metrics(results)
        per_cycle    = compute_per_cycle_metrics(results)
        per_scenario = compute_per_scenario_metrics(results)
        trans_df     = compute_transition_analysis(results)
        samples_df   = build_prediction_samples(results)
        _sec("Full Evaluation — Results")
        print_metrics_block(metrics, bio, cfg["horizon"])
        if not per_stage.empty:
            print(f"\n  Per-Stage Accuracy (+24h):")
            for _, row in per_stage.iterrows():
                if row["n"] > 0:
                    print(f"    {row['stage']:<25}  n={int(row['n']):>4}  "
                          f"acc={row.get('acc_24h', 'N/A')}  "
                          f"F1w={row.get('f1_weighted_24h', 'N/A')}")
        if not per_cycle.empty:
            print(f"\n  Per-Cycle Accuracy (+24h):")
            for _, row in per_cycle.iterrows():
                print(f"    Cycle {row['cycle_id']} ({row['origin_type']})  "
                      f"n={int(row['n'])}  "
                      f"acc={row.get('acc_24h', 'N/A')}")

    # ── MODE 5: Biological consistency only ──────────────────────────────────
    elif test_mode == 5:
        windows = extract_windows(df, cfg["enc_len"], cycles,
                                  max_windows=cfg["n_windows"], scenario_id=scenario_id)
        if not windows:
            sys.exit("[ERROR] No windows available.")
        functional   = run_functional_tests(arts, windows)
        results      = run_batch_inference(windows, arts, "bio")
        n_valid      = sum(1 for r in results if r["pred"] is not None)
        bio          = run_biological_checks(results)
        metrics      = {}
        per_stage = per_cycle = per_scenario = trans_df = _empty_df
        samples_df   = build_prediction_samples(results, n=50)

    # ── MODE 6: Stage transition focused ─────────────────────────────────────
    elif test_mode == 6:
        print("  Extracting near-stage-boundary (transition) windows ...")
        windows = extract_transition_windows(df, cfg["enc_len"], cycles,
                                             max_windows=cfg["n_windows"])
        if not windows:
            print("  [!] No transition windows found — falling back to approaching-transition scenario.")
            windows = extract_windows(df, cfg["enc_len"], cycles,
                                      max_windows=cfg["n_windows"], scenario_id=2)
        functional   = run_functional_tests(arts, windows)
        results      = run_batch_inference(windows, arts, "trans")
        n_valid      = sum(1 for r in results if r["pred"] is not None)
        if cfg["print_table"]:
            print_prediction_table(results)
        bio          = run_biological_checks(results)
        metrics      = compute_all_metrics(results, cfg["horizon"])
        per_stage    = compute_per_stage_metrics(results)
        per_cycle    = compute_per_cycle_metrics(results)
        per_scenario = _empty_df
        trans_df     = compute_transition_analysis(results)
        samples_df   = build_prediction_samples(results)
        _sec("Transition Analysis")
        if not trans_df.empty:
            near24 = trans_df[trans_df["actual_trans_24h"]]
            if len(near24) > 0:
                det_rate = near24["correct_24h"].mean()
                h2n_bias = near24["h2n_error_h"].mean()
                print(f"  Expected transitions within 24h : {len(near24)}")
                print(f"    Detection accuracy : {det_rate:.2%}")
                print(f"    Timing bias        : {h2n_bias:+.1f}h")
                skips = near24[near24["stage_skip"] > 0]
                if len(skips) > 0:
                    print(f"    Stage skips (pred) : {len(skips)}")
        print_metrics_block(metrics, bio, cfg["horizon"])

    else:
        sys.exit(f"[ERROR] Unknown test mode: {test_mode}")

    # ── Save all outputs ──────────────────────────────────────────────────────
    save_outputs(
        out_dir        = out_dir,
        test_cfg       = test_cfg,
        runtime_choices= runtime_choices,
        functional     = functional,
        bio            = bio,
        metrics        = metrics,
        per_stage      = per_stage,
        per_cycle      = per_cycle,
        per_scenario   = per_scenario,
        trans_df       = trans_df,
        samples_df     = samples_df,
        results        = results,
        save_plots     = cfg["save_plots"],
        show_plots     = cfg["show_plots"],
    )

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(
        out_dir        = out_dir,
        test_cfg       = test_cfg,
        runtime_choices= runtime_choices,
        functional     = functional,
        bio            = bio,
        metrics        = metrics,
        n_valid        = n_valid,
    )


if __name__ == "__main__":
    main()
