#!/usr/bin/env python3
"""
test_growth_progression_model.py

Real-time growth-progression model tester for AgriTwin-GH.

Simulates a live greenhouse sensor stream at hourly resolution, maintains a
72-hour rolling feature buffer, and runs inference with the best saved
growth-progression model (LSTM or RF).

The feature engineering pipeline (rolling stats, lags, stage history,
cumulative exposure, interaction terms, day/night segmented features)
exactly mirrors Sections G1-G7 of growth_progression.ipynb.

Usage:
  # Fast simulation (default: 0.5 s per tick)
  python scripts/test_growth_model_realtime.py

  # Starting at the 'flowering' stage, 0.1 s per tick
  python scripts/test_growth_model_realtime.py --stage flowering --tick-secs 0.1

  # Force LSTM, run 200 ticks, save log
  python scripts/test_growth_model_realtime.py --model lstm --max-ticks 200 --log-json logs/realtime_test.json

  # Use a specific saved run
  python scripts/test_growth_model_realtime.py --run-id growth_progression_20260306_212928

Arguments:
  --tick-secs  Wall-clock seconds between ticks (default: 0.5)
  --stage      Starting growth stage            (default: seedling)
  --max-ticks  Stop after N ticks; 0 = unlimited (default: 500)
  --model      auto | rf | lstm                 (default: auto)
  --log-json   Path to save per-tick JSON log   (default: none)
  --run-id     Specific artifact run ID to load (default: latest)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
LOG = logging.getLogger("gp_realtime")

# Suppress TensorFlow noise unless debugging
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# ---------------------------------------------------------------------------
# Repo structure
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[1]
MODELS_DIR    = ROOT / "src" / "agritwin_gh" / "models"
ARTIFACTS_BASE = MODELS_DIR / "artifacts"

# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------
STAGE_ORDER = [
    "seedling",
    "early_veg",
    "flowering_initiation",
    "flowering",
    "unripe",
    "ripe",
]

STAGE_TO_INT = {s: i for i, s in enumerate(STAGE_ORDER)}

# ---------------------------------------------------------------------------
# Biological thresholds — mirror notebook G4
# ---------------------------------------------------------------------------
TEMP_FAV_MIN = 18.0   # °C
TEMP_FAV_MAX = 26.0   # °C
VPD_OPT_MIN  =  0.4   # kPa
VPD_OPT_MAX  =  1.2   # kPa
VPD_STRESS   =  1.5   # kPa

# Sensors used for rolling and lag features (7 continuous, excluding binary cols)
_CONTINUOUS_SENSORS = [
    "temperature", "humidity", "air_velocity",
    "co2", "solar_radiation", "vpd", "dew_point",
]
_ROLLING_WINDOWS = [6, 12, 24, 72]
_ROLLING_STATS   = ["mean", "max", "min", "std"]
_LAG_STEPS       = [1, 2, 3, 6, 12, 24]

# ---------------------------------------------------------------------------
# Per-stage simulation parameters — realistic Dindigul, Tamil Nadu conditions
# ---------------------------------------------------------------------------
_STAGE_SIM = {
    "seedling": {
        "temp_day": (22.0, 26.0), "temp_night": (18.0, 22.0),
        "humidity": (68.0, 80.0), "co2": (400, 520),
        "solar_peak": (250, 500), "air_vel": (0.10, 0.35),
    },
    "early_veg": {
        "temp_day": (21.0, 26.0), "temp_night": (17.0, 21.0),
        "humidity": (62.0, 76.0), "co2": (420, 600),
        "solar_peak": (300, 550), "air_vel": (0.10, 0.40),
    },
    "flowering_initiation": {
        "temp_day": (19.0, 25.0), "temp_night": (15.0, 20.0),
        "humidity": (58.0, 72.0), "co2": (450, 650),
        "solar_peak": (350, 650), "air_vel": (0.15, 0.45),
    },
    "flowering": {
        "temp_day": (18.0, 24.0), "temp_night": (14.0, 19.0),
        "humidity": (54.0, 68.0), "co2": (480, 700),
        "solar_peak": (350, 700), "air_vel": (0.15, 0.50),
    },
    "unripe": {
        "temp_day": (20.0, 27.0), "temp_night": (16.0, 22.0),
        "humidity": (52.0, 66.0), "co2": (430, 650),
        "solar_peak": (300, 650), "air_vel": (0.10, 0.45),
    },
    "ripe": {
        "temp_day": (22.0, 30.0), "temp_night": (18.0, 24.0),
        "humidity": (48.0, 62.0), "co2": (380, 520),
        "solar_peak": (350, 600), "air_vel": (0.10, 0.40),
    },
}

# =============================================================================
# Physics helpers
# =============================================================================

def _vpd(T: float, RH: float) -> float:
    """Vapour-pressure deficit in kPa (Magnus formula)."""
    es = 0.6108 * math.exp(17.27 * T / (T + 237.3))
    return max(0.0, round(es * (1.0 - RH / 100.0), 4))


def _dewpoint(T: float, RH: float) -> float:
    """Dew-point temperature in °C (Magnus formula)."""
    alpha = math.log(max(RH / 100.0, 1e-9)) + 17.625 * T / (243.04 + T)
    return round(243.04 * alpha / (17.625 - alpha), 3)


# =============================================================================
# Sensor simulation
# =============================================================================

def simulate_reading(ts: datetime, stage: str, rng: random.Random) -> dict[str, float]:
    """
    Generate one synthetic hourly sensor reading for *ts* and *stage*.

    Derived quantities (VPD, dew_point) are computed from fundamental
    measurements so they are physically consistent.
    """
    p = _STAGE_SIM.get(stage, _STAGE_SIM["seedling"])
    hour = ts.hour
    is_day = 6 <= hour < 18
    day_night_flag = 1 if is_day else 0

    # Temperature — sinusoidal day/night profile with small Gaussian noise
    if is_day:
        lo, hi = p["temp_day"]
        phase = math.sin(math.pi * (hour - 6) / 12.0)
    else:
        lo, hi = p["temp_night"]
        phase = 0.4
    temp = lo + (hi - lo) * phase + rng.gauss(0.0, 0.3)
    temp = round(max(p["temp_night"][0] - 1.0, min(p["temp_day"][1] + 2.0, temp)), 2)

    # Humidity — inversely correlated with temperature during the day
    hum_lo, hum_hi = p["humidity"]
    humidity = hum_hi - (hum_hi - hum_lo) * phase + rng.gauss(0.0, 1.0)
    humidity = round(max(30.0, min(99.0, humidity)), 1)

    # CO2 — stable with slight nighttime elevation
    co2_lo, co2_hi = p["co2"]
    co2 = round(rng.uniform(co2_lo, co2_hi) + (30.0 if not is_day else 0.0), 1)

    # Solar radiation — sinusoidal daytime curve
    if is_day:
        sol_lo, sol_hi = p["solar_peak"]
        solar = max(0.0, (sol_lo + (sol_hi - sol_lo) * phase) * rng.uniform(0.70, 1.05))
    else:
        solar = 0.0
    solar = round(solar, 2)

    # Air velocity — uniform random draw
    av_lo, av_hi = p["air_vel"]
    air_vel = round(rng.uniform(av_lo, av_hi), 3)

    # Derived
    vpd       = _vpd(temp, humidity)
    dew_point = _dewpoint(temp, humidity)

    return {
        "temperature":    temp,
        "humidity":       humidity,
        "air_velocity":   air_vel,
        "co2":            co2,
        "solar_radiation": solar,
        "day_night_flag": day_night_flag,
        "vpd":            vpd,
        "dew_point":      dew_point,
    }


# =============================================================================
# Feature engineering (replicates growth_progression.ipynb G1-G7)
# =============================================================================

def engineer_features(buf: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the full feature engineering pipeline from notebook Sections G1-G7
    to a buffer DataFrame.

    Parameters
    ----------
    buf : pd.DataFrame
        DatetimeIndex named 'datetime'.  Must contain columns:
        temperature, humidity, air_velocity, co2, solar_radiation,
        day_night_flag, vpd, dew_point, leaf_wetness_proxy,
        growth_stage (str), stage_int (int).

    Returns
    -------
    pd.DataFrame
        Same rows as *buf* with all engineered feature columns appended.
        The last row corresponds to the current tick.

    Notes
    -----
    Cumulative exposure features (G4) accumulate from the buffer start,
    not from the historical dataset start.  This is an acceptable
    approximation for real-time inference; in production, persist the
    running totals between calls.
    """
    df = buf.copy()
    # All engineered columns are accumulated in a dict to avoid DataFrame
    # fragmentation (pandas PerformanceWarning when inserting columns one at a time).
    new_cols: dict[str, Any] = {}

    # ── G1: Rolling statistics ────────────────────────────────────────────────
    # 7 continuous sensors × 4 windows × 4 stats = 112 features
    for col in _CONTINUOUS_SENSORS:
        for w in _ROLLING_WINDOWS:
            roll_w = df[col].rolling(w, min_periods=1)
            for stat in _ROLLING_STATS:
                new_cols[f"{col}_rolling_{stat}_{w}h"] = getattr(roll_w, stat)()

    # ── G2: Lag features ──────────────────────────────────────────────────────
    # 7 sensors × 6 lags = 42 features
    for col in _CONTINUOUS_SENSORS:
        for lag in _LAG_STEPS:
            new_cols[f"{col}_lag_{lag}h"] = df[col].shift(lag)

    # ── G3: Stage-history features ────────────────────────────────────────────
    stage_change_flag = (df["stage_int"] != df["stage_int"].shift(1)).astype(int)
    if len(stage_change_flag) > 0:
        stage_change_flag.iloc[0] = 1  # first row always marks a stage start
    stage_run_id = stage_change_flag.cumsum()

    run_start_map = (
        df.assign(stage_run_id=stage_run_id)
        .reset_index()
        .groupby("stage_run_id")["datetime"]
        .first()
    )
    stage_run_start = stage_run_id.map(run_start_map)
    stage_age_hours = (
        df.index.to_series() - pd.Series(stage_run_start.values, index=df.index)
    ).dt.total_seconds() / 3600.0

    new_cols["current_stage_encoded"] = df["stage_int"].values
    new_cols["stage_change_flag"]     = stage_change_flag.values
    new_cols["stage_run_id"]          = stage_run_id.values
    new_cols["stage_age_hours"]       = stage_age_hours.values

    # ── Merge G1-G3 into df now (G4 needs stage_run_id in df) ─────────────────
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    new_cols = {}

    # ── G4: Cumulative exposure features ──────────────────────────────────────
    fav_temp  = ((df["temperature"] >= TEMP_FAV_MIN) &
                 (df["temperature"] <= TEMP_FAV_MAX)).astype(float)
    vpd_opt   = ((df["vpd"] >= VPD_OPT_MIN) &
                 (df["vpd"] <= VPD_OPT_MAX)).astype(float)
    vpd_str   = (df["vpd"] > VPD_STRESS).astype(float)
    night_msk = (df["day_night_flag"] == 0).astype(float)

    # Global accumulation (from buffer start)
    new_cols["cumulative_solar_exposure"]          = df["solar_radiation"].cumsum()
    new_cols["cumulative_favorable_temp_hours"]    = fav_temp.cumsum()
    new_cols["cumulative_vpd_optimal_hours"]       = vpd_opt.cumsum()
    new_cols["cumulative_vpd_stress_hours"]        = vpd_str.cumsum()
    new_cols["cumulative_night_humidity_exposure"] = (df["humidity"] * night_msk).cumsum()

    # Within-stage accumulation (resets at each stage transition via groupby)
    for out_col, src in [
        ("stage_solar_exposure",       df["solar_radiation"]),
        ("stage_favorable_temp_hours", fav_temp),
        ("stage_vpd_stress_hours",     vpd_str),
        ("stage_vpd_optimal_hours",    vpd_opt),
    ]:
        tmp_s = pd.Series(src.values, index=df.index, name="_tmp")
        new_cols[out_col] = df.groupby("stage_run_id")["stage_run_id"].transform(
            lambda _: None  # placeholder
        )  # replaced below
        # Correct within-stage cumsum via concat-based groupby
        new_cols[out_col] = (
            pd.concat([df["stage_run_id"].rename("srid"), tmp_s], axis=1)
            .groupby("srid")["_tmp"]
            .cumsum()
            .values
        )

    # ── G5: Interaction terms ─────────────────────────────────────────────────
    new_cols["dewpoint_spread"]         = df["temperature"].values - df["dew_point"].values
    new_cols["temperature_x_humidity"]  = df["temperature"].values * df["humidity"].values
    new_cols["temperature_x_vpd"]       = df["temperature"].values * df["vpd"].values
    new_cols["humidity_x_vpd"]          = df["humidity"].values    * df["vpd"].values
    new_cols["radiation_x_temperature"] = df["solar_radiation"].values * df["temperature"].values
    new_cols["radiation_x_vpd"]         = df["solar_radiation"].values * df["vpd"].values

    # ── G6: Day/night segmented rolling features ──────────────────────────────
    _DN_W  = 24
    day_s  = lambda col: df[col].where(df["day_night_flag"] == 1)
    nght_s = lambda col: df[col].where(df["day_night_flag"] == 0)

    day_temp_mean  = day_s("temperature").rolling(_DN_W, min_periods=1).mean()
    nght_temp_mean = nght_s("temperature").rolling(_DN_W, min_periods=1).mean()
    day_hum_mean   = day_s("humidity").rolling(_DN_W, min_periods=1).mean()
    nght_hum_mean  = nght_s("humidity").rolling(_DN_W, min_periods=1).mean()

    new_cols["day_temperature_mean_24h"]      = day_temp_mean
    new_cols["night_temperature_mean_24h"]    = nght_temp_mean
    new_cols["day_humidity_mean_24h"]         = day_hum_mean
    new_cols["night_humidity_mean_24h"]       = nght_hum_mean
    new_cols["day_vpd_mean_24h"]              = day_s("vpd").rolling(_DN_W, min_periods=1).mean()
    new_cols["night_vpd_mean_24h"]            = nght_s("vpd").rolling(_DN_W, min_periods=1).mean()
    new_cols["day_solar_mean_24h"]            = day_s("solar_radiation").rolling(_DN_W, min_periods=1).mean()
    new_cols["day_co2_mean_24h"]              = day_s("co2").rolling(_DN_W, min_periods=1).mean()
    new_cols["night_co2_mean_24h"]            = nght_s("co2").rolling(_DN_W, min_periods=1).mean()
    new_cols["night_humidity_std_24h"]        = nght_s("humidity").rolling(_DN_W, min_periods=2).std()
    new_cols["diurnal_temperature_range_24h"] = day_temp_mean - nght_temp_mean
    new_cols["diurnal_humidity_range_24h"]    = day_hum_mean  - nght_hum_mean

    # ── Final merge ───────────────────────────────────────────────────────────
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


# =============================================================================
# Artifact loading
# =============================================================================

def load_artifacts(run_dir: Path, load_lstm: bool = True) -> dict[str, Any]:
    """
    Load all model artifacts from *run_dir* and return them in a flat dict.

    Parameters
    ----------
    run_dir :
        Path to the artifacts sub-directory for this run.
    load_lstm :
        Set to False to skip loading the Keras LSTM model when only RF
        inference is needed (avoids the slow TensorFlow initialisation).

    Loaded:
      feature_schema.json    → feature_cols, window_n, class encoding
      thresholds_config.json → alert_lead_h, repeat_interval_h, confidence_threshold
      best_model_summary.json → best_model_type, run_id, artifact_paths
      lstm_metrics.json      → y_time_mean_h, y_time_std_h (regression normalisation)
      scaler_*.pkl           → fitted StandardScaler
      label_encoder_stage_*.pkl → fitted LabelEncoder
      growth_progression_lstm_*.keras  (only when load_lstm=True)
      growth_progression_rf_classifier_*.pkl  (if available)
      growth_progression_rf_regressor_*.pkl   (if available)
    """
    # ── Feature schema ────────────────────────────────────────────────────────
    schema = json.loads((run_dir / "feature_schema.json").read_text())
    feature_cols = schema["feature_cols"]
    window_n     = schema["window_N"]
    int_to_stage = {
        int(k): v
        for k, v in schema["label_encoder_classes"]["int_to_stage"].items()
    }
    stage_to_int = schema["label_encoder_classes"]["stage_to_int"]

    # ── Alert thresholds ──────────────────────────────────────────────────────
    thresh = json.loads((run_dir / "thresholds_config.json").read_text())
    alert_lead_h         = thresh["regressor_alert_thresholds"]["alert_lead_h"]
    repeat_interval_h    = thresh["regressor_alert_thresholds"]["repeat_interval_h"]
    confidence_threshold = thresh["classifier_confidence_threshold"]

    # ── Best model summary ────────────────────────────────────────────────────
    best_summary     = json.loads((run_dir / "best_model_summary.json").read_text())
    best_model_type  = best_summary["best_model"]
    run_id           = best_summary["run_id"]

    # ── LSTM regression normalisation constants ───────────────────────────────
    lstm_metrics  = json.loads((run_dir / "lstm_metrics.json").read_text())
    y_time_mean_h = lstm_metrics["normalization"]["y_time_mean_h"]
    y_time_std_h  = lstm_metrics["normalization"]["y_time_std_h"]

    # ── Preprocessing objects ─────────────────────────────────────────────────
    import joblib

    scaler_path    = next(run_dir.glob("scaler_*.pkl"), None)
    label_enc_path = next(run_dir.glob("label_encoder_stage_*.pkl"), None)

    if scaler_path is None:
        raise FileNotFoundError(f"No scaler_*.pkl found in {run_dir}")
    if label_enc_path is None:
        raise FileNotFoundError(f"No label_encoder_stage_*.pkl found in {run_dir}")

    scaler    = joblib.load(scaler_path)
    label_enc = joblib.load(label_enc_path)

    # ── Models ────────────────────────────────────────────────────────────────
    rf_clf_path = MODELS_DIR / f"growth_progression_rf_classifier_{run_id}.pkl"
    rf_reg_path = MODELS_DIR / f"growth_progression_rf_regressor_{run_id}.pkl"
    lstm_path   = MODELS_DIR / f"growth_progression_lstm_{run_id}.keras"

    clf_rf = joblib.load(rf_clf_path) if rf_clf_path.exists() else None
    reg_rf = joblib.load(rf_reg_path) if rf_reg_path.exists() else None

    lstm_model = None
    if load_lstm and lstm_path.exists():
        LOG.info("Loading LSTM model (this may take a moment) ...")
        import keras
        lstm_model = keras.models.load_model(str(lstm_path))
        LOG.info("LSTM model loaded.")
    elif not load_lstm:
        LOG.info("Skipping LSTM model load (RF inference only).")

    LOG.info(
        "Artifacts loaded: run=%s | n_features=%d | window_N=%d | best_model=%s",
        run_id, len(feature_cols), window_n, best_model_type,
    )

    return dict(
        feature_cols=feature_cols,
        window_n=window_n,
        int_to_stage=int_to_stage,
        stage_to_int=stage_to_int,
        alert_lead_h=alert_lead_h,
        repeat_interval_h=repeat_interval_h,
        confidence_threshold=confidence_threshold,
        best_model_type=best_model_type,
        run_id=run_id,
        y_time_mean_h=y_time_mean_h,
        y_time_std_h=y_time_std_h,
        scaler=scaler,
        label_enc=label_enc,
        clf_rf=clf_rf,
        reg_rf=reg_rf,
        lstm_model=lstm_model,
    )


# =============================================================================
# Inference
# =============================================================================

def run_inference(
    buf_feat: pd.DataFrame,
    arts: dict,
    model_type: str,
) -> dict[str, Any]:
    """
    Run growth-stage and time-to-transition inference on the current buffer.

    Parameters
    ----------
    buf_feat :
        Full engineered feature buffer (all computed rows).
        Last row = current observation.
    arts :
        Loaded artifacts dict from load_artifacts().
    model_type :
        "rf"  — tabular Random Forest inference on last row.
        "lstm" — sequence LSTM inference on last window_n rows.

    Returns
    -------
    dict
        stage_pred   : predicted next stage name (str)
        stage_conf   : classifier confidence for that class (float 0-1)
        time_pred_h  : predicted hours until stage transition (float)
        clf_probs    : full class probability vector (list[float])
        raw_reg_out  : raw regressor output before denormalisation (float)
    """
    feature_cols  = arts["feature_cols"]
    scaler        = arts["scaler"]
    int_to_stage  = arts["int_to_stage"]
    window_n      = arts["window_n"]
    y_time_mean_h = arts["y_time_mean_h"]
    y_time_std_h  = arts["y_time_std_h"]

    # Align columns to the exact FEATURE_COLS order used during training.
    # Any column that doesn't exist in the buffer is filled with 0.0.
    feat_df = buf_feat.reindex(columns=feature_cols, fill_value=0.0)

    # NaN handling:
    #   ffill  — propagate the most recent valid value (lag / rolling warmup)
    #   bfill  — fill remaining leading NaN (first rows before rolling kicks in)
    #   fillna — last resort zero-fill
    feat_df = feat_df.ffill().bfill().fillna(0.0)

    # ── LSTM inference ────────────────────────────────────────────────────────
    if model_type == "lstm":
        lstm_model = arts["lstm_model"]

        # Extract the last window_n rows
        tail = feat_df.tail(window_n)

        # Pad with the oldest available row if buffer hasn't filled yet
        if len(tail) < window_n:
            pad_n   = window_n - len(tail)
            pad_row = tail.iloc[[0]].values if len(tail) > 0 else np.zeros((1, len(feature_cols)))
            pad     = np.repeat(pad_row, pad_n, axis=0)
            window_arr = np.vstack([pad, tail.values]).astype(np.float32)
        else:
            window_arr = tail.values.astype(np.float32)  # (window_n, n_feat)

        # Scale: reshape to 2-D, transform, reshape back
        n_feat     = window_arr.shape[1]
        scaled_2d  = scaler.transform(window_arr)          # (window_n, n_feat)
        X_input    = scaled_2d[np.newaxis, :, :]           # (1, window_n, n_feat)

        pred_out = lstm_model.predict(X_input, verbose=0)

        # Dual-output model: [clf_probs (1,6), reg_norm (1,1)]
        if isinstance(pred_out, (list, tuple)) and len(pred_out) == 2:
            clf_probs  = np.asarray(pred_out[0]).ravel()   # (6,)
            reg_norm   = float(np.asarray(pred_out[1]).ravel()[0])
        else:
            # Single-output fallback — treat as classification only
            clf_probs = np.asarray(pred_out).ravel()
            reg_norm  = 0.0

        raw_reg_out = reg_norm
        pred_class  = int(np.argmax(clf_probs))
        stage_pred  = int_to_stage.get(pred_class, f"class_{pred_class}")
        stage_conf  = float(clf_probs[pred_class])
        time_pred_h = max(0.0, reg_norm * y_time_std_h + y_time_mean_h)

    # ── RF inference ──────────────────────────────────────────────────────────
    else:
        clf_rf = arts["clf_rf"]
        reg_rf = arts["reg_rf"]

        X_row    = feat_df.iloc[[-1]].values.astype(np.float64)  # (1, n_feat)
        X_scaled = scaler.transform(X_row)

        # Classifier
        if clf_rf is not None:
            clf_probs_arr = clf_rf.predict_proba(X_scaled)[0]  # (n_classes,)
            pred_class    = int(np.argmax(clf_probs_arr))
            stage_pred    = int_to_stage.get(pred_class, f"class_{pred_class}")
            stage_conf    = float(clf_probs_arr[pred_class])
            clf_probs     = clf_probs_arr.tolist()
        else:
            stage_pred = "unknown"
            stage_conf = 0.0
            clf_probs  = []

        # Regressor (RF predicts raw hours — no denormalisation needed)
        if reg_rf is not None:
            raw_reg_out = float(reg_rf.predict(X_scaled)[0])
            time_pred_h = max(0.0, raw_reg_out)
        else:
            raw_reg_out = float("nan")
            time_pred_h = float("nan")

    return dict(
        stage_pred=stage_pred,
        stage_conf=stage_conf,
        time_pred_h=time_pred_h,
        clf_probs=clf_probs if isinstance(clf_probs, list) else clf_probs.tolist(),
        raw_reg_out=raw_reg_out,
    )


# =============================================================================
# Display helpers
# =============================================================================

_PROB_BAR_WIDTH = 20


def _prob_bar(prob: float) -> str:
    """Render a compact ASCII probability bar for *prob* in [0, 1]."""
    filled = round(prob * _PROB_BAR_WIDTH)
    return "[" + "#" * filled + "." * (_PROB_BAR_WIDTH - filled) + "]"


def print_tick(
    tick: int,
    ts: datetime,
    current_stage: str,
    reading: dict,
    result: dict,
    arts: dict,
    alert: bool,
    model_type: str,
    warmup: bool,
) -> None:
    """Print a formatted per-tick diagnostic to stdout."""
    int_to_stage = arts["int_to_stage"]
    n_cls        = len(int_to_stage)

    sep = "=" * 74 if alert else "-" * 74
    print(sep)

    if alert:
        print("  *** STAGE TRANSITION ALERT — transition predicted within "
              f"{arts['alert_lead_h']}h ***")

    period = "DAY " if reading["day_night_flag"] else "NGHT"
    warmup_tag = "  [WARMUP]" if warmup else ""
    print(
        f"  Tick #{tick:04d}{warmup_tag}  |  "
        f"Sim time: {ts.strftime('%Y-%m-%d %H:%M')}  [{period}]"
    )
    print(
        f"  Model: {model_type.upper():<4}  |  "
        f"Current stage : {current_stage}"
    )
    print()

    # Sensor readings
    print(
        f"  Sensors  "
        f"T={reading['temperature']:5.1f}C  "
        f"RH={reading['humidity']:5.1f}%  "
        f"VPD={reading['vpd']:5.3f} kPa  "
        f"CO2={reading['co2']:5.0f} ppm  "
        f"Rad={reading['solar_radiation']:5.0f} W/m2  "
        f"Vel={reading['air_velocity']:4.2f} m/s"
    )
    print(
        f"           "
        f"Dew={reading['dew_point']:5.1f}C"
    )
    print()

    # Prediction summary
    time_str = (
        f"{result['time_pred_h']:.1f} h"
        if not math.isnan(result["time_pred_h"])
        else "n/a"
    )
    alert_tag = "  <-- ALERT" if alert else ""
    print(
        f"  Prediction   next stage : {result['stage_pred']:<25} "
        f"(conf={result['stage_conf']:.1%})"
    )
    print(
        f"               time-to-transition : {time_str}{alert_tag}"
    )
    print()

    # Class probability breakdown
    if result["clf_probs"]:
        print("  Stage probabilities:")
        for cls_idx, prob in enumerate(result["clf_probs"]):
            stage_name = int_to_stage.get(cls_idx, str(cls_idx))
            marker = " <-- predicted" if cls_idx == STAGE_TO_INT.get(result["stage_pred"], -1) else ""
            print(
                f"    {stage_name:<25} {_prob_bar(prob)} {prob:5.1%}{marker}"
            )


# =============================================================================
# Interactive configuration menu
# =============================================================================

def _interactive_setup() -> argparse.Namespace:
    """
    Display a step-by-step configuration menu and return a Namespace
    that matches exactly what argparse would produce in main().

    Called automatically when the script is run without any CLI arguments.
    """

    def _pick(prompt: str, options: list[str], default: int = 0) -> int:
        """Print numbered options and return the 0-based index chosen by the user."""
        print(f"\n  {prompt}")
        for i, opt in enumerate(options):
            tag = "  <-- default" if i == default else ""
            print(f"    [{i + 1}] {opt}{tag}")
        while True:
            try:
                raw = input(f"  Choice [1-{len(options)}, Enter={default + 1}]: ").strip()
            except EOFError:
                return default
            if raw == "":
                return default
            if raw.isdigit() and 1 <= int(raw) <= len(options):
                return int(raw) - 1
            print(f"  Invalid -- enter a number between 1 and {len(options)}.")

    print()
    print("=" * 64)
    print("  AgriTwin-GH  |  Growth Progression Model Tester")
    print("  Interactive Setup  --  press Enter to accept the default")
    print("=" * 64)

    # -- 1. Starting growth stage ------------------------------------------
    stage_labels = [
        "seedling               - germination & root establishment",
        "early_veg              - leaf/stem expansion",
        "flowering_initiation   - bud formation begins",
        "flowering              - anthesis & pollination",
        "unripe                 - fruit development",
        "ripe                   - harvest-ready",
    ]
    stage_idx = _pick("Starting growth stage:", stage_labels, default=0)
    stage = STAGE_ORDER[stage_idx]

    # -- 2. Inference model ------------------------------------------------
    model_labels = [
        "auto   - use best model recorded in artifacts",
        "rf     - Random Forest  (fast, no TF warmup)",
        "lstm   - LSTM neural network  (requires TF initialisation)",
    ]
    model_vals = ["auto", "rf", "lstm"]
    model_idx  = _pick("Inference model:", model_labels, default=0)
    model      = model_vals[model_idx]

    # -- 3. Simulation speed -----------------------------------------------
    speed_labels = [
        "instant  (0.00 s/tick) - run as fast as possible",
        "fast     (0.10 s/tick)",
        "normal   (0.50 s/tick)",
        "slow     (2.00 s/tick) - closer to real-time feel",
    ]
    speed_vals = [0.0, 0.1, 0.5, 2.0]
    speed_idx  = _pick("Simulation speed (wall-clock seconds per simulated hour):",
                       speed_labels, default=2)
    tick_secs  = speed_vals[speed_idx]

    # ── 4. Number of ticks ────────────────────────────────────────────────────
    ticks_labels = [
        "50      - quick sanity check",
        "200     - short run",
        "500     - standard run",
        "unlimited - run until Ctrl+C",
    ]
    ticks_vals = [50, 200, 500, 0]
    ticks_idx  = _pick("Number of ticks to simulate:", ticks_labels, default=2)
    max_ticks  = ticks_vals[ticks_idx]

    # -- 5. JSON log -------------------------------------------------------
    log_labels = [
        "No   - do not save a log file",
        "Yes  - save to logs/realtime_test_<timestamp>.json",
    ]
    log_idx = _pick("Save a per-tick JSON log?", log_labels, default=0)
    if log_idx == 1:
        ts_tag   = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_json = f"logs/realtime_test_{ts_tag}.json"
    else:
        log_json = ""

    # -- Summary ----------------------------------------------------------
    print()
    print("=" * 64)
    print("  Configuration summary")
    print(f"    Starting stage : {stage}")
    print(f"    Model          : {model}")
    print(f"    Speed          : {tick_secs:.2f} s / tick")
    print(f"    Max ticks      : {max_ticks if max_ticks > 0 else 'unlimited'}")
    print(f"    JSON log       : {log_json if log_json else 'disabled'}")
    print("=" * 64)
    print()

    return argparse.Namespace(
        stage=stage,
        model=model,
        tick_secs=tick_secs,
        max_ticks=max_ticks,
        log_json=log_json,
        run_id="",
    )


# =============================================================================
# Main simulation loop
# =============================================================================

def main() -> None:
    # Ensure UTF-8 output on Windows where the console default may be cp1252
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Real-time growth-progression model tester.",
    )
    parser.add_argument(
        "--tick-secs", type=float, default=0.5,
        help="Wall-clock seconds between ticks (default: 0.5).",
    )
    parser.add_argument(
        "--stage", type=str, default="seedling", choices=STAGE_ORDER,
        help="Starting growth stage (default: seedling).",
    )
    parser.add_argument(
        "--max-ticks", type=int, default=500,
        help="Stop after N ticks; 0 = unlimited (default: 500).",
    )
    parser.add_argument(
        "--model", type=str, default="auto", choices=["auto", "rf", "lstm"],
        help="Model to use for inference (default: auto = use best_model from artifacts).",
    )
    parser.add_argument(
        "--log-json", type=str, default="",
        help="Path to save per-tick JSON log (default: none).",
    )
    parser.add_argument(
        "--run-id", type=str, default="",
        help="Specific artifact run ID to load (default: latest).",
    )

    # When no CLI arguments are given, launch the interactive setup menu
    # instead of falling back silently to all defaults.
    if len(sys.argv) == 1:
        args = _interactive_setup()
    else:
        args = parser.parse_args()

    # ── Locate artifact directory ─────────────────────────────────────────────
    if args.run_id:
        run_dir = ARTIFACTS_BASE / args.run_id
        if not run_dir.exists():
            LOG.error("Artifact directory not found: %s", run_dir)
            sys.exit(1)
    else:
        candidates = sorted(ARTIFACTS_BASE.glob("growth_progression_*"), reverse=True)
        if not candidates:
            LOG.error(
                "No growth_progression_* artifact directories found under %s",
                ARTIFACTS_BASE,
            )
            sys.exit(1)
        run_dir = candidates[0]
        LOG.info("Latest run: %s", run_dir.name)

    # ── Load artifacts ────────────────────────────────────────────────────────
    # Load best_model_summary first to decide whether LSTM needs to be loaded.
    _best_meta = json.loads(
        (run_dir / "best_model_summary.json").read_text()
    )
    _auto_model = _best_meta["best_model"]
    _need_lstm  = (args.model == "lstm") or (args.model == "auto" and _auto_model == "lstm")

    arts = load_artifacts(run_dir, load_lstm=_need_lstm)

    # ── Resolve model type ────────────────────────────────────────────────────
    model_type = args.model if args.model != "auto" else arts["best_model_type"]

    if model_type == "lstm" and arts["lstm_model"] is None:
        LOG.warning("LSTM model file not found; falling back to RF.")
        model_type = "rf"

    if model_type == "rf" and arts["clf_rf"] is None:
        LOG.error("RF classifier not found. Cannot run inference.")
        sys.exit(1)

    LOG.info("Active inference model: %s", model_type.upper())

    # ── Initial state ─────────────────────────────────────────────────────────
    rng                 = random.Random(42)
    current_stage       = args.stage
    current_stage_int   = STAGE_TO_INT[current_stage]

    # Simulation clock: start on 2025-03-01 at 06:00 so the first ticks are daytime
    sim_ts = datetime(2025, 3, 1, 6, 0, 0)

    # Rolling buffer of raw rows.  We keep _BUFFER_CAPACITY rows so that
    # rolling (max window=72) and lag (max lag=24) features are accurate.
    _BUFFER_CAPACITY = 300
    buffer_rows: list[tuple[datetime, dict]] = []

    # Alert suppression: track last alert time in simulated hours (ticks)
    last_alert_tick: float = -float("inf")

    # Warmup: first window_n ticks collect data before inference is meaningful
    window_n = arts["window_n"]

    tick_logs: list[dict] = []
    tick = 0

    print()
    print("=" * 74)
    print("  AgriTwin-GH - Growth Progression Real-Time Model Test")
    print(f"  Run       : {arts['run_id']}")
    print(f"  Model     : {model_type.upper()}")
    print(f"  Stage     : {current_stage}  (starting)")
    print(f"  Tick rate : {args.tick_secs:.2f} s / simulated hour")
    print(f"  Max ticks : {args.max_ticks if args.max_ticks > 0 else 'unlimited'}")
    print(f"  Alert lead: {arts['alert_lead_h']} h | "
          f"Conf threshold: {arts['confidence_threshold']:.0%}")
    print(f"  Warmup    : {window_n} ticks before LSTM window is fully populated")
    print("=" * 74)
    print()
    LOG.info("Press Ctrl+C to stop.\n")

    try:
        while True:
            tick += 1
            if args.max_ticks > 0 and tick > args.max_ticks:
                LOG.info("max-ticks=%d reached. Stopping.", args.max_ticks)
                break

            # ── Generate synthetic sensor reading ──────────────────────────────
            reading = simulate_reading(sim_ts, current_stage, rng)

            # ── Append to buffer ───────────────────────────────────────────────
            buffer_rows.append((sim_ts, {**reading,
                                         "leaf_wetness_proxy": 0,  # placeholder
                                         "growth_stage": current_stage,
                                         "stage_int": current_stage_int}))
            if len(buffer_rows) > _BUFFER_CAPACITY:
                buffer_rows = buffer_rows[-_BUFFER_CAPACITY:]

            # ── Build buffer DataFrame ─────────────────────────────────────────
            idx = pd.DatetimeIndex(
                [r[0] for r in buffer_rows], name="datetime"
            )
            buf = pd.DataFrame([r[1] for r in buffer_rows], index=idx)

            # leaf_wetness_proxy: humidity > 85 for 3 consecutive hours
            high_hum = (buf["humidity"] > 85).astype(int)
            buf["leaf_wetness_proxy"] = (
                high_hum.rolling(window=3, min_periods=1).sum() >= 3
            ).astype(int)

            # ── Feature engineering ────────────────────────────────────────────
            try:
                buf_feat = engineer_features(buf)
            except Exception as exc:
                LOG.warning("Feature engineering error at tick %d: %s", tick, exc)
                sim_ts += timedelta(hours=1)
                time.sleep(args.tick_secs)
                continue

            # ── Inference ──────────────────────────────────────────────────────
            warmup = (tick < window_n)

            try:
                result = run_inference(buf_feat, arts, model_type)
            except Exception as exc:
                LOG.warning("Inference error at tick %d: %s", tick, exc)
                sim_ts += timedelta(hours=1)
                time.sleep(args.tick_secs)
                continue

            # ── Alert logic ────────────────────────────────────────────────────
            time_since_last = tick - last_alert_tick
            alert = (
                not warmup
                and not math.isnan(result["time_pred_h"])
                and result["time_pred_h"] <= arts["alert_lead_h"]
                and result["stage_conf"]  >= arts["confidence_threshold"]
                and time_since_last       >= arts["repeat_interval_h"]
            )
            if alert:
                last_alert_tick = tick

            # ── Print diagnostic ───────────────────────────────────────────────
            print_tick(
                tick, sim_ts, current_stage, reading,
                result, arts, alert, model_type, warmup,
            )

            # ── JSON log ───────────────────────────────────────────────────────
            if args.log_json:
                tick_logs.append({
                    "tick":          tick,
                    "sim_ts":        sim_ts.isoformat(),
                    "current_stage": current_stage,
                    "sensors":       reading,
                    "prediction":    {
                        "stage_pred":  result["stage_pred"],
                        "stage_conf":  round(result["stage_conf"], 4),
                        "time_pred_h": round(result["time_pred_h"], 2),
                        "clf_probs":   [round(p, 4) for p in result["clf_probs"]],
                    },
                    "alert_fired":   alert,
                })

            # ── Advance simulation clock ───────────────────────────────────────
            sim_ts += timedelta(hours=1)

            # ── Sleep ─────────────────────────────────────────────────────────
            if args.tick_secs > 0:
                time.sleep(args.tick_secs)

    except KeyboardInterrupt:
        print("\n\n[Stopped by user]")

    # ── Save JSON log ──────────────────────────────────────────────────────────
    if args.log_json and tick_logs:
        log_path = Path(args.log_json)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fh:
            json.dump(tick_logs, fh, indent=2)
        LOG.info("Tick log saved: %s  (%d ticks)", log_path, len(tick_logs))

    LOG.info("Simulation ended after %d ticks.", min(tick, args.max_ticks) if args.max_ticks > 0 else tick)


if __name__ == "__main__":
    main()
