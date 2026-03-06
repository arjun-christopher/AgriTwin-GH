#!/usr/bin/env python3
"""
test_disease_progression_model.py
===================================
Interactive inference-validation script for the AgriTwin-GH
Disease Progression Risk Forecasting model.

No real greenhouse dataset required — six realistic synthetic scenarios
are generated from first principles (diurnal physics, disease biology)
and fed through the full inference pipeline.

What this script does
─────────────────────
1.  Auto-discovers the latest trained run and loads all artefacts.
2.  Presents an interactive terminal menu to choose:
      • Which model to run  (Random Forest / LSTM / Both)
      • Which scenario to simulate  (6 choices + Run All)
      • Simulation speed  (Fast / Real-time stream / Slow-live stream)
3.  Synthesises hourly greenhouse physics data for the chosen scenario.
4.  Optionally streams hourly sensor readings to the terminal in real time.
5.  Runs the full feature-engineering pipeline (Sections D–F) and
    model inference for all four forecast horizons and five diseases.
6.  Prints a formatted system-style disease-risk alert report.

Usage
─────
    python scripts/test_disease_progression_model.py

    # (No CLI arguments — all choices are made through the interactive menu)
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATH SETUP
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent          # scripts/ → repo root

MODELS_DIR    = REPO_ROOT / "src" / "agritwin_gh" / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS  (must match training notebook CONFIG / DISEASE_THRESHOLDS)
# ─────────────────────────────────────────────────────────────────────────────
DISEASES   = ["late_blight", "leaf_mold", "powdery_mildew",
              "early_blight", "spider_mites"]
HORIZON_H  = [6, 12, 24, 48]
WINDOW_N   = 24          # LSTM lookback (hours)
HIGH_THRESH   = 67.0
MEDIUM_THRESH = 34.0

RISK_LABEL_BINS = {
    "low":    [0,  33],
    "medium": [34, 66],
    "high":   [67, 100],
}

DISEASE_THRESHOLDS = {
    "late_blight": {
        "rh_min": 90.0, "vpd_max": 0.40, "required_hours_24h": 6,
        "night_boost": 15, "night_boost_cap": 15,
        "low_airflow_thresh": 0.5, "low_airflow_boost": 10, "low_airflow_boost_cap": 10,
    },
    "leaf_mold": {
        "rh_min": 90.0, "vpd_max": 0.50, "required_hours_24h": 5,
        "low_airflow_thresh": 1.5, "low_airflow_boost": 10, "low_airflow_boost_cap": 10,
        "night_boost": 10, "night_boost_cap": 10,
    },
    "powdery_mildew": {
        "rh_min": 70.0, "rh_max": 90.0, "vpd_min": 0.50, "vpd_max": 1.20,
        "required_hours_24h": 6,
        "low_airflow_thresh": 0.5, "low_airflow_boost": 10, "low_airflow_boost_cap": 10,
    },
    "early_blight": {
        "temp_min": 24.0, "temp_max": 30.0, "rh_min": 85.0,
        "required_hours_24h": 4,
        "radiation_boost_thresh": 200.0, "radiation_boost": 12, "radiation_boost_cap": 12,
    },
    "spider_mites": {
        "temp_min": 28.0, "rh_max": 55.0, "vpd_min": 1.50,
        "required_hours_24h": 1,
        "radiation_boost_thresh": 150.0, "radiation_boost": 15, "radiation_boost_cap": 15,
    },
}

FE_CONFIG = {
    "base_vars":   ["temp", "humidity", "air_velocity", "co2",
                    "solar_radiation", "vpd", "dew_point"],
    "lag_vars":    ["temp", "humidity", "vpd", "air_velocity",
                    "solar_radiation", "dew_point"],
    "roll_windows": [6, 12, 24],
    "lags":        [1, 2, 3, 6, 12, 24],
    "diseases":    DISEASES,
    "window_N":    WINDOW_N,
    "horizon_H":   HORIZON_H,
}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATA GENERATOR
# ═════════════════════════════════════════════════════════════════════════════

def _vpd(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Tetens formula: VPD (kPa) from temperature (°C) and RH (%)."""
    es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    return np.clip(es * (1.0 - rh_pct / 100.0), 0.0, None)


def _dew_point(temp_c: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """Magnus approximation: dew point (°C)."""
    return temp_c - (100.0 - rh_pct) / 5.0


def _solar_radiation(hours: np.ndarray, peak_wm2: float = 450.0) -> np.ndarray:
    """Half-sine solar curve: rises at 06:00, peaks at 13:00, sets at 20:00."""
    rad = np.zeros(len(hours))
    for i, h in enumerate(hours % 24):
        if 6 <= h <= 20:
            angle = np.pi * (h - 6) / 14.0
            rad[i] = peak_wm2 * np.sin(angle)
    return rad


def _diurnal_temp(hours: np.ndarray,
                  t_min: float, t_max: float,
                  noise_std: float = 0.3) -> np.ndarray:
    """Smooth diurnal temperature: min at ~06:00, max at ~14:00."""
    h = hours % 24
    phase = (h - 10) / 24.0 * 2 * np.pi      # 10 h = midday reference
    temp = t_min + (t_max - t_min) * (0.5 - 0.5 * np.cos(phase))
    return temp + np.random.normal(0, noise_std, len(hours))


def _diurnal_rh(temp: np.ndarray,
                rh_night_mean: float, rh_day_mean: float,
                noise_std: float = 1.5) -> np.ndarray:
    """RH varies inversely with temperature within the day/night envelope."""
    t_min, t_max = temp.min(), temp.max()
    if t_max == t_min:
        rh = np.full(len(temp), rh_night_mean)
    else:
        rh = rh_night_mean - (temp - t_min) / (t_max - t_min) * (rh_night_mean - rh_day_mean)
    rh += np.random.normal(0, noise_std, len(temp))
    return np.clip(rh, 5.0, 100.0)


def _air_velocity(hours: np.ndarray,
                  av_base: float, av_day_bonus: float = 1.2,
                  noise_std: float = 0.15) -> np.ndarray:
    """Ventilation fans ramp up during the day; lower at night."""
    h = hours % 24
    is_day = ((h >= 7) & (h <= 19)).astype(float)
    av = av_base + av_day_bonus * is_day + np.random.normal(0, noise_std, len(hours))
    return np.clip(av, 0.10, 10.0)


def _co2(hours: np.ndarray, day_night_flag: np.ndarray,
         co2_base: float = 600.0) -> np.ndarray:
    """CO₂ builds up at night (no photosynthesis), consumed by day."""
    h = hours % 24
    # Night accumulation, daytime draw-down
    co2 = co2_base + 200 * (1 - day_night_flag) - 100 * day_night_flag
    co2 += np.random.normal(0, 15, len(hours))
    return np.clip(co2, 380.0, 1400.0)


def generate_scenario(
    start_dt: pd.Timestamp,
    n_hours: int,
    t_min: float,         # °C — overnight low
    t_max: float,         # °C — daytime high
    rh_night: float,      # % — night RH
    rh_day: float,        # % — daytime RH
    av_base: float,       # m/s — base air velocity (night)
    solar_peak: float,    # W/m² — peak solar radiation
    co2_base: float = 600.0,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """
    Generate one synthetic greenhouse scenario.

    Returns a DataFrame with DatetimeIndex (hourly) and columns:
        temp, humidity, air_velocity, co2, solar_radiation,
        vpd, dew_point, day_night_flag
    """
    np.random.seed(rng_seed)
    idx   = pd.date_range(start=start_dt, periods=n_hours, freq="h")
    hours = np.arange(n_hours, dtype=float) + start_dt.hour

    temp    = _diurnal_temp(hours, t_min, t_max)
    rh      = _diurnal_rh(temp, rh_night, rh_day)
    solar   = _solar_radiation(hours, solar_peak)
    dn_flag = ((hours % 24 >= 6) & (hours % 24 <= 18)).astype(int)
    av      = _air_velocity(hours, av_base)
    vpd     = _vpd(temp, rh)
    dew     = _dew_point(temp, rh)
    co2     = _co2(hours, dn_flag, co2_base)

    df = pd.DataFrame({
        "temp":           temp.round(2),
        "humidity":       rh.round(2),
        "air_velocity":   av.round(2),
        "co2":            co2.round(1),
        "solar_radiation":solar.round(1),
        "vpd":            vpd.round(3),
        "dew_point":      dew.round(2),
        "day_night_flag": dn_flag,
    }, index=idx)
    df.index.name = "datetime"
    return df


# Six scenarios — engineered for agronomic realism and model novelty
SCENARIOS = [
    # ── 1. Monsoon Late Blight Crisis ─────────────────────────────────────────
    # Three days of persistent cool, damp, cloudy conditions. Sustained RH≥92%
    # for 10+ hours each night; VPD stays below 0.35 kPa throughout.
    # Expects HIGH late_blight + HIGH leaf_mold at most horizons.
    dict(
        name        = "Monsoon Late-Blight Crisis",
        description = (
            "3-day monsoon event: persistent cloud cover, RH 92–96%, VPD<0.35, "
            "poor ventilation. Classic Late Blight + Leaf Mold incubation window."
        ),
        start_dt    = pd.Timestamp("2025-08-12 00:00"),
        n_hours     = 72,
        t_min       = 18.5, t_max   = 23.0,
        rh_night    = 94.5, rh_day  = 91.0,
        av_base     = 0.55, solar_peak = 90.0,
        co2_base    = 650.0, rng_seed = 1,
    ),

    # ── 2. Summer Spider-Mite Heat Wave ─────────────────────────────────────
    # Three brutal midsummer days: 36 °C peak, RH drops to 36% midday,
    # VPD exceeds 2.2 kPa. 2+ hours per day of temp≥28 + RH≤55 + VPD≥1.5.
    # Expects HIGH spider_mites; late_blight / leaf_mold stay Low.
    dict(
        name        = "Summer Spider-Mite Heat Wave",
        description = (
            "Prolonged heatwave: 34–36°C, RH 36–52%, VPD>2.0 kPa. "
            "Classic spider-mite explosion trigger with intense solar load."
        ),
        start_dt    = pd.Timestamp("2025-06-20 00:00"),
        n_hours     = 72,
        t_min       = 29.0, t_max   = 36.5,
        rh_night    = 52.0, rh_day  = 36.0,
        av_base     = 2.8,  solar_peak = 580.0,
        co2_base    = 480.0, rng_seed = 2,
    ),

    # ── 3. Powdery-Mildew Corridor ────────────────────────────────────────────
    # Moderate autumn climate that falls perfectly into the powdery-mildew
    # sweet spot: RH 72–87%, VPD 0.55–1.10, low airflow.
    # Late Blight / Leaf Mold are borderline; Powdery Mildew dominates.
    dict(
        name        = "Powdery-Mildew Corridor",
        description = (
            "Autumn moderate climate: RH 72–87%, VPD 0.55–1.1 kPa, cool nights. "
            "Powdery mildew sweet spot — often missed in reactve monitoring."
        ),
        start_dt    = pd.Timestamp("2025-10-05 00:00"),
        n_hours     = 72,
        t_min       = 19.0, t_max   = 26.5,
        rh_night    = 86.5, rh_day  = 73.0,
        av_base     = 0.45, solar_peak = 220.0,
        co2_base    = 700.0, rng_seed = 3,
    ),

    # ── 4. Early-Blight Warm-Humid Day Cycle ─────────────────────────────────
    # Warm, partially cloudy afternoons (24–29°C) with moderate humidity
    # and strong midday radiation spikes. Exactly the Early Blight trigger.
    # Novel: conditions look "manageable" by humidity but the temperature-RH
    # combination and radiation load drive substantial Early Blight risk.
    dict(
        name        = "Early-Blight Warm-Humid Cycle",
        description = (
            "Warm partly-cloudy stretch: 24–29°C, RH 85–90%, 300–400 W/m² solar. "
            "Radiation stress on top of sustained moderate humidity — a "
            "frequently underestimated Early Blight incubator."
        ),
        start_dt    = pd.Timestamp("2025-09-01 00:00"),
        n_hours     = 72,
        t_min       = 23.5, t_max   = 29.0,
        rh_night    = 90.0, rh_day  = 85.0,
        av_base     = 1.8,  solar_peak = 360.0,
        co2_base    = 620.0, rng_seed = 4,
    ),

    # ── 5. Night-Raid Fog Intrusion ───────────────────────────────────────────
    # Novel scenario: pleasant, well-managed daytime → sudden RH spike after
    # dusk (fog/dew intrusion through vents, common near coastal areas).
    # Night: RH 93–97%, VPD → 0.10, air velocity collapses to 0.6 m/s.
    # Day:   RH 65–75%, VPD 1.2–1.5.
    # Models must detect the night-specific risk WITHOUT being confused by
    # the benign daytime readings — tests night-segmented rolling features.
    dict(
        name        = "Night-Raid Fog Intrusion",
        description = (
            "Coastal fog intrusion post-sunset: daytime looks healthy "
            "(RH 65 %, AV 3 m/s) but nights see RH 94 %, AV 0.6 m/s. "
            "Tests night-segmented feature tracking — impossible to catch "
            "without rolling night-exposure columns."
        ),
        start_dt    = pd.Timestamp("2025-07-15 00:00"),
        n_hours     = 72,
        # Night and day conditions are very different — generate manually:
        t_min       = 21.0, t_max   = 28.0,
        rh_night    = 95.0, rh_day  = 68.0,
        av_base     = 0.60, solar_peak = 310.0,
        co2_base    = 580.0, rng_seed = 5,
    ),

    # ── 6. Healthy Well-Managed Greenhouse ───────────────────────────────────
    # Control / baseline: precision climate control keeps all parameters in the
    # "green zone" — good airflow, moderate humidity, no temperature extremes.
    # Expects LOW on all diseases. Validates the model doesn't false-alarm.
    dict(
        name        = "Healthy Well-Managed Baseline",
        description = (
            "Precision-controlled spring climate: RH 62–72%, temp 22–26°C, "
            "AV 2.5 m/s, VPD 1.1–1.4 kPa. All diseases should score Low. "
            "Validates false-alarm resistance."
        ),
        start_dt    = pd.Timestamp("2025-04-10 00:00"),
        n_hours     = 72,
        t_min       = 21.5, t_max   = 26.0,
        rh_night    = 72.0, rh_day  = 62.0,
        av_base     = 2.50, solar_peak = 280.0,
        co2_base    = 550.0, rng_seed = 6,
    ),
]


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — FEATURE ENGINEERING  (replicated from notebook Sections D + F)
# ═════════════════════════════════════════════════════════════════════════════

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def compute_binary_condition(df: pd.DataFrame, disease: str) -> pd.Series:
    """Hourly 0/1 flag — identical to notebook cell D2."""
    th = DISEASE_THRESHOLDS[disease]
    T, RH, AV, VP = df["temp"], df["humidity"], df["air_velocity"], df["vpd"]

    if disease == "late_blight":
        cond = (RH >= th["rh_min"]) & (VP <= th["vpd_max"])
    elif disease == "leaf_mold":
        cond = (RH >= th["rh_min"]) & (VP <= th["vpd_max"])
    elif disease == "powdery_mildew":
        cond = ((RH >= th["rh_min"]) & (RH <= th["rh_max"])
                & (VP >= th["vpd_min"]) & (VP <= th["vpd_max"]))
    elif disease == "early_blight":
        cond = (T >= th["temp_min"]) & (T <= th["temp_max"]) & (RH >= th["rh_min"])
    elif disease == "spider_mites":
        cond = (T >= th["temp_min"]) & (RH <= th["rh_max"]) & (VP >= th["vpd_min"])
    else:
        raise ValueError(f"Unknown disease: {disease}")
    return cond.astype(int)


def compute_rolling_exposures(df: pd.DataFrame, disease: str) -> pd.DataFrame:
    """Rolling exposure windows — identical to notebook cell D3."""
    df_out = df.copy()
    cond   = compute_binary_condition(df, disease)
    df_out[f"cond_{disease}"] = cond

    for w in [6, 12, 24]:
        df_out[f"exposure_count_{w}h_{disease}"] = (
            cond.rolling(w, min_periods=1).sum().astype(int)
        )
    night_mask = (df["day_night_flag"] == 0).astype(int)
    df_out[f"night_exposure_24h_{disease}"] = (
        (cond * night_mask).rolling(24, min_periods=1).sum().astype(int)
    )
    return df_out


def compute_risk_index(df: pd.DataFrame, disease: str) -> pd.Series:
    """Risk index 0–100 — identical to notebook cell D4."""
    th      = DISEASE_THRESHOLDS[disease]
    req24   = th["required_hours_24h"]
    exp24   = df[f"exposure_count_24h_{disease}"].values.astype(float)
    night24 = df[f"night_exposure_24h_{disease}"].values.astype(float)
    av_m24  = df["air_velocity"].rolling(24, min_periods=1).mean().values
    sr_m24  = df["solar_radiation"].rolling(24, min_periods=1).mean().values

    risk = np.zeros(len(df))
    for i in range(len(df)):
        base      = 100.0 * min(1.0, exp24[i] / req24)
        modifiers = 0.0

        if "night_boost" in th:
            nb = min(th["night_boost_cap"], th["night_boost"] * (night24[i] / 24.0))
            modifiers += nb

        if "low_airflow_thresh" in th:
            av_ratio = _clamp(av_m24[i] / th["low_airflow_thresh"])
            ab = min(th["low_airflow_boost_cap"],
                     th["low_airflow_boost"] * (1.0 - av_ratio))
            modifiers += ab

        if "radiation_boost_thresh" in th:
            sr_excess = _clamp(
                (sr_m24[i] - th["radiation_boost_thresh"]) / th["radiation_boost_thresh"]
            )
            rb = min(th["radiation_boost_cap"], th["radiation_boost"] * sr_excess)
            modifiers += rb

        risk[i] = min(100.0, max(0.0, base + modifiers))
    return pd.Series(risk, index=df.index, name=f"risk_{disease}")


def run_disease_risk_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrates D2-D5 for all five diseases (matches notebook)."""
    df_e = df.copy()
    for d in DISEASES:
        df_e = compute_rolling_exposures(df_e, d)
        df_e[f"risk_{d}"] = compute_risk_index(df_e, d)
    return df_e


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour_sin"]  = np.sin(2 * np.pi * df.index.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df.index.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df.index.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df.index.month / 12)
    return df


def compute_rolling_stats(df: pd.DataFrame) -> pd.DataFrame:
    df_out = df.copy()
    for var in FE_CONFIG["base_vars"]:
        if var not in df.columns:
            continue
        s = df[var]
        for W in FE_CONFIG["roll_windows"]:
            r = s.rolling(window=W, min_periods=1)
            df_out[f"{var}_mean_{W}h"] = r.mean()
            df_out[f"{var}_max_{W}h"]  = r.max()
            df_out[f"{var}_min_{W}h"]  = r.min()
            df_out[f"{var}_std_{W}h"]  = r.std().fillna(0.0)
    return df_out


def compute_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df_out  = df.copy()
    lag_cols = []
    for var in FE_CONFIG["lag_vars"]:
        if var not in df.columns:
            continue
        for lag in FE_CONFIG["lags"]:
            col = f"{var}_lag{lag}"
            df_out[col] = df[var].shift(lag)
            lag_cols.append(col)
    if lag_cols:
        df_out[lag_cols] = df_out[lag_cols].bfill().ffill()
    return df_out


def compute_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["temp_x_rh"]      = df["temp"]  * df["humidity"]
    df["vpd_x_rh"]       = df["vpd"]   * df["humidity"]
    df["dewpoint_spread"] = df["temp"]  - df["dew_point"]
    return df


def compute_day_night_rolling(df: pd.DataFrame) -> pd.DataFrame:
    df_out   = df.copy()
    is_night = (df["day_night_flag"] == 0)
    for var in FE_CONFIG["base_vars"]:
        if var not in df.columns:
            continue
        night_series = df[var].where(is_night)
        for W in FE_CONFIG["roll_windows"]:
            col = f"{var}_night_mean_{W}h"
            df_out[col] = (
                night_series.rolling(W, min_periods=1).mean().ffill().bfill()
            )
    return df_out


def build_feature_matrix(df_enriched: pd.DataFrame):
    """Full feature matrix construction — mirrors notebook cell F6."""
    df_fe = add_cyclical_time_features(df_enriched)
    df_fe = compute_rolling_stats(df_fe)
    df_fe = compute_lag_features(df_fe)
    df_fe = compute_interaction_terms(df_fe)
    df_fe = compute_day_night_rolling(df_fe)
    return df_fe


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3A — LSTM ARCHITECTURE (replicated from notebook cell J2)
# ═════════════════════════════════════════════════════════════════════════════

def build_lstm_model(n_timesteps, n_features, n_diseases, horizon_H, lstm_cfg):
    """
    Rebuild the exact same multi-head LSTM architecture used during training.
    Used to load weights without triggering Keras Lambda deserialization.

    Architecture
    ────────────
    Input (N, F)
      └─ LSTM(units_1, return_seq=True) → [LayerNorm] → Dropout
          └─ LSTM(units_2, return_seq=False) → [LayerNorm] → Dropout
              └─ Dense(dense_units, relu)   ← shared encoder
                  ├─ Head H6  : Dense(32) → Dense(D) → Clip[0,100]
                  ├─ Head H12 : Dense(32) → Dense(D) → Clip[0,100]
                  ├─ Head H24 : Dense(32) → Dense(D) → Clip[0,100]
                  └─ Head H48 : Dense(32) → Dense(D) → Clip[0,100]
    """
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    mc  = lstm_cfg
    inp = keras.Input(shape=(n_timesteps, n_features), name="input_seq")

    x = layers.LSTM(mc["units_1"], return_sequences=True,  name="lstm_1")(inp)
    if mc.get("layer_norm", False):
        x = layers.LayerNormalization(name="ln_1")(x)
    x = layers.Dropout(mc["dropout_1"], name="drop_1")(x)

    x = layers.LSTM(mc["units_2"], return_sequences=False, name="lstm_2")(x)
    if mc.get("layer_norm", False):
        x = layers.LayerNormalization(name="ln_2")(x)
    x = layers.Dropout(mc["dropout_2"], name="drop_2")(x)

    x = layers.Dense(mc["dense_units"], activation="relu", name="shared_dense")(x)

    outputs = {}
    for H in horizon_H:
        h   = layers.Dense(32, activation="relu", name=f"head_H{H}")(x)
        out = layers.Dense(n_diseases, name=f"output_H{H}")(h)
        # Use a named function (not an inline lambda) so output_shape is inferrable
        out = layers.Lambda(
            lambda t: tf.clip_by_value(t, 0.0, 100.0),
            output_shape=lambda s: s,
            name=f"clip_H{H}",
        )(out)
        outputs[f"output_H{H}"] = out

    model = keras.Model(inputs=inp, outputs=outputs, name="dp_lstm")
    return model


def _load_lstm_weights(keras_path: Path, model):
    """
    Load weights from a .keras archive without deserializing the full model config.

    Keras 3 stores weights inside the .keras zip as model.weights.h5 in its own
    internal format (not legacy HDF5 groups). The cleanest way to load them is to
    call model.load_weights() directly on the .keras file — Keras 3 handles the
    zip extraction and weight assignment automatically.
    """
    keras_path = Path(keras_path)
    model.load_weights(str(keras_path))
    return model


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — ARTIFACT DISCOVERY & LOADING
# ═════════════════════════════════════════════════════════════════════════════

def find_latest_run(run_id_override: str = None) -> dict:
    """
    Auto-discover the latest training run from the artifacts directory.
    Returns a dict with paths to all required artefacts.
    """
    if run_id_override:
        run_dir = ARTIFACTS_DIR / run_id_override
        if not run_dir.exists():
            sys.exit(f"[ERROR] Run directory does not exist: {run_dir}")
        run_id = run_id_override
    else:
        run_dirs = sorted(
            [d for d in ARTIFACTS_DIR.iterdir()
             if d.is_dir() and d.name.startswith("dp_")],
            key=lambda d: d.stat().st_mtime,
            reverse=True,
        )
        if not run_dirs:
            sys.exit(
                f"[ERROR] No training runs found in {ARTIFACTS_DIR}.\n"
                "        Run the training notebook first."
            )
        run_dir = run_dirs[0]
        run_id  = run_dir.name

    print(f"[ARTEFACT] Run ID  : {run_id}")
    print(f"           Run dir : {run_dir}")

    paths = {
        "run_id":         run_id,
        "run_dir":        run_dir,
        "scaler":         run_dir / "scaler.pkl",
        "feature_schema": run_dir / "feature_schema.json",
        "rf_models":       {},
        "lstm":            None,
    }

    # Locate RF models (flat in models dir, dp_rf_H<h>_<run_id>.joblib)
    for H in HORIZON_H:
        candidates = list(MODELS_DIR.glob(f"dp_rf_H{H}_*.joblib"))
        if candidates:
            paths["rf_models"][H] = sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]

    # Locate LSTM model (lstm_<run_id>.keras, flat in models dir)
    lstm_candidates = list(MODELS_DIR.glob("lstm_*.keras"))
    if lstm_candidates:
        paths["lstm"] = sorted(lstm_candidates, key=lambda p: p.stat().st_mtime)[-1]

    return paths


def load_artefacts(paths: dict, model_type: str = "both") -> dict:
    """Load scaler, feature schema, RF models, and LSTM model."""
    import joblib

    artefacts = {}

    # Feature schema
    if not paths["feature_schema"].exists():
        sys.exit(f"[ERROR] feature_schema.json not found: {paths['feature_schema']}")
    with open(paths["feature_schema"]) as f:
        schema = json.load(f)
    artefacts["feature_columns"] = schema["feature_columns"]
    artefacts["n_features"]      = schema["n_features"]
    print(f"[ARTEFACT] Feature schema loaded  — {artefacts['n_features']} features")

    # Scaler
    if not paths["scaler"].exists():
        sys.exit(f"[ERROR] scaler.pkl not found: {paths['scaler']}")
    artefacts["scaler"] = joblib.load(paths["scaler"])
    print(f"[ARTEFACT] StandardScaler loaded  — fitted on training data")

    # RandomForest models
    artefacts["rf_models"] = {}
    if model_type in ("both", "rf"):
        for H, p in paths["rf_models"].items():
            if p and p.exists():
                artefacts["rf_models"][H] = joblib.load(p)
                print(f"[ARTEFACT] RF model H={H:>2}h loaded  ← {p.name}")
            else:
                print(f"[WARN]     RF model H={H}h NOT FOUND — will be skipped.")

    # LSTM model — rebuild architecture from saved params + load weights
    artefacts["lstm_model"] = None
    if model_type in ("both", "lstm"):
        if paths["lstm"] and paths["lstm"].exists():
            # Read lstm hyper-params from the evaluation_report.json saved in the run dir
            report_path = paths["run_dir"] / "evaluation_report.json"
            if not report_path.exists():
                print(f"[WARN]     evaluation_report.json not found — cannot rebuild LSTM.")
            else:
                with open(report_path) as f:
                    eval_report = json.load(f)
                lstm_cfg = eval_report["models"]["LSTM"]["params"]
                n_features = artefacts["n_features"]

                print(f"[ARTEFACT] Rebuilding LSTM architecture from saved params …")
                model = build_lstm_model(
                    n_timesteps=WINDOW_N,
                    n_features=n_features,
                    n_diseases=len(DISEASES),
                    horizon_H=HORIZON_H,
                    lstm_cfg=lstm_cfg,
                )
                # Load weights directly from the .keras zip (bypasses Lambda deserialisation)
                _load_lstm_weights(paths["lstm"], model)
                artefacts["lstm_model"] = model
                print(f"[ARTEFACT] LSTM weights loaded     ← {paths['lstm'].name}")
        else:
            print(f"[WARN]     LSTM .keras file NOT FOUND — will be skipped.")

    return artefacts


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — INFERENCE HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def prepare_features(raw_df: pd.DataFrame,
                     feature_columns: list,
                     scaler) -> np.ndarray:
    """
    Apply full feature engineering + scaling to a raw sensor DataFrame.
    Returns a 2D float32 array (n_rows, n_features) aligned to feature_columns.
    """
    # D pipeline (disease risk)
    df_enriched = run_disease_risk_pipeline(raw_df)
    # F pipeline (full feature matrix)
    df_fe = build_feature_matrix(df_enriched)

    # Select & reorder columns to match training schema
    missing = [c for c in feature_columns if c not in df_fe.columns]
    if missing:
        print(f"  [WARN] {len(missing)} feature columns missing from engineered matrix "
              f"— filling with 0: {missing[:5]}{'…' if len(missing) > 5 else ''}")
        for c in missing:
            df_fe[c] = 0.0

    X = df_fe[feature_columns].values.astype(np.float32)
    X_scaled = scaler.transform(X).astype(np.float32)
    return X_scaled, df_fe


def rf_predict(rf_models: dict, X_scaled_2d: np.ndarray) -> dict:
    """
    Run all RF models (one per horizon) on each row of X_scaled_2d.

    Returns:  {H: np.ndarray  shape (n_rows, 5)}
    """
    results = {}
    for H, model in rf_models.items():
        preds = model.predict(X_scaled_2d)          # (n_rows, 5)
        results[H] = np.clip(preds, 0.0, 100.0)
    return results


def lstm_predict(lstm_model, X_scaled_2d: np.ndarray) -> dict:
    """
    Slide a window of WINDOW_N hours over X_scaled_2d and run LSTM inference.
    Uses the last WINDOW_N rows as the single prediction window (most recent).

    Returns: {H: np.ndarray  shape (1, 5)}
    """
    n_rows, n_feats = X_scaled_2d.shape
    if n_rows < WINDOW_N:
        raise ValueError(
            f"Need at least {WINDOW_N} hours of data for LSTM; got {n_rows}."
        )
    # Build windows: shape (n_windows, WINDOW_N, n_feats)
    n_windows = n_rows - WINDOW_N + 1
    windows   = np.stack(
        [X_scaled_2d[i:i + WINDOW_N] for i in range(n_windows)],
        axis=0
    )  # (n_windows, 24, F)

    preds_raw = lstm_model.predict(windows, verbose=0)   # dict {output_HH: (n_windows, 5)}
    results   = {}
    for H in HORIZON_H:
        key = f"output_H{H}"
        if key in preds_raw:
            results[H] = np.clip(preds_raw[key], 0.0, 100.0)
        else:
            print(f"  [WARN] LSTM output key '{key}' not found — skipping H={H}")
    return results


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — REPORT FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

def risk_label(score: float) -> str:
    if score >= HIGH_THRESH:   return "🔴 HIGH  "
    if score >= MEDIUM_THRESH: return "🟡 MEDIUM"
    return "🟢 LOW   "


def format_risk_table(
    disease_risks: dict,   # {disease: {H: score}}
    model_name: str,
) -> str:
    """Render a formatted risk table for one model prediction."""
    H_cols = "".join(f"  H{H:>3}h" for H in HORIZON_H)
    header = f"  {'Disease':<22}{'Label':<12}{H_cols}"
    sep    = "  " + "─" * (len(header) - 2)
    rows   = [f"\n  ┌── {model_name} Predictions ──", header, sep]

    for d in DISEASES:
        h_scores = disease_risks.get(d, {})
        agg      = max(h_scores.get(h, 0) for h in HORIZON_H if h <= 24)
        label    = risk_label(agg)
        scores   = "".join(f"  {h_scores.get(H, 0.0):>5.1f}" for H in HORIZON_H)
        rows.append(f"  {d:<22}{label}{scores}")

    rows.append("  └" + "─" * (len(header) - 3))
    return "\n".join(rows)


def format_alert(disease_risks: dict, model_name: str) -> str:
    """Print HIGH-risk alert lines for a single model result."""
    alerts = []
    for d in DISEASES:
        h_scores = disease_risks.get(d, {})
        max24    = max(h_scores.get(h, 0) for h in HORIZON_H if h <= 24)
        if max24 >= HIGH_THRESH:
            peak_h = min(
                (h for h in HORIZON_H if h <= 24),
                key=lambda h: abs(h_scores.get(h, 0) - max24),
            )
            alerts.append(
                f"  ⚠  [{model_name}] {d:<22} score={max24:5.1f}  "
                f"(first HIGH horizon: {peak_h}h ahead)"
            )
    return "\n".join(alerts) if alerts else f"  ✅  [{model_name}] No HIGH-risk diseases forecast."


def print_scenario_report(
    scenario_meta: dict,
    rf_disease_risks: dict,   # {disease: {H: score}} or None
    lstm_disease_risks: dict, # {disease: {H: score}} or None
    last_row: pd.Series,
):
    width = 72
    print("\n" + "═" * width)
    print(f"  SCENARIO : {scenario_meta['name']}")
    print(f"  START    : {scenario_meta['start_dt']}  ({scenario_meta['n_hours']}h window)")
    print(f"  CONTEXT  : {scenario_meta['description']}")
    print("─" * width)
    print("  LAST SENSOR READING (most recent hour):")
    print(f"    Temp={last_row.get('temp', 0):.1f}°C  "
          f"RH={last_row.get('humidity', 0):.1f}%  "
          f"VPD={last_row.get('vpd', 0):.3f}kPa  "
          f"AV={last_row.get('air_velocity', 0):.2f}m/s  "
          f"SR={last_row.get('solar_radiation', 0):.0f}W/m²  "
          f"{'Day' if last_row.get('day_night_flag', 0) else 'Night'}")
    print("─" * width)

    if rf_disease_risks:
        print(format_risk_table(rf_disease_risks, "Random Forest"))
        print(format_alert(rf_disease_risks, "RF"))

    if lstm_disease_risks:
        print(format_risk_table(lstm_disease_risks, "LSTM"))
        print(format_alert(lstm_disease_risks, "LSTM"))

    print("═" * width)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — INTERACTIVE MENU UTILITIES  +  MAIN
# ═════════════════════════════════════════════════════════════════════════════

_W = 72   # console width constant


def _banner(run_id: str, has_rf: bool, has_lstm: bool) -> None:
    """Print ASCII welcome banner after artefacts are loaded."""
    models_str = "  +  ".join(
        filter(None, [
            "Random Forest" if has_rf   else None,
            "LSTM"          if has_lstm else None,
        ])
    )
    horizons_str = " / ".join(f"H{h}h" for h in HORIZON_H)
    diseases_str = " · ".join(DISEASES)

    print("\n" + "╔" + "═" * (_W - 2) + "╗")
    print("║" + "  AgriTwin-GH  ·  Disease Progression Risk Forecasting".center(_W - 2) + "║")
    print("║" + "  Interactive Simulation Mode".center(_W - 2) + "║")
    print("╠" + "═" * (_W - 2) + "╣")
    print(f"║  Run ID   : {run_id}".ljust(_W - 1) + "║")
    print(f"║  Models   : {models_str}".ljust(_W - 1) + "║")
    print(f"║  Horizons : {horizons_str}".ljust(_W - 1) + "║")
    print(f"║  Diseases : {diseases_str}".ljust(_W - 1) + "║")
    print("╚" + "═" * (_W - 2) + "╝")


def _menu(prompt: str, options: list) -> int:
    """
    Print a numbered menu, validate user input, return 0-based index.
    Keeps prompting until a valid integer in range is entered.
    """
    print(f"\n  ┌{'─' * (_W - 4)}┐")
    print(f"  │  {prompt}".ljust(_W - 1) + "│")
    print(f"  ├{'─' * (_W - 4)}┤")
    for i, opt in enumerate(options, 1):
        line = f"  │  [{i}] {opt}"
        print(line.ljust(_W - 1) + "│")
    print(f"  └{'─' * (_W - 4)}┘")
    while True:
        try:
            raw = input(f"\n  → Enter choice (1–{len(options)}): ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return idx
            print(f"  ✗  Please enter a number between 1 and {len(options)}.")
        except (ValueError, EOFError):
            print("  ✗  Invalid input — please enter a number.")


def _stream_sensor_row(dt, row: dict, step: int, total: int, delay: float) -> None:
    """Print one row of the synthetic sensor feed with a progress bar."""
    bar_len = 20
    filled  = int(bar_len * step / total)
    bar     = "█" * filled + "░" * (bar_len - filled)
    dn      = "☀  day " if row.get("day_night_flag", 0) else "🌙 night"
    print(
        f"  {dt.strftime('%Y-%m-%d %H:%M')}  [{bar}] {step:>3}/{total}  "
        f"{dn}  "
        f"T={row.get('temp', 0):>5.1f}°C  "
        f"RH={row.get('humidity', 0):>4.0f}%  "
        f"VPD={row.get('vpd', 0):.2f}kPa  "
        f"AV={row.get('air_velocity', 0):.1f}m/s  "
        f"SR={row.get('solar_radiation', 0):>4.0f}W/m²",
        flush=True,
    )
    if delay > 0:
        time.sleep(delay)


def main():
    print("\n" + "═" * _W)
    print("  AgriTwin-GH  ·  Disease Progression Risk Forecasting System")
    print("  Initialising …")
    print("═" * _W)
    print(f"  Repo root  : {REPO_ROOT}")

    # ── 1. Discover & load artefacts ──────────────────────────────────────────
    print("\n[1/4] Discovering training artefacts …")
    paths     = find_latest_run()
    artefacts = load_artefacts(paths, model_type="both")

    feature_columns = artefacts["feature_columns"]
    scaler          = artefacts["scaler"]
    rf_models       = artefacts.get("rf_models", {})
    lstm_model      = artefacts.get("lstm_model")

    has_rf   = bool(rf_models)
    has_lstm = lstm_model is not None

    if not has_rf and not has_lstm:
        sys.exit("[ERROR] No models available. Check the models directory.")

    _banner(paths["run_id"], has_rf, has_lstm)

    # ── 2. Menu: model selection ──────────────────────────────────────────────
    print("\n[2/4] Configure simulation …")

    if has_rf and has_lstm:
        model_options = [
            "Both models  (Random Forest + LSTM)  ← recommended — compare predictions",
            "Random Forest only  — fast, snapshot-based predictor",
            "LSTM only           — sequence-aware, trend-sensitive predictor",
        ]
        model_keys = ["both", "rf", "lstm"]
    elif has_rf:
        model_options = ["Random Forest  (LSTM artefact not found)"]
        model_keys    = ["rf"]
    else:
        model_options = ["LSTM  (Random Forest artefacts not found)"]
        model_keys    = ["lstm"]

    model_idx  = _menu("Which model would you like to run?", model_options)
    model_type = model_keys[model_idx]

    # Release whichever model is not needed
    if model_type == "rf":
        lstm_model = None
    elif model_type == "lstm":
        rf_models  = {}

    # ── 3. Menu: scenario selection ───────────────────────────────────────────
    desc_limit = _W - 12   # characters available after bullet indent

    def _short_desc(d: str) -> str:
        return d if len(d) <= desc_limit else d[:desc_limit - 1] + "…"

    scenario_options = [
        f"{s['name']:<35} — {_short_desc(s['description'])}"
        for s in SCENARIOS
    ] + ["Run ALL scenarios sequentially"]

    scen_idx = _menu("Select a greenhouse scenario to simulate:", scenario_options)
    selected = SCENARIOS if scen_idx == len(SCENARIOS) else [SCENARIOS[scen_idx]]

    # ── 4. Menu: simulation speed ─────────────────────────────────────────────
    speed_options = [
        "Fast        — no delay, show final predictions only",
        "Real-time   — stream hourly sensor feed at 0.05 s/row  (~4 s for 72 h)",
        "Slow-live   — stream at 0.15 s/row, comfortable to watch (~11 s for 72 h)",
    ]
    speed_idx = _menu("Select simulation speed / display style:", speed_options)
    row_delay = [0.0, 0.05, 0.15][speed_idx]

    show_features = False
    if speed_idx == 0:
        fv_idx        = _menu("Show feature-matrix detail?",
                              ["No  — predictions only",
                               "Yes — print first 8 feature columns & matrix shape"])
        show_features = (fv_idx == 1)

    # ── 5. Run scenarios ──────────────────────────────────────────────────────
    speed_label = ["fast", "real-time stream", "slow-live stream"][speed_idx]
    print(f"\n[3/4] Starting simulation  "
          f"·  model={model_type.upper()}  "
          f"·  {len(selected)} scenario(s)  "
          f"·  {speed_label}")

    t_start = time.time()

    for s_num, scen in enumerate(selected, 1):
        # ── Scenario header ───────────────────────────────────────────────────
        print(f"\n{'╔' + '═' * (_W - 2) + '╗'}")
        print(f"║  Scenario {s_num}/{len(selected)} : {scen['name']}".ljust(_W - 1) + "║")
        desc_line = f"║  {scen['description']}"
        if len(desc_line) > _W - 1:
            desc_line = desc_line[:_W - 2] + "…"
        print(desc_line.ljust(_W - 1) + "║")
        print(f"╚{'═' * (_W - 2)}╝")

        # ── Generate synthetic sensor data ────────────────────────────────────
        print(f"\n  Generating {scen['n_hours']}h of synthetic greenhouse sensor data …")
        raw_df = generate_scenario(**{
            k: v for k, v in scen.items()
            if k not in ("name", "description")
        })

        # ── Stream sensor rows (or fast summary) ──────────────────────────────
        if row_delay > 0:
            print(f"\n  {'─' * (_W - 4)}")
            print(f"  LIVE SENSOR FEED  —  {scen['n_hours']} hours  "
                  f"({row_delay:.2f} s/row)")
            print(f"  {'─' * (_W - 4)}")
            n = len(raw_df)
            for i, (dt, row) in enumerate(raw_df.iterrows(), 1):
                _stream_sensor_row(dt, row.to_dict(), i, n, row_delay)
            print(f"  {'─' * (_W - 4)}")
            print("  ✓ Sensor stream complete — running inference …")
        else:
            last     = raw_df.iloc[-1]
            last_dt  = raw_df.index[-1]
            dn_label = "Day" if last.get("day_night_flag", 0) else "Night"
            print(f"\n  Last reading  ({last_dt.strftime('%Y-%m-%d %H:%M')}  ·  {dn_label}):")
            print(f"    T={last.get('temp', 0):.1f}°C  "
                  f"RH={last.get('humidity', 0):.0f}%  "
                  f"VPD={last.get('vpd', 0):.3f} kPa  "
                  f"AV={last.get('air_velocity', 0):.2f} m/s  "
                  f"SR={last.get('solar_radiation', 0):.0f} W/m²")

        # ── Feature engineering ───────────────────────────────────────────────
        X_scaled, df_fe = prepare_features(raw_df, feature_columns, scaler)

        if show_features:
            print(f"\n  Feature matrix shape : {X_scaled.shape}")
            print(f"  First 8 features     : {feature_columns[:8]}")

        # ── RF inference ──────────────────────────────────────────────────────
        rf_disease_risks = None
        if rf_models and model_type in ("both", "rf"):
            if row_delay > 0:
                print(f"\n  [RF]   Running Random Forest across {len(rf_models)} horizons …",
                      end=" ", flush=True)
            rf_preds = rf_predict(rf_models, X_scaled)
            rf_disease_risks = {
                d: {H: float(rf_preds[H][-1, d_idx])
                    for H in HORIZON_H if H in rf_preds}
                for d_idx, d in enumerate(DISEASES)
            }
            if row_delay > 0:
                print("done ✓")

        # ── LSTM inference ────────────────────────────────────────────────────
        lstm_disease_risks = None
        if lstm_model is not None and model_type in ("both", "lstm"):
            if row_delay > 0:
                print(f"  [LSTM] Sliding {WINDOW_N}-step window over sensor sequence …",
                      end=" ", flush=True)
            lstm_preds = lstm_predict(lstm_model, X_scaled)
            lstm_disease_risks = {
                d: {H: float(lstm_preds[H][-1, d_idx])
                    for H in HORIZON_H if H in lstm_preds}
                for d_idx, d in enumerate(DISEASES)
            }
            if row_delay > 0:
                print("done ✓")

        # ── Formatted report ──────────────────────────────────────────────────
        last_raw = raw_df.iloc[-1].to_dict()
        print_scenario_report(
            scenario_meta      = scen,
            rf_disease_risks   = rf_disease_risks,
            lstm_disease_risks = lstm_disease_risks,
            last_row           = last_raw,
        )

        # Pause between scenarios when running all
        if len(selected) > 1 and s_num < len(selected):
            print(f"\n  ↩  Press Enter to continue to scenario {s_num + 1} …", end="")
            try:
                input()
            except EOFError:
                print()

    elapsed = time.time() - t_start
    print(f"\n[4/4] Simulation complete  "
          f"·  {len(selected)} scenario(s)  "
          f"·  elapsed {elapsed:.1f} s")
    print("═" * _W + "\n")


if __name__ == "__main__":
    main()
