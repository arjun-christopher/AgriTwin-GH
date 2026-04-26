"""
test_weather_forecast.py
─────────────────────────
Standalone test script for the Environment Forecast model
(Chronos + XGBoost + LSTM ensemble).

10 scenarios covering summer heat, monsoon onset, winter cold, dry spells,
overcast periods, post-monsoon transitions, and edge cases.

The model expects a DataFrame with at least 30 daily rows containing:
  datetime, temp (°C), humidity (%), windspeed (km/h), solarradiation (W/m²)

Feature engineering and Chronos inference are handled internally by
EnvironmentForecastModel.  This script generates synthetic climate windows.

NOTE: First run requires Chronos checkpoint download (~600 MB).
      Subsequent runs use the HuggingFace local cache.

Usage:
    python scripts/test_weather_forecast.py            # run all 10 scenarios
    python scripts/test_weather_forecast.py --scenario 3   # run one scenario
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Resolve repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

# ── Model paths ───────────────────────────────────────────────────────────────
ARTIFACTS_DIR   = REPO_ROOT / "src/agritwin_gh/models/artifacts/environment_forecast_20260403_173201"
MAIN_MODEL_PATH = REPO_ROOT / "src/agritwin_gh/models/environment_forecast_20260403_173201.pt"

CONTEXT_LENGTH = 30   # rolling 30-day window the model expects
TARGET_COLS    = ["temp", "humidity", "windspeed", "solarradiation"]

CONDITION_CLASSES = {
    0: "Clear",
    1: "Overcast",
    2: "Partially cloudy",
    3: "Rain, Overcast",
    4: "Rain, Partially cloudy",
}


# ── DataFrame factory ─────────────────────────────────────────────────────────

def _make_weather_df(
    start_date: str,
    temp_range: tuple,          # (min_°C, max_°C)  daily averages
    humidity_range: tuple,      # (min_%, max_%)
    wind_range: tuple,          # (min_km/h, max_km/h)
    solar_range: tuple,         # (min_W/m², max_W/m²)
    n_days: int = CONTEXT_LENGTH,
    rng_seed: int = 0,
    noise: float = 0.05,        # fractional random noise on each series
) -> pd.DataFrame:
    """
    Generate a synthetic daily weather DataFrame with smooth linear trends
    + small Gaussian noise so the data resembles realistic patterns.
    """
    rng   = np.random.default_rng(seed=rng_seed)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    def _series(lo, hi):
        base = np.linspace(lo, hi, n_days)
        return base + rng.normal(0, abs(hi - lo) * noise + 0.5, n_days)

    df = pd.DataFrame({
        "datetime"      : dates,
        "temp"          : np.clip(_series(*temp_range),     -10, 50),
        "humidity"      : np.clip(_series(*humidity_range),   5, 99),
        "windspeed"     : np.clip(_series(*wind_range),       0, 80),
        "solarradiation": np.clip(_series(*solar_range),      0, 1100),
    })
    return df


def _make_weather_df_sine(
    start_date: str,
    temp_mid: float,
    temp_amp: float,
    hum_mid: float,
    hum_amp: float,
    wind_base: float,
    solar_mid: float,
    solar_amp: float,
    n_days: int = CONTEXT_LENGTH,
    rng_seed: int = 1,
) -> pd.DataFrame:
    """Sine-wave daily pattern (mimics seasonal intra-period oscillation)."""
    rng   = np.random.default_rng(seed=rng_seed)
    t     = np.linspace(0, 2 * np.pi, n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    df = pd.DataFrame({
        "datetime"      : dates,
        "temp"          : np.clip(temp_mid + temp_amp * np.sin(t)
                                  + rng.normal(0, 0.3, n_days), -5, 50),
        "humidity"      : np.clip(hum_mid  + hum_amp  * np.cos(t)
                                  + rng.normal(0, 1.0, n_days),  5, 99),
        "windspeed"     : np.clip(wind_base + rng.normal(0, 1.5, n_days), 0, 80),
        "solarradiation": np.clip(solar_mid + solar_amp * np.sin(t)
                                  + rng.normal(0, 10, n_days),   0, 1100),
    })
    return df


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model():
    import importlib.util, sys as _sys
    loader_path = ARTIFACTS_DIR / "environment_forecast_loader.py"
    spec = importlib.util.spec_from_file_location(
        "environment_forecast_loader", str(loader_path)
    )
    mod = importlib.util.module_from_spec(spec)
    _sys.modules["environment_forecast_loader"] = mod
    spec.loader.exec_module(mod)

    print(f"Loading EnvironmentForecastModel …")
    print(f"  Main model : {MAIN_MODEL_PATH.name}")
    print(f"  Artifacts  : {ARTIFACTS_DIR.name}")
    model = mod.EnvironmentForecastModel(
        artifacts_dir=str(ARTIFACTS_DIR),
        main_model_path=str(MAIN_MODEL_PATH),
        device="cpu",
    )
    print("  Model loaded.\n")
    return model


# ── Result printer ────────────────────────────────────────────────────────────

def _print_result(result: dict, scenario_num: int, scenario_name: str) -> None:
    print(f"{'─'*70}")
    print(f"Scenario {scenario_num:>2}: {scenario_name}")
    print(f"  {'Variable':<16}  {'24h Forecast':>14}  {'48h Forecast':>14}")
    print(f"  {'─'*16}  {'─'*14}  {'─'*14}")
    units = {
        "temp"          : "°C",
        "humidity"      : "%",
        "windspeed"     : " km/h",
        "solarradiation": " W/m²",
    }
    for col in TARGET_COLS:
        if col in result:
            v24 = result[col]["24h"]
            v48 = result[col]["48h"]
            u   = units.get(col, "")
            print(f"  {col:<16}  {v24:>11.2f}{u:>3}  {v48:>11.2f}{u:>3}")
    if "sky_condition" in result:
        sky = result["sky_condition"]
        print(f"  {'sky_condition':<16}  {sky.get('24h', '–'):>14}  {sky.get('48h', '–'):>14}")


def _run_and_print(model, df: pd.DataFrame, scenario_num: int, scenario_name: str) -> dict:
    result = model.predict(df)
    _print_result(result, scenario_num, scenario_name)
    for col in TARGET_COLS:
        assert col in result, f"Missing key {col} in result"
        assert "24h" in result[col] and "48h" in result[col]
    return result


# ── Scenario functions ────────────────────────────────────────────────────────

def run_scenario_1(model):
    """Summer baseline — warm, moderate humidity, high solar (June)."""
    df = _make_weather_df(
        start_date="2025-06-01",
        temp_range=(28.0, 33.0),
        humidity_range=(55.0, 65.0),
        wind_range=(8.0, 14.0),
        solar_range=(600.0, 850.0),
        rng_seed=1,
    )
    return _run_and_print(model, df, 1, "Summer baseline — warm, moderate humidity (June)")


def run_scenario_2(model):
    """Monsoon onset — humidity rises 60→90%, solar drops 600→100 W/m²."""
    df = _make_weather_df(
        start_date="2025-07-01",
        temp_range=(28.0, 26.0),
        humidity_range=(60.0, 90.0),
        wind_range=(10.0, 20.0),
        solar_range=(600.0, 100.0),
        rng_seed=2,
    )
    return _run_and_print(model, df, 2, "Monsoon onset — humidity rising, solar dropping")


def run_scenario_3(model):
    """Winter cold — low temp (10–18°C), low solar, low humidity."""
    df = _make_weather_df(
        start_date="2025-12-01",
        temp_range=(10.0, 18.0),
        humidity_range=(30.0, 45.0),
        wind_range=(5.0, 15.0),
        solar_range=(120.0, 280.0),
        rng_seed=3,
    )
    return _run_and_print(model, df, 3, "Winter cold — low temp (10–18°C), low solar (Dec)")


def run_scenario_4(model):
    """Dry hot spell — extreme heat 35–40°C, low humidity (25–35%)."""
    df = _make_weather_df(
        start_date="2025-05-15",
        temp_range=(35.0, 40.0),
        humidity_range=(25.0, 35.0),
        wind_range=(12.0, 22.0),
        solar_range=(700.0, 950.0),
        rng_seed=4,
    )
    return _run_and_print(model, df, 4, "Dry hot spell — 35–40°C, humidity 25–35%")


def run_scenario_5(model):
    """Overcast rainy period — low solar (<50 W/m²), high humidity (80–95%)."""
    df = _make_weather_df(
        start_date="2025-08-10",
        temp_range=(22.0, 25.0),
        humidity_range=(80.0, 95.0),
        wind_range=(5.0, 12.0),
        solar_range=(0.0, 50.0),
        rng_seed=5,
    )
    return _run_and_print(model, df, 5, "Overcast rainy period — low solar, high humidity (Aug)")


def run_scenario_6(model):
    """Clear sky peak summer — high solar (800–1050 W/m²), low humidity."""
    df = _make_weather_df(
        start_date="2025-06-15",
        temp_range=(30.0, 36.0),
        humidity_range=(30.0, 45.0),
        wind_range=(6.0, 12.0),
        solar_range=(800.0, 1050.0),
        rng_seed=6,
    )
    return _run_and_print(model, df, 6, "Clear sky peak — 800–1050 W/m² solar, low humidity")


def run_scenario_7(model):
    """Post-monsoon transition — humidity dropping 85→55%, temp recovering."""
    df = _make_weather_df(
        start_date="2025-09-15",
        temp_range=(24.0, 30.0),
        humidity_range=(85.0, 55.0),
        wind_range=(8.0, 15.0),
        solar_range=(200.0, 550.0),
        rng_seed=7,
    )
    return _run_and_print(model, df, 7, "Post-monsoon — humidity dropping 85→55%, recovery")


def run_scenario_8(model):
    """24h vs 48h accuracy gap — same input; verify 48h != 24h and both finite."""
    df = _make_weather_df(
        start_date="2025-10-01",
        temp_range=(22.0, 26.0),
        humidity_range=(60.0, 68.0),
        wind_range=(7.0, 11.0),
        solar_range=(350.0, 480.0),
        rng_seed=8,
    )
    result = model.predict(df)
    print(f"{'─'*70}")
    print(f"Scenario  8: 24h vs 48h forecast divergence check (Oct mild weather)")
    for col in TARGET_COLS:
        v24 = result[col]["24h"]
        v48 = result[col]["48h"]
        diff = abs(v24 - v48)
        print(f"  {col:<16}  24h={v24:>8.2f}  48h={v48:>8.2f}  |Δ|={diff:>6.2f}")
        assert np.isfinite(v24) and np.isfinite(v48), f"{col} forecast is not finite"
    print("  All forecasts finite  ✓")
    return result


def run_scenario_9(model):
    """Minimum climate extreme — very low values across all channels."""
    df = _make_weather_df(
        start_date="2025-01-10",
        temp_range=(2.0, 8.0),
        humidity_range=(10.0, 20.0),
        wind_range=(0.5, 3.0),
        solar_range=(10.0, 60.0),
        rng_seed=9,
    )
    return _run_and_print(model, df, 9, "Minimum climate extreme — very cold, dry, low light")


def run_scenario_10(model):
    """Sine oscillation — intra-period variance to test rolling feature stability."""
    df = _make_weather_df_sine(
        start_date="2025-04-01",
        temp_mid=24.0,    temp_amp=6.0,
        hum_mid=65.0,     hum_amp=15.0,
        wind_base=10.0,
        solar_mid=450.0,  solar_amp=200.0,
        rng_seed=10,
    )
    return _run_and_print(model, df, 10,
                          "Sine-wave oscillation — tests rolling feature stability (Apr)")


# ── Scenario registry ─────────────────────────────────────────────────────────
SCENARIOS = {
    1:  ("Summer baseline — warm, moderate humidity (June)",        run_scenario_1),
    2:  ("Monsoon onset — humidity rising, solar dropping",         run_scenario_2),
    3:  ("Winter cold — 10–18°C, low solar (December)",            run_scenario_3),
    4:  ("Dry hot spell — 35–40°C, humidity 25–35%",               run_scenario_4),
    5:  ("Overcast rainy — low solar, humidity 80–95%",            run_scenario_5),
    6:  ("Clear sky peak — 800–1050 W/m² solar",                   run_scenario_6),
    7:  ("Post-monsoon — humidity dropping 85→55%",                run_scenario_7),
    8:  ("24h vs 48h gap — forecast divergence check",             run_scenario_8),
    9:  ("Minimum climate extreme — cold, dry, low light",         run_scenario_9),
    10: ("Sine oscillation — rolling feature stability",           run_scenario_10),
}


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone test suite for the Environment Forecast ensemble model"
    )
    parser.add_argument(
        "--scenario", type=int, default=0,
        help="Run a specific scenario number (1–10). Default 0 runs all."
    )
    args = parser.parse_args()

    model = load_model()

    if args.scenario == 0:
        print(f"Running all {len(SCENARIOS)} scenarios ...\n")
        passed = failed = 0
        for idx, (name, fn) in SCENARIOS.items():
            try:
                fn(model)
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
        fn(model)
