# Monthly Snapshot Reference

AgriTwin-GH records aggregated greenhouse telemetry into the database at every calendar-month boundary.  Each row in `monthly_snapshots` is a complete month-level summary for one crop cycle — combining sensor averages, resource totals, per-actuator billing, MPC performance, disease peaks, and AI model run counts.

---

## Quick Start

### 1 — Apply the schema

**PostgreSQL**
```bash
psql -d agritwin_db -f database/schema/monthly_snapshots.sql
```

**SQLite** (dev default) — tables are created automatically when the feature is enabled and the first step is ingested.  However, to inspect the schema independently:
```bash
sqlite3 data/processed/agritwin.db < database/schema/monthly_snapshots.sql
```

### 2 — Enable the feature

Set the env var before starting `main.py`:
```powershell
# Windows PowerShell
$env:AGRITWIN_MONTHLY_DB = "1"
python main.py
```
```bash
# Linux / macOS
AGRITWIN_MONTHLY_DB=1 python main.py
```

The default is `0` (disabled) so the DT loop runs normally without attempting DB writes.

### 3 — Seed mock data for demo

```bash
python scripts/seed_monthly_mock.py
```

Inserts 2 crop cycles (cycle-001, cycle-002) with 3 monthly snapshot rows.  Safe to run multiple times — uses `ON CONFLICT DO NOTHING`.

### 4 — Display the table

```bash
# Compact table (20 rows)
python scripts/show_monthly_snapshots.py

# Last 5 rows only
python scripts/show_monthly_snapshots.py --limit 5

# Filter to one cycle
python scripts/show_monthly_snapshots.py --cycle cycle-001

# Detailed multi-line output per row
python scripts/show_monthly_snapshots.py --detail
```

### 5 — Run tests

```bash
uv run python tests/test_monthly_snapshot.py
```

---

## Architecture

```
main.py (AGRITWIN_MONTHLY_DB=1)
│
└─► LoopService.start_loop()
       └─► MonthlySnapshotService.initialise()
               └─► INSERT INTO crop_cycles (cycle_label)

    LoopService.run_one_step()   (every 5 min)
       └─► accumulate_resources()          ← existing path
       └─► MonthlySnapshotService.ingest_step()
               └─► MonthlyAccumulator.accumulate()

    When calendar month changes:
       └─► MonthlySnapshotService._flush_month()
               └─► INSERT INTO monthly_snapshots
               └─► MonthlyAccumulator reset

    LoopService.stop_loop()
       └─► MonthlySnapshotService.flush_current_month()
               └─► Partial-month row written on shutdown
```

### Key components

| Component | File | Role |
|---|---|---|
| `MonthlyAccumulator` | `services/monthly_snapshot_service.py` | In-memory per-step running totals / min / max |
| `MonthlySnapshotService` | `services/monthly_snapshot_service.py` | Ingests steps, detects rollover, writes DB rows |
| `LoopService` | `services/loop_service.py` | Calls `ingest_step()` after each DT step |
| Schema | `database/schema/monthly_snapshots.sql` | Two tables: `crop_cycles` + `monthly_snapshots` |

---

## Table: `crop_cycles`

One row per crop run (seedling → ripe journey).  Created at `LoopService.start_loop()` when `AGRITWIN_MONTHLY_DB=1`.

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Auto-incrementing integer primary key |
| `cycle_label` | TEXT UNIQUE | Human-readable label, e.g. `cycle-001` |
| `started_at` | TIMESTAMPTZ | When `start_loop()` was called |
| `ended_at` | TIMESTAMPTZ | When the cycle was marked complete (Ripe stage reached) |
| `completed` | BOOLEAN | `TRUE` once the crop reaches the Ripe stage |
| `crop_type` | TEXT | Always `'tomato'` for now (extensible) |
| `notes` | TEXT | Free-text notes for demo or operator use |

---

## Table: `monthly_snapshots`

One row per calendar month per crop cycle.  Written automatically at each month boundary while the DT loop is running.

### Identity

| Column | Type | Description |
|---|---|---|
| `id` | SERIAL PK | Auto-incrementing row ID |
| `cycle_id` | INTEGER FK | References `crop_cycles.id` |
| `billing_month` | TEXT | `"YYYY-MM"` — the calendar month this row covers |
| `month_label` | TEXT | Human-readable, e.g. `"April 2026"` |
| `month_start_ts` | TIMESTAMPTZ | Timestamp of the first step in this month |
| `month_end_ts` | TIMESTAMPTZ | Timestamp of the last step before rollover |
| `snapshot_recorded_at` | TIMESTAMPTZ | When the row was written to the DB |

Unique constraint: `(cycle_id, billing_month)`.

### Step counters

| Column | Type | Description |
|---|---|---|
| `total_steps` | INTEGER | Total 5-minute DT steps in the month |
| `mpc_solve_count` | INTEGER | Steps where MPC CVXPY solver ran |
| `image_classify_count` | INTEGER | Steps where the CNN image classifiers ran |

### Growth stage

| Column | Type | Description |
|---|---|---|
| `stage_at_month_start` | TEXT | Stage label at start of month (e.g. `"seedling"`) |
| `stage_at_month_end` | TEXT | Stage label at end of month |
| `stage_idx_start` | INTEGER | 0-based stage index at month start |
| `stage_idx_end` | INTEGER | 0-based stage index at month end |
| `stage_transitions` | INTEGER | Number of stage changes that occurred during the month |
| `stage_transition_log` | JSONB/TEXT | Array of `{from_stage, to_stage, step, ts}` objects |

Stage index mapping: 0=seedling, 1=early vegetative, 2=flowering initiation, 3=flowering, 4=unripe, 5=ripe.

### Sensor averages & extremes

All values computed as arithmetic mean over `total_steps` steps.

| Column | Unit | Description |
|---|---|---|
| `avg_indoor_temp` | °C | Mean indoor temperature |
| `avg_indoor_humidity` | % RH | Mean relative humidity |
| `avg_co2` | ppm | Mean CO₂ concentration |
| `avg_soil_moisture` | % | Mean volumetric soil moisture |
| `avg_light_intensity` | lux | Mean light intensity (LED + natural) |
| `avg_vpd` | kPa | Mean vapour pressure deficit |
| `avg_leaf_wetness` | 0–1 | Mean leaf wetness proxy |
| `avg_disease_risk_score` | 0–1 | Mean composite disease risk |
| `min_indoor_temp` / `max_indoor_temp` | °C | Monthly temperature range |
| `min_indoor_humidity` / `max_indoor_humidity` | % | Monthly humidity range |
| `min_co2` / `max_co2` | ppm | Monthly CO₂ range |
| `min_soil_moisture` / `max_soil_moisture` | % | Monthly soil moisture range |
| `max_disease_risk_score` | 0–1 | Peak risk score in the month |

### Setpoint errors (averages)

Signed deviation of actual sensor value from MPC setpoint.  Positive = above setpoint, negative = below.

| Column | Unit |
|---|---|
| `avg_setpt_err_temp` | °C |
| `avg_setpt_err_humidity` | % |
| `avg_setpt_err_soil` | % |
| `avg_setpt_err_co2` | ppm |
| `avg_setpt_err_light` | lux |
| `avg_setpt_err_vpd` | kPa |

### Resource totals

| Column | Unit | Description |
|---|---|---|
| `total_energy_kwh` | kWh | Total electrical energy consumed this month |
| `total_water_l` | L | Total irrigation water used this month |
| `total_cost_inr` | ₹ | `energy_kwh × ₹7.00 + water_l × ₹0.004` |

### Per-actuator breakdown

Energy (kWh) and cost (₹) for each of the 7 actuators.

| Actuator key | Rated power | Columns |
|---|---|---|
| `fan_speed` | 0.75 kW | `act_fan_speed_kwh`, `act_fan_speed_inr` |
| `vent_opening` | 0.10 kW | `act_vent_opening_kwh`, `act_vent_opening_inr` |
| `heater_output` | 3.00 kW | `act_heater_output_kwh`, `act_heater_output_inr` |
| `led_intensity` | 2.00 kW | `act_led_intensity_kwh`, `act_led_intensity_inr` |
| `fogger_duty` | 0.20 kW | `act_fogger_duty_kwh`, `act_fogger_duty_inr` |
| `co2_valve_pct` | 0.05 kW | `act_co2_valve_kwh`, `act_co2_valve_inr` |
| `irrigation_qty` | 0.15 kW + 10 L/step | `act_irrigation_kwh`, `act_irrigation_inr`, `act_irrigation_water_l` |

Irrigation cost = energy cost + water cost (`water_l × ₹0.004`).

### MPC solver summary

Averages are computed over steps where MPC ran (`mpc_solve_count` steps).

| Column | Type | Description |
|---|---|---|
| `avg_mpc_cost` | REAL | Mean CVXPY objective value (dimensionless) |
| `mpc_converge_count` | INTEGER | Steps where solver converged (no fallback) |
| `avg_mpc_fan` … `avg_mpc_irrig` | REAL | Mean actuator setpoints issued by MPC (0–1) |

### External weather averages

Averages over `total_steps` steps.

| Column | Unit | Description |
|---|---|---|
| `avg_ext_temp` | °C | Mean outdoor temperature |
| `avg_ext_humidity` | % RH | Mean outdoor relative humidity |
| `avg_solar_radiation` | W/m² | Mean global solar irradiance |
| `avg_wind_speed` | km/h | Mean wind speed |

### Disease peaks

Peak (maximum) severity values recorded during the month for each of the five disease models.  Severity is a 0–1 normalised LSTM output.

| Column | Disease |
|---|---|
| `peak_early_blight_sev` | Early Blight |
| `peak_late_blight_sev` | Late Blight |
| `peak_leaf_mold_sev` | Leaf Mold |
| `peak_powdery_mildew_sev` | Powdery Mildew |
| `peak_spider_mites_sev` | Spider Mites |
| `disease_alert_steps` | Steps with `disease_risk_score > 0.3` |

### AI model run counts

| Column | Description |
|---|---|
| `growth_lstm_runs` | Steps where the growth-progression LSTM ran |
| `disease_lstm_runs` | Steps where the disease-progression LSTM ran |
| `weather_forecast_runs` | Steps where the weather forecast model ran |

---

## Environment Variables Summary

| Variable | Default | Description |
|---|---|---|
| `AGRITWIN_MONTHLY_DB` | `0` | `1` = enable monthly snapshot persistence |
| `AGRITWIN_BACKGROUND_LOOP` | `1` | DT loop advances automatically every 5 min |
| `AGRITWIN_NO_FRONTEND` | `0` | `1` = skip Vite dev server |

---

## Demo Queries

### List all months for all cycles
```sql
SELECT cc.cycle_label, ms.billing_month, ms.month_label,
       ms.total_steps, ms.total_energy_kwh, ms.total_water_l, ms.total_cost_inr,
       ms.stage_at_month_start, ms.stage_at_month_end
FROM monthly_snapshots ms
JOIN crop_cycles cc ON cc.id = ms.cycle_id
ORDER BY ms.billing_month, cc.cycle_label;
```

### Resource totals by cycle
```sql
SELECT cc.cycle_label,
       COUNT(ms.id)                    AS months_recorded,
       SUM(ms.total_energy_kwh)        AS lifetime_kwh,
       SUM(ms.total_water_l)           AS lifetime_water_l,
       SUM(ms.total_cost_inr)          AS lifetime_cost_inr
FROM monthly_snapshots ms
JOIN crop_cycles cc ON cc.id = ms.cycle_id
GROUP BY cc.cycle_label
ORDER BY cc.cycle_label;
```

### Peak disease month
```sql
SELECT billing_month,
       GREATEST(peak_early_blight_sev, peak_late_blight_sev,
                peak_leaf_mold_sev, peak_powdery_mildew_sev,
                peak_spider_mites_sev) AS worst_disease_sev
FROM monthly_snapshots
ORDER BY worst_disease_sev DESC
LIMIT 1;
```

---

## Files Created / Modified

| Path | Purpose |
|---|---|
| `database/schema/monthly_snapshots.sql` | DDL — two-table schema |
| `src/agritwin_gh/services/monthly_snapshot_service.py` | Core service + mock data & demo helpers |
| `src/agritwin_gh/services/loop_service.py` | Wired `ingest_step()` + flush on stop |
| `main.py` | `AGRITWIN_MONTHLY_DB` env var documented |
| `scripts/seed_monthly_mock.py` | Demo seed script (2 cycles, 3 rows) |
| `scripts/show_monthly_snapshots.py` | Demo display script (compact + detail modes) |
| `tests/test_monthly_snapshot.py` | 6 unit tests (in-memory SQLite) |
| `docs/MONTHLY_SNAPSHOT_REFERENCE.md` | This file |
