-- ============================================================
-- AgriTwin-GH  –  Monthly Snapshot Schema
-- Table:
--   • crop_cycles        — one row per crop run (seedling → ripe)
--   • monthly_snapshots  — one row per calendar-month per crop cycle
--
-- Apply with:
--   psql -U <user> -d agritwin_db -f database/schema/monthly_snapshots.sql
--
-- See docs/MONTHLY_SNAPSHOT_REFERENCE.md for full field descriptions.
-- ============================================================


-- ─── Crop Cycles ─────────────────────────────────────────────
-- Each row represents one complete or ongoing crop run.
-- A new run is created every time main.py is started with the
-- AGRITWIN_MONTHLY_DB=1 env-var set.  The run_number auto-increments
-- globally so cycle IDs are stable and unique across all crops.

CREATE TABLE IF NOT EXISTS crop_cycles (
    id                  SERIAL PRIMARY KEY,

    -- Human-readable cycle identifier, e.g. "cycle-001"
    cycle_label         VARCHAR(20)  NOT NULL UNIQUE,

    -- Wallclock timestamps (UTC)
    started_at          TIMESTAMP    NOT NULL DEFAULT NOW(),
    ended_at            TIMESTAMP,               -- NULL while cycle is still running
    completed           BOOLEAN      NOT NULL DEFAULT FALSE,

    -- Metadata
    crop_type           VARCHAR(50)  NOT NULL DEFAULT 'tomato',
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_crop_cycles_label  ON crop_cycles (cycle_label);
CREATE INDEX IF NOT EXISTS idx_crop_cycles_started ON crop_cycles (started_at DESC);


-- ─── Monthly Snapshots ───────────────────────────────────────
-- One row per calendar month per crop cycle.
-- Automatically inserted by MonthlySnapshotService at the
-- end of each month (detected by LoopService when the billing
-- month rolls over).

CREATE TABLE IF NOT EXISTS monthly_snapshots (
    id                  SERIAL PRIMARY KEY,

    -- Foreign key to crop_cycles
    cycle_id            INTEGER      NOT NULL REFERENCES crop_cycles(id) ON DELETE CASCADE,

    -- Calendar month this snapshot covers, e.g. "2026-04"
    billing_month       VARCHAR(7)   NOT NULL,       -- "YYYY-MM"
    month_label         VARCHAR(20)  NOT NULL,       -- "April 2026"

    -- Timestamps (UTC)
    snapshot_recorded_at TIMESTAMP   NOT NULL DEFAULT NOW(),
    month_start_ts      TIMESTAMP    NOT NULL,       -- first DT step in this month
    month_end_ts        TIMESTAMP    NOT NULL,       -- last DT step in this month

    -- ── Step counters ─────────────────────────────────────────
    total_steps         INTEGER      NOT NULL DEFAULT 0,
    mpc_solve_count     INTEGER      NOT NULL DEFAULT 0,
    image_classify_count INTEGER     NOT NULL DEFAULT 0,

    -- ── Growth stage tracking ─────────────────────────────────
    -- Stage at the first and last step of the month (from the INPUT log line)
    stage_at_month_start   VARCHAR(30)  NOT NULL DEFAULT 'seedling',
    stage_at_month_end     VARCHAR(30)  NOT NULL DEFAULT 'seedling',
    -- Stage index (0=Seedling … 5=Ripe) for quick range queries
    stage_idx_start        SMALLINT     NOT NULL DEFAULT 0,
    stage_idx_end          SMALLINT     NOT NULL DEFAULT 0,
    -- Number of stage transitions observed during the month
    stage_transitions       SMALLINT    NOT NULL DEFAULT 0,
    -- JSON array of { from_stage, to_stage, step, ts } objects
    stage_transition_log    JSONB       NOT NULL DEFAULT '[]',

    -- ── Sensor averages (from INPUT lines) ───────────────────
    avg_indoor_temp         FLOAT,      -- °C
    avg_indoor_humidity     FLOAT,      -- %
    avg_co2                 FLOAT,      -- ppm
    avg_soil_moisture       FLOAT,      -- %
    avg_light_intensity     FLOAT,      -- lux
    avg_vpd                 FLOAT,      -- kPa
    avg_leaf_wetness        FLOAT,      -- proxy 0-1
    avg_disease_risk_score  FLOAT,      -- 0-1

    -- ── Sensor min / max (from INPUT lines) ──────────────────
    min_indoor_temp         FLOAT,
    max_indoor_temp         FLOAT,
    min_indoor_humidity     FLOAT,
    max_indoor_humidity     FLOAT,
    min_co2                 FLOAT,
    max_co2                 FLOAT,
    min_soil_moisture       FLOAT,
    max_soil_moisture       FLOAT,
    max_disease_risk_score  FLOAT,

    -- ── DT setpoint errors (monthly averages) ────────────────
    avg_setpt_err_temp      FLOAT,
    avg_setpt_err_humidity  FLOAT,
    avg_setpt_err_soil      FLOAT,
    avg_setpt_err_co2       FLOAT,
    avg_setpt_err_light     FLOAT,
    avg_setpt_err_vpd       FLOAT,

    -- ── Resource totals for the month ────────────────────────
    total_energy_kwh        FLOAT        NOT NULL DEFAULT 0.0,
    total_water_l           FLOAT        NOT NULL DEFAULT 0.0,
    total_cost_inr          FLOAT        NOT NULL DEFAULT 0.0,

    -- ── Per-actuator energy and cost breakdown ────────────────
    -- Each field = cumulative kWh consumed by that actuator this month
    act_fan_speed_kwh       FLOAT        NOT NULL DEFAULT 0.0,
    act_vent_opening_kwh    FLOAT        NOT NULL DEFAULT 0.0,
    act_heater_output_kwh   FLOAT        NOT NULL DEFAULT 0.0,
    act_led_intensity_kwh   FLOAT        NOT NULL DEFAULT 0.0,
    act_fogger_duty_kwh     FLOAT        NOT NULL DEFAULT 0.0,
    act_co2_valve_kwh       FLOAT        NOT NULL DEFAULT 0.0,
    act_irrigation_kwh      FLOAT        NOT NULL DEFAULT 0.0,

    -- Corresponding INR cost per actuator (energy @ ₹7/kWh + water @ ₹4/kL)
    act_fan_speed_inr       FLOAT        NOT NULL DEFAULT 0.0,
    act_vent_opening_inr    FLOAT        NOT NULL DEFAULT 0.0,
    act_heater_output_inr   FLOAT        NOT NULL DEFAULT 0.0,
    act_led_intensity_inr   FLOAT        NOT NULL DEFAULT 0.0,
    act_fogger_duty_inr     FLOAT        NOT NULL DEFAULT 0.0,
    act_co2_valve_inr       FLOAT        NOT NULL DEFAULT 0.0,
    act_irrigation_inr      FLOAT        NOT NULL DEFAULT 0.0,

    -- Water consumed by irrigation specifically (L)
    act_irrigation_water_l  FLOAT        NOT NULL DEFAULT 0.0,

    -- ── MPC solve summary ────────────────────────────────────
    -- Averages across all MPC solves in this month
    avg_mpc_cost            FLOAT,
    mpc_converge_count      INTEGER      NOT NULL DEFAULT 0,

    -- Most-used actuator levels during MPC solves (0-1 scale, averaged)
    avg_mpc_fan             FLOAT,
    avg_mpc_vent            FLOAT,
    avg_mpc_heat            FLOAT,
    avg_mpc_led             FLOAT,
    avg_mpc_co2v            FLOAT,
    avg_mpc_fog             FLOAT,
    avg_mpc_irrig           FLOAT,

    -- ── Weather (outdoor) monthly averages ───────────────────
    avg_ext_temp            FLOAT,      -- °C
    avg_ext_humidity        FLOAT,      -- %
    avg_solar_radiation     FLOAT,      -- W/m²
    avg_wind_speed          FLOAT,      -- km/h

    -- ── Disease risk monthly summary ─────────────────────────
    -- Peak 24h-severity across all diseases for each disease category
    peak_early_blight_sev   FLOAT       NOT NULL DEFAULT 0.0,
    peak_late_blight_sev    FLOAT       NOT NULL DEFAULT 0.0,
    peak_leaf_mold_sev      FLOAT       NOT NULL DEFAULT 0.0,
    peak_powdery_mildew_sev FLOAT       NOT NULL DEFAULT 0.0,
    peak_spider_mites_sev   FLOAT       NOT NULL DEFAULT 0.0,
    disease_alert_steps     INTEGER     NOT NULL DEFAULT 0,  -- steps with risk > 0.3

    -- ── AI model cadences this month ─────────────────────────
    growth_lstm_runs        INTEGER     NOT NULL DEFAULT 0,
    disease_lstm_runs       INTEGER     NOT NULL DEFAULT 0,
    weather_forecast_runs   INTEGER     NOT NULL DEFAULT 0,

    -- Unique constraint: one row per month per cycle
    CONSTRAINT uq_monthly_snapshot UNIQUE (cycle_id, billing_month)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_ms_cycle_id     ON monthly_snapshots (cycle_id);
CREATE INDEX IF NOT EXISTS idx_ms_billing_month ON monthly_snapshots (billing_month DESC);
CREATE INDEX IF NOT EXISTS idx_ms_recorded_at  ON monthly_snapshots (snapshot_recorded_at DESC);
