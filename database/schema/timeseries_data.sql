-- ============================================================
-- AgriTwin-GH  –  Complete Timeseries Schema
-- Tables:
--   • weather_data
--   • greenhouse_data
--   • disease_progression
--   • growth_progression_hourly
--   • growth_progression_stage_summary
--   • growth_progression_cycle_summary
--   • growth_progression_metadata
--   • realtime_greenhouse_stream
--
-- Apply with:
--   psql -U <user> -d agritwin_db -f database/schema/timeseries_data.sql
-- ============================================================

-- ─── Prerequisites ──────────────────────────────────────────
-- Enable TimescaleDB extension (requires TimescaleDB to be installed):
-- CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;


-- ─── Weather Data ────────────────────────────────────────────
-- Source: data/external/Weather Data/dindigul_weather_<year>.csv

CREATE TABLE IF NOT EXISTS weather_data (
    id                  SERIAL PRIMARY KEY,
    datetime            TIMESTAMP NOT NULL,        -- Observation timestamp
    datetime_epoch      INTEGER,                   -- Unix epoch equivalent
    temp                FLOAT,                     -- Temperature in °C
    humidity            FLOAT,                     -- Relative humidity (%)
    windspeed           FLOAT,                     -- Wind speed in km/h
    solarradiation      FLOAT,                     -- Solar radiation in W/m²
    sunrise             VARCHAR(20),
    sunrise_epoch       FLOAT,
    sunset              VARCHAR(20),
    sunset_epoch        FLOAT,
    conditions          VARCHAR(100),              -- e.g. "Partially cloudy"
    description         VARCHAR(500)
);

CREATE INDEX IF NOT EXISTS idx_weather_datetime
    ON weather_data (datetime DESC);

-- TimescaleDB hypertable (run only if TimescaleDB is installed)
-- SELECT create_hypertable('weather_data', 'datetime',
--     if_not_exists => TRUE,
--     chunk_time_interval => INTERVAL '1 month');


-- ─── Greenhouse Indoor Conditions ───────────────────────────
-- Source: data/processed/Greenhouse Indoor Conditions/dindigul_greenhouse_indoor_<year>.csv
-- Merges 2024 and 2025 into a single table.

CREATE TABLE IF NOT EXISTS greenhouse_data (
    id                  SERIAL PRIMARY KEY,
    datetime            TIMESTAMP NOT NULL,        -- Observation timestamp
    indoor_temp         FLOAT,                     -- Indoor temperature in °C
    indoor_humidity     FLOAT,                     -- Indoor relative humidity (%)
    indoor_air_velocity FLOAT,                     -- Air velocity in m/s
    indoor_co2          FLOAT,                     -- CO₂ concentration in ppm
    solarradiation      FLOAT,                     -- Solar radiation in W/m²
    day_night_flag      INTEGER,                   -- 1 = Day, 0 = Night
    vpd                 FLOAT,                     -- Vapor Pressure Deficit in kPa
    dew_point           FLOAT,                     -- Dew point in °C
    leaf_wetness_proxy  FLOAT                      -- Leaf wetness proxy indicator
);

CREATE INDEX IF NOT EXISTS idx_greenhouse_datetime
    ON greenhouse_data (datetime DESC);

-- TimescaleDB hypertable (run only if TimescaleDB is installed)
-- SELECT create_hypertable('greenhouse_data', 'datetime',
--     if_not_exists => TRUE,
--     chunk_time_interval => INTERVAL '1 week');


-- ─── Disease Progression ────────────────────────────────────

CREATE TABLE IF NOT EXISTS disease_progression (
    id                          SERIAL PRIMARY KEY,
    timestamp                   TIMESTAMP NOT NULL,

    -- Cycle / stage identity
    cycle_id                    INTEGER NOT NULL,
    cycle_label                 VARCHAR(50),
    season_label                VARCHAR(50),
    stage_name                  VARCHAR(50),
    stage_index                 INTEGER,

    -- Time features
    days_from_cycle_start       FLOAT,
    day_of_year                 INTEGER,
    week_of_year                INTEGER,
    hour                        INTEGER,
    hours_in_current_stage      FLOAT,
    stage_progress_pct          FLOAT,
    total_cycle_progress_pct    FLOAT,
    is_stage_transition         BOOLEAN,

    -- Snapshot indoor environment
    indoor_temp                 FLOAT,
    indoor_humidity             FLOAT,
    indoor_air_velocity         FLOAT,
    indoor_co2                  FLOAT,
    solarradiation              FLOAT,
    day_night_flag              FLOAT,
    vpd                         FLOAT,
    dew_point                   FLOAT,
    leaf_wetness_proxy          FLOAT,

    -- Rolling / derived features
    temperature_rolling_mean_24h  FLOAT,
    humidity_rolling_mean_24h     FLOAT,
    vpd_proxy                     FLOAT,
    cumulative_gdd_like_index     FLOAT,

    -- Disease-specific columns
    disease_name                VARCHAR(50)  NOT NULL,
    disease_present_flag        INTEGER,
    disease_cycle_id            INTEGER,
    disease_cycle_stage         VARCHAR(30),
    outbreak_trigger_flag       INTEGER,
    control_action_flag         INTEGER,
    control_action_type         VARCHAR(50),
    stage_susceptibility_score  FLOAT,
    disease_risk_score          FLOAT,
    hours_since_disease_onset   FLOAT,
    current_infection_pct       FLOAT,
    infection_growth_rate_hourly FLOAT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_disease_prog_timestamp
    ON disease_progression (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_disease_prog_cycle_disease
    ON disease_progression (cycle_id, disease_name);
CREATE INDEX IF NOT EXISTS idx_disease_prog_present
    ON disease_progression (disease_present_flag)
    WHERE disease_present_flag = 1;

-- TimescaleDB hypertable (run only if TimescaleDB is installed)
-- SELECT create_hypertable('disease_progression', 'timestamp',
--     if_not_exists => TRUE,
--     chunk_time_interval => INTERVAL '1 month');


-- ─── Growth Progression – Hourly ────────────────────────────

CREATE TABLE IF NOT EXISTS growth_progression_hourly (
    id                              SERIAL PRIMARY KEY,
    timestamp                       TIMESTAMP NOT NULL,

    -- Cycle / identity
    cycle_id                        INTEGER NOT NULL,
    cycle_label                     VARCHAR(50),
    season_window                   VARCHAR(100),
    real_or_synthetic_flag          VARCHAR(20),

    -- Time features
    year                            INTEGER,
    month                           INTEGER,
    day_of_year                     INTEGER,
    week_of_year                    INTEGER,
    hour                            INTEGER,
    season_label                    VARCHAR(50),
    days_from_cycle_start           FLOAT,

    -- Stage / progression
    stage_name                      VARCHAR(50),
    stage_index                     INTEGER,
    hours_in_current_stage          FLOAT,
    days_in_current_stage           FLOAT,
    stage_duration_hours            INTEGER,
    stage_duration_days             INTEGER,
    stage_progress_pct              FLOAT,
    total_cycle_progress_pct        FLOAT,
    estimated_days_to_next_stage    FLOAT,
    estimated_hours_to_next_stage   FLOAT,
    is_stage_transition             BOOLEAN,

    -- Environment
    indoor_temp                     FLOAT,
    indoor_humidity                 FLOAT,
    indoor_air_velocity             FLOAT,
    indoor_co2                      FLOAT,
    solarradiation                  FLOAT,
    day_night_flag                  FLOAT,
    vpd                             FLOAT,
    dew_point                       FLOAT,
    leaf_wetness_proxy              FLOAT,

    -- Engineered features
    temperature_rolling_mean_24h    FLOAT,
    humidity_rolling_mean_24h       FLOAT,
    vpd_proxy                       FLOAT,
    light_period_flag               INTEGER,
    cumulative_gdd_like_index       FLOAT
);

CREATE INDEX IF NOT EXISTS idx_growth_hourly_timestamp
    ON growth_progression_hourly (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_growth_hourly_cycle_stage
    ON growth_progression_hourly (cycle_id, stage_index);

-- SELECT create_hypertable('growth_progression_hourly', 'timestamp',
--     if_not_exists => TRUE,
--     chunk_time_interval => INTERVAL '1 month');


-- ─── Growth Progression – Stage Summary ─────────────────────

CREATE TABLE IF NOT EXISTS growth_progression_stage_summary (
    id                    SERIAL PRIMARY KEY,
    cycle_id              INTEGER NOT NULL,
    stage_index           INTEGER NOT NULL,
    stage_name            VARCHAR(50) NOT NULL,
    hourly_rows           INTEGER,
    start_timestamp       TIMESTAMP,
    end_timestamp         TIMESTAMP,
    actual_days           FLOAT,
    max_stage_prog        FLOAT,
    mean_gdd              FLOAT,
    mean_indoor_temp      FLOAT,
    mean_indoor_humidity  FLOAT,
    mean_vpd              FLOAT,
    mean_solarradiation   FLOAT,
    std_indoor_temp       FLOAT,
    std_indoor_humidity   FLOAT,
    std_vpd               FLOAT,
    std_solarradiation    FLOAT
);

CREATE INDEX IF NOT EXISTS idx_stage_summary_cycle_stage
    ON growth_progression_stage_summary (cycle_id, stage_index);


-- ─── Growth Progression – Cycle Summary ─────────────────────

CREATE TABLE IF NOT EXISTS growth_progression_cycle_summary (
    id                          SERIAL PRIMARY KEY,
    cycle_id                    INTEGER NOT NULL UNIQUE,
    cycle_label                 VARCHAR(50),
    season_window               VARCHAR(100),
    cycle_start                 DATE,
    cycle_end                   DATE,
    total_days                  INTEGER,
    days_seedling               INTEGER,
    days_early_vegetative       INTEGER,
    days_flowering_initiation   INTEGER,
    days_flowering              INTEGER,
    days_unripe                 INTEGER,
    days_ripe                   INTEGER,
    total_hourly_rows           INTEGER
);


-- ─── Growth Progression – Metadata ──────────────────────────

CREATE TABLE IF NOT EXISTS growth_progression_metadata (
    id                  SERIAL PRIMARY KEY,
    project             VARCHAR(100),
    crop                VARCHAR(100),
    location            VARCHAR(100),
    greenhouse_system   VARCHAR(100),
    notebook_name       VARCHAR(200),
    created_on          TIMESTAMP,
    total_hourly_rows   INTEGER,
    earliest_timestamp  TIMESTAMP,
    latest_timestamp    TIMESTAMP,
    metadata_json       JSONB,          -- Full raw metadata JSON
    loaded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);


-- ─── Real-Time Closed-Loop Stream ───────────────────────────
-- Written by:   scripts/run_realtime_loop.py
-- Read by:      RealtimeMPCInputPreparation.get_latest_greenhouse_row()
--
-- One row per 5-minute DT simulation step in a real-time MPC run.
-- Column names deliberately mirror greenhouse_data so that
-- MPCInputPreparation can read from either table via a single dict schema.
--
-- Source values:
--   "bootstrap"  — initial sensor values seeded from greenhouse_data
--   "dt_sim"     — state produced by DigitalTwinEngine.step() (ARX physics)

CREATE TABLE IF NOT EXISTS realtime_greenhouse_stream (
    id                      SERIAL PRIMARY KEY,

    -- Run identity
    run_id                  VARCHAR(64)    NOT NULL,   -- rt_YYYYMMDD_HHMMSS
    step_index              INTEGER        NOT NULL,   -- step counter within run
    source                  VARCHAR(20)    NOT NULL DEFAULT 'dt_sim',
                                                       -- bootstrap | dt_sim

    -- Timestamps
    datetime                TIMESTAMP      NOT NULL,   -- logical simulation time (UTC)
    created_at              TIMESTAMP      NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Indoor climate (mirrors greenhouse_data columns)
    indoor_temp             FLOAT,                     -- °C
    indoor_humidity         FLOAT,                     -- %
    indoor_air_velocity     FLOAT,                     -- m/s
    indoor_co2              FLOAT,                     -- ppm
    solarradiation          FLOAT,                     -- W/m²
    day_night_flag          INTEGER,                   -- 1=day, 0=night
    vpd                     FLOAT,                     -- kPa
    dew_point               FLOAT,                     -- °C
    leaf_wetness_proxy      FLOAT,                     -- 0–1

    -- Derived state
    soil_moisture           FLOAT,                     -- % (0–100)
    disease_risk_score      FLOAT,                     -- 0–1
    growth_stage            VARCHAR(50),               -- canonical label
    growth_stage_index      INTEGER,                   -- 0–5
    disease_classification  VARCHAR(100),              -- top disease class

    -- Applied actuators (outputs of MPCSolver.solve())
    fan_speed               FLOAT,                     -- duty 0–1
    vent_opening            FLOAT,                     -- fraction 0–1
    heater_output           FLOAT,                     -- duty 0–1
    led_intensity           FLOAT,                     -- duty 0–1
    fogger_duty             FLOAT,                     -- duty 0–1
    co2_valve_pct           FLOAT,                     -- 0–1
    irrigation_qty          FLOAT,                     -- L/step

    -- MPC decision metadata
    mpc_ran                 BOOLEAN        DEFAULT FALSE,
    mpc_converged           BOOLEAN        DEFAULT FALSE,
    mpc_fallback_used       BOOLEAN        DEFAULT FALSE,
    step_cost               FLOAT,                     -- MPC objective value
    solve_time_ms           FLOAT,                     -- solver wall time ms

    -- Crop progression context
    hours_in_current_stage  FLOAT,                     -- elapsed h in stage
    stage_progress_pct      FLOAT,                     -- 0–100
    hours_to_stage_transition FLOAT,                   -- LSTM-estimated hours

    -- Resource accounting
    step_energy_kwh         FLOAT,                     -- kWh this step
    cumulative_energy_kwh   FLOAT,                     -- kWh since run start
    cumulative_water_litres FLOAT,                     -- L since run start

    -- Alert level
    alert_level             VARCHAR(10)    DEFAULT 'GREEN'  -- GREEN|YELLOW|RED
);

CREATE INDEX IF NOT EXISTS idx_rt_stream_run_step
    ON realtime_greenhouse_stream (run_id, step_index);

CREATE INDEX IF NOT EXISTS idx_rt_stream_datetime
    ON realtime_greenhouse_stream (datetime DESC);

-- TimescaleDB hypertable (run only if TimescaleDB is installed)
-- SELECT create_hypertable('realtime_greenhouse_stream', 'datetime',
--     if_not_exists => TRUE,
--     chunk_time_interval => INTERVAL '1 day');

