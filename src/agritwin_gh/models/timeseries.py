"""
Time-series data models for AgriTwin-GH
Supports both regular PostgreSQL and TimescaleDB
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Boolean, Date, Index, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class WeatherData(Base):
    """Weather data time-series table"""
    __tablename__ = 'weather_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    datetime = Column(DateTime, nullable=False, index=True)
    datetime_epoch = Column(Integer)
    temp = Column(Float, comment='Temperature in Celsius')
    humidity = Column(Float, comment='Humidity percentage')
    windspeed = Column(Float, comment='Wind speed in km/h')
    solarradiation = Column(Float, comment='Solar radiation in W/m²')
    sunrise = Column(String(20))
    sunrise_epoch = Column(Float)
    sunset = Column(String(20))
    sunset_epoch = Column(Float)
    conditions = Column(String(100))
    description = Column(String(500))
    
    # Create composite index for time-based queries
    __table_args__ = (
        Index('idx_weather_datetime', 'datetime'),
    )
    
    def __repr__(self):
        return f"<WeatherData(datetime={self.datetime}, temp={self.temp}, humidity={self.humidity})>"


class GreenhouseData(Base):
    """Greenhouse indoor conditions time-series table"""
    __tablename__ = 'greenhouse_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    datetime = Column(DateTime, nullable=False, index=True)
    indoor_temp = Column(Float, comment='Indoor temperature in Celsius')
    indoor_humidity = Column(Float, comment='Indoor humidity percentage')
    indoor_air_velocity = Column(Float, comment='Indoor air velocity in m/s')
    indoor_co2 = Column(Float, comment='CO2 concentration in ppm')
    solarradiation = Column(Float, comment='Solar radiation in W/m²')
    day_night_flag = Column(Integer, comment='Day (1) or Night (0) indicator')
    vpd = Column(Float, comment='Vapor Pressure Deficit in kPa')
    dew_point = Column(Float, comment='Dew point temperature in Celsius')
    leaf_wetness_proxy = Column(Float, comment='Leaf wetness proxy indicator')
    
    # Create composite index for time-based queries
    __table_args__ = (
        Index('idx_greenhouse_datetime', 'datetime'),
    )
    
    def __repr__(self):
        return f"<GreenhouseData(datetime={self.datetime}, indoor_temp={self.indoor_temp}, indoor_humidity={self.indoor_humidity})>"


class DiseaseProgressionData(Base):
    """Disease progression time-series table (hourly, per-disease per-cycle)"""
    __tablename__ = 'disease_progression'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Cycle / stage identity
    cycle_id = Column(Integer, nullable=False, comment='Crop cycle ID (1-4)')
    cycle_label = Column(String(50), comment='e.g. kharif_2024')
    season_label = Column(String(50), comment='e.g. southwest_monsoon')
    stage_name = Column(String(50), comment='Growth stage name')
    stage_index = Column(Integer, comment='Growth stage index (0-5)')

    # Time features
    days_from_cycle_start = Column(Float)
    day_of_year = Column(Integer)
    week_of_year = Column(Integer)
    hour = Column(Integer)
    hours_in_current_stage = Column(Float)
    stage_progress_pct = Column(Float)
    total_cycle_progress_pct = Column(Float)
    is_stage_transition = Column(Boolean)

    # Environment (snapshot from indoor conditions)
    indoor_temp = Column(Float)
    indoor_humidity = Column(Float)
    indoor_air_velocity = Column(Float)
    indoor_co2 = Column(Float)
    solarradiation = Column(Float)
    day_night_flag = Column(Float)
    vpd = Column(Float)
    dew_point = Column(Float)
    leaf_wetness_proxy = Column(Float)

    # Rolling / derived features
    temperature_rolling_mean_24h = Column(Float)
    humidity_rolling_mean_24h = Column(Float)
    vpd_proxy = Column(Float)
    cumulative_gdd_like_index = Column(Float)

    # Disease-specific columns
    disease_name = Column(String(50), nullable=False, comment='Disease type (e.g. early_blight)')
    disease_present_flag = Column(Integer, comment='1 = active outbreak, 0 = absent')
    disease_cycle_id = Column(Integer, comment='Disease cycle ID (unique per outbreak)')
    disease_cycle_stage = Column(String(30), comment='none | latent | active | decline')
    outbreak_trigger_flag = Column(Integer, comment='1 if this row triggered an outbreak')
    control_action_flag = Column(Integer, comment='1 if control action was applied')
    control_action_type = Column(String(50), comment='none | fungicide | biocontrol | pruning')
    stage_susceptibility_score = Column(Float)
    disease_risk_score = Column(Float)
    hours_since_disease_onset = Column(Float, nullable=True)
    current_infection_pct = Column(Float)
    infection_growth_rate_hourly = Column(Float)

    __table_args__ = (
        Index('idx_disease_prog_timestamp', 'timestamp'),
        Index('idx_disease_prog_cycle_disease', 'cycle_id', 'disease_name'),
    )

    def __repr__(self):
        return (f"<DiseaseProgressionData(timestamp={self.timestamp}, "
                f"cycle_id={self.cycle_id}, disease={self.disease_name})>")


class GrowthProgressionHourly(Base):
    """Hourly tomato growth progression time-series table"""
    __tablename__ = 'growth_progression_hourly'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)

    # Cycle / identity
    cycle_id = Column(Integer, nullable=False)
    cycle_label = Column(String(50))
    season_window = Column(String(100))
    real_or_synthetic_flag = Column(String(20))

    # Time features
    year = Column(Integer)
    month = Column(Integer)
    day_of_year = Column(Integer)
    week_of_year = Column(Integer)
    hour = Column(Integer)
    season_label = Column(String(50))
    days_from_cycle_start = Column(Float)

    # Stage / progression
    stage_name = Column(String(50))
    stage_index = Column(Integer)
    hours_in_current_stage = Column(Float)
    days_in_current_stage = Column(Float)
    stage_duration_hours = Column(Integer)
    stage_duration_days = Column(Integer)
    stage_progress_pct = Column(Float)
    total_cycle_progress_pct = Column(Float)
    estimated_days_to_next_stage = Column(Float)
    estimated_hours_to_next_stage = Column(Float)
    is_stage_transition = Column(Boolean)

    # Environment
    indoor_temp = Column(Float)
    indoor_humidity = Column(Float)
    indoor_air_velocity = Column(Float)
    indoor_co2 = Column(Float)
    solarradiation = Column(Float)
    day_night_flag = Column(Float)
    vpd = Column(Float)
    dew_point = Column(Float)
    leaf_wetness_proxy = Column(Float)

    # Engineered features
    temperature_rolling_mean_24h = Column(Float)
    humidity_rolling_mean_24h = Column(Float)
    vpd_proxy = Column(Float)
    light_period_flag = Column(Integer)
    cumulative_gdd_like_index = Column(Float)

    __table_args__ = (
        Index('idx_growth_hourly_timestamp', 'timestamp'),
        Index('idx_growth_hourly_cycle_stage', 'cycle_id', 'stage_index'),
    )

    def __repr__(self):
        return (f"<GrowthProgressionHourly(timestamp={self.timestamp}, "
                f"cycle_id={self.cycle_id}, stage={self.stage_name})>")


class GrowthProgressionStageSummary(Base):
    """Per-stage aggregated summary of growth progression cycles"""
    __tablename__ = 'growth_progression_stage_summary'

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(Integer, nullable=False)
    stage_index = Column(Integer, nullable=False)
    stage_name = Column(String(50), nullable=False)
    hourly_rows = Column(Integer)
    start_timestamp = Column(DateTime)
    end_timestamp = Column(DateTime)
    actual_days = Column(Float)
    max_stage_prog = Column(Float)
    mean_gdd = Column(Float)
    mean_indoor_temp = Column(Float)
    mean_indoor_humidity = Column(Float)
    mean_vpd = Column(Float)
    mean_solarradiation = Column(Float)
    std_indoor_temp = Column(Float)
    std_indoor_humidity = Column(Float)
    std_vpd = Column(Float)
    std_solarradiation = Column(Float)

    __table_args__ = (
        Index('idx_stage_summary_cycle_stage', 'cycle_id', 'stage_index'),
    )

    def __repr__(self):
        return (f"<GrowthProgressionStageSummary(cycle_id={self.cycle_id}, "
                f"stage={self.stage_name})>")


class GrowthProgressionCycleSummary(Base):
    """Per-cycle summary metadata for growth progression"""
    __tablename__ = 'growth_progression_cycle_summary'

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(Integer, nullable=False, unique=True)
    cycle_label = Column(String(50))
    season_window = Column(String(100))
    cycle_start = Column(Date)
    cycle_end = Column(Date)
    total_days = Column(Integer)
    days_seedling = Column(Integer)
    days_early_vegetative = Column(Integer)
    days_flowering_initiation = Column(Integer)
    days_flowering = Column(Integer)
    days_unripe = Column(Integer)
    days_ripe = Column(Integer)
    total_hourly_rows = Column(Integer)

    def __repr__(self):
        return (f"<GrowthProgressionCycleSummary(cycle_id={self.cycle_id}, "
                f"label={self.cycle_label})>")


class GrowthProgressionMetadata(Base):
    """Stores the dataset-level metadata JSON for growth progression"""
    __tablename__ = 'growth_progression_metadata'

    id = Column(Integer, primary_key=True, autoincrement=True)
    project = Column(String(100))
    crop = Column(String(100))
    location = Column(String(100))
    greenhouse_system = Column(String(100))
    notebook_name = Column(String(200))
    created_on = Column(DateTime)
    total_hourly_rows = Column(Integer)
    earliest_timestamp = Column(DateTime)
    latest_timestamp = Column(DateTime)
    metadata_json = Column(JSONB, comment='Full metadata JSON blob')
    loaded_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<GrowthProgressionMetadata(project={self.project}, crop={self.crop})>"


class RealtimeGreenhouseStream(Base):
    """Real-time sensor stream table for the closed-loop DT + MPC controller.

    Each row represents one 5-minute DT simulation step written during a
    ``run_realtime_loop.py`` run.  Column names mirror ``greenhouse_data``
    so that ``MPCInputPreparation.get_latest_greenhouse_row()`` (and its
    subclass override) can use a unified return schema.

    Source legend
    -------------
    ``"bootstrap"``  — initial state seeded from historical ``greenhouse_data``
    ``"dt_sim"``     — state produced by the Digital-Twin ARX physics step
    """
    __tablename__ = "realtime_greenhouse_stream"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # Run identity
    run_id = Column(String(64), nullable=False, index=True,
                    comment="Unique ID of the realtime loop run (rt_YYYYMMDD_HHMMSS)")
    step_index = Column(Integer, nullable=False, comment="Step counter within this run")
    source = Column(String(20), nullable=False, default="dt_sim",
                    comment="bootstrap | dt_sim")

    # Timestamp
    datetime = Column(DateTime, nullable=False, index=True,
                      comment="Logical simulation timestamp (UTC)")
    created_at = Column(DateTime, nullable=False, server_default=func.now(),
                        comment="Wall-clock time the row was inserted")

    # ── Greenhouse climate (mirrors greenhouse_data columns) ──────────
    indoor_temp = Column(Float, comment="Indoor temperature °C")
    indoor_humidity = Column(Float, comment="Indoor relative humidity %")
    indoor_air_velocity = Column(Float, comment="Indoor air velocity m/s")
    indoor_co2 = Column(Float, comment="CO₂ concentration ppm")
    solarradiation = Column(Float, comment="Solar / LED radiation W/m²")
    day_night_flag = Column(Integer, comment="1=day, 0=night")
    vpd = Column(Float, comment="Vapour Pressure Deficit kPa")
    dew_point = Column(Float, comment="Dew-point temperature °C")
    leaf_wetness_proxy = Column(Float, comment="Leaf wetness proxy 0–1")

    # ── Derived state ─────────────────────────────────────────────────
    soil_moisture = Column(Float, comment="Soil moisture % (0–100)")
    disease_risk_score = Column(Float, comment="Composite disease risk 0–1")
    growth_stage = Column(String(50), comment="Canonical growth-stage label")
    growth_stage_index = Column(Integer, comment="Growth stage index 0–5")
    disease_classification = Column(String(100), comment="Top disease class from classifier")

    # ── Applied actuators ─────────────────────────────────────────────
    fan_speed = Column(Float, comment="Fan speed duty 0–1")
    vent_opening = Column(Float, comment="Vent opening fraction 0–1")
    heater_output = Column(Float, comment="Heater duty 0–1")
    led_intensity = Column(Float, comment="LED intensity duty 0–1")
    fogger_duty = Column(Float, comment="Fogger duty cycle 0–1")
    co2_valve_pct = Column(Float, comment="CO₂ valve opening 0–1")
    irrigation_qty = Column(Float, comment="Irrigation quantity L/step")

    # ── MPC decision metadata ─────────────────────────────────────────
    mpc_ran = Column(Boolean, default=False, comment="True if MPC solved this step")
    mpc_converged = Column(Boolean, default=False, comment="True if solver converged")
    mpc_fallback_used = Column(Boolean, default=False,
                                comment="True if baseline fallback was used")
    step_cost = Column(Float, comment="MPC step cost (objective value)")
    solve_time_ms = Column(Float, comment="Solver wall time ms")

    # ── Progression context ───────────────────────────────────────────
    hours_in_current_stage = Column(Float, comment="Elapsed hours in current stage")
    stage_progress_pct = Column(Float, comment="Stage progress 0–100")
    hours_to_stage_transition = Column(Float, comment="LSTM-estimated hours to next stage")

    # ── Resource accounting ───────────────────────────────────────────
    step_energy_kwh = Column(Float, comment="Energy consumed this step kWh")
    cumulative_energy_kwh = Column(Float, comment="Cumulative energy kWh since run start")
    cumulative_water_litres = Column(Float, comment="Cumulative water L since run start")

    # ── Alert ─────────────────────────────────────────────────────────
    alert_level = Column(String(10), default="GREEN",
                         comment="GREEN | YELLOW | RED alert level")

    __table_args__ = (
        Index("idx_rt_stream_run_step", "run_id", "step_index"),
        Index("idx_rt_stream_datetime", "datetime"),
    )

    def __repr__(self) -> str:
        return (
            f"<RealtimeGreenhouseStream(run_id={self.run_id!r}, "
            f"step={self.step_index}, ts={self.datetime}, "
            f"stage={self.growth_stage}, risk={self.disease_risk_score})>"
        )


# TimescaleDB-specific functions (optional, requires timescaledb extension)
def create_hypertables(engine):
    """
    Convert tables to TimescaleDB hypertables for better time-series performance
    Only run this if TimescaleDB extension is installed in PostgreSQL
    """
    with engine.connect() as conn:
        try:
            # Create TimescaleDB extension if not exists
            conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            conn.commit()

            hypertable_configs = [
                ('weather_data', 'datetime', "INTERVAL '1 month'"),
                ('greenhouse_data', 'datetime', "INTERVAL '1 week'"),
                ('disease_progression', 'timestamp', "INTERVAL '1 month'"),
                ('growth_progression_hourly', 'timestamp', "INTERVAL '1 month'"),
            ]

            for table, time_col, interval in hypertable_configs:
                conn.execute(f"""
                    SELECT create_hypertable('{table}', '{time_col}',
                        if_not_exists => TRUE,
                        chunk_time_interval => {interval}
                    );
                """)

            conn.commit()
            print("✓ TimescaleDB hypertables created successfully")
            return True
        except Exception as e:
            print(f"Note: Could not create hypertables (TimescaleDB may not be installed): {e}")
            print("Tables will work with regular PostgreSQL")
            return False

