"""
Load CSV time-series data into PostgreSQL
Handles weather, greenhouse indoor conditions, disease progression,
and growth progression data.

Usage:
    python scripts/load_timeseries_to_postgres.py --all
    python scripts/load_timeseries_to_postgres.py --weather
    python scripts/load_timeseries_to_postgres.py --greenhouse
    python scripts/load_timeseries_to_postgres.py --disease
    python scripts/load_timeseries_to_postgres.py --growth
    python scripts/load_timeseries_to_postgres.py --create-tables-only
    python scripts/load_timeseries_to_postgres.py --timescaledb  # Enable TimescaleDB
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agritwin_gh.models.timeseries import (
    Base,
    WeatherData,
    GreenhouseData,
    DiseaseProgressionData,
    GrowthProgressionHourly,
    GrowthProgressionStageSummary,
    GrowthProgressionCycleSummary,
    GrowthProgressionMetadata,
    create_hypertables,
)
from src.agritwin_gh.utils.database import get_db_manager


def create_tables(engine, use_timescaledb: bool = False):
    """
    Create database tables
    
    Args:
        engine: SQLAlchemy engine
        use_timescaledb: Whether to create TimescaleDB hypertables
    """
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Tables created successfully")
    
    if use_timescaledb:
        print("\nConverting to TimescaleDB hypertables...")
        create_hypertables(engine)


def load_weather_data(session, data_dir: Path, year: int = None):
    """
    Load weather data from CSV files
    
    Args:
        session: SQLAlchemy session
        data_dir: Path to data directory
        year: Optional year filter (2024, 2025)
    """
    weather_dir = data_dir / 'external' / 'Weather Data'
    
    if year:
        csv_files = [weather_dir / f'dindigul_weather_{year}.csv']
    else:
        csv_files = list(weather_dir.glob('dindigul_weather_*.csv'))
    
    total_records = 0
    
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Warning: File not found: {csv_file}")
            continue
        
        print(f"\nLoading weather data from: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # Convert datetime string to datetime object
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Batch insert for better performance
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            record = WeatherData(
                datetime=row['datetime'],
                datetime_epoch=int(row['datetimeEpoch']) if pd.notna(row['datetimeEpoch']) else None,
                temp=float(row['temp']) if pd.notna(row['temp']) else None,
                humidity=float(row['humidity']) if pd.notna(row['humidity']) else None,
                windspeed=float(row['windspeed']) if pd.notna(row['windspeed']) else None,
                solarradiation=float(row['solarradiation']) if pd.notna(row['solarradiation']) else None,
                sunrise=str(row['sunrise']) if pd.notna(row['sunrise']) else None,
                sunrise_epoch=float(row['sunriseEpoch']) if pd.notna(row['sunriseEpoch']) else None,
                sunset=str(row['sunset']) if pd.notna(row['sunset']) else None,
                sunset_epoch=float(row['sunsetEpoch']) if pd.notna(row['sunsetEpoch']) else None,
                conditions=str(row['conditions']) if pd.notna(row['conditions']) else None,
                description=str(row['description']) if pd.notna(row['description']) else None,
            )
            records.append(record)
            
            # Batch commit every 1000 records
            if len(records) >= 1000:
                session.bulk_save_objects(records)
                session.commit()
                records = []
        
        # Commit remaining records
        if records:
            session.bulk_save_objects(records)
            session.commit()
        
        total_records += len(df)
        print(f"✓ Loaded {len(df)} weather records from {csv_file.name}")
    
    print(f"\n✓ Total weather records loaded: {total_records}")
    return total_records


def load_greenhouse_data(session, data_dir: Path, year: int = None):
    """
    Load greenhouse indoor conditions data from CSV files
    
    Args:
        session: SQLAlchemy session
        data_dir: Path to data directory
        year: Optional year filter (2024, 2025)
    """
    greenhouse_dir = data_dir / 'processed' / 'Greenhouse Indoor Conditions'
    
    if year:
        csv_files = [greenhouse_dir / f'dindigul_greenhouse_indoor_{year}.csv']
    else:
        csv_files = list(greenhouse_dir.glob('dindigul_greenhouse_indoor_*.csv'))
    
    total_records = 0
    
    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"Warning: File not found: {csv_file}")
            continue
        
        print(f"\nLoading greenhouse data from: {csv_file.name}")
        df = pd.read_csv(csv_file)
        
        # Convert datetime string to datetime object
        df['datetime'] = pd.to_datetime(df['datetime'])
        
        # Batch insert for better performance
        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing records"):
            record = GreenhouseData(
                datetime=row['datetime'],
                indoor_temp=float(row['indoor_temp']) if pd.notna(row['indoor_temp']) else None,
                indoor_humidity=float(row['indoor_humidity']) if pd.notna(row['indoor_humidity']) else None,
                indoor_air_velocity=float(row['indoor_air_velocity']) if pd.notna(row['indoor_air_velocity']) else None,
                indoor_co2=float(row['indoor_CO2']) if pd.notna(row['indoor_CO2']) else None,
                solarradiation=float(row['solarradiation']) if pd.notna(row['solarradiation']) else None,
                day_night_flag=int(row['day_night_flag']) if pd.notna(row['day_night_flag']) else None,
                vpd=float(row['vpd']) if pd.notna(row['vpd']) else None,
                dew_point=float(row['dew_point']) if pd.notna(row['dew_point']) else None,
                leaf_wetness_proxy=float(row['leaf_wetness_proxy']) if pd.notna(row['leaf_wetness_proxy']) else None,
            )
            records.append(record)
            
            # Batch commit every 1000 records
            if len(records) >= 1000:
                session.bulk_save_objects(records)
                session.commit()
                records = []
        
        # Commit remaining records
        if records:
            session.bulk_save_objects(records)
            session.commit()
        
        total_records += len(df)
        print(f"✓ Loaded {len(df)} greenhouse records from {csv_file.name}")
    
    print(f"\n✓ Total greenhouse records loaded: {total_records}")
    return total_records


def load_disease_progression_data(session, data_dir: Path):
    """
    Load disease progression hourly data from CSV.

    Args:
        session: SQLAlchemy session
        data_dir: Path to project data directory
    """
    csv_file = data_dir / 'processed' / 'Disease Progression' / \
               'tomato_disease_progression_synthetic_hourly.csv'

    if not csv_file.exists():
        print(f"Warning: File not found: {csv_file}")
        return 0

    print(f"\nLoading disease progression data from: {csv_file.name}")
    df = pd.read_csv(csv_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    bool_map = {'True': True, 'False': False, True: True, False: False}

    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing disease records"):
        record = DiseaseProgressionData(
            timestamp=row['timestamp'],
            cycle_id=int(row['cycle_id']),
            cycle_label=str(row['cycle_label']) if pd.notna(row['cycle_label']) else None,
            season_label=str(row['season_label']) if pd.notna(row['season_label']) else None,
            stage_name=str(row['stage_name']) if pd.notna(row['stage_name']) else None,
            stage_index=int(row['stage_index']) if pd.notna(row['stage_index']) else None,
            days_from_cycle_start=float(row['days_from_cycle_start']) if pd.notna(row['days_from_cycle_start']) else None,
            day_of_year=int(row['day_of_year']) if pd.notna(row['day_of_year']) else None,
            week_of_year=int(row['week_of_year']) if pd.notna(row['week_of_year']) else None,
            hour=int(row['hour']) if pd.notna(row['hour']) else None,
            hours_in_current_stage=float(row['hours_in_current_stage']) if pd.notna(row['hours_in_current_stage']) else None,
            stage_progress_pct=float(row['stage_progress_pct']) if pd.notna(row['stage_progress_pct']) else None,
            total_cycle_progress_pct=float(row['total_cycle_progress_pct']) if pd.notna(row['total_cycle_progress_pct']) else None,
            is_stage_transition=bool_map.get(row['is_stage_transition'], None),
            indoor_temp=float(row['indoor_temp']) if pd.notna(row['indoor_temp']) else None,
            indoor_humidity=float(row['indoor_humidity']) if pd.notna(row['indoor_humidity']) else None,
            indoor_air_velocity=float(row['indoor_air_velocity']) if pd.notna(row['indoor_air_velocity']) else None,
            indoor_co2=float(row['indoor_CO2']) if pd.notna(row['indoor_CO2']) else None,
            solarradiation=float(row['solarradiation']) if pd.notna(row['solarradiation']) else None,
            day_night_flag=float(row['day_night_flag']) if pd.notna(row['day_night_flag']) else None,
            vpd=float(row['vpd']) if pd.notna(row['vpd']) else None,
            dew_point=float(row['dew_point']) if pd.notna(row['dew_point']) else None,
            leaf_wetness_proxy=float(row['leaf_wetness_proxy']) if pd.notna(row['leaf_wetness_proxy']) else None,
            temperature_rolling_mean_24h=float(row['temperature_rolling_mean_24h']) if pd.notna(row['temperature_rolling_mean_24h']) else None,
            humidity_rolling_mean_24h=float(row['humidity_rolling_mean_24h']) if pd.notna(row['humidity_rolling_mean_24h']) else None,
            vpd_proxy=float(row['vpd_proxy']) if pd.notna(row['vpd_proxy']) else None,
            cumulative_gdd_like_index=float(row['cumulative_gdd_like_index']) if pd.notna(row['cumulative_gdd_like_index']) else None,
            disease_name=str(row['disease_name']),
            disease_present_flag=int(row['disease_present_flag']) if pd.notna(row['disease_present_flag']) else None,
            disease_cycle_id=int(row['disease_cycle_id']) if pd.notna(row['disease_cycle_id']) else None,
            disease_cycle_stage=str(row['disease_cycle_stage']) if pd.notna(row['disease_cycle_stage']) else None,
            outbreak_trigger_flag=int(row['outbreak_trigger_flag']) if pd.notna(row['outbreak_trigger_flag']) else None,
            control_action_flag=int(row['control_action_flag']) if pd.notna(row['control_action_flag']) else None,
            control_action_type=str(row['control_action_type']) if pd.notna(row['control_action_type']) else None,
            stage_susceptibility_score=float(row['stage_susceptibility_score']) if pd.notna(row['stage_susceptibility_score']) else None,
            disease_risk_score=float(row['disease_risk_score']) if pd.notna(row['disease_risk_score']) else None,
            hours_since_disease_onset=float(row['hours_since_disease_onset']) if pd.notna(row['hours_since_disease_onset']) else None,
            current_infection_pct=float(row['current_infection_pct']) if pd.notna(row['current_infection_pct']) else None,
            infection_growth_rate_hourly=float(row['infection_growth_rate_hourly']) if pd.notna(row['infection_growth_rate_hourly']) else None,
        )
        records.append(record)

        if len(records) >= 1000:
            session.bulk_save_objects(records)
            session.commit()
            records = []

    if records:
        session.bulk_save_objects(records)
        session.commit()

    count = len(df)
    print(f"✓ Loaded {count} disease progression records from {csv_file.name}")
    return count


def load_growth_progression_data(session, data_dir: Path):
    """
    Load all growth progression artefacts:
      - hourly time-series
      - stage summary
      - cycle summary
      - metadata JSON

    Args:
        session: SQLAlchemy session
        data_dir: Path to project data directory
    """
    growth_dir = data_dir / 'processed' / 'Growth Progression'

    total = 0
    bool_map = {'True': True, 'False': False, True: True, False: False}

    # ── 1. Hourly time-series ──────────────────────────────────────────────
    hourly_file = growth_dir / 'tomato_growth_progression_synthetic_hourly.csv'
    if not hourly_file.exists():
        print(f"Warning: File not found: {hourly_file}")
    else:
        print(f"\nLoading growth progression hourly data from: {hourly_file.name}")
        df = pd.read_csv(hourly_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        records = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing hourly growth records"):
            record = GrowthProgressionHourly(
                timestamp=row['timestamp'],
                cycle_id=int(row['cycle_id']),
                cycle_label=str(row['cycle_label']) if pd.notna(row['cycle_label']) else None,
                season_window=str(row['season_window']) if pd.notna(row['season_window']) else None,
                real_or_synthetic_flag=str(row['real_or_synthetic_flag']) if pd.notna(row['real_or_synthetic_flag']) else None,
                year=int(row['year']) if pd.notna(row['year']) else None,
                month=int(row['month']) if pd.notna(row['month']) else None,
                day_of_year=int(row['day_of_year']) if pd.notna(row['day_of_year']) else None,
                week_of_year=int(row['week_of_year']) if pd.notna(row['week_of_year']) else None,
                hour=int(row['hour']) if pd.notna(row['hour']) else None,
                season_label=str(row['season_label']) if pd.notna(row['season_label']) else None,
                days_from_cycle_start=float(row['days_from_cycle_start']) if pd.notna(row['days_from_cycle_start']) else None,
                stage_name=str(row['stage_name']) if pd.notna(row['stage_name']) else None,
                stage_index=int(row['stage_index']) if pd.notna(row['stage_index']) else None,
                hours_in_current_stage=float(row['hours_in_current_stage']) if pd.notna(row['hours_in_current_stage']) else None,
                days_in_current_stage=float(row['days_in_current_stage']) if pd.notna(row['days_in_current_stage']) else None,
                stage_duration_hours=int(row['stage_duration_hours']) if pd.notna(row['stage_duration_hours']) else None,
                stage_duration_days=int(row['stage_duration_days']) if pd.notna(row['stage_duration_days']) else None,
                stage_progress_pct=float(row['stage_progress_pct']) if pd.notna(row['stage_progress_pct']) else None,
                total_cycle_progress_pct=float(row['total_cycle_progress_pct']) if pd.notna(row['total_cycle_progress_pct']) else None,
                estimated_days_to_next_stage=float(row['estimated_days_to_next_stage']) if pd.notna(row['estimated_days_to_next_stage']) else None,
                estimated_hours_to_next_stage=float(row['estimated_hours_to_next_stage']) if pd.notna(row['estimated_hours_to_next_stage']) else None,
                is_stage_transition=bool_map.get(row['is_stage_transition'], None),
                indoor_temp=float(row['indoor_temp']) if pd.notna(row['indoor_temp']) else None,
                indoor_humidity=float(row['indoor_humidity']) if pd.notna(row['indoor_humidity']) else None,
                indoor_air_velocity=float(row['indoor_air_velocity']) if pd.notna(row['indoor_air_velocity']) else None,
                indoor_co2=float(row['indoor_CO2']) if pd.notna(row['indoor_CO2']) else None,
                solarradiation=float(row['solarradiation']) if pd.notna(row['solarradiation']) else None,
                day_night_flag=float(row['day_night_flag']) if pd.notna(row['day_night_flag']) else None,
                vpd=float(row['vpd']) if pd.notna(row['vpd']) else None,
                dew_point=float(row['dew_point']) if pd.notna(row['dew_point']) else None,
                leaf_wetness_proxy=float(row['leaf_wetness_proxy']) if pd.notna(row['leaf_wetness_proxy']) else None,
                temperature_rolling_mean_24h=float(row['temperature_rolling_mean_24h']) if pd.notna(row['temperature_rolling_mean_24h']) else None,
                humidity_rolling_mean_24h=float(row['humidity_rolling_mean_24h']) if pd.notna(row['humidity_rolling_mean_24h']) else None,
                vpd_proxy=float(row['vpd_proxy']) if pd.notna(row['vpd_proxy']) else None,
                light_period_flag=int(row['light_period_flag']) if pd.notna(row['light_period_flag']) else None,
                cumulative_gdd_like_index=float(row['cumulative_gdd_like_index']) if pd.notna(row['cumulative_gdd_like_index']) else None,
            )
            records.append(record)

            if len(records) >= 1000:
                session.bulk_save_objects(records)
                session.commit()
                records = []

        if records:
            session.bulk_save_objects(records)
            session.commit()

        total += len(df)
        print(f"✓ Loaded {len(df)} hourly growth progression records")

    # ── 2. Stage summary ──────────────────────────────────────────────────
    stage_file = growth_dir / 'tomato_growth_progression_stage_summary.csv'
    if not stage_file.exists():
        print(f"Warning: File not found: {stage_file}")
    else:
        print(f"\nLoading growth stage summary from: {stage_file.name}")
        df = pd.read_csv(stage_file)
        df['start_timestamp'] = pd.to_datetime(df['start_timestamp'])
        df['end_timestamp'] = pd.to_datetime(df['end_timestamp'])

        records = [
            GrowthProgressionStageSummary(
                cycle_id=int(row['cycle_id']),
                stage_index=int(row['stage_index']),
                stage_name=str(row['stage_name']),
                hourly_rows=int(row['hourly_rows']) if pd.notna(row['hourly_rows']) else None,
                start_timestamp=row['start_timestamp'],
                end_timestamp=row['end_timestamp'],
                actual_days=float(row['actual_days']) if pd.notna(row['actual_days']) else None,
                max_stage_prog=float(row['max_stage_prog']) if pd.notna(row['max_stage_prog']) else None,
                mean_gdd=float(row['mean_gdd']) if pd.notna(row['mean_gdd']) else None,
                mean_indoor_temp=float(row['mean_indoor_temp']) if pd.notna(row['mean_indoor_temp']) else None,
                mean_indoor_humidity=float(row['mean_indoor_humidity']) if pd.notna(row['mean_indoor_humidity']) else None,
                mean_vpd=float(row['mean_vpd']) if pd.notna(row['mean_vpd']) else None,
                mean_solarradiation=float(row['mean_solarradiation']) if pd.notna(row['mean_solarradiation']) else None,
                std_indoor_temp=float(row['std_indoor_temp']) if pd.notna(row['std_indoor_temp']) else None,
                std_indoor_humidity=float(row['std_indoor_humidity']) if pd.notna(row['std_indoor_humidity']) else None,
                std_vpd=float(row['std_vpd']) if pd.notna(row['std_vpd']) else None,
                std_solarradiation=float(row['std_solarradiation']) if pd.notna(row['std_solarradiation']) else None,
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(records)
        session.commit()
        print(f"✓ Loaded {len(records)} stage summary rows")

    # ── 3. Cycle summary ──────────────────────────────────────────────────
    cycle_file = growth_dir / 'tomato_growth_progression_cycle_summary.csv'
    if not cycle_file.exists():
        print(f"Warning: File not found: {cycle_file}")
    else:
        print(f"\nLoading growth cycle summary from: {cycle_file.name}")
        df = pd.read_csv(cycle_file)
        df['cycle_start'] = pd.to_datetime(df['cycle_start']).dt.date
        df['cycle_end'] = pd.to_datetime(df['cycle_end']).dt.date

        records = [
            GrowthProgressionCycleSummary(
                cycle_id=int(row['cycle_id']),
                cycle_label=str(row['cycle_label']) if pd.notna(row['cycle_label']) else None,
                season_window=str(row['season_window']) if pd.notna(row['season_window']) else None,
                cycle_start=row['cycle_start'],
                cycle_end=row['cycle_end'],
                total_days=int(row['total_days']) if pd.notna(row['total_days']) else None,
                days_seedling=int(row['days_seedling']) if pd.notna(row['days_seedling']) else None,
                days_early_vegetative=int(row['days_early_vegetative']) if pd.notna(row['days_early_vegetative']) else None,
                days_flowering_initiation=int(row['days_flowering_initiation']) if pd.notna(row['days_flowering_initiation']) else None,
                days_flowering=int(row['days_flowering']) if pd.notna(row['days_flowering']) else None,
                days_unripe=int(row['days_unripe']) if pd.notna(row['days_unripe']) else None,
                days_ripe=int(row['days_ripe']) if pd.notna(row['days_ripe']) else None,
                total_hourly_rows=int(row['total_hourly_rows']) if pd.notna(row['total_hourly_rows']) else None,
            )
            for _, row in df.iterrows()
        ]
        session.bulk_save_objects(records)
        session.commit()
        print(f"✓ Loaded {len(records)} cycle summary rows")

    # ── 4. Metadata JSON ──────────────────────────────────────────────────
    meta_file = growth_dir / 'tomato_growth_progression_metadata.json'
    if not meta_file.exists():
        print(f"Warning: File not found: {meta_file}")
    else:
        print(f"\nLoading growth progression metadata from: {meta_file.name}")
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        date_coverage = meta.get('date_coverage', {})
        record = GrowthProgressionMetadata(
            project=meta.get('project'),
            crop=meta.get('crop'),
            location=meta.get('location'),
            greenhouse_system=meta.get('greenhouse_system'),
            notebook_name=meta.get('notebook_name'),
            created_on=datetime.fromisoformat(meta['created_on']) if meta.get('created_on') else None,
            total_hourly_rows=meta.get('total_hourly_rows'),
            earliest_timestamp=datetime.fromisoformat(date_coverage['earliest_timestamp']) if date_coverage.get('earliest_timestamp') else None,
            latest_timestamp=datetime.fromisoformat(date_coverage['latest_timestamp']) if date_coverage.get('latest_timestamp') else None,
            metadata_json=meta,
        )
        session.add(record)
        session.commit()
        print("✓ Loaded growth progression metadata")

    return total


def main():
    parser = argparse.ArgumentParser(
        description='Load time-series data into PostgreSQL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load all data
  python scripts/load_timeseries_to_postgres.py --all

  # Load only weather data
  python scripts/load_timeseries_to_postgres.py --weather

  # Load only greenhouse indoor conditions (2024 + 2025 merged)
  python scripts/load_timeseries_to_postgres.py --greenhouse

  # Load disease progression data
  python scripts/load_timeseries_to_postgres.py --disease

  # Load growth progression data (hourly + summaries + metadata)
  python scripts/load_timeseries_to_postgres.py --growth

  # Create tables only (no data load)
  python scripts/load_timeseries_to_postgres.py --create-tables-only

  # Enable TimescaleDB support
  python scripts/load_timeseries_to_postgres.py --all --timescaledb

  # Load specific year (weather / greenhouse only)
  python scripts/load_timeseries_to_postgres.py --weather --year 2024
        """
    )

    parser.add_argument('--all', action='store_true', help='Load all data types')
    parser.add_argument('--weather', action='store_true', help='Load weather data')
    parser.add_argument('--greenhouse', action='store_true', help='Load greenhouse indoor conditions (2024+2025)')
    parser.add_argument('--disease', action='store_true', help='Load disease progression data')
    parser.add_argument('--growth', action='store_true', help='Load growth progression data (hourly + summaries + metadata)')
    parser.add_argument('--create-tables-only', action='store_true', help='Only create tables, do not load data')
    parser.add_argument('--timescaledb', action='store_true', help='Enable TimescaleDB hypertables')
    parser.add_argument('--year', type=int, choices=[2024, 2025], help='Load data for specific year only (weather/greenhouse)')
    parser.add_argument('--drop-existing', action='store_true', help='Drop existing tables before creating new ones')

    args = parser.parse_args()

    # Default to --all if no specific option provided
    if not any([args.all, args.weather, args.greenhouse, args.disease, args.growth, args.create_tables_only]):
        args.all = True

    # Get database manager
    print("Connecting to database...")
    db_manager = get_db_manager()
    engine = db_manager.engine

    # Display connection info
    db_type = db_manager.config.get('database', {}).get('type', 'sqlite')
    print(f"Database type: {db_type}")
    print(f"Connection: {engine.url}\n")

    # Drop existing tables if requested
    if args.drop_existing:
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)
        print("✓ Tables dropped\n")

    # Create tables
    create_tables(engine, use_timescaledb=args.timescaledb)

    if args.create_tables_only:
        print("\n✓ Tables created successfully (data load skipped)")
        return

    # Get data directory
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'

    # Load data
    session = db_manager.get_session()

    try:
        start_time = datetime.now()

        if args.all or args.weather:
            load_weather_data(session, data_dir, year=args.year)

        if args.all or args.greenhouse:
            load_greenhouse_data(session, data_dir, year=args.year)

        if args.all or args.disease:
            load_disease_progression_data(session, data_dir)

        if args.all or args.growth:
            load_growth_progression_data(session, data_dir)

        elapsed = datetime.now() - start_time
        print(f"\n{'='*60}")
        print(f"✓ Data loading completed successfully!")
        print(f"Time elapsed: {elapsed.total_seconds():.2f} seconds")
        print(f"{'='*60}")

    except Exception as e:
        print(f"\n✗ Error during data loading: {e}")
        session.rollback()
        raise
    finally:
        session.close()
        db_manager.close()


if __name__ == '__main__':
    main()
