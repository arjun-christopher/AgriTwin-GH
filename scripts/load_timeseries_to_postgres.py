"""
Load CSV time-series data into PostgreSQL
Handles weather data and greenhouse indoor conditions data

Usage:
    python scripts/load_timeseries_to_postgres.py --all
    python scripts/load_timeseries_to_postgres.py --weather
    python scripts/load_timeseries_to_postgres.py --greenhouse
    python scripts/load_timeseries_to_postgres.py --create-tables-only
    python scripts/load_timeseries_to_postgres.py --timescaledb  # Enable TimescaleDB
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agritwin.models.timeseries import Base, WeatherData, GreenhouseData, create_hypertables
from src.agritwin.utils.database import get_db_manager


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
  
  # Load only greenhouse data
  python scripts/load_timeseries_to_postgres.py --greenhouse
  
  # Create tables only (no data load)
  python scripts/load_timeseries_to_postgres.py --create-tables-only
  
  # Enable TimescaleDB support
  python scripts/load_timeseries_to_postgres.py --all --timescaledb
  
  # Load specific year
  python scripts/load_timeseries_to_postgres.py --weather --year 2024
        """
    )
    
    parser.add_argument('--all', action='store_true', help='Load all data types')
    parser.add_argument('--weather', action='store_true', help='Load weather data')
    parser.add_argument('--greenhouse', action='store_true', help='Load greenhouse data')
    parser.add_argument('--create-tables-only', action='store_true', help='Only create tables, do not load data')
    parser.add_argument('--timescaledb', action='store_true', help='Enable TimescaleDB hypertables')
    parser.add_argument('--year', type=int, choices=[2024, 2025], help='Load data for specific year only')
    parser.add_argument('--drop-existing', action='store_true', help='Drop existing tables before creating new ones')
    
    args = parser.parse_args()
    
    # Default to --all if no specific option is provided
    if not any([args.all, args.weather, args.greenhouse, args.create_tables_only]):
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
