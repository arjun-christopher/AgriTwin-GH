"""
Example queries for time-series data in PostgreSQL
Demonstrates various ways to query and analyze the data

Usage:
    python scripts/query_timeseries_examples.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.agritwin.utils.database import get_db_manager
from src.agritwin.models.timeseries import WeatherData, GreenhouseData
from sqlalchemy import func, and_, or_


def example_basic_queries(session):
    """Basic query examples"""
    print("\n" + "="*60)
    print("BASIC QUERIES")
    print("="*60)
    
    # Count total records
    weather_count = session.query(WeatherData).count()
    greenhouse_count = session.query(GreenhouseData).count()
    print(f"\nTotal weather records: {weather_count}")
    print(f"Total greenhouse records: {greenhouse_count}")
    
    # Get latest records
    latest_weather = session.query(WeatherData).order_by(
        WeatherData.datetime.desc()
    ).first()
    print(f"\nLatest weather record: {latest_weather.datetime}")
    print(f"  Temperature: {latest_weather.temp}°C")
    print(f"  Humidity: {latest_weather.humidity}%")
    
    # Get date range
    date_range = session.query(
        func.min(WeatherData.datetime).label('start'),
        func.max(WeatherData.datetime).label('end')
    ).first()
    print(f"\nData range: {date_range.start} to {date_range.end}")


def example_filtering(session):
    """Filtering examples"""
    print("\n" + "="*60)
    print("FILTERING QUERIES")
    print("="*60)
    
    # High temperature days
    hot_days = session.query(WeatherData).filter(
        WeatherData.temp > 30.0
    ).count()
    print(f"\nDays with temperature > 30°C: {hot_days}")
    
    # Rainy days
    rainy_days = session.query(WeatherData).filter(
        WeatherData.conditions.like('%Rain%')
    ).count()
    print(f"Rainy days: {rainy_days}")
    
    # Date range query
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 2, 1)
    january_data = session.query(WeatherData).filter(
        and_(
            WeatherData.datetime >= start_date,
            WeatherData.datetime < end_date
        )
    ).count()
    print(f"Weather records in January 2025: {january_data}")
    
    # Greenhouse conditions - high CO2
    high_co2 = session.query(GreenhouseData).filter(
        GreenhouseData.indoor_co2 > 800
    ).count()
    print(f"Hours with CO2 > 800 ppm: {high_co2}")


def example_aggregations(session):
    """Aggregation examples"""
    print("\n" + "="*60)
    print("AGGREGATION QUERIES")
    print("="*60)
    
    # Average temperature by month
    print("\nAverage temperature by month:")
    monthly_avg = session.query(
        func.date_trunc('month', WeatherData.datetime).label('month'),
        func.avg(WeatherData.temp).label('avg_temp'),
        func.avg(WeatherData.humidity).label('avg_humidity'),
        func.count().label('records')
    ).group_by('month').order_by('month').all()
    
    for record in monthly_avg:
        print(f"  {record.month.strftime('%Y-%m')}: "
              f"Temp={record.avg_temp:.1f}°C, "
              f"Humidity={record.avg_humidity:.1f}%, "
              f"Records={record.records}")
    
    # Daily statistics
    print("\nDaily temperature statistics (first 5 days of 2025):")
    daily_stats = session.query(
        func.date_trunc('day', WeatherData.datetime).label('day'),
        func.min(WeatherData.temp).label('min_temp'),
        func.max(WeatherData.temp).label('max_temp'),
        func.avg(WeatherData.temp).label('avg_temp')
    ).filter(
        WeatherData.datetime >= datetime(2025, 1, 1)
    ).group_by('day').order_by('day').limit(5).all()
    
    for record in daily_stats:
        print(f"  {record.day.strftime('%Y-%m-%d')}: "
              f"Min={record.min_temp:.1f}°C, "
              f"Max={record.max_temp:.1f}°C, "
              f"Avg={record.avg_temp:.1f}°C")


def example_joining_tables(session):
    """Join query examples"""
    print("\n" + "="*60)
    print("JOIN QUERIES")
    print("="*60)
    
    # Join weather and greenhouse data by date
    print("\nIndoor vs Outdoor conditions (first 5 records):")
    joined_data = session.query(
        GreenhouseData.datetime,
        GreenhouseData.indoor_temp,
        GreenhouseData.indoor_humidity,
        WeatherData.temp.label('outdoor_temp'),
        WeatherData.humidity.label('outdoor_humidity')
    ).join(
        WeatherData,
        func.date_trunc('day', GreenhouseData.datetime) == 
        func.date_trunc('day', WeatherData.datetime)
    ).order_by(GreenhouseData.datetime).limit(5).all()
    
    for record in joined_data:
        temp_diff = record.indoor_temp - record.outdoor_temp
        print(f"  {record.datetime.strftime('%Y-%m-%d %H:%M')}: "
              f"Indoor={record.indoor_temp:.1f}°C, "
              f"Outdoor={record.outdoor_temp:.1f}°C, "
              f"Diff={temp_diff:+.1f}°C")


def example_pandas_integration(engine):
    """Using pandas for analysis"""
    print("\n" + "="*60)
    print("PANDAS INTEGRATION")
    print("="*60)
    
    # Load data to DataFrame
    query = """
    SELECT 
        datetime,
        temp,
        humidity,
        solarradiation,
        conditions
    FROM weather_data
    WHERE datetime >= '2025-01-01'
    ORDER BY datetime
    LIMIT 10
    """
    
    df = pd.read_sql_query(query, engine)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print("\nWeather data as DataFrame:")
    print(df.to_string(index=False))
    
    # Basic statistics
    print("\nStatistics:")
    print(df[['temp', 'humidity', 'solarradiation']].describe())


def example_advanced_analysis(session):
    """Advanced analysis examples"""
    print("\n" + "="*60)
    print("ADVANCED ANALYSIS")
    print("="*60)
    
    # Correlation between outdoor and indoor conditions
    print("\nAverage indoor-outdoor temperature difference by hour:")
    hourly_diff = session.query(
        func.extract('hour', GreenhouseData.datetime).label('hour'),
        func.avg(GreenhouseData.indoor_temp).label('avg_indoor'),
        func.avg(GreenhouseData.solarradiation).label('avg_solar')
    ).group_by('hour').order_by('hour').all()
    
    for record in hourly_diff[:6]:  # Show first 6 hours
        print(f"  Hour {int(record.hour):02d}:00 - "
              f"Avg Indoor Temp: {record.avg_indoor:.1f}°C, "
              f"Avg Solar: {record.avg_solar:.1f} W/m²")
    
    # High-risk conditions (example)
    print("\nHigh-risk conditions (High temp + Low humidity):")
    risk_conditions = session.query(
        func.count().label('count'),
        func.avg(GreenhouseData.indoor_temp).label('avg_temp'),
        func.avg(GreenhouseData.indoor_humidity).label('avg_humidity')
    ).filter(
        and_(
            GreenhouseData.indoor_temp > 30,
            GreenhouseData.indoor_humidity < 60
        )
    ).first()
    
    print(f"  Occurrences: {risk_conditions.count}")
    print(f"  Avg Temperature: {risk_conditions.avg_temp:.1f}°C")
    print(f"  Avg Humidity: {risk_conditions.avg_humidity:.1f}%")


def main():
    """Run all example queries"""
    print("\n" + "="*60)
    print("TIME-SERIES DATA QUERY EXAMPLES")
    print("="*60)
    
    # Get database manager
    db_manager = get_db_manager()
    engine = db_manager.engine
    session = db_manager.get_session()
    
    try:
        # Run examples
        example_basic_queries(session)
        example_filtering(session)
        example_aggregations(session)
        example_joining_tables(session)
        example_pandas_integration(engine)
        example_advanced_analysis(session)
        
        print("\n" + "="*60)
        print("All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        raise
    finally:
        session.close()
        db_manager.close()


if __name__ == '__main__':
    main()
