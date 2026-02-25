"""
Time-series data models for AgriTwin-GH
Supports both regular PostgreSQL and TimescaleDB
"""

from datetime import datetime
from sqlalchemy import Column, Integer, Float, String, DateTime, Index
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
            
            # Convert weather_data to hypertable
            conn.execute("""
                SELECT create_hypertable('weather_data', 'datetime',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 month'
                );
            """)
            
            # Convert greenhouse_data to hypertable
            conn.execute("""
                SELECT create_hypertable('greenhouse_data', 'datetime',
                    if_not_exists => TRUE,
                    chunk_time_interval => INTERVAL '1 week'
                );
            """)
            
            conn.commit()
            print("✓ TimescaleDB hypertables created successfully")
            return True
        except Exception as e:
            print(f"Note: Could not create hypertables (TimescaleDB may not be installed): {e}")
            print("Tables will work with regular PostgreSQL")
            return False
