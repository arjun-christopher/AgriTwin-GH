# PostgreSQL Time-Series Data Setup Guide

This guide explains how to set up PostgreSQL and load your time-series data (weather and greenhouse conditions) into the database.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Database Setup Options](#database-setup-options)
4. [Loading Data](#loading-data)
5. [TimescaleDB (Optional)](#timescaledb-optional)
6. [Querying Data](#querying-data)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### 1. Install Required Python Packages
```powershell
uv add -r requirements.txt
```

### 2. Install PostgreSQL
Choose one of these options:

#### Option A: Local PostgreSQL Installation
Download and install from [PostgreSQL official website](https://www.postgresql.org/download/):
- Download the installer for Windows
- Run the installer and remember your password
- Default port is 5432

#### Option B: Docker (Recommended for Development)
```powershell
# Run PostgreSQL in Docker
docker run --name agritwin-postgres `
  -e POSTGRES_PASSWORD=yourpassword `
  -e POSTGRES_DB=agritwin_db `
  -p 5432:5432 `
  -d postgres:15

# Or with TimescaleDB support (recommended for time-series)
docker run --name agritwin-timescaledb `
  -e POSTGRES_PASSWORD=yourpassword `
  -e POSTGRES_DB=agritwin_db `
  -p 5432:5432 `
  -d timescale/timescaledb:latest-pg15
```

#### Option C: Cloud PostgreSQL
- AWS RDS PostgreSQL
- Azure Database for PostgreSQL
- Google Cloud SQL for PostgreSQL
- DigitalOcean Managed Databases

---

## Quick Start

### Step 1: Configure Environment Variables

Create a `.env` file in your project root:
```bash
# Copy from example
cp .env.example .env
```

Edit `.env` with your database credentials:
```dotenv
DB_USER=postgres
DB_PASSWORD=yourpassword
DB_NAME=agritwin_db
DB_HOST=localhost
DB_PORT=5432
```

### Step 2: Configure Database Settings

Create `config/settings.local.yaml`:
```yaml
# Copy from example
cp config/settings.local.example.yaml config/settings.local.yaml
```

Edit `config/settings.local.yaml`:
```yaml
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
  # Credentials are loaded from .env
```

### Step 3: Load All Data

```powershell
# Load all data (weather + greenhouse)
python scripts/load_timeseries_to_postgres.py --all
```

That's it! Your data is now in PostgreSQL.

---

## Database Setup Options

### Create Tables Only (No Data)
```powershell
python scripts/load_timeseries_to_postgres.py --create-tables-only
```

### Load Specific Data Types
```powershell
# Load only weather data
python scripts/load_timeseries_to_postgres.py --weather

# Load only greenhouse data
python scripts/load_timeseries_to_postgres.py --greenhouse
```

### Load Specific Year
```powershell
# Load only 2024 data
python scripts/load_timeseries_to_postgres.py --all --year 2024

# Load only 2025 data
python scripts/load_timeseries_to_postgres.py --all --year 2025
```

### Drop and Recreate Tables
```powershell
# Warning: This will delete existing data!
python scripts/load_timeseries_to_postgres.py --all --drop-existing
```

---

## TimescaleDB (Optional)

TimescaleDB is a PostgreSQL extension optimized for time-series data. It provides:
- Faster queries on time-series data
- Automatic data partitioning (chunks)
- Built-in time-series functions
- Better compression

### Installing TimescaleDB

#### Option 1: Docker (Easiest)
```powershell
docker run --name agritwin-timescaledb `
  -e POSTGRES_PASSWORD=yourpassword `
  -e POSTGRES_DB=agritwin_db `
  -p 5432:5432 `
  -d timescale/timescaledb:latest-pg15
```

#### Option 2: Existing PostgreSQL
Follow [TimescaleDB installation guide](https://docs.timescale.com/install/latest/)

### Loading Data with TimescaleDB
```powershell
python scripts/load_timeseries_to_postgres.py --all --timescaledb
```

This will:
1. Create standard PostgreSQL tables
2. Convert them to TimescaleDB hypertables
3. Load the data

### TimescaleDB Benefits for Your Data

**Weather Data** (daily records):
- Chunked by 1 month intervals
- Faster queries for date ranges
- Better compression (~50% space savings)

**Greenhouse Data** (hourly records):
- Chunked by 1 week intervals
- Optimized for high-frequency queries
- Automated retention policies possible

---

## Querying Data

### Using Python (SQLAlchemy)

```python
from src.agritwin.utils.database import get_db_session
from src.agritwin.models.timeseries import WeatherData, GreenhouseData
from datetime import datetime

# Get database session
session = get_db_session()

# Query weather data for a date range
weather_data = session.query(WeatherData).filter(
    WeatherData.datetime >= datetime(2025, 1, 1),
    WeatherData.datetime < datetime(2025, 2, 1)
).all()

# Query greenhouse data with conditions
hot_conditions = session.query(GreenhouseData).filter(
    GreenhouseData.indoor_temp > 30.0,
    GreenhouseData.indoor_humidity < 60.0
).all()

# Time-series aggregation
from sqlalchemy import func

avg_temp_by_day = session.query(
    func.date_trunc('day', WeatherData.datetime).label('day'),
    func.avg(WeatherData.temp).label('avg_temp'),
    func.max(WeatherData.temp).label('max_temp'),
    func.min(WeatherData.temp).label('min_temp')
).group_by('day').all()

session.close()
```

### Using pandas

```python
import pandas as pd
from sqlalchemy import create_engine

# Create engine
engine = create_engine('postgresql://postgres:yourpassword@localhost:5432/agritwin_db')

# Query to DataFrame
weather_df = pd.read_sql_query(
    "SELECT * FROM weather_data WHERE datetime >= '2025-01-01'",
    engine
)

greenhouse_df = pd.read_sql_query(
    "SELECT * FROM greenhouse_data WHERE indoor_temp > 30",
    engine
)
```

### Direct SQL Queries

Connect using psql or any PostgreSQL client:
```sql
-- Connect
psql -U postgres -d agritwin_db

-- View table structure
\d weather_data
\d greenhouse_data

-- Count records
SELECT COUNT(*) FROM weather_data;
SELECT COUNT(*) FROM greenhouse_data;

-- Average temperature by month
SELECT 
    DATE_TRUNC('month', datetime) as month,
    AVG(temp) as avg_temp,
    AVG(humidity) as avg_humidity
FROM weather_data
GROUP BY month
ORDER BY month;

-- Indoor vs outdoor temperature correlation
SELECT 
    w.datetime,
    g.indoor_temp,
    w.temp as outdoor_temp,
    (g.indoor_temp - w.temp) as temp_difference
FROM greenhouse_data g
JOIN weather_data w ON DATE_TRUNC('hour', g.datetime) = DATE_TRUNC('hour', w.datetime)
WHERE w.datetime >= '2025-01-01';

-- TimescaleDB specific queries (if enabled)
-- Time-weighted average
SELECT time_bucket('1 day', datetime) as day,
       time_weight('Average', datetime, temp) as weighted_avg_temp
FROM weather_data
GROUP BY day;
```

---

## Database Schema

### weather_data Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| datetime | DateTime | Timestamp (indexed) |
| datetime_epoch | Integer | Unix timestamp |
| temp | Float | Temperature (°C) |
| humidity | Float | Humidity (%) |
| windspeed | Float | Wind speed (km/h) |
| solarradiation | Float | Solar radiation (W/m²) |
| sunrise | String | Sunrise time |
| sunrise_epoch | Float | Sunrise unix timestamp |
| sunset | String | Sunset time |
| sunset_epoch | Float | Sunset unix timestamp |
| conditions | String | Weather conditions |
| description | String | Detailed description |

### greenhouse_data Table
| Column | Type | Description |
|--------|------|-------------|
| id | Integer | Primary key |
| datetime | DateTime | Timestamp (indexed) |
| indoor_temp | Float | Indoor temperature (°C) |
| indoor_humidity | Float | Indoor humidity (%) |
| indoor_air_velocity | Float | Air velocity (m/s) |
| indoor_co2 | Float | CO2 concentration (ppm) |
| solarradiation | Float | Solar radiation (W/m²) |
| day_night_flag | Integer | Day (1) or Night (0) |
| vpd | Float | Vapor Pressure Deficit (kPa) |
| dew_point | Float | Dew point temperature (°C) |
| leaf_wetness_proxy | Float | Leaf wetness indicator |

---

## Troubleshooting

### Connection Issues

**Error: could not connect to server**
- Check PostgreSQL is running: `docker ps` or check Windows services
- Verify host and port in `.env`
- Check firewall settings

**Error: password authentication failed**
- Verify credentials in `.env`
- Check PostgreSQL user permissions

### Data Loading Issues

**Error: relation "weather_data" does not exist**
```powershell
# Create tables first
python scripts/load_timeseries_to_postgres.py --create-tables-only
```

**Error: column "X" does not exist**
- Your CSV structure might differ from expected
- Check CSV headers match the model definitions

**Slow loading**
- Use `--year` to load data incrementally
- Check database connection (local is faster)
- Ensure adequate disk space

### TimescaleDB Issues

**Error: extension "timescaledb" does not exist**
- TimescaleDB is not installed
- Use regular PostgreSQL or install TimescaleDB
- Load without `--timescaledb` flag

---

## Performance Tips

1. **Indexes**: Already created on `datetime` columns for fast queries
2. **Batch Loading**: Script uses batch commits (1000 records) for speed
3. **Connection Pooling**: Configured in database.py for concurrent access
4. **TimescaleDB**: Recommended for production time-series workloads
5. **Compression**: TimescaleDB auto-compresses old data

---

## Next Steps

After loading data:
1. Create visualization dashboards using Grafana or custom tools
2. Implement real-time data ingestion pipelines
3. Set up data retention policies
4. Create materialized views for common aggregations
5. Implement alerting based on thresholds

---

## Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [pandas SQL Documentation](https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html)
