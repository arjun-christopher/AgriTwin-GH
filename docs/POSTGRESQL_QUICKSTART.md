# PostgreSQL Quick Start - AgriTwin-GH

> **Current setup** — The project runs PostgreSQL via Docker using the TimescaleDB image.  The container is named `agritwin-timescaledb` and is already configured.  `psql` does **not** need to be installed locally — use `docker exec` to run queries instead.

## 1. Setup (One-time)

### Install packages
```powershell
uv add -r requirements.txt
```

### Start PostgreSQL (Docker — TimescaleDB)

The project uses TimescaleDB on PostgreSQL 15.  Start the container with:
```powershell
docker run --name agritwin-timescaledb `
  -e POSTGRES_PASSWORD=agritwin-gh `
  -e POSTGRES_DB=agritwin_db `
  -p 5432:5432 -d `
  timescale/timescaledb:latest-pg15
```

If the container already exists, just start it:
```powershell
docker start agritwin-timescaledb
```

Check it is running:
```powershell
docker ps --format "{{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"
```

### Environment — already configured

Credentials are set in `.env` (git-ignored) and `config/settings.local.yaml`:

`.env`:
```dotenv
DB_USER=postgres
DB_PASSWORD=agritwin-gh
DB_NAME=agritwin_db
DB_HOST=localhost
DB_PORT=5432
```

`config/settings.local.yaml`:
```yaml
database:
  type: "postgresql"
  host: "localhost"
  port: 5432
```

### Apply schemas

Pipe SQL files directly into the Docker container — no local `psql` needed:
```powershell
# Monthly snapshots schema
Get-Content database/schema/monthly_snapshots.sql | docker exec -i agritwin-timescaledb psql -U postgres -d agritwin_db
```

## 2. Load Data

### Load everything
```powershell
python scripts/load_timeseries_to_postgres.py --all
```

### With TimescaleDB
```powershell
python scripts/load_timeseries_to_postgres.py --all --timescaledb
```

### Load specific data
```powershell
# Only weather data
python scripts/load_timeseries_to_postgres.py --weather

# Only greenhouse data
python scripts/load_timeseries_to_postgres.py --greenhouse

# Specific year
python scripts/load_timeseries_to_postgres.py --all --year 2025
```

## 3. Query Data

### Run examples
```powershell
python scripts/query_timeseries_examples.py
```

### Use in your code
```python
from src.agritwin.utils.database import get_db_session
from src.agritwin.models.timeseries import WeatherData, GreenhouseData
from datetime import datetime

session = get_db_session()

# Query weather data
weather = session.query(WeatherData).filter(
    WeatherData.datetime >= datetime(2025, 1, 1)
).all()

# Query greenhouse data
greenhouse = session.query(GreenhouseData).filter(
    GreenhouseData.indoor_temp > 30
).all()

session.close()
```

### Use with pandas
```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:yourpassword@localhost:5432/agritwin_db')
df = pd.read_sql_query("SELECT * FROM weather_data LIMIT 100", engine)
```

## 4. Connect to PostgreSQL

Since `psql` is served inside Docker, connect via `docker exec`:
```powershell
# Open interactive psql session
docker exec -it agritwin-timescaledb psql -U postgres -d agritwin_db

# View tables
\dt

# Query
SELECT COUNT(*) FROM weather_data;
SELECT * FROM greenhouse_data LIMIT 10;
SELECT * FROM crop_cycles;
SELECT billing_month, stage_at_month_start, stage_at_month_end, total_cost_inr FROM monthly_snapshots;
```

Or run a one-liner without entering the shell:
```powershell
docker exec -i agritwin-timescaledb psql -U postgres -d agritwin_db -c "\dt"
```

## 5. Useful Scripts

| Command | Description |
|---------|-------------|
| `--all` | Load all data types |
| `--weather` | Load weather data only |
| `--greenhouse` | Load greenhouse data only |
| `--year 2024` | Load specific year |
| `--timescaledb` | Enable TimescaleDB |
| `--create-tables-only` | Create tables without data |
| `--drop-existing` | Drop and recreate tables |

## Tables Schema

### weather_data
- datetime, temp, humidity, windspeed, solarradiation
- conditions, description, sunrise, sunset

### greenhouse_data
- datetime, indoor_temp, indoor_humidity, indoor_air_velocity
- indoor_co2, solarradiation, vpd, dew_point, leaf_wetness_proxy

## Resources
- Full documentation: `docs/POSTGRESQL_SETUP.md`
- Example queries: `scripts/query_timeseries_examples.py`
- Database models: `src/agritwin/models/timeseries.py`
