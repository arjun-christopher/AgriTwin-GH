# AgriTwin-GH 🌱

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://arjun-christopher.github.io/AgriTwin-GH/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **An Advanced Digital Twin System for Precision Greenhouse Agriculture**

AgriTwin-GH is a comprehensive cyber-physical system combining real-time environmental monitoring, ML-based disease and growth stage detection, physics-based digital twin simulation, and model predictive control for intelligent greenhouse management.

## 🌟 Overview

- **Predictive Disease Management** — Risk indexing from environmental sensor data to prevent fungal outbreaks before they occur
- **Growth-Aware Control** — Adaptive MPC policies that respond to detected crop development stages
- **Stage Transition Forecasting** — Multi-task LSTM predicts the next growth stage and hours to transition directly from sensor time-series, enabling proactive greenhouse interventions
- **Digital Twin Simulation** — Physics-based virtual replica enabling what-if scenario analysis
- **Image Intelligence** — EfficientNet classifiers for tomato leaf disease detection and growth stage classification
- **Data Infrastructure** — PostgreSQL + TimescaleDB for time-series, MinIO for image object storage
- **Operator Decision Support** — Visual dashboards and non-verbal alert systems

## ✅ Components

| Component | Description | Status |
|-----------|-------------|--------|
| **Synthetic Data Generator** | Configurable greenhouse sensor data generation | ✅ Complete |
| **Indoor Dataset Generator** | Passive greenhouse physics model from outdoor weather | ✅ Complete |
| **Disease Classifier** | EfficientNetB0 — 6-class tomato leaf disease classification | ✅ Complete |
| **Growth Stage Classifier** | EfficientNetB3 — 6-stage tomato plant growth classification | ✅ Complete |
| **Growth Progression Model** | Multi-task LSTM predicting current stage, next stage, and hours to transition from sensor time-series | ✅ Complete |
| **Disease Progression Model** | Baseline + LSTM/GRU progression pipeline predicting per-disease current presence, 24h infection severity, and 24h trend (absent/emerging/reducing/stable/worsening) from hourly sensor time-series | ✅ Complete |
| **Greenhouse Weather Forecast Model** | Chronos + XGBoost + LSTM ensemble forecasting 24h/48h indoor climate conditions for digital twin and control | ✅ Complete |
| **Digital Twin Simulator** | Physics-based greenhouse model for scenario simulation | ✅ Complete |
| **MPC-Like Control Policy** | Model predictive control for actuator management | ✅ Complete |
| **Real-Time Closed-Loop** | DB→AI→MPC→DB autonomous control cycle with in-memory context buffers | ✅ Complete |
| **What-If Analysis** | Comparative scenario evaluation and decision support | ✅ Complete |
| **Non-Verbal Alerts** | Visual operator notifications for critical events | ✅ Complete |
| **Dashboard Visualizations** | Interactive monitoring and performance comparison | ✅ Complete |
| **Resource Tracking** | Energy and water usage optimization and reporting | ✅ Complete |
| **Time-Series Database** | PostgreSQL + TimescaleDB hypertables for sensor data | ✅ Complete |
| **Image Storage** | MinIO (S3-compatible) with PostgreSQL metadata indexing | ✅ Complete |
| **Monthly Snapshots** | Per-month aggregated sensor, resource, MPC, and disease summary stored to SQLite/PostgreSQL | ✅ Complete |
| **Frontend Dashboard** | React 19 + Tailwind v4 SPA — HomeDashboard, Detailed Insights, Manual Override | ✅ Complete |
| **3D Greenhouse Scene** | Unity-based 3D greenhouse with 10 actuator controllers, 15 crop plants (6-stage visual progression), time-of-day environment, and a central JSON-driven state applier for Python/FastAPI integration | ✅ Complete |

## 📊 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      AgriTwin-GH System                          │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│   Sensors    │    │  Digital Twin    │    │  Actuators   │
│ (Monitoring) │───▶│   (Simulation)   │───▶│  (Control)   │
└──────────────┘    └──────────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────┐
│ Disease Risk │    │  Growth Stage    │    │ MPC Control  │
│  Detection   │    │   Detection      │    │   Policy     │
└──────────────┘    └──────────────────┘    └──────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   Dashboard &    │
                    │  Operator Panel  │
                    └──────────────────┘
```

## 🗄️ Data Infrastructure

- **TimescaleDB** — Hypertable storage for weather and indoor greenhouse time-series (5-min and hourly) → [Database Reference](docs/DATABASE_REFERENCE.md)
- **MinIO** — S3-compatible image object storage with PostgreSQL metadata indexing → [Image Storage Setup](docs/IMAGE_STORAGE_SETUP.md)
- **Monthly Snapshots** — Per-month aggregated summaries (sensor averages, energy/water totals, per-actuator billing, MPC convergence rate, disease peaks, AI run counts) stored to SQLite (dev) or PostgreSQL (prod) → [Monthly Snapshot Reference](docs/MONTHLY_SNAPSHOT_REFERENCE.md)
- **Indoor Dataset** — Passive greenhouse physics model deriving indoor conditions from outdoor weather data → [Dataset Guide](docs/INDOOR_GREENHOUSE_DATASET.md)
- **Data Directory Guide** — Structure and management conventions → [Data Guide](docs/DATA.md)

### Database Schemas

| File | Description |
|------|-------------|
| `database/schema/timeseries_data.sql` | Sensor hypertables — weather and indoor greenhouse time-series |
| `database/schema/image_metadata.sql` | Image object store metadata — disease scans and growth stage images |
| `database/schema/monthly_snapshots.sql` | Monthly aggregation tables — `monthly_snapshots` and `monthly_actuator_energy` |

### Monthly Snapshot Quick Enable

```powershell
# Windows PowerShell
$env:AGRITWIN_MONTHLY_DB = "1"
python main.py
```
```bash
# Linux / macOS
AGRITWIN_MONTHLY_DB=1 python main.py
```

Apply the schema first (SQLite auto-creates on first ingest; PostgreSQL requires explicit apply):
```bash
psql -d agritwin_db -f database/schema/monthly_snapshots.sql
```

Seed mock data and inspect:
```bash
python scripts/seed_monthly_mock.py       # inserts 3 rows across 2 crop cycles
python scripts/show_monthly_snapshots.py  # compact table view
python scripts/show_monthly_snapshots.py --detail  # full per-row breakdown
```

See [Monthly Snapshot Reference](docs/MONTHLY_SNAPSHOT_REFERENCE.md) for the full schema, field reference, and API integration.

## 🤖 ML Models

- **Disease Classifier** — EfficientNetB0, 6 classes: Early Blight, Late Blight, Leaf Mold, Powdery Mildew, Septoria Leaf Spot, Spider Mites + Healthy → [Disease Classification](docs/TOMATO_DISEASE_CLASSIFICATION.md)
- **Growth Stage Classifier** — EfficientNetB3, 6 stages: Seedling → Early Vegetative → Flowering Initiation → Flowering → Unripe → Ripe, with TTA support → [Growth Stage Classification](docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md)
- **Growth Progression Model** — Multi-task LSTM trained on hourly sensor time-series; simultaneously predicts the current growth stage, next stage, hours until stage transition, and 24h/48h transition probability in a single forward pass → [Growth Progression Model](docs/TOMATO_GROWTH_PROGRESSION_MODEL.md)
- **Disease Progression Model** — Baseline + LSTM/GRU progression workflow on hourly greenhouse sensor time-series; predicts per-disease current presence, 24h infection severity, and 24h trend labels (absent, emerging, reducing, stable, worsening) → [Disease Progression Model](docs/TOMATO_DISEASE_PROGRESSION_MODEL.md)
- **Greenhouse Weather Forecast Model** — Chronos time-series foundation model + XGBoost + LSTM ensemble for 24h/48h indoor climate forecasting, feeding the digital twin and control policies → [Weather Forecast Model](docs/WEATHER_FORECAST_MODEL.md)

## 🎮 Model Predictive Control (MPC)

- **MPC Module** — Receding-horizon SLSQP optimisation over a 12-hour prediction window; integrates growth stage detection, disease risk scoring, and weather forecasts to compute optimal actuator actions every 5 minutes
- **Stage-Aware Constraints** — Dynamically adjusts environmental bounds (temperature, humidity, CO₂) based on detected growth stage and disease risk, with soft penalty-based enforcement
- **Cost Function** — Nine-term objective balancing setpoint tracking, disease suppression, humidity exposure, energy costs, water costs, and actuator efficiency
- **Comparison Framework** — Evaluated against a rule-based baseline controller; all three scenarios (standard, disease-pressure, stage-transition) show positive yield improvement and reduced resource costs → [MPC Complete Guide](docs/MPC_COMPLETE_GUIDE.md)

## 🔄 Digital Twin Closed-Loop

A production-grade real-time closed-loop layer (`realtime_core.py`) that connects the PostgreSQL database, all AI inference pipelines, and the MPC solver into a single autonomous control cycle:

- **DB → AI → MPC → DB** — Each 5-minute step reads live sensor rows, runs weather forecast (Chronos/XGBoost/LSTM ensemble), disease progression (LSTM/GRU), and growth stage progression (multi-task LSTM), feeds results into the MPC solver, advances the digital twin physics model, and writes the output back to `realtime_greenhouse_stream`
- **In-Memory Context Buffers** — Bootstrapped from historical hypertables at startup; grown with each step so all AI models always have a full look-back window without repeated DB queries
- **Multi-Rate Cadence** — DT physics every 5 min · MPC solve every 15 min (configurable) · hold steps carry forward last actuator trajectory between solves
- **CLI Runner** — `scripts/run_realtime_loop.py` (`--steps`, `--stage`, `--days-elapsed`, `--mpc-every`, `--no-images`, `--dry-run`) with per-step console output and NDJSON artifact logs → [DT Closed-Loop Guide](docs/DT_LOOP_GUIDE.md)

## 🛠️ Technology Stack

- **Python 3.8+** · **UV** (package manager)
- **FastAPI + Uvicorn** — ASGI backend with 15 REST endpoints
- **TensorFlow / Keras** — EfficientNet model training and inference
- **NumPy · Pandas** — Data processing and analysis
- **Matplotlib · Seaborn** — Visualization and dashboards
- **PostgreSQL 15 + TimescaleDB** — Time-series database
- **MinIO** — S3-compatible object storage
- **Jupyter Notebooks** — Interactive demonstrations
- **MkDocs Material** — Documentation website

## 🚀 Quick Start

```powershell
git clone https://github.com/arjun-christopher/AgriTwin-GH.git
cd AgriTwin-GH
python setup.py           # installs uv, sets up venv, syncs deps, and optionally downloads the dataset
jupyter notebook feature_demos/
```

> **`setup.py`** handles everything automatically across 9 steps: installs uv (if missing), initialises the project, creates the virtual environment, syncs dependencies, creates `.env` from `.env.example`, creates `config/settings.local.yaml`, ensures all required data and log directories exist, runs `npm install` for the React frontend, and optionally downloads the Kaggle dataset with an interactive token setup. A formatted manual-steps guide is printed at the end for anything that requires human action (PostgreSQL, MinIO, API keys, CUDA). You can also run the steps manually:
>
> ```powershell
> uv venv && uv sync
> .venv\Scripts\activate   # Windows — or `source .venv/bin/activate` on macOS/Linux
> ```

## 📚 Documentation

| Guide | Description |
|-------|-------------|
| [Feature Demos Guide](feature_demos/FEATURE_DEMOS_GUIDE.md) | Walkthrough of all 6 interactive notebooks |
| [Database Reference](docs/DATABASE_REFERENCE.md) | Schema, queries, and time-series data guide |
| [PostgreSQL Quick Start](docs/POSTGRESQL_QUICKSTART.md) | Database setup and data loading |
| [Image Storage Setup](docs/IMAGE_STORAGE_SETUP.md) | MinIO + PostgreSQL image pipeline |
| [Monthly Snapshot Reference](docs/MONTHLY_SNAPSHOT_REFERENCE.md) | Per-month aggregated sensor, resource, MPC, and disease summary — schema, API, scripts |
| [Indoor Greenhouse Dataset](docs/INDOOR_GREENHOUSE_DATASET.md) | Synthetic dataset generation methodology |
| [Disease Classification](docs/TOMATO_DISEASE_CLASSIFICATION.md) | EfficientNetB0 leaf disease model |
| [Growth Stage Classification](docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md) | EfficientNetB3 growth stage model |
| [Growth Progression Model](docs/TOMATO_GROWTH_PROGRESSION_MODEL.md) | Multi-task LSTM for stage transition forecasting from sensor time-series |
| [Disease Progression Model](docs/TOMATO_DISEASE_PROGRESSION_MODEL.md) | Baseline + LSTM/GRU disease progression forecasting for per-disease presence, 24h severity, and 24h trend labels |
| [Weather Forecast Model](docs/WEATHER_FORECAST_MODEL.md) | Chronos + XGBoost + LSTM ensemble for 24h/48h greenhouse climate forecasting |
| [MPC Complete Guide](docs/MPC_COMPLETE_GUIDE.md) | Model predictive control module: solver tuning, constraint strategy, cost function design, and end-to-end evaluation |
| [DT Closed-Loop Guide](docs/DT_LOOP_GUIDE.md) | Real-time DB→AI→MPC→DB closed-loop layer: architecture, data flow, cadence, and CLI runner reference |
| [DT Loop Streaming Guide](docs/DT_LOOP_STREAMING_GUIDE.md) | Per-step data flow, log format walkthrough, cadence reference, AI model refresh, and FAQ |
| [FastAPI Backend & API Guide](docs/FASTAPI_API_GUIDE.md) | All 15 REST endpoints, layer-by-layer architecture, runtime state, override mechanism, 3D fields, and testing |
| [Frontend UI Reference](docs/FRONTEND_UI_REFERENCE.md) | React dashboard — pages, components, and live API data-binding contract |
| [Greenhouse 3D Model Reference](docs/GREENHOUSE_3D_MODEL_REFERENCE.md) | Unity scene architecture — C# controllers, GameObject hierarchy, JSON state schema, and FastAPI integration guide |
| [Disease Progression Synthetic Dataset](docs/DISEASE_PROGRESSION_SYNTHETIC_DOCUMENTATION.md) | Technical documentation for the synthetic disease progression dataset — generation methodology, disease dynamics, and feature schema |
| [Deployment Guide](docs/DOCS_DEPLOYMENT.md) | MkDocs documentation site setup |

**[📖 View Full Documentation →](https://arjun-christopher.github.io/AgriTwin-GH/)**

## 🌐 FastAPI Backend

A FastAPI + Uvicorn server exposes 16 REST endpoints backed by the live DT loop and `RuntimeStore`:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dt/state` | Full DT snapshot — sensors, crop, actuators, 3D scene context |
| POST | `/api/dt/override` | Enter override mode with custom env/crop values |
| POST | `/api/dt/override/sim` | Enter sim override mode (stage + start time) |
| DELETE | `/api/dt/override` | Return to live DT data |
| POST | `/api/dt/preset/{id}` | Apply a named preset (e.g. `high-growth`, `disease-alert`) |
| GET | `/api/intelligence/disease` | Per-pathogen risk scores with confidence and trend |
| GET | `/api/intelligence/growth` | Growth stage transition forecasts |
| GET | `/api/weather/current` | Outdoor conditions + 24h forecast |
| GET | `/api/actuators/state` | Current actuator levels |
| POST | `/api/actuators/set` | Override individual actuator levels |
| GET | `/api/resources/monthly` | Energy (kWh) + water (L) usage and INR cost |
| GET | `/api/media/latest` | Latest disease scan and growth stage image |
| GET | `/api/media/stage-images` | Rolling gallery of growth stage captures |
| GET | `/api/media/disease-scans` | Rolling gallery of disease scan images |
| GET | `/api/system/health` | 6-subsystem health check |

Interactive docs at `http://localhost:8000/docs`. See [FastAPI Backend & API Guide](docs/FASTAPI_API_GUIDE.md) for the full architecture, layer reference, and testing guide.

```powershell
.venv\Scripts\Activate.ps1
python main.py   # → FastAPI on http://localhost:8000
```
## 🌿 3D Greenhouse Model

A Unity-based 3D greenhouse scene that mirrors the live digital twin state in real time via a JSON-driven central controller:

- **GreenhouseStateApplier** — Central C# orchestrator that reads a JSON state file (or FastAPI response), detects changes, and dispatches to all sub-controllers automatically
- **10 Actuator Controllers** — Fluorescent lights, heater, humidifier, window fan, vent, and water tank, each with status indicators, particle effects, and audio
- **15 Crop Plants** — Each with a `CropStageController` (6-stage visual model swap: Seedling → Ripe) and a `CropHealthIndicator` (green / yellow / red RGB health lights with blinking)
- **Environment System** — `TimeOfDayController` drives skybox, directional light, fog, and night lights; `CropHealthIndicator` maps disease risk scores to visual health states
- **FastAPI Integration** — The scene is designed to consume the `/api/dt/state` JSON payload and apply the full greenhouse state without code changes

See [Greenhouse 3D Model Reference](docs/GREENHOUSE_3D_MODEL_REFERENCE.md) for the complete C# script reference, GameObject hierarchy, JSON schema, and integration guide.
## �️ Frontend Dashboard

A React 19 + Tailwind v4 single-page application providing a real-time operator interface for the greenhouse digital twin.

- **HomeDashboard** — live sensor metrics strip, crop stage progression track, actuator status grid, camera frames, resource usage and cost summary
- **Detailed Insights** — full indoor sensor readings with optimal ranges, per-pathogen disease risk bars, outdoor weather + 24h forecast, rolling stage and leaf-scan image galleries, growth stage transition spotlight
- **Manual Override** — live/override mode toggle, editable growth stage + day-in-stage + start time, per-actuator on/off toggles, apply and reset-all actions

All pages are wired to live FastAPI endpoints. The service layer (`src/agritwin_gh/frontend/src/services/api.js`) makes real `fetch()` calls to `http://localhost:8000/api/*`. Run `python main.py` to start the backend, then `npm run dev` for the dev server. See [FastAPI Backend & API Guide](docs/FASTAPI_API_GUIDE.md) for the full endpoint reference, override mode, and 3D integration fields.

```powershell
cd src/agritwin_gh/frontend
npm install
npm run dev      # → http://localhost:5173
```

## 📂 Repository Structure

```
AgriTwin-GH/
├── setup.py                # Interactive setup — uv install, venv, deps, Kaggle dataset
├── main.py                 # FastAPI entry point (Uvicorn + DT loop startup)
├── feature_demos/          # Interactive Jupyter notebook demonstrations (01–06)
├── notebooks/              # ML training notebooks (disease & growth stage classifiers)
├── scripts/                # Utility scripts
│   ├── seed_monthly_mock.py        # Seed 3 monthly snapshot rows for demo
│   ├── show_monthly_snapshots.py   # Display monthly snapshot table (--detail, --cycle, --limit)
│   ├── load_timeseries_to_postgres.py  # Load sensor data into PostgreSQL/TimescaleDB
│   └── classify_input_leaf.py      # Run leaf disease classifier on an input image
├── database/
│   └── schema/
│       ├── timeseries_data.sql         # Sensor hypertables (weather + indoor)
│       ├── image_metadata.sql          # MinIO image metadata index
│       └── monthly_snapshots.sql       # Monthly aggregation tables
├── src/agritwin_gh/        # Core library — API, models, services, utils
│   └── frontend/           # React 19 + Tailwind v4 dashboard SPA
├── tests/                  # Unit and smoke tests
├── data/                   # Raw, processed, and external datasets
└── docs/                   # Documentation source files
```

## 🔬 Research Areas

- Cyber-physical system design for controlled environment agriculture
- Digital twin technology and physics-based simulation
- Transfer learning for plant pathology and phenology classification
- Model predictive control in greenhouse environments
- Human-machine interface design for agricultural systems

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**[📖 View Live Documentation](https://arjun-christopher.github.io/AgriTwin-GH/)**