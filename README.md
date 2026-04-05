# AgriTwin-GH 🌱

<div align="center">

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://arjun-christopher.github.io/AgriTwin-GH/)
[![Python](https://img.shields.io/badge/python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Unity](https://img.shields.io/badge/Unity-WebGL-000000?logo=unity&logoColor=white)](https://unity.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-15+-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![Git LFS](https://img.shields.io/badge/Git_LFS-enabled-F05032?logo=git&logoColor=white)](https://git-lfs.com/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**An Advanced Digital Twin System for Precision Greenhouse Agriculture**

*Combining real-time sensor fusion, physics-based simulation, multi-model ML inference, and model predictive control into a unified cyber-physical platform for intelligent tomato cultivation.*

[📖 View Full Documentation →](https://arjun-christopher.github.io/AgriTwin-GH/)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Components](#-components)
- [System Architecture](#-system-architecture)
- [Data Infrastructure](#️-data-infrastructure)
- [ML Models](#-ml-models)
- [Model Predictive Control](#-model-predictive-control-mpc)
- [Digital Twin Closed-Loop](#-digital-twin-closed-loop)
- [Technology Stack](#️-technology-stack)
- [Quick Start](#-quick-start)
- [Documentation](#-documentation)
- [FastAPI Backend](#-fastapi-backend)
- [3D Greenhouse Model](#-3d-greenhouse-model)
- [Frontend Dashboard](#️-frontend-dashboard)
- [Repository Structure](#-repository-structure)
- [Research Areas](#-research-areas)
- [License](#-license)

---

AgriTwin-GH is a comprehensive cyber-physical system combining real-time environmental monitoring, ML-based disease and growth stage detection, physics-based digital twin simulation, and model predictive control for intelligent greenhouse management.

## 🌟 Overview

| Capability | Description |
|-----------|-------------|
| 🦠 **Predictive Disease Management** | Risk indexing from environmental sensor data to prevent fungal outbreaks before they occur |
| 🌱 **Growth-Aware Control** | Adaptive MPC policies that respond to detected crop development stages |
| 🔮 **Stage Transition Forecasting** | Multi-task LSTM predicts next growth stage and hours to transition directly from sensor time-series, enabling proactive interventions |
| 🤖 **Digital Twin Simulation** | Physics-based virtual replica enabling what-if scenario analysis |
| 📸 **Image Intelligence** | EfficientNet classifiers for tomato leaf disease detection and growth stage classification |
| 🗄️ **Data Infrastructure** | PostgreSQL + TimescaleDB for time-series, MinIO for image object storage |
| 🧊 **3D Visualization** | Unity WebGL greenhouse scene driven live from the FastAPI state endpoint |
| 👁️ **Operator Decision Support** | React dashboard, visual alerts, and non-verbal notification systems |

---

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
| **3D Greenhouse Scene** | Unity WebGL greenhouse with 10 actuator controllers, 15 crop plants (6-stage visual progression), time-of-day environment, and a central JSON-driven state applier for Python/FastAPI integration | ✅ Complete |

---

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
                    ┌────────────────────────────────┐
                    │   FastAPI  ·  React Dashboard   │
                    │    Unity 3D Greenhouse Scene    │
                    └────────────────────────────────┘
```

---

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

---

## 🤖 ML Models

| Model | Architecture | Task | Guide |
|-------|-------------|------|-------|
| **Disease Classifier** | EfficientNetB0 | 6-class leaf disease detection — Early Blight, Late Blight, Leaf Mold, Powdery Mildew, Septoria Leaf Spot, Spider Mites + Healthy | [→](docs/TOMATO_DISEASE_CLASSIFICATION.md) |
| **Growth Stage Classifier** | EfficientNetB3 | 6-stage growth classification — Seedling → Early Vegetative → Flowering Initiation → Flowering → Unripe → Ripe, with TTA support | [→](docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md) |
| **Growth Progression Model** | Multi-task LSTM | Simultaneously predicts current stage, next stage, hours until stage transition, and 24h/48h transition probability from hourly sensor time-series | [→](docs/TOMATO_GROWTH_PROGRESSION_MODEL.md) |
| **Disease Progression Model** | Baseline + LSTM/GRU | Per-disease current presence, 24h infection severity, and 24h trend labels (absent, emerging, reducing, stable, worsening) from hourly sensor time-series | [→](docs/TOMATO_DISEASE_PROGRESSION_MODEL.md) |
| **Greenhouse Weather Forecast** | Chronos + XGBoost + LSTM Ensemble | 24h/48h indoor climate forecasting feeding the digital twin and MPC control policies | [→](docs/WEATHER_FORECAST_MODEL.md) |

---

## 🎮 Model Predictive Control (MPC)

- **MPC Module** — Receding-horizon SLSQP optimisation over a 12-hour prediction window; integrates growth stage detection, disease risk scoring, and weather forecasts to compute optimal actuator actions every 5 minutes
- **Stage-Aware Constraints** — Dynamically adjusts environmental bounds (temperature, humidity, CO₂) based on detected growth stage and disease risk, with soft penalty-based enforcement
- **Cost Function** — Nine-term objective balancing setpoint tracking, disease suppression, humidity exposure, energy costs, water costs, and actuator efficiency
- **Comparison Framework** — Evaluated against a rule-based baseline controller; all three scenarios (standard, disease-pressure, stage-transition) show positive yield improvement and reduced resource costs

→ [MPC Complete Guide](docs/MPC_COMPLETE_GUIDE.md)

---

## 🔄 Digital Twin Closed-Loop

A production-grade real-time closed-loop layer (`realtime_core.py`) that connects the PostgreSQL database, all AI inference pipelines, and the MPC solver into a single autonomous control cycle:

- **DB → AI → MPC → DB** — Each 5-minute step reads live sensor rows, runs weather forecast (Chronos/XGBoost/LSTM ensemble), disease progression (LSTM/GRU), and growth stage progression (multi-task LSTM), feeds results into the MPC solver, advances the digital twin physics model, and writes the output back to `realtime_greenhouse_stream`
- **In-Memory Context Buffers** — Bootstrapped from historical hypertables at startup; grown with each step so all AI models always have a full look-back window without repeated DB queries
- **Multi-Rate Cadence** — DT physics every 5 min · MPC solve every 15 min (configurable) · hold steps carry forward last actuator trajectory between solves
- **CLI Runner** — `scripts/run_realtime_loop.py` with flags `--steps`, `--stage`, `--days-elapsed`, `--mpc-every`, `--no-images`, `--dry-run` — per-step console output and NDJSON artifact logs

→ [DT Closed-Loop Guide](docs/DT_LOOP_GUIDE.md) · [Streaming Guide](docs/DT_LOOP_STREAMING_GUIDE.md)

---

## 🛠️ Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Language & Runtime** | Python 3.13 · UV (package manager & venv) |
| **Backend** | FastAPI · Uvicorn (ASGI) · SQLAlchemy |
| **ML / DL** | TensorFlow · Keras · PyTorch · Chronos · LightGBM · XGBoost · CatBoost · Scikit-learn |
| **Data** | NumPy · Pandas · PyArrow · Statsmodels |
| **Visualization** | Matplotlib · Seaborn |
| **Database** | PostgreSQL 15 · TimescaleDB · SQLite |
| **Object Storage** | MinIO (S3-compatible) |
| **Frontend** | React 19 · Tailwind CSS v4 · Vite |
| **3D Scene** | Unity (WebGL build) · C# |
| **Infra / DevOps** | Git LFS · MkDocs Material · Docker (MinIO) |
| **Notebooks** | Jupyter · Feature demo notebooks (01–06) |

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Purpose | Link |
|-------------|---------|------|
| Python 3.13 | Backend runtime | [python.org](https://www.python.org/downloads/) |
| Node.js LTS | React frontend | [nodejs.org](https://nodejs.org/) |
| Git LFS | Unity WebGL binary tracking | [git-lfs.com](https://git-lfs.com/) |
| PostgreSQL 15+ *(optional)* | Production time-series storage | [postgresql.org](https://www.postgresql.org/) |
| MinIO *(optional)* | Image object storage | [min.io](https://min.io/) |

### Installation

```powershell
git lfs install
git clone https://github.com/arjun-christopher/AgriTwin-GH.git
cd AgriTwin-GH
python setup.py
```

> **`setup.py`** automates 9 steps: installs uv, initialises the project, creates the virtual environment, syncs all Python dependencies, creates `.env` from `.env.example`, creates `config/settings.local.yaml`, ensures all required data and log directories exist, runs `npm install` for the React frontend, and optionally downloads the Kaggle dataset. A formatted manual-steps guide is printed at the end for everything that requires human action (PostgreSQL, MinIO, API keys, CUDA).

Manual alternative:

```powershell
uv venv && uv sync
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux
```

### Running the Application

```powershell
# Full stack — FastAPI backend + Vite dev server (auto-launched)
python main.py

# Backend only
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Interactive notebook demos
jupyter notebook feature_demos/
```

---

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

---

## 🌐 FastAPI Backend

A FastAPI + Uvicorn server exposes 16 REST endpoints backed by the live DT loop and `RuntimeStore`. Interactive API docs at `http://localhost:8000/docs`.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/dt/state` | Full DT snapshot — sensors, crop, actuators, 3D scene context |
| `POST` | `/api/dt/override` | Enter override mode with custom env/crop values |
| `POST` | `/api/dt/override/sim` | Enter sim override mode (stage + start time) |
| `DELETE` | `/api/dt/override` | Return to live DT data |
| `POST` | `/api/dt/preset/{id}` | Apply a named preset (e.g. `high-growth`, `disease-alert`) |
| `GET` | `/api/intelligence/disease` | Per-pathogen risk scores with confidence and trend |
| `GET` | `/api/intelligence/growth` | Growth stage transition forecasts |
| `GET` | `/api/weather/current` | Outdoor conditions + 24h forecast |
| `GET` | `/api/actuators/state` | Current actuator levels |
| `POST` | `/api/actuators/set` | Override individual actuator levels |
| `GET` | `/api/resources/monthly` | Energy (kWh) + water (L) usage and INR cost |
| `GET` | `/api/media/latest` | Latest disease scan and growth stage image |
| `GET` | `/api/media/stage-images` | Rolling gallery of growth stage captures |
| `GET` | `/api/media/disease-scans` | Rolling gallery of disease scan images |
| `GET` | `/api/system/health` | 6-subsystem health check |

→ [FastAPI Backend & API Guide](docs/FASTAPI_API_GUIDE.md)

```powershell
.venv\Scripts\Activate.ps1
python main.py   # → FastAPI on http://localhost:8000
```

---

## 🌿 3D Greenhouse Model

A Unity-based 3D greenhouse scene that mirrors the live digital twin state in real time via a JSON-driven central controller. The compiled WebGL build lives in `src/agritwin_gh/build/` and is tracked in this repository via **Git LFS** (the Brotli-compressed binary bundles).

| Component | Description |
|-----------|-------------|
| **GreenhouseStateApplier** | Central C# orchestrator — reads a JSON state file (or `/api/dt/state` FastAPI response), detects changes, and dispatches to all sub-controllers automatically |
| **10 Actuator Controllers** | Fluorescent lights, heater, humidifier, window fan, vent, and water tank — each with status indicators, particle effects, and audio |
| **15 Crop Plants** | Each with a `CropStageController` (6-stage visual model swap: Seedling → Ripe) and a `CropHealthIndicator` (green / yellow / red RGB health lights with blinking) |
| **Environment System** | `TimeOfDayController` drives skybox, directional light, fog, and night lights; disease risk scores are mapped to visual health states |

```
src/agritwin_gh/build/
├── Build/
│   ├── build.data.br          # Scene + asset data      (~18 MB, Git LFS)
│   ├── build.wasm.br          # Unity runtime (WASM)    (~ 6 MB, Git LFS)
│   ├── build.framework.js.br  # JS framework loader     (Git LFS)
│   └── build.loader.js        # Bootstrap loader
├── StreamingAssets/
│   └── greenhouse_state.json  # Default DT state payload consumed by the scene
├── TemplateData/              # WebGL template assets (CSS, icons, logos)
└── index.html                 # Entry point — open in browser or embed in FastAPI
```

→ [Greenhouse 3D Model Reference](docs/GREENHOUSE_3D_MODEL_REFERENCE.md)

---

## �️ Frontend Dashboard

A React 19 + Tailwind v4 single-page application providing a real-time operator interface for the greenhouse digital twin.

| Page | Description |
|------|-------------|
| **HomeDashboard** | Live sensor metrics strip, crop stage progression track, actuator status grid, camera frames, resource usage and cost summary |
| **Detailed Insights** | Full indoor sensor readings with optimal ranges, per-pathogen disease risk bars, outdoor weather + 24h forecast, rolling stage and leaf-scan image galleries, growth stage transition spotlight |
| **Manual Override** | Live/override mode toggle, editable growth stage + day-in-stage + start time, per-actuator on/off toggles, apply and reset-all actions |

All pages make real `fetch()` calls to `http://localhost:8000/api/*` via `src/agritwin_gh/frontend/src/services/api.js`.

```powershell
cd src/agritwin_gh/frontend
npm install
npm run dev      # → http://localhost:5173
```

→ [Frontend UI Reference](docs/FRONTEND_UI_REFERENCE.md)

---

## 📂 Repository Structure

```
AgriTwin-GH/
├── setup.py                    # 9-step interactive setup — uv, venv, deps, .env, npm, Kaggle
├── main.py                     # FastAPI entry point (Uvicorn + DT loop + Vite dev server)
├── config/                     # App configuration (settings.yaml, settings.local.yaml, MPC config)
├── feature_demos/              # Interactive Jupyter notebook demonstrations (01–06)
├── notebooks/                  # ML training notebooks (disease & growth stage classifiers)
├── scripts/                    # Utility and data pipeline scripts
│   ├── load_timeseries_to_postgres.py  # Load CSV sensor data into PostgreSQL / TimescaleDB
│   ├── upload_images_to_minio.py       # Bulk image upload to MinIO buckets
│   ├── seed_monthly_mock.py            # Seed 3 monthly snapshot rows for demo
│   ├── show_monthly_snapshots.py       # Display monthly snapshot table (--detail, --cycle)
│   ├── run_realtime_loop.py            # CLI runner for the DT closed-loop
│   └── classify_input_leaf.py          # Run leaf disease classifier on an input image
├── database/
│   └── schema/
│       ├── timeseries_data.sql         # Sensor hypertables (weather + indoor conditions)
│       ├── image_metadata.sql          # MinIO image metadata index
│       └── monthly_snapshots.sql       # Monthly aggregation tables
├── src/agritwin_gh/            # Core Python library — API, models, services, utils
│   ├── api/                    # FastAPI routers and RuntimeStore
│   ├── core/                   # Digital twin physics, MPC solver, realtime loop
│   ├── models/                 # ML model wrappers and inference pipelines
│   ├── services/               # Database, MinIO, and sensor service layers
│   ├── frontend/               # React 19 + Tailwind v4 dashboard SPA
│   └── build/                  # Unity WebGL greenhouse scene (Git LFS for .br binaries)
│       ├── Build/              # Brotli-compressed WASM + data bundles (LFS-tracked)
│       ├── StreamingAssets/    # greenhouse_state.json — default DT state for the 3D scene
│       └── TemplateData/       # WebGL template CSS, icons, and Unity branding
├── tests/                      # Unit and smoke tests
├── data/                       # Raw, processed, and external datasets
│   ├── raw/                    # Unprocessed source files
│   ├── processed/              # Cleaned and feature-engineered outputs
│   └── external/               # Kaggle dataset — disease images, growth stages, weather CSV
└── docs/                       # MkDocs documentation source files
```

---

## 🔬 Research Areas

- Cyber-physical system design for controlled environment agriculture
- Digital twin technology and physics-based simulation
- Transfer learning for plant pathology and phenology classification
- Model predictive control in greenhouse environments
- Human-machine interface design for agricultural systems

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

**[📖 View Live Documentation](https://arjun-christopher.github.io/AgriTwin-GH/)**

</div>