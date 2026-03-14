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
| **Disease Progression Model** | Multi-stream GRU with cross-disease attention predicting infection %, active status, and net change at 24h and 48h horizons for 5 diseases simultaneously | ✅ Complete |
| **Greenhouse Weather Forecast Model** | Chronos + XGBoost + LSTM ensemble forecasting 24h/48h indoor climate conditions for digital twin and control | ✅ Complete |
| **Digital Twin Simulator** | Physics-based greenhouse model for scenario simulation | ✅ Complete |
| **MPC-Like Control Policy** | Model predictive control for actuator management | ✅ Complete |
| **What-If Analysis** | Comparative scenario evaluation and decision support | ✅ Complete |
| **Non-Verbal Alerts** | Visual operator notifications for critical events | ✅ Complete |
| **Dashboard Visualizations** | Interactive monitoring and performance comparison | ✅ Complete |
| **Resource Tracking** | Energy and water usage optimization and reporting | ✅ Complete |
| **Time-Series Database** | PostgreSQL + TimescaleDB hypertables for sensor data | ✅ Complete |
| **Image Storage** | MinIO (S3-compatible) with PostgreSQL metadata indexing | ✅ Complete |

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
- **Indoor Dataset** — Passive greenhouse physics model deriving indoor conditions from outdoor weather data → [Dataset Guide](docs/INDOOR_GREENHOUSE_DATASET.md)
- **Data Directory Guide** — Structure and management conventions → [Data Guide](docs/DATA.md)

## 🤖 ML Models

- **Disease Classifier** — EfficientNetB0, 6 classes: Early Blight, Late Blight, Leaf Mold, Powdery Mildew, Septoria Leaf Spot, Spider Mites + Healthy → [Disease Classification](docs/TOMATO_DISEASE_CLASSIFICATION.md)
- **Growth Stage Classifier** — EfficientNetB3, 6 stages: Seedling → Early Vegetative → Flowering Initiation → Flowering → Unripe → Ripe, with TTA support → [Growth Stage Classification](docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md)
- **Growth Progression Model** — Multi-task LSTM trained on hourly sensor time-series; simultaneously predicts the current growth stage, next stage, hours until stage transition, and 24h/48h transition probability in a single forward pass → [Growth Progression Model](docs/TOMATO_GROWTH_PROGRESSION_MODEL.md)
- **Disease Progression Model** — Multi-stream GRU with cross-disease co-infection attention; predicts infection %, active status, and net change at 24h and 48h horizons for all 5 diseases simultaneously across healthy, single-disease, and multi-disease crop scenarios → [Disease Progression Model](docs/TOMATO_DISEASE_PROGRESSION_MODEL.md)
- **Greenhouse Weather Forecast Model** — Chronos time-series foundation model + XGBoost + LSTM ensemble for 24h/48h indoor climate forecasting, feeding the digital twin and control policies → [Weather Forecast Model](docs/WEATHER_FORECAST_MODEL.md)

## 🛠️ Technology Stack

- **Python 3.8+** · **UV** (package manager)
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

> **`setup.py`** handles everything automatically — installing uv (if missing), initialising the project, creating the virtual environment, syncing dependencies, and optionally downloading the Kaggle dataset with an interactive token setup. You can also run the steps manually:
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
| [Indoor Greenhouse Dataset](docs/INDOOR_GREENHOUSE_DATASET.md) | Synthetic dataset generation methodology |
| [Disease Classification](docs/TOMATO_DISEASE_CLASSIFICATION.md) | EfficientNetB0 leaf disease model |
| [Growth Stage Classification](docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md) | EfficientNetB3 growth stage model |
| [Growth Progression Model](docs/TOMATO_GROWTH_PROGRESSION_MODEL.md) | Multi-task LSTM for stage transition forecasting from sensor time-series |
| [Disease Progression Model](docs/TOMATO_DISEASE_PROGRESSION_MODEL.md) | Multi-stream GRU with cross-disease attention for 24h/48h disease progression forecasting |
| [Weather Forecast Model](docs/WEATHER_FORECAST_MODEL.md) | Chronos + XGBoost + LSTM ensemble for 24h/48h greenhouse climate forecasting |
| [Deployment Guide](docs/DOCS_DEPLOYMENT.md) | MkDocs documentation site setup |

**[📖 View Full Documentation →](https://arjun-christopher.github.io/AgriTwin-GH/)**

## 📂 Repository Structure

```
AgriTwin-GH/
├── setup.py                # Interactive setup — uv install, venv, deps, Kaggle dataset
├── feature_demos/          # Interactive Jupyter notebook demonstrations (01–06)
├── notebooks/              # ML training notebooks (disease & growth stage classifiers)
├── scripts/                # Data loading, upload, and classification scripts
├── src/agritwin_gh/        # Core library — API, models, services, utils
├── data/                   # Raw, processed, and external datasets
├── database/               # PostgreSQL schema files
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