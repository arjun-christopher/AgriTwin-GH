# AgriTwin-GH 🌱

[![Documentation](https://img.shields.io/badge/docs-online-brightgreen.svg)](https://arjun-christopher.github.io/AgriTwin-GH/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **An Advanced Digital Twin System for Precision Greenhouse Agriculture**

AgriTwin-GH is a comprehensive cyber-physical system that combines real-time environmental monitoring, disease risk prediction, growth stage detection, and model predictive control for intelligent greenhouse management. It goes beyond traditional climate control systems by integrating machine learning, simulation-based decision support, and automated resource optimization.

## 🌟 Overview

Traditional greenhouse systems focus on basic climate control. AgriTwin-GH extends this paradigm with:

- **Predictive Disease Management** — Machine learning-based risk indexing to prevent diseases before they occur
- **Growth-Aware Control** — Adaptive control policies that adjust to detected crop development stages
- **Digital Twin Simulation** — Virtual replica enabling what-if scenario analysis and predictive optimization
- **Resource Efficiency** — Intelligent tracking and optimization of energy and water consumption
- **Operator Decision Support** — Visual dashboards and non-verbal alert systems for human-machine collaboration

## 🚀 Key Features

### ✅ Implemented Components

| Component | Description | Status |
|-----------|-------------|--------|
| **Synthetic Data Generator** | Realistic greenhouse sensor data with configurable parameters | ✅ Complete |
| **Disease Risk Index** | ML-based fungal disease risk prediction from environmental data | ✅ Complete |
| **Growth Stage Detection** | Automated crop phenology classification (germination → harvest) | ✅ Complete |
| **Digital Twin Simulator** | Physics-based greenhouse model for scenario simulation | ✅ Complete |
| **MPC-Like Control Policy** | Model predictive control for actuator management | ✅ Complete |
| **What-If Analysis** | Comparative scenario evaluation and decision support | ✅ Complete |
| **Non-Verbal Alerts** | Visual operator notifications for critical events | ✅ Complete |
| **Dashboard Visualizations** | Interactive monitoring and performance comparison tools | ✅ Complete |
| **Resource Tracking** | Energy and water usage optimization and reporting | ✅ Complete |

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

## 🛠️ Technology Stack

- **Python 3.8+** — Core implementation language
- **UV** — Fast Python package manager
- **NumPy & Pandas** — Data processing and analysis
- **Matplotlib & Seaborn** — Visualization and dashboards
- **Jupyter Notebooks** — Interactive demonstrations
- **MkDocs Material** — Documentation website

## 🚀 Quick Start

### Prerequisites

This project uses **UV** as the Python package manager for faster dependency management and virtual environment handling.

#### Install UV (if not already installed)

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arjun-christopher/AgriTwin-GH.git
   cd AgriTwin-GH
   ```

2. **Create virtual environment and install dependencies:**
   ```bash
   uv venv
   uv pip install -e .
   ```

3. **Activate the virtual environment:**
   
   **Windows:**
   ```powershell
   .venv\Scripts\activate
   ```
   
   **macOS/Linux:**
   ```bash
   source .venv/bin/activate
   ```

4. **Run feature demonstrations:**
   ```bash
   jupyter notebook feature_demos/
   ```

### Development Installation

For development with additional tools:
```bash
uv pip install -e ".[dev]"
```

## 📚 Documentation

**[📖 View Full Documentation →](https://arjun-christopher.github.io/AgriTwin-GH/)**

### Quick Links

- **[🚀 Setup Guide](SETUP_GUIDE.md)** — Comprehensive UV package manager setup and usage instructions
- **[🤝 Contributing Guide](CONTRIBUTING.md)** — Guidelines for contributing to the project
- **[Feature Demonstrations Guide](feature_demos/FEATURE_DEMOS_GUIDE.md)** — Comprehensive walkthrough of all system capabilities with interactive notebooks
- **[Deployment Instructions](docs/DOCS_DEPLOYMENT.md)** — Documentation website setup and configuration guide

## 🎯 Use Cases

- **Research** — Digital twin modeling in controlled environment agriculture
- **Commercial Agriculture** — Smart greenhouse operations and management
- **Education** — Teaching cyber-physical systems and precision agriculture
- **IoT Development** — Reference implementation for agricultural IoT platforms

## 📂 Repository Structure

```
AgriTwin-GH/
├── feature_demos/          # Interactive Jupyter notebook demonstrations
│   ├── 01_uv_setup_and_imports.ipynb
│   ├── 02_synthetic_greenhouse_data_generator.ipynb
│   ├── 03_disease_risk_index_and_growth_stage.ipynb
│   ├── 04_digital_twin_simulator_and_whatif.ipynb
│   ├── 05_control_policy_mpc_like_actions_and_nonverbal_alerts.ipynb
│   ├── 06_dashboard_visualizations_comparison_ready.ipynb
│   ├── FEATURE_DEMOS_GUIDE.md
│   ├── data/               # Generated datasets
│   └── figures/            # Visualization outputs
├── docs/                   # Documentation source files
└── .github/workflows/      # CI/CD automation

```

## 🔬 Research & Development

This system demonstrates advanced concepts in:
- Cyber-physical system design for agriculture
- Digital twin technology and simulation
- Model predictive control in greenhouse environments
- Machine learning for agricultural risk assessment
- Human-machine interface design for agricultural systems

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

**[View Live Documentation](https://arjun-christopher.github.io/AgriTwin-GH/)** | **[Feature Guide](feature_demos/FEATURE_DEMOS_GUIDE.md)**
