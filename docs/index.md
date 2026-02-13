# AgriTwin-GH 🌱

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

### Implemented Components

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
- **NumPy & Pandas** — Data processing and analysis
- **Matplotlib & Seaborn** — Visualization and dashboards
- **Jupyter Notebooks** — Interactive demonstrations

## 📚 Quick Navigation

### 📖 Feature Demonstrations
**[Complete Feature Guide →](../feature_demos/FEATURE_DEMOS_GUIDE.md)**

Comprehensive walkthrough of all system capabilities with interactive Jupyter notebooks covering:

1. Environment setup and dependencies
2. Synthetic greenhouse data generation
3. Disease risk indexing and growth stage detection
4. Digital twin simulation and what-if analysis
5. Control policies with MPC-like actions and alerts
6. Dashboard visualizations and performance comparisons

### 🔧 Documentation
- **[Deployment Instructions](DOCS_DEPLOYMENT.md)** — Documentation website setup guide

## 🎯 Use Cases

- **Research** — Digital twin modeling in controlled environment agriculture
- **Commercial Agriculture** — Smart greenhouse operations and management
- **Education** — Teaching cyber-physical systems and precision agriculture
- **IoT Development** — Reference implementation for agricultural IoT platforms

## 🔬 Research & Development

This system demonstrates advanced concepts in:

- Cyber-physical system design for agriculture
- Digital twin technology and simulation
- Model predictive control in greenhouse environments
- Machine learning for agricultural risk assessment
- Human-machine interface design for agricultural systems

---

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

---

**[View on GitHub](https://github.com/arjun-christopher/AgriTwin-GH)** | **[Feature Guide](../feature_demos/FEATURE_DEMOS_GUIDE.md)**
