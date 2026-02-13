# AgriTwin-GH Feature Demonstrations

> **Comprehensive Digital Twin System for Smart Greenhouse Management**

This folder contains a complete demonstration of the **AgriTwin-GH** system — an intelligent digital twin platform for precision greenhouse control, disease risk management, and resource optimization. The demonstrations are presented as a series of interactive Jupyter notebooks that showcase advanced features beyond baseline greenhouse monitoring systems.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Folder Hierarchy](#folder-hierarchy)
4. [Prerequisites & Setup](#prerequisites--setup)
5. [Notebook Descriptions](#notebook-descriptions)
6. [Data Files](#data-files)
7. [Generated Figures](#generated-figures)
8. [Workflow](#workflow)
9. [Key Features](#key-features)
10. [Technical Details](#technical-details)
11. [Usage Instructions](#usage-instructions)
12. [Expected Outputs](#expected-outputs)

---

## 🌟 Overview

### What is AgriTwin-GH?

**AgriTwin-GH** is an advanced digital twin system designed for smart greenhouse management. It combines:
- **Real-time environmental monitoring** (temperature, humidity, CO2, soil conditions)
- **Disease risk prediction** using machine learning
- **Growth stage detection** for crop-specific optimization
- **Model Predictive Control (MPC)** for autonomous actuator management
- **What-if scenario simulation** for decision support
- **Resource optimization** (energy and water usage tracking)
- **Non-verbal human-machine interface** for operator alerts

### Why This Matters

Traditional greenhouse systems focus only on basic temperature and humidity control. AgriTwin-GH goes beyond by:
- **Preventing diseases** before they occur through risk indexing
- **Optimizing growth conditions** based on detected crop stage
- **Reducing resource consumption** through intelligent control
- **Enabling predictive analysis** via digital twin simulation
- **Providing actionable insights** through visual dashboards and alerts

### Target Audience

These demonstrations are designed for:
- **Researchers** studying digital twins in agriculture
- **Greenhouse operators** evaluating smart control systems
- **Students** learning about cyber-physical systems and IoT
- **Engineers** implementing precision agriculture solutions
- **Anyone** with no prior knowledge who wants to understand smart greenhouse technology in depth

---

## 🏗️ System Architecture

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

**Sequential Workflow:**
1. **Setup** → Install dependencies and configure environment
2. **Data Generation** → Create synthetic greenhouse sensor data
3. **Risk & Stage Analysis** → Compute disease risk and detect growth stages
4. **Digital Twin** → Calibrate simulation model for what-if scenarios
5. **Control Policy** → Implement MPC-like control with alerts
6. **Visualization** → Generate dashboards comparing to baseline systems

---

## 📁 Folder Hierarchy

```
feature_demos/
│
├── README.md                                          # This documentation file
│
├── 01_uv_setup_and_imports.ipynb                     # Environment setup & dependency installation
├── 02_synthetic_greenhouse_data_generator.ipynb      # Synthetic data generation (30 days)
├── 03_disease_risk_index_and_growth_stage.ipynb      # ML-based risk & stage detection
├── 04_digital_twin_simulator_and_whatif.ipynb        # Grey-box model calibration & simulation
├── 05_control_policy_mpc_like_actions_and_nonverbal_alerts.ipynb  # MPC control & HMI alerts
├── 06_dashboard_visualizations_comparison_ready.ipynb # Publication-ready dashboards
│
├── data/                                              # Generated data files
│   ├── greenhouse_data_5min.csv                      # 5-minute resolution sensor data (8,640 samples)
│   ├── greenhouse_data_hourly.csv                    # Hourly aggregated sensor data
│   ├── greenhouse_data_with_risk_and_stage.csv       # Enhanced data with ML predictions
│   ├── events_log.csv                                # Actuator events and interventions
│   └── feature_comparison.csv                        # AgriTwin-GH vs baseline capabilities
│
└── figures/                                           # Generated visualizations
    ├── data_generation_overview.png                  # Synthetic data generation summary
    ├── fig_baseline_temp_humidity.png                # Baseline system equivalent plot
    ├── fig_dashboard_snapshot.png                    # Complete dashboard visualization
    ├── fig_disease_risk_index.png                    # Disease risk trends over time
    ├── fig_growth_stage_timeline.png                 # Crop growth stage progression
    ├── fig_whatif_fan_on_off.png                     # What-if scenario comparison
    ├── fig_control_vs_nocontrol_resources.png        # Controlled vs uncontrolled resource usage
    ├── digital_twin_validation.png                   # Model prediction accuracy
    ├── environmental_forecasting.png                 # Multi-step environmental predictions
    ├── lstm_disease_risk_prediction.png              # LSTM-based risk forecasting
    ├── lstm_temporal_progression.png                 # Temporal disease risk evolution
    ├── model_comparison_importance.png               # ML model feature importance
    ├── stage_feature_importance.png                  # Growth stage classifier features
    ├── alert_timeline.png                            # Non-verbal alert history
    └── operator_panel.png                            # HMI operator interface mockup
```

---

## 🔧 Prerequisites & Setup

### System Requirements

- **Python:** 3.12.1 or higher
- **Operating System:** Linux, macOS, or Windows with WSL
- **Memory:** Minimum 4GB RAM (8GB recommended)
- **Storage:** At least 500MB free space for data and figures

### Required Software

- **Jupyter Notebook** or **JupyterLab**
- **uv** package manager (fast Python package installer)
  - Installation: `pip install uv`

### Dependencies

All dependencies are automatically installed in **Notebook 01**. Key packages include:

**Data Processing:**
- `numpy` (1.26.4)
- `pandas` (2.2.1)
- `scipy` (1.12.0)

**Machine Learning:**
- `scikit-learn` (1.4.1.post1)
- `statsmodels` (0.14.1)

**Visualization:**
- `matplotlib` (3.8.3)
- `seaborn` (0.13.2)
- `plotly` (5.19.0)

**Jupyter Extensions:**
- `ipywidgets` (8.1.2)
- `ipython` (8.22.1)

**Note:** Complete list of 41 packages with versions is available in Notebook 01.

---

## 📓 Notebook Descriptions

### 01. Environment Setup and Imports
**File:** `01_uv_setup_and_imports.ipynb`

**Purpose:**  
Prepares the computational environment for all subsequent notebooks.

**What it does:**
- Installs all 41 required Python packages using the `uv` package manager
- Verifies successful imports and displays version information
- Configures default plotting styles for publication-quality figures
- Creates necessary directories (`data/`, `figures/`)
- Runs diagnostic tests to ensure environment is ready

**Key Outputs:**
- Confirmation of Python 3.12.1 environment
- List of installed package versions
- Test visualization proving matplotlib/seaborn functionality

**Estimated Runtime:** 2-3 minutes (first run with package installation)

**Who should run this:**  
Everyone — this is the mandatory first step before any other notebook.

---

### 02. Synthetic Greenhouse Data Generator
**File:** `02_synthetic_greenhouse_data_generator.ipynb`

**Purpose:**  
Generates realistic synthetic greenhouse sensor data for testing and demonstration.

**What it does:**
- Creates **30 days** of synthetic sensor readings at **5-minute intervals** (8,640 total samples)
- Simulates **10 sensor types:**
  - **Environmental:** Temperature (°C), Humidity (%), Light intensity (lux), CO₂ concentration (ppm)
  - **Soil:** Moisture (%), pH level, Electrical conductivity (mS/cm)
  - **Derived metrics:** Leaf wetness (0-1), Ventilation rate (%), Air circulation (%)
- Models **4 crop growth stages:** Vegetative → Flowering → Fruiting → Harvest
- Implements realistic patterns:
  - **Diurnal cycles** (day/night temperature/light variations)
  - **Stage-aware setpoints** (different optimal conditions per growth stage)
  - **Inter-variable correlations** (e.g., temperature affects humidity)
  - **Sensor noise and drift**
  - **Missing data patterns** (realistic sensor failures)
- Generates **event log** tracking 86 discrete events:
  - Irrigation cycles
  - Ventilation changes
  - Heating/cooling activations

**Outputs:**
- `greenhouse_data_5min.csv` — High-resolution time series (8,640 rows × 13 columns)
- `greenhouse_data_hourly.csv` — Aggregated hourly statistics
- `events_log.csv` — Timestamped actuator events
- `data_generation_overview.png` — Visual summary of generated data

**Scientific Basis:**
- Temperature ranges based on tomato crop requirements
- Humidity-temperature inverse relationship (psychrometric principles)
- CO₂ enrichment strategies per growth stage
- Soil moisture dynamics following irrigation patterns

**Estimated Runtime:** 30-60 seconds

**Why synthetic data?**  
Allows controlled experimentation without requiring real greenhouse hardware. The data exhibits realistic physical relationships suitable for training machine learning models.

---

### 03. Disease Risk Index and Growth Stage Detection
**File:** `03_disease_risk_index_and_growth_stage.ipynb`

**Purpose:**  
Implements disease risk assessment and machine learning-based crop growth stage detection.

**What it does:**

#### Disease Risk Indexing
Computes a **Disease Risk Index (0-100 scale)** every 5 minutes based on:

**For Tomato Crops:**
- **Leaf Mold Risk:**
  - Triggers: High humidity (>80%) + leaf wetness + temperatures 18-25°C
  - Peak risk when all conditions align
- **Spider Mite Risk:**
  - Triggers: Hot conditions (>28°C) + low humidity (<50%)
  - Common in dry greenhouse environments

**For Strawberry Crops (also modeled):**
- **Powdery Mildew Risk:** Moderate humidity (60-75%) + specific temperature range
- **Leaf Scorch Risk:** High temperature + dry soil conditions

**Risk Components:**
- **Instantaneous risk score:** Current environmental conditions
- **Risk budget:** Cumulative high-risk minutes over rolling windows
- **Risk velocity:** Rate of risk increase (early warning)

#### Growth Stage Detection
Implements a **RandomForest classifier** to automatically detect crop growth stage:

**Features used:**
- Day index (time since planting)
- Temperature statistics (mean, min, max)
- Light integral (cumulative daily light)
- Soil moisture trends
- Humidity patterns

**Outputs:**
- Stage label (Vegetative / Flowering / Fruiting / Harvest)
- Confidence percentage (model certainty)
- Feature importance rankings

**Machine Learning Details:**
- **Algorithm:** RandomForest (100 trees)
- **Training:** Supervised learning on labeled synthetic data
- **Performance:** ~95% accuracy on test set
- **Interpretability:** Feature importance helps understand crop signals

**Key Outputs:**
- `greenhouse_data_with_risk_and_stage.csv` — Enhanced dataset with risk scores and stage predictions
- `fig_disease_risk_index.png` — Time series of disease risk
- `fig_growth_stage_timeline.png` — Stage progression over 30 days
- `stage_feature_importance.png` — Which features matter for stage detection

**Practical Use:**
- **Preventive action:** Intervene when risk exceeds threshold (e.g., 65/100)
- **Stage-aware control:** Adjust climate setpoints based on detected stage
- **Operator alerts:** Visual indicators when disease risk enters danger zone

**Estimated Runtime:** 1-2 minutes

---

### 04. Digital Twin Simulator and What-If Analysis
**File:** `04_digital_twin_simulator_and_whatif.ipynb`

**Purpose:**  
Develops a grey-box digital twin model for greenhouse simulation and scenario analysis.

**What it does:**

#### Digital Twin Model
Implements a **grey-box state-space model** that predicts:
- Temperature (°C)
- Humidity (%)
- CO₂ concentration (ppm)
- Soil moisture (%)

**Model Structure:**
```
x(t+1) = f(x(t), u(t), disturbances)

where:
  x(t) = current environmental state
  u(t) = actuator commands (vent, fan, heater, irrigation, LED, CO₂ injection)
  f = learned state-transition function
```

**Modeling Approach:**
- **Grey-box:** Combines physics-inspired structure with data-driven parameter fitting
- **Method:** Ridge regression or ARX (AutoRegressive with eXogenous inputs)
- **Time step:** 5 minutes (configurable)

#### Model Calibration
- **Training data:** 8,640 synthetic samples (from Notebook 02)
- **Train/test split:** 80/20
- **Metrics:**
  - Temperature: R² > 0.95, MAE < 0.5°C
  - Humidity: R² > 0.92, MAE < 2%
  - CO₂: R² > 0.90, MAE < 50 ppm

#### What-If Scenarios
Enables rapid simulation of hypothetical scenarios:

**Example Questions:**
- "What happens to temperature if I turn the fan ON for 2 hours?"
- "How does humidity change if ventilation increases by 20%?"
- "What's the energy cost of maintaining 25°C on a hot day?"

**Workflow:**
1. Specify actuator actions (e.g., fan_speed = 100%)
2. Run simulation forward in time
3. Compare predicted outcomes to baseline
4. Visualize differences

**Outputs:**
- `digital_twin_validation.png` — Predicted vs actual values (model accuracy)
- `fig_whatif_fan_on_off.png` — Example scenario: fan ON vs OFF comparison
- `environmental_forecasting.png` — Multi-step ahead predictions

**Control Applications:**
- **Model Predictive Control (MPC):** Optimize actuator commands by simulating future states
- **Energy optimization:** Test different control strategies virtually
- **Risk-free experimentation:** Evaluate strategies without affecting real crops

**Technical Notes:**
- Model update frequency: Every new data point (adaptive learning possible)
- Computational cost: <100ms per simulation step (real-time capable)
- Uncertainty quantification: Confidence intervals on predictions

**Estimated Runtime:** 2-3 minutes (model training + validation)

---

### 05. Control Policy, MPC-like Actions, and Non-Verbal Alerts
**File:** `05_control_policy_mpc_like_actions_and_nonverbal_alerts.ipynb`

**Purpose:**  
Implements an intelligent control system with MPC-style optimization and a non-verbal operator interface.

**What it does:**

#### MPC-like Controller
Implements a **Model Predictive Control (MPC)** approach:

**Control Objectives:**
1. **Climate regulation:** Maintain temperature, humidity, CO₂ near stage-specific setpoints
2. **Disease prevention:** Keep disease risk index below 65/100
3. **Resource efficiency:** Minimize energy (kWh) and water (liters) consumption
4. **Actuator protection:** Enforce cooldown periods to prevent rapid cycling

**Stage-Specific Setpoints:**

| Growth Stage | Temperature | Humidity | CO₂  |
|--------------|-------------|----------|------|
| Vegetative   | 22°C        | 70%      | 800  |
| Flowering    | 20°C        | 65%      | 1000 |
| Fruiting     | 21°C        | 60%      | 1200 |
| Harvest      | 20°C        | 55%      | 900  |

**Actuator Commands:**
- **Heating:** ON/OFF (deadband control)
- **Cooling/Ventilation:** 0-100% (proportional)
- **Fan:** Variable speed 0-100%
- **Irrigation:** ON when soil moisture < threshold
- **LED grow lights:** ON/OFF (photoperiod control)
- **CO₂ injection:** Pulse injection when below setpoint

**Control Logic:**
- **Proportional control:** Actuator intensity proportional to error
- **Deadband zones:** Avoid oscillations (e.g., ±1°C around setpoint)
- **Cooldown enforcement:** Minimum 15 minutes between heater cycles
- **Disease-aware:** Reduce humidity aggressively if risk >65

#### Resource Tracking
Monitors cumulative consumption:
- **Energy (kWh):**
  - Heater: 3 kW × runtime
  - Fan: 0.5 kW × runtime
  - LED: 0.2 kW × runtime
- **Water (liters):** 50L per irrigation event

#### Non-Verbal Alert System
Implements a **color-coded Human-Machine Interface (HMI):**

**Alert Levels:**
- 🟢 **GREEN (Normal):** All parameters within acceptable range, disease risk <40
- 🟡 **YELLOW (Warning):** One parameter near limit OR disease risk 40-65
- 🔴 **RED (Alert):** Parameter out of range OR disease risk >65

**Visual Elements:**
- Color-coded status indicator
- Icon-based notifications (thermometer, droplet, leaf, etc.)
- Timestamp of last alert change
- Quick-glance metric summary

**Operator Panel Features:**
- Current environmental state (temperature, humidity, CO₂, soil)
- Active actuator states (ON/OFF indicators)
- Disease risk gauge (0-100 with color zones)
- Resource usage (kWh, liters)
- Alert timeline (history of status changes)

#### Controlled vs Uncontrolled Comparison
Simulates two scenarios:
1. **With control:** MPC actively manages actuators
2. **Without control (baseline):** Minimal intervention

**Metrics Compared:**
- Temperature/humidity stability
- Disease risk reduction
- Energy consumption
- Water usage

**Typical Results:**
- **Disease risk:** 40% lower with control
- **Energy usage:** 15% lower (optimized heating/cooling)
- **Climate stability:** 3x better (lower standard deviation)

**Outputs:**
- `fig_control_vs_nocontrol_resources.png` — Resource usage comparison
- `alert_timeline.png` — History of alert status changes
- `operator_panel.png` — Non-verbal HMI mockup

**Estimated Runtime:** 3-4 minutes (full simulation with control)

---

### 06. Dashboard Visualizations (Comparison Ready)
**File:** `06_dashboard_visualizations_comparison_ready.ipynb`

**Purpose:**  
Creates publication-quality dashboards comparing AgriTwin-GH to baseline greenhouse systems.

**What it does:**

#### Baseline-Compatible Visualizations
Recreates standard greenhouse monitoring plots:
- **Figure 2 equivalent:** Temperature and humidity time series (7-day window)
- **Simple dashboard:** Current readings + actuator states + basic alerts

**Purpose:** Demonstrates that AgriTwin-GH includes all baseline features PLUS enhancements.

#### Enhanced Visualizations (AgriTwin-GH Exclusive)
Showcases novel capabilities not available in baseline systems:

1. **Disease Risk Dashboard:**
   - Real-time risk index (0-100 scale)
   - Per-disease breakdowns (leaf mold, spider mites, etc.)
   - Risk budget cumulative tracking
   - Predictive risk forecasting (LSTM-based)

2. **Growth Stage Tracking:**
   - Automated stage detection timeline
   - Confidence scores per stage
   - Stage-aware setpoint visualization

3. **Digital Twin Validation:**
   - Predicted vs actual comparisons
   - Model accuracy metrics (R², MAE, RMSE)
   - What-if scenario comparisons

4. **Control Performance:**
   - Controlled vs uncontrolled resource usage
   - Energy/water savings quantification
   - Climate stability improvements

5. **Operator Interface:**
   - Non-verbal alert system demonstration
   - Icon-based status indicators
   - Quick-glance metric panels

#### Feature Comparison Table
Generates `feature_comparison.csv` documenting capabilities:

| Feature | Baseline System | AgriTwin-GH |
|---------|-----------------|-------------|
| Temperature monitoring | ✅ Yes | ✅ Yes |
| Humidity monitoring | ✅ Yes | ✅ Yes |
| Actuator control | ✅ Basic | ✅ Advanced (MPC) |
| Disease risk indexing | ❌ No | ✅ Yes |
| Growth stage detection | ❌ No | ✅ Yes |
| Digital twin simulation | ❌ No | ✅ Yes |
| What-if scenarios | ❌ No | ✅ Yes |
| Resource optimization | ❌ No | ✅ Yes |
| Non-verbal alerts | ❌ No | ✅ Yes |
| Predictive forecasting | ❌ No | ✅ Yes |

#### Publication Quality
All figures saved with:
- **DPI:** 300 (publication standard)
- **Format:** PNG (with transparency support)
- **Consistent styling:** Color palette, fonts, sizes
- **Descriptive filenames:** Easy identification

**Visualization Period:**
- Main focus: Days 8-14 (representative 7-day window)
- Reason: Shows all growth stages and typical risk patterns

**Outputs:**
All figures in `figures/` directory:
- `fig_dashboard_snapshot.png` — Complete dashboard view
- `fig_baseline_temp_humidity.png` — Baseline system equivalent
- `model_comparison_importance.png` — ML model feature rankings
- `lstm_disease_risk_prediction.png` — Predictive risk modeling
- `lstm_temporal_progression.png` — Temporal risk evolution
- And 10 more publication-ready visualizations

**Estimated Runtime:** 2-3 minutes (generating all figures)

---

## 📊 Data Files

### Input Data (Auto-Generated)

#### `greenhouse_data_5min.csv`
**Generated by:** Notebook 02  
**Size:** 8,640 rows × 13 columns  
**Time resolution:** 5 minutes  
**Duration:** 30 days  
**Columns:**
- `timestamp` — ISO 8601 datetime
- `temp_c` — Temperature (°C)
- `humidity_pct` — Relative humidity (%)
- `light_lux` — Light intensity (lux)
- `co2_ppm` — CO₂ concentration (ppm)
- `soil_moisture_pct` — Soil moisture (%)
- `soil_ph` — Soil pH (4-8 range)
- `soil_ec_ms_cm` — Electrical conductivity (mS/cm)
- `leaf_wetness` — Proxy for leaf wetness (0-1)
- `vent_rate_pct` — Ventilation opening (%)
- `air_circulation_pct` — Fan speed (%)
- `growth_stage` — Categorical stage label
- `day_index` — Days since planting

**Use cases:**
- Training machine learning models
- Control algorithm testing
- Digital twin calibration

---

#### `greenhouse_data_hourly.csv`
**Generated by:** Notebook 02  
**Size:** 720 rows × 13 columns  
**Time resolution:** 1 hour (aggregated from 5-minute data)  
**Aggregation method:** Mean for most columns, mode for categorical

**Use cases:**
- Trend analysis
- Long-term pattern visualization
- Reduced computational load for certain analyses

---

#### `greenhouse_data_with_risk_and_stage.csv`
**Generated by:** Notebook 03  
**Size:** 8,640 rows × 20+ columns  
**Extends:** `greenhouse_data_5min.csv` with additional ML-derived columns:
- `disease_risk_index` — Composite risk score (0-100)
- `leaf_mold_risk` — Tomato leaf mold specific risk
- `spider_mite_risk` — Spider mite infestation risk
- `powdery_mildew_risk` — Strawberry powdery mildew risk (if applicable)
- `risk_budget_6h` — Cumulative high-risk minutes (6-hour window)
- `predicted_stage` — ML-predicted growth stage
- `stage_confidence_pct` — Prediction confidence (0-100%)

**Use cases:**
- Control policy decisions
- Alert generation
- Risk trend analysis
- Stage-aware actuator management

---

#### `events_log.csv`
**Generated by:** Notebook 02  
**Size:** ~86 rows × 4 columns  
**Columns:**
- `timestamp` — Event occurrence time
- `event_type` — Type of event (irrigation, vent_change, heating_on, etc.)
- `description` — Human-readable event description
- `value` — Numeric value if applicable (e.g., new vent_rate)

**Event types:**
- `irrigation_start` / `irrigation_stop`
- `vent_change` (ventilation adjustment)
- `heating_on` / `heating_off`
- `fan_speed_change`
- `co2_injection_pulse`

**Use cases:**
- Correlating actuator actions with environmental changes
- Control algorithm validation
- Event-driven analysis

---

#### `feature_comparison.csv`
**Generated by:** Notebook 06  
**Size:** ~10 rows × 3 columns  
**Columns:**
- `Feature` — Feature/capability name
- `Baseline_System` — Present in baseline? (Yes/No)
- `AgriTwin_GH` — Present in AgriTwin-GH? (Yes/Advanced/etc.)

**Use cases:**
- Capability comparison tables for reports
- Justification for advanced features
- Marketing/presentation material

---

## 🖼️ Generated Figures

All figures are saved in the `figures/` directory in **PNG format at 300 DPI** for publication quality.

### Data Generation & Validation

#### `data_generation_overview.png`
**Source:** Notebook 02  
**Shows:** Summary of synthetic data generation  
**Panels:** Temperature, humidity, light, CO₂, soil moisture over 30 days  
**Purpose:** Validate that synthetic data exhibits realistic patterns

---

#### `digital_twin_validation.png`
**Source:** Notebook 04  
**Shows:** Predicted vs actual environmental values  
**Metrics:** R², MAE, RMSE for each variable  
**Purpose:** Demonstrate digital twin model accuracy

---

### Disease Risk & Growth Stage

#### `fig_disease_risk_index.png`
**Source:** Notebook 03  
**Shows:** Disease risk index (0-100) over time  
**Features:**
- Composite risk line
- Per-disease breakdowns (leaf mold, spider mites)
- Risk threshold line (65)
- High-risk zones highlighted

**Purpose:** Demonstrate disease risk indexing capability

---

#### `fig_growth_stage_timeline.png`
**Source:** Notebook 03  
**Shows:** Detected growth stages over 30 days  
**Features:**
- Stage transitions (Vegetative → Flowering → Fruiting → Harvest)
- Confidence scores per prediction
- Stage duration bars

**Purpose:** Validate growth stage detection algorithm

---

#### `stage_feature_importance.png`
**Source:** Notebook 03  
**Shows:** RandomForest feature importance for stage classification  
**Features:** Bar chart of most influential features (day index, temperature, light, etc.)  
**Purpose:** Explain what signals drive stage detection

---

#### `lstm_disease_risk_prediction.png`
**Source:** Notebook 06  
**Shows:** LSTM-based disease risk forecasting  
**Features:**
- Historical risk (blue)
- Predicted risk 12 hours ahead (orange)
- Confidence intervals

**Purpose:** Demonstrate predictive disease risk capabilities

---

#### `lstm_temporal_progression.png`
**Source:** Notebook 06  
**Shows:** How disease risk evolves over multiple time horizons  
**Purpose:** Show temporal patterns in risk progression

---

### Digital Twin & What-If Analysis

#### `fig_whatif_fan_on_off.png`
**Source:** Notebook 04  
**Shows:** Comparison of two scenarios: fan ON vs fan OFF  
**Panels:** Temperature and humidity trajectories  
**Purpose:** Demonstrate what-if scenario simulation

---

#### `environmental_forecasting.png`
**Source:** Notebook 04  
**Shows:** Multi-step ahead environmental predictions  
**Variables:** Temperature, humidity, CO₂ (1-4 hours ahead)  
**Purpose:** Validate predictive modeling for MPC

---

### Control & Resource Management

#### `fig_control_vs_nocontrol_resources.png`
**Source:** Notebook 05  
**Shows:** Controlled vs uncontrolled resource consumption  
**Metrics:**
- Energy usage (kWh)
- Water usage (liters)
- Disease risk reduction

**Purpose:** Quantify benefits of intelligent control

---

#### `fig_baseline_temp_humidity.png`
**Source:** Notebook 06  
**Shows:** Temperature and humidity time series (7-day window)  
**Style:** Matches baseline system "Figure 2" format  
**Purpose:** Direct comparison to baseline capabilities

---

### Operator Interface

#### `alert_timeline.png`
**Source:** Notebook 05  
**Shows:** History of alert status changes (Green/Yellow/Red)  
**Features:** Color-coded timeline with timestamps  
**Purpose:** Demonstrate non-verbal alert system

---

#### `operator_panel.png`
**Source:** Notebook 05  
**Shows:** Mockup of operator HMI panel  
**Elements:**
- Current environmental metrics
- Actuator status indicators
- Disease risk gauge
- Resource usage counters
- Alert status

**Purpose:** Visualize proposed operator interface

---

#### `fig_dashboard_snapshot.png`
**Source:** Notebook 06  
**Shows:** Complete dashboard with all AgriTwin-GH features  
**Panels:**
- Environmental time series
- Disease risk trends
- Growth stage indicator
- Resource usage
- Alert status

**Purpose:** Comprehensive system overview for presentations

---

### Model Analysis

#### `model_comparison_importance.png`
**Source:** Notebook 06  
**Shows:** Feature importance comparison across different ML models  
**Purpose:** Compare RandomForest, Gradient Boosting, SVM for stage detection

---

## 🔄 Workflow

### Sequential Execution Order

The notebooks are designed to be executed **in numerical order**. Each notebook builds upon outputs from previous notebooks.

```
START
  │
  ├─▶ [1] Setup Environment
  │     └─▶ Install packages, configure settings
  │
  ├─▶ [2] Generate Synthetic Data
  │     └─▶ Creates: greenhouse_data_5min.csv, events_log.csv
  │
  ├─▶ [3] Compute Risk & Stage
  │     └─▶ Creates: greenhouse_data_with_risk_and_stage.csv
  │
  ├─▶ [4] Calibrate Digital Twin
  │     └─▶ Creates: digital_twin_validation.png
  │
  ├─▶ [5] Run Control Simulation
  │     └─▶ Creates: alert_timeline.png, resource comparisons
  │
  └─▶ [6] Generate Dashboards
        └─▶ Creates: All publication figures
  
END
```

### Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────┐
│ Notebook 01: Setup                                           │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Python environment ready
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Notebook 02: Data Generator                                  │
│  Outputs: greenhouse_data_5min.csv, events_log.csv           │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Raw sensor data
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Notebook 03: Risk & Stage Analysis                           │
│  Input: greenhouse_data_5min.csv                             │
│  Output: greenhouse_data_with_risk_and_stage.csv             │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Enhanced data with ML predictions
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Notebook 04: Digital Twin                                    │
│  Input: greenhouse_data_with_risk_and_stage.csv              │
│  Output: Calibrated simulation model                         │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Predictive model ready
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Notebook 05: Control Policy                                  │
│  Input: All previous outputs                                 │
│  Output: Control simulation results, alerts                  │
└──────────────────────────────────────────────────────────────┘
                            │
                            │ Complete system demonstration
                            ▼
┌──────────────────────────────────────────────────────────────┐
│ Notebook 06: Dashboards                                      │
│  Input: All previous outputs                                 │
│  Output: Publication-quality visualizations                  │
└──────────────────────────────────────────────────────────────┘
```

### Dependency Matrix

| Notebook | Depends On | Produces | Used By |
|----------|------------|----------|---------|
| 01 | None | Environment setup | All |
| 02 | 01 | Raw sensor data | 03, 04, 05, 06 |
| 03 | 01, 02 | Risk & stage predictions | 04, 05, 06 |
| 04 | 01, 02, 03 | Digital twin model | 05, 06 |
| 05 | 01, 02, 03, 04 | Control results | 06 |
| 06 | 01, 02, 03, 04, 05 | Final dashboards | None (end product) |

---

## ✨ Key Features

### What Makes AgriTwin-GH Different?

#### 1. **Disease Risk Indexing** 🦠
- **What it is:** Real-time calculation of disease probability based on environmental conditions
- **How it works:** Multi-factor scoring considering temperature, humidity, leaf wetness, soil conditions
- **Why it matters:** Enables preventive action before disease symptoms appear
- **Example:** If humidity stays >80% for 6 hours with leaf wetness present, leaf mold risk rises to CRITICAL

#### 2. **Automated Growth Stage Detection** 🌱
- **What it is:** Machine learning classifier that identifies crop developmental stage
- **How it works:** RandomForest model trained on temporal patterns, environmental exposure, soil trends
- **Why it matters:** Different stages need different climate setpoints (e.g., flowering needs more CO₂)
- **Example:** System automatically detects flowering stage and increases CO₂ from 800 to 1000 ppm

#### 3. **Digital Twin Simulation** 🔮
- **What it is:** Virtual replica of the greenhouse that predicts future states
- **How it works:** Grey-box model combining physics equations with data-driven parameters
- **Why it matters:** Test control strategies without risking real crops
- **Example:** "If I increase ventilation by 20%, temperature will drop 2°C in 30 minutes"

#### 4. **Model Predictive Control (MPC)** 🎯
- **What it is:** Advanced control algorithm that optimizes actuator commands
- **How it works:** Simulates future scenarios, chooses actions minimizing cost function
- **Why it matters:** Balances multiple objectives (climate, disease risk, energy, water)
- **Example:** Controller delays heating to avoid risk spike, saves 15% energy vs baseline

#### 5. **Non-Verbal Operator Interface** 🚦
- **What it is:** Color-coded alert system requiring no technical expertise
- **How it works:** Green (OK) / Yellow (Warning) / Red (Alert) with icon-based notifications
- **Why it matters:** Operators make informed decisions without reading complex data
- **Example:** Red alert + leaf icon = disease risk critical, reduce humidity immediately

#### 6. **Resource Optimization** 💧⚡
- **What it is:** Tracking and minimization of energy (kWh) and water (liters) consumption
- **How it works:** Controller considers resource cost in decision-making
- **Why it matters:** Reduces operational costs while maintaining crop health
- **Example:** Intelligent control saves ~15% energy and 20% water vs uncontrolled baseline

#### 7. **What-If Scenario Analysis** 🤔
- **What it is:** Ability to simulate hypothetical interventions before implementing
- **How it works:** Digital twin runs forward in time with specified actuator actions
- **Why it matters:** Risk-free evaluation of strategies
- **Example:** "If I run fan at 100% for 2 hours, disease risk drops 15 points but energy cost is $2.50"

#### 8. **Predictive Forecasting** 📈
- **What it is:** LSTM-based models predicting disease risk and environmental trends hours ahead
- **How it works:** Recurrent neural network trained on temporal sequences
- **Why it matters:** Enables proactive rather than reactive management
- **Example:** System predicts risk will exceed threshold in 4 hours, suggests ventilation increase now

---

## 🔬 Technical Details

### Machine Learning Models

#### Growth Stage Classifier
- **Algorithm:** RandomForest (ensemble of 100 decision trees)
- **Input features:** 8 features (day index, temp stats, light integral, soil trends)
- **Output:** Stage label + confidence percentage
- **Training data:** 8,640 labeled samples
- **Performance:** ~95% accuracy on test set
- **Update frequency:** Every 5 minutes (real-time inference)
- **Computational cost:** <50ms per prediction

**Why RandomForest?**
- Handles non-linear relationships
- Resistant to overfitting
- Interpretable (feature importance)
- No hyperparameter tuning needed
- Fast inference

#### Disease Risk Models
- **Approach:** Rule-based scoring with fuzzy logic
- **Inputs:** Temperature, humidity, leaf wetness, soil moisture, pH
- **Outputs:** Per-disease scores (0-100) + composite index
- **Diseases modeled:**
  - Tomato: Leaf mold, Spider mites
  - Strawberry: Powdery mildew, Leaf scorch (optional)
- **Update frequency:** Every 5 minutes
- **Threshold:** Risk >65 triggers alerts

**Scoring Logic Example (Leaf Mold):**
```python
risk = 0
if humidity > 80%: risk += 40
if leaf_wetness > 0.6: risk += 30
if 18°C < temp < 25°C: risk += 30
if risk_budget_6h > 180 minutes: risk += 20
return min(risk, 100)
```

#### Digital Twin Model
- **Type:** Grey-box state-space model
- **Method:** Ridge regression / ARX
- **States:** Temperature, Humidity, CO₂, Soil moisture
- **Inputs (actuators):** Vent rate, fan speed, heater, irrigation, LED, CO₂ injection
- **Time step:** 5 minutes
- **Training:** 6,912 samples (80% of dataset)
- **Validation:** 1,728 samples (20% of dataset)
- **Performance:**
  - Temperature: R² = 0.95, MAE = 0.4°C
  - Humidity: R² = 0.92, MAE = 1.8%
  - CO₂: R² = 0.90, MAE = 45 ppm

**Model Equations (simplified):**
```
T(t+1) = α₁·T(t) + β₁·Heater(t) - γ₁·Vent(t) + δ₁·Tamb(t)
H(t+1) = α₂·H(t) - β₂·Vent(t) + γ₂·Irrigation(t)
CO₂(t+1) = α₃·CO₂(t) + β₃·CO₂_injection(t) - γ₃·Vent(t)
```
Parameters (α, β, γ, δ) are learned from data.

#### LSTM Risk Predictor (Notebook 06)
- **Architecture:** 2-layer LSTM with 64 hidden units per layer
- **Input:** 48 time steps (4 hours) of environmental data
- **Output:** Disease risk 12 steps ahead (1 hour forecast)
- **Training:** 7,000 sequences with sliding window
- **Performance:** MAE = 8.2 on risk index (0-100 scale)
- **Use case:** Early warning system

---

### Control Algorithms

#### MPC-like Controller (Notebook 05)
**Objective Function:**
```
minimize: w₁·(T - T_target)² + w₂·(H - H_target)² 
          + w₃·DiseaseRisk + w₄·Energy + w₅·Water

subject to:
  - Tmin ≤ T ≤ Tmax
  - Hmin ≤ H ≤ Hmax
  - DiseaseRisk ≤ 65
  - Actuator cooldown constraints
```

**Weights (tunable):**
- w₁ (temp error) = 10
- w₂ (humidity error) = 8
- w₃ (disease risk) = 15
- w₄ (energy) = 5
- w₅ (water) = 3

**Control Loop:**
1. Measure current state
2. Detect growth stage → update setpoints
3. Compute disease risk
4. Run digital twin to predict next state
5. Optimize actuator commands
6. Apply commands (if cooldown allows)
7. Track resource usage
8. Update alert status
9. Wait 5 minutes, repeat

**Actuator Constraints:**
- Heater: Minimum 15 minutes between ON cycles
- Irrigation: Minimum 2 hours between events
- Vent: Rate change ≤20% per 5 minutes
- Fan: Speed change ≤10% per 5 minutes

---

### Data Pipeline

#### Data Flow Architecture
```
Raw Sensors (10 types, 5-min resolution)
  ↓
Preprocessing (outlier removal, interpolation)
  ↓
Feature Engineering (risk metrics, trends)
  ↓
Machine Learning (stage detection, risk indexing)
  ↓
Digital Twin (state prediction)
  ↓
Control Algorithm (actuator optimization)
  ↓
Resource Tracking (energy, water)
  ↓
Alert System (Green/Yellow/Red)
  ↓
Dashboard Visualization
```

#### Time Series Processing
- **Sampling rate:** 5 minutes (configurable)
- **Missing data handling:** Linear interpolation (max gap: 30 minutes)
- **Outlier detection:** 3-sigma rule
- **Smoothing:** Rolling median (window: 3 samples for noise reduction)
- **Aggregation:** Hourly mean for long-term trends

#### CSV Data Format
All CSV files use:
- **Encoding:** UTF-8
- **Delimiter:** Comma (`,`)
- **Timestamp format:** ISO 8601 (`YYYY-MM-DDTHH:MM:SS`)
- **Decimal separator:** Period (`.`)
- **Missing values:** Empty string or `NaN`

---

### Software Stack

#### Core Dependencies
```
Python 3.12.1
├── NumPy 1.26.4          (numerical computing)
├── Pandas 2.2.1          (data manipulation)
├── Matplotlib 3.8.3      (plotting)
├── Seaborn 0.13.2        (statistical visualization)
├── scikit-learn 1.4.1    (machine learning)
├── SciPy 1.12.0          (scientific computing)
└── Plotly 5.19.0         (interactive plots)
```

#### Package Manager
- **uv:** Fast, reliable Python package installer (Rust-based)
- **Advantages over pip:**
  - 10-100x faster dependency resolution
  - Deterministic installs
  - Better error messages

#### Notebook Environment
- **Jupyter Notebook 7.1.0** or **JupyterLab 4.1.2**
- **IPython 8.22.1** (enhanced REPL)
- **ipywidgets 8.1.2** (interactive controls)

---

### Computational Requirements

#### Performance Metrics

| Task | Runtime | Memory | CPU |
|------|---------|--------|-----|
| Setup (Notebook 01) | 2-3 min | 200 MB | Low |
| Data generation (Notebook 02) | 30-60 sec | 150 MB | Medium |
| Risk/stage detection (Notebook 03) | 1-2 min | 250 MB | Medium |
| Digital twin training (Notebook 04) | 2-3 min | 300 MB | High |
| Control simulation (Notebook 05) | 3-4 min | 400 MB | High |
| Dashboard generation (Notebook 06) | 2-3 min | 350 MB | Medium |
| **Total (full pipeline)** | **~15 min** | **<500 MB** | **Medium** |

#### Scalability
- **Tested with:** 30 days (8,640 samples)
- **Scales to:** 1+ year (100,000+ samples) with minor performance degradation
- **Bottlenecks:** Plot rendering (can disable for large datasets)
- **Optimization:** Use hourly data for long-term analysis

---

## 📖 Usage Instructions

### First-Time Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/arjun-christopher/AgriTwin-GH.git
   cd AgriTwin-GH/feature_demos
   ```

2. **Install Jupyter:**
   ```bash
   pip install jupyter
   # or for JupyterLab:
   pip install jupyterlab
   ```

3. **Start Jupyter:**
   ```bash
   jupyter notebook
   # or:
   jupyter lab
   ```

4. **Open Notebook 01:**
   - Navigate to `01_uv_setup_and_imports.ipynb`
   - Run all cells (Cell → Run All)
   - Wait for package installation (~2-3 minutes)
   - Verify no errors

### Sequential Execution

**Follow this order strictly:**

```bash
01 → 02 → 03 → 04 → 05 → 06
```

**For each notebook:**
1. Open the notebook file
2. Read the introductory markdown cells
3. Run all cells sequentially (Shift+Enter or Cell → Run All)
4. Review outputs and visualizations
5. Check that expected files are created in `data/` or `figures/`
6. Proceed to next notebook

### Running Individual Notebooks

If you want to run only a subset:

- **Just setup:** Run Notebook 01
- **Data generation only:** Run Notebooks 01 → 02
- **Risk analysis only:** Run Notebooks 01 → 02 → 03
- **Full pipeline:** Run all notebooks 01 → 06

**Important:** You cannot skip dependencies. For example, Notebook 04 requires outputs from Notebooks 02 and 03.

### Customization Options

#### Modify Data Generation Parameters (Notebook 02)
```python
# Change simulation duration
num_days = 30  # Default: 30 days (increase for longer simulations)

# Change time resolution
time_step_minutes = 5  # Default: 5 minutes

# Change crop type
crop_type = "tomato"  # Options: "tomato", "strawberry", "lettuce"

# Modify growth stage durations
stage_durations = {
    "vegetative": 7,   # days
    "flowering": 10,
    "fruiting": 10,
    "harvest": 3
}
```

#### Adjust Disease Risk Thresholds (Notebook 03)
```python
# Change alert threshold
disease_risk_threshold = 65  # Default: 65/100 (lower = more sensitive)

# Modify risk weights
leaf_mold_weight = 0.4  # Contribution to composite risk
spider_mite_weight = 0.3
```

#### Tune Control Parameters (Notebook 05)
```python
# Change setpoints
temp_setpoint_vegetative = 22  # °C
humidity_setpoint_vegetative = 70  # %

# Modify control aggressiveness
proportional_gain_temp = 5.0  # Higher = more aggressive
deadband_temp = 1.0  # °C (tolerance around setpoint)

# Adjust resource weights
energy_cost_per_kwh = 0.12  # USD
water_cost_per_liter = 0.002  # USD
```

### Troubleshooting

#### Problem: Package installation fails (Notebook 01)
**Solution:**
```bash
# Upgrade uv
pip install --upgrade uv

# Retry installation
uv pip install numpy pandas matplotlib seaborn scikit-learn scipy statsmodels plotly ipywidgets
```

#### Problem: "File not found" error in Notebook 03+
**Solution:**
- Ensure you ran Notebook 02 completely
- Check that `data/greenhouse_data_5min.csv` exists
- Re-run Notebook 02 if necessary

#### Problem: Out of memory error
**Solution:**
- Reduce `num_days` in Notebook 02 (try 7 or 14 days)
- Close other applications
- Restart Jupyter kernel (Kernel → Restart)

#### Problem: Plots not displaying
**Solution:**
```python
# Add at top of notebook
%matplotlib inline

# Or use notebook backend
%matplotlib notebook
```

#### Problem: Slow execution
**Solution:**
- Use hourly data instead of 5-minute data for Notebooks 04-06
- Disable interactive plots (use static matplotlib only)
- Reduce number of RandomForest trees (change `n_estimators=100` to `50`)

---

## 📦 Expected Outputs

### After Running All Notebooks

#### Directory Structure
```
feature_demos/
├── data/                      (5 CSV files, ~10 MB total)
├── figures/                   (15 PNG files, ~8 MB total)
└── *.ipynb                    (6 notebooks with executed outputs)
```

#### File Sizes (Approximate)
- `greenhouse_data_5min.csv`: 2.5 MB
- `greenhouse_data_hourly.csv`: 50 KB
- `greenhouse_data_with_risk_and_stage.csv`: 3.2 MB
- `events_log.csv`: 5 KB
- `feature_comparison.csv`: 1 KB
- Each PNG figure: 200-800 KB

#### Total Storage: ~20 MB

### Key Results You Should See

#### 1. Synthetic Data Quality
- Realistic diurnal temperature cycles (day/night ~10°C difference)
- Inverse humidity-temperature relationship
- CO₂ enrichment periods during daylight
- Soil moisture oscillations around irrigation events

#### 2. Disease Risk Detection
- Clear risk peaks during high-humidity periods
- Different disease profiles (leaf mold vs spider mites)
- Risk budget accumulation visible
- Alert thresholds crossed ~3-5 times over 30 days

#### 3. Growth Stage Accuracy
- Stage transitions at expected day indices:
  - Vegetative: Days 0-7
  - Flowering: Days 8-17
  - Fruiting: Days 18-27
  - Harvest: Days 28-30
- Confidence >80% for most predictions

#### 4. Digital Twin Performance
- Temperature predictions: R² > 0.95
- Humidity predictions: R² > 0.92
- What-if scenarios show expected physical behavior (e.g., fan ON → temp drop)

#### 5. Control Effectiveness
- **Disease risk reduction:** ~40% lower with control
- **Energy savings:** ~15% lower consumption
- **Water savings:** ~20% lower usage
- **Climate stability:** 3x better (lower standard deviation)

#### 6. Dashboard Quality
- All plots display clearly
- Colors consistent (matplotlib defaults or custom palette)
- Axis labels and titles present
- Legends positioned correctly
- No overlapping text

---

## 🎓 Learning Outcomes

After completing these demonstrations, you will understand:

### Conceptual Understanding
- ✅ What a **digital twin** is and how it works
- ✅ How **disease risk** can be quantified from environmental data
- ✅ Why **growth stages** matter for crop management
- ✅ What **Model Predictive Control (MPC)** does
- ✅ How **machine learning** applies to agriculture
- ✅ The difference between **reactive** and **proactive** control

### Technical Skills
- ✅ Processing time-series sensor data with Pandas
- ✅ Training RandomForest classifiers with scikit-learn
- ✅ Calibrating grey-box models for simulation
- ✅ Implementing control algorithms in Python
- ✅ Creating publication-quality visualizations
- ✅ Designing human-machine interfaces for operators

### Domain Knowledge
- ✅ Greenhouse environmental requirements for tomatoes
- ✅ Common greenhouse diseases and risk factors
- ✅ Actuator types and control constraints
- ✅ Energy and water consumption patterns
- ✅ Industry-standard performance metrics

### System Design
- ✅ How to structure a cyber-physical system
- ✅ Data pipeline design for IoT applications
- ✅ Integration of sensing, modeling, and control
- ✅ Alert system design for non-technical users

---

## 🔗 Related Resources

### Project Documentation
- `/docs/Documents/` — Project reports and presentations
- `/docs/Base Research Papers/` — Foundational literature
- `/docs/General Research Papers/` — Relevant academic papers

### Key Papers Referenced
1. **Digital Twin Technology in Greenhouse** — Conceptual framework
2. **Integrating Digital Twins and MPC for Sustainable Greenhouse Management** — Control methodology
3. **A Digital Twin-Based Framework for Precision Tomato Cultivation** — Crop-specific implementation
4. **Advances in intelligent and autonomous greenhouse systems** — State-of-the-art review

### External Links
- [scikit-learn Documentation](https://scikit-learn.org/stable/) — Machine learning library
- [Pandas Documentation](https://pandas.pydata.org/docs/) — Data manipulation
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/) — Plotting examples
- [Model Predictive Control Basics](https://en.wikipedia.org/wiki/Model_predictive_control) — Theory

---

## 🏆 Advanced Usage

### For Researchers

#### Modify ML Models
Replace RandomForest with other algorithms:
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# In Notebook 03
model = GradientBoostingClassifier(n_estimators=100)
# or
model = SVC(kernel='rbf', probability=True)
```

#### Add New Diseases
Extend disease risk models:
```python
def calculate_botrytis_risk(temp, humidity, leaf_wetness):
    """Gray mold risk for grapes/strawberries"""
    risk = 0
    if humidity > 85: risk += 50
    if 15 <= temp <= 20: risk += 30
    if leaf_wetness > 0.7: risk += 20
    return min(risk, 100)
```

#### Implement Advanced Control
Replace MPC-like with true MPC using optimization:
```python
from scipy.optimize import minimize

def mpc_objective(u, x_current, setpoints, digital_twin):
    """Optimize over prediction horizon"""
    cost = 0
    x = x_current
    for t in range(horizon):
        x = digital_twin.predict(x, u[t])
        cost += (x['temp'] - setpoints['temp'])**2
        cost += (x['humidity'] - setpoints['humidity'])**2
        cost += 0.1 * u[t]['energy']  # Energy penalty
    return cost

optimal_u = minimize(mpc_objective, initial_guess, constraints=...)
```

### For Operators

#### Deploy to Real Greenhouse
1. **Replace synthetic data** with real sensor inputs:
   ```python
   # Instead of loading CSV
   sensor_data = read_from_real_sensors()  # MQTT, REST API, etc.
   ```

2. **Connect actuator outputs** to real hardware:
   ```python
   # Instead of simulating
   if heater_command == "ON":
       send_to_actuator("heater", "ON")  # GPIO, Modbus, etc.
   ```

3. **Run as continuous service:**
   ```bash
   # Convert notebook to Python script
   jupyter nbconvert --to script 05_control_policy*.ipynb
   
   # Run continuously
   while true; do python 05_control_policy.py; sleep 300; done
   ```

4. **Set up monitoring dashboard:**
   - Use Grafana + InfluxDB for time-series visualization
   - Stream data to cloud platform (AWS IoT, Azure IoT Hub)
   - Set up SMS/email alerts for critical events

---

## 🤝 Contributing

This demonstration is part of the **AgriTwin-GH** research project. If you have:
- **Bug reports** — Open an issue on GitHub
- **Feature requests** — Suggest enhancements
- **Questions** — Reach out to project maintainers
- **Improvements** — Submit pull requests

---

## 📄 License

Refer to the main repository for licensing information.

---

## 📧 Contact

For questions or collaboration inquiries, contact the AgriTwin-GH team through the GitHub repository:
**https://github.com/arjun-christopher/AgriTwin-GH**

---

## 📝 Citation

If you use this work in research, please cite:

```
AgriTwin-GH: A Digital Twin System for Smart Greenhouse Management
Arjun Christopher et al.
GitHub Repository: https://github.com/arjun-christopher/AgriTwin-GH
Year: 2026
```

---

## 🎉 Acknowledgments

This work builds upon:
- Open-source Python scientific computing ecosystem (NumPy, Pandas, scikit-learn)
- Research in digital twins for agriculture
- Model predictive control methodologies
- Machine learning for crop disease prediction

Special thanks to the greenhouse automation research community for advancing precision agriculture technologies.

---

## 📌 Quick Reference Card

### Notebook Sequence
```
01_Setup → 02_Data → 03_Risk → 04_Twin → 05_Control → 06_Dashboard
```

### Key Files Generated
```
data/greenhouse_data_5min.csv                    (Raw sensors)
data/greenhouse_data_with_risk_and_stage.csv     (ML-enhanced)
figures/fig_dashboard_snapshot.png                (Main dashboard)
figures/fig_disease_risk_index.png                (Risk trends)
figures/fig_control_vs_nocontrol_resources.png    (Savings proof)
```

### System Capabilities
- ✅ Temperature/Humidity/CO₂/Soil monitoring
- ✅ Disease risk indexing (leaf mold, spider mites, etc.)
- ✅ Growth stage detection (Vegetative/Flowering/Fruiting/Harvest)
- ✅ Digital twin simulation (grey-box model)
- ✅ MPC-like control (multi-objective optimization)
- ✅ Resource tracking (energy kWh, water liters)
- ✅ Non-verbal alerts (Green/Yellow/Red HMI)
- ✅ What-if scenario analysis

### Performance Targets
- Disease risk reduction: **~40%**
- Energy savings: **~15%**
- Water savings: **~20%**
- Digital twin accuracy: **R² > 0.90**
- Stage detection accuracy: **~95%**

---

**Version:** 1.0  
**Last Updated:** February 2026  

---

**End of Documentation**
