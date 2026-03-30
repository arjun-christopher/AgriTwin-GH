# AgriTwin-GH: Model Predictive Control (MPC) — Complete Guide

> **Who is this for?**  
> This guide is written so that anyone — from a curious beginner with no control theory background to an experienced ML engineer — can understand what the MPC module does, how every file fits together, how to run it, and how to extend it.

---

## Table of Contents

1. [What is MPC? (Plain English)](#1-what-is-mpc-plain-english)
2. [How AgriTwin-GH Uses MPC](#2-how-agritwin-gh-uses-mpc)
3. [System Architecture at a Glance](#3-system-architecture-at-a-glance)
4. [Full Data Flow Diagram](#4-full-data-flow-diagram)
5. [File Tree — All 26 Source Files](#5-file-tree--all-26-source-files)
6. [File-by-File Reference](#6-file-by-file-reference)
   - [constants.py](#61-constantspy)
   - [state.py](#62-statepy)
   - [setpoints.py](#63-setpointspy)
   - [constraints.py](#64-constraintspy)
   - [greenhouse_model.py](#65-greenhouse_modelpy)
   - [baseline_controller.py](#66-baseline_controllerpy)
   - [cost_function.py](#67-cost_functionpy)
   - [mpc_solver.py](#68-mpc_solverpy)
   - [disturbance.py](#69-disturbancepy)
   - [disease_penalty.py](#610-disease_penaltypy)
   - [growth_weights.py](#611-growth_weightspy)
   - [state_fusion.py](#612-state_fusionpy)
   - [mpc_input_preparation.py](#613-mpc_input_preparationpy)
   - [digital_twin_output.py](#614-digital_twin_outputpy)
   - [image_streamer.py](#615-image_streamerpy)
   - [evaluation.py](#616-evaluationpy)
   - [runner.py](#617-runnerpy)
   - [config.py](#618-configpy)
   - [experiment_runner.py](#619-experiment_runnerpy)
   - [weather_adaptation.py](#620-weather_adaptationpy)
   - [utils.py](#621-utilspy)
   - [evaluation_metrics.py](#622-evaluation_metricspy)
   - [__init__.py](#623-__init__py)
7. [Key Data Structures](#7-key-data-structures)
   - [FusedState](#71-fusedstate)
   - [MPCSolution](#72-mpcsolution)
   - [DigitalTwinStepPayload](#73-digitaltwinsteppayload)
   - [ComparisonMetrics](#74-comparisonmetrics)
8. [Canonical Labels Reference](#8-canonical-labels-reference)
9. [Configuration Guide](#9-configuration-guide)
10. [How to Run the MPC Module](#10-how-to-run-the-mpc-module)
11. [Test Scripts](#11-test-scripts)
12. [Artifact & Logging Strategy](#12-artifact--logging-strategy)
13. [Assumptions & Design Decisions](#13-assumptions--design-decisions)
14. [Extension Points](#14-extension-points)
15. [Phased Build Roadmap](#15-phased-build-roadmap)

---

## 1. What is MPC? (Plain English)

Imagine you are driving a car on a winding road. You constantly look ahead, predict where the road curves, and steer *now* to prepare for what is coming. You do not just react to the curve when it is already under you — you use foresight.

**Model Predictive Control (MPC)** works the same way for automated systems:

1. **The model**: A mathematical description of how the system (greenhouse) changes when you take an action (turn on heater, open vent).
2. **The prediction horizon**: A window of time into the future (e.g. 12 hours). The controller simulates what will happen over this window for different action sequences.
3. **The optimisation**: Find the sequence of actions that keeps the greenhouse closest to the desired targets (temperature, humidity, etc.) while using the least energy and keeping disease risk low.
4. **Receding horizon (the clever part)**: Only the *first* action from the best sequence is actually applied. At the very next timestep, the whole prediction+optimisation is repeated with fresh sensor data. This continuously corrects for model error and disturbances (like unexpected weather).

**Why not just a simple rule-based controller?**  
Rules like "if temp > 25 °C, turn on fan" cannot look ahead. They react after the problem has already happened. MPC anticipates problems and pre-emptively acts, resulting in less crop stress, lower energy waste, and more stable conditions — especially critical during flowering or when disease risk is elevated.

---

## 2. How AgriTwin-GH Uses MPC

AgriTwin-GH is a **digital twin for a smart greenhouse** growing tomatoes. It:

- Monitors the physical greenhouse via sensors stored in a **PostgreSQL database**.
- Has trained **AI models** for weather forecasting (24 h/48 h), disease progression (LSTM severity per disease), and growth stage classification/progression (LSTM).
- Retrieves **plant images** from **MinIO** object storage for disease/growth classification.
- Uses the **MPC module** (`src/agritwin_gh/mpc/`) to translate all that sensor and AI data into optimal actuator commands every **5 minutes**.
- Compares MPC performance against a **rule-based baseline controller** to demonstrate improvement.
- Outputs structured payloads to a **dashboard / digital twin UI**.

The MPC module has no GUI of its own — it is the *brain* that other parts of the system consume.

---

## 3. System Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        AGRITWIN-GH MPC MODULE                            │
│                  src/agritwin_gh/mpc/  (26 Python files)                 │
│                                                                          │
│   DATABASE             AI MODELS              CONTROL ENGINE             │
│   ─────────            ─────────              ──────────────             │
│   PostgreSQL ──► MPCInputPreparation ──► StateFusion.fuse()             │
│   (sensor data,                         (FusedState)                     │
│    disease data,                              │                          │
│    growth data,                               ▼                          │
│    weather data)                       MPCSolver.solve()                 │
│                                        (SLSQP optimiser)                 │
│   MinIO ──────► ImageStreamer ──────►       │                            │
│   (crop images)                             ▼                            │
│                                   DigitalTwinOutput.format_step()        │
│   AI Models ─► WeatherDisturbance  (DigitalTwinStepPayload)             │
│                DiseaseRiskPenalty              │                          │
│                GrowthStageWeights              ▼                          │
│                                        Dashboard / API                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Seven layers of operation (one 5-minute step):**

| Step | What happens |
|------|-------------|
| 1 | Query DB for latest greenhouse sensor readings and ML model predictions |
| 2 | Retrieve a plant image from MinIO; run disease & growth classifiers |
| 3 | Run weather forecast, disease progression, and growth stage models |
| 4 | `StateFusion.fuse()` assembles all data into a single `FusedState` |
| 5 | `MPCSolver.solve()` runs SLSQP optimisation over the next 12 hours |
| 6 | Extract `first_action` from the solution; apply it to the greenhouse model |
| 7 | Format into `DigitalTwinStepPayload`; log; yield to dashboard |

---

## 4. Full Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: EXTERNAL DATA                          │
│  ┌────────────────┐   ┌────────────────┐   ┌───────────────────────┐   │
│  │  PostgreSQL DB  │   │  MinIO images  │   │  Trained model files  │   │
│  │  (agritwin_db) │   │  (agritwin-    │   │  (data/processed/     │   │
│  │                │   │   images)      │   │   models/artifacts/)  │   │
│  └───────┬────────┘   └───────┬────────┘   └──────────┬────────────┘   │
└──────────┼────────────────────┼───────────────────────┼────────────────┘
           │                    │                        │
┌──────────┼────────────────────┼───────────────────────┼────────────────┐
│          │     LAYER 1: DATA ACQUISITION               │                │
│          ▼                    ▼                        ▼                │
│  mpc_input_preparation.py  image_streamer.py       (model loaders)      │
│  ─────────────────────────────────────────────────────────────────     │
│  • get_latest_greenhouse_row()    • get_random_disease_image()          │
│  • get_weather_context_df()       • get_random_growth_stage_image()     │
│  • get_disease_progression_df()                                         │
│  • get_growth_progression_df()                                          │
│  • get_recent_actuator_state()                                          │
└──────────┬────────────────────┬────────────────────────────────────────┘
           │                    │
┌──────────┼────────────────────┼────────────────────────────────────────┐
│          ▼     LAYER 2: AI MODEL WRAPPERS                               │
│  ┌───────────────────┐  ┌───────────────────┐  ┌─────────────────────┐ │
│  │ disturbance.py    │  │ disease_penalty.py │  │ growth_weights.py   │ │
│  │ ─────────────     │  │ ─────────────────  │  │ ───────────────     │ │
│  │ WeatherDisturbance│  │ DiseaseRiskPenalty │  │ GrowthStageWeights  │ │
│  │ Forecast          │  │ compute_risk_score │  │ get_weights()       │ │
│  │ get_forecast()    │  │ predict_severity   │  │ predict_transition()│ │
│  │  → 24h weather    │  │  → penalty float   │  │  → weight dict      │ │
│  │    per 5-min step │  │  → severity 24h/48h│  │  → stage transition │ │
│  └──────────┬────────┘  └─────────┬──────────┘  └──────────┬──────────┘ │
└─────────────┼──────────────────────┼─────────────────────────┼──────────┘
              │                      │                          │
┌─────────────┼──────────────────────┼─────────────────────────┼──────────┐
│             ▼   LAYER 3: STATE FUSION                                    │
│         ┌──────────────────────────────────────────────────────────┐    │
│         │                    state_fusion.py                        │    │
│         │                StateFusion.fuse(timestamp)                │    │
│         │                                                           │    │
│         │  Combines ALL of the above into a single FusedState:     │    │
│         │   • GreenhouseState (9 sensor variables)                  │    │
│         │   • WeatherForecast (24h disturbance sequence)            │    │
│         │   • GrowthStage + hours-to-transition                     │    │
│         │   • DiseaseClassification + severity_24h / _48h          │    │
│         │   • StageSetpoint (target values for current stage)       │    │
│         │   • ConstraintSet (per-stage tightened bounds)            │    │
│         │   • DiseaseRiskScore [0, 1]                               │    │
│         │   • ImagePayload (URL, MinIO key)                         │    │
│         └──────────────────────────┬────────────────────────────────┘   │
└────────────────────────────────────┼──────────────────────────────────┘
                                     │
┌────────────────────────────────────┼──────────────────────────────────┐
│            LAYER 4: MPC SOLVER      ▼                                   │
│   ┌─────────────────────────────────────────────────────────────┐      │
│   │                        mpc_solver.py                         │      │
│   │                   MPCSolver.solve(fused_state)               │      │
│   │                                                              │      │
│   │  Uses:                                                       │      │
│   │   • greenhouse_model.py  (step-forward simulation)          │      │
│   │   • cost_function.py     (stage-aware cost J(u))            │      │
│   │   • constraints.py       (scipy bounds + rate limits)        │      │
│   │   • weather_adaptation.py (weather-tightened constraints)   │      │
│   │                                                              │      │
│   │  Objective: min_u Σ ℓ(x_k, u_k, u_{k-1}) + V_f(x_N)      │      │
│   │  Solver: scipy SLSQP (falls back to RuleBasedController)    │      │
│   │  Output: MPCSolution (first_action + predicted trajectory)  │      │
│   └──────────────────────────┬────────────────────────────────┘      │
│                               │                                        │
│   ┌───────────────────────────┴────────────────────────────────┐      │
│   │                  baseline_controller.py                     │      │
│   │              RuleBasedController.compute_action()           │      │
│   │        (runs in parallel for comparison purposes)           │      │
│   └──────────────────────────┬──────────────────────────────────┘      │
└────────────────────────────────┼──────────────────────────────────────┘
                                 │
┌────────────────────────────────┼──────────────────────────────────────┐
│            LAYER 5: OUTPUT FORMATTING                                   │
│                                ▼                                        │
│   digital_twin_output.py → DigitalTwinStepPayload                      │
│                           → DigitalTwinTrajectoryPayload                │
│                                │                                        │
│   evaluation.py → ComparisonMetrics (MPC vs Baseline statistics)       │
└────────────────────────────────┬──────────────────────────────────────┘
                                 │
                                 ▼
                    Dashboard / Digital Twin UI
```

---

## 5. File Tree — All 26 Source Files

```
src/agritwin_gh/mpc/
│
├── __init__.py                  # Public API exports
│
│── Core data types ─────────────────────────────────────────────
├── constants.py                 # Enums, label maps, index dicts
├── state.py                     # GreenhouseState, ActuatorState, FusedState, payloads
├── setpoints.py                 # Stage-specific target setpoints
├── constraints.py               # Actuator + environment constraint sets
│
│── Physics model ───────────────────────────────────────────────
├── greenhouse_model.py          # Linear ARX greenhouse transition model
│
│── Controllers ─────────────────────────────────────────────────
├── baseline_controller.py       # Rule-based heuristic (for comparison)
├── cost_function.py             # Quadratic stage cost + terminal cost
├── mpc_solver.py                # Receding-horizon SLSQP optimiser
│
│── AI model wrappers ───────────────────────────────────────────
├── disturbance.py               # Weather forecast → disturbance sequence
├── disease_penalty.py           # Disease risk score + LSTM severity penalty
├── growth_weights.py            # Growth-stage cost weights + LSTM transition
├── weather_adaptation.py        # Weather-stress constraint tightening
│
│── Data pipeline ───────────────────────────────────────────────
├── mpc_input_preparation.py     # All PostgreSQL queries for MPC inputs
├── image_streamer.py            # MinIO image metadata retrieval (5-min TTL cache)
├── state_fusion.py              # Assembles FusedState from all sources
│
│── Orchestration ───────────────────────────────────────────────
├── runner.py                    # MPCRunner — top-level control loop
├── experiment_runner.py         # Batch simulation + save + compare
│
│── Output & evaluation ─────────────────────────────────────────
├── digital_twin_output.py       # Formats step + trajectory payloads
├── evaluation.py                # MPC vs Baseline comparisons + plotting
├── evaluation_metrics.py        # Metric computation (tracking error, energy, etc.)
│
│── Configuration ───────────────────────────────────────────────
├── config.py                    # MPCConfig dataclass + YAML loader
│
│── Shared utilities ────────────────────────────────────────────
└── utils.py                     # discover_latest_artifact() helper
```

---

## 6. File-by-File Reference

### 6.1 `constants.py`

**Purpose**: Single source of truth for all symbolic constants used across the MPC module. No classes — just module-level definitions.

**What it defines:**

| Constant | Type | Description |
|----------|------|-------------|
| `DT_MINUTES` | `int` | Control timestep: **5 minutes** |
| `STATE_VARIABLES` | `tuple[str, ...]` | 9 greenhouse state variable names |
| `CONTROL_VARIABLES` | `tuple[str, ...]` | 7 actuator names |
| `GROWTH_STAGES` | `tuple[str, ...]` | 6 canonical growth stage strings |
| `DISEASE_CATEGORIES` | `tuple[str, ...]` | 6 canonical disease/health strings |
| `GROWTH_STAGE_DB_MAP` | `dict` | DB integer code → stage string |
| `GROWTH_STAGE_IMAGE_SUBCATEGORY` | `dict` | Stage string → MinIO image subfolder |
| `IMAGE_SUBCATEGORY_MAP` | `dict` | Disease string → MinIO image subfolder |
| `STAGE_INDEX` | `dict` | Stage string → integer index (0–5) |
| `DISEASE_INDEX` | `dict` | Disease string → integer index (0–5) |

**State variables (9):**

```
indoor_temp, indoor_humidity, co2_level, soil_moisture,
light_intensity, outdoor_temp, outdoor_humidity, vpd, leaf_wetness
```

**Control (actuator) variables (7):**

```
fan_speed, vent_opening, irrigation_qty, heater_output,
led_intensity, co2_valve_pct, fogger_duty
```

**Inputs**: Nothing (pure definitions).  
**Outputs**: Constants used by every other MPC file via `from .constants import ...`.

---

### 6.2 `state.py`

**Purpose**: All dataclasses for the MPC state machine. This is the shared language between every MPC component.

**Classes:**

#### `GreenhouseState`
Represents the physical greenhouse at a single point in time.

```python
@dataclass
class GreenhouseState:
    indoor_temp: float        # °C
    indoor_humidity: float    # % RH
    co2_level: float          # ppm
    soil_moisture: float      # % volumetric
    light_intensity: float    # μmol/m²/s
    outdoor_temp: float       # °C
    outdoor_humidity: float   # % RH
    vpd: float                # kPa (vapour pressure deficit)
    leaf_wetness: float       # 0–1 proxy
    timestamp: datetime | None = None
```

Key methods:
- `to_numpy()` → `ndarray(9,)` — for feeding into the physics model
- `from_numpy(arr)` → `GreenhouseState` — inverse
- `from_db_row(row: dict)` → `GreenhouseState` — construct from DB record
- `to_dict()` → `dict` — JSON-serialisable

#### `ActuatorState`
Represents all 7 actuator settings.

```python
@dataclass
class ActuatorState:
    fan_speed: float         # 0–1 normalised fraction
    vent_opening: float      # 0–1 fraction
    irrigation_qty: float    # litres per step
    heater_output: float     # 0–1 fraction
    led_intensity: float     # 0–1 fraction  
    co2_valve_pct: float     # 0–1 fraction
    fogger_duty: float       # 0–1 duty cycle
```

Key methods:
- `to_numpy()` → `ndarray(7,)`
- `clip(constraints)` → clips values to constraint bounds
- `from_numpy(arr)` → `ActuatorState`

#### `WeatherState`
Single-timestep weather snapshot used inside the disturbance forecast sequence.

#### `MPCState`
Composite of `GreenhouseState` + `ActuatorState` + optional metadata.

#### `FusedState`
The rich state assembled by `StateFusion` — see [Section 7.1](#71-fusedstate).

#### `ControllerDecisionContext`
Structured traceability record capturing *why* a particular action was taken (solver config, weights, disease context, weather stress).

#### `DigitalTwinStepPayload` / `DigitalTwinTrajectoryPayload`
Formatted outputs — see [Section 7.3](#73-digitaltwinsteppayload).

**Inputs**: Raw floats or DB rows.  
**Outputs**: Typed dataclass instances used everywhere else.

---

### 6.3 `setpoints.py`

**Purpose**: Defines the *target* (setpoint) values for every growth stage. The MPC cost function penalises deviations from these targets.

**Key class:**

```python
@dataclass
class StageSetpoint:
    temp: float             # °C target
    temp_tol: float         # ± tolerance (deadband)
    humidity: float         # % RH target
    hum_tol: float          # ± tolerance
    soil_moisture: float    # % target
    co2: float              # ppm target
    light: float            # μmol/m²/s target
    vpd: float              # kPa target
    disease_risk_max: float # risk threshold [0,1]
```

**Key function:**

```python
def get_setpoint(stage: str) -> StageSetpoint:
    """Return the StageSetpoint for a canonical growth stage name."""
```

Grows from `config/mpc_config.yaml` (the `setpoints:` section). Raises `ValueError` if stage is unrecognised.

**Why this matters**: Without stage-aware setpoints, the MPC would try to keep the greenhouse at the same conditions regardless of whether the plant is a seedling or in full flower. Tomatoes need warmer, more humid conditions when young and progressively cooler, drier conditions at flowering and fruit development.

**Inputs**: Stage name string (one of the 6 canonical stages).  
**Outputs**: `StageSetpoint` dataclass.

---

### 6.4 `constraints.py`

**Purpose**: All operational limits that the MPC must respect. Separates *hard* box bounds (actuator physical limits) from *soft* environmental safety ranges.

**Key class:**

```python
@dataclass
class ConstraintSet:
    # Actuator box bounds (hard)
    actuator_bounds: dict[str, tuple[float, float]]
    
    # Rate-of-change limits (prevent actuator shock)
    actuator_rate_limits: dict[str, float]
    
    # Minimum steps between changes (cooldown)
    actuator_cooldown_steps: dict[str, int]
    
    # Environmental safe ranges (soft/penalty)
    env_bounds: dict[str, tuple[float, float]]
    
    # Resource budgets
    daily_water_budget_litres: float
    daily_energy_budget_kwh: float
```

Key methods:
- `to_bounds()` → list of `(lo, hi)` tuples in scipy format
- `to_scipy()` → list of scipy constraint dicts for rate limits

**Key functions:**
- `get_default_constraints(stage: str) -> ConstraintSet` — stage-aware defaults
- `tighten_constraints_for_disease(cs, risk_score)` — narrows humidity bounds when disease risk is high

**Why rate limits matter**: Without rate limits, the optimiser might oscillate — e.g. toggling the heater on/off every 5 minutes to minimise cost. Rate limits ensure smooth, realistic actuator behaviour.

**Inputs**: Growth stage string, optional disease risk score.  
**Outputs**: `ConstraintSet` instance ready for scipy.

---

### 6.5 `greenhouse_model.py`

**Purpose**: The mathematical model of how the greenhouse responds to actuator actions and external weather. This is the model in "Model Predictive Control."

**Class: `GreenhouseTransitionModel`**

What it models (physics sub-models):

| Variable | Model type | Key effects |
|----------|-----------|------------|
| `indoor_temp` | ARX (Auto-Regressive with eXogenous inputs) | heater_output (+), fan_speed (−), vent_opening (−), outdoor_temp |
| `indoor_humidity` | ARX | fogger_duty (+), vent_opening (−), outdoor_humidity |
| `co2_level` | ARX | co2_valve_pct (+), fan/vent (−) |
| `soil_moisture` | Water balance | irrigation_qty (+), evaporation (−) |
| `light_intensity` | Direct sum | led_intensity + solar contribution via vent/outdoor |
| `vpd` | Tetens equation | derived from temp + humidity |
| `leaf_wetness` | Proxy | fogger, humidity, time-of-day |

**Named bounds constants** (guard-rails for physical plausibility):

```python
TEMP_BOUNDS  = (0.0, 50.0)    # °C
HUM_BOUNDS   = (0.0, 100.0)   # %
SM_BOUNDS    = (0.0, 100.0)   # %
CO2_BOUNDS   = (200.0, 5000.0) # ppm
LIGHT_BOUNDS = (0.0, 2000.0)  # μmol/m²/s
```

**Key methods:**

```python
def step(
    state: GreenhouseState,
    actuators: ActuatorState,
    disturbance: dict | None = None,  # weather at this timestep
) -> GreenhouseState:
    """Advance state by one DT_MINUTES step."""

def simulate(
    initial_state: GreenhouseState,
    actuator_sequence: list[ActuatorState],
    disturbance_sequence: list[dict] | None = None,
) -> list[GreenhouseState]:
    """Simulate over an actuator sequence (used by solver)."""

def calibrate(
    historical_data: pd.DataFrame,
) -> dict:
    """Fit ARX coefficients to real greenhouse data using Ridge regression."""
```

**Why ARX?**: ARX models are linear, which means the MPC optimisation is well-posed for SLSQP (Sequential Least Squares Programming). They are fast to evaluate (microseconds per step), enabling a 144-step 12-hour horizon to be optimised in milliseconds. A neural network plant model would be more accurate but orders of magnitude slower.

**Inputs**: `GreenhouseState` + `ActuatorState` + optional weather disturbance dict.  
**Outputs**: Next `GreenhouseState`.

---

### 6.6 `baseline_controller.py`

**Purpose**: A deterministic rule-based controller that mimics what a manual/basic automated system would do. It serves as the performance *baseline* — the MPC is evaluated by how much better it does compared to this.

**Class: `RuleBasedController`**

**Rule priority (highest → lowest):**

1. **Disease risk** — if `risk_score > 0.6`: emergency ventilation (max fan + vent), reduce fogger
2. **Temperature** — if too hot: fan + vent ON; if too cold: heater ON
3. **Humidity** — if too high: vent + fan; if too low: fogger ON
4. **Soil moisture** — if too dry: irrigate
5. **CO₂** — if too low: co2 valve ON
6. **Light** — if too dark (daytime only): LED ON

Energy cost coefficients (kWh per fractional unit):

```python
ENERGY_COST_COEFFICIENTS = {
    "fan_speed":      0.5,
    "vent_opening":   0.0,  # passive, no energy
    "irrigation_qty": 0.02, # per litre
    "heater_output":  2.0,
    "led_intensity":  0.3,
    "co2_valve_pct":  0.01,
    "fogger_duty":    0.15,
}
```

**Key method:**

```python
def compute_action(
    state: GreenhouseState,
    growth_stage: str,
    disease_risk_score: float,
    weather_forecast: list[dict] | None = None,
) -> ActuatorState:
    """Apply rule hierarchy and return deterministic actuator settings."""
```

**Output dataclass:**

```python
@dataclass
class BaselineControlPayload:
    actuators: ActuatorState
    triggered_rules: list[str]   # which rules fired
    energy_kwh: float
    water_litres: float
```

**Why keep a baseline?**: Real-world value of MPC is only demonstrated by comparison. If MPC uses 10% less energy with 20% better disease risk suppression than the baseline, that is the measurable ROI. The `evaluation.py` module computes exactly this.

**Inputs**: `GreenhouseState`, `growth_stage`, `disease_risk_score`.  
**Outputs**: `ActuatorState` + triggered rule names.

---

### 6.7 `cost_function.py`

**Purpose**: Defines the objective function J(u) that the MPC minimises. "Cost" = how bad the current state + actions are. Lower cost = closer to targets, lower energy use.

**Cost terms (stage-aware):**

| Term | Formula | Penalises |
|------|---------|----------|
| Temperature tracking | `w_T · (T - T_sp)²` | Deviation from setpoint |
| Humidity tracking | `w_H · (H - H_sp)²` | Deviation from setpoint |
| Soil moisture | `w_S · (S - S_sp)²` | Deviation from setpoint |
| CO₂ | `w_CO2 · (CO2 - CO2_sp)²` | Deviation from setpoint |
| VPD | `w_vpd · (vpd - vpd_sp)²` | Vapour pressure stress |
| Light | `w_L · (light - light_sp)²` | Light deficit or excess |
| Disease risk | `w_D · disease_penalty` | Elevated disease risk |
| Energy | `w_E · Σ E_i · u_i` | Energy consumption |
| Water | `w_W · irrigation_qty` | Water consumption |
| Actuator switching | `w_sw · Σ \|Δu_i\|` | Rapid actuator changes |

**Key classes:**

```python
class StageCost:
    def evaluate(
        state: GreenhouseState,
        setpoint: StageSetpoint,
        actuators: ActuatorState,
        prev_actuators: ActuatorState | None,
        disease_penalty: float = 0.0,
    ) -> float:
        """Compute single-step running cost."""

class TerminalCost:
    def evaluate(
        final_state: GreenhouseState,
        setpoint: StageSetpoint,
    ) -> float:
        """Terminal penalty at end of horizon (discourages drifting)."""

class CostBuilder:
    def build(
        weights: dict[str, float],
        disease_penalty_fn: Callable | None = None,
    ) -> Callable:
        """Factory: returns a callable cost function bound to given weights."""
```

**Stage weight multipliers** (from config, applied on top of base weights):

| Weight | Seedling | Veg | Flower Init | Flowering | Unripe | Ripe |
|--------|---------|-----|------------|-----------|--------|------|
| temperature | 1.2× | 1.1× | 1.4× | 1.5× | 1.2× | 0.9× |
| humidity | 1.5× | 1.2× | 1.2× | 1.5× | 1.2× | 0.9× |
| disease_risk | 0.8× | 1.0× | 1.3× | 1.5× | 1.5× | 1.2× |

**Inputs**: State, setpoint, actuators, weights.  
**Outputs**: A single float cost value (lower = better).

---

### 6.8 `mpc_solver.py`

**Purpose**: The MPC engine. Takes `FusedState` and returns the optimal actuator sequence by minimising the cost function subject to constraints.

**Mathematical structure:**

```
Decision variable:
    u = [u_0, u_1, …, u_{N-1}]   ∈  R^{7 × N}
    (flattened: 7 actuators × N control horizon steps)

Objective:
    min_u  J(u) = Σ_{k=0}^{N-1}  ℓ(x_k, u_k, u_{k-1})  +  V_f(x_N)
    where  x_{k+1} = f(x_k, u_k, d_k)    [greenhouse model]
           d_k                              [weather disturbance at step k]

Constraints:
    u_lo ≤ u_k ≤ u_hi            (actuator physical bounds)
    |u_k − u_{k-1}| ≤ Δu_max     (rate-of-change limits)
    irrigation_qty ≥ 0            (included in bounds)
```

**Class: `MPCSolver`**

```python
def solve(
    fused: FusedState,
    previous_control: ActuatorState | None = None,
) -> MPCSolution:
    """Run SLSQP optimisation. Falls back to RuleBasedController if it fails."""
```

**Fallback mechanism:** If SLSQP does not converge (status ≠ 0), the solver automatically calls `RuleBasedController.compute_action()` and marks `solution.fallback_used = True`. This means the system *never* returns no answer — it always produces safe actuator commands.

**Warm start:** The previous solution's tail is used as the initial guess for the next step. This dramatically speeds up convergence in practice.

**Single-shooting formulation:** The entire future trajectory `x_1, x_2, ..., x_N` is computed by rolling forward `greenhouse_model.step()` inside the objective function. The decision variable is only the actuator sequence.

**Output: `MPCSolution`**

```python
@dataclass
class MPCSolution:
    first_action: ActuatorState         # Apply this actuator command NOW
    predicted_states: list[GreenhouseState] # Predicted trajectory (for display)
    optimal_actuator_sequence: ndarray  # Full u* (shape: N×7)
    total_cost: float                   # Achieved objective value
    cost_breakdown: dict[str, float]    # Per-term cost contributions
    converged: bool                     # Whether SLSQP succeeded
    fallback_used: bool                 # Whether rule fallback was used
    solve_time_ms: float                # Wall-clock solve time
    n_iterations: int                   # SLSQP iterations
    n_function_evals: int               # Function evaluations
    solver_status: int                  # scipy result status
    last_weather_stress_summary: dict   # Weather adaptation applied
    last_constraint_tightening: dict    # Constraint tightening applied
```

**Inputs**: `FusedState`, optional previous `ActuatorState`.  
**Outputs**: `MPCSolution`.

---

### 6.9 `disturbance.py`

**Purpose**: Wraps the trained weather forecast model and converts its 24h/48h predictions into a per-step disturbance sequence for the MPC solver.

**Class: `WeatherDisturbanceForecast`**

```python
def get_forecast(
    context_df: pd.DataFrame,
    horizon_steps: int,          # number of 5-min steps to generate
) -> list[dict]:
    """Return a list of weather dicts, one per control step.
    
    Each dict contains: {outdoor_temp, outdoor_humidity, solar_irradiance}
    
    Implementation:
    - Calls EnvironmentForecastModel.predict() for 24h and 48h forecasts
    - Linearly interpolates between 24h and 48h breakpoints
    - Subdivides to 5-minute resolution via uniform interpolation
    """
```

**Auto-discovery:** The model artifact is located by `discover_latest_artifact("environment_forecast")` from `utils.py`. Override by passing an explicit `run_id`.

**Inputs**: Recent weather DataFrame from DB, horizon step count.  
**Outputs**: `list[dict]` — one weather snapshot per MPC step.

---

### 6.10 `disease_penalty.py`

**Purpose**: Quantifies how bad the disease situation is today and how much worse it is projected to get. Feeds directly into the MPC cost function.

**Class: `DiseaseRiskPenalty`**

```python
def compute_risk_score(
    state: GreenhouseState,
    disease_classification: str,
    severity_current: float,
) -> float:
    """Sigmoid rule-based risk score in [0, 1].
    
    high humidity + elevated severity + disease label → risk near 1.0
    healthy + low severity + low humidity → risk near 0.0
    """

def predict_severity_24h(
    context_df: pd.DataFrame,
) -> dict[str, float]:
    """LSTM severity forecast at 24h per disease category."""

def predict_severity_48h(
    context_df: pd.DataFrame,
) -> dict[str, float]:
    """LSTM severity forecast at 48h per disease category."""

def compute_penalty(
    risk_score: float,
    severity_24h: dict,
    severity_48h: dict,
    current_stage: str,
) -> float:
    """Combine current risk + future projections into single cost penalty."""
```

Why project 24h and 48h ahead? The MPC horizon is 12 hours. But disease development is a *slow process* — early blight might look mild now but be severe tomorrow. Including the 24h/48h projections in the cost penalises conditions that are likely to lead to disease escalation even if the current reading is safe.

**Inputs**: Greenhouse state, disease label, severity value, context DataFrame.  
**Outputs**: Risk score in [0, 1]; penalty float.

---

### 6.11 `growth_weights.py`

**Purpose**: Makes the MPC *stage-aware*. Provides different cost weights for different growth stages and predicts how many hours until the plant transitions to the next stage.

**Class: `GrowthStageWeights`**

```python
def get_weights(
    stage: str,
    base_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    """Return cost weight vector for the given growth stage.
    
    Applies stage_weight_multipliers from config on top of base weights.
    E.g. at flowering, temperature weight is 1.5× and disease_risk is 1.5×.
    """

def predict_transition(
    context_df: pd.DataFrame,
    current_stage: str,
) -> dict:
    """Use GrowthProgressionModel LSTM to predict:
    
    Returns:
        {
            "next_stage": str,
            "hours_to_transition": float,
            "confidence": float
        }
    """
```

**Stage weight design rationale:**

| Stage | Key priority | Reason |
|-------|------------|--------|
| Seedling | Humidity, soil moisture | Young roots vulnerable to drying |
| Early Vegetative | CO₂, light | Rapid leaf area expansion phase |
| Flowering Initiation | Temperature, CO₂, disease | Temperature critical for pollen viability |
| Flowering | Temperature, humidity, disease | Peak vulnerability; fruit set determines yield |
| Unripe | Disease risk, soil moisture | Fruit development needs stable water |
| Ripe | Minimal intervention | Plant nearing end of cycle |

**Inputs**: Growth stage string, optional DataFrame with recent growth data.  
**Outputs**: Dict of cost weights; dict with transition prediction.

---

### 6.12 `state_fusion.py`

**Purpose**: The "brain assembler." Takes all independent data sources (DB, MinIO, AI models) and fuses them into a single `FusedState` ready for the MPC solver.

**Class: `StateFusion`**

Constructor dependencies (injected by `MPCRunner`):
```
config, input_prep, weather, disease_penalty,
growth_weights, image_streamer,
disease_classifier (optional callable),
growth_classifier (optional callable)
```

**Key method:**

```python
def fuse(timestamp: datetime) -> FusedState:
    """Full pipeline:
    
    1. Query latest greenhouse sensor row → GreenhouseState
    2. Query recent actuator state → ActuatorState
    3. Retrieve plant image → ImagePayload (cached 5 min)
    4. Run disease classifier on image → disease_classification
    5. Run growth classifier on image → growth_stage
    6. Query disease progression DataFrame → severity context
    7. predict_severity_24h / _48h → disease projections
    8. compute_risk_score → disease_risk_score
    9. Query growth progression DataFrame
    10. predict_transition → next_stage, hours_to_transition
    11. get_forecast → weather disturbance sequence
    12. get_setpoint(growth_stage) → StageSetpoint
    13. get_default_constraints(growth_stage) → ConstraintSet
    14. get_weights(growth_stage) → cost_weights
    15. Assemble and return FusedState
    """
```

`_GROWTH_CLASSIFIER_LABEL_MAP` is derived from `GROWTH_STAGE_IMAGE_SUBCATEGORY` (no hardcoded strings).

**Inputs**: A `datetime` timestamp (used to query DB at that point in time).  
**Outputs**: `FusedState` — see [Section 7.1](#71-fusedstate).

---

### 6.13 `mpc_input_preparation.py`

**Purpose**: All database queries the MPC needs. Isolates SQL/SQLAlchemy logic from business logic.

**Class: `MPCInputPreparation`**

Key methods:

```python
def get_latest_greenhouse_row(
    timestamp: datetime | None = None,
) -> dict:
    """Query latest (or at-timestamp) row from greenhouse_data table."""

def get_weather_context_df(
    lookback_hours: int = 48,
) -> pd.DataFrame:
    """Fetch recent weather records for forecast model context window."""

def get_disease_progression_df(
    cycle_id: int | None = None,
    lookback_hours: int = 72,
) -> pd.DataFrame:
    """Fetch disease progression records for LSTM context."""

def get_growth_progression_df(
    cycle_id: int | None = None,
    lookback_hours: int = 168,  # 1 week
) -> pd.DataFrame:
    """Fetch growth stage progression records for LSTM context."""

def get_recent_actuator_state(
    timestamp: datetime | None = None,
) -> dict | None:
    """Fetch the most recent actuator command applied."""

def get_latest_cycle_id(self) -> int:
    """Return MAX(cycle_id) from crop_cycles table."""
```

All methods use the injected SQLAlchemy `Session` — no raw SQL strings. Uses parameterised queries, preventing SQL injection.

**Inputs**: SQLAlchemy Session (injected), optional timestamps.  
**Outputs**: `pd.DataFrame` or `dict` records.

---

### 6.14 `digital_twin_output.py`

**Purpose**: Formats the raw MPC solution into structured payloads that the dashboard and API can consume directly. Handles step-by-step and full trajectory formatting.

**Class: `DigitalTwinOutput`**

Key methods:

```python
def format_step(
    fused: FusedState,
    actuators: ActuatorState,
    predicted_next: GreenhouseState | None,
    step_cost: float,
    energy_kwh: float,
    water_litres: float,
    cost_breakdown: dict,
    solver_converged: bool,
    weather_stress: dict | None,
    tightened_constraints: dict | None,
    decision_context: ControllerDecisionContext,
    solver_performance: dict,
) -> DigitalTwinStepPayload:
    """Build one-step dashboard payload."""

def format_trajectory(
    steps: list[DigitalTwinStepPayload],
) -> DigitalTwinTrajectoryPayload:
    """Aggregate list of step payloads into a trajectory summary."""
```

**Inputs**: MPC solution components.  
**Outputs**: `DigitalTwinStepPayload` / `DigitalTwinTrajectoryPayload` — see [Section 7.3](#73-digitaltwinsteppayload).

---

### 6.15 `image_streamer.py`

**Purpose**: Retrieves plant image metadata from MinIO for the disease/growth classifiers. Has a **5-minute TTL cache** so the MPC loop does not hammer MinIO storage on every 5-minute step.

**Class: `ImageStreamer`**

```python
def get_random_disease_image(
    disease_label: str,
) -> ImagePayload:
    """Return metadata for a random image matching the disease label.
    
    Uses IMAGE_SUBCATEGORY_MAP to find the MinIO subfolder.
    Results are cached for 5 minutes (_TTL = 300 seconds).
    """

def get_random_growth_stage_image(
    stage_label: str,
) -> ImagePayload:
    """Return metadata for a random image matching the growth stage.
    
    Uses GROWTH_STAGE_IMAGE_SUBCATEGORY to find the MinIO subfolder.
    """
```

**`ImagePayload`** contains:
```python
@dataclass
class ImagePayload:
    minio_key: str         # Object key in MinIO bucket
    bucket: str            # Bucket name ("agritwin-images")
    disease_label: str     # Canonical disease string
    growth_stage: str      # Canonical stage string
    timestamp: datetime    # When image was taken
    presigned_url: str | None = None  # Optional pre-signed URL for direct browser access
```

**Inputs**: Disease label or growth stage label string.  
**Outputs**: `ImagePayload`.

---

### 6.16 `evaluation.py`

**Purpose**: Compares MPC performance against the baseline controller across a full simulation run. Computes metrics and generates plots.

**Class: `BaselineVsMPCEvaluator`**

```python
def compute_all(
    mpc_trajectory: DigitalTwinTrajectoryPayload,
    baseline_trajectory: DigitalTwinTrajectoryPayload,
) -> ComparisonMetrics:
    """Compute all comparison metrics."""

def plot_comparison(
    mpc_trajectory: DigitalTwinTrajectoryPayload,
    baseline_trajectory: DigitalTwinTrajectoryPayload,
    output_dir: Path,
) -> None:
    """Save comparison plots as PNG files."""
```

**Metrics computed** (see [Section 7.4](#74-comparisonmetrics)):
- Temperature RMSE
- Humidity RMSE  
- Disease risk (mean and time-above-threshold)
- Energy consumption (kWh)
- Water usage (litres)
- Setpoint tracking score
- Constraint violation count

**Inputs**: Two `DigitalTwinTrajectoryPayload` objects (MPC vs. Baseline).  
**Outputs**: `ComparisonMetrics`; optional plots in `data/processed/mpc_results/<run_id>/figures/`.

---

### 6.17 `runner.py`

**Purpose**: The top-level orchestrator. One `MPCRunner` instance drives the entire real-time (or replay) MPC loop.

**Class: `MPCRunner`**

Constructor:
```python
MPCRunner(
    session: Session,           # Live SQLAlchemy Session
    config: MPCConfig | None,   # If None, loads from YAML
    disease_classifier: Any,    # Optional: callable wrapping predict_image()
    growth_classifier: Any,     # Optional: callable wrapping predict_growth_stage()
    device: str = "cpu",        # PyTorch device for weather model
)
```

On construction, `MPCRunner.__init__` instantiates:
- `MPCInputPreparation`
- `ImageStreamer`
- `WeatherDisturbanceForecast`
- `DiseaseRiskPenalty`
- `GrowthStageWeights`
- `StateFusion`
- `DigitalTwinOutput`
- `GreenhouseTransitionModel`
- `MPCSolver`

**Key methods:**

```python
def run_single_step(
    timestamp: datetime | None = None,
) -> DigitalTwinStepPayload:
    """Execute one full MPC control step (5 minutes of real greenhouse time).
    
    Steps:
    1. StateFusion.fuse() → FusedState
    2. GrowthStageWeights.get_weights() → adaptive cost weights
    3. MPCSolver.solve() → MPCSolution (first_action + trajectory)
    4. _estimate_energy() → energy_kwh
    5. Build ControllerDecisionContext (traceability)
    6. DigitalTwinOutput.format_step() → DigitalTwinStepPayload
    7. Log step summary
    """

def run_simulation(
    start_time: datetime,
    end_time: datetime,
) -> DigitalTwinTrajectoryPayload:
    """Loop run_single_step() from start_time to end_time in DT_MINUTES steps."""

def run_simulation_iter(
    start_time: datetime,
    end_time: datetime,
) -> Generator[DigitalTwinStepPayload, None, None]:
    """Streaming variant — yields one payload per step. For live dashboards."""
```

**Inputs**: Live DB session, timestamps.  
**Outputs**: `DigitalTwinStepPayload` (single step) or `DigitalTwinTrajectoryPayload` (full simulation).

---

### 6.18 `config.py`

**Purpose**: Loads and validates the MPC configuration from `config/mpc_config.yaml`. Exposes all settings as a single typed dataclass.

**Class: `MPCConfig`**

Selected fields:

```python
@dataclass
class MPCConfig:
    # Timing
    dt_minutes: int = 5
    prediction_horizon_hours: float = 12.0
    control_horizon_hours: float = 6.0

    # Solver
    solver_method: str = "SLSQP"
    solver_max_iter: int = 200
    solver_ftol: float = 1e-6

    # Cost weights (base, overridden per stage)
    cost_weight_vector: dict[str, float] = field(default_factory=lambda: {
        "temperature": 1.0, "humidity": 1.0, "soil_moisture": 0.8,
        "co2": 0.5, "vpd": 0.6, "light": 0.4,
        "disease_risk": 2.0, "energy": 0.3, "water": 0.3,
        "actuator_switching": 0.1,
    })

    # Per-stage weight multipliers (dict of dicts)
    stage_weight_multipliers: dict[str, dict[str, float]] = field(...)

    # Resource budgets
    daily_water_budget_litres: float = 500.0
    daily_energy_budget_kwh: float = 100.0

    # Model artifact IDs (auto-discovered if None)
    environment_forecast_run_id: str | None = None
    disease_progression_run_id: str | None = None
    growth_progression_run_id: str | None = None
    disease_classifier_run_id: str | None = None
    growth_classifier_run_id: str | None = None

    # Run identity
    run_id: str = field(default_factory=lambda: ...)  # mpc_YYYYMMDD_HHMMSS
```

**Key function:**

```python
def load_mpc_config(path: str | Path | None = None) -> MPCConfig:
    """Load from config/mpc_config.yaml (or given path). Returns MPCConfig."""
```

**Inputs**: Path to YAML or nothing (uses default).  
**Outputs**: `MPCConfig` instance.

---

### 6.19 `experiment_runner.py`

**Purpose**: Runs a full paired experiment: MPC simulation + Baseline simulation over the same time window, saves all artefacts, and returns comparison metrics.

**Key function:**

```python
def run_experiment(
    session: Session,
    start_time: datetime,
    end_time: datetime,
    config: MPCConfig | None = None,
    output_dir: Path | None = None,
) -> ComparisonMetrics:
    """
    1. Run MPCRunner.run_simulation() → mpc_trajectory
    2. Run BaselineRunner.run_simulation() → baseline_trajectory
    3. BaselineVsMPCEvaluator.compute_all() → metrics
    4. Save trajectory Parquet files, metrics JSON, config snapshot
    5. BaselineVsMPCEvaluator.plot_comparison() → PNG figures
    6. Return ComparisonMetrics
    """
```

Uses canonical disease labels (`"healthy leaves"`, `"early blight"`) and canonical stage indexing (`stage_label_to_index("flowering")`) — no hardcoded integers.

**Output files saved to:** `data/processed/mpc_results/<run_id>/`

**Inputs**: DB session, time range, optional config.  
**Outputs**: `ComparisonMetrics` + files on disk.

---

### 6.20 `weather_adaptation.py`

**Purpose**: Dynamically adjusts MPC constraints based on weather stress signals — tightening ventilation and humidity bounds when extreme weather is forecast.

**Key classes and functions:**

```python
@dataclass
class WeatherAdaptiveModifiers:
    humidity_upper_tighten: float = 0.0   # reduce upper humidity bound by this
    temp_lower_tighten: float = 0.0       # raise lower temp bound by this
    vent_force_min: float = 0.0           # force minimum vent opening

def compute_weather_adaptation(
    weather_forecast: list[dict],
    current_constraints: ConstraintSet,
) -> tuple[ConstraintSet, WeatherAdaptiveModifiers]:
    """Return tightened constraints + modifiers applied."""
```

**Example:** If the 6-hour forecast shows outdoor humidity > 90% (high mould risk), the indoor humidity upper bound is reduced by 5%, and a minimum vent opening is enforced to increase air exchange.

**Inputs**: Weather disturbance sequence, current `ConstraintSet`.  
**Outputs**: Tightened `ConstraintSet` + `WeatherAdaptiveModifiers` (for logging).

---

### 6.21 `utils.py`

**Purpose**: Shared utility used by multiple model-loading files. Prevents code duplication.

**Key function:**

```python
def discover_latest_artifact(objective: str) -> str | None:
    """Scan data/processed/models/artifacts/ for the newest run_id
    matching the given objective prefix.
    
    Example: discover_latest_artifact("environment_forecast")
    Returns: "environment_forecast_20260226_141843" (or None if not found)
    
    The convention is: <objective>_<YYYYMMDD>_<HHMMSS>
    Artifacts are sorted by timestamp; latest wins.
    """
```

Used by: `disturbance.py`, `disease_penalty.py`, `growth_weights.py`, `image_streamer.py`.

**Inputs**: Objective string prefix.  
**Outputs**: `str` run ID or `None`.

---

### 6.22 `evaluation_metrics.py`

**Purpose**: Pure computation functions for comparison metrics. Separated from `evaluation.py` to keep that class lean (single responsibility).

**Key functions:**

```python
def rmse(predicted: np.ndarray, actual: np.ndarray) -> float:
def mae(predicted: np.ndarray, actual: np.ndarray) -> float:
def time_above_threshold(series: np.ndarray, threshold: float) -> float:
def tracking_score(trajectory, setpoints) -> float:  # higher = better
def constraint_violation_count(trajectory, constraints) -> int:
```

**Inputs**: NumPy arrays or trajectory payloads.  
**Outputs**: Scalar metrics.

---

### 6.23 `__init__.py`

**Purpose**: Defines the public API of the MPC package.

**Exports:**

```python
from agritwin_gh.mpc import (
    MPCRunner,
    MPCConfig,
    load_mpc_config,
    FusedState,
    MPCSolution,
    DigitalTwinStepPayload,
    DigitalTwinTrajectoryPayload,
    ComparisonMetrics,
    discover_latest_artifact,
)
```

Only these symbols need to be imported by code *outside* the MPC package. Internal files use relative imports.

---

## 7. Key Data Structures

### 7.1 `FusedState`

The single most important data structure — combines *everything* the MPC needs to know to make a decision.

```python
@dataclass
class FusedState:
    # ── Current physical state ───────────────────────────
    greenhouse: GreenhouseState          # 9 sensor readings
    actuators: ActuatorState             # 7 latest actuator settings
    timestamp: datetime

    # ── Growth information ───────────────────────────────
    growth_stage: str                    # e.g. "flowering"
    next_stage: str                      # e.g. "unripe"
    hours_to_transition: float           # e.g. 38.5
    growth_stage_confidence: float       # classifier confidence [0,1]

    # ── Disease information ──────────────────────────────
    disease_classification: str          # e.g. "early blight"
    disease_confidence: float            # classifier confidence [0,1]
    severity_current: float              # current severity [0,1]
    severity_24h: dict[str, float]       # per-disease LSTM forecast
    severity_48h: dict[str, float]       # per-disease LSTM forecast
    disease_risk_score: float            # derived risk [0,1]
    disease_penalty: float               # cost-function penalty

    # ── Weather forecast ─────────────────────────────────
    weather_disturbance: list[dict]      # per-step forecast (len = horizon_steps)

    # ── MPC targets ──────────────────────────────────────
    setpoint: StageSetpoint              # targets for current stage
    constraints: ConstraintSet           # bounds for current stage
    cost_weights: dict[str, float]       # adaptive weights for current stage

    # ── Image metadata ───────────────────────────────────
    image: ImagePayload                  # latest plant image reference
```

---

### 7.2 `MPCSolution`

The output of `MPCSolver.solve()`:

```python
@dataclass
class MPCSolution:
    first_action: ActuatorState           # THE command to apply right now
    predicted_states: list[GreenhouseState] # future trajectory prediction
    optimal_actuator_sequence: ndarray    # shape (N_steps, 7)
    total_cost: float                     # J(u*)
    cost_breakdown: dict[str, float]      # {term: value} for debugging
    converged: bool
    fallback_used: bool                   # True if rule-based fallback was used
    solve_time_ms: float
    n_iterations: int
    n_function_evals: int
    solver_status: int                    # scipy: 0 = success
    last_weather_stress_summary: dict
    last_constraint_tightening: dict
```

---

### 7.3 `DigitalTwinStepPayload`

The formatted output sent to the dashboard after each 5-minute step:

```python
@dataclass
class DigitalTwinStepPayload:
    run_id: str
    step_index: int
    timestamp: datetime

    # Current state snapshot
    state: GreenhouseState
    actuators: ActuatorState
    predicted_next_state: GreenhouseState | None

    # Growth & disease
    growth_stage: str
    disease_classification: str
    disease_risk_score: float
    alert_level: str               # "none" | "low" | "medium" | "high"

    # Performance
    step_cost: float
    energy_kwh: float
    water_litres: float
    cost_breakdown: dict[str, float]
    solver_converged: bool

    # Contextual info
    image: ImagePayload
    weather_stress: dict | None
    tightened_constraints: dict | None
    decision_context: ControllerDecisionContext
    solver_performance: dict
```

---

### 7.4 `ComparisonMetrics`

Output of `BaselineVsMPCEvaluator.compute_all()`:

```python
@dataclass
class ComparisonMetrics:
    # Tracking accuracy
    mpc_temp_rmse: float
    baseline_temp_rmse: float
    mpc_humidity_rmse: float
    baseline_humidity_rmse: float

    # Resource efficiency
    mpc_total_energy_kwh: float
    baseline_total_energy_kwh: float
    energy_savings_pct: float          # (baseline - mpc) / baseline × 100

    mpc_total_water_litres: float
    baseline_total_water_litres: float
    water_savings_pct: float

    # Disease management
    mpc_mean_disease_risk: float
    baseline_mean_disease_risk: float
    mpc_time_above_risk_threshold: float   # hours above 0.5
    baseline_time_above_risk_threshold: float

    # Overall
    mpc_tracking_score: float          # higher = better
    baseline_tracking_score: float
    mpc_constraint_violations: int
    baseline_constraint_violations: int
    n_steps: int
    run_id: str
```

---

## 8. Canonical Labels Reference

These exact strings must be used everywhere in the codebase. Any deviation will cause label-map lookups to fail.

### Growth Stages

| Index | Canonical Name | DB Code | MinIO Subfolder |
|-------|---------------|---------|-----------------|
| 0 | `"seedling"` | 1 | `"seedling"` |
| 1 | `"early vegetative"` | 2 | `"early_vegetative"` |
| 2 | `"flowering initiation"` | 3 | `"flowering_initiation"` |
| 3 | `"flowering"` | 4 | `"flowering"` |
| 4 | `"unripe"` | 5 | `"unripe"` |
| 5 | `"ripe"` | 6 | `"ripe"` |

### Disease / Health Categories

| Index | Canonical Name | MinIO Subfolder | Notes |
|-------|---------------|-----------------|-------|
| 0 | `"healthy leaves"` | `"Healthy"` | Healthy |
| 1 | `"early blight"` | `"Early_Blight"` | *Alternaria solani* |
| 2 | `"late blight"` | `"Late_Blight"` | *Phytophthora infestans* |
| 3 | `"leaf mold"` | `"Leaf_Mold"` | *Fulvia fulva* |
| 4 | `"yellow leaf curl virus"` | `"Tomato_Yellow_Leaf_Curl_Virus"` | TYLCV |
| 5 | `"mosaic virus"` | `"Tomato_Mosaic_Virus"` | ToMV |

---

## 9. Configuration Guide

### File location

```
config/mpc_config.yaml
```

Referenced from `config/settings.yaml`:
```yaml
mpc:
  config_file: "config/mpc_config.yaml"
```

### Full annotated configuration

```yaml
mpc:
  # ── Timing ──────────────────────────────────────────────────────
  dt_minutes: 5                        # Control timestep (5 minutes)
  prediction_horizon_hours: 12         # Look 12 hours ahead
  control_horizon_hours: 6             # Optimise first 6 hours actively

  # ── Solver ──────────────────────────────────────────────────────
  solver:
    method: "SLSQP"                    # scipy.optimize method
    max_iter: 200                      # Max SLSQP iterations
    ftol: 1.0e-6                       # Convergence tolerance
    verbose: false                     # Set true to see solver output

  # ── Base cost weights (multiplied by per-stage factors below) ──
  cost_weights:
    temperature: 1.0
    humidity: 1.0
    soil_moisture: 0.8
    co2: 0.5
    vpd: 0.6
    light: 0.4
    disease_risk: 2.0                  # High — disease is the #1 profit risk
    energy: 0.3
    water: 0.3
    actuator_switching: 0.1

  # ── Per-stage multipliers on the weights above ─────────────────
  stage_weight_multipliers:
    seedling:
      temperature: 1.2
      humidity: 1.5
      disease_risk: 0.8
      soil_moisture: 1.3
    early vegetative:
      temperature: 1.1
      humidity: 1.2
      light: 1.3
      co2: 1.1
    flowering initiation:
      temperature: 1.4
      humidity: 1.2
      disease_risk: 1.3
      co2: 1.3
    flowering:
      temperature: 1.5
      humidity: 1.5
      disease_risk: 1.5
      co2: 1.5
    unripe:
      temperature: 1.2
      humidity: 1.2
      disease_risk: 1.5
      soil_moisture: 1.3
    ripe:
      temperature: 0.9
      humidity: 0.9
      disease_risk: 1.2

  # ── Target setpoints per growth stage ─────────────────────────
  setpoints:
    seedling:
      temp: 24.0
      temp_tol: 2.0
      humidity: 75.0
      hum_tol: 8.0
      soil_moisture: 70.0
      co2: 600
      light: 200
      vpd: 0.6
      disease_risk_max: 0.4
    # ... (similar blocks for all 6 stages)
    flowering:
      temp: 21.0       # Cooler to maximise pollen viability
      temp_tol: 1.5    # Tighter tolerance at critical stage
      humidity: 62.0
      hum_tol: 5.0
      soil_moisture: 60.0
      co2: 1000        # High CO2 improves fruit set
      light: 450
      vpd: 1.0
      disease_risk_max: 0.35    # Strictest threshold at flowering

  # ── Actuator physical bounds ────────────────────────────────────
  actuator_bounds:
    fan_speed: [0.0, 1.0]            # Normalised fraction 0–1
    vent_opening: [0.0, 1.0]
    irrigation_qty: [0.0, 50.0]      # Litres per 5-minute step
    heater_output: [0.0, 1.0]
    led_intensity: [0.0, 1.0]
    co2_valve_pct: [0.0, 1.0]
    fogger_duty: [0.0, 1.0]

  actuator_rate_limits:              # Max change per step
    fan_speed: 0.2
    vent_opening: 0.15
    irrigation_qty: 50.0             # No ramp limit on irrigation
    heater_output: 0.2
    led_intensity: 0.15
    co2_valve_pct: 0.2
    fogger_duty: 0.3

  actuator_cooldown_steps:           # Min steps between changes
    irrigation_qty: 12               # 12 × 5 min = 60 min cooldown
    co2_valve_pct: 6                 # 6 × 5 min = 30 min cooldown
    vent_opening: 2                  # 2 × 5 min = 10 min cooldown

  # ── Environmental safe-range bounds ────────────────────────────
  env_bounds:
    indoor_temp: [10.0, 40.0]
    indoor_humidity: [30.0, 95.0]
    co2: [300.0, 2000.0]
    soil_moisture: [20.0, 95.0]
    light_intensity: [0.0, 1200.0]
    vpd: [0.2, 2.5]

  # ── Resource budgets ────────────────────────────────────────────
  resources:
    daily_water_budget_litres: 500.0
    daily_energy_budget_kwh: 100.0

  # ── Energy cost model ───────────────────────────────────────────
  energy_costs:
    fan_kw_per_unit: 0.5
    heater_kw_per_unit: 2.0
    led_kw_per_unit: 0.3
    co2_kw_per_unit: 0.01
    fogger_kw_per_unit: 0.15
    pump_kwh_per_event: 0.02

  # ── Model artifact IDs (leave null for auto-discovery) ─────────
  model_artifacts:
    environment_forecast_run_id: null
    disease_progression_run_id: null
    growth_progression_run_id: null
    disease_classifier_run_id: null
    growth_classifier_run_id: null

  # ── Image streaming ─────────────────────────────────────────────
  image_stream_interval_minutes: 5
```

---

## 10. How to Run the MPC Module

### Prerequisites

```
python >= 3.13
postgresql (agritwin_db running)
minio (agritwin-images bucket populated)
trained model artifacts in data/processed/models/artifacts/
```

### Environment setup

```powershell
# Option A: uv (recommended)
uv sync
$env:PYTHONPATH = "e:\AgriTwin-GH\src"

# Option B: pip editable install
pip install -e .
```

### Single-step execution (simplest possible use)

```python
import datetime
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from agritwin_gh.mpc import MPCRunner, load_mpc_config

# 1. Connect to database
engine = create_engine("postgresql://user:pass@localhost/agritwin_db")

with Session(engine) as session:
    # 2. Create runner (loads all sub-systems automatically)
    runner = MPCRunner(session=session)

    # 3. Execute one 5-minute MPC step
    payload = runner.run_single_step()

    # 4. Inspect the result
    print(f"Growth stage : {payload.growth_stage}")
    print(f"Disease risk : {payload.disease_risk_score:.3f}")
    print(f"Alert level  : {payload.alert_level}")
    print(f"Heater       : {payload.actuators.heater_output:.2f}")
    print(f"Fan speed    : {payload.actuators.fan_speed:.2f}")
    print(f"Vent opening : {payload.actuators.vent_opening:.2f}")
    print(f"Step cost    : {payload.step_cost:.4f}")
    print(f"Energy (kWh) : {payload.energy_kwh:.4f}")
    print(f"Solver OK?   : {payload.solver_converged}")
```

### Full 24-hour simulation

```python
import datetime
from sqlalchemy.orm import Session

from agritwin_gh.mpc import MPCRunner

with Session(engine) as session:
    runner = MPCRunner(session=session)

    start = datetime.datetime(2026, 3, 30, 6, 0)
    end   = datetime.datetime(2026, 3, 31, 6, 0)

    trajectory = runner.run_simulation(start_time=start, end_time=end)

    print(f"Steps completed : {len(trajectory.steps)}")
    print(f"Total cost      : {trajectory.total_cost:.2f}")
    print(f"Total energy    : {trajectory.total_energy_kwh:.2f} kWh")
    print(f"Total water     : {trajectory.total_water_litres:.2f} L")
```

### Streaming to a live dashboard

```python
with Session(engine) as session:
    runner = MPCRunner(session=session)

    for payload in runner.run_simulation_iter(start, end):
        # Called once every 5 minutes of simulation time
        dashboard.push(payload.to_dict())   # your dashboard integration
```

### Running a full experiment (MPC vs. Baseline comparison)

```python
from agritwin_gh.mpc.experiment_runner import run_experiment

with Session(engine) as session:
    metrics = run_experiment(
        session=session,
        start_time=datetime.datetime(2026, 3, 30, 0, 0),
        end_time=datetime.datetime(2026, 4, 6, 0, 0),  # 7-day run
    )

    print(f"MPC energy savings   : {metrics.energy_savings_pct:.1f}%")
    print(f"MPC mean disease risk: {metrics.mpc_mean_disease_risk:.3f}")
    print(f"Baseline mean risk   : {metrics.baseline_mean_disease_risk:.3f}")
    print(f"MPC constraint viols : {metrics.mpc_constraint_violations}")
```

Results are automatically saved to `data/processed/mpc_results/<run_id>/`.

### Using a custom configuration

```python
from agritwin_gh.mpc import MPCRunner, load_mpc_config

config = load_mpc_config("path/to/my_custom_mpc_config.yaml")

# Override one field programmatically
config.prediction_horizon_hours = 6.0
config.solver_max_iter = 500

with Session(engine) as session:
    runner = MPCRunner(session=session, config=config)
    payload = runner.run_single_step()
```

### Plugging in your own disease / growth classifiers

```python
from agritwin_gh.disease_inference import predict_image   # your model
from agritwin_gh.growth_stage_inference import predict_growth_stage

with Session(engine) as session:
    runner = MPCRunner(
        session=session,
        disease_classifier=predict_image,
        growth_classifier=predict_growth_stage,
        device="cuda",   # use GPU for weather forecast model
    )
    payload = runner.run_single_step()
```

---

## 11. Test Scripts

All tests live in `tests/` and follow pytest conventions. Run them:

```powershell
$env:PYTHONPATH = "e:\AgriTwin-GH\src"
uv run pytest tests/ -v
```

### `test_state.py` — State dataclasses

```python
import numpy as np
from agritwin_gh.mpc.state import GreenhouseState, ActuatorState

def test_greenhouse_state_roundtrip():
    """GreenhouseState → numpy → GreenhouseState must be lossless."""
    original = GreenhouseState(
        indoor_temp=22.5, indoor_humidity=68.0, co2_level=800.0,
        soil_moisture=65.0, light_intensity=400.0,
        outdoor_temp=18.0, outdoor_humidity=55.0, vpd=0.9, leaf_wetness=0.1,
    )
    arr = original.to_numpy()
    assert arr.shape == (9,)
    recovered = GreenhouseState.from_numpy(arr)
    assert abs(recovered.indoor_temp - original.indoor_temp) < 1e-9

def test_actuator_state_clip():
    """Actuator values outside bounds are clipped, not errored."""
    from agritwin_gh.mpc.constraints import get_default_constraints
    cs = get_default_constraints("flowering")
    a = ActuatorState(
        fan_speed=2.0,       # exceeds max of 1.0
        vent_opening=0.5,
        irrigation_qty=-5.0, # below min of 0.0
        heater_output=0.3,
        led_intensity=0.8,
        co2_valve_pct=0.4,
        fogger_duty=0.2,
    )
    clipped = a.clip(cs)
    assert clipped.fan_speed == 1.0
    assert clipped.irrigation_qty == 0.0
```

### `test_greenhouse_model.py` — Physics plausibility

```python
from agritwin_gh.mpc.state import GreenhouseState, ActuatorState
from agritwin_gh.mpc.greenhouse_model import GreenhouseTransitionModel

def test_heater_warms_greenhouse():
    """Turning heater on should raise indoor temperature."""
    model = GreenhouseTransitionModel()
    state = GreenhouseState(
        indoor_temp=15.0, indoor_humidity=70.0, co2_level=600.0,
        soil_moisture=60.0, light_intensity=200.0,
        outdoor_temp=10.0, outdoor_humidity=50.0, vpd=0.5, leaf_wetness=0.1,
    )
    heater_on  = ActuatorState(0.2, 0.1, 0.0, 1.0, 0.3, 0.2, 0.1)  # heater=1.0
    heater_off = ActuatorState(0.2, 0.1, 0.0, 0.0, 0.3, 0.2, 0.1)  # heater=0.0

    next_on  = model.step(state, heater_on)
    next_off = model.step(state, heater_off)

    assert next_on.indoor_temp > next_off.indoor_temp

def test_simulate_trajectory_length():
    """simulate() must return len(actuator_sequence) + 1 states."""
    model = GreenhouseTransitionModel()
    s0 = GreenhouseState(22.0, 68.0, 800.0, 65.0, 400.0, 18.0, 55.0, 0.9, 0.1)
    acts = [ActuatorState(0.3, 0.2, 0.0, 0.2, 0.5, 0.3, 0.1)] * 10
    trajectory = model.simulate(s0, acts)
    assert len(trajectory) == 11  # initial + 10 steps
```

### `test_cost_function.py` — Cost function

```python
from agritwin_gh.mpc.cost_function import StageCost, CostBuilder
from agritwin_gh.mpc.setpoints import get_setpoint

def test_cost_at_setpoint_is_minimal():
    """When state exactly matches setpoint, tracking cost should be near zero."""
    sp = get_setpoint("flowering")
    at_setpoint = GreenhouseState(
        indoor_temp=sp.temp, indoor_humidity=sp.humidity,
        co2_level=sp.co2, soil_moisture=sp.soil_moisture,
        light_intensity=sp.light,
        outdoor_temp=18.0, outdoor_humidity=55.0, vpd=sp.vpd, leaf_wetness=0.1,
    )
    cost_fn = CostBuilder().build(weights={"temperature": 1.0, "humidity": 1.0, ...})
    cost = cost_fn(at_setpoint, sp, ActuatorState(0,0,0,0,0,0,0), None, 0.0)
    assert cost < 0.1

def test_high_disease_risk_increases_cost():
    """High disease penalty should increase total cost."""
    cost_low  = compute_cost(disease_penalty=0.0)
    cost_high = compute_cost(disease_penalty=1.0)
    assert cost_high > cost_low
```

### `test_mpc_solver.py` — Solver

```python
from agritwin_gh.mpc.mpc_solver import MPCSolver
from agritwin_gh.mpc.config import load_mpc_config

def test_solver_returns_valid_actuators():
    """Solver must return an ActuatorState with all values in valid range."""
    config = load_mpc_config()
    # (build a mock FusedState...)
    solver = MPCSolver(config=config, model=GreenhouseTransitionModel())
    solution = solver.solve(fused=mock_fused_state)

    a = solution.first_action
    assert 0.0 <= a.fan_speed    <= 1.0
    assert 0.0 <= a.heater_output <= 1.0
    assert a.irrigation_qty >= 0.0

def test_solver_fallback_on_infeasible():
    """If constraints are infeasible, fallback_used must be True."""
    # Inject contradictory constraints...
    solution = solver.solve(fused=infeasible_fused_state)
    assert solution.fallback_used is True
    # But still returns valid actuators
    assert solution.first_action is not None

def test_cost_does_not_increase_over_iterations():
    """Running solve twice with the warm-start should not produce higher cost."""
    sol1 = solver.solve(fused=fused_state)
    sol2 = solver.solve(fused=fused_state)  # warm-started
    # Cost should be stable (not necessarily decreasing, but not diverging)
    assert abs(sol2.total_cost - sol1.total_cost) < 1.0
```

### `test_baseline_controller.py` — Baseline

```python
from agritwin_gh.mpc.baseline_controller import RuleBasedController

def test_high_risk_triggers_emergency_ventilation():
    """Disease risk > 0.6 must set fan_speed and vent_opening to maximum."""
    ctrl = RuleBasedController()
    action = ctrl.compute_action(
        state=normal_state,
        growth_stage="flowering",
        disease_risk_score=0.8,   # HIGH
    )
    assert action.fan_speed == 1.0
    assert action.vent_opening == 1.0

def test_cold_greenhouse_triggers_heater():
    """Temperature below setpoint − tolerance must turn on heater."""
    cold_state = GreenhouseState(indoor_temp=15.0, ...)  # vs setpoint 21 °C
    ctrl = RuleBasedController()
    action = ctrl.compute_action(cold_state, "flowering", disease_risk_score=0.1)
    assert action.heater_output > 0.0
```

### `test_state_fusion.py` — Integration

```python
from unittest.mock import MagicMock
from agritwin_gh.mpc.state_fusion import StateFusion

def test_fuse_returns_valid_fused_state():
    """StateFusion.fuse() should return a FusedState with all required fields."""
    # Use mocked sub-components to avoid DB dependency in unit test
    mock_input_prep = MagicMock()
    mock_input_prep.get_latest_greenhouse_row.return_value = {
        "indoor_temp": 22.0, "indoor_humidity": 68.0, ...
    }
    # ... (other mocks)

    fusion = StateFusion(config=load_mpc_config(), input_prep=mock_input_prep, ...)
    fused = fusion.fuse(timestamp=datetime.datetime.now())

    assert fused.growth_stage in GROWTH_STAGES
    assert 0.0 <= fused.disease_risk_score <= 1.0
    assert fused.setpoint is not None
    assert len(fused.weather_disturbance) > 0
```

### `test_runner_integration.py` — End-to-end (with mocked DB)

```python
from unittest.mock import MagicMock, patch
from agritwin_gh.mpc import MPCRunner

def test_single_step_pipeline():
    """Full pipeline from DB mock to DigitalTwinStepPayload."""
    mock_session = MagicMock()

    with patch("agritwin_gh.mpc.mpc_input_preparation.MPCInputPreparation") as MockPrep:
        MockPrep.return_value.get_latest_greenhouse_row.return_value = {
            "indoor_temp": 22.0, ...
        }
        runner = MPCRunner(session=mock_session)
        payload = runner.run_single_step()

    assert payload.step_index == 0
    assert payload.growth_stage in GROWTH_STAGES
    assert hasattr(payload, "actuators")
    assert hasattr(payload, "solver_converged")
```

---

## 12. Artifact & Logging Strategy

### Run ID convention

Every MPC run is identified by:
```
mpc_<YYYYMMDD>_<HHMMSS>
```
Example: `mpc_20260330_143022`

This mirrors the existing repo convention (`disease_20260226_141843`, `growth_stage_20260302_170744`).

### Output files per run

```
data/processed/mpc_results/mpc_20260330_143022/
│
├── trajectory_mpc.parquet             # MPC state + actuator trajectory
├── trajectory_baseline.parquet        # Baseline controller trajectory
├── comparison_metrics.json            # ComparisonMetrics serialised
├── run_config.json                    # MPCConfig snapshot (reproducibility)
├── fused_states.parquet               # All FusedState snapshots (for replay)
├── step_payloads.jsonl                # DigitalTwinStepPayload per step (newline JSON)
│
└── figures/
    ├── temperature_comparison.png
    ├── humidity_comparison.png
    ├── disease_risk_comparison.png
    ├── resource_consumption.png
    ├── actuator_profiles.png
    └── metrics_bar_chart.png
```

### Structured logs

```
logs/mpc/mpc_run_mpc_20260330_143022.json
```

Each log entry (one per step) contains:
```json
{
  "timestamp": "2026-03-30T14:35:22",
  "step_index": 1,
  "solve_time_ms": 42.3,
  "solver_status": 0,
  "cost_breakdown": {
    "temperature": 0.042,
    "humidity": 0.118,
    "disease_risk": 0.003,
    "energy": 0.021
  },
  "constraint_violations": [],
  "applied_actuators": {"fan_speed": 0.45, "heater_output": 0.0, ...},
  "observed_state_summary": {"indoor_temp": 22.1, ...}
}
```

Python logger name: `agritwin_gh.mpc`

### Calibrated model artifacts

When `GreenhouseTransitionModel.calibrate()` is called on historical data, the fitted parameters are saved to:
```
src/agritwin_gh/models/artifacts/greenhouse_model_<run_id>/
├── calibrated_params.json
└── calibration_report.json      # R², residual stats per sub-model
```

---

## 13. Assumptions & Design Decisions

### Design decisions

| Decision | What was chosen | Why |
|----------|----------------|-----|
| Solver | scipy SLSQP | Linear ARX model makes the problem convex; SLSQP is fast (≪ 1 s solve), no external solver dependency |
| Fallback | RuleBasedController | Guarantees safe actuator output even on infeasibility; zero extra dependency |
| Step time | 5 minutes | Matches DB write frequency; fast enough to respond to disturbances, slow enough to avoid actuator wear |
| Horizon | 12 h predict, 6 h control | Covers typical weather cycle; control horizon shorter to limit decision variables |
| Plant model | ARX (linear) | Fast enough for real-time; calibratable from historical DB data with Ridge regression |
| Formulation | Single-shooting | Simpler implementation; adequate for short horizons with linear model |
| State vector | 9 variables | Covers all measureable indoor quantities; VPD and leaf wetness are derived but important for disease/transpiration |

### Assumptions

| ID | Assumption | Consequence |
|----|-----------|------------|
| A1 | Soil moisture is not in the DB | `GreenhouseTransitionModel` derives it via water balance; initialised from config default |
| A2 | Greenhouse model is linear (ARX) | SLSQP is appropriate; for a neural-network plant model switch to CasADi/IPOPT |
| A3 | Weather forecast gives 24h and 48h points | Intermediate values are linearly interpolated — adequate for a 12-hour horizon |
| A4 | Disease LSTM predicts at 24h granularity | Penalty uses horizon-level projections, not per-step; extension point: higher-resolution model |
| A5 | One crop cycle active at a time | `get_latest_cycle_id()` returns MAX(cycle_id); multi-cycle support is an extension |
| A6 | Actuators are independent | No cross-coupling constraints (e.g. "no heater while vent open"); can add as linear inequalities |
| A7 | Image streamer returns metadata only | Dashboard fetches actual bytes from MinIO; keeps MPC loop fast |
| A8 | All trained models already exist | Auto-discovery via `discover_latest_artifact()`; if no model found, model wrappers raise `RuntimeError` with clear message |

---

## 14. Extension Points

| What you want to add | Where to change | What changes |
|---------------------|----------------|-------------|
| **Nonlinear plant model** (neural network) | `greenhouse_model.py` | Replace ARX with PyTorch/TF net; switch solver in `mpc_solver.py` to CasADi/IPOPT |
| **Stochastic MPC** | `mpc_solver.py` | Sample scenario trees from weather forecast uncertainty; optimise expected cost |
| **Multi-zone greenhouse** | `state.py`, `greenhouse_model.py` | State vector becomes matrix (zones × variables); model becomes zone graph |
| **Real-time sensor feed** (MQTT) | `mpc_input_preparation.py` | Replace `get_latest_greenhouse_row()` with MQTT subscription |
| **Reinforcement learning agent** | `runner.py` | Plug in RL policy alongside `MPCSolver` as alternative controller with same interface |
| **New disease** | `constants.py` | Add entry to `DISEASE_CATEGORIES` and `IMAGE_SUBCATEGORY_MAP` |
| **New actuator** | `constants.py`, `state.py`, `constraints.py` | Add field to `ActuatorState`, extend bounds/rates in config |
| **Economic MPC** | `cost_function.py` | Add electricity price signal, crop market price, to cost terms |
| **Online model calibration** | `greenhouse_model.py`, `runner.py` | Periodically re-fit Ridge coefficients on recent data inside the runner loop |
| **FastAPI service** | New `services/mpc_service.py` | Wrap `MPCRunner.run_single_step()` in an HTTP endpoint |
| **WebSocket streaming** | New `api/mpc_endpoints.py` | Use `run_simulation_iter()` generator with WebSocket push |

---

## 15. Phased Build Roadmap

The MPC module was built in 10 ordered phases. Understanding this helps you know what each file depends on.

| Phase | Files | What was built |
|-------|-------|---------------|
| 1 | `constants.py`, `state.py` | All canonical enums, state dataclasses with numpy/DB roundtrip |
| 2 | `setpoints.py`, `config.py` | Stage-specific targets, YAML config loader |
| 3 | `greenhouse_model.py` | ARX transition model: step/simulate/calibrate |
| 4 | `baseline_controller.py`, `constraints.py` | Rule-based controller, constraint sets |
| 5 | `cost_function.py`, `mpc_solver.py` | Stage-aware cost, SLSQP optimiser with fallback |
| 6 | `disturbance.py`, `mpc_input_preparation.py` | Weather forecast wrapper, DB queries |
| 7 | `disease_penalty.py` | Disease risk score + LSTM severity penalty |
| 8 | `growth_weights.py`, `state_fusion.py` | Adaptive weights, full state-fusion pipeline |
| 9 | `digital_twin_output.py`, `image_streamer.py`, `runner.py` | Output formatters, MinIO image retrieval, top-level orchestrator |
| 10 | `evaluation.py`, `evaluation_metrics.py`, `experiment_runner.py` | MPC vs. Baseline comparison, plots, full experiment runner |
| + | `weather_adaptation.py`, `utils.py` | Weather-adaptive constraint tightening, shared artifact discovery |

---

*AgriTwin-GH MPC Complete Guide — `src/agritwin_gh/mpc/` (26 files)*
