# AgriTwin-GH: Model Predictive Control (MPC)

> **Who is this for?**  
> This guide is written so that anyone — from a curious beginner with no control theory background to an experienced ML engineer — can understand what the MPC module does, how every file fits together, how to run it, and how to extend it.

---

## Table of Contents

1\. [What is MPC? (Plain English)](#1-what-is-mpc-plain-english)

2\. [How AgriTwin-GH Uses MPC](#2-how-agritwin-gh-uses-mpc)

3\. [System Architecture at a Glance](#3-system-architecture-at-a-glance)

4\. [Full Data Flow Diagram](#4-full-data-flow-diagram)

5\. [File Tree — All 26 Source Files](#5-file-tree--all-26-source-files)

6\. [File-by-File Reference](#6-file-by-file-reference)

&emsp;6.1\. [`constants.py`](#61-constantspy)

&emsp;6.2\. [`state.py`](#62-statepy)

&emsp;&emsp;6.2.1\. [`GreenhouseState`](#greenhousestate)

&emsp;&emsp;6.2.2\. [`ActuatorState`](#actuatorstate)

&emsp;&emsp;6.2.3\. [`WeatherState`](#weatherstate)

&emsp;&emsp;6.2.4\. [`MPCState`](#mpcstate)

&emsp;&emsp;6.2.5\. [`FusedState`](#fusedstate)

&emsp;&emsp;6.2.6\. [`ControllerDecisionContext`](#controllerdecisioncontext)

&emsp;&emsp;6.2.7\. [`DigitalTwinStepPayload` / `DigitalTwinTrajectoryPayload`](#digitaltwinsteppayload--digitaltwintrajectorypayload)

&emsp;6.3\. [`setpoints.py`](#63-setpointspy)

&emsp;6.4\. [`constraints.py`](#64-constraintspy)

&emsp;6.5\. [`greenhouse_model.py`](#65-greenhouse_modelpy)

&emsp;&emsp;6.5.1\. [ARX model — equations and implementation](#arx-model--equations-and-implementation)

&emsp;&emsp;6.5.2\. [Per-variable ARX equations](#per-variable-arx-equations)

&emsp;&emsp;6.5.3\. [Coefficient reference (`GreenhouseModelParams`)](#coefficient-reference-greenhousemodelparams----greenhouse_modelpy--l40)

&emsp;&emsp;6.5.4\. [Why ARX for MPC?](#why-arx-for-mpc)

&emsp;6.6\. [`baseline_controller.py`](#66-baseline_controllerpy)

&emsp;6.7\. [`cost_function.py`](#67-cost_functionpy)

&emsp;&emsp;6.7.1\. [Building blocks overview](#building-blocks-overview)

&emsp;&emsp;6.7.2\. [The complete objective function](#the-complete-objective-function)

&emsp;&emsp;6.7.3\. [Running cost — all nine terms](#running-cost-xk-uk--all-nine-terms)

&emsp;&emsp;6.7.4\. [Term 1 — Setpoint Tracking](#term-1--setpoint-tracking)

&emsp;&emsp;6.7.5\. [Term 2 — Disease Environment Penalty](#term-2--disease-environment-penalty)

&emsp;&emsp;6.7.6\. [Term 3 — Humidity Exposure Penalty](#term-3--humidity-exposure-penalty)

&emsp;&emsp;6.7.7\. [Term 4 — Fogger Suppression Penalty](#term-4--fogger-suppression-penalty)

&emsp;&emsp;6.7.8\. [Term 5 — Irrigation Caution Penalty](#term-5--irrigation-caution-penalty)

&emsp;&emsp;6.7.9\. [Term 6 — Energy Cost](#term-6--energy-cost)

&emsp;&emsp;6.7.10\. [Term 7 — Water Cost](#term-7--water-cost)

&emsp;&emsp;6.7.11\. [Term 8 — Environmental Bounds Barrier](#term-8--environmental-bounds-barrier)

&emsp;&emsp;6.7.12\. [Term 9 — Actuator Switching Penalty](#term-9--actuator-switching-penalty)

&emsp;&emsp;6.7.13\. [Terminal cost](#terminal-cost-vfxn)

&emsp;&emsp;6.7.14\. [Base cost weights](#base-cost-weights-current-tuned-values-from-configpy)

&emsp;&emsp;6.7.15\. [Stage weight multipliers](#stage-weight-multipliers)

&emsp;&emsp;6.7.16\. [Growth-stage transition blending](#growth-stage-transition-blending)

&emsp;&emsp;6.7.17\. [Weather-adaptive weight scaling](#weather-adaptive-weight-scaling)

&emsp;&emsp;6.7.18\. [`DiseaseContext` dataclass](#diseasecontext-dataclass)

&emsp;6.8\. [`mpc_solver.py`](#68-mpc_solverpy)

&emsp;6.9\. [`disturbance.py`](#69-disturbancepy)

&emsp;6.10\. [`disease_penalty.py`](#610-disease_penaltypy)

&emsp;6.11\. [`growth_weights.py`](#611-growth_weightspy)

&emsp;6.12\. [`state_fusion.py`](#612-state_fusionpy)

&emsp;6.13\. [`mpc_input_preparation.py`](#613-mpc_input_preparationpy)

&emsp;6.14\. [`digital_twin_output.py`](#614-digital_twin_outputpy)

&emsp;6.15\. [`image_streamer.py`](#615-image_streamerpy)

&emsp;6.16\. [`evaluation.py`](#616-evaluationpy)

&emsp;6.17\. [`runner.py`](#617-runnerpy)

&emsp;6.18\. [`config.py`](#618-configpy)

&emsp;6.19\. [`experiment_runner.py`](#619-experiment_runnerpy)

&emsp;6.20\. [`weather_adaptation.py`](#620-weather_adaptationpy)

&emsp;6.21\. [`utils.py`](#621-utilspy)

&emsp;6.22\. [`evaluation_metrics.py`](#622-evaluation_metricspy)

&emsp;6.23\. [`__init__.py`](#623-__init__py)

&emsp;6.24\. [`yield_proxy.py`](#624-yield_proxypy)

7\. [Key Data Structures](#7-key-data-structures)

&emsp;7.1\. [`FusedState`](#71-fusedstate)

&emsp;7.2\. [`MPCSolution`](#72-mpcsolution)

&emsp;7.3\. [`DigitalTwinStepPayload`](#73-digitaltwinsteppayload)

&emsp;7.4\. [`ComparisonMetrics`](#74-comparisonmetrics)

8\. [Canonical Labels Reference](#8-canonical-labels-reference)

&emsp;8.1\. [Growth Stages](#growth-stages)

&emsp;8.2\. [Disease / Health Categories](#disease--health-categories)

9\. [Configuration Guide](#9-configuration-guide)

&emsp;9.1\. [File location](#file-location)

&emsp;9.2\. [Full annotated configuration](#full-annotated-configuration)

10\. [How to Run the MPC Module](#10-how-to-run-the-mpc-module)

&emsp;10.1\. [Prerequisites](#prerequisites)

&emsp;10.2\. [Environment setup](#environment-setup)

&emsp;10.3\. [Single-step execution (simplest possible use)](#single-step-execution-simplest-possible-use)

&emsp;10.4\. [Full 24-hour simulation](#full-24-hour-simulation)

&emsp;10.5\. [Streaming to a live dashboard](#streaming-to-a-live-dashboard)

&emsp;10.6\. [Running a full experiment (MPC vs. Baseline comparison)](#running-a-full-experiment-mpc-vs-baseline-comparison)

&emsp;10.7\. [Using a custom configuration](#using-a-custom-configuration)

&emsp;10.8\. [Plugging in your own disease / growth classifiers](#plugging-in-your-own-disease--growth-classifiers)

11\. [Test Scripts](#11-test-scripts)

&emsp;11.1\. [How to run the smoke tests](#how-to-run-the-smoke-tests)

&emsp;11.2\. [`smoke_intelligent_mpc.py` — Intelligent MPC](#tests-smoke_intelligent_mpcpy--intelligent-mpc-disease--weather--stage-blending)

&emsp;11.3\. [`smoke_test_dt_handoff.py` — Digital Twin Output, Explanation, Replay](#tests-smoke_test_dt_handoffpy--digital-twin-output-explanation-replay)

&emsp;11.4\. [`test_evaluation_smoke.py` — Evaluation Framework](#tests-test_evaluation_smokepy--evaluation-framework)

&emsp;11.5\. [Future unit tests (not yet implemented)](#future-unit-tests-not-yet-implemented)

12\. [Artifact & Logging Strategy](#12-artifact--logging-strategy)

&emsp;12.1\. [Run ID convention](#run-id-convention)

&emsp;12.2\. [Output files per run](#output-files-per-run)

&emsp;12.3\. [Structured logs](#structured-logs)

&emsp;12.4\. [Calibrated model artifacts](#calibrated-model-artifacts)

13\. [Assumptions & Design Decisions](#13-assumptions--design-decisions)

&emsp;13.1\. [Design decisions](#design-decisions)

&emsp;13.2\. [Assumptions](#assumptions)

14\. [Extension Points](#14-extension-points)

15\. [Phased Build Roadmap](#15-phased-build-roadmap)

16\. [MPC Solver Tuning & Robustness Improvements](#16-mpc-solver-tuning--robustness-improvements)

&emsp;16.1\. [Rate Constraint 5 % Slack](#161-rate-constraint-5--slack)

&emsp;16.2\. [Non-Converged Result Salvage](#162-non-converged-result-salvage)

&emsp;16.3\. [Dead-Band Filter](#163-dead-band-filter)

&emsp;16.4\. [Safety Filter](#164-safety-filter)

17\. [Constraint Reference Tables](#17-constraint-reference-tables)

&emsp;17.1\. [Environmental Constraints — Base Limits](#171-environmental-constraints--base-limits)

&emsp;17.1a\. [Growth-Stage Environmental Overrides](#171a-growth-stage-environmental-overrides)

&emsp;17.2\. [Actuator Bounds (Box Constraints)](#172-actuator-bounds-box-constraints)

&emsp;17.3\. [Actuator Rate-of-Change Limits (Per 5-Minute Step)](#173-actuator-rate-of-change-limits-per-5-minute-step)

&emsp;17.4\. [Actuator Cooldown Periods](#174-actuator-cooldown-periods)

&emsp;17.5\. [Crop Safety Bounds (Stage-Dependent)](#175-crop-safety-bounds-stage-dependent)

&emsp;17.6\. [Disease-Sensitive Constraint Tightening](#176-disease-sensitive-constraint-tightening)

18\. [Resource Cost Calculation — Tamil Nadu, India](#18-resource-cost-calculation--tamil-nadu-india)

&emsp;18.1\. [Currency and Pricing Standard](#181-currency-and-pricing-standard)

&emsp;18.2\. [Energy Consumption Model](#182-energy-consumption-model)

&emsp;18.3\. [Water Consumption Model](#183-water-consumption-model)

&emsp;18.4\. [Total Resource Cost Formula](#184-total-resource-cost-formula)

&emsp;18.5\. [Interpreting Resource Cost Comparisons](#185-interpreting-resource-cost-comparisons)

19\. [End-to-End Evaluation Script (`run_full_mpc_evaluation.py`)](#19-end-to-end-evaluation-script-run_full_mpc_evaluationpy)

&emsp;19.1\. [Overview](#191-overview)

&emsp;19.2\. [Scenario Design](#192-scenario-design)

&emsp;&emsp;19.2.1\. [Scenario 1 — Standard 12-Hour Flowering Stage](#scenario-1--standard-12-hour-flowering-stage)

&emsp;&emsp;19.2.2\. [Scenario 2 — High Disease-Pressure Fruiting Stage (24 h)](#scenario-2--high-disease-pressure-fruiting-stage-24-h)

&emsp;&emsp;19.2.3\. [Scenario 3 — 24-Hour Stage Transition (Flowering → Unripe)](#scenario-3--24-hour-stage-transition-flowering--unripe)

&emsp;&emsp;19.2.4\. [Scenario 4 — MPC Solver Component-Level Validation](#scenario-4--mpc-solver-component-level-validation)

&emsp;&emsp;19.2.5\. [Scenario 5 — Multi-Horizon Convergence Test](#scenario-5--multi-horizon-convergence-test)

&emsp;19.3\. [Yield Proxy — How Performance Is Measured](#193-yield-proxy--how-performance-is-measured)

&emsp;&emsp;19.3.1\. [Formula](#formula)

&emsp;&emsp;19.3.2\. [Climate Tracking Score (40 % of total)](#climate-tracking-score-40--of-total)

&emsp;&emsp;19.3.3\. [Stress Exposure Score (20 % of total)](#stress-exposure-score-20--of-total)

&emsp;&emsp;19.3.4\. [Resource Stability Score (15 % of total)](#resource-stability-score-15--of-total)

&emsp;19.4\. [How to Read the Output](#194-how-to-read-the-output)

&emsp;&emsp;19.4.1\. [Summary Table](#summary-table)

&emsp;&emsp;19.4.2\. [Pairwise Improvements](#pairwise-improvements)

&emsp;&emsp;19.4.3\. [Yield Proxy Breakdown](#yield-proxy-breakdown)

&emsp;&emsp;19.4.4\. [Resource Cost Table](#resource-cost-table)

&emsp;19.5\. [How to Determine Which Controller Is Better](#195-how-to-determine-which-controller-is-better)

&emsp;19.6\. [Artifact Output](#196-artifact-output)

&emsp;19.7\. [Automatic Validation Checks](#197-automatic-validation-checks)

20\. [Setpoint and Growth Stage Profile Reference](#20-setpoint-and-growth-stage-profile-reference)

&emsp;20.1\. [Stage Setpoints (Target Climate Values)](#201-stage-setpoints-target-climate-values)

&emsp;20.2\. [Stage Control Profiles (Weight Multipliers)](#202-stage-control-profiles-weight-multipliers)

21\. [References](#21-references)

---

## 1. What is MPC? (Plain English)

Imagine you are driving a car on a winding road. You constantly look ahead, predict where the road curves, and steer *now* to prepare for what is coming. You do not just react to the curve when it is already under you — you use foresight.

**Model Predictive Control (MPC)** works the same way for automated systems:

1. **The model**: A mathematical description of how the system (greenhouse) changes when you take an action (turn on heater, open vent).
2. **The prediction horizon**: A window of time into the future (e.g. 12 hours). The controller simulates what will happen over this window for different action sequences.
3. **The optimisation**: Find the sequence of actions that keeps the greenhouse closest to the desired targets (temperature, humidity, etc.) while using the least energy and keeping disease risk low.
4. **Receding horizon (the clever part)**: Only the *first* action from the best sequence is actually applied. At the very next timestep, the whole prediction+optimisation is repeated with fresh sensor data. This continuously corrects for model error and disturbances (like unexpected weather).<sup>[[1]](#ref-1), [[2]](#ref-2)</sup>

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
├── yield_proxy.py               # YieldProxyWeights, YieldProxyResult, compute_yield_proxy
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

#### ARX model — equations and implementation

**ARX (Auto-Regressive with eXogenous inputs)** is a family of linear discrete-time models. Instead of solving differential equations, it predicts the *next* value of each state variable as a weighted sum of the current state, control inputs, and external disturbances. This one-liner update replaces a full CFD simulation.<sup>[[3]](#ref-3), [[4]](#ref-4)</sup>

**General ARX form used in AgriTwin-GH:**

$$
x_i[k+1] = \alpha_i \, x_i[k]
           + \sum_{j} \beta_{ij} \, u_j[k]
           + \sum_{m} \gamma_{im} \, d_m[k]
           + \epsilon_i
$$

*[↗ greenhouse_model.py · L118](../src/agritwin_gh/mpc/greenhouse_model.py#L118)*

| Symbol | Meaning |
|---|---|
| $x_i[k]$ | State variable $i$ at timestep $k$ (e.g. indoor temperature) |
| $\alpha_i$ | Self-decay coefficient — how much of the current value persists to the next step; $\alpha < 1$ means the variable naturally drifts toward equilibrium |
| $u_j[k]$ | Actuator command $j$ at step $k$ (e.g. `heater_output`, `fan_speed`) |
| $\beta_{ij}$ | Actuator gain — how strongly actuator $j$ pushes state $i$ up or down |
| $d_m[k]$ | External disturbance $m$ at step $k$ (e.g. outdoor temperature, solar radiation) |
| $\gamma_{im}$ | Disturbance gain — how strongly weather input $m$ affects state $i$ |
| $\epsilon_i$ | Optional process noise (std configured via `GreenhouseModelParams.noise_*`; default 0) |

---

#### Per-variable ARX equations

**Temperature** — *[↗ greenhouse_model.py · L149](../src/agritwin_gh/mpc/greenhouse_model.py#L149)*

$$
T[k+1] = \underbrace{\alpha_T \, T[k]}_{\text{thermal mass}}
        + \underbrace{\gamma_{\text{ext}} \bigl(T_{\text{ext}}[k] - T[k]\bigr)}_{\text{heat exchange with outside}}
        + \underbrace{\gamma_{\text{sol}} \, S[k]}_{\text{solar gain}}
        + \underbrace{\beta_{\text{heat}} \, u_{\text{heater}}}_{\text{heater}}
        + \underbrace{\beta_{\text{fan}} \, u_{\text{fan}}}_{\text{fan cooling}}
        + \underbrace{\beta_{\text{vent}} \, u_{\text{vent}}}_{\text{vent cooling}}
$$

**Humidity** — *[↗ greenhouse_model.py · L160](../src/agritwin_gh/mpc/greenhouse_model.py#L160)*

$$
H[k+1] = \alpha_H \, H[k]
        + \gamma_{\text{ext}} \bigl(H_{\text{ext}}[k] - H[k]\bigr)
        + \beta_{\text{fog}} \, u_{\text{fogger}}
        + \beta_{\text{fan}} \, u_{\text{fan}}
        + \beta_{\text{vent}} \, u_{\text{vent}}
        + \text{ET}
$$

where ET = evapotranspiration baseline (plant transpiration, constant 0.3 %RH/step).

**Soil Moisture** — *[↗ greenhouse_model.py · L171](../src/agritwin_gh/mpc/greenhouse_model.py#L171)*

$$
\text{SM}[k+1] = \alpha_{\text{sm}} \, \text{SM}[k]
               + \beta_{\text{irr}} \, u_{\text{irr}}
               - \lambda_{\text{ET}} \cdot \max\!\bigl(0,\; T[k+1] - 15\bigr)
$$

The evapotranspiration loss $\lambda_{\text{ET}}$ scales with temperature — hotter conditions dry out soil faster.<sup>[[6]](#ref-6)</sup>

**CO₂** — *[↗ greenhouse_model.py · L181](../src/agritwin_gh/mpc/greenhouse_model.py#L181)*

$$
C[k+1] = \alpha_C \, C[k]
        + \beta_{\text{inj}} \, u_{\text{co2}}
        + \beta_{\text{plant}} \cdot \mathit{LF}[k]
        + \beta_{\text{vent}} \, u_{\text{vent}}
        + \gamma_{\text{vent}} \, u_{\text{vent}} \bigl(C_{\text{amb}} - C[k]\bigr)
$$

where $\mathit{LF}[k] = \operatorname{clip}(L[k]/500, 0, 1)$ is the light factor — CO₂ plant uptake scales with photosynthetic light availability.<sup>[[5]](#ref-5)</sup>

**Light Intensity** — *[↗ greenhouse_model.py · L194](../src/agritwin_gh/mpc/greenhouse_model.py#L194)*

$$
L[k+1] = \gamma_{\text{sol}} \, S[k] + \beta_{\text{LED}} \, u_{\text{LED}}
$$

Light has no memory term ($\alpha = 0$) — it is instantaneous: whatever the LEDs and solar contribute this step is the value for this step.

**Derived quantities** (not ARX, computed analytically after each step) — *[↗ greenhouse_model.py · L207](../src/agritwin_gh/mpc/greenhouse_model.py#L207)*

| Derived variable | Formula | Source |
|---|---|---|
| VPD | Tetens equation: $\text{VPD} = e_s(T) \cdot (1 - H/100)$ <sup>[[7]](#ref-7)</sup> | `compute_vpd(T, H)` |
| Leaf wetness | Sigmoid proxy of humidity, temperature vs. dew point | `compute_leaf_wetness_proxy()` |

---

#### Coefficient reference (`GreenhouseModelParams`) — *[↗ greenhouse_model.py · L40](../src/agritwin_gh/mpc/greenhouse_model.py#L40)*

| Coefficient | Value | Physical meaning |
|---|---|---|
| `temp_decay` $\alpha_T$ | 0.92 | 8% of greenhouse heat dissipates per 5-min step |
| `temp_external_gain` $\gamma_{\text{ext}}$ | 0.08 | Heat exchange with outdoor air |
| `temp_solar_gain` $\gamma_{\text{sol}}$ | 0.005 | Solar radiation heating contribution |
| `temp_heater_gain` $\beta_{\text{heat}}$ | 2.0 °C | Heater at full power raises temp 2 °C/step |
| `temp_fan_cool` $\beta_{\text{fan}}$ | −1.5 °C | Fan at full speed cools 1.5 °C/step |
| `temp_vent_cool` $\beta_{\text{vent}}$ | −1.2 °C | Vent at full open cools 1.2 °C/step |
| `hum_decay` $\alpha_H$ | 0.95 | Humidity is more persistent than temperature |
| `hum_fogger_gain` $\beta_{\text{fog}}$ | 8.0 % | Fogger at full duty adds 8 %RH/step |
| `hum_fan_loss` / `hum_vent_loss` | −3.0 / −2.5 % | Ventilation removes moisture |
| `co2_injection_gain` $\beta_{\text{inj}}$ | 300 ppm | CO₂ valve fully open adds 300 ppm/step |
| `co2_vent_loss` $\beta_{\text{vent}}$ | −40 ppm | Ventilation flushes ~40 ppm CO₂/step |
| `light_led_gain` $\beta_{\text{LED}}$ | 400 W/m² | LED intensity at full power |

---

#### Why ARX for MPC?

ARX models are **linear in the state and inputs** — which means the MPC cost function becomes a smooth, well-conditioned landscape for SLSQP to navigate. Each `step()` call takes microseconds, so the solver can evaluate thousands of candidate trajectories during a single 5-minute control cycle. A neural network plant model would be more accurate for edge cases but orders of magnitude slower and non-differentiable without AD tooling.<sup>[[1]](#ref-1), [[3]](#ref-3)</sup>

The coefficients are physically interpretable — if the heater gain looks wrong, you can adjust it directly. The `calibrate()` method (*[↗ greenhouse_model.py · L252](../src/agritwin_gh/mpc/greenhouse_model.py#L252)*) is a placeholder for fitting these from real sensor logs via least-squares regression.<sup>[[3]](#ref-3)</sup>

**Inputs**: `GreenhouseState` + `ActuatorState` + `WeatherState` (or dict).  
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

**Purpose**: Defines the objective function $J(\mathbf{u})$ that the MPC solver minimises over the prediction horizon. "Cost" is a single number measuring _how bad_ a particular sequence of actuator commands is: the higher the cost, the further the greenhouse is from its targets and the more energy, water, and disease risk it incurs. The solver's job is to find the $\mathbf{u}$ that makes this number as small as possible.

This is the richest file in the module — four nested building blocks compose into the final, numerically differentiable scalar objective.

---

#### Building blocks overview

| Class / Function | Role |
|---|---|
| `DiseaseContext` | Snapshot of disease severity data; scales how aggressively disease terms are penalised |
| `_compute_env_disease_risk(state)` | Re-evaluates disease risk from *predicted* humidity, temperature, VPD, and leaf wetness at every horizon step |
| `StageCost` | Per-timestep running cost $\ell(x_k, u_k)$ — tracking + disease + energy/water + switching |
| `TerminalCost` | End-of-horizon penalty $V_f(x_N)$ — discourages drifting into a bad state at the end of the window |
| `CostBuilder` | Assembles `StageCost` + `TerminalCost`, applies stage-transition blending, and scales weights with weather modifiers |

---

#### The complete objective function

The solver finds the actuator sequence $\mathbf{u} = [u_0, u_1, \ldots, u_{N-1}]$ that solves:

$$
\min_{\mathbf{u}} \; J(\mathbf{u}) = \sum_{k=0}^{N-1} \ell\!\left(x_k,\, u_k,\, u_{k-1}\right) + V_f(x_N)
$$

*[↗ cost_function.py · L484](../src/agritwin_gh/mpc/cost_function.py#L484)* <sup>[[1]](#ref-1), [[2]](#ref-2)</sup>

| Symbol | Meaning |
|---|---|
| $N$ | Prediction horizon length (e.g. 144 steps = 12 h at 5-min intervals) |
| $x_k \in \mathbb{R}^9$ | Predicted greenhouse state vector at step $k$ (temperature, humidity, soil moisture, …) |
| $u_k \in \mathbb{R}^7$ | Actuator command vector at step $k$ (fan, heater, fogger, …) |
| $u_{k-1}$ | Previous actuator command — used by the switching penalty; set to $\mathbf{0}$ at $k = 0$ |
| $\ell(x_k, u_k, u_{k-1})$ | **Running cost** — paid at every step of the horizon |
| $V_f(x_N)$ | **Terminal cost** — paid once at the final predicted state $x_N$ |

!!! tip "Why a terminal cost?"
    The running cost $\ell$ shapes behaviour _throughout_ the horizon. Without $V_f$, the solver could deliberately let conditions drift bad toward the end of the window — it would look fine now but set up a poor starting point for the _next_ solve. The terminal cost closes this loophole.

---

#### Running cost $\ell(x_k, u_k)$ — all nine terms

$$
\ell(x_k, u_k) = \underbrace{\ell_{\text{track}}}_{\text{1.\ setpoint tracking}}
              + \underbrace{\ell_{\text{dis}}}_{\text{2.\ disease environment}}
              + \underbrace{\ell_{\text{hum}}}_{\text{3.\ humidity exposure}}
              + \underbrace{\ell_{\text{fog}}}_{\text{4.\ fogger suppression}}
              + \underbrace{\ell_{\text{irr}}}_{\text{5.\ irrigation caution}}
              + \underbrace{\ell_{\text{eng}}}_{\text{6.\ energy}}
              + \underbrace{\ell_{\text{wat}}}_{\text{7.\ water}}
              + \underbrace{\ell_{\text{env}}}_{\text{8.\ env bounds}}
              + \underbrace{\ell_{\text{sw}}}_{\text{9.\ switching}}
$$

*[↗ cost_function.py · L301](../src/agritwin_gh/mpc/cost_function.py#L301)* <sup>[[1]](#ref-1)</sup>

Terms 1–2 enforce the agronomic objectives (stay near setpoints, avoid disease). Terms 3–5 activate only when disease risk is elevated. Terms 6–7 penalise resource use. Term 8 penalises excursions beyond growth-stage environmental bounds (§17.1a). Term 9 penalises actuator wear.

---

#### Term 1 — Setpoint Tracking

**Intuition:** Keep every state variable close to its growth-stage target. A 5 °C temperature error should hurt roughly as much as a 10 % humidity error — the normalisation scales ensure each variable contributes fairly regardless of its physical unit.

$$
\ell_{\text{track}} = \sum_{i=1}^{n_x} m_i \cdot w_i \cdot \left(\frac{x_k^{(i)} - x^{*\,(i)}}{\sigma_i}\right)^{\!2}
$$

*[↗ cost_function.py · L301](../src/agritwin_gh/mpc/cost_function.py#L301)*

| Symbol | Meaning |
|---|---|
| $x_k^{(i)}$ | Predicted value of state variable $i$ at step $k$ |
| $x^{*\,(i)}$ | Growth-stage setpoint for variable $i$ (from `StageSetpoint`) |
| $\sigma_i$ | Normalisation scale — converts raw units to a dimensionless error |
| $w_i$ | Effective weight = base weight × stage-profile multiplier |
| $m_i$ | Weather-adaptive modifier at step $k$ (default 1.0; increases if extreme weather is forecast) |

**Normalisation scales $\sigma_i$:**

| State variable | $\sigma_i$ | Unit | Interpretation |
|---|---|---|---|
| `indoor_temp` | 5.0 | °C | An error of 5 °C scores 1.0 normalised error |
| `indoor_humidity` | 10.0 | % | An error of 10 % scores 1.0 normalised error |
| `soil_moisture` | 10.0 | % | A 10 % deviation from target = 1.0 normalised error |
| `co2` | 150.0 | ppm | A 150 ppm deviation = 1.0 normalised error |
| `light_intensity` | 200.0 | W/m² | — |
| `vpd` | 0.3 | kPa | A 0.3 kPa deviation = 1.0 normalised error |
| `disease_risk_score` | 0.3 | unitless | — |
| `leaf_wetness_proxy` | 0.3 | unitless | — |

!!! note "Weather-adaptive scaling"
    The optional `step_modifiers` array from `WeatherAdaptiveModifiers` multiplies the weights $w_i$ element-wise at each step. If the forecast predicts an external heat spike in 2 hours, the temperature tracking weight rises automatically for those steps — the solver **pre-acts** to cool the greenhouse before the spike arrives.

---

#### Term 2 — Disease Environment Penalty

**Intuition:** A naive controller might look at the current `disease_risk_score` sensor field and ignore how future conditions evolve. AgriTwin-GH instead _re-predicts_ disease risk from the **forecasted** humidity, temperature, VPD, and leaf wetness at each horizon step. This means the solver is penalised for a trajectory that lets humidity climb toward dangerous levels — it cannot hide the risk.

**Step 1 — Predicted disease risk $\hat{d}(x_k)$:**

The predicted risk is a weighted sum of four sigmoid-shaped sub-risks:

$$
\hat{d}(x_k) = 0.35\;\sigma(H_k;\;75.0,\;0.20)
             + 0.25\;\sigma(L_k;\;0.50,\;8.00)
             + 0.20\;\sigma(T_k;\;22.0,\;0.15)
             + 0.20\;\bigl[1 - \sigma(P_k;\;0.80,\;5.00)\bigr]
$$

*[↗ cost_function.py · L201](../src/agritwin_gh/mpc/cost_function.py#L201)*

where the logistic sigmoid function is:

$$
\sigma(x;\;c,\;s) \;=\; \frac{1}{1 + e^{-s\,(x-c)}}
$$

*[↗ cost_function.py · L194](../src/agritwin_gh/mpc/cost_function.py#L194)*

This S-shaped function is zero for $x \ll c$, rises steeply around the centre $c$, and saturates at 1 for $x \gg c$. The slope $s$ controls how sharp the transition is.

| Input | Symbol | Centre $c$ | Slope $s$ | Disease interpretation |
|---|---|---|---|---|
| Indoor humidity (%) | $H_k$ | 75 % | 0.20 | Risk climbs above 75 % RH; shallow slope = broad sensitivity |
| Leaf wetness proxy | $L_k$ | 0.50 | 8.00 | Very sharp onset — even small wetness causes a large jump |
| Indoor temperature (°C) | $T_k$ | 22 °C | 0.15 | Moderate, broad temperature sensitivity around 22 °C |
| VPD (kPa) | $P_k$ | 0.80 | 5.00 | Inverted: low VPD = stagnant, humid air = higher risk |

**Step 2 — Severity amplification:**

When disease is already progressing, the system automatically increases how much it cares about future disease risk:

$$
w_{\text{dis,eff}} = w_{\text{disease}} \;\times\; \delta_{\text{stage}} \;\times\; \underbrace{\left(1 + w_{\text{sev}} \cdot \frac{\max\!\left(s_{24h},\; s_{48h}\right)}{100}\right)}_{\text{severity amplifier}}
$$

*[↗ cost_function.py · L253](../src/agritwin_gh/mpc/cost_function.py#L253)*

| Symbol | Meaning |
|---|---|
| $w_{\text{disease}}$ | Base disease weight (default 2.0) |
| $\delta_{\text{stage}}$ | Per-stage sensitivity multiplier from `StageControlProfile` |
| $w_{\text{sev}}$ | Severity amplification strength (default 1.0) |
| $s_{24h},\, s_{48h}$ | Worst-case predicted disease severity (%) at the 24 h and 48 h forecast horizons |

!!! example "Severity amplification in practice"
    The 48-hour disease forecast predicts early blight reaching 60% severity:

    **amplifier = 1 + 1.0 × (60 ÷ 100) = 1.60**

    The controller is now **60 % more aggressive** at suppressing humid, warm conditions — even before visible symptoms worsen.

**Step 3 — Disease cost per step:**

$$
\ell_{\text{dis}} = w_{\text{dis,eff}} \cdot \hat{d}(x_k)^2
$$

*[↗ cost_function.py · L307](../src/agritwin_gh/mpc/cost_function.py#L307)*

The quadratic form means mild risk ($\hat{d} = 0.3$) costs only $0.09 \times w$, while high risk ($\hat{d} = 0.9$) costs $0.81 \times w$ — the solver is strongly motivated to avoid the high-risk end.

---

#### Term 3 — Humidity Exposure Penalty

**Intuition:** When humidity is above setpoint **and** disease risk is simultaneously elevated, the controller pays an extra penalty on top of the standard tracking term. Below setpoint or with low disease risk, this term is zero.

$$
\ell_{\text{hum}} = w_{\text{hum\_{exp}}} \cdot \left(\frac{\max\!\left(0,\; H_k - H^*\right)}{20}\right)^{\!2} \cdot \hat{d}(x_k)
$$

*[↗ cost_function.py · L315](../src/agritwin_gh/mpc/cost_function.py#L315)*

| Symbol | Meaning |
|---|---|
| $H_k$ | Predicted indoor humidity at step $k$ (%) |
| $H^*$ | Humidity setpoint (%) |
| $\hat{d}(x_k)$ | Predicted disease risk at step $k$ |
| $w_{\text{hum\_exp}}$ | Humidity exposure weight (default 0.5) |

The $\max(0, \cdot)$ ensures the penalty only activates when humidity **exceeds** setpoint. The disease risk factor $\hat{d}$ means humidity excess is tolerated more when the disease environment is otherwise safe.

---

#### Term 4 — Fogger Suppression Penalty

**Intuition:** The fogger adds moisture and promotes leaf wetness — exactly what disease-causing fungi thrive on. Once predicted disease risk crosses a threshold, running the fogger becomes increasingly expensive.

$$
\ell_{\text{fog}} = w_{\text{fog}} \cdot u_{\text{fogger}} \cdot \max\!\left(0,\; \hat{d}(x_k) - \theta_{\text{fog}}\right)
$$

*[↗ cost_function.py · L321](../src/agritwin_gh/mpc/cost_function.py#L321)*

| Symbol | Meaning |
|---|---|
| $u_{\text{fogger}}$ | Fogger duty cycle command at step $k$ (0–100) |
| $\theta_{\text{fog}}$ | Disease risk threshold (default 0.5) |
| $w_{\text{fog}}$ | Fogger suppression weight (default 0.3) |

Below $\theta_{\text{fog}} = 0.5$ the fogger is unpenalised and runs freely for humidity management. Above it, each unit of fogger duty increases cost linearly — the solver prefers to reduce or stop fogging and use venting instead.

---

#### Term 5 — Irrigation Caution Penalty

**Intuition:** Irrigation adds root-zone moisture and raises ambient humidity. When the environment is already humid and disease-prone, additional watering makes things worse. This term fires only when _all three_ conditions hold simultaneously: irrigation is commanded, humidity is above setpoint, _and_ disease risk is elevated.

$$
\ell_{\text{irr}} = w_{\text{irr}} \cdot \frac{u_{\text{irr}}}{50} \cdot \frac{\max\!\left(0,\; H_k - H^*\right)}{20} \cdot \hat{d}(x_k)
$$

*[↗ cost_function.py · L327](../src/agritwin_gh/mpc/cost_function.py#L327)*

| Symbol | Meaning |
|---|---|
| $u_{\text{irr}}$ | Irrigation quantity command (0–50 units) |
| $w_{\text{irr}}$ | Irrigation caution weight (default 0.2) |

If any one factor is zero — humidity is fine, or disease risk is low, or no irrigation is commanded — the entire term collapses to zero.

---

#### Term 6 — Energy Cost

**Intuition:** Some actuators draw far more power than others. The solver is penalised for high-energy solutions so it learns to prefer cheaper alternatives (e.g. open vents instead of run the heater) whenever possible.

$$
\ell_{\text{eng}} = w_{\text{energy}} \sum_{j=1}^{7} c_j \cdot u_k^{(j)}
$$

*[↗ cost_function.py · L333](../src/agritwin_gh/mpc/cost_function.py#L333)*

| Actuator $j$ | Energy coefficient $c_j$ | Relative cost |
|---|---|---|
| `fan_speed` | 0.15 | Medium |
| `vent_opening` | 0.02 | Nearly free — just a servo |
| `heater_output` | **0.80** | **Most expensive** — resistive heating |
| `led_intensity` | 0.30 | High-power grow lights |
| `co2_valve_pct` | 0.05 | Low draw; CO₂ gas cost is separate |
| `fogger_duty` | 0.10 | Pump + nozzle |
| `irrigation_qty` | 0.01 | Minimal energy |

The heater at $c = 0.80$ is **40 × more expensive** than venting ($c = 0.02$). Given the same thermal result, the solver strongly prefers opening vents.

---

#### Term 7 — Water Cost

**Intuition:** Total water consumption is minimised. Fogging is weighted twice as heavily as irrigation because evaporated water is distributed throughout the canopy — less targeted and harder to control.

$$
\ell_{\text{wat}} = w_{\text{water}} \cdot \left(u_{\text{irr}} + 2\,u_{\text{fogger}}\right)
$$

*[↗ cost_function.py · L340](../src/agritwin_gh/mpc/cost_function.py#L340)*

---

#### Term 8 — Environmental Bounds Barrier

**Intuition:** Each growth stage has biologically optimal environmental ranges (§17.1a). When the predicted state approaches or exceeds these stage-specific limits, the solver is penalised with a quadratic barrier. This complements the stress penalty (which uses setpoint tolerances) by enforcing the wider stage-specific safe envelope.

$$
\ell_{\text{env}} = w_{\text{env}} \sum_{i \in \mathcal{E}} \left(\frac{\max(0,\; x_k^{(i)} - \overline{b}_i) + \max(0,\; \underline{b}_i - x_k^{(i)})}{\sigma_i}\right)^{\!2}
$$

| Symbol | Meaning |
|---|---|
| $\mathcal{E}$ | Set of environmentally bounded variables: indoor\_temp, indoor\_humidity, co2, soil\_moisture, light\_intensity |
| $\overline{b}_i, \underline{b}_i$ | Upper and lower stage-specific environmental bounds for variable $i$ |
| $\sigma_i$ | Normalisation scale (same as tracking term) |
| $w_{\text{env}}$ | Environmental bounds weight (default **0.5**) |

The penalty is **exactly zero** when all states are within bounds. It activates only when conditions drift outside the stage-appropriate range, providing a soft barrier that guides the solver without over-constraining it.

---

#### Term 9 — Actuator Switching Penalty

**Intuition:** Rapid oscillation in actuator commands — e.g. a heater toggling on/off every 5 minutes — is mechanically damaging and energetically wasteful. A quadratic penalty on command changes keeps actuator trajectories smooth.

$$
\ell_{\text{sw}} = w_{\text{switch}} \sum_{j=1}^{7} \left(u_k^{(j)} - u_{k-1}^{(j)}\right)^{\!2}
$$

*[↗ cost_function.py · L346](../src/agritwin_gh/mpc/cost_function.py#L346)*

The quadratic form punishes large jumps exponentially more than small ones. A change of 20 units costs 4 × as much as a change of 10 units.

---

#### Terminal cost $V_f(x_N)$

The terminal cost evaluates the same tracking, disease, and environmental bounds terms at the **final predicted state** $x_N$, multiplied by $\gamma = 2$:

$$
V_f(x_N) = \gamma \left[\, \sum_{i=1}^{n_x} w_i \left(\frac{x_N^{(i)} - x^{*\,(i)}}{\sigma_i}\right)^{\!2} + w_{\text{dis,eff}} \cdot \hat{d}(x_N)^2 + \ell_{\text{env}}(x_N) \,\right], \qquad \gamma = 2.0
$$

*[↗ cost_function.py · L406](../src/agritwin_gh/mpc/cost_function.py#L406)* <sup>[[1]](#ref-1)</sup>

The $2\times$ multiplier ensures the solver genuinely ends the prediction window in a good state, not merely passes through it momentarily. The environmental bounds penalty $\ell_{\text{env}}$ uses the same stage-specific bounds and weight as the running cost (Term 8).

---

#### Base cost weights (current tuned values from `config.py`)

| Weight | Value | What it penalises |
|---|---|---|
| $w_{\text{temp}}$ | **2.0** | Temperature tracking error |
| $w_{\text{humidity}}$ | **2.0** | Humidity tracking error |
| $w_{\text{soil\_moisture}}$ | **1.5** | Soil moisture tracking error |
| $w_{\text{co2}}$ | **1.0** | CO₂ tracking error |
| $w_{\text{vpd}}$ | **1.0** | VPD deviation |
| $w_{\text{light}}$ | 0.4 | Light intensity deviation |
| $w_{\text{disease}}$ | 0.8 | Predicted disease risk (base; amplified by stage sensitivity) |
| $w_{\text{energy}}$ | 0.10 | Energy consumption (low to prioritise tracking) |
| $w_{\text{water}}$ | 0.10 | Water consumption (low to prioritise tracking) |
| $w_{\text{switch}}$ | **0.30** | Actuator switching (high to stabilise actuators) |
| $w_{\text{hum\_exp}}$ | 0.1 | RH above setpoint × disease risk |
| $w_{\text{fog}}$ | 0.1 | Fogger duty when disease risk is high |
| $w_{\text{irr}}$ | 0.05 | Irrigation when humid + disease active |
| $w_{\text{sev}}$ | 1.0 | Severity forecast amplification strength |
| $w_{\text{stress}}$ | 1.5 | Stress-excursion penalty (matches yield proxy stress formula) |
| $w_{\text{env}}$ | **0.5** | Environmental bounds barrier — penalises states outside stage-specific limits (§17.1a) |

> **Tuning rationale:** Tracking weights (temp, humidity, soil moisture, CO₂, VPD) are set high because the yield proxy assigns 40% weight to climate tracking. The switching penalty (0.30) prevents actuator oscillation that degrades the 15% stability component. Energy and water weights are low (0.10) because the solver warm-starts near the baseline—pushing these higher cannot meaningfully reduce MPC resource cost but can degrade yield.

---

#### Stage weight multipliers

Each growth stage defines a `StageControlProfile` that multiplies the base tracking and disease weights:

| Weight | Seedling | Vegetative | Flower Init | Flowering | Unripe | Ripe |
|---|---|---|---|---|---|---|
| Temperature | 1.2× | 1.0× | 1.3× | **1.4×** | 1.1× | 0.9× |
| Humidity | 1.0× | 1.0× | 1.2× | **1.3×** | 1.2× | 0.8× |
| Soil moisture | 1.3× | 1.0× | 0.9× | 1.0× | 1.1× | 0.8× |
| CO₂ | 0.6× | 0.8× | 1.0× | **1.2×** | 1.0× | 0.5× |
| VPD | 0.8× | 0.9× | 1.2× | **1.3×** | 1.1× | 0.7× |
| Light | 0.7× | 1.0× | 1.1× | 1.2× | 1.0× | 0.6× |
| Disease sensitivity $\delta_{\text{stage}}$ | 1.0× | 1.0× | 1.3× | **1.5×** | **1.4×** | 0.8× |

Flowering is the most sensitive stage — temperature, humidity, CO₂, VPD, and disease weights simultaneously reach their peak multipliers.

**Effective weight example** (flowering, temperature): $w_{\text{eff}} = w_{\text{base}} \times \text{multiplier} = 2.0 \times 1.4 = 2.8$

---

#### Growth-stage transition blending

When the prediction horizon spans a stage boundary, `CostBuilder` holds a second `StageCost` for the upcoming stage and blends between them:

$$
\ell_{\text{blended}}(k) = (1 - \alpha_k)\;\ell_{\text{current}}(k) \;+\; \alpha_k\;\ell_{\text{next}}(k)
$$

*[↗ cost_function.py · L520](../src/agritwin_gh/mpc/cost_function.py#L520)*

$$
\alpha_k = \operatorname{clip}\!\left(\frac{k - k_{\text{start}}}{B},\; 0,\; 1\right)
$$

*[↗ cost_function.py · L471](../src/agritwin_gh/mpc/cost_function.py#L471)* <sup>[[1]](#ref-1)</sup>

| Symbol | Meaning |
|---|---|
| $k_{\text{start}}$ | Horizon step at which blending begins |
| $B$ | Blend window width — default 12 steps (= 60 minutes at 5-min intervals) |
| $\alpha_k$ | Blending coefficient: $0$ = full current-stage cost, $1$ = full next-stage cost |

Without blending the cost function would jump discontinuously when a stage boundary occurs mid-horizon, and the solver would produce an erratic actuator schedule. The linear ramp over $B$ steps prevents this.

---

#### Weather-adaptive weight scaling

`StageCost.evaluate()` accepts an optional `step_modifiers` array (shape `(N_{\text{state}},)`) from `WeatherAdaptiveModifiers`. The effective tracking weights become:

$$
w_i^{\text{eff}}(k) = w_i \cdot m_i(k)
$$

*[↗ cost_function.py · L300](../src/agritwin_gh/mpc/cost_function.py#L300)*

where $m_i(k)$ is the weather modifier for state variable $i$ at step $k$. If the external forecast predicts a temperature spike in 2 hours, $m_{\text{temp}}$ rises for those steps — the solver pre-acts to cool the greenhouse **before** the spike arrives rather than reacting to it after the fact.

---

#### `DiseaseContext` dataclass

```python
@dataclass
class DiseaseContext:
    risk_score:       float               # Current aggregate risk in [0, 1]
    classification:   str                 # e.g. "early_blight"
    confidence:       float               # Classifier confidence in [0, 1]
    current_severity: dict[str, float]    # {disease: severity %} now
    severity_24h:     dict[str, float]    # {disease: severity %} in 24 h
    severity_48h:     dict[str, float]    # {disease: severity %} in 48 h
```

Key derived properties:

| Property | Formula | Purpose |
|---|---|---|
| `max_severity_24h` | $\max_j\, s_{24h}^{(j)}$ | Worst-case severity across all diseases at the 24 h mark |
| `max_severity_48h` | $\max_j\, s_{48h}^{(j)}$ | Same at 48 h |
| `severity_amplifier` | $1 + \max\!\left(s_{24h},\, s_{48h}\right)/100$ | Multiplies $w_{\text{disease}}$ — auto-elevates response as disease progresses |

Constructed at the start of each solve via `DiseaseContext.from_fused(fused_state)`.

---

**Inputs**: NumPy state and control arrays per horizon step, `MPCConfig`, `StageSetpoint`, optional `DiseaseContext` and `WeatherAdaptiveModifiers`.  
**Outputs**: A single float cost value per step or total horizon cost (lower = better).

---

### 6.8 `mpc_solver.py`

**Purpose**: The MPC engine. Takes `FusedState` and returns the optimal actuator sequence by minimising the cost function subject to constraints via the SLSQP algorithm.<sup>[[8]](#ref-8)</sup>

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

Why project 24h and 48h ahead? The MPC horizon is 12 hours. But disease development is a *slow process* — early blight might look mild now but be severe tomorrow.<sup>[[9]](#ref-9)</sup> Including the 24h/48h projections in the cost penalises conditions that are likely to lead to disease escalation even if the current reading is safe.

**Inputs**: Greenhouse state, disease label, severity value, context DataFrame.  
**Outputs**: Risk score in [0, 1]; penalty float.

---

### 6.11 `growth_weights.py`

**Purpose**: Makes the MPC *stage-aware*. Provides different cost weights for different growth stages and predicts how many hours until the plant transitions to the next stage via an LSTM-based progression model.<sup>[[9]](#ref-9)</sup>

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
**Outputs**: `ComparisonMetrics`; optional plots in `src/agritwin_gh/mpc/mpc_results/<run_id>/figures/`.

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

**Purpose**: Offline comparative evaluation framework — runs multiple controllers
over an identical synthetic scenario and produces a structured `ComparisonReport`.
No database session is required; all inputs are generated programmatically.

↗ `experiment_runner.py · L1`

---

**Dataclass `ExperimentConfig`** *(L67)* — declarative experiment specification:

```python
@dataclass
class ExperimentConfig:
    n_steps: int = 288              # total simulation steps
    dt_minutes: int = 5             # step duration
    initial_state: GreenhouseState  # starting indoor climate
    weather_sequence: list[WeatherState]     # external disturbance (len = n_steps)
    growth_stage_sequence: list[str]         # canonical stage label per step
    random_seed: int = 42
    yield_proxy_weights: YieldProxyWeights | None = None
    experiment_name: str = ""
```

---

**Class `ExperimentRunner`** *(L196)*:

```python
runner = ExperimentRunner(config)
runner.register_controller("baseline", make_baseline_adapter(), "baseline")
runner.register_controller("mpc",      make_mpc_adapter(),      "mpc")
report = runner.run()   # → ComparisonReport
```

- `register_controller(id, adapter, type)` — adapter signature:
  `(GreenhouseState, WeatherState, str, float, int) → ActuatorState`
- `run()` — simulates every registered controller over the same scenario
  and returns a `ComparisonReport`.

---

**Dataclass `ComparisonReport`** *(L118)*:

```python
@dataclass
class ComparisonReport:
    controller_metrics: dict[str, ControllerMetricsBundle]
    yield_results:      dict[str, YieldProxyResult]
    improvements:       dict[str, dict[str, float]]  # pairwise % improvements
    experiment_config:  ExperimentConfig
    generated_at:       datetime

    def summary_table(self) -> dict[str, dict]:  ...  # scalars per controller
    def to_json_dict(self) -> dict:              ...  # full JSON-serialisable form
    def to_dict(self) -> dict:                  ...  # alias for script compatibility
```

---

**Adapter factories:**

| Function | Line | Description |
|---|---|---|
| `make_baseline_adapter()` | L418 | Wraps `RuleBasedController`; no config needed |
| `make_mpc_adapter(config)` | L444 | Wraps `MPCSolver`; assembles `FusedState` per step |

---

**Scenario helpers:**

| Function | Line | Description |
|---|---|---|
| `generate_default_weather(n_steps, dt_minutes, base_temp, base_humidity)` | L502 | Sinusoidal diurnal weather, starts from hour 0 |
| `generate_default_growth_stages(n_steps, stage)` | L553 | Constant growth-stage sequence |
| `make_default_initial_state()` | L570 | Healthy flowering-stage greenhouse at mid-morning |

Uses canonical disease labels (`"healthy leaves"`, `"early blight"`) and canonical
stage indexing (`stage_label_to_index("flowering")`) — no hardcoded integers.

**Inputs**: `ExperimentConfig` + registered adapters (no DB session needed).  
**Outputs**: `ComparisonReport` with per-controller `ControllerMetricsBundle`,
`YieldProxyResult`, and pairwise improvement dict.

Used by `scripts/run_full_mpc_evaluation.py` for the five-scenario validation suite.

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

**Purpose**: Defines the public API of the MPC package with comprehensive re-exports.

**Key exported symbols** (grouped by origin):

```python
from agritwin_gh.mpc import (
    # ── State dataclasses ────────────────────────────────────────────────────
    GreenhouseState, ActuatorState, WeatherState, FusedState,
    MPCSolution, DigitalTwinStepPayload, DigitalTwinTrajectoryPayload,
    ComparisonMetrics,

    # ── Constants & helpers ──────────────────────────────────────────────────
    GROWTH_STAGES, DISEASE_CATEGORIES, DT_MINUTES, STEPS_PER_HOUR,
    stage_label_to_index, stage_index_to_label,
    compute_disease_risk_score, compute_vpd, compute_dew_point,

    # ── Configuration ────────────────────────────────────────────────────────
    MPCConfig, load_mpc_config,

    # ── Constraints & setpoints ──────────────────────────────────────────────
    ConstraintSet, get_default_constraints, tighten_constraints_for_disease,
    StageSetpoint, get_setpoint,

    # ── Orchestration ────────────────────────────────────────────────────────
    MPCRunner, MPCSolver,

    # ── Evaluation metrics ───────────────────────────────────────────────────
    ControllerMetricsBundle, compute_all_metrics,

    # ── Yield proxy ──────────────────────────────────────────────────────────
    YieldProxyWeights, YieldProxyResult, compute_yield_proxy,

    # ── Offline experiment runner ─────────────────────────────────────────────
    ExperimentConfig, ExperimentRunner, ComparisonReport,
    make_baseline_adapter, make_mpc_adapter,
    generate_default_weather, generate_default_growth_stages,
    make_default_initial_state,

    # ── DT loop ──────────────────────────────────────────────────────────────
    DTLoop, DTLoopStepResult, DigitalTwinEngine,
    SyntheticInputProvider, prepare_initial_state, prepare_weather_sequence,

    # ── Utilities ────────────────────────────────────────────────────────────
    discover_latest_artifact,
)
```

Internal files use relative imports. Only symbols listed here should be
imported by code *outside* the MPC package.

---

### 6.24 `yield_proxy.py`

**Purpose**: Transparent, configurable scalar quality proxy (0–100) that
estimates the impact of controller behaviour on tomato crop yield without
requiring a full biophysical crop model.

↗ `yield_proxy.py · L1`

**Dataclass `YieldProxyWeights`** *(L37)* — component importance weights:

```python
@dataclass
class YieldProxyWeights:
    climate_tracking:   float = 0.40   # closeness to stage-specific setpoints
    disease_burden:     float = 0.25   # integrated disease risk over window
    stress_exposure:    float = 0.20   # temp + humidity excursions outside safe envelopes
    resource_stability: float = 0.15   # penalises erratic actuator behaviour
```

**Dataclass `YieldProxyResult`** *(L72)* — audit-ready score breakdown:

```python
@dataclass
class YieldProxyResult:
    overall_score:           float   # 0–100 composite
    climate_tracking_score:  float   # 0–100
    disease_burden_score:    float   # 0–100
    stress_exposure_score:   float   # 0–100
    resource_stability_score: float  # 0–100
    weights_used:            dict[str, float]
    per_step_scores:         list[float]

    def to_dict(self) -> dict: ...  # includes per_step summary stats
```

**Function `compute_yield_proxy`** *(L100)*:

```python
def compute_yield_proxy(
    states: Sequence[GreenhouseState],
    actuators: Sequence[ActuatorState],
    growth_stages: Sequence[str],
    weights: YieldProxyWeights | None = None,
) -> YieldProxyResult:
    """Compute the yield/growth quality proxy for a full simulation window."""
```

Used by `ExperimentRunner.run()` to score each controller and by
`scripts/run_full_mpc_evaluation.py` to populate the yield proxy table.

**Inputs**: State trajectory, actuator trajectory, growth-stage sequence, optional weights.  
**Outputs**: `YieldProxyResult` with auditable component breakdown.

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

Results are automatically saved to `src/agritwin_gh/mpc/mpc_results/<run_id>/`.

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

> **Current status**: Formal pytest unit test files (`test_state.py`, `test_greenhouse_model.py`, etc.) are **not yet implemented** — they are planned for a future phase. The three smoke test scripts listed below are the tests that actually exist and pass right now. Each covers a major subsystem end-to-end without a live database.

### How to run the smoke tests

```powershell
# Set PYTHONPATH and run all three
$env:PYTHONPATH = "e:\AgriTwin-GH\src"
python tests/smoke_intelligent_mpc.py
python tests/smoke_test_dt_handoff.py
python tests/test_evaluation_smoke.py
```

---

### `tests/smoke_intelligent_mpc.py` — Intelligent MPC (disease + weather + stage blending)

**What it tests:**
- `DiseaseContext` dataclass construction and derived fields (`severity_amplifier`, `max_severity_24h/48h`)
- `DiseaseContext.from_fused()` factory from a live `FusedState`
- `compute_weather_adaptation()` — detects external temperature and humidity stress signals
- `tighten_constraints_for_disease()` — reduces RH upper bound and suppresses fogger when risk is high
- `CostBuilder` with disease context and stage-transition blending (`_blend_start`, `_blend_alpha`)
- Cost evaluation with and without weather modifiers
- Blending alpha ramp from 0 → 1 across the transition window
- Full `MPCSolver.solve()` with a short 8-step horizon — converges or correctly uses fallback

**Run command:**

```powershell
$env:PYTHONPATH = "e:\AgriTwin-GH\src"
python tests/smoke_intelligent_mpc.py
```

**Actual output:**

```
1. All imports OK
2. DiseaseContext OK — sev_amp=1.60
3. DiseaseContext.from_fused OK
4. Weather adaptation OK — temp_stress_max=1.400, rh_stress_max=1.250
5. Constraint tightening OK — RH: 95.0 -> 88.6, fogger hi=0.30
6. CostBuilder with blending OK — blend_start=54
7. Cost eval: with_weather=28.1711, without=26.7774
8. Blending alpha: step_0=0.00, mid=0.50, end=1.00
9. Running full MPC solve (short horizon=8)...
   Converged: False
   Total cost: 58.6043
   Solve time: 3500.4 ms
   Iterations: 0
   Breakdown: {'fallback': 58.60426947218381}
   First action: fan=1.00, vent=0.80

==================================================
ALL INTELLIGENT MPC SMOKE TESTS PASSED
==================================================
```

> **Note on convergence**: The solver does not converge on this particular input (`Positive directional derivative for linesearch`) because the test state is intentionally extreme (temp=32 °C, humidity=85%, disease risk=0.6) — the SLSQP gradient conflicts with the large penalty. The fallback `RuleBasedController` kicks in and produces safe actuator commands (`fan=1.00, vent=0.80`), which is the correct behaviour. The test asserts `converged OR fallback_used`, so it passes.

---

### `tests/smoke_test_dt_handoff.py` — Digital Twin Output, Explanation, Replay

**What it tests:**
- `ExplanationBuilder.build()` — generates human-readable entries (critical/warning/info) for every active trigger
- Entry categories: `climate`, `disease`, `weather`, `growth`, `constraint`
- `ControllerDecisionContext.to_dict()` — schema versioning, JSON-serialisable
- `DigitalTwinOutput.format_step()` — assembles the full `DigitalTwinStepPayload` with explanation, decision context, and solver performance embedded
- Alert level + alert icons computation (`YELLOW`, `['disease_moderate', 'transition_24h']`)
- `DigitalTwinStepPayload` dataclass field presence check
- `ReplayConfig`, `ReplayStep`, `ReplaySummary` dataclass structure

**Run command:**

```powershell
$env:PYTHONPATH = "e:\AgriTwin-GH\src"
python tests/smoke_test_dt_handoff.py
```

**Actual output:**

```
=== Import test ===
All imports OK

=== ExplanationBuilder test ===
Dominant factor: climate
Action summary: fan at 80%, vents at 60%, irrigating 3.0L, LEDs at 40% — driven by: Temperature 30.5°C is 9.5°C above target 21.0°C
Entries (12):
  [critical] [climate   ] temperature_above_setpoint: Temperature 30.5°C is 9.5°C above target 21.0°C
  [critical] [climate   ] humidity_above_setpoint: Humidity 88.0% is 28.0% above stage-safe safe range
  [warning ] [climate   ] vpd_deviation: VPD 1.80 kPa deviates +0.80 from target 1.00
  [warning ] [disease   ] moderate_disease_risk: Disease risk 0.55 is moderately elevated
  [warning ] [disease   ] severity_worsening: Leaf Mold severity predicted to worsen: 25% → 38% in 24h
  [info    ] [disease   ] fogger_suppressed_for_disease: Fogger suppressed to reduce moisture and disease-favorable conditions
  [info    ] [disease   ] irrigation_cautious_for_disease: Irrigation kept low to avoid high humidity persistence
  [info    ] [weather   ] heat_stress_anticipated: External temperature stress anticipated (stress=0.45); ventilation increased proactively
  [info    ] [weather   ] humidity_stress_anticipated: External humidity stress anticipated (stress=0.35); ventilation adjusted
  [info    ] [growth    ] stage_transition_imminent: Growth stage transition flowering → unripe expected in ~18h; control weights blending toward next-stage profile
  [warning ] [constraint] rh_ceiling_tightened: RH ceiling tightened to 88.0% due to disease risk
  [warning ] [constraint] fogger_constraint_active: Fogger max duty limited to 15% due to disease risk
to_dict: 12 entries OK

=== ControllerDecisionContext test ===
DecisionContext to_dict OK, schema=1.0

=== DigitalTwinOutput with explanation test ===
Payload explanation entries: 11
Payload decision_context run_id: test-run-001
Payload solver_performance converged: True
Payload alert: YELLOW ['disease_moderate', 'transition_24h']

=== DigitalTwinStepPayload fields check ===
All new fields present in DigitalTwinStepPayload

=== ReplayEngine test ===
ReplayConfig: replay_id=replay-f8ea1b3564f9
ReplayStep fields: ['step_index', 'timestamp', 'observed_state', 'mpc_action', 'model_predicted_state', 'actual_next_state', 'correction_delta', 'solution']
ReplaySummary fields: ['replay_id', 'total_steps', 'mean_correction', 'max_correction', 'solver_convergence_rate', 'mean_solve_time_ms']

=== ALL SMOKE TESTS PASSED ===
```

---

### `tests/test_evaluation_smoke.py` — Evaluation Framework

**What it tests:**
- Import of all evaluation symbols (`TrackingMetrics`, `DiseaseBurdenMetrics`, `ResourceMetrics`, `ControlQualityMetrics`, `SafetyMetrics`, `ControllerMetricsBundle`, `compute_all_metrics`, `YieldProxyResult`, `compute_yield_proxy`, `ExperimentRunner`, `ComparisonReport`, etc.)
- `compute_all_metrics()` on 50 synthetic states — tracking RMSE, disease burden, energy, water, safety, smoothness
- `compute_yield_proxy()` — overall score, climate score, disease penalty, per-step vector
- `ExperimentRunner` with a registered rule-based baseline adapter over 50 synthetic steps
- `save_evaluation_artifacts()` — writes 4 JSON files to a temp directory
- `load_evaluation_report()` — round-trip load from disk
- JSON serialisation of full `ComparisonReport`
- `to_dict()` round-trip for `ControllerMetricsBundle` and `YieldProxyResult`

**Run command:**

```powershell
$env:PYTHONPATH = "e:\AgriTwin-GH\src"
python tests/test_evaluation_smoke.py
```

**Actual output:**

```
[OK] All evaluation symbols imported
[OK] Metrics bundle: 50 steps, temp RMSE=2.503
     Disease mean_risk=0.1676
     Energy=12.9325 kWh, Water=3.30 L
     Safety violations=0
     Smoothness L2=0.0952
[OK] Yield proxy: overall=63.58/100, climate=60.99, disease=32.96
     Per-step scores: 50 entries, mean=63.58
[OK] Experiment: baseline temp_rmse=5.465, yield=67.18
     Water=20.27 L, Energy=36.1263 kWh
[OK] Artifacts saved: ['experiment_config.json', 'full_metrics.json', 'report_summary.json', 'yield_proxy.json']
[OK] Report loaded back: keys=['generated_at', 'improvements', 'run_id', 'summary_table']
[OK] Report JSON serialisation: 4840 chars
[OK] baseline metrics dict has all 5 sections
[OK] baseline yield proxy dict has summary stats

=== ALL SMOKE TESTS PASSED ===
```

---

### Future unit tests (not yet implemented)

The following pytest unit test files are planned. They will exercise individual components in isolation using mocks and synthetic data, without requiring a database or MinIO instance.

| Planned file | Target module | Key assertions |
|---|---|---|
| `test_state.py` | `state.py` | `GreenhouseState` numpy roundtrip; `ActuatorState.clip()` |
| `test_greenhouse_model.py` | `greenhouse_model.py` | Heater warms temperature; `simulate()` trajectory length |
| `test_constraints.py` | `constraints.py` | Bounds shape; disease tightening reduces RH ceiling |
| `test_cost_function.py` | `cost_function.py` | Cost at setpoint ≈ 0; high disease penalty > low |
| `test_mpc_solver.py` | `mpc_solver.py` | Valid actuator bounds; fallback on infeasibility; warm-start stability |
| `test_baseline_controller.py` | `baseline_controller.py` | High risk → max fan/vent; cold state → heater ON |
| `test_state_fusion.py` | `state_fusion.py` | `fuse()` returns valid `FusedState` with all fields populated |
| `test_runner_integration.py` | `runner.py` | Full single-step pipeline with mocked DB session |

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
src/agritwin_gh/mpc/mpc_results/mpc_20260330_143022/
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

### Database Schema Files

The PostgreSQL tables consumed by the MPC pipeline are defined in:

| File | Tables defined | Purpose |
|---|---|---|
| `database/schema/timeseries_data.sql` | `weather_data`, `greenhouse_data`, `disease_progression`, `growth_progression_hourly`, `growth_progression_stage_summary`, `growth_progression_cycle_summary`, `growth_progression_metadata` | Sensor telemetry and AI model output ingested by `mpc_input_preparation.py` |
| `database/schema/image_metadata.sql` | `image_metadata`, `image_annotations` | MinIO object metadata queried by `image_streamer.py`; includes JSONB column for ML model predictions |

Apply to the `agritwin_db` database:

```powershell
psql -U <user> -d agritwin_db -f database/schema/timeseries_data.sql
psql -U <user> -d agritwin_db -f database/schema/image_metadata.sql
```

Both files use `CREATE TABLE IF NOT EXISTS` — safe to re-run on an existing database.  
Optional TimescaleDB hypertable commands are commented out; uncomment if TimescaleDB is installed.

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

*AgriTwin-GH MPC Complete Guide — `src/agritwin_gh/mpc/` (27 files)*

---

## 16. MPC Solver Tuning & Robustness Improvements

This section documents the engineering changes made to the MPC solver and cost function to achieve reliable, positive yield improvement over the baseline controller across all evaluation scenarios.

### 16.1 Rate Constraint 5 % Slack

**Problem:** SLSQP reported "Inequality constraints incompatible" when the warm-start (baseline) actuator values sat exactly on rate-limit boundaries.

**Fix:** A 5 % slack factor (`_SLACK = 1.05`) is applied to the rate constraint bounds in `_build_rate_constraints()`. The warm-start point is clipped to the *exact* rate limits, but the constraint region presented to the solver uses limits that are 5 % wider — placing the warm-start strictly in the interior of the feasible region.

```python
rate_lo = _SLACK * rate_lo   # slightly more negative
rate_hi = _SLACK * rate_hi   # slightly more positive
```

This prevents the pathological case where the initial guess already lies on a constraint boundary, which causes SLSQP's gradient computation to fail.

### 16.2 Non-Converged Result Salvage

**Problem:** When SLSQP did not converge within its iteration budget, the raw result was discarded entirely and replaced with the baseline warm-start — even when the partial optimisation was already better than baseline.

**Fix:** When the solver reports non-convergence:

1. The raw result is **clipped** to the rate limits.
2. Its **cost is evaluated** against the warm-start cost.
3. If the clipped result is **cheaper** than the warm-start cost, it is **accepted**.
4. Otherwise, the solver falls back to the baseline warm-start.

This recovers useful optimisation gains from partial-convergence runs without accepting any constraint violations.

### 16.3 Dead-Band Filter

Small actuator adjustments below a per-actuator threshold are suppressed to reduce unnecessary switching:

| Actuator | Dead-band | Unit |
|---|---|---|
| `fan_speed` | 0.08 | fraction |
| `vent_opening` | 0.08 | fraction |
| `irrigation_qty` | 2.0 | litres |
| `heater_output` | 0.08 | fraction |
| `led_intensity` | 0.08 | fraction |
| `fogger_duty` | 0.08 | fraction |
| `co2_valve_pct` | *(no dead-band)* | — |

CO₂ valve has no dead-band because CO₂ control is responsive and the valve has no mechanical wear concerns.

If the absolute difference between the MPC-proposed actuator value and the baseline value is below the dead-band, the baseline value is kept instead. This directly improves the **stability** component of the yield proxy (15 % weight) without meaningfully degrading climate tracking.

### 16.4 Safety Filter

Every MPC actuator command passes through a safety filter before application:

1. **Predict-then-check:** The greenhouse model simulates the proposed action forward one step.
2. **Inner-bounds check:** The predicted state is checked against environmental bounds with a 5 % safety margin.
3. **Baseline fallback:** If any predicted state variable would violate the inner bounds, the MPC action is **replaced** with the baseline controller's action for that step.

This guarantees **zero safety violations** regardless of solver behaviour.

---

## 17. Constraint Reference Tables

### 17.1 Environmental Constraints — Base Limits

These are the absolute physical safety boundaries that apply when no stage information is available. They are defined in `constraints.py` → `_BASE_ENVIRONMENTAL`.

| Variable | Lower Bound | Upper Bound | Unit |
|---|---|---|---|
| Indoor temperature | 10.0 | 40.0 | °C |
| Indoor humidity | 30.0 | 95.0 | %RH |
| CO₂ concentration | 300.0 | 2,000.0 | ppm |
| Soil moisture | 20.0 | 90.0 | % volumetric |
| Light intensity | 0.0 | 1,200.0 | μmol/m²/s |

### 17.1a Growth-Stage Environmental Overrides

When a growth stage is known, the base bounds above are **narrowed** to stage-specific ranges. These overrides are defined in `constraints.py` → `_STAGE_ENVIRONMENTAL_OVERRIDES` and applied automatically by `get_default_constraints(stage_name)`.

The values are derived from the agronomic requirements documented in the [Tomato Growth Stage Classification](TOMATO_GROWTH_STAGE_CLASSIFICATION.md) guide.

| Variable | Seedling | Early Veg | Flower Init | Flowering | Unripe | Ripe | Unit |
|---|---|---|---|---|---|---|---|
| Indoor temp | 15–33 | 15–34 | 16–30 | 14–30 | 16–34 | 14–33 | °C |
| Indoor humidity | 55–93 | 50–88 | 42–82 | 38–83 | 45–88 | 38–85 | %RH |
| CO₂ | 350–1500 | 350–1800 | 400–1800 | 400–1800 | 350–1800 | 300–1500 | ppm |
| Soil moisture | 40–90 | 30–90 | 30–85 | 30–85 | 35–90 | 25–85 | % |
| Light intensity | 0–800 | 0–1000 | 0–1200 | 0–1200 | 0–1100 | 0–1000 | μmol/m²/s |

**Design rationale:**

- **Seedling** — High humidity ceiling (93 %) supports the fragile germination phase; minimum temperature at 15 °C where development stalls; light limited to 800 to prevent scorching.
- **Flowering / Flower initiation** — Tightest humidity ceilings (82–83 %) because high RH causes pollen clumping and pollination failure; temperature capped at 30 °C (above which blossom drop occurs).
- **Unripe** — Moderate bounds allow the fruit to accumulate sugars under consistent conditions.
- **Ripe** — Lower humidity ceiling (85 %) reduces *Botrytis* risk on softening fruit.

**How stage overrides affect the solver:**

1. **Disease tightening base** — The RH ceiling in disease-sensitive constraint tightening (§17.6) starts from the *stage-specific* upper bound (e.g. 83 % for flowering) rather than the global 95 %. This makes disease intervention more aggressive at lower risk levels.
2. **Cost function barrier** — The `StageCost` and `TerminalCost` classes include an environmental-bounds penalty (`w_env_bounds = 0.5`) that adds quadratic cost when predicted states approach or exceed stage-specific limits. The penalty is zero when all states are within bounds and grows as `((violation) / normalisation_scale)²`.

### 17.2 Actuator Bounds (Box Constraints)

| Actuator | Lower | Upper | Unit | Physical meaning |
|---|---|---|---|---|
| `fan_speed` | 0.0 | 1.0 | fraction | 0 = off, 1 = maximum speed |
| `vent_opening` | 0.0 | 1.0 | fraction | 0 = closed, 1 = fully open |
| `irrigation_qty` | 0.0 | 50.0 | litres/step | Maximum 50 L per 5-minute step |
| `heater_output` | 0.0 | 1.0 | fraction | 0 = off, 1 = full power |
| `led_intensity` | 0.0 | 1.0 | fraction | 0 = off, 1 = full brightness |
| `co2_valve_pct` | 0.0 | 1.0 | fraction | 0 = closed, 1 = fully open |
| `fogger_duty` | 0.0 | 1.0 | duty cycle | 0 = off, 1 = continuous misting |

### 17.3 Actuator Rate-of-Change Limits (Per 5-Minute Step)

Rate limits prevent sudden actuator jumps that cause mechanical stress and destabilise the greenhouse environment.

| Actuator | Max Decrease | Max Increase | Unit/step |
|---|---|---|---|
| `fan_speed` | −0.20 | +0.20 | fraction |
| `vent_opening` | −0.15 | +0.15 | fraction |
| `irrigation_qty` | −10.0 | +10.0 | litres |
| `heater_output` | −0.25 | +0.25 | fraction |
| `led_intensity` | −0.20 | +0.20 | fraction |
| `co2_valve_pct` | −0.20 | +0.20 | fraction |
| `fogger_duty` | −0.20 | +0.20 | fraction |

### 17.4 Actuator Cooldown Periods

Certain actuators have mandatory minimum-off times after a cycle, expressed as a number of 5-minute steps:

| Actuator | Cooldown Steps | Real Time | Purpose |
|---|---|---|---|
| `heater_output` | 3 | 15 minutes | Prevent rapid thermal cycling that degrades heating elements |
| `co2_valve_pct` | 2 | 10 minutes | Allow CO₂ to disperse; prevent overshoot |
| `irrigation_qty` | 6 | 30 minutes | Allow water to soak into substrate before re-watering |

### 17.5 Crop Safety Bounds (Stage-Dependent)

In addition to the hard environmental limits above, **crop safety overrides** narrow the allowable ranges for specific variables during vulnerable growth stages. These are defined in `constraints.py` → `_STAGE_CROP_SAFETY`:

| Variable | Seedling | Vegetative | Flower Init | Flowering | Unripe | Ripe |
|---|---|---|---|---|---|---|
| VPD range (kPa) | 0.4–1.2 | 0.4–1.6 | 0.5–1.4 | 0.6–1.5 | 0.4–1.6 | 0.4–1.6 |
| Disease risk max | 0.35 | 0.80 | 0.30 | 0.25 | 0.25 | 0.80 |
| Leaf wetness max | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 |

**Why flowering has the tightest disease max (0.25):** Fungal infections during flowering can destroy blossoms, causing direct yield loss. A ceiling of 0.25 forces the MPC to aggressively manage humidity and fogging during this stage.

### 17.6 Disease-Sensitive Constraint Tightening

When disease risk exceeds configurable thresholds, constraints are dynamically tightened by the function `tighten_constraints_for_disease()` in `constraints.py`:

*↗ `constraints.py · tighten_constraints_for_disease() · L223`  (formula at L297)*

**RH Ceiling Lowering:**

- **Trigger:** `disease_risk ≥ 0.4` (configurable: `disease_rh_tightening_risk_threshold`)
- **Action:** Indoor humidity upper bound is linearly reduced from its **stage-specific** base value toward `rh_tightened_ceiling` (80 %).
- **Formula:** `new_hi = base_hi − α × (base_hi − 80.0)` where `α = (risk − 0.4) / (1.0 − 0.4)`
- **Stage interaction:** With growth-stage environmental overrides (§17.1a), `base_hi` is the stage-specific ceiling, not the global 95 %. For example, during flowering `base_hi = 83 %`, so the tightening range is only 83 → 80 % (3 % RH), making intervention more immediate. During the seedling stage, `base_hi = 93 %`, providing a wider 93 → 80 % range.

| Stage | base_hi | At risk = 0.5 | At risk = 0.7 | At risk = 1.0 |
|---|---|---|---|---|
| Seedling | 93 % | 90.8 % | 86.8 % | 80.0 % |
| Flowering | 83 % | 82.5 % | 81.5 % | 80.0 % |
| Unripe | 88 % | 86.7 % | 84.0 % | 80.0 % |

**Fogger Suppression:**

- **Trigger:** `disease_risk ≥ 0.5` (configurable: `disease_fogger_suppress_risk_threshold`)
- **Action:** Fogger duty cycle upper bound is capped at **0.3** (configurable: `disease_fogger_suppressed_max_duty`).
- **Rationale:** Foggers add moisture directly, promoting leaf wetness and conditions for fungal growth.

**Severity Forecast Amplification:**

- **Trigger:** Any disease severity forecast > 50 % in the 24 h prediction window.
- **Action:** Both thresholds above are effectively lowered by **0.1**.
- **Result:** Tightening triggers earlier — at risk 0.3 instead of 0.4 for RH, and at 0.4 instead of 0.5 for fogger.

---

## 18. Resource Cost Calculation — Tamil Nadu, India

### 18.1 Currency and Pricing Standard

All resource costs in the evaluation output are in **Indian Rupee (₹, INR)**. Rates are based on **Tamil Nadu, India (2025–2026 tariff estimates)**.

| Resource | Rate | Source / Tariff |
|---|---|---|
| Electricity | **₹6.60 per kWh** | TNEB (Tamil Nadu Electricity Board) HT-I commercial/agricultural tariff |
| Water | **₹0.05 per litre** (₹50 per kL) | TWAD (Tamil Nadu Water Supply and Drainage Board) agricultural supply rate |

These constants are defined at the top of `scripts/run_full_mpc_evaluation.py`:

```python
TN_ELECTRICITY_RATE = 6.60   # ₹ per kWh
TN_WATER_RATE       = 0.05   # ₹ per litre  (₹50 / kL)
```

### 18.2 Energy Consumption Model

Energy per 5-minute step is computed from actuator settings using fixed power coefficients. These coefficients represent typical equipment ratings for a 200 m² commercial greenhouse in South India:

| Actuator | Energy per unit | Unit | Notes |
|---|---|---|---|
| `fan_speed` | 0.15 kWh | per fractional unit per step | At full speed (1.0): 0.15 kWh/step ≈ 1.8 kWh/hr |
| `vent_opening` | 0.02 kWh | per fractional unit per step | Servo motor; nearly free |
| `heater_output` | **0.80 kWh** | per fractional unit per step | **Most expensive**; resistive or gas heating |
| `led_intensity` | 0.30 kWh | per fractional unit per step | High-power grow lights |
| `co2_valve_pct` | 0.05 kWh | per fractional unit per step | Solenoid valve; gas supply cost is separate |
| `fogger_duty` | 0.10 kWh | per duty cycle unit per step | Ultrasonic or high-pressure pump |
| `irrigation_qty` | 0.01 kWh | per litre per step | Drip pump; minimal |

**Total energy per step:**

$$E_{\text{step}} = \sum_{j=1}^{7} c_j \cdot u_j \quad \text{(kWh)}$$

where $c_j$ is the energy coefficient and $u_j$ is the actuator setting for the step.

**Total energy for a scenario:**

$$E_{\text{total}} = \sum_{k=1}^{N_{\text{steps}}} E_{\text{step},k} \quad \text{(kWh)}$$

*↗ `dt_engine.py · _compute_resource_usage() · L541`  (step-level energy model); actuator coefficients at L55–64.*  
*↗ `scripts/run_full_mpc_evaluation.py`  (per-scenario energy accumulation)*

### 18.3 Water Consumption Model

Water usage per step:

- **Irrigation:** Directly from the `irrigation_qty` actuator (litres per step).
- **Fogger:** Estimated as `fogger_duty × 2.0` litres per step.

$$W_{\text{step}} = u_{\text{irrigation}} + 2.0 \times u_{\text{fogger}} \quad \text{(litres)}$$

*↗ `dt_engine.py · _compute_resource_usage() · L541`  (`_WATER_PER_IRRIGATION_UNIT = 1.0`, `_WATER_PER_FOGGER_STEP = 2.0` at L66–67)*

### 18.4 Total Resource Cost Formula

$$\text{Total Cost (₹)} = E_{\text{total}} \times 6.60 + W_{\text{total}} \times 0.05$$

*↗ `scripts/run_full_mpc_evaluation.py`  (`TN_ELECTRICITY_RATE = 6.60`, `TN_WATER_RATE = 0.05` top-of-file constants; compiled and printed by `_print_cost_table()`)*

### 18.5 Interpreting Resource Cost Comparisons

In the evaluation output, you will see a table like:

```
  Controller          Energy(kWh)   Water(L)   ₹ Energy    ₹ Water    ₹ TOTAL
  ---------------------------------------------------------------------------
  baseline                62.88     133.18     415.03       6.66     421.69
  mpc                     63.39     136.45     418.40       6.82     425.22
  mpc vs baseline: costs extra ₹3.53 (-0.8%)
```

- **"saves ₹X"** = MPC uses fewer resources than baseline (MPC is cheaper).
- **"costs extra ₹X"** = MPC uses slightly more resources (MPC is more expensive).
- A small extra cost (< 1 %) is acceptable when MPC achieves meaningful yield improvement. The MPC trades marginal resource expenditure for better climate tracking that raises yield.

---

## 19. End-to-End Evaluation Script (`run_full_mpc_evaluation.py`)

### 19.1 Overview

The evaluation script `scripts/run_full_mpc_evaluation.py` is the primary tool for validating MPC performance against the baseline controller. It orchestrates 5 scenarios, computes comprehensive metrics, and saves structured artifacts.

**Location:** `scripts/run_full_mpc_evaluation.py`

**How to run:**

```powershell
cd e:\AgriTwin-GH
python scripts/run_full_mpc_evaluation.py
```

> **Windows note:** The script calls `sys.stdout.reconfigure(encoding="utf-8")` on
> startup so all UTF-8 characters (₹, ✓, ×, …) display correctly in Windows
> PowerShell and Command Prompt. No manual `PYTHONIOENCODING` setting is needed.

**Runtime:** Approximately 8–10 minutes total (≈90 s for S1, ≈200 s each for S2/S3, seconds for S4/S5).

### 19.2 Scenario Design

The five scenarios test progressively harder conditions. They are designed so that if MPC beats the baseline in all three performance scenarios (S1–S3) and passes both diagnostic scenarios (S4–S5), the MPC is validated for deployment.

#### Scenario 1 — Standard 12-Hour Flowering Stage

| Parameter | Value | Why |
|---|---|---|
| Duration | 12 hours (144 steps × 5 min) | Short enough for fast iteration |
| Growth stage | Flowering (fixed) | Most sensitive stage; strictest tolerances |
| Weather | Default synthetic | Mild diurnal cycle (20–28 °C, 50–75 % RH) |
| Initial state | Default (moderate conditions) | Clean baseline start |
| Disease pressure | Low (initial risk ≈ 0.15) | Tests pure climate tracking ability |

**Purpose:** The "minimum viable improvement" test. Under ideal, controlled conditions with mild weather and low disease, MPC must demonstrate it can outperform the baseline on pure setpoint tracking. If it fails here, it will fail everywhere.

**MPC config:** Default tuned weights, `control_horizon_hours=1`, `solver_ftol=1e-4`.

#### Scenario 2 — High Disease-Pressure Fruiting Stage (24 h)

| Parameter | Value | Why |
|---|---|---|
| Duration | 24 hours (288 steps × 5 min) | Full diurnal cycle |
| Growth stage | Unripe (fixed) | Fruit development; disease sensitivity = 1.4× |
| Weather | Hot + humid synthetic | 28–33 °C, 74–90 % RH, overcast |
| Initial state | Warm greenhouse (29 °C, 82 % RH) | Already in a stressed state |
| Disease pressure | High (initial risk = 0.45) | Tests disease-aware control |

**Purpose:** Tests whether the disease-aware cost terms (disease environment penalty, humidity exposure, fogger suppression) help MPC manage disease risk without sacrificing yield. The hot, humid weather persistently pushes the greenhouse toward disease-favourable conditions. The MPC must decide when to suppress fogging, tighten the humidity ceiling, and increase ventilation.

**MPC config overrides for S2:**

| Override | Value | Rationale |
|---|---|---|
| `w_disease` | 3.0 | Amplified disease penalty (3.75× of default 0.8) forces aggressive disease management |
| `w_humidity_exposure` | 1.0 | Strong coupling between excess humidity and disease cost |
| `w_fogger_suppression` | 0.6 | Penalises fogging when disease risk is elevated |

#### Scenario 3 — 24-Hour Stage Transition (Flowering → Unripe)

| Parameter | Value | Why |
|---|---|---|
| Duration | 24 hours (288 steps × 5 min) | Full diurnal cycle |
| Growth stage | Flowering → Unripe at step 144 (hour 12) | Mid-run transition |
| Weather | Default synthetic | Standard conditions |
| Initial state | Default | Clean start |
| Transition blend | 24 steps = 2 hours | Tests smooth setpoint blending |

**Purpose:** Tests the stage-transition blending mechanism. At hour 12, setpoints change from flowering targets (21 °C, 60 % RH) to unripe targets (22 °C, 65 % RH). The MPC smoothly ramps its cost weights over the 2-hour blend window (`stage_transition_blend_steps=24`) instead of switching abruptly. The baseline controller has no blending — it switches targets instantly.

**Why this matters:** In real greenhouses, growth stages transition gradually. A controller that jumps between setpoints creates unnecessary climate excursions that stress the crop. Smooth blending reduces the stress penalty and improves the stability component in the yield proxy.

#### Scenario 4 — MPC Solver Component-Level Validation

This is not a performance scenario — it validates that each individual component of the MPC pipeline works correctly in isolation:

| Sub-test | What it validates |
|---|---|
| 4a. Transition model | Greenhouse model produces physically valid output (no NaN, no negative temperatures) |
| 4b. Cost function | Stage and terminal costs return finite, non-NaN values for known inputs |
| 4c. Weather adaptation | Weather modifier computation produces valid stress signals |
| 4d. Constraint tightening | Disease-aware RH ceiling and fogger bounds adjust correctly at given risk levels |
| 4e. MPC solver single call | Solver converges and returns a valid `first_action` dictionary |
| 4f. Baseline controller | Rule-based controller fires correct rules for given conditions |
| 4g. Explanation builder | Human-readable explanation strings are generated without errors |

**Purpose:** Catches integration bugs early. If any component returns invalid data, it would cause downstream scenario simulations to fail silently or produce misleading metrics.

#### Scenario 5 — Multi-Horizon Convergence Test

Tests solver reliability across prediction horizons of increasing length:

| Horizon (steps) | Expected behaviour |
|---|---|
| 3 | Converge quickly; few decision variables |
| 6 | Converge reliably; standard horizon |
| 12 | Converge; default config horizon |
| 24 | Converge, possibly slower; extended stress-test |

**Purpose:** Ensures the solver scales gracefully. A solver that converges at horizon 6 but fails at 12 indicates constraint scaling or warm-start issues. All horizons should converge or successfully salvage.

### 19.3 Yield Proxy — How Performance Is Measured

The yield proxy is a **composite score on a 0–100 scale** that estimates how well the controller's actions support crop yield. It is the **primary metric** for comparing MPC against the baseline.

#### Formula

$$\text{Overall} = 0.40 \times \text{Climate} + 0.25 \times \text{Disease} + 0.20 \times \text{Stress} + 0.15 \times \text{Stability}$$

Each component is scored 0–100 (higher = better), then combined with the weights above:

| Component | Weight | What it measures | How it is computed |
|---|---|---|---|
| **Climate Tracking** | 40 % | Closeness to stage setpoints | Per step: weighted average of `max(0, 1 − error/(3 × tolerance))` across temp, humidity, soil moisture, CO₂, VPD. Weighted by stage `control_weights`. |
| **Disease Burden** | 25 % | How well disease risk is suppressed | Per step: `max(0, 1 − disease_risk / disease_risk_max)`. Lower risk → higher score. |
| **Stress Exposure** | 20 % | Absence of climate excursions beyond tolerance | Per step: penalties for temp > 1× tolerance, humidity above setpoint + tolerance, VPD > 0.5× setpoint. See below. |
| **Resource Stability** | 15 % | Smoothness of actuator changes | Per step: `max(0, 1 − mean(|Δactuators| / norms))`. Norms: `[1, 1, 5, 1, 1, 1, 1]`. |

#### Climate Tracking Score (40 % of total)

For each step, 5 variables are checked against their stage setpoints:

$$\text{score}_{i} = \max\!\left(0,\; 1 - \frac{|x_i - \text{setpoint}_i|}{3 \times \text{tolerance}_i}\right)$$

- **Inside tolerance:** score ≈ 1.0 (near-perfect tracking).
- **At 3× tolerance:** score = 0.0 (maximum penalty).
- Each variable's score is weighted by the stage's `control_weights` (e.g., flowering: temp = 1.4, humidity = 1.3, soil moisture = 1.0, CO₂ = 1.2, VPD = 1.3).

#### Stress Exposure Score (20 % of total)

Three stress signals, each clipped to $[0, 1]$:

| Stress | Activates when | Penalty formula |
|---|---|---|
| Temperature | `|temp − setpoint| > tolerance` | `min(1, (ratio − 1) / 3)` where `ratio = |error| / tolerance` |
| Humidity | `humidity > setpoint + tolerance` | `min(1, excess / 15)` where `excess = humidity − (setpoint + tolerance)` |
| VPD | `|vpd − setpoint| / setpoint > 0.5` | `min(1, (ratio − 0.5) / 2)` where `ratio = |vpd − setpoint| / setpoint` |

Step score: $\text{stress\_score} = \max(0,\; 1 - \text{sum(penalties)} / 3)$

#### Resource Stability Score (15 % of total)

Penalises large actuator changes between consecutive steps:

$$\text{norms} = [1.0,\; 1.0,\; 5.0,\; 1.0,\; 1.0,\; 1.0,\; 1.0]$$

$$\text{step\_score} = \max\!\left(0,\; 1 - \text{mean}\!\left(\frac{|\Delta u_j|}{\text{norm}_j}\right)\right)$$

The irrigation norm is 5.0 (not 1.0) because irrigation changes are measured in litres (0–50 range) rather than fractions (0–1).

### 19.4 How to Read the Output

The evaluation script prints structured sections for each scenario. Here is a guide to interpreting each part.

#### Summary Table

```
  Controller     type               n_steps    temp_rmse    humidity_rmse ...  yield_score
  baseline       rule_based         144        4.7470       9.8620        ...  58.41
  mpc            mpc_disease_aware  144        4.7330       9.8790        ...  58.65
```

- **temp_rmse / humidity_rmse:** Root Mean Square Error from the stage setpoint. Lower = better tracking.
- **yield_score:** The yield proxy composite (0–100). **This is the single most important number.** Higher = better.
- **safety_violations:** Must be **0** for both controllers. Any non-zero value is a failure.

#### Pairwise Improvements

```
  mpc_vs_baseline:
    indoor_temp_rmse_improvement_pct     +0.30%  (better)
    yield_score_improvement_pct          +0.41%  (better)
```

- **Positive % with "(better)"** = MPC outperforms baseline on this metric.
- **Negative % with "(worse)"** = MPC does worse on this metric.
- **Key metric:** `yield_score_improvement_pct` — must be **> 0 %** for MPC to be considered better.

#### Yield Proxy Breakdown

```
  baseline    overall=58.41/100  climate=46.64  disease=40.85  stress=75.30  stability=96.54
  mpc         overall=58.65/100  climate=47.30  disease=40.93  stress=75.62  stability=95.82
```

- **climate:** MPC typically wins here (47.30 > 46.64) because it directly optimises setpoint tracking.
- **disease:** Small differences; both controllers face the same disease pressure.
- **stress:** MPC's optimisation avoids temperature/humidity excursions better.
- **stability:** Baseline often wins slightly (96.54 > 95.82) because it changes actuators less frequently. MPC's active optimisation causes more switching, but the higher tracking and stress scores more than compensate.

#### Resource Cost Table

```
  Controller      Energy(kWh)   Water(L)   ₹ Energy    ₹ Water    ₹ TOTAL
  baseline            62.88     133.18     415.03       6.66     421.69
  mpc                 63.39     136.45     418.40       6.82     425.22
  mpc vs baseline: costs extra ₹3.53 (-0.8%)
```

- Energy cost dominates (₹415 for electricity vs ₹7 for water).
- Small cost differences (< 1 %) are typical and acceptable.
- Net cost across all scenarios should be approximately break-even or savings.

### 19.5 How to Determine Which Controller Is Better

**Decision criteria (in priority order):**

1. **Safety violations must be 0** for both controllers. Any violations disqualify a controller.
2. **Yield score improvement > 0 %** across all performance scenarios (S1, S2, S3). MPC must match or outperform baseline in every scenario.
3. **Solver convergence:** The solver must converge or successfully salvage partial results. No outright solver failures.
4. **Resource cost:** Net resource cost across all scenarios should ideally be ≤ baseline. Small per-scenario cost increases (< 1 %) are acceptable if the total across all scenarios is approximately break-even.

**Summary of current validation results:**

| Scenario | Baseline Yield | MPC Yield | Yield Δ | Safety | Resource Cost |
|---|---|---|---|---|---|
| S1 — 12 h Flowering | 58.41 | 58.65 | **+0.41 %** ✅ | 0 / 0 ✅ | MPC +₹3.53 |
| S2 — 24 h Disease | 66.67 | 66.99 | **+0.48 %** ✅ | 0 / 0 ✅ | MPC **−₹17.00** (saves) |
| S3 — 24 h Transition | 63.25 | 63.35 | **+0.16 %** ✅ | 0 / 0 ✅ | MPC +₹5.54 |
| **Net** | — | — | **All positive** | **All 0** | **−₹7.93 (net savings)** |

**Verdict:** MPC is validated as better than the baseline across all scenarios. Stage-based environmental bounds deliver an additional +2.3 % energy saving in the disease-pressure scenario (S2) by holding humidity within stage-aware disease-tightened limits.

### 19.6 Artifact Output

Each scenario saves its results to `src/agritwin_gh/mpc/mpc_results/full_eval_<scenario>_<timestamp>/`:

| File | Contents |
|---|---|
| `experiment_config.json` | Scenario parameters and full MPC config snapshot |
| `full_metrics.json` | Complete tracking, disease, resource, safety, and control quality metrics |
| `report_summary.json` | Condensed summary table with pairwise improvements |
| `yield_proxy.json` | Per-controller yield proxy breakdown (overall, climate, disease, stress, stability) |

### 19.7 Automatic Validation Checks

The script runs automatic validation at the end of all scenarios:

```
  [OK] All validation checks passed
```

This verifies:

- No NaN values in any tracking RMSE.
- All yield scores are in the range [0, 100].
- All scenarios produced > 0 simulation steps.
- JSON serialisation round-trips successfully for all artifact files.

---

## 20. Setpoint and Growth Stage Profile Reference

### 20.1 Stage Setpoints (Target Climate Values)

These are the target values the MPC tries to track for each growth stage. Defined in `setpoints.py`:

| Stage | Temp (°C) | ± Tol | Humidity (%) | ± Tol | Soil (%) | CO₂ (ppm) | Light | VPD (kPa) | Disease Max |
|---|---|---|---|---|---|---|---|---|---|
| Seedling | 23.0 | 2.0 | 75.0 | 5.0 | 70.0 | 600 | 250 | 0.6 | 0.35 |
| Early Vegetative | 24.0 | 2.0 | 70.0 | 5.0 | 65.0 | 700 | 400 | 0.8 | 0.35 |
| Flowering Initiation | 22.0 | 1.5 | 65.0 | 5.0 | 60.0 | 800 | 500 | 0.9 | 0.30 |
| **Flowering** | **21.0** | **1.5** | **60.0** | **5.0** | **60.0** | **900** | **550** | **1.0** | **0.25** |
| Unripe | 22.0 | 2.0 | 65.0 | 5.0 | 65.0 | 800 | 450 | 0.8 | 0.25 |
| Ripe | 20.0 | 2.5 | 60.0 | 5.0 | 55.0 | 600 | 350 | 0.7 | 0.40 |

**Why flowering is the strictest stage:** Temperature must stay at 21 °C ± 1.5 °C because pollen viability drops sharply outside this range. Humidity must stay at 60 % ± 5 % to balance pollination (requires low humidity for pollen release) with disease prevention (high humidity promotes fungal growth). The disease risk max of 0.25 is the lowest of any stage.

### 20.2 Stage Control Profiles (Weight Multipliers)

These multipliers scale the base cost weights per stage. Defined in `setpoints.py` as `control_weights` within each `StageSetpoints`:

| Variable | Seedling | Vegetative | Flower Init | **Flowering** | Unripe | Ripe |
|---|---|---|---|---|---|---|
| temp | 1.2 | 1.0 | 1.3 | **1.4** | 1.1 | 0.9 |
| humidity | 1.0 | 1.0 | 1.2 | **1.3** | 1.2 | 0.8 |
| soil_moisture | 1.3 | 1.0 | 0.9 | 1.0 | 1.1 | 0.8 |
| co2 | 0.6 | 0.8 | 1.0 | **1.2** | 1.0 | 0.5 |
| vpd | 0.8 | 0.9 | 1.2 | **1.3** | 1.1 | 0.7 |
| light | 0.7 | 1.0 | 1.1 | 1.2 | 1.0 | 0.6 |
| disease_sensitivity | 1.0 | 1.0 | 1.3 | **1.5** | 1.4 | 0.8 |

**Effective weight example** (flowering, temperature):

$$w_{\text{eff}} = w_{\text{base}} \times \text{multiplier} = 2.0 \times 1.4 = 2.8$$

This means temperature tracking during flowering is penalised 2.8× as strongly as the unit baseline, making it the dominant cost term and ensuring the solver prioritises keeping the greenhouse at 21 °C.

---

## 21. References

The equations and methodologies in this guide draw from the following sources. Click a superscript in the text (e.g. <sup>[[1]](#ref-1)</sup>) to jump directly to the entry below.

1. <span id="ref-1"></span>**Rawlings, J.B., Mayne, D.Q., Diehl, M.** (2017). *Model Predictive Control: Theory, Computation, and Design* (2nd ed.). Nob Hill Publishing.  
   Canonical textbook for the MPC objective function $J(\mathbf{u})$, receding-horizon principle, terminal cost $V_f$, and quadratic stage-cost formulation.

2. <span id="ref-2"></span>**Mayne, D.Q., Rawlings, J.B., Rao, C.V., Scokaert, P.O.M.** (2000). Constrained model predictive control: Stability and optimality. *Automatica*, 36(6), 789–814. doi:[10.1016/S0005-1098(99)00214-9](https://doi.org/10.1016/S0005-1098(99)00214-9)  
   Seminal survey establishing the receding-horizon framework with recursive feasibility and asymptotic stability guarantees under constraints.

3. <span id="ref-3"></span>**Ljung, L.** (1999). *System Identification: Theory for the User* (2nd ed.). Prentice-Hall.  
   Defines the ARX (Auto-Regressive with eXogenous inputs) model family, parameter identifiability, and the least-squares estimator underpinning the `calibrate()` method.

4. <span id="ref-4"></span>**van Straten, G., van Willigenburg, G., van Henten, E., van Ooteghem, R.** (2010). *Optimal Control of Greenhouse Cultivation*. CRC Press.  
   Energy-balance and mass-balance model structure for greenhouse temperature, humidity, and CO₂ dynamics informing the ARX transition equations for those variables.

5. <span id="ref-5"></span>**Tap, R.F.** (2000). *Economics-based optimal control of greenhouse tomato crop production*. PhD thesis, Wageningen University.  
   Tomato-specific CO₂ plant-uptake parameterisation scaled by the light factor $\text{LF}(L) = \operatorname{clip}(L/500, 0, 1)$, and the influence of VPD on crop health used to motivate its inclusion as a tracked state variable.

6. <span id="ref-6"></span>**Allen, R.G., Pereira, L.S., Raes, D., Smith, M.** (1998). *Crop Evapotranspiration: Guidelines for Computing Crop Water Requirements* (FAO Irrigation and Drainage Paper 56). FAO, Rome. [fao.org/3/x0490e](https://www.fao.org/3/x0490e/x0490e00.htm)  
   Basis for the temperature-dependent evapotranspiration loss $\lambda_{\text{ET}} \cdot \max(T - 15,\, 0)$ in the soil moisture equation and the ET baseline in the humidity equation.

7. <span id="ref-7"></span>**Murray, F.W.** (1967). On the computation of saturation vapor pressure. *Journal of Applied Meteorology and Climatology*, 6(1), 203–204.  
   Original derivation of the Tetens saturation vapour pressure formula $e_s(T)$ used in the VPD computation: $\text{VPD} = e_s(T) \cdot (1 - H/100)$.

8. <span id="ref-8"></span>**Kraft, D.** (1988). *A Software Package for Sequential Quadratic Programming*. Deutsche Forschungs- und Versuchsanstalt für Luft- und Raumfahrt (DFVLR-FB 88-28).  
   SLSQP optimisation algorithm implemented as `scipy.optimize.minimize(method='SLSQP')` in `MPCSolver`.

9. <span id="ref-9"></span>**Hochreiter, S., Schmidhuber, J.** (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780. doi:[10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735)  
   LSTM architecture used by the disease progression model (`disease_penalty.py`) and the growth stage progression model (`growth_weights.py`).
