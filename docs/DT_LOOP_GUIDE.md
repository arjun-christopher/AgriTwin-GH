# AgriTwin-GH: Digital Twin Closed-Loop Layer

> **Who is this for?**  
> This guide is written so that anyone — from a student seeing a greenhouse control system for the first time to an experienced developer extending the codebase — can understand what the DT layer does, how every file fits together, every formula it uses, and exactly how to run and extend it.

---

## Table of Contents

1\. [What is a Digital Twin? (Plain English)](#1-what-is-a-digital-twin-plain-english)

2\. [How the DT Layer Fits into AgriTwin-GH](#2-how-the-dt-layer-fits-into-agritwin-gh)

3\. [System Architecture at a Glance](#3-system-architecture-at-a-glance)

4\. [Full Data Flow Diagram](#4-full-data-flow-diagram)

5\. [File Tree — All 12 Source Files](#5-file-tree--all-12-source-files)

6\. [File-by-File Reference](#6-file-by-file-reference)

&emsp;6.1\. [`dt_state.py`](#61-dt_statepy)

&emsp;&emsp;6.1.1\. [`DTSnapshot`](#dtsnapshot)

&emsp;&emsp;6.1.2\. [`DTStepInput`](#dtstepinput)

&emsp;&emsp;6.1.3\. [`DTDiagnostics`](#dtdiagnostics)

&emsp;&emsp;6.1.4\. [`DTStepOutput`](#dtstepoutput)

&emsp;6.2\. [`dt_interface.py`](#62-dt_interfacepy)

&emsp;6.3\. [`dt_engine.py`](#63-dt_enginepy)

&emsp;&emsp;6.3.1\. [8-Phase step workflow](#8-phase-step-workflow)

&emsp;&emsp;6.3.2\. [Effect attribution — the formulas](#effect-attribution--the-formulas)

&emsp;&emsp;6.3.3\. [Disease environment flags](#disease-environment-flags)

&emsp;&emsp;6.3.4\. [Resource accounting](#resource-accounting)

&emsp;6.4\. [`dt_runtime_prep.py`](#64-dt_runtime_preppy)

&emsp;&emsp;6.4.1\. [Initial state formula](#initial-state-formula)

&emsp;&emsp;6.4.2\. [Synthetic weather formula](#synthetic-weather-formula)

&emsp;&emsp;6.4.3\. [`build_fused_state`](#build_fused_state)

&emsp;&emsp;6.4.4\. [`build_mpc_solver`](#build_mpc_solver)

&emsp;6.5\. [`dt_loop.py`](#65-dt_looppy)

&emsp;&emsp;6.5.1\. [Multi-rate cadence](#multi-rate-cadence)

&emsp;&emsp;6.5.2\. [Event-triggered MPC re-solve](#event-triggered-mpc-re-solve)

&emsp;&emsp;6.5.3\. [`DTLoopStepResult`](#dtloopstepresult)

&emsp;&emsp;6.5.4\. [Per-step run workflow](#per-step-run-workflow)

&emsp;6.6\. [`dt_logger.py`](#66-dt_loggerpy)

&emsp;&emsp;6.6.1\. [`DTLoopRunSummary`](#dtlooprunsummary)

&emsp;&emsp;6.6.2\. [`DTLoopLogger`](#dtlooplogger)

&emsp;&emsp;6.6.3\. [Console output format](#console-output-format)

&emsp;6.7\. [`dt_input_provider.py`](#67-dt_input_providerpy)

&emsp;&emsp;6.7.1\. [`DTInputProvider` protocol](#dtinputprovider-protocol)

&emsp;&emsp;6.7.2\. [`SyntheticInputProvider`](#syntheticinputprovider)

&emsp;&emsp;6.7.3\. [`DatabaseInputProvider`](#databaseinputprovider)

&emsp;&emsp;6.7.4\. [`ImageObservation`](#imageobservation)

&emsp;6.8\. [`dt_image_observer.py`](#68-dt_image_observerpy)

&emsp;6.9\. [`dt_output_writer.py`](#69-dt_output_writerpy)

&emsp;6.10\. [`dt_artifact_manager.py`](#610-dt_artifact_managerpy)

&emsp;6.11\. [`realtime_core.py` (core engine)](#611-realtime_corepy-core-engine)

&emsp;6.12\. [`run_realtime_loop.py` (CLI)](#612-run_realtime_looppy-cli)

7\. [Key Data Structures](#7-key-data-structures)

&emsp;7.1\. [`DTLoopStepResult`](#71-dtloopstepresult)

&emsp;7.2\. [`DTDiagnostics`](#72-dtdiagnostics)

&emsp;7.3\. [`DTLoopRunSummary`](#73-dtlooprunsummary)

&emsp;7.4\. [`ImageObservation`](#74-imageobservation)

8\. [Mathematical Formulas — Complete Reference](#8-mathematical-formulas--complete-reference)

&emsp;8.1\. [Disease risk score](#81-disease-risk-score)

&emsp;8.2\. [Diurnal weather model](#82-diurnal-weather-model)

&emsp;8.3\. [Initial state temperature](#83-initial-state-temperature)

&emsp;8.4\. [Effect attribution decomposition](#84-effect-attribution-decomposition)

&emsp;8.5\. [Energy and water accounting](#85-energy-and-water-accounting)

9\. [Cadence Reference](#9-cadence-reference)

10\. [Event-Trigger Thresholds](#10-event-trigger-thresholds)

11\. [Disease Environment Flag Thresholds](#11-disease-environment-flag-thresholds)

12\. [How to Run](#12-how-to-run)

&emsp;12.1\. [Prerequisites](#121-prerequisites)

&emsp;12.2\. [Environment setup](#122-environment-setup)

&emsp;12.3\. [Interactive mode](#123-interactive-mode)

&emsp;12.4\. [Non-interactive mode](#124-non-interactive-mode)

&emsp;12.5\. [Cleanup after a run](#125-cleanup-after-a-run)

&emsp;12.6\. [Full CLI flag reference](#126-full-cli-flag-reference)

13\. [Artifact Output Layout](#13-artifact-output-layout)

&emsp;13.1\. [Run folder structure](#131-run-folder-structure)

&emsp;13.2\. [File contents](#132-file-contents)

14\. [Test Script](#14-test-script)

15\. [Assumptions & Design Decisions](#15-assumptions--design-decisions)

&emsp;15.1\. [Design decisions](#151-design-decisions)

&emsp;15.2\. [Assumptions](#152-assumptions)

16\. [Known Limitations](#16-known-limitations)

17\. [Recommended Next Extensions](#17-recommended-next-extensions)

18\. [Production Closed-Loop — Implemented](#18-production-closed-loop--implemented)

19\. [References](#19-references)

---

## 1. What is a Digital Twin? (Plain English)

Imagine you are a pilot with two cockpits: the real airplane, and a perfect
computer simulation of the same airplane running in parallel. Every few
seconds the simulation receives the same sensor readings as the real plane —
speed, altitude, engine temperature — and simulates what will happen next.
If the simulation predicts an engine failure in 10 minutes, the pilot can
act *now*, not when the failure actually happens. This parallel simulation is
a **digital twin**.

In AgriTwin-GH, the "real airplane" is the physical tomato greenhouse.
The sensors report temperature, humidity, CO₂, soil moisture, and disease
risk every 5 minutes. The digital twin:

1. **Mirrors** the current greenhouse state (temperature, humidity, etc.)
2. **Predicts** how that state will change given the actuators (heater, fan,
   fogger) and the weather forecast
3. **Optimises** actuator commands every 15 minutes using Model Predictive
   Control (see [`MPC_COMPLETE_GUIDE.md`](MPC_COMPLETE_GUIDE.md))
4. **Reports** what it decided and why — per-actuator effect attribution,
   energy/water used, disease risk — for humans and dashboards to read

**Why not just react?**  
Rule-based control ("if temperature > 30 °C, turn on fan") reacts *after*
the problem arrives. The DT predicts the problem before it reaches the crop
and adjusts proactively. This is especially important during flowering —
a 5-minute heat spike at the wrong moment can cause blossom drop and reduce
yield permanently.<sup>[[1]](#ref-1)</sup>

---

## 2. How the DT Layer Fits into AgriTwin-GH

AgriTwin-GH has two execution modes:

| Mode | What it does | Database needed? |
|------|-------------|-----------------|
| **Full production** | Reads real sensor data from PostgreSQL, retrieves crop images from MinIO, runs live MPC | Yes |
| **DT closed-loop (this module)** | Generates synthetic weather + initial state, runs MPC in a simulated loop, saves JSON artifacts | **No** |

The DT layer is the second mode. It was designed so the entire control
system can be exercised, validated, and demonstrated on any laptop — with
no database, no MinIO, no running greenhouse required.

At the same time, the DT layer uses the **exact same MPC solver, ARX
physics model, and cost function** as the production system. Swapping in a
real sensor feed requires changing only the *input provider* class — the
loop itself does not change at all.

---

## 3. System Architecture at a Glance

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      AGRITWIN-GH DT CLOSED-LOOP LAYER                    │
│              src/agritwin_gh/mpc/  (12 DT-specific files)                │
│                                                                          │
│   INPUT                 SIMULATION CORE              OUTPUT              │
│   ─────                 ───────────────              ──────              │
│   DTInputProvider  ──►  DTLoop                  ──►  DTLoopLogger        │
│   (synthetic or DB)     │   every 5 min:             (console + JSON)    │
│                         │   DigitalTwinEngine                            │
│                         │   (ARX physics)         ►  DTOutputWriter      │
│                         │                           (4 artifact files)   │
│                         │   every 15 min:                                │
│                         │   MPCSolver             ►  DTArtifactManager   │
│                         │   (SLSQP optimiser)       (run-ID folder)      │
│                         │                                                │
│                         │   every 30 min:                                │
│   ImageObserver    ──►  │   image refresh hook                           │
│   (synthetic or MinIO)                                                   │
└──────────────────────────────────────────────────────────────────────────┘
```

**Three rates operating simultaneously:**

| Rate | Period | What happens |
|------|--------|-------------|
| DT step | every 5 min | ARX physics advances the greenhouse state by one timestep |
| MPC solve | every 15 min | Optimal actuator trajectory recomputed for the next 1 hour |
| Image refresh | every 30 min | Disease/growth image observation hook fires |

---

## 4. Full Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         LAYER 0: INPUTS                                  │
│  ┌─────────────────────────────┐  ┌──────────────────────────────────┐  │
│  │   SyntheticInputProvider    │  │  SyntheticImageObserver          │  │
│  │   ─────────────────────     │  │  ──────────────────────────      │  │
│  │   • get_initial_state()     │  │  • observe(growth_stage, state,  │  │
│  │     (time-of-day GreenhouseState)   step_index, timestamp)        │  │
│  │   • get_weather_sequence()  │  │    → ImageObservation            │  │
│  │     (diurnal WeatherState[])│  │  (MinIOImageObserver ✅)         │  │
│  │   DatabaseInputProvider ✅ implemented  │                          │  │
│  └────────────┬────────────────┘  └─────────────────┬────────────────┘  │
└───────────────┼─────────────────────────────────────┼───────────────────┘
                │                                      │ (every 30 min)
┌───────────────┼──────────────────────────────────────┼───────────────────┐
│               ▼      LAYER 1: DTLoop                 │                   │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                          dt_loop.py                                │  │
│  │                    DTLoop.run(n_steps)                             │  │
│  │                                                                    │  │
│  │  For each 5-minute step:                                           │  │
│  │   1. Decide: MPC due? (step % 3 == 0)  Image due? (step % 6 == 0) │  │
│  │   2. If MPC: build FusedState → MPCSolver.solve() → new action     │  │
│  │   3. Build DTStepInput (state + action + weather + crop context)   │  │
│  │   4. DigitalTwinEngine.step(DTStepInput) → DTStepOutput           │  │
│  │   5. Check should_force_mpc_update(next_state)?                    │  │
│  │      If yes: re-solve + re-step.                                   │  │
│  │   6. Yield DTLoopStepResult                                        │  │
│  │   7. state = next_state  (chain forward)                           │  │
│  └───────────────────────────────┬────────────────────────────────────┘  │
└──────────────────────────────────┼────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼────────────────────────────────────────┐
│    LAYER 2: DT ENGINE             ▼                                        │
│  ┌───────────────────────────────────────────────────────────────────┐    │
│  │                          dt_engine.py                             │    │
│  │               DigitalTwinEngine.step(DTStepInput)                 │    │
│  │                                                                   │    │
│  │  Phase 1:  GreenhouseTransitionModel.step() → next_state         │    │
│  │  Phase 2:  Recompute disease risk from post-step env              │    │
│  │  Phase 3:  Per-actuator effect attribution (Δ decomposition)      │    │
│  │  Phase 4:  Disease-environment flags (4 boolean conditions)       │    │
│  │  Phase 5:  Energy (kWh) + water (L) accounting                   │    │
│  │  Phase 6:  State delta (variable-by-variable change)              │    │
│  │  Phase 7:  Bounds clamping detection                              │    │
│  │  Phase 8:  Setpoint error (actual − target per variable)          │    │
│  │                 → DTStepOutput (next_state + diagnostics)         │    │
│  └───────────────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼────────────────────────────────────────┐
│    LAYER 3: MPC SOLVER            │                                        │
│  ┌────────────────────────────────────────────────────────────────────┐   │
│  │                    dt_runtime_prep.py                              │   │
│  │                        build_fused_state()                        │   │
│  │   state + stage + disease_risk + weather_forecast → FusedState    │   │
│  │                              │                                     │   │
│  │                              ▼                                     │   │
│  │                      mpc_solver.py                                 │   │
│  │                  MPCSolver.solve(fused_state)                      │   │
│  │          → MPCSolution.first_action (ActuatorState)               │   │
│  └────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼────────────────────────────────────────┐
│    LAYER 4: OUTPUT                ▼                                        │
│  ┌──────────────────┐  ┌────────────────────┐  ┌─────────────────────┐   │
│  │ DTLoopLogger     │  │ DTOutputWriter      │  │ DTArtifactManager   │   │
│  │ ───────────────  │  │ ─────────────────── │  │ ─────────────────── │   │
│  │ Console output   │  │ 4 JSON artifact     │  │ Run-ID folder       │   │
│  │ per-step lines   │  │ streams per step    │  │ metadata + summary  │   │
│  │ Run summary      │  │ (state, MPC, diag,  │  │ run_metadata.json   │   │
│  │ dt_loop_*.json   │  │  summary)           │  │ summary.json        │   │
│  └──────────────────┘  └────────────────────┘  └─────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. File Tree — All 12 Source Files

```
src/agritwin_gh/mpc/
│
│── DT data types ────────────────────────────────────────────────
├── dt_state.py              # DTSnapshot, DTStepInput, DTDiagnostics, DTStepOutput
│
│── DT plant interface ───────────────────────────────────────────
├── dt_interface.py          # DigitalTwinPlant — thin public wrapper
│
│── DT physics engine ────────────────────────────────────────────
├── dt_engine.py             # DigitalTwinEngine — 8-phase step with effect attribution
│
│── DT runtime helpers ───────────────────────────────────────────
├── dt_runtime_prep.py       # prepare_initial_state, prepare_weather_sequence,
│                            # build_fused_state, build_mpc_solver
│
│── DT orchestrator ──────────────────────────────────────────────
├── dt_loop.py               # DTLoop — multi-rate closed-loop (DT+MPC+image)
│                            # DTLoopStepResult, should_force_mpc_update
│
│── DT console + export ──────────────────────────────────────────
├── dt_logger.py             # DTLoopLogger, DTLoopRunSummary
│
│── DT input abstraction ─────────────────────────────────────────
├── dt_input_provider.py     # DTInputProvider (Protocol) + SyntheticInputProvider
│                            # + DatabaseInputProvider (live PostgreSQL feed)
│                            # ImageObservation dataclass
│
│── DT image hook ────────────────────────────────────────────────
├── dt_image_observer.py     # ImageObserver (Protocol) + SyntheticImageObserver
│
│── DT output abstraction ────────────────────────────────────────
├── dt_output_writer.py      # DTOutputWriter (Protocol) + JsonFileOutputWriter
│                            # + fanout_step_to_writer()
│
│── DT run folder management ─────────────────────────────────────
├── dt_artifact_manager.py   # DTArtifactManager — timestamped run folder
│
│── Realtime closed-loop core ────────────────────────────────────
├── realtime_core.py          # RealtimeLoop, RealtimeLoopConfig — importable engine
│                             # RealtimeMPCInputPreparation, RealtimeStepResult
│                             # RealtimeRunSummary, seed_initial_state()
│                             # write_step_to_stream(), ensure_stream_table()
│                             # estimate_energy(), diurnal helpers
│
scripts/
└── run_realtime_loop.py      # Thin CLI / test harness — imports from realtime_core
                              # (console display, artifact saving, signal handling,
                              #  interactive prompts, argparse)
```

---

## 6. File-by-File Reference

### 6.1 `dt_state.py`

**Purpose**: All typed dataclasses that flow through the DT simulation layer.
Nothing is computed here — this file is pure data definitions.
It re-uses the core `GreenhouseState`, `ActuatorState`, and `WeatherState`
from `state.py` rather than duplicating them.

**Why data classes?** Python `@dataclass` decorators auto-generate
`__init__`, `__repr__`, and `__eq__` methods from typed field declarations.
This eliminates boilerplate, provides IDE auto-complete, and makes the
type of every field explicit.

---

#### `DTSnapshot`

A complete "photograph" of the greenhouse at a single moment in time.
Think of it as the DT's answer to the question: *"What was the full state
of the greenhouse at step 47?"*

```python
@dataclass
class DTSnapshot:
    greenhouse_state: GreenhouseState   # 9 indoor climate variables
    growth_stage:     str               # e.g. "flowering"
    disease_classification: str         # e.g. "healthy leaves"
    disease_severity:  dict[str, float] # per-disease severity scores
    actuators_applied: ActuatorState    # what actuators were active
    weather:           WeatherState     # outdoor conditions at this step
    timestamp:         datetime         # logical simulation time
    step_index:        int              # step number within the run
    metadata:          dict             # arbitrary extra fields
```

Key method:
- `to_dict()` → JSON-serialisable `dict` — used by the output writer.

**Inputs**: Components assembled by `DTLoop` after each step.  
**Outputs**: Stored inside `DTStepOutput` for logging.

---

#### `DTStepInput`

Everything the `DigitalTwinEngine` needs to simulate one 5-minute step
forward. Think of it as the "question" you ask the simulator: *"Given this
state, this action, and this weather — what happens next?"*

```python
@dataclass
class DTStepInput:
    current_state:           GreenhouseState  # state at start of step
    action:                  ActuatorState    # actuator commands to apply
    weather:                 WeatherState     # outdoor weather this step
    growth_stage:            str              # canonical stage label
    disease_risk_score:      float            # carried-forward risk [0,1]
    disease_classification:  str              # current disease label
    disease_severity:        dict[str, float] # per-disease severity
    dt_minutes:              int = 5          # step duration
    step_index:              int = 0          # ordinal within run
    timestamp:               datetime         # logical time
```

Key method:
- `to_dict()` → JSON-serialisable `dict`

---

#### `DTDiagnostics`

Everything the engine reports *about* the step — not the physics result
itself, but all the diagnostic information around it. Think of it as the
flight data recorder attached to each step.

```python
@dataclass
class DTDiagnostics:
    energy_kwh:               float            # kWh consumed this step
    water_litres:             float            # litres used this step
    disease_risk_recomputed:  float            # fresh risk from new state
    state_delta:              dict[str, float] # {variable: new − old}
    bounds_clamped:           list[str]        # variables that hit limits
    setpoint_error:           dict[str, float] # {variable: actual − target}
    effect_attribution:       dict[str, dict]  # per-actuator contributions
    disease_environment_flags: dict[str, bool] # 4 boolean risk conditions
    step_compute_ms:          float            # wall-clock time for this step
```

`effect_attribution` is the richest field: for each state variable
(e.g. `indoor_temp`) it gives a breakdown like:

```json
{
  "indoor_temp": {
    "natural_decay":    -0.4500,
    "weather_exchange":  0.1600,
    "solar_heating":     1.5000,
    "heater":            2.2500,
    "fan_cooling":      -0.1200,
    "vent_cooling":     -0.0800
  }
}
```

This is explained in detail in [Section 8.4](#84-effect-attribution-decomposition).

---

#### `DTStepOutput`

The "answer" returned by one simulation step. Contains the predicted
next state plus all diagnostics.

```python
@dataclass
class DTStepOutput:
    next_state:   GreenhouseState  # predicted greenhouse state after step
    diagnostics:  DTDiagnostics    # resource usage, attribution, flags
    snapshot:     DTSnapshot       # full DT photograph at this timestep
```

**Inputs**: Produced by `DigitalTwinEngine.step()`.  
**Outputs**: Consumed by `DTLoop.run()` which wraps it in `DTLoopStepResult`.

---

### 6.2 `dt_interface.py`

**Purpose**: A thin, stable public interface for running DT simulations.
Downstream code that only wants to step the physics model without knowing
about the engine internals uses this class.

**Class: `DigitalTwinPlant`**

```python
class DigitalTwinPlant:
    def __init__(
        self,
        model_params: GreenhouseModelParams | None = None,
        dt_minutes: int = 5,
    ) -> None: ...

    def step(self, step_input: DTStepInput) -> DTStepOutput: ...
    def simulate(self, step_inputs: list[DTStepInput]) -> list[DTStepOutput]: ...
```

**Why a separate interface class?**  
`DigitalTwinEngine` is the rich internal implementation. `DigitalTwinPlant`
is the stable public surface. If the engine internals change (e.g. new
diagnostics), imports of `DigitalTwinPlant` in other modules keep working
without change. This follows the *Facade* design pattern.<sup>[[2]](#ref-2)</sup>

**`step()`**: Runs a single 5-minute simulation forward. Returns
`DTStepOutput` with the new state and all diagnostics.

**`simulate()`**: Runs multiple steps in sequence, chaining the output
state of each step as the input state of the next. Only the
first `step_input.current_state` matters; subsequent ones are overwritten
automatically.

**Inputs**: `DTStepInput` (or a list of them).  
**Outputs**: `DTStepOutput` (or a list of them).

---

### 6.3 `dt_engine.py`

**Purpose**: The core of the DT physics layer. Every 5-minute simulation
step passes through this class. It is the most technically detailed
file in the DT module.

**Class: `DigitalTwinEngine`**

```python
class DigitalTwinEngine:
    def __init__(
        self,
        model_params: GreenhouseModelParams | None = None,
        dt_minutes: int = 5,
    ) -> None: ...

    def step(self, step_input: DTStepInput) -> DTStepOutput: ...
    def simulate(self, step_inputs: list[DTStepInput]) -> list[DTStepOutput]: ...
```

The engine does **not** contain its own physics model. It delegates
all state transition computation to `GreenhouseTransitionModel.step()`
(the ARX model in `greenhouse_model.py`). Then it wraps that result with
eight phases of diagnostic computation.

#### 8-Phase Step Workflow

```
DTStepInput
    │
    ▼  Phase 1: _update_climate()
    │   Calls GreenhouseTransitionModel.step(state, action, weather)
    │   → next_state (9 ARX equations, physically clamped)
    │
    ▼  Phase 2: _recompute_disease_risk()
    │   Sigmoid risk score from post-step humidity + leaf wetness + temp + stage
    │   → next_state.disease_risk_score updated in-place
    │
    ▼  Phase 3: _compute_effect_attribution()
    │   Decomposes Δ for 5 state variables into named actuator contributions
    │   → effect_attribution: {variable: {source: contribution}}
    │
    ▼  Phase 4: _assess_disease_environment()
    │   4 boolean flags: high_humidity_risk, high_leaf_wetness,
    │   disease_temp_band, fogger_disease_concern
    │   → disease_environment_flags
    │
    ▼  Phase 5: _compute_resource_usage()
    │   Energy kWh + water litres for this step
    │   → (energy_kwh, water_litres)
    │
    ▼  Phase 6: _compute_state_delta()
    │   {variable: next_val − current_val}
    │   → state_delta
    │
    ▼  Phase 7: _detect_bounds_clamped()
    │   Which variables hit physical bounds (e.g. CO₂ < 300 or > 2500 ppm)
    │   → bounds_clamped: list[str]
    │
    ▼  Phase 8: _compute_setpoint_error()
    │   Signed (actual − target) per variable for the current growth stage
    │   → setpoint_error: {variable: float}
    │
    ▼
DTStepOutput(next_state, diagnostics, snapshot)
```

---

#### Effect Attribution — The Formulas

For each state variable the engine computes the **linear contribution**
of every actuator and natural process using the same coefficients as
the ARX model. The contributions sum to the pre-clamping delta.

All formula notation: `p` = `GreenhouseModelParams`, one set of coefficients.

**Temperature** (`indoor_temp`):

$$
\Delta T = \underbrace{(p_{decay}^T - 1)\cdot T}_\text{natural decay}
         + \underbrace{p_{ext}^T \cdot (T_{ext} - T)}_\text{weather exchange}
         + \underbrace{p_{solar}^T \cdot \text{solar}}_\text{solar heating}
         + \underbrace{p_{heat}^T \cdot u_{heat}}_\text{heater}
         + \underbrace{p_{fan}^T \cdot u_{fan}}_\text{fan cooling}
         + \underbrace{p_{vent}^T \cdot u_{vent}}_\text{vent cooling}
$$

- $p_{decay}^T < 1$ → natural thermal drift toward outdoor temperature
- $p_{ext}^T > 0$ → outdoor air infiltration through the envelope
- $p_{solar}^T > 0$ → solar gain through glazing (kW/m² → °C)
- $p_{heat}^T > 0$ → heater contribution (major warming actuator)
- $p_{fan}^T < 0$ → fan evaporative + convective cooling (negative)
- $p_{vent}^T < 0$ → vent passive cooling via outside air (negative)

**Humidity** (`indoor_humidity`):

$$
\Delta RH = (p_{decay}^{RH} - 1)\cdot RH
          + p_{ext}^{RH} \cdot (RH_{ext} - RH)
          + p_{fog}^{RH} \cdot u_{fog}
          + p_{fan}^{RH} \cdot u_{fan}
          + p_{vent}^{RH} \cdot u_{vent}
          + p_{ET}^{RH}
$$

- $p_{fog}^{RH} > 0$ → fogger raises humidity (primary humidifier)
- $p_{fan}^{RH} < 0$ → fan dries via exhaust and enhanced evaporation
- $p_{ET}^{RH}$ → constant plant evapotranspiration term

**Soil moisture** (`soil_moisture`):

$$
\Delta SM = (p_{decay}^{SM} - 1)\cdot SM
          + p_{irrig}^{SM} \cdot u_{irrig}
          - p_{ET}^{SM} \cdot \max(\hat{T}_{next} - 15,\; 0)
$$

where $\hat{T}_{next}$ is the approximated next temperature (pre-clamping).
The evapotranspiration loss scales with temperature excess above 15 °C —
hotter conditions lose more water through plant transpiration and bare-soil
evaporation.<sup>[[3]](#ref-3)</sup>

**CO₂** (`co2`):

$$
\Delta CO_2 = (p_{decay}^{CO_2} - 1)\cdot CO_2
            + p_{inj}^{CO_2} \cdot u_{CO_2}
            + p_{uptake}^{CO_2} \cdot \phi_{light}
            + p_{vent}^{CO_2} \cdot u_{vent}
            + p_{ext}^T \cdot u_{vent} \cdot (420 - CO_2)
$$

where $\phi_{light} = \min(1, I_{light} / 500)$ is a dimensionless light
availability factor (PAR proxy). Plants consume CO₂ proportionally to
available light — a process called photosynthesis.<sup>[[4]](#ref-4)</sup>
The last term models drift toward outdoor ambient CO₂ (420 ppm) when the
vent is open.

**Light** (`light_intensity`):

$$
I_{light} = p_{solar}^{light} \cdot \text{solar} + p_{LED}^{light} \cdot u_{LED}
$$

Light intensity is *memoryless* — it depends entirely on current solar
radiation and LED duty. The "delta" is expressed relative to the previous
step's intensity for consistency with other attribution outputs.

---

#### Disease Environment Flags

Four boolean flags computed from the post-step state:

| Flag | Condition | Agronomic rationale |
|------|-----------|---------------------|
| `high_humidity_risk` | $RH > 80\%$ | Most foliar fungal pathogens (powdery mildew, leaf mold) require sustained high humidity for spore germination |
| `high_leaf_wetness` | $LW > 0.5$ | Free moisture on leaf surfaces is the primary infection requirement for late blight (*Phytophthora infestans*)<sup>[[5]](#ref-5)</sup> |
| `disease_temp_band` | $18 \le T \le 25\,°C$ **AND** $RH > 80\%$ | Classic "disease triangle" — temperature + humidity jointly favourable for tomato fungal pathogens<sup>[[6]](#ref-6)</sup> |
| `fogger_disease_concern` | $u_{fog} > 0$ **AND** $\text{risk} > 0.45$ | Fogging raises humidity and leaf wetness while disease risk is already elevated — may accelerate progression |

These flags are **diagnostic only** — they do not alter the physics.
They feed into dashboard alerts and MPC constraint tightening upstream.

---

#### Resource Accounting

Actuator power draws (rated peak, kW):

| Actuator | Rated power (kW) |
|----------|-----------------|
| Fan (`fan_speed`) | 0.75 |
| Vent motor (`vent_opening`) | 0.05 |
| Heater (`heater_output`) | 5.00 |
| LED bars (`led_intensity`) | 1.20 |
| Fogger (`fogger_duty`) | 0.30 |
| CO₂ valve (`co2_valve_pct`) | 0.10 |
| Irrigation pump (`irrigation_qty`) | 0.01 |

Full formulas are given in [Section 8.5](#85-energy-and-water-accounting).

**Inputs**: `DTStepInput` (state + action + weather + crop context).  
**Outputs**: `DTStepOutput` (next state + 8 diagnostic fields).

---

### 6.4 `dt_runtime_prep.py`

**Purpose**: DB-free helper functions that build the initial state, weather
sequence, MPC solver, and `FusedState` from runtime context only.
This is the bridge between pure synthesis and the production DB-backed path.

**Key functions:**

---

#### `prepare_initial_state`

```python
def prepare_initial_state(
    growth_stage: str,
    timestamp: datetime | None = None,
    base_temp: float = 23.0,
) -> GreenhouseState:
```

Builds a physically plausible initial greenhouse state anchored to the
real wall-clock hour so simulations starting at midnight do not begin with
full midday solar radiation.

The formula is explained in [Section 8.3](#83-initial-state-temperature).

Also computes:
- `vpd` — vapour pressure deficit from temp + humidity (Antoine equation)
- `leaf_wetness_proxy` — dew-point proximity proxy
- `disease_risk_score` — sigmoid multi-factor score
- `growth_stage_index` — integer index for the solver

---

#### `prepare_weather_sequence`

```python
def prepare_weather_sequence(
    n_steps: int,
    start_time: datetime | None = None,
    dt_minutes: int = 5,
    base_temp: float = 20.0,
    diurnal_amp: float = 8.0,
) -> list[WeatherState]:
```

Generates a synthetic outdoor weather sequence aligned to the real
wall-clock hour. Full formula in [Section 8.2](#82-diurnal-weather-model).

Each `WeatherState` contains:

| Field | Units | Model |
|-------|-------|-------|
| `temp_external` | °C | Sinusoidal diurnal with peak ~15:00 |
| `humidity_external` | % RH | Anti-correlated cosine (high at night) |
| `solar_radiation` | W/m² | Bell curve between 06:00 and 20:00 |
| `windspeed` | m/s | Slow sinusoidal variation around 2.0 m/s |
| `conditions` | string | "clear" if solar > 200 W/m², else "cloudy" |
| `timestamp` | datetime | Logical simulation timestamp |

---

#### `build_fused_state`

```python
def build_fused_state(
    state: GreenhouseState,
    growth_stage: str,
    disease_risk: float,
    weather_forecast: list[dict] | None = None,
    timestamp: datetime | None = None,
) -> FusedState:
```

Assembles the `FusedState` that `MPCSolver.solve()` requires, without
touching the database. It is the DB-free equivalent of
`StateFusion.fuse()` from the production path.

Populates:
- `greenhouse_state` — current indoor climate
- `growth_stage` + `growth_stage_index` — for cost function weights
- `disease_risk_score` — fresh analytical score
- `disease_classification` — "healthy leaves" if risk < 0.3, else "early blight"
- `setpoint` — stage-aware targets from `setpoints.py`
- `constraints` — stage-aware bounds from `constraints.py`
- `weather_forecast` — 12-step look-ahead list for the solver

---

#### `build_mpc_solver`

```python
def build_mpc_solver(config: MPCConfig | None = None) -> MPCSolver:
```

Creates an `MPCSolver` with tighter horizons suitable for closed-loop
execution:

| Parameter | Production default | DT closed-loop override |
|-----------|-------------------|------------------------|
| `prediction_horizon_hours` | configured in YAML | **1 hour** |
| `control_horizon_hours` | configured in YAML | **1 hour** |
| `solver_max_iter` | configured in YAML | **300** |
| `solver_ftol` | configured in YAML | **1 × 10⁻⁵** |

Shorter horizons mean each MPC solve takes ~20–50 ms instead of several
seconds, making it practical to solve every 15 minutes in real time.

**Inputs**: Nothing required (all defaults work).  
**Outputs**: `MPCSolver` instance, `FusedState`, `GreenhouseState`, `list[WeatherState]`.

---

### 6.5 `dt_loop.py`

**Purpose**: The central orchestrator. This file is the "director" that
coordinates the DT engine, the MPC solver, the image observer, and the
output writer on a precise multi-rate cadence.

---

#### Multi-Rate Cadence

```
Step 0   → DT step + MPC solve + image refresh (all three fire)
Step 1   → DT step only
Step 2   → DT step only
Step 3   → DT step + MPC solve
Step 4   → DT step only
Step 5   → DT step only
Step 6   → DT step + MPC solve + image refresh
...
```

Expressed as constants:

```python
MPC_CADENCE_STEPS   = 3   # MPC fires every 3 steps = every 15 minutes
IMAGE_CADENCE_STEPS = 6   # Image fires every 6 steps = every 30 minutes
```

The loop checks: `mpc_due = (step % MPC_CADENCE_STEPS == 0)`.

---

#### Event-Triggered MPC Re-Solve

**Problem**: If the greenhouse state suddenly deteriorates between scheduled
MPC solves (e.g. a disease risk spike), waiting until the next 15-minute
cadence may be too late.

**Solution**: After every DT step, `should_force_mpc_update(next_state)` is
called. If any threshold is breached, the MPC solver runs immediately —
even if the cadence has not elapsed. The DT step is then re-run with the
newly computed action.

Thresholds:

| Condition | Threshold | Reason |
|-----------|-----------|--------|
| `indoor_humidity` | > 85 % | Rapid humidification needs immediate fan/vent response |
| `disease_risk_score` | > 0.55 | Pathogen-favourable conditions warrant proactive actuator changes |
| `indoor_temp` | < 12 °C or > 38 °C | Crop safety risk — outside tomato viable range |

These are accessible as module-level constants:
`_FORCE_MPC_RH_THRESH`, `_FORCE_MPC_RISK_THRESH`,
`_FORCE_MPC_TEMP_LO`, `_FORCE_MPC_TEMP_HI`.

---

#### `DTLoopStepResult`

The aggregated result of one closed-loop iteration — everything the logger
and output writer need from a single step:

```python
@dataclass
class DTLoopStepResult:
    step_index:            int
    timestamp:             datetime
    current_state:         GreenhouseState   # state BEFORE this step
    weather_used:          WeatherState      # outdoor weather this step
    next_state:            GreenhouseState   # state AFTER this step
    diagnostics:           DTDiagnostics
    snapshot:              DTSnapshot
    action_applied:        ActuatorState
    mpc_ran_this_step:     bool
    mpc_forced:            bool              # True = event-triggered, not cadence
    mpc_solution:          MPCSolution | None
    mpc_cost:              float | None
    image_refresh_this_step: bool
    image_observation:     ImageObservation | None
    cadence_info:          dict              # step_in_mpc_cycle, step_in_image_cycle, etc.
```

`to_dict()` makes the full result JSON-serialisable.

---

#### Per-Step Run Workflow

Inside `DTLoop.run()`, each step follows this sequence:

```
step k
├── Check mpc_due = (k % 3 == 0)
├── Check image_due = (k % 6 == 0)
│
├── [If mpc_due]
│   └── _run_mpc(state, stage, weather, k, ts)
│       ├── compute_disease_risk_score(state)
│       ├── weather_forecast = weather_seq[k : k+12]  (12-step look-ahead)
│       ├── build_fused_state() → FusedState
│       └── MPCSolver.solve(fused) → MPCSolution
│           └── current_action = solution.first_action
│
├── [If image_due]
│   └── ImageObserver.observe(stage, state, k, ts) → ImageObservation
│
├── Build DTStepInput(state, current_action, weather[k], ...)
├── DigitalTwinEngine.step(dt_input) → DTStepOutput (8 phases)
│
├── [If NOT mpc_due AND should_force_mpc_update(next_state)]
│   ├── _run_mpc(state, ...) → new MPCSolution
│   ├── current_action = solution.first_action
│   └── Re-run DigitalTwinEngine.step(dt_input) with new action
│
├── yield DTLoopStepResult(...)
└── state = next_state  ── advance the chain
```

**`DTLoop` constructor parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `growth_stage` | *(required)* | One of the 6 canonical stage labels |
| `start_time` | `datetime.now()` | Simulation anchor time |
| `n_steps` | 288 | Total 5-minute steps (288 = 24 hours) |
| `dt_minutes` | 5 | Step duration |
| `mpc_cadence_steps` | 3 | Steps between MPC solves (3 = 15 min) |
| `image_cadence_steps` | 6 | Steps between image refreshes (6 = 30 min) |
| `weather_base_temp` | 20.0 | Mean outdoor temperature (°C) |
| `weather_diurnal_amp` | 8.0 | Half-range of daily temperature swing (°C) |
| `input_provider` | `SyntheticInputProvider` | Plug in `DatabaseInputProvider` here |
| `image_observer` | `SyntheticImageObserver` | Plug in `MinIOImageObserver` here |
| `auto_advance_stage` | `False` | Auto-advance growth stage when `STAGE_DURATION_HOURS` elapses |
| `days_elapsed` | `0.0` | Hours already spent in the current stage (sets initial offset) |
| `session` | `None` | SQLAlchemy `Session` — when provided, auto-selects `DatabaseInputProvider` + `MinIOImageObserver` |

**Validation** (raises `ValueError` immediately in `__init__` if violated):
- `growth_stage` must be one of the 6 canonical labels
- `n_steps >= 1`
- `dt_minutes >= 1`
- `mpc_cadence_steps >= 1`
- `image_cadence_steps >= 1`

**MPC weather look-ahead**: The solver receives 12 future `WeatherState`
steps (1 hour of look-ahead) sliced from the pre-generated sequence.
This allows the solver to see upcoming solar radiation and temperature
swings when optimising the actuator trajectory.

**Inputs**: `DTInputProvider`, `ImageObserver`, configuration.  
**Outputs**: Generator of `DTLoopStepResult` (one per step).

---

### 6.6 `dt_logger.py`

**Purpose**: Lightweight step accumulator, console printer, and JSON
exporter for the DT loop. Does not depend on any database or external
service.

---

#### `DTLoopRunSummary`

Post-run aggregate statistics computed from all accumulated steps:

```python
@dataclass
class DTLoopRunSummary:
    total_steps:                 int
    total_mpc_solves:            int
    total_forced_mpc:            int     # event-triggered re-solves
    total_image_refreshes:       int
    total_energy_kwh:            float
    total_water_litres:          float
    mean_temp:                   float   # mean of next_state.indoor_temp
    mean_humidity:               float
    mean_disease_risk:           float
    min_temp:                    float   # running min
    max_temp:                    float   # running max
    min_humidity:                float
    max_humidity:                float
    min_soil_moisture:           float
    max_soil_moisture:           float
    mean_setpoint_error_temp:    float   # mean |actual − target| for temp
    mean_setpoint_error_humidity: float
    start_time:                  datetime
    end_time:                    datetime
    growth_stage:                str
```

`to_dict()` serialises all fields for JSON output. `min_temp` /
`max_temp` etc. are maintained as running min/max accumulators — no
buffering of all step values required.

---

#### `DTLoopLogger`

```python
class DTLoopLogger:
    def log_step(self, result: DTLoopStepResult) -> None: ...
    def summary(self) -> DTLoopRunSummary: ...
    def save(self, path: str | Path) -> Path: ...
```

**`log_step()`**: Called once per step by the DT loop runner. Accumulates
into running totals, stores a compact per-step dict, and optionally prints
to stdout.

**`summary()`**: Computes `DTLoopRunSummary` from all accumulated totals.
Can be called at any time during the run (not just at the end).

**`save(path)`**: Writes a JSON file with:
```json
{
  "run_summary": { ...DTLoopRunSummary fields... },
  "steps": [
    { "step": 0, "ts": "...", "T": 20.7, "RH": 61.1, "SM": 64.8, ... },
    ...
  ]
}
```

**`console_every` parameter**: Set to `1` to print every step, `3` to
print every 3rd step (quiet mode), `0` to suppress all output.

---

#### Console Output Format

Every printed step follows this format:

```
  step    0  18:54  T= 20.7°C  RH= 61.1%  SM= 64.8%  CO2= 705.9  risk=0.180 [MPC,IMG]
           ↳ act: fan=0.00 heat=0.00 vent=0.00 irr=0.00 fog=0.00
```

| Field | Meaning |
|-------|---------|
| `step` | Step index within the run |
| `HH:MM` | Logical simulation time |
| `T=` | Indoor temperature (°C) |
| `RH=` | Indoor relative humidity (%) |
| `SM=` | Soil moisture (%) |
| `CO2=` | Indoor CO₂ (ppm) |
| `risk=` | Disease risk score [0, 1] |
| `[MPC]` | MPC solve ran on cadence this step |
| `[MPC!]` | MPC solve was event-triggered |
| `[IMG]` | Image observation refresh this step |
| `↳ act:` | Actuator commands applied (only on MPC steps) |

**Inputs**: `DTLoopStepResult` objects from `DTLoop.run()`.  
**Outputs**: Console lines, in-memory summary, JSON log file.

---

### 6.7 `dt_input_provider.py`

**Purpose**: Defines the `DTInputProvider` protocol (interface) and two
concrete implementations: `SyntheticInputProvider` (offline/evaluation)
and `DatabaseInputProvider` (production, backed by PostgreSQL + AI models).
The protocol is what makes the DT layer *pluggable* — swap implementations
by changing a single constructor argument.

Also defines `ImageObservation` (the value object returned by image
refresh cycles).

---

#### `DTInputProvider` Protocol

```python
@runtime_checkable
class DTInputProvider(Protocol):
    @property
    def growth_stage(self) -> str: ...

    @property
    def start_time(self) -> datetime: ...

    def get_initial_state(self) -> GreenhouseState: ...

    def get_weather_sequence(
        self,
        n_steps: int,
        dt_minutes: int,
    ) -> list[WeatherState]: ...
```

Python `Protocol` is like an interface in Java or C#. Any class that
implements these four methods automatically satisfies the contract —
no inheritance required. The `@runtime_checkable` decorator allows
`isinstance(obj, DTInputProvider)` to work at runtime.

---

#### `SyntheticInputProvider`

The built-in implementation — uses synthetic generators:

```python
class SyntheticInputProvider:
    def __init__(
        self,
        growth_stage: str,
        start_time: datetime | None = None,
        base_temp: float = 20.0,
        diurnal_amp: float = 8.0,
    ) -> None: ...
```

- `get_initial_state()` → calls `prepare_initial_state()` from `dt_runtime_prep.py`
- `get_weather_sequence()` → calls `prepare_weather_sequence()` from `dt_runtime_prep.py`

Validates `growth_stage` against `GROWTH_STAGES` on construction and
raises `ValueError` immediately if the stage is unknown.

---

#### `DatabaseInputProvider`

Production implementation — reads live greenhouse state and uses the
AI weather-forecast model:

```python
class DatabaseInputProvider:
    def __init__(
        self,
        session: Session,
        growth_stage: str,
        weather_run_id: str | None = None,
        device: str = "cpu",
    ) -> None: ...
```

| Method | What it does |
|--------|-------------|
| `get_initial_state()` | Queries `greenhouse_data` via `MPCInputPreparation.get_latest_greenhouse_row()` → `GreenhouseState.from_db_row()`. Falls back to synthetic if table empty. |
| `get_weather_sequence()` | Queries 30 days of `weather_data` → runs `WeatherDisturbanceForecast` AI model for a 48-hour hourly forecast → expands to per-step `WeatherState` list. Falls back to sinusoidal model if DB is empty or model fails. |
| `start_time` property | Returns timestamp of the most recent `greenhouse_data` row, or `datetime.now()` as fallback. |

**Usage:**

```python
from sqlalchemy.orm import Session
from agritwin_gh.mpc import DTLoop, DatabaseInputProvider

provider = DatabaseInputProvider(session, growth_stage="flowering")
loop = DTLoop(growth_stage="flowering", input_provider=provider)
for result in loop.run(n_steps=288):
    ...
```

Exported from `agritwin_gh.mpc.__init__` so it is importable at the
package top level.

---

#### `ImageObservation`

Returned by any `ImageObserver.observe()` call:

```python
@dataclass
class ImageObservation:
    growth_stage_image_key: str   # e.g. "flowering/synthetic_0006.jpg"
    growth_stage_label:     str   # e.g. "flowering"
    disease_image_key:      str   # e.g. "healthy_leaves/synthetic_0006.jpg"
    disease_label:          str   # e.g. "healthy leaves"
    timestamp:              datetime
    source:                 str   # "synthetic" | "minio"
```

Mirrors the shape of `ImagePayload` from `state.py` but is a standalone
value object specific to the DT layer. `source = "synthetic"` is set by
`SyntheticImageObserver`; a production `MinIOImageObserver` would set
`source = "minio"`.

---

### 6.8 `dt_image_observer.py`

**Purpose**: Decouples image observation from DT state physics. The
observer fires at image-refresh cadence (every 30 minutes) and produces
`ImageObservation` metadata — it never modifies the greenhouse state.

**Protocol:**

```python
@runtime_checkable
class ImageObserver(Protocol):
    def observe(
        self,
        growth_stage: str,
        state: GreenhouseState,
        step_index: int,
        timestamp: datetime,
    ) -> ImageObservation: ...
```

**`SyntheticImageObserver`** — the built-in implementation:

1. Looks up the growth-stage image subcategory from `GROWTH_STAGE_IMAGE_SUBCATEGORY`
   (constant map in `constants.py`) and builds a synthetic key like
   `"flowering/synthetic_0006.jpg"`.
2. Derives the disease label from the current `disease_risk_score`:
   - risk < 0.3 → `"healthy leaves"`
   - 0.3 ≤ risk < 0.5 → `"early blight"`
   - risk ≥ 0.5 → `"late blight"`
3. Returns an `ImageObservation` with `source = "synthetic"`.

**`MinIOImageObserver`** — production implementation (DB + MinIO):

```python
class MinIOImageObserver:
    def __init__(
        self,
        session: Session,
        cache_ttl_sec: float = 300.0,
    ) -> None: ...
```

1. Wraps `ImageStreamer(session, cache_ttl_sec=...)` — the existing
   DB-backed image service with per-category TTL caching.
2. On `observe()`, derives the disease label from the current risk score
   (same thresholds as `SyntheticImageObserver`), then calls
   `ImageStreamer.get_random_growth_stage_image(growth_stage)` and
   `ImageStreamer.get_random_disease_image(disease_label)`.
3. If either image is missing (empty table, unknown category),
   falls back transparently to `SyntheticImageObserver` for that step
   and logs the fallback.
4. Returns `ImageObservation(source="minio")`.

**Usage:**

```python
# Explicit observer:
loop = DTLoop(growth_stage="flowering",
              image_observer=MinIOImageObserver(session))

# Or let DTLoop auto-select when a session is provided:
loop = DTLoop(growth_stage="flowering", session=db_session)
#  → auto-creates MinIOImageObserver(session) internally
```

**Inputs**: Growth stage label, current state, step index, timestamp.  
**Outputs**: `ImageObservation` value object.

---

### 6.9 `dt_output_writer.py`

**Purpose**: Separates the simulation results into four independent
artifact streams and defines the `DTOutputWriter` protocol so a future
database writer can replace the file writer with zero loop changes.

**Four streams:**

| Stream | Written by | What it contains |
|--------|-----------|-----------------|
| State log | `write_state_step()` | Per-step: `current_state` → `weather` → `next_state` (6 key variables each) |
| MPC actions | `write_mpc_action()` | Per MPC-solve step: all 7 actuator values + MPC cost |
| Diagnostics | `write_diagnostics()` | Per-step: energy, water, effect attribution, disease flags |
| Summary | `write_summary()` | Run-level aggregates (one entry per run) |

**`DTOutputWriter` Protocol:**

```python
@runtime_checkable
class DTOutputWriter(Protocol):
    def write_state_step(self, record: dict) -> None: ...
    def write_mpc_action(self, record: dict) -> None: ...
    def write_diagnostics(self, record: dict) -> None: ...
    def write_summary(self, summary: dict) -> None: ...
    def flush(self) -> dict[str, str]: ...  # returns {stream_name: path}
```

**`JsonFileOutputWriter`** — the built-in implementation:

Buffers all records in memory during the run, then on `flush()` writes:

```
dt_state_<tag>.json       → list of state-step dicts
dt_mpc_actions_<tag>.json → list of MPC action dicts
dt_diagnostics_<tag>.json → list of diagnostic dicts
dt_summary_<tag>.json     → run summary dict
```

**`fanout_step_to_writer(result, writer)`**:

This free function decomposes one `DTLoopStepResult` into the four
streams and routes each part to the correct writer method. Keeps the
loop code clean — it just calls `fanout_step_to_writer(result, writer)`
once per step.

State record structure (one entry in `dt_state_run.json`):
```json
{
  "step": 3,
  "ts": "2026-04-02 19:09:00",
  "current_state": { "indoor_temp": 20.7, "indoor_humidity": 61.1, ... },
  "weather": { "temp_external": 18.2, "solar_radiation": 320.0, ... },
  "next_state": { "indoor_temp": 18.9, "indoor_humidity": 53.1, ..., "vpd": 1.234 }
}
```

**Inputs**: `DTLoopStepResult` objects.  
**Outputs**: 4 JSON files per run.

---

### 6.10 `dt_artifact_manager.py`

**Purpose**: Creates a uniquely named run folder for every simulation run
and provides a factory for the per-step writer. Every run is isolated —
no two runs overwrite each other.

**Class: `DTArtifactManager`**

```python
class DTArtifactManager:
    def __init__(
        self,
        base_dir: str | Path = "logs/dt_runs",
        run_id: str | None = None,
    ) -> None: ...

    @property
    def run_id(self) -> str: ...         # e.g.  "dt_run_20260402_185434"
    @property
    def run_dir(self) -> Path: ...       # e.g.  logs/dt_runs/dt_run_20260402_185434/

    def create_output_writer(self) -> JsonFileOutputWriter: ...
    def save_run_metadata(self, metadata: dict) -> Path: ...
    def save_summary(self, summary: dict) -> Path: ...
```

**`create_output_writer()`**: Returns a `JsonFileOutputWriter` pointed at
the run directory, using `"run"` as the file tag. This produces the
consistently named files:

```
dt_state_run.json
dt_mpc_actions_run.json
dt_diagnostics_run.json
dt_summary_run.json
```

**`save_run_metadata(metadata)`**: Writes `run_metadata.json` in the run
folder. Typically populated with: growth stage, base temp, hours,
`dt_minutes`, cadence settings, force-MPC thresholds, CLI arguments.

**`save_summary(summary)`**: Writes `summary.json` — the enriched
`DTLoopRunSummary.to_dict()` output with min/max ranges and tracking errors.

**Run ID convention**: `dt_run_YYYYMMDD_HHMMSS` — matches the pattern used
by `save_evaluation_artifacts()` in `evaluation.py` for consistency across
the whole repository.

**Inputs**: Configuration dict, `DTLoopRunSummary`.  
**Outputs**: Run folder with 6 files (see [Section 13](#13-artifact-output-layout)).

---

### 6.11 `realtime_core.py` (core engine)

**Purpose**: All reusable closed-loop DT + MPC orchestration logic in one
importable module. Lives in `src/agritwin_gh/mpc/` so it can be consumed
by production services, web backends, notebooks, or evaluation harnesses
without any CLI / console / signal dependencies.

**Design principle**: No `argparse`, no `print`, no `signal` — pure
business logic only.

**Key exports:**

| Name | Kind | Role |
|------|------|------|
| `RealtimeLoopConfig` | `@dataclass` | All parameters for one session: `growth_stage`, `days_elapsed`, `total_steps`, `mpc_every`, `device`, `no_images`, `dry_run`, `run_id`, `start_ts`, `auto_advance_stage` |
| `RealtimeLoop` | class | Orchestrator — call `setup()`, then `run()` or individual `step(i)` calls |
| `RealtimeStepResult` | `@dataclass` | Output of each `step()` call: payload, hours, energy, wall time |
| `RealtimeRunSummary` | `@dataclass` | Returned by `run()` / `finish()`: totals over all steps |
| `RealtimeMPCInputPreparation` | class | `MPCInputPreparation` subclass that reads `realtime_greenhouse_stream` |
| `ensure_stream_table()` | function | Creates `realtime_greenhouse_stream` if absent |
| `seed_initial_state()` | function | Writes step-0 bootstrap row from `greenhouse_data` (or diurnal defaults) |
| `write_step_to_stream()` | function | Persists one `DigitalTwinStepPayload` as a DB row |
| `estimate_energy()` | function | kWh estimate for one DT step from actuator duty cycles |
| `STAGE_DURATION_HOURS` | dict | Canonical stage durations (hours) |
| `PRIOR_STAGE_HOURS` | dict | Cumulative hours before each stage |
| `RunRegistry` | class | Persistent JSON index of all runs in `logs/realtime/registry.json` |

**`RealtimeLoop` API:**

```python
from agritwin_gh.mpc.realtime_core import RealtimeLoop, RealtimeLoopConfig

cfg = RealtimeLoopConfig(
    growth_stage="flowering",
    days_elapsed=5.0,
    total_steps=288,     # 24 h
    mpc_every=3,
)
loop = RealtimeLoop(cfg, session=db_session)
loop.setup()             # seeds bootstrap DB row
summary = loop.run(
    on_step=my_callback,         # called after each step
    should_stop=lambda: flag,    # Ctrl-C / cancel support
)
```

Or step-by-step:

```python
loop.setup()
for i in range(1, cfg.total_steps + 1):
    result: RealtimeStepResult = loop.step(i)
    # result.payload, result.wall_time_ms, result.cumulative_energy_kwh …
summary = loop.finish()
```

**Data flow per step (inside `RealtimeLoop.step()`):**

```
realtime_greenhouse_stream (DB)
  → RealtimeMPCInputPreparation.get_latest_greenhouse_row()
  → StateFusion.fuse()          [9-step AI pipeline]
  → MPCSolver.solve()           → ActuatorState
  → DigitalTwinEngine.step()    → DTStepOutput
  → current_state updated
  → write_step_to_stream()      → realtime_greenhouse_stream (DB)
```

---

### 6.12 `run_realtime_loop.py` (CLI)

**Purpose**: Thin CLI / test-evaluation harness that wraps `RealtimeLoop`.
Located in `scripts/` because it is an executable, not a library.
Contains **only** console / filesystem concerns — all domain logic is
imported from `realtime_core`.

**What lives here** (and only here):

| Responsibility | Functions |
|---|---|
| Interactive prompts | `prompt_growth_stage()`, `prompt_days_elapsed()`, `prompt_steps()` |
| Console display | `print_banner()`, `print_step_line()`, `print_run_summary()` |
| Artifact saving | `setup_artifact_dir()`, `save_step_artifact()`, `save_run_manifest()` |
| Signal handling | `_handle_signal()`, `_stop_requested` flag |
| CLI wiring | `build_parser()`, `run_from_cli()`, `main()` |

**CLI → core wiring (inside `run_from_cli()`):**

```python
config = RealtimeLoopConfig(growth_stage=..., days_elapsed=..., ...)
loop   = RealtimeLoop(config, session, disease_classifier=..., growth_classifier=...)
loop.setup()
summary = loop.run(on_step=on_step, should_stop=should_stop)
```

**Requires**: A running PostgreSQL instance with `agritwin_db` reachable
and populated `greenhouse_data` / `weather_data` tables (see
[Section 18](#18-production-closed-loop--implemented) and
`docs/POSTGRESQL_QUICKSTART.md`).

**Run identity**: `run_id = f"rt_{now:%Y%m%d_%H%M%S}"` — e.g. `rt_20260402_185434`

**Graceful shutdown**: Handles `SIGINT` / `SIGTERM` via `_stop_requested` flag
passed as `should_stop` callback to `loop.run()`.

**`--show-delete-query`**: Prints SQL `DELETE` / `TRUNCATE` statements for
`realtime_greenhouse_stream` and exits.

---

## 7. Key Data Structures

### 7.1 `DTLoopStepResult`

The primary output of each loop iteration. Aggregates all DT engine,
MPC, and image observer outputs into a single object.

| Field | Type | Description |
|-------|------|-------------|
| `step_index` | `int` | Step number (0-based) |
| `timestamp` | `datetime` | Logical simulation time |
| `current_state` | `GreenhouseState` | State **before** this step |
| `weather_used` | `WeatherState` | Outdoor weather during this step |
| `next_state` | `GreenhouseState` | State **after** this step |
| `diagnostics` | `DTDiagnostics` | Energy, water, attribution, flags |
| `snapshot` | `DTSnapshot` | Full DT snapshot at this moment |
| `action_applied` | `ActuatorState` | Actuator commands used this step |
| `mpc_ran_this_step` | `bool` | True if MPC solved on cadence |
| `mpc_forced` | `bool` | True if MPC was event-triggered |
| `mpc_solution` | `MPCSolution` or `None` | Full solver result |
| `mpc_cost` | `float` or `None` | Total objective function value |
| `image_refresh_this_step` | `bool` | True if image observer fired |
| `image_observation` | `ImageObservation` or `None` | Image metadata |
| `cadence_info` | `dict` | `{mpc_due, mpc_forced, image_due, step_in_mpc_cycle, step_in_image_cycle}` |

---

### 7.2 `DTDiagnostics`

Detailed per-step reporting bundle. The most information-dense structure
in the DT layer.

| Field | Type | Description |
|-------|------|-------------|
| `energy_kwh` | `float` | Energy consumed this 5-min step |
| `water_litres` | `float` | Water used (irrigation + fogging) |
| `disease_risk_recomputed` | `float` | Fresh risk score from post-step state |
| `state_delta` | `dict` | `{var: new − old}` for all climate variables |
| `bounds_clamped` | `list[str]` | Variables that hit physical bounds |
| `setpoint_error` | `dict` | `{var: actual − target}` |
| `effect_attribution` | `dict[str, dict]` | Per-actuator contribution breakdown |
| `disease_environment_flags` | `dict[str, bool]` | 4 pathogen risk conditions |
| `step_compute_ms` | `float` | Time to compute this step (wall clock) |

---

### 7.3 `DTLoopRunSummary`

Post-run aggregate statistics. Computed from all accumulated step records
without buffering every individual step value.

| Field | Type | Description |
|-------|------|-------------|
| `total_steps` | `int` | Number of 5-min steps run |
| `total_mpc_solves` | `int` | Cadence + forced MPC solves |
| `total_forced_mpc` | `int` | Event-triggered re-solves only |
| `total_image_refreshes` | `int` | Image observer calls |
| `total_energy_kwh` | `float` | Sum of all per-step energy |
| `total_water_litres` | `float` | Sum of all per-step water |
| `mean_temp` | `float` | Mean post-step indoor temp |
| `mean_humidity` | `float` | Mean post-step indoor humidity |
| `mean_disease_risk` | `float` | Mean post-step risk score |
| `min_temp` / `max_temp` | `float` | Temperature range |
| `min_humidity` / `max_humidity` | `float` | Humidity range |
| `min_soil_moisture` / `max_soil_moisture` | `float` | Soil moisture range |
| `mean_setpoint_error_temp` | `float` | Mean |actual − target| for temperature |
| `mean_setpoint_error_humidity` | `float` | Mean |actual − target| for humidity |
| `start_time` / `end_time` | `datetime` | Logical simulation window |
| `growth_stage` | `str` | Stage label for this run |

---

### 7.4 `ImageObservation`

Value object produced by the image observer at 30-minute refresh intervals.

| Field | Type | Example |
|-------|------|---------|
| `growth_stage_image_key` | `str` | `"flowering/synthetic_0006.jpg"` |
| `growth_stage_label` | `str` | `"flowering"` |
| `disease_image_key` | `str` | `"healthy_leaves/synthetic_0006.jpg"` |
| `disease_label` | `str` | `"healthy leaves"` |
| `timestamp` | `datetime` | Logical simulation time |
| `source` | `str` | `"synthetic"` or `"minio"` |

---

## 8. Mathematical Formulas — Complete Reference

### 8.1 Disease Risk Score

The disease risk score is a weighted sigmoid combination of three factors.
It is computed in `constants.py` by `compute_disease_risk_score()` and
called in two places: `dt_runtime_prep.prepare_initial_state()` and
`dt_engine._recompute_disease_risk()`.

Define the sigmoid function: $\sigma(x, c, k) = \dfrac{1}{1 + e^{-k(x-c)}}$

**Three sub-scores:**

$$
s_{RH} = \sigma(RH,\; 75,\; 0.2)
\qquad \text{(humidity risk, centre 75\% RH)}
$$

$$
s_{LW} = \sigma(LW,\; 0.5,\; 8.0)
\qquad \text{(leaf-wetness risk, centre 0.5)}
$$

$$
s_{T} = \sigma(T,\; 22,\; 0.15)
\qquad \text{(temperature risk, centre 22°C)}
$$

**Weighted combination:**

$$
r_{base} = 0.40 \cdot s_{RH} + 0.35 \cdot s_{LW} + 0.25 \cdot s_{T}
$$

**Growth-stage modifier** (amplifies risk during vulnerable stages):

$$
r_{final} = r_{base} \times \begin{cases}
  1.2 & \text{if stage} \in \{\text{flowering}, \text{unripe}\} \\
  1.1 & \text{if stage} = \text{ripe} \\
  1.0 & \text{otherwise}
\end{cases}
$$

Result is clamped to $[0, 1]$.

**Why these weights?** Humidity and leaf wetness are the dominant drivers
of tomato fungal disease — free water on leaf surfaces enables spore
germination and mycelial growth.<sup>[[5]](#ref-5)</sup> Temperature
contributes but is secondary.<sup>[[6]](#ref-6)</sup>

*↗ `constants.py · compute_disease_risk_score() · L204`  (`sigmoid()` helper · L175)*

---

### 8.2 Diurnal Weather Model

Used in `prepare_weather_sequence()`. All formulas are anchored to the
real wall-clock hour so the simulation starts with time-of-day-correct
weather.

Let $h_0$ = start hour (fractional, e.g. 18.5 = 18:30).  
At step $i$: current hour fraction $h_i = (h_0 + i \cdot \Delta t_{min}/60) \bmod 24$

**Phase angle** (peak at 15:00):

$$
\phi_i = \frac{2\pi (h_i - 6)}{24}
$$

**Outdoor temperature** (sinusoidal diurnal):

$$
T_{ext,i} = T_{base} + A_{diurnal} \cdot \sin(\phi_i)
$$

where $T_{base}$ = base temperature (default 20 °C), $A_{diurnal}$ =
half-range amplitude (default 8 °C). Peak at ~15:00, minimum at ~03:00.

**Outdoor humidity** (anti-correlated with temperature):

$$
RH_{ext,i} = \text{clip}\!\left(60 + 15 \cdot \cos(\phi_i),\; 30,\; 95\right)
$$

Higher humidity at night/dawn when temperature is low — matches
observed meteorological patterns.<sup>[[7]](#ref-7)</sup>

**Solar radiation** (half-sine between 06:00 and 20:00):

$$
G_{solar,i} = \begin{cases}
  800 \cdot \sin\!\left(\dfrac{\pi (h_i - 6)}{14}\right) & \text{if } 6 \le h_i \le 20 \\
  0 & \text{otherwise}
\end{cases}
$$

**Wind speed** (slow oscillation):

$$
v_{wind,i} = 2.0 + 1.5 \cdot |\sin(\phi_i)|
$$

*↗ `dt_runtime_prep.py · prepare_weather_sequence() · L116`  (real-time, anchored to wall-clock hour)*  
*↗ `experiment_runner.py · generate_default_weather() · L502`  (offline evaluation, starts from hour 0)*

---

### 8.3 Initial State Temperature

Used in `prepare_initial_state()`. The indoor temperature follows a
mild diurnal variation around the base temperature, lagged slightly
from the outdoor profile because greenhouses have thermal mass.

$$
T_{init} = T_{base} + 2 \cdot \sin\!\left(\frac{2\pi (h - 6)}{24}\right)
$$

$h$ = current hour fraction. Amplitude ±2 °C reflects the fact that a
well-managed greenhouse has less temperature variation than outdoors.

**Indoor light** follows a bell curve only during daylight (06:00–20:00):

$$
I_{init} = \begin{cases}
  350 \cdot \sin\!\left(\dfrac{\pi (h - 6)}{14}\right) & \text{if } 6 \le h \le 20 \\
  0 & \text{otherwise}
\end{cases}
$$

The constant humidity = 65 % RH is a typical well-managed greenhouse
condition at startup before the control loop takes effect.

*↗ `dt_runtime_prep.py · prepare_initial_state() · L57`  (temperature formula · L68, light formula · L72)*  
*↗ `experiment_runner.py · make_default_initial_state() · L570`  (offline fixed-value default)*

---

### 8.4 Effect Attribution Decomposition

Full derivation of all attribution terms. See also [Section 6.3](#63-dt_enginepy).

All terms are pre-clamp estimates. Actual deltas may differ where
physical bounds are active (e.g. CO₂ cannot fall below 300 ppm).

**Notation**: $p$ = `GreenhouseModelParams`, $u$ = actuator, state
variable subscript identifies the variable.

**Temperature:**

| Source | Formula |
|--------|---------|
| Natural decay | $(p^T_{decay} - 1) \cdot T$ |
| Weather exchange | $p^T_{ext} \cdot (T_{ext} - T)$ |
| Solar heating | $p^T_{solar} \cdot G_{solar}$ |
| Heater | $p^T_{heat} \cdot u_{heat}$ |
| Fan cooling | $p^T_{fan} \cdot u_{fan}$ |
| Vent cooling | $p^T_{vent} \cdot u_{vent}$ |

**Humidity:**

| Source | Formula |
|--------|---------|
| Natural decay | $(p^{RH}_{decay} - 1) \cdot RH$ |
| Weather exchange | $p^{RH}_{ext} \cdot (RH_{ext} - RH)$ |
| Fogger | $p^{RH}_{fog} \cdot u_{fog}$ |
| Fan drying | $p^{RH}_{fan} \cdot u_{fan}$ |
| Vent drying | $p^{RH}_{vent} \cdot u_{vent}$ |
| Evapotranspiration | $p^{RH}_{ET}$ (constant) |

**Soil moisture:**

| Source | Formula |
|--------|---------|
| Natural drying | $(p^{SM}_{decay} - 1) \cdot SM$ |
| Irrigation | $p^{SM}_{irrig} \cdot u_{irrig}$ |
| Evapotranspiration | $-p^{SM}_{ET} \cdot \max(\hat{T}_{next} - 15, 0)$ |

**CO₂:**

| Source | Formula |
|--------|---------|
| Natural decay | $(p^{CO_2}_{decay} - 1) \cdot CO_2$ |
| CO₂ injection | $p^{CO_2}_{inj} \cdot u_{CO_2}$ |
| Plant uptake | $p^{CO_2}_{uptake} \cdot \min(1, I_{light}/500)$ |
| Vent loss | $p^{CO_2}_{vent} \cdot u_{vent}$ |
| Ambient drift | $p^T_{ext} \cdot u_{vent} \cdot (420 - CO_2)$ |

**Light intensity:**

| Source | Formula |
|--------|---------|
| Solar fraction | $p^{light}_{solar} \cdot G_{solar}$ |
| LED contribution | $p^{light}_{LED} \cdot u_{LED}$ |
| Previous offset | $-I_{light}$ (to express as delta) |

*↗ `dt_engine.py · _compute_effect_attribution() · L336`*

---

### 8.5 Energy and Water Accounting

Used in `dt_engine._compute_resource_usage()`.

**Energy (kWh per 5-minute step):**

$$
E_{step} = \Delta t_{hours} \cdot \sum_{a} P_a \cdot u_a
$$

where $\Delta t_{hours} = 5/60$ and $P_a$ is the rated power (kW) of
actuator $a$, and $u_a \in [0,1]$ is the duty fraction.

| Actuator $a$ | $P_a$ (kW) |
|-------------|-----------|
| `fan_speed` | 0.75 |
| `vent_opening` | 0.05 |
| `heater_output` | 5.00 |
| `led_intensity` | 1.20 |
| `fogger_duty` | 0.30 |
| `co2_valve_pct` | 0.10 |
| `irrigation_qty` | 0.01 |

Example: heater at 50% duty for one 5-minute step:
$E = (5/60) \times 5.0 \times 0.5 = 0.208 \text{ kWh}$

**Water (litres per 5-minute step):**

$$
W_{step} = u_{irrig} \times 1.0 + u_{fog} \times 2.0
$$

- Drip irrigation: 1.0 L per unit of `irrigation_qty`
- Fogger: 2.0 L per step at full duty (proportional to `fogger_duty`)

This model is consistent with `MPCRunner._estimate_energy()` in
`runner.py` to ensure the DT layer reports the same resource costs
as the full production path.<sup>[[8]](#ref-8)</sup>

*↗ `dt_engine.py · _compute_resource_usage() · L541`*

---

## 9. Cadence Reference

| Event | Period | Condition | What fires |
|-------|--------|-----------|-----------|
| DT step | Every 5 min | Every step | `DigitalTwinEngine.step()` |
| MPC solve (scheduled) | Every 15 min | `step % 3 == 0` | `MPCSolver.solve()` → new actuator command |
| MPC re-solve (event) | Any step | `should_force_mpc_update()` = True | Immediate re-solve + re-step |
| Image observation | Every 30 min | `step % 6 == 0` | `ImageObserver.observe()` |

**Total MPC solves in a 24-hour run** (288 steps, no forced re-solves):
$288 / 3 = 96$ scheduled solves.

**MPC look-ahead window**: 12 future weather steps = 1 hour of forecast.
This is the weather slice `weather_seq[k : k+12]` passed to the solver
at each MPC step.

---

## 10. Event-Trigger Thresholds

Defined as module-level constants in `dt_loop.py`:

```python
_FORCE_MPC_RH_THRESH:   float = 85.0   # % relative humidity
_FORCE_MPC_RISK_THRESH: float = 0.55   # disease risk score [0,1]
_FORCE_MPC_TEMP_LO:     float = 12.0   # °C lower safety bound
_FORCE_MPC_TEMP_HI:     float = 38.0   # °C upper safety bound
```

Any **one** of these conditions being true triggers an early MPC
re-solve and a re-run of the DT step with the new action.

The temperature bounds [12, 38] °C correspond to the outer survival
range for *Solanum lycopersicum* (tomato). Below 12 °C, chilling injury
occurs; above 38 °C, pollen viability drops to near zero.<sup>[[1]](#ref-1)</sup>

---

## 11. Disease Environment Flag Thresholds

Defined as module-level constants in `dt_engine.py`:

```python
_HIGH_HUMIDITY_THRESH:    float = 80.0   # % RH
_HIGH_LEAF_WETNESS_THRESH: float = 0.5   # leaf-wetness proxy [0,1]
_DISEASE_TEMP_LO:         float = 18.0   # °C
_DISEASE_TEMP_HI:         float = 25.0   # °C
_FOGGER_RISK_THRESH:      float = 0.45   # disease risk score
```

The [18, 25] °C temperature band is the optimal growth range for most
tomato foliar fungal pathogens — powdery mildew (*Leveillula taurica*,
*Oidium neolycopersici*), early blight (*Alternaria solani*), and leaf
mold (*Fulvia fulva*).<sup>[[6]](#ref-6)</sup>

---

## 12. How to Run

### 12.1 Prerequisites

- Python 3.10+
- All packages from `requirements.txt` installed
- PostgreSQL `agritwin_db` running and reachable (see `docs/POSTGRESQL_QUICKSTART.md`)
- Schema applied:

```powershell
psql -U <user> -d agritwin_db -f database/schema/timeseries_data.sql
psql -U <user> -d agritwin_db -f database/schema/image_metadata.sql
```

- `.env` file at the repo root with `DATABASE_URL` set (see `config/settings.local.example.yaml`)

Install dependencies:

```bash
pip install -r requirements.txt
```

### 12.2 Environment Setup

The `src/` directory must be on the Python path. From the repo root:

```bash
# Windows PowerShell
$env:PYTHONPATH = "src"

# Linux / macOS / Git Bash
export PYTHONPATH=src
```

`run_realtime_loop.py` automatically adds `src/` to `sys.path` when run
from the `scripts/` directory, so explicit `PYTHONPATH` setting is only
required for the smoke test.

### 12.3 Interactive Mode

Launches a menu to select the growth stage and prompts for days elapsed
and number of steps:

```bash
python scripts/run_realtime_loop.py
```

Output:
```
╔══════════════════════════════════════════════════════════╗
║          AgriTwin-GH  Real-Time Closed-Loop MPC          ║
╚══════════════════════════════════════════════════════════╝

── Current Growth Stage ───────────────────────────────
  1. Seedling
  2. Early Vegetative
  3. Flowering Initiation
  4. Flowering
  5. Unripe
  6. Ripe

  Enter stage number (1–6): 4
```

### 12.4 Non-Interactive Mode

```bash
# 24-hour run (288 steps × 5 min), flowering stage, MPC every 15 min
python scripts/run_realtime_loop.py --stage flowering --steps 288

# 7-day run with GPU weather model
python scripts/run_realtime_loop.py --stage unripe --steps 2016 --device cuda

# MPC every step (aggressive, 5-minute re-solve)
python scripts/run_realtime_loop.py --stage flowering --steps 72 --mpc-every 1

# Dry run — simulate without writing to DB or saving artifacts
python scripts/run_realtime_loop.py --stage flowering --steps 12 --dry-run

# Skip image classifiers (faster, no MinIO required)
python scripts/run_realtime_loop.py --stage ripe --steps 144 --no-images

# Show DELETE queries for stream table cleanup, then exit
python scripts/run_realtime_loop.py --show-delete-query

# Smoke test (synthetic, no DB required)
$env:PYTHONPATH = "src"
python tests/smoke_test_dt_loop.py
```

### 12.5 Cleanup After a Run

After testing, remove accumulated rows from the stream table:

```sql
-- Delete a specific run
DELETE FROM realtime_greenhouse_stream WHERE run_id = 'rt_20260402_185434';

-- Delete all rows older than 7 days
DELETE FROM realtime_greenhouse_stream
WHERE created_at < NOW() - INTERVAL '7 days';

-- Full table reset
TRUNCATE TABLE realtime_greenhouse_stream RESTART IDENTITY;
```

Or use the built-in helper to print these queries:

```bash
python scripts/run_realtime_loop.py --show-delete-query
```

### 12.6 Full CLI Flag Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--steps` | `int` | `0` (interactive) | Number of 5-minute simulation steps. 288 = 24 h, 2016 = 7 days |
| `--stage` | `str` | *(interactive)* | Growth stage. Options: `seedling`, `early vegetative`, `flowering initiation`, `flowering`, `unripe`, `ripe` |
| `--days-elapsed` | `float` | *(interactive)* | Days already elapsed in the current stage (used to compute stage progress) |
| `--mpc-every` | `int` | `3` | Solve MPC every N steps. Default 3 = every 15 min. Set 1 for every 5 min. |
| `--device` | `str` | `cpu` | PyTorch device for weather-forecast ensemble: `cpu`, `cuda`, or `mps` |
| `--no-images` | flag | off | Disable image-classification step (skips MinIO, faster) |
| `--dry-run` | flag | off | Run without writing to DB or saving artifacts (safe for testing) |
| `--show-delete-query` | flag | off | Print SQL DELETE queries for the stream table and exit |

---

## 13. Artifact Output Layout

### 13.1 Run Folder Structure

Every production run creates a uniquely named folder and a corresponding
set of rows in the PostgreSQL stream table:

```
logs/
└── realtime/
    └── rt_20260402_185434/
        ├── manifest.json             ← run configuration and summary
        └── steps.ndjson              ← one compact JSON record per step (NDJSON)

PostgreSQL:
└── realtime_greenhouse_stream
    └── rows with run_id = 'rt_20260402_185434'   ← full state + actuators + diagnostics
```

The synthetic DT loop (`DTLoop` + `JsonFileOutputWriter`) continues to
produce the original four-file layout under `logs/dt_runs/`:

```
logs/
└── dt_runs/
    └── dt_run_20260402_185434/
        ├── run_metadata.json         ← configuration snapshot
        ├── dt_state_run.json         ← per-step climate state
        ├── dt_mpc_actions_run.json   ← actuator commands (MPC steps only)
        ├── dt_diagnostics_run.json   ← energy, water, attribution, flags
        ├── dt_summary_run.json       ← writer-level summary
        └── summary.json              ← enriched run-level summary (min/max/errors)

logs/
└── dt_loop_20260402_185434.json      ← backward-compatible combined log
```

### 13.2 File Contents

**`run_metadata.json`** — configuration and thresholds used for this run:
```json
{
  "run_id": "dt_run_20260402_185434",
  "growth_stage": "flowering",
  "hours": 1.0,
  "n_steps": 12,
  "dt_minutes": 5,
  "weather_base_temp": 22.0,
  "mpc_cadence_steps": 3,
  "image_cadence_steps": 6,
  "force_mpc_rh_thresh": 85.0,
  "force_mpc_risk_thresh": 0.55,
  "force_mpc_temp_lo": 12.0,
  "force_mpc_temp_hi": 38.0
}
```

**`dt_state_run.json`** — array of per-step state records (current → weather → next):
```json
[
  {
    "step": 0,
    "ts": "2026-04-02 18:54:34",
    "current_state": { "indoor_temp": 20.7, "indoor_humidity": 61.1, ... },
    "weather": { "temp_external": 18.2, "solar_radiation": 320.0, ... },
    "next_state": { "indoor_temp": 18.9, "indoor_humidity": 53.1, "vpd": 1.234, ... }
  }, ...
]
```

**`dt_mpc_actions_run.json`** — actuator commands only on MPC-solve steps:
```json
[
  {
    "step": 0, "ts": "...",
    "fan_speed": 0.0, "heater_output": 0.25, "vent_opening": 0.0,
    "irrigation_qty": 0.0, "fogger_duty": 0.20,
    "mpc_cost": 49.45, "mpc_forced": false
  }, ...
]
```

**`dt_diagnostics_run.json`** — energy, water, attribution per step:
```json
[
  {
    "step": 0, "energy_kwh": 0.002083, "water_litres": 0.0,
    "disease_risk_recomputed": 0.130,
    "state_delta": { "indoor_temp": -1.8, "indoor_humidity": -8.0, ... },
    "effect_attribution": {
      "indoor_temp": { "heater": 2.25, "natural_decay": -0.45, ... }
    },
    "disease_environment_flags": {
      "high_humidity_risk": false, "fogger_disease_concern": false, ...
    }
  }, ...
]
```

**`summary.json`** — enriched run-level summary with min/max and tracking errors:
```json
{
  "total_steps": 12,
  "total_mpc_solves": 4,
  "mean_temp": 18.19,
  "min_temp": 16.5,
  "max_temp": 20.7,
  "mean_setpoint_error_temp": 2.81,
  "mean_setpoint_error_humidity": 7.48,
  "total_energy_kwh": 1.9276,
  "total_water_litres": 5.31
}
```

---

## 14. Test Script

**`tests/smoke_test_dt_loop.py`** — 12-section, 64-check comprehensive
smoke test covering the full DT stack.

```bash
$env:PYTHONPATH = "src"
python tests/smoke_test_dt_loop.py
```

| Section | What it tests |
|---------|--------------|
| 1. Imports | All DT module imports succeed |
| 2. SyntheticInputProvider | growth_stage, start_time, initial state validity, weather length, field names |
| 3. DT engine single step | Returns `DTStepOutput`, `GreenhouseState`, diagnostics have expected fields |
| 4. DTLoop construction + validation | Constructs successfully, invalid stage raises `ValueError`, negative n_steps raises `ValueError` |
| 5. Multi-step execution | 12 steps run, MPC fires on cadence, state evolves, `current_state` and `weather_used` populated |
| 6. Image observer | Fires on cadence, `ImageObservation` has `growth_stage_label`, `disease_label`, `source = "synthetic"` |
| 7. Output writer + fanout | 4 artifact files created, state record has `current_state`, `weather`, `next_state` sub-objects |
| 8. DTArtifactManager | `run_id` set, `run_dir` created, `run_metadata.json` saved, `summary.json` saved |
| 9. DTLoopRunSummary enriched fields | All min/max/error fields present and valid |
| 10. Error handling | Bad stage rejected, force MPC fires on extreme state, not on safe state |
| 11. Schema round-trip | `to_dict()` returns a `dict`, has all expected keys, JSON-serialisable |
| 12. FusedState ↔ MPCSolver | `build_fused_state()` produces valid `FusedState`, `MPCSolver.solve()` returns `MPCSolution` |

Expected output:
```
============================================================
  DT Loop — Comprehensive Smoke Test
============================================================
  ...
============================================================
  Results: 64/64 passed, 0 failed
  STATUS: ALL CHECKS PASSED
============================================================
```

---

## 15. Assumptions & Design Decisions

### 15.1 Design Decisions

**1. No physics duplication.**  
`DigitalTwinEngine` delegates all state transition computation to
`GreenhouseTransitionModel.step()`. It never reimplements the ARX
equations. This guarantees that DT predictions and MPC predictions
are always consistent — they use the same model.

**2. Pluggable interfaces.**  
`DTInputProvider`, `ImageObserver`, and `DTOutputWriter` are Python
`Protocol` types — structural interfaces that any compatible class
satisfies without inheritance. This makes swapping from synthetic to
production sources a local change (one constructor argument).

**3. Multi-rate cadence, single thread.**  
The three rates (5 min DT, 15 min MPC, 30 min image) are managed by
integer modulo checks inside a single `for` loop. No threads, no background
processes. This makes the loop deterministic and easy to debug. For real-time
deployment, each rate can be moved to its own thread or coroutine.

**4. Timestamp-aligned synthetic weather.**  
Weather is anchored to the real wall-clock hour so simulations "feel"
realistic regardless of when they are started. A 3 PM run starts with
warm, sunny, high-solar weather; a midnight run starts with cool, humid,
zero-solar weather.

**5. Event-triggered MPC.**  
Safety conditions (temperature out of range, disease risk spike) trigger
an early MPC re-solve instead of waiting for the next cadence boundary.
This is a simple heuristic inspired by model predictive safety filters
in the control literature.<sup>[[9]](#ref-9)</sup>

**6. Four-stream artifact output.**  
Separating state, MPC actions, diagnostics, and summary into four files
means each file can be read and processed independently. A tool that only
cares about energy usage only needs to open `dt_diagnostics_run.json`.

### 15.2 Assumptions

1. **Constant growth stage** *(now optional)* — by default the simulation
   runs an entire session at a fixed growth stage.  Set
   `auto_advance_stage=True` (on `DTLoop` or `RealtimeLoopConfig`) to
   enable automatic stage transitions.  The loop tracks cumulative hours
   and advances to the next canonical stage when `STAGE_DURATION_HOURS`
   elapses.  `days_elapsed` sets the initial offset.

2. **Synthetic weather is representative** — the sinusoidal diurnal model
   is reasonable for a clear day in Tamil Nadu. Multi-day cloud cover,
   monsoon conditions, or step-change weather events are not modelled.

3. **Actuator delays** *(now configurable)* — by default actuators
   respond instantaneously (backward compatible).  Set the
   `lag_*_minutes` parameters on `GreenhouseModelParams` to enable
   first-order response lag per actuator channel:
   `lag_fan_minutes`, `lag_vent_minutes`, `lag_heater_minutes`,
   `lag_fogger_minutes`, `lag_co2_valve_minutes`, `lag_led_minutes`,
   `lag_irrigation_minutes`.  Each positive value introduces
   `alpha = 1 - exp(-dt / tau)` smoothing.  Call
   `GreenhouseTransitionModel.reset_actuator_state()` between
   independent simulations.

4. **Linear ARX model** — the greenhouse physics are approximated by a
   linear discrete-time model. Real greenhouses are nonlinear (e.g.
   evapotranspiration depends on plant leaf area index which grows over
   time). The ARX model is calibrated to a mid-season "typical" plant.

5. **Single zone** — the model treats the entire greenhouse as one
   spatially uniform zone. Real greenhouses may have temperature
   gradients of 2–5 °C from roof to floor.

---

## 16. Known Limitations

> **Scope note**: The DT layer is the *physics simulation + loop
> orchestration* wrapper around the MPC pipeline.  Features like the LSTM
> disease model, AI weather forecasts, growth-stage weights, and image
> classifiers live in the **MPC layer** (`StateFusion.fuse()`).
> The production loop (`run_realtime_loop.py`) already invokes all of
> those AI models via `StateFusion`.  Limitations listed here are
> specific to the DT engine or the loop orchestration itself.

1. **Single-threaded loop** — `DigitalTwinEngine.step()` and
   `MPCSolver.solve()` run synchronously in the same thread.  The
   MPC solve blocks for ~20–50 ms per cadence step.  For strict
   real-time scheduling, consider an `asyncio` coroutine or a
   background-thread solver.

2. **DT engine recomputes disease risk analytically** — after the ARX
   physics step, `DigitalTwinEngine` recalculates disease risk using
   the sigmoid formula (`compute_disease_risk_score`).  This is a
   *diagnostic-only* cross-check — the authoritative risk score comes
   from the MPC pipeline (`DiseaseRiskPenalty` LSTM via `StateFusion`).
   The DT engine does not override the MPC's risk estimate.

---

## 17. Recommended Next Extensions

> **What is already done in the MPC layer** (not repeated here):
> AI weather forecasts, LSTM disease severity, growth-stage weights
> and transition prediction, image classification (disease +
> growth stage), cost-weight adaptation, and `DigitalTwinOutput`
> formatting are all implemented and consumed by the production
> loop via `StateFusion.fuse()`.  See
> [`MPC_COMPLETE_GUIDE.md`](MPC_COMPLETE_GUIDE.md) for details.

The extensions below are **DT-layer / infrastructure** improvements:

1. **Grafana / live dashboard export** — stream per-step dicts
   to a Grafana dashboard via InfluxDB or a WebSocket endpoint.

2. **Multi-zone physics model** — extend `GreenhouseTransitionModel`
   to handle $N$ spatially distinct zones with heat and moisture
   exchange between them.

---

## 18. Production Closed-Loop — Implemented

The production closed-loop is fully implemented across two layers:

- **`src/agritwin_gh/mpc/realtime_core.py`** — importable core engine (`RealtimeLoop`, `RealtimeLoopConfig`, DB helpers, diurnal defaults). No CLI dependencies — safe to import from any context.
- **`scripts/run_realtime_loop.py`** — thin CLI / test harness. Wraps `RealtimeLoop` with interactive prompts, console display, artifact saving, and signal handling.

This section documents the implemented architecture.

### 18.0 Database Schema Prerequisites

Before running the production loop, ensure the schema has been applied:

| Schema file | Tables created | Used by |
|---|---|---|
| `database/schema/timeseries_data.sql` | `greenhouse_data`, `weather_data`, `disease_progression`, `growth_progression_hourly`, `realtime_greenhouse_stream`, … | `RealtimeMPCInputPreparation` → `StateFusion` → `write_step_to_stream()` |
| `database/schema/image_metadata.sql` | `image_metadata`, `image_annotations` | `MinIOImageObserver` → `ImageStreamer` |

```powershell
psql -U <user> -d agritwin_db -f database/schema/timeseries_data.sql
psql -U <user> -d agritwin_db -f database/schema/image_metadata.sql
```

Both files use `CREATE TABLE IF NOT EXISTS` — safe to re-run. See
`docs/POSTGRESQL_QUICKSTART.md` for full setup instructions.

---

### 18.1 Implemented System Architecture

**Data flow (per step):**

```
realtime_greenhouse_stream  (PostgreSQL)
  │
  │  RealtimeMPCInputPreparation.get_latest_greenhouse_row()
  ▼
StateFusion.fuse()           ← 9-step AI pipeline
  │                             (weather forecast, disease risk, growth weights,
  │                              image classification, state fusion)
  ▼
MPCSolver.solve(fused)       ← SLSQP optimisation over 1-hour horizon
  │
  ▼
DigitalTwinEngine.step()     ← 8-phase ARX physics + diagnostics
  │                             (effect attribution, energy/water, disease flags)
  ▼
write_step_to_stream()       → realtime_greenhouse_stream  (PostgreSQL)
  │
  ▼
save_step_artifact()         → logs/realtime/<run_id>/steps.ndjson
  │
  ▼  next iteration …
```

**Key components:**

| Class / function | File | Role |
|-----------------|------|------|
| `RealtimeMPCInputPreparation` | `mpc/realtime_core.py` | Reads from `realtime_greenhouse_stream` instead of `greenhouse_data`; all other tables unchanged |
| `DatabaseInputProvider` | `mpc/dt_input_provider.py` | Full `DTInputProvider` for the synthetic `DTLoop` backed by PostgreSQL |
| `RealtimeGreenhouseStream` | `models/timeseries.py` | SQLAlchemy ORM for `realtime_greenhouse_stream` |
| `seed_initial_state()` | `mpc/realtime_core.py` | Bootstraps stream row 0 from real `greenhouse_data` (or synthetic defaults) |
| `write_step_to_stream()` | `mpc/realtime_core.py` | Persists each `DigitalTwinStepPayload` as a new DB row |
| `RealtimeLoop` | `mpc/realtime_core.py` | Orchestrator — `setup()` → `run()` / `step(i)` — importable by any integration |
| `DigitalTwinEngine` | `mpc/dt_engine.py` | Called on every step (MPC + non-MPC) — full physics + diagnostics |

### 18.2 `MinIOImageObserver` — Implemented

`MinIOImageObserver` is now fully implemented in `dt_image_observer.py`.
See [Section 6.8](#68-dt_image_observerpy) for the full API.

```python
from agritwin_gh.mpc import DTLoop, MinIOImageObserver

# Explicit observer:
loop = DTLoop(
    growth_stage="flowering",
    image_observer=MinIOImageObserver(session, cache_ttl_sec=300.0),
)

# Or auto-select via session:
loop = DTLoop(growth_stage="flowering", session=db_session)
```

Falls back to `SyntheticImageObserver` when a MinIO image category is
empty — logged at `INFO` level.

### 18.3 `RunRegistry` — Implemented

`RunRegistry` (in `realtime_core.py`) indexes all completed runs in
`logs/realtime/registry.json`.  Automatically wired into the CLI
harness (`save_run_manifest()` in `run_realtime_loop.py`).

```python
from agritwin_gh.mpc import RunRegistry

reg = RunRegistry()
all_runs  = reg.load()                         # full registry
flowering = reg.list_runs("flowering", limit=5) # filtered, newest first
entry     = reg.get_run("rt_20260402_185434")   # single lookup
```

Each entry stores: `run_id`, `growth_stage`, `days_elapsed_at_start`,
`planned_steps`, `steps_run`, `start_ts`, `end_ts`, `simulated_hours`,
`total_energy_kwh`, `total_water_litres`, `total_mpc_cost`,
`mpc_every_steps`, `auto_advance_stage`, `images_enabled`, `device`,
`artifact_dir`.

### 18.4 Auto-Advancing Growth Stage — Implemented

Both `DTLoop` and `RealtimeLoop` support automatic growth-stage transitions.

```python
# Offline DTLoop:
loop = DTLoop(
    growth_stage="seedling",
    auto_advance_stage=True,
    days_elapsed=13.5,     # 13.5 days into seedling (336 h total)
    n_steps=576,           # 48 hours — will cross into early vegetative
)

# Production RealtimeLoop:
cfg = RealtimeLoopConfig(
    growth_stage="flowering",
    days_elapsed=5.0,
    total_steps=288,
    auto_advance_stage=True,
)
```

When `hours_in_stage >= STAGE_DURATION_HOURS[stage]`, the loop advances
to the next entry in `GROWTH_STAGES` and resets the hour counter.
Transitions are logged at `INFO` level.

### 18.5 Actuator Response Lag — Implemented

`GreenhouseModelParams` now exposes 7 per-channel first-order lag time
constants (all default 0.0 — instantaneous, backward compatible):

| Parameter | Actuator | Typical real-world lag |
|---|---|---|
| `lag_fan_minutes` | Fan | 0.5–2 min (motor spool) |
| `lag_vent_minutes` | Vent | 1–3 min (motor travel) |
| `lag_heater_minutes` | Heater | 2–5 min (element warm-up) |
| `lag_fogger_minutes` | Fogger | 0.5–1 min (pressure build) |
| `lag_co2_valve_minutes` | CO₂ valve | 0.5–1 min (valve travel) |
| `lag_led_minutes` | LED | ~0 (electronic) |
| `lag_irrigation_minutes` | Irrigation | 0.5–2 min (pump/valve) |

When `tau > 0`, the effective actuator output follows:

$$
u_{eff}(k) = u_{eff}(k{-}1) + \alpha \cdot \bigl(u_{cmd}(k) - u_{eff}(k{-}1)\bigr),
\quad \alpha = 1 - e^{-\Delta t / \tau}
$$

Call `GreenhouseTransitionModel.reset_actuator_state()` between
independent simulations to clear the lag memory.  `simulate()` saves and
restores the effective-actuator state automatically so each MPC rollout
is independent.

---

## 19. References

<a name="ref-1">[1]</a> Heuvelink, E. (2005). *Tomatoes*. CABI Publishing.
Temperature tolerance and flowering biology of *Solanum lycopersicum*.

<a name="ref-2">[2]</a> Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994).
*Design Patterns: Elements of Reusable Object-Oriented Software*.
Addison-Wesley. Facade pattern (pp. 185–193).

<a name="ref-3">[3]</a> Allen, R.G., Pereira, L.S., Raes, D., & Smith, M. (1998).
*Crop evapotranspiration — Guidelines for computing crop water requirements.*
FAO Irrigation and Drainage Paper 56. FAO, Rome.

<a name="ref-4">[4]</a> Farquhar, G.D., von Caemmerer, S., & Berry, J.A. (1980).
A biochemical model of photosynthetic CO₂ assimilation in leaves of C₃ species.
*Planta*, 149(1), 78–90.

<a name="ref-5">[5]</a> Fry, W.E. (2008). *Phytophthora infestans*: the plant (and R gene)
destroyer. *Molecular Plant Pathology*, 9(3), 385–402.
Leaf wetness as primary infection requirement.

<a name="ref-6">[6]</a> Jones, J.B., Jones, J.P., Stall, R.E., & Zitter, T.A. (1991).
*Compendium of Tomato Diseases*. American Phytopathological Society Press.
Temperature and humidity ranges for major tomato pathogens.

<a name="ref-7">[7]</a> Campbell, G.S., & Norman, J.M. (2000).
*An Introduction to Environmental Biophysics* (2nd ed.). Springer.
Diurnal temperature and humidity patterns in semi-arid environments (Ch. 4).

<a name="ref-8">[8]</a> Mortensen, L.M. (1987). Review: CO₂ enrichment in greenhouses.
Crop responses. *Scientia Horticulturae*, 33(1–2), 1–25.
Actuator power consumption models for greenhouse climate control.

<a name="ref-9">[9]</a> Wabersich, K.P., & Zeilinger, M.N. (2021).
A predictive safety filter for learning-based control of constrained
nonlinear dynamical systems. *Automatica*, 129, 109597.
Event-triggered safety filter concept underpinning `should_force_mpc_update()`.
