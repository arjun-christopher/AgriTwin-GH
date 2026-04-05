# Greenhouse Unity Scene

> **Complete documentation for all greenhouse controllers, central state applier, and backend integration patterns.**

---

## Table of Contents

1. [Overview](#overview)
2. [Complete Project Reference](#complete-project-reference)
   - [All C# Scripts — Quick Reference](#all-c-scripts--quick-reference)
   - [Scene GameObject Hierarchy](#scene-gameobject-hierarchy)
   - [Growth Stages — Complete Reference](#growth-stages--complete-reference)
   - [Actuators — Complete List](#actuators--complete-list)
   - [Environment Systems](#environment-systems)
   - [Complete Data Flow](#complete-data-flow)
   - [File Structure in Project](#file-structure-in-project)
3. [GreenhouseStateApplier — Central Controller](#greenhousestateapplier--central-controller)
4. [Individual Actuator Controllers](#individual-actuator-controllers)
5. [Environment Controllers](#environment-controllers)
6. [JSON Schema & Format](#json-schema--format)
7. [Setup Instructions](#setup-instructions)
8. [Live Testing in Play Mode](#live-testing-in-play-mode)
9. [Future FastAPI / Python Backend Integration](#future-fastapi--python-backend-integration)
10. [Architecture & Design Patterns](#architecture--design-patterns)

---

## Overview

This greenhouse Unity scene is composed of:
- **10 actuator controllers** (lights, effects, audio) for manual state control
- **3 environment controllers** (crop stages, time of day, crop health)
- **1 central state applier** (`GreenhouseStateApplier.cs`) that orchestrates everything from a JSON file

All scripts are designed for **Python backend integration**. The central applier reads a JSON file at runtime, detects changes, and applies the full greenhouse state automatically. This allows testing backend-style control patterns locally inside Unity before connecting to FastAPI.

---

---

# Complete Project Reference

## All C# Scripts — Quick Reference

| Script | Purpose | Type | Key Method(s) |
|--------|---------|------|---|
| **GreenhouseStateApplier.cs** | Central orchestrator; reads JSON, detects changes, applies state | Manager | `SetState()` dispatches to all controllers |
| **FluorescentLightController.cs** | Controls fluorescent light (status + spots) | Actuator | `SetState(bool)` |
| **HeaterController.cs** | Controls heater (indicator + glow) | Actuator | `SetState(bool)` |
| **EnergyCanisterController.cs** | Controls energy canister (status light) | Actuator | `SetState(bool)` |
| **HumidifierController.cs** | Controls humidifier (light + fog + audio) | Actuator | `SetState(bool)` |
| **WindowFanController.cs** | Controls window fan (light + particles + audio) | Actuator | `SetState(bool)` |
| **VentController.cs** | Controls vent system (light + particles + audio) | Actuator | `SetState(bool)` |
| **WaterTankFloorController.cs** | Controls water tank (light + audio) | Actuator | `SetState(bool)` |
| **CropStageController.cs** | Controls one crop's visual stage | Environment | `SetStageByName(string)` |
| **TimeOfDayController.cs** | Controls lighting, skybox, fog, night lights | Environment | `SetTimeOfDayByName(string)` |
| **CropHealthIndicator.cs** | Controls RGB health lights + blinking | Environment | `SetGreen()`, `SetYellow()`, `SetRed()`, `SetBlinkGreen(bool)` |
| **FreeCameraController.cs** | Provides free camera controls | Utility | (no public state methods) |

---

## Scene GameObject Hierarchy

Your Unity scene should have this structure:

```
Scene/
├── GreenhouseManager (Empty GameObject)
│   └── [GreenhouseStateApplier component attached]
│
├── Actuators/ (folder, organizing lights/effects)
│   ├── FluorescentLight
│   │   └── [FluorescentLightController component]
│   ├── Heater
│   │   └── [HeaterController component]
│   ├── EnergyCanister
│   │   └── [EnergyCanisterController component]
│   ├── Humidifier
│   │   └── [HumidifierController component]
│   │       └── StatusIndicator (Light child)
│   │       └── FogEffect (ParticleSystem child)
│   │       └── AudioSource (parent object)
│   ├── WindowFan
│   │   └── [WindowFanController component]
│   │       └── StatusIndicator (Light child)
│   │       └── AirFlow (ParticleSystem child)
│   │       └── AudioSource (parent object)
│   ├── Vent
│   │   └── [VentController component]
│   │       └── StatusIndicator (Light child)
│   │       └── AirFlow (ParticleSystem child)
│   │       └── AudioSource (parent object)
│   └── WaterTank
│       └── [WaterTankFloorController component]
│           └── StatusIndicator (Light child)
│           └── AudioSource (same object)
│
├── Crops/ (folder, organizing 15 crop plants)
│   ├── Crop_01
│   │   └── [CropStageController component]
│   │       ├── Seedling (Model)
│   │       ├── Vegetative (Model)
│   │       ├── FloweringInitiation (Model)
│   │       ├── Flowering (Model)
│   │       ├── Unripe (Model)
│   │       └── Ripe (Model)
│   │   └── CropHealth (ChildObject)
│   │       └── [CropHealthIndicator component]
│   │           ├── GreenLight (Light)
│   │           ├── YellowLight (Light)
│   │           └── RedLight (Light)
│   ├── Crop_02 ... Crop_15
│   │   └── [same structure as Crop_01]
│
├── Environment/ (folder)
│   └── [TimeOfDayController component on main light]
│       └── DirectionalLight (sun/moon)
│       └── NightLights (array of GameObjects for night)
│
├── Camera (MainCamera)
│   └── [FreeCameraController component]
│
└── [Skybox, Fog, Lighting settings]
```

---

## Growth Stages — Complete Reference

When a crop grows, it progresses through 6 distinct stages. Each stage has a corresponding GameObject model that becomes visible/invisible.

### Stage Progression (0 → 5)

| Index | Name | Visual | Duration (Typical) | Description |
|-------|------|--------|---|---|
| 0 | **Seedling** | Tiny sprout | Days 0–7 | Initial germination; very small plant |
| 1 | **Vegetative** | Young plant | Days 7–21 | Leaf growth; expanding stem |
| 2 | **FloweringInitiation** | Budding | Days 21–28 | Buds beginning to form |
| 3 | **Flowering** | Blooming | Days 28–42 | Full flowers open; peak beauty |
| 4 | **Unripe** | Young fruit | Days 42–56 | Fruit formed but not mature |
| 5 | **Ripe** | Ready to harvest | Days 56+ | Fully mature; ready for pickup |

### JSON Stage Values (Case-Insensitive)

```json
"cropStage": { "stage": "Seedling" }
"cropStage": { "stage": "Vegetative" }
"cropStage": { "stage": "FloweringInitiation" }
"cropStage": { "stage": "Flowering" }
"cropStage": { "stage": "Unripe" }
"cropStage": { "stage": "Ripe" }
```

**Variants accepted** (all auto-converted):
- `"flowering_initiation"` → `FloweringInitiation`
- `"RIPE"` → `Ripe`
- `"vegetative"` → `Vegetative`

### Key Behaviors by Stage

- **Seedling/Vegetative:** Health indicator can show any color (Green/Yellow/Red)
- **All stages up to Unripe:** No blinking, even if manually set
- **Ripe (all crops):** Health **automatically forced to Green + blinking**, overriding JSON
- **Any stage change:** Emits `SetBlinkGreen(false)` unless all crops are Ripe

---

## Actuators — Complete List

All 8 actuators follow the unified **`SetState(bool)`** pattern: `true` = on, `false` = off.

### Group 1: Lighting Actuators

#### 1. FluorescentLight
- **Controls:** Status indicator (green), main spot light, point light
- **Behavior:** All 3 lights turn on/off together
- **Inspector Fields:** `statusIndicatorLight`, `fluorescentSpotLight`, `fluorescentPointLight`
- **JSON:** `"fluorescentLight": { "isOn": true | false }`

#### 2. Heater
- **Controls:** Status indicator (green), heater glow effect
- **Behavior:** Both lights turn on/off together
- **Inspector Fields:** `statusIndicatorLight`, `heaterLight`
- **JSON:** `"heater": { "isOn": true | false }`

#### 3. EnergyCanister
- **Controls:** Energy status light
- **Behavior:** Light on/off
- **Inspector Fields:** `statusLight`
- **JSON:** `"energyCanister": { "isOn": true | false }`

### Group 2: Effects with Audio

#### 4. Humidifier
- **Controls:** Status indicator, fog particles, audio playback
- **Behavior:** When on → light on, particles play, audio continuous. When off → all stop.
- **Inspector Fields:** `statusIndicatorLight`, `fogParticle`
- **Auto-Found:** `fogAudio` (from parent)
- **JSON:** `"humidifier": { "isOn": true | false }`

#### 5. WindowFan
- **Controls:** Status indicator, air flow particles, audio playback
- **Behavior:** When on → light on, particles play, audio continuous loop. When off → all stop.
- **Inspector Fields:** `statusIndicatorLight`, `airFlowParticle`
- **Auto-Found:** `fanAudio` (from parent)
- **JSON:** `"windowFan": { "isOn": true | false }`

#### 6. Vent
- **Controls:** Status indicator, air flow particles, audio trigger
- **Behavior:** When on → light on, particles play, audio plays **once per state change**. When off → particles stop.
- **Inspector Fields:** `statusIndicatorLight`, `airFlowParticle`
- **Auto-Found:** `ventAudio` (from parent)
- **JSON:** `"vent": { "isOn": true | false }`

#### 7. WaterTankFloor
- **Controls:** Status indicator, audio playback
- **Behavior:** When on → light on, audio continuous. When off → both stop.
- **Auto-Found:** `statusLight` (from StatusIndicator child), `audioSource` (self)
- **JSON:** `"waterTankFloor": { "isOn": true | false }`

### Group 3: Utility Lights

*No additional actuators; lights are controlled by environment systems*

---

## Environment Systems

### Time of Day

Affects **global lighting, skybox, fog, and night lights**.

| Time | Lighting | Skybox | Fog | Sun Angle | Night Lights |
|------|----------|--------|-----|-----------|---|
| **Morning** | Warm (0.75, 0.72, 0.65) | Clear sky | Warm fog (on) | 25°, 30° | Off |
| **Afternoon** | Bright (1.0, 1.0, 1.0) | Bright blue | Off | 60°, 0° | Off |
| **Evening** | Orange (1.0, 0.55, 0.3) | Orange sky | Orange fog (on) | 15°, 220° | Off |
| **Night** | Dark blue (0.4, 0.45, 0.6) | Night sky | Dim fog (on) | -10°, 0° | On |

**JSON Values (Case-Insensitive):**
```json
"timeOfDay": { "time": "Morning" | "Afternoon" | "Evening" | "Night" }
```

### Crop Health Indicator

Controls RGB status lights showing crop health and maturity.

| State | Color | Light | Blink? | Meaning |
|-------|-------|-------|--------|---------|
| **Green** | RGB Green | Bright | Auto* | Healthy |
| **Yellow** | RGB Yellow | Medium | Never | Stressed |
| **Red** | RGB Red | Bright | Never | Critical |

*Blinking occurs **only when all crops are Ripe**, regardless of JSON `blinkGreen` value.

**JSON Values (Case-Insensitive):**
```json
"cropHealth": {
    "state": "Green" | "Yellow" | "Red",
    "blinkGreen": false  // IGNORED — computed from crop stages
}
```

**Blink Speed:** Configurable in `CropHealthIndicator` Inspector (`blinkSpeed = 2.0` = 2 blinks/sec)

---

## Complete Data Flow

```
JSON File (greenhouse_state.json)
    ↓
GreenhouseStateApplier.Update() [every 1s]
    ↓
Change Detected? (compare file content hash)
    ↓ YES
ParseJson() → GreenhouseState object
    ↓
ApplyState() → dispatches to 8 controllers + 3 environment systems
    ↓
Each Controller.SetState() or SetStageByName() or SetTimeOfDayByName()
    ↓
Scene updates: Lights on/off, Particles play/stop, Models swap, Fog changes
    ↓
GreenhouseStateApplier.LateUpdate() [every frame]
    ↓
Re-assert health color (so CropStageController.SetGreen() doesn't override JSON)
    ↓
Scene persists health color every frame
```

---

## File Structure in Project

```
Assets/
├── Scripts/
│   ├── GreenhouseStateApplier.cs          [Central orchestrator]
│   ├── CropStageController.cs              [Crop stage visuals]
│   ├── CropHealthIndicator.cs              [RGB health lights]
│   ├── TimeOfDayController.cs              [Global lighting/sky]
│   ├── FluorescentLightController.cs       [Light actuator]
│   ├── HeaterController.cs                 [Light actuator]
│   ├── EnergyCanisterController.cs         [Light actuator]
│   ├── HumidifierController.cs             [Effects + audio]
│   ├── WindowFanController.cs              [Effects + audio]
│   ├── VentController.cs                   [Effects + audio]
│   ├── WaterTankFloorController.cs         [Effects + audio]
│   ├── FreeCameraController.cs             [Utility]
│   └── GREENHOUSE_INTEGRATION_GUIDE.md     [This document]
│
└── StreamingAssets/
    └── greenhouse_state.json               [Runtime state file]
```

---



## What It Does

`GreenhouseStateApplier.cs` is the **single central point** that controls:
- All 8 actuators (lights, heater, canister, humidifier, fan, vent, water tank)
- All 15 crops (stage progression)
- Time of day (lighting, skybox, fog)
- Crop health indicator (color + auto-blinking based on stage)

It reads a JSON file from disk at a **configurable polling interval**, detects changes, and applies updates automatically during Play Mode.

### Key Features

✅ **Reads JSON at runtime** — no code changes needed to update scene state  
✅ **Detects changes automatically** — only reapplies when file content changes  
✅ **Supports any crop count** — JSON array can have 15, 50, or 100 crops  
✅ **Safe missing references** — logs clear warnings but never crashes  
✅ **Health color auto-logic** — forces Green+blink when all crops are Ripe  
✅ **LateUpdate assertion** — health color persists even if other scripts modify it  
✅ **Production-ready** — clean error handling, verbose logging, easy to extend  

---

## Inspector Setup

1. **Create empty GameObject** in your scene (e.g., `"GreenhouseManager"`)
2. **Attach `GreenhouseStateApplier`** component
3. **Assign every field** in the Inspector by dragging scene GameObjects:

### Actuators to Assign

- `Fluorescent Light` — GameObject with `FluorescentLightController`
- `Heater` — GameObject with `HeaterController`
- `Energy Canister` — GameObject with `EnergyCanisterController`
- `Humidifier` — GameObject with `HumidifierController`
- `Window Fan` — GameObject with `WindowFanController`
- `Vent` — GameObject with `VentController`
- `Water Tank Floor` — GameObject with `WaterTankFloorController`

### Environment & Crop to Assign

- `Crop Stages` (**Array**) — Drag all 15 crop GameObjects here (one per slot)
- `Time Of Day` — GameObject with `TimeOfDayController`
- `Crop Health` — GameObject with `CropHealthIndicator`

### Settings

- **Json File Path:** (default) `Assets/StreamingAssets/greenhouse_state.json`
- **Poll Interval Seconds:** `1.0` (check file every 1 second)
- **Verbose Logging:** `true` (see detailed console output)
- **Force Refresh:** Checkbox in Play Mode to instantly re-apply without waiting

---

## JSON File Location

**Path:** `Assets/StreamingAssets/greenhouse_state.json`

If the `StreamingAssets` folder doesn't exist, create it:
```
Assets/
├── StreamingAssets/
│   └── greenhouse_state.json
```

---

---

# Individual Actuator Controllers

## Architecture

All actuator scripts follow the **unified control pattern**:

- **Protected State Variables:** `[SerializeField] private bool isOn` prevents accidental manual manipulation
- **Single Entry Point:** `SetState(bool state)` is the only method the backend calls
- **Debug Mode:** Optional `debugManualControl` toggle for editor testing
- **Simple & Direct:** Changes apply immediately, no complex state tracking

---

## 1. FluorescentLightController.cs

**Purpose:** Controls a fluorescent light fixture with status indicator and point lights

**Controls:**
- Status indicator light (green)
- Fluorescent spot light (main light)
- Fluorescent point light (additional brightness)

**Public Methods:**
- `SetState(bool state)` — **main entry point** (pass `true` to turn on)
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `statusIndicatorLight` — Light component for status
- `fluorescentSpotLight` — Main spotlight
- `fluorescentPointLight` — Additional point light for ambiance
- `debugManualControl` — Edit in Play Mode for testing

---

## 2. HeaterController.cs

**Purpose:** Controls a heater with indicator light and glow effect

**Controls:**
- Status indicator light (green)
- Heater glow/point light

**Public Methods:**
- `SetState(bool state)` — **main entry point**
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `statusIndicatorLight` — Status indicator
- `heaterLight` — Heater glow effect
- `debugManualControl` — Edit in Play Mode for testing

---

## 3. EnergyCanisterController.cs

**Purpose:** Controls an energy canister with status light

**Controls:**
- Status point light

**Public Methods:**
- `SetState(bool state)` — **main entry point**
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `statusLight` — Energy status indicator
- `debugManualControl` — Edit in Play Mode for testing

---

## 4. HumidifierController.cs

**Purpose:** Controls a humidifier with fog particle effect and audio

**Controls:**
- Status indicator light
- Fog particle system
- Audio source (auto-found from parent)

**Public Methods:**
- `SetState(bool state)` — **main entry point**
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `statusIndicatorLight` — Status indicator
- `fogParticle` — Particle system for fog effect
- `debugManualControl` — Edit in Play Mode for testing

**Auto-Found:**
- `fogAudio` — Looks for AudioSource on parent object

---

## 5. WindowFanController.cs

**Purpose:** Controls a window fan with air flow effect and audio

**Controls:**
- Status indicator light
- Air flow particle system
- Audio source (continuous playback)

**Public Methods:**
- `SetState(bool state)` — **main entry point**
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `statusIndicatorLight` — Status indicator
- `airFlowParticle` — Particle system for air flow
- `debugManualControl` — Edit in Play Mode for testing

**Auto-Found:**
- `fanAudio` — Looks for AudioSource on parent object

---

## 6. VentController.cs

**Purpose:** Controls a ventilation system with air flow and audio trigger

**Controls:**
- Status indicator light
- Air flow particle system
- Audio source (plays once per state change)

**Public Methods:**
- `SetState(bool state)` — **main entry point**
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `statusIndicatorLight` — Status indicator
- `airFlowParticle` — Particle system for air flow
- `debugManualControl` — Edit in Play Mode for testing

**Auto-Found:**
- `ventAudio` — Looks for AudioSource on parent object

**Note:** Audio plays only on state changes, not continuously (unlike fan)

---

## 7. WaterTankFloorController.cs

**Purpose:** Controls a water tank with status light and audio

**Controls:**
- Status indicator light (child of `StatusIndicator`)
- Audio source (on same object, continuous playback)

**Public Methods:**
- `SetState(bool state)` — **main entry point**
- `TurnOn()` / `TurnOff()` — convenience helpers

**Inspector Fields:**
- `debugManualControl` — Edit in Play Mode for testing

**Auto-Found:**
- `statusLight` — Looks in children for `StatusIndicator` child, then its Light
- `audioSource` — Looks on same GameObject for AudioSource

---

---

# Environment Controllers

## 8. CropStageController.cs

**Purpose:** Controls a single crop's visual stage and updates its health indicator

**Growth Stages (0-5):**
- `Seedling` (0) — Initial stage
- `Vegetative` (1) — Growth phase
- `FloweringInitiation` (2) — Transition to flowering
- `Flowering` (3) — Bloom phase
- `Unripe` (4) — Fruit developing
- `Ripe` (5) — Ready to harvest

**Public Methods:**
- `SetStage(GrowthStage stage)` — Pass enum directly
- `SetStageByIndex(int index)` — Pass 0-5, clamped automatically
- `SetStageByName(string stageName)` — Pass string like `"Ripe"` or `"flowering_initiation"`
- `NextStage()` / `PreviousStage()` — Increment/decrement

**Inspector Fields:**
- `seedling` / `vegetative` / `floweringInitiation` / `flowering` / `unripe` / `ripe` — GameObjects for each stage (only one visible at a time)
- `healthIndicator` — Reference to the CropHealthIndicator (auto-found from parent/children)
- `debugManualControl` — Edit in Play Mode for testing

**Auto-Behavior:**
- When set to `Ripe`, automatically calls `healthIndicator.SetGreen()` and `SetBlinkGreen(true)`
- When set to any other stage, calls `SetBlinkGreen(false)`

---

## 9. TimeOfDayController.cs

**Purpose:** Controls scene lighting, skybox, fog, and night lights

**Time Periods:**
- `Morning` — Warm light, clear sky, some fog
- `Afternoon` — Bright sun, no fog, high intensity
- `Evening` — Orange light, orange sky, fog
- `Night` — Dark blue light, night sky, dim fog, night lights on

**Public Methods:**
- `SetTimeOfDay(TimeOfDay timeOfDay)` — Pass enum directly
- `SetTimeOfDayByName(string timeName)` — Pass `"Morning"`, `"Afternoon"`, `"Evening"`, or `"Night"`

**Inspector Fields:**
- `directionalLight` — Main sun/moon light
- `morningSkybox` / `afternoonSkybox` / `eveningSkybox` / `nightSkybox` — Material for each time
- `nightLights` — Array of GameObjects to enable only at night
- `enableKeyboardDebugControl` — Press 1/2/3/4 to switch times in Play Mode
- `debugManualControl` — Edit in Play Mode for testing

---

## 10. CropHealthIndicator.cs

**Purpose:** Controls green/yellow/red status lights with optional blinking

**Health States:**
- `Green` — Healthy (can blink if all crops are Ripe)
- `Yellow` — Moderate stress
- `Red` — Critical stress

**Public Methods:**
- `SetState(HealthState state)` — Pass enum directly
- `SetGreen()` / `SetYellow()` / `SetRed()` — Direct helpers
- `SetBlinkGreen(bool shouldBlink)` — Enable/disable green blinking

**Inspector Fields:**
- `greenLight` / `yellowLight` / `redLight` — Light components for each state
- `blinkGreen` — Manual blink toggle (read-only in normal use; `GreenhouseStateApplier` controls this)
- `blinkSpeed` — Speed of green blink (2.0 = 2 blinks per second)
- `debugManualControl` — Edit in Play Mode for testing

**Auto-Behavior (via GreenhouseStateApplier):**
- Green light blinks when **ALL 15 crops are in Ripe stage**
- If even one crop is not Ripe, blinking stops (even if manually set)
- If JSON says `state: "Yellow"` or `"Red"` but all crops are Ripe, automatically forced to `Green + blink`

---

## 11. FreeCameraController.cs

**Purpose:** Provides free camera movement and look controls

**Controls:**
- Right-click + move mouse to look around (sensitivity: 0.15)
- WASD to move forward/back/strafe
- Shift to double speed (2x movement)
- Mouse scroll to zoom in/out
- Confined to `boundsBox` if assigned

**Inspector Fields:**
- `boundsBox` — Assign a Collider (usually a cube) to constrain camera movement
- (Requires `CharacterController` component on same GameObject)

---

---

# JSON Schema & Format

## Complete JSON Structure

```json
{
    "fluorescentLight":  { "isOn": true | false },
    "heater":            { "isOn": true | false },
    "energyCanister":    { "isOn": true | false },
    "humidifier":        { "isOn": true | false },
    "windowFan":         { "isOn": true | false },
    "vent":              { "isOn": true | false },
    "waterTankFloor":    { "isOn": true | false },
    
    "cropStage":  { "stage": "Seedling|Vegetative|FloweringInitiation|Flowering|Unripe|Ripe" },
    
    "timeOfDay":  { "time": "Morning|Afternoon|Evening|Night" },
    
    "cropHealth": {
        "state": "Green|Yellow|Red",
        "blinkGreen": false  // IGNORED — computed from crop stages
    }
}
```

### Important Notes

- **All string values are case-insensitive** (e.g., `"Ripe"`, `"RIPE"`, `"ripe"` all work)
- **Crop stage applies to all crops** — the single `cropStage` value is applied to every crop in the Inspector array
- **Any key can be omitted** — missing keys default to no action (safe partial JSON)
- **Crop stage underscore variants** are auto-converted (e.g., `"flowering_initiation"` → `"FloweringInitiation"`)
- **`blinkGreen` field** is kept in schema for compatibility but is **never read** — it is always computed from crop stage

---

## Sample JSON File

```json
{
    "fluorescentLight":  { "isOn": true  },
    "heater":            { "isOn": false },
    "energyCanister":    { "isOn": true  },
    "humidifier":        { "isOn": true  },
    "windowFan":         { "isOn": false },
    "vent":              { "isOn": true  },
    "waterTankFloor":    { "isOn": false },
    "cropStage":         { "stage": "Ripe" },
    "timeOfDay":         { "time":  "Morning" },
    "cropHealth":        { "state": "Green", "blinkGreen": false }
}
```

---

---

# Setup Instructions

## Step 1: Verify File Structure

Ensure your project has:
```
Assets/
├── Scripts/
│   ├── GreenhouseStateApplier.cs
│   ├── CropStageController.cs
│   ├── CropHealthIndicator.cs
│   ├── TimeOfDayController.cs
│   ├── FluorescentLightController.cs
│   ├── HeaterController.cs
│   ├── EnergyCanisterController.cs
│   ├── HumidifierController.cs
│   ├── WindowFanController.cs
│   ├── VentController.cs
│   ├── WaterTankFloorController.cs
│   ├── FreeCameraController.cs
│   └── GREENHOUSE_INTEGRATION_GUIDE.md (this file)
└── StreamingAssets/
    └── greenhouse_state.json
```

## Step 2: Create Manager GameObject

1. In the Hierarchy, right-click → **Create Empty**
2. Name it `"GreenhouseManager"`
3. Attach `GreenhouseStateApplier` component to it

## Step 3: Assign Inspector References

In the Inspector for `GreenhouseStateApplier`, drag:

### Actuators (8 items)
1. `Fluorescent Light` ← GameObject with `FluorescentLightController`
2. `Heater` ← GameObject with `HeaterController`
3. `Energy Canister` ← GameObject with `EnergyCanisterController`
4. `Humidifier` ← GameObject with `HumidifierController`
5. `Window Fan` ← GameObject with `WindowFanController`
6. `Vent` ← GameObject with `VentController`
7. `Water Tank Floor` ← GameObject with `WaterTankFloorController`

### Crops & Environment (3 items)
8. `Crop Stages` array — Set Size = 15 (or however many crops you have), then drag each crop GameObject into Element 0–14. **Note:** All crops receive the same `cropStage` value from the JSON.
9. `Time Of Day` ← GameObject with `TimeOfDayController`
10. `Crop Health` ← GameObject with `CropHealthIndicator`

## Step 4: Create JSON File

1. Create folder: `Assets/StreamingAssets/`
2. Create file: `greenhouse_state.json`
3. Paste the sample JSON content (see [Sample JSON File](#sample-json-file) above)

## Step 5: Play & Test

Press **Play** — the GreenhouseStateApplier reads the JSON file and applies the state automatically within 1 second (default poll interval).

---

---

# Live Testing in Play Mode

## How to Test Dynamic Updates

While Play Mode is running:

1. **Open the JSON file** in VS Code, Notepad++, or any text editor:
   ```
   Assets/StreamingAssets/greenhouse_state.json
   ```

2. **Make a change** (Examples):
   - Turn on heater: `"heater": { "isOn": false }` → `"heater": { "isOn": true }`
   - Change crop stage (applies to all 15 crops): `"cropStage": { "stage": "Seedling" }` → `"cropStage": { "stage": "Ripe" }`
   - Set health to Yellow: `"state": "Green"` → `"state": "Yellow"`
   - Change time: `"time": "Morning"` → `"time": "Night"`

3. **Save the file** (Ctrl+S)

4. **Within 1 second**, the scene updates automatically:
   - Lights turn on/off
   - Particles start/stop
   - Audio plays/stops
   - Crop models swap stages
   - Sky and fog change
   - Health indicator colors shift

5. **Watch the Console** for debug messages:
   ```
   [GreenhouseStateApplier] Crop[0] stage -> Ripe
   [GreenhouseStateApplier] CropHealth -> Green (blinking) [auto-overridden: all crops Ripe]
   [GreenhouseStateApplier] All states applied successfully.
   ```

## Force Refresh

In the Inspector during Play Mode, tick the **Force Refresh** checkbox to immediately re-apply without waiting for the next poll interval. The checkbox auto-resets after use.

---

---

# Future FastAPI / Python Backend Integration

## Swapping File to API

The current script reads from a local JSON file. To connect to a real Python FastAPI backend:

### Step 1: Locate the Read Method

Find this in `GreenhouseStateApplier.cs`:

```csharp
string ReadJsonFromDisk()
{
    try
    {
        if (!File.Exists(_resolvedFilePath))
        {
            Debug.LogWarning("[GreenhouseStateApplier] JSON file not found: " + _resolvedFilePath);
            return null;
        }

        return File.ReadAllText(_resolvedFilePath);
    }
    catch (Exception ex)
    {
        Debug.LogError("[GreenhouseStateApplier] Failed to read JSON file: " + ex.Message);
        return null;
    }
}
```

### Step 2: Replace with FastAPI Call

```csharp
private IEnumerator ReadJsonFromApi()
{
    string apiUrl = "http://localhost:8000/state";  // Your FastAPI endpoint

    UnityWebRequest request = UnityWebRequest.Get(apiUrl);
    yield return request.SendWebRequest();

    if (request.result == UnityWebRequest.Result.Success)
    {
        return request.downloadHandler.text;
    }
    else
    {
        Debug.LogError("[GreenhouseStateApplier] API request failed: " + request.error);
        return null;
    }
}
```

### Step 3: Update TryReadAndApply

Change from synchronous to coroutine:

```csharp
// Before (synchronous)
string json = ReadJsonFromDisk();

// After (coroutine)
StartCoroutine(ReadJsonFromApiCoroutine());

IEnumerator ReadJsonFromApiCoroutine()
{
    yield return StartCoroutine(ReadJsonFromApi());
    // Continue with parsing and applying...
}
```

**That's it.** Everything else in the pipeline (parsing, change detection, applying to controllers) stays exactly the same. The individual controller scripts need zero changes.

---

---

# Architecture & Design Patterns

## Why This Structure?

### Separation of Concerns

- **Individual Controllers** handle: lights, particles, audio, UI
- **GreenhouseStateApplier** handles: orchestration, JSON parsing, change detection
- **Backend** (future) handles: business logic, state computation

Each layer knows nothing about lower layers and can be tested independently.

### Unified Control Pattern

Every actuator exposes `SetState(bool)` — a single, predictable method signature that the central applier can call uniformly. This makes it trivial to add new actuators later.

### LateUpdate for Health Persistence

`CropStageController.Update()` might call `SetGreen()` (because a crop is Ripe). If `GreenhouseStateApplier.Update()` happens to run *after* that, it would override the health you just set in the JSON. 

**Solution:** `GreenhouseStateApplier` uses `LateUpdate()` to re-assert health color **after all Update() calls finish**. This guarantees the health color you set in JSON persists every frame, regardless of execution order.

### All-Ripe Override Logic

When all 15 crops are Ripe, the logic is:
1. **Always** set health to Green (even if JSON says Yellow/Red)
2. **Always** enable blinking (even if JSON says `"blinkGreen": false`)
3. **Automatically** detect when this condition starts/stops

This is business logic: "A fully mature crop garden should always show green + blinking." The JSON cannot override this — it's part of the scene's DNA.

### Fault Tolerance

- Missing JSON keys → safely skipped (no error)
- Null references → warning logged, skipped (no crash)
- Invalid JSON → error logged, previous state retained (no crash)
- File encoding issues → caught and logged (no silent failures)

---

## Future Extensibility

### Adding More Crops (or Fewer)

Since all crops share a **single `cropStage` value**, adding or removing crops is trivial:

1. Create new crop GameObject(s) in the scene with `CropStageController` component
2. In `GreenhouseStateApplier` Inspector, increase `Crop Stages` array Size
3. Drag the new crop(s) into the new array slots
4. **Done.** No JSON or code changes required. The single `cropStage` value applies to all crops automatically.

### Adding a New Actuator

1. Create the new controller script (e.g., `SoilSensorController.cs`)
2. Add to `GreenhouseStateApplier`:
   ```csharp
   public SoilSensorController soilSensor;
   
   void ApplySoilSensor(ActuatorState s)
   {
       if (s == null) return;
       if (!AssertRef(soilSensor, "SoilSensorController")) return;
       soilSensor.SetState(s.isOn);
       Log("SoilSensor -> " + s.isOn);
   }
   ```
3. Call in `ApplyState()`:
   ```csharp
   ApplySoilSensor(state.soilSensor);
   ```
4. Update `GreenhouseState` data class:
   ```csharp
   public ActuatorState soilSensor;
   ```
5. Update JSON schema and sample file
6. **Done.** No other scripts touched.

---

## Testing Individual Controllers

You can test any controller in isolation by manually calling its methods in the editor:

```csharp
// Test FluorescentLight
fluorescent.SetState(true);
fluorescent.TurnOff();

// Test CropStage
crop.SetStageByName("Ripe");
crop.NextStage();

// Test Health
health.SetRed();
health.SetBlinkGreen(true);
```

Or enable `debugManualControl` in the Inspector to toggle values in Play Mode without touching code.

---

## Shader Updates

### Lit Shader - Leaf Green Default Color

- **File:** `Library/PackageCache/com.unity.render-pipelines.universal@*/Shaders/Lit.shader`
- **Change:** Default `_BaseColor` updated to leaf green **(0.298, 0.686, 0.314, 1)**
- **Impact:** All materials using the Universal Render Pipeline Lit shader default to leaf green unless overridden
- **Hex Value:** `#4CAF50`

---

## Common Integration Patterns

### Pattern 1: Direct Local Testing (Current)

```
JSON File (disk)
    ↓
GreenhouseStateApplier (reads file every 1s)
    ↓
Individual Controllers (apply state)
    ↓
Scene (lights, particles, audio)
```

### Pattern 2: FastAPI Backend (Future)

```
FastAPI Backend (Python)
    ↓ GET /state
    ↓
GreenhouseStateApplier (polls API every 1s)
    ↓
Individual Controllers (apply state)
    ↓
Scene (lights, particles, audio)
```

**Only the first read step changes. Everything after is identical.**

---

## Debugging Checklist

| Issue | Solution |
|-------|----------|
| Script not compiling | Check for missing references in Inspector |
| Scene not updating on JSON change | Verify file path is correct; check Console for errors |
| Health color won't change to Red | Check `debugManualControl` is `false`; ensure not all crops are Ripe |
| Crop stage not changing | Verify `cropStages` array is assigned and not empty |
| No blink even though all Ripe | Verify `cropHealth` is assigned; check health state is "Green" or all-Ripe override should apply |
| Reference warnings in Console | Drag missing controller from scene into the Inspector slot |

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial release: 8 actuators, 3 environment controllers, central JSON applier |
| - | Support for 15 crops (expandable) |
| - | LateUpdate health persistence |
| - | All-Ripe auto-blink override logic |

---

# Quick Start Summary

## What You Have

✅ **12 C# Scripts** — all production-ready, well-commented  
✅ **8 Actuator Controllers** — lights, effects, audio (all use `SetState(bool)`)  
✅ **3 Environment Controllers** — crop stages, time of day, health indicator  
✅ **1 Central Applier** — reads JSON, auto-applies state  
✅ **1 Sample JSON File** — ready to edit and test  
✅ **1 Integration Guide** — this document (everything you need)  

## What You Do

1. **Create `GreenhouseManager` GameObject** in your scene
2. **Attach `GreenhouseStateApplier`** component
3. **Drag 10 controller GameObjects** into Inspector slots
4. **Set JSON file path** (default: `Assets/StreamingAssets/greenhouse_state.json`)
5. **Press Play** — scene auto-updates as you edit the JSON

## Key Concepts

### Unified Control
Every actuator is commanded **identically**: `SetState(bool)`. You pass `true` or `false`. That's it.

### Single Crop Stage Value
The JSON has **one** `cropStage` value (not 15). It applies to all 15 crops simultaneously. This matches your real project where the backend sends a single stage value.

### Auto-Blink Logic
Green blinking **automatically activates** when all crops are Ripe. You cannot manually disable it — it's built in. Remove it from the JSON or set another state (Yellow/Red) stops blinking.

### Change Detection
The applier only re-applies when file content **actually changes**. Touching the file without changing content = no re-apply (efficient).

### LateUpdate Guarantee
Health color persists every frame, even if `CropStageController.Update()` tries to modify it. The applier always has the final say.

---

## Example JSON Commands

### Fully Ripe Greenhouse

```json
{
    "fluorescentLight":  { "isOn": true  },
    "heater":            { "isOn": true  },
    "energyCanister":    { "isOn": true  },
    "humidifier":        { "isOn": true  },
    "windowFan":         { "isOn": true  },
    "vent":              { "isOn": true  },
    "waterTankFloor":    { "isOn": true  },
    "cropStage":         { "stage": "Ripe" },
    "timeOfDay":         { "time": "Afternoon" },
    "cropHealth":        { "state": "Green", "blinkGreen": false }
}
```
**Result:** All lights on, all particles running, all audio playing, every crop shows Ripe model, green light **automatically blinking**.

### Seedling Startup (Morning)

```json
{
    "fluorescentLight":  { "isOn": true  },
    "heater":            { "isOn": false },
    "energyCanister":    { "isOn": false },
    "humidifier":        { "isOn": true  },
    "windowFan":         { "isOn": false },
    "vent":              { "isOn": false },
    "waterTankFloor":    { "isOn": true  },
    "cropStage":         { "stage": "Seedling" },
    "timeOfDay":         { "time": "Morning" },
    "cropHealth":        { "state": "Green", "blinkGreen": false }
}
```
**Result:** Lights on for growth, heater off (seedlings sensitive), humidifier on, all crops show Seedling model, morning lighting, green light (no blink).

### Stressed Crop (Emergency)

```json
{
    "fluorescentLight":  { "isOn": true  },
    "heater":            { "isOn": true  },
    "energyCanister":    { "isOn": true  },
    "humidifier":        { "isOn": false },
    "windowFan":         { "isOn": true  },
    "vent":              { "isOn": true  },
    "waterTankFloor":    { "isOn": false },
    "cropStage":         { "stage": "Flowering" },
    "timeOfDay":         { "time": "Night" },
    "cropHealth":        { "state": "Red", "blinkGreen": false }
}
```
**Result:** All systems active, humidifier off, crops in Flowering, night lighting, **red health light** (critical stress).

---

## Common Workflows

### Workflow 1: Develop Locally vs Integrate with Backend

```
Phase 1: Local Testing (NOW)
  └─ Edit JSON manually
  └─ Watch scene update in real-time
  └─ Verify all actuators work
  └─ Test all combinations

Phase 2: Backend Integration (FUTURE)
  └─ Replace ReadJsonFromDisk() with UnityWebRequest
  └─ Point to your FastAPI /state endpoint
  └─ Everything else unchanged
  └─ Drop into production
```

### Workflow 2: Adding a New Actuator

```
1. Create MyNewController.cs with SetState(bool)
2. Add to GreenhouseStateApplier class:
   - Public field: public MyNewController myNew;
   - Apply method: void ApplyMyNew(ActuatorState s) { ... }
   - Call in ApplyState(): ApplyMyNew(state.myNew);
3. Update GreenhouseState data class:
   - public ActuatorState myNew;
4. Update JSON schema and sample file
5. Done. No other scripts touched.
```

### Workflow 3: Debugging Why Health Won't Change

Check in this order:
1. Verify `cropHealth` component is assigned in Inspector
2. Verify `debugManualControl` is `false` on CropHealthIndicator
3. Check Console for `[GreenhouseStateApplier]` messages
4. If all crops are Ripe, health is **forced to Green** (this is intentional)
5. Try `forceRefresh` checkbox in Inspector to re-apply immediately

---

## File Format Quick Reference

| Field | Type | Valid Values |
|-------|------|---|
| `fluorescentLight.isOn` | bool | `true`, `false` |
| `heater.isOn` | bool | `true`, `false` |
| `energyCanister.isOn` | bool | `true`, `false` |
| `humidifier.isOn` | bool | `true`, `false` |
| `windowFan.isOn` | bool | `true`, `false` |
| `vent.isOn` | bool | `true`, `false` |
| `waterTankFloor.isOn` | bool | `true`, `false` |
| `cropStage.stage` | string | `Seedling`, `Vegetative`, `FloweringInitiation`, `Flowering`, `Unripe`, `Ripe` (case-insensitive) |
| `timeOfDay.time` | string | `Morning`, `Afternoon`, `Evening`, `Night` (case-insensitive) |
| `cropHealth.state` | string | `Green`, `Yellow`, `Red` (case-insensitive) |
| `cropHealth.blinkGreen` | bool | `true`, `false` (IGNORED — auto-computed) |

---

## Performance Notes

- **Polling Interval:** Default 1.0 seconds between file checks (configurable, 0.1–5.0 recommended)
- **CPU Impact:** Negligible — only string comparison on poll, no parsing until change detected
- **Memory:** ~5KB JSON file + state object in heap
- **Update Frequency:** All controllers updated within same Update() call once per poll
- **LateUpdate Overhead:** One extra `ApplyCropHealth()` call per frame (minimal)

---

**End of Integration Guide**
