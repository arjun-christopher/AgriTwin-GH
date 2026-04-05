# AgriTwin-GH — FastAPI Backend & Frontend Integration Guide

> **Status:** Complete — all endpoints implemented, smoke-tested, and live with the DT loop.  
> **Last updated:** April 2026  
> **Covers:** Architecture, all files created, layer-by-layer reference, endpoint surface,
> runtime state, live vs override mode, 3D-ready data, environment variables,
> running & testing, design decisions, and future roadmap.

---

## Table of Contents

1. [Overview](#1-overview)
2. [What Was There Before](#2-what-was-there-before)
3. [Full Architecture](#3-full-architecture)
4. [File Tree — Every New File](#4-file-tree--every-new-file)
5. [Layer-by-Layer Reference](#5-layer-by-layer-reference)  
   5.1 [Schema Layer](#51-schema-layer)  
   5.2 [API Layer — Route Handlers](#52-api-layer--route-handlers)  
   5.3 [Service Layer](#53-service-layer)  
   5.4 [Core / Runtime Layer](#54-core--runtime-layer)
6. [Files Modified (Not New)](#6-files-modified-not-new)
7. [Existing MPC Modules Reused](#7-existing-mpc-modules-reused)
8. [API Endpoint Reference](#8-api-endpoint-reference)
9. [Runtime State Management](#9-runtime-state-management)  
   9.1 [Data flow](#91-data-flow)  
   9.2 [Thread safety](#92-thread-safety)  
   9.3 [Override isolation](#93-override-isolation)
10. [Live Mode vs Override Mode](#10-live-mode-vs-override-mode)
11. [Manual Override Mechanism](#11-manual-override-mechanism)
12. [3D Integration Fields](#12-3d-integration-fields)
13. [Environment Variables](#13-environment-variables)
14. [Running the System](#14-running-the-system)
15. [Testing & Verification](#15-testing--verification)
16. [Design Decisions & Constraints](#16-design-decisions--constraints)
17. [Known Limitations](#17-known-limitations)
18. [Future Work](#18-future-work)

---

## 1. Overview

AgriTwin-GH already contained a complete MPC + Digital-Twin engine
(`src/agritwin_gh/mpc/`) and a React frontend (`src/agritwin_gh/frontend/`).
Before the integration the frontend's `api.js` was a stub returning mock data and
`main.py` was a `print("Hello from agritwin-gh!")` placeholder.

This guide documents the FastAPI integration that:
- makes all 15 `api.js` fetch calls hit real endpoints,
- drives them from the live MPC/DT loop,
- allows operators to switch between autonomous (`live`) and manual (`override`) modes,
- exposes 3D-ready scene metadata in every response (for a future Unity/WebGL layer),
- and is fully smoke-tested without requiring a running database.

**Key design rule:** the existing MPC/DT engine files are **never moved or modified**.
FastAPI wraps them; it does not replace them.

---

## 2. What Was There Before

| Location | Content | Status before integration |
|----------|---------|--------------------------|
| `src/agritwin_gh/api/__init__.py` | Empty shell | Pre-planned stub |
| `src/agritwin_gh/services/__init__.py` | Empty shell | Pre-planned stub |
| `src/agritwin_gh/core/__init__.py` | Empty shell | Pre-planned stub |
| `src/agritwin_gh/schemas/` | Did not exist | — |
| `src/agritwin_gh/mpc/` | 34 files — full MPC + DT engine | Production-ready, untouched |
| `src/agritwin_gh/models/` | Disease + growth inference models | Available |
| `src/agritwin_gh/utils/database.py` | `DatabaseManager` with SQLAlchemy | Available |
| `src/agritwin_gh/frontend/src/services/api.js` | 14 mock stubs | Stubbed with dummy data |
| `main.py` | `print("Hello from agritwin-gh!")` | Placeholder |
| `pyproject.toml` | No `fastapi` or `uvicorn` | Required adding |

---

## 3. Full Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        REACT FRONTEND (Vite, port 5173)                  │
│  src/agritwin_gh/frontend/src/services/api.js                            │
│  → 15 real fetch() calls to http://localhost:8000/api/*                  │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ HTTP + JSON (CORS pre-configured)
┌───────────────────────────────────▼──────────────────────────────────────┐
│                     FASTAPI APP  (Uvicorn, port 8000)                    │
│  main.py  →  agritwin_gh.api.app.create_app()                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    ROUTE HANDLERS  (thin adapters)                 │  │
│  │  agritwin_gh/api/routes/                                           │  │
│  │    dt.py · actuators.py · weather.py · intelligence.py            │  │
│  │    resources.py · media.py · system.py                            │  │
│  │                                                                    │  │
│  │  All routes: validate input → call service → return schema        │  │
│  │  Zero DB calls, zero MPC logic in route files                     │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │                   SERVICE LAYER  (business logic)                  │  │
│  │  agritwin_gh/services/                                             │  │
│  │    DashboardService    — read facade over RuntimeStore             │  │
│  │    ControlService      — write facade: overrides, presets          │  │
│  │    LoopService         — async background DT loop                  │  │
│  │    SystemService       — health checks from staleness              │  │
│  │    MediaService        — PostgreSQL/MinIO image queries            │  │
│  │                                                                    │  │
│  │  Wraps ──► MPC/DT core modules (import only, no copying)          │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │                  RUNTIME STORE  (in-process singleton)             │  │
│  │  agritwin_gh/core/runtime_store.py  →  RuntimeStore               │  │
│  │                                                                    │  │
│  │  Holds last DT step result; updated by the background loop        │  │
│  │  Route handlers read from it; never block on MPC solve            │  │
│  │  All as_*_response() projections live here                        │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    │ import only (no HTTP)
┌───────────────────────────────────▼──────────────────────────────────────┐
│                   MPC / DT ENGINE  (completely unchanged)                │
│  src/agritwin_gh/mpc/                                                    │
│    dt_loop.py        ← multi-rate closed-loop generator                 │
│    dt_engine.py      ← ARX physics step                                 │
│    dt_input_provider.py ← CSVInputProvider + AI models                  │
│    dt_runtime_prep.py   ← build_fused_state, build_mpc_solver           │
│    mpc_solver.py     ← CVXPY optimiser                                  │
│    realtime_core.py  ← STAGE_DURATION_HOURS, production DB loop         │
│    constants.py      ← GROWTH_STAGES, disease labels                    │
│    ...                                                                   │
│                                                                          │
│  src/agritwin_gh/models/                                                 │
│    disease_inference.py      ← Keras disease classifier                 │
│    growth_stage_inference.py ← Keras growth-stage classifier            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. File Tree — Every New File

```
src/agritwin_gh/
│
├── schemas/                         ← NEW: Pydantic v2 response/request models
│   ├── __init__.py                  Re-exports every public schema
│   ├── enums.py                     SystemStatus, AlertSeverity, GrowthStage, ActuatorMode
│   ├── dt_schemas.py                DTStateResponse, SceneContext, ActuatorVisualState, ThreeDPayload
│   ├── actuator_schemas.py          ActuatorStateResponse, ActuatorSetRequest, ActuatorSetResponse
│   ├── intelligence_schemas.py      DiseaseRisksResponse, DiseaseRisk, GrowthIntelResponse
│   ├── weather_schemas.py           WeatherResponse, WeatherForecastPoint
│   ├── resource_schemas.py          ResourcesResponse, ResourceMetric
│   ├── media_schemas.py             LatestMediaResponse, StageImagesResponse, DiseaseImagesResponse
│   └── system_schemas.py            SystemHealthResponse, HealthRow
│
├── api/                             ← EXTENDED (was empty __init__.py only)
│   ├── __init__.py                  (unchanged)
│   ├── app.py                       create_app() factory — ASGI app, CORS, 7 routers under /api
│   ├── dependencies.py              DI providers: get_runtime_store, get_dashboard_service,
│   │                                  get_control_service, get_media_service, get_db_session
│   └── routes/
│       ├── __init__.py
│       ├── dt.py                    GET /api/dt/state, POST /api/dt/override,
│       │                              POST /api/dt/override/sim, DELETE /api/dt/override,
│       │                              POST /api/dt/preset/{preset_id}
│       ├── actuators.py             GET /api/actuators/state, POST /api/actuators/set
│       ├── weather.py               GET /api/weather/current
│       ├── intelligence.py          GET /api/intelligence/disease, GET /api/intelligence/growth
│       ├── resources.py             GET /api/resources/monthly
│       ├── media.py                 GET /api/media/latest, /stage-images, /disease-scans
│       └── system.py               GET /api/system/health
│
├── services/                        ← EXTENDED (was empty __init__.py only)
│   ├── __init__.py                  (unchanged)
│   ├── dashboard_service.py         Read facade over RuntimeStore — one method per endpoint group
│   ├── control_service.py           Write facade — overrides, presets, actuator levels → RuntimeStore
│   ├── loop_service.py              Async background loop: drives DTLoop, commits LatestState
│   ├── system_service.py            Derives SystemHealthResponse from RuntimeStore staleness
│   ├── media_service.py             Queries PostgreSQL / MinIO; degrades to empty lists if unavailable
│   ├── actuator_service.py          Thin projection service (public surface of services/)
│   ├── weather_service.py           Thin projection service (public surface of services/)
│   ├── intelligence_service.py      Thin projection service (public surface of services/)
│   ├── resource_service.py          Thin projection service (public surface of services/)
│   └── dt_service.py                Placeholder skeleton for Phase 2 (live DB loop)
│
├── core/                            ← EXTENDED (was empty __init__.py only)
│   ├── __init__.py                  (unchanged)
│   ├── runtime_store.py             Thread-safe singleton (~1400 lines): LatestState, OverrideConfig,
│   │                                  all as_*_response() projection methods
│   └── lifespan.py                  FastAPI lifespan: startup init, optional background loop,
│                                      shutdown cleanup, TESTING guard
│
└── mpc/                             ← UNCHANGED — all files untouched
    └── (no changes)

main.py                              ← UPDATED: Uvicorn entry point calling create_app()
tests/test_api_smoke.py              ← NEW: pytest smoke suite (15 endpoints, TestClient)
scripts/sample_api_responses.py      ← NEW: stdlib-only manual endpoint sampler
```

---

## 5. Layer-by-Layer Reference

### 5.1 Schema Layer

**Location:** `src/agritwin_gh/schemas/`

Pure Pydantic v2 `BaseModel` classes — no business logic, no MPC imports.
Deliberately decoupled from MPC dataclasses so the API surface can evolve
independently. All fields use **snake_case**; the React frontend's `api.js`
maps to camelCase where needed.

| File | Key types defined |
|------|------------------|
| `enums.py` | `SystemStatus`, `AlertSeverity`, `GrowthStage`, `ActuatorMode` |
| `dt_schemas.py` | `DTStateResponse`, `SceneContext`, `ActuatorVisualState`, `ThreeDPayload`, override schemas |
| `actuator_schemas.py` | `ActuatorStateResponse`, `ActuatorSetRequest`, `ActuatorSetResponse` |
| `intelligence_schemas.py` | `DiseaseRisksResponse`, `DiseaseRisk`, `GrowthIntelResponse` |
| `weather_schemas.py` | `WeatherResponse`, `WeatherForecastPoint` |
| `resource_schemas.py` | `ResourcesResponse`, `ResourceMetric` |
| `media_schemas.py` | `LatestMediaResponse`, `StageImagesResponse`, `DiseaseImagesResponse`, `ImageMeta` |
| `system_schemas.py` | `SystemHealthResponse`, `HealthRow` |

Import from `src/agritwin_gh/schemas/__init__.py` only — never from individual schema files.
This prevents deep coupling if files are reorganised.

---

### 5.2 API Layer — Route Handlers

**Location:** `src/agritwin_gh/api/routes/`

Every route module is a thin adapter:

```
request arrives → path/query validation → call service method → return schema
```

Rules enforced across all route files:
- **No** database calls
- **No** MPC logic
- All handlers are `async`
- All routers are mounted under `/api` (no path prefix needed in `api.js`)
- FastAPI `Depends()` injects `DashboardService` (reads) and `ControlService` (writes)

**`app.py`** — `create_app()` factory:

```python
app = FastAPI(title="AgriTwin-GH", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=_CORS_ORIGINS, ...)
app.include_router(dt_router,          prefix="/api")
app.include_router(actuators_router,   prefix="/api")
app.include_router(weather_router,     prefix="/api")
app.include_router(intelligence_router,prefix="/api")
app.include_router(resources_router,   prefix="/api")
app.include_router(media_router,       prefix="/api")
app.include_router(system_router,      prefix="/api")
```

CORS origins pre-configured: `localhost:5173` (Vite dev), `localhost:4173`
(Vite preview) → `localhost:8000` (FastAPI).

---

### 5.3 Service Layer

**Location:** `src/agritwin_gh/services/`

Services are plain Python classes, not FastAPI dependencies, so they can be
unit-tested independently. Lazy imports inside methods prevent the circular
import issues common in large MPC codebases.

| Service | Role | Active? |
|---------|------|---------|
| `DashboardService` | Read facade over `RuntimeStore`; one method per endpoint group | ✅ Active |
| `ControlService` | Write facade: overrides, presets, actuator levels → `RuntimeStore` | ✅ Active |
| `LoopService` | Owns `DTLoop` lifecycle; `run_one_step()` feeds `RuntimeStore` every tick | ✅ Active |
| `SystemService` | Derives `SystemHealthResponse` from `RuntimeStore` staleness thresholds | ✅ Active |
| `MediaService` | Queries PostgreSQL / MinIO for image metadata; degrades to empty lists | ✅ Active |
| `ActuatorService` | Thin read projection for `services/__init__.py` public surface | ✅ Active |
| `WeatherService` | Thin read projection for `services/__init__.py` public surface | ✅ Active |
| `IntelligenceService` | Thin read projection for `services/__init__.py` public surface | ✅ Active |
| `ResourceService` | Thin read projection for `services/__init__.py` public surface | ✅ Active |
| `DTService` | Skeleton stub for Phase 2 (live DB-backed loop) | 🔲 Placeholder |

**`LoopService`** is the active loop driver (not `DTService`). It:
1. Creates `CSVInputProvider` (3 AI models started concurrently at init)
2. Creates `DTLoop` generator with `auto_advance_stage=True`
3. Calls `next(generator)` on each tick
4. Writes 10–15 `_phase()` log lines per step to `logs/dt_loop_YYYYMMDD.log`
5. Calls `RuntimeStore.update_from_step_result(result)`

See [DT_LOOP_STREAMING_GUIDE.md](DT_LOOP_STREAMING_GUIDE.md) for a complete
walk-through of every step of the loop.

---

### 5.4 Core / Runtime Layer

**Location:** `src/agritwin_gh/core/`

#### `runtime_store.py` — The State Bridge

Thread-safe singleton (~1400 lines). The single source of truth for every
`GET` endpoint. Background loop writes; HTTP handlers read — never the reverse.

**What it stores:**

| Snapshot type | Fields |
|---------------|--------|
| `ClimateSnapshot` | temperature, humidity, CO₂, soil moisture, light, VPD, leaf wetness, disease risk |
| `ActuatorSnapshot` | fan, vent, heater, LED, CO₂ valve, fogger, irrigation — levels + mode |
| `WeatherSnapshot` | outdoor temperature, humidity, solar, wind, conditions |
| `DiseaseSnapshot` | overall risk + per-pathogen breakdown (5 diseases) |
| `GrowthSnapshot` | current stage, next stage, hours to transition, confidence |
| `ResourceSnapshot` | cumulative kWh and litres since loop start |
| `MediaSnapshot` | presigned MinIO image URLs or local path references |
| `OverrideConfig` | active flag + operator-set target values (never merged into `LatestState`) |

**Projection methods** (`as_*_response()`) — called by every `GET` endpoint:

```python
store.as_dt_state_response()       → DTStateResponse
store.as_actuator_state_response() → ActuatorStateResponse
store.as_weather_response()        → WeatherResponse
store.as_disease_risks_response()  → DiseaseRisksResponse
store.as_growth_intel_response()   → GrowthIntelResponse
store.as_resources_response()      → ResourcesResponse
store.as_system_health_response()  → SystemHealthResponse
```

Each projection runs under `threading.RLock` — concurrent HTTP requests
cannot observe a partial write.

#### `lifespan.py` — FastAPI Startup / Shutdown

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── STARTUP ────────────────────────────────────────────────────────
    store   = RuntimeStore()
    service = LoopService(store=store)
    service.start_loop(
        growth_stage      = os.getenv("AGRITWIN_GROWTH_STAGE", "seedling"),
        total_steps       = int(os.getenv("AGRITWIN_TOTAL_STEPS", "25632")),
        auto_advance_stage= _env_bool("AGRITWIN_AUTO_ADVANCE_STAGE", True),
    )
    service.run_one_step()          # seed RuntimeStore before first request
    if _env_bool("AGRITWIN_BACKGROUND_LOOP", True) and not _testing:
        asyncio.create_task(service.run_background_loop(interval_seconds=...))
    app.state.store   = store
    app.state.service = service
    yield
    # ── SHUTDOWN ───────────────────────────────────────────────────────
    service.stop_loop()
```

`TESTING=1` disables both the background task and all DB connections so
`pytest` can run with no live infrastructure.

---

## 6. Files Modified (Not New)

| File | What changed |
|------|-------------|
| `main.py` | Replaced `print()` placeholder with `uvicorn.run(create_app(), ...)` |
| `src/agritwin_gh/frontend/src/services/api.js` | Fully rewritten: 14 mock stubs → 15 real `fetch()` functions + `deleteDtOverride()` |
| `src/agritwin_gh/frontend/src/pages/HomeDashboard.jsx` | Wired to live API: `fetchDtState`, `fetchWeather`, `fetchResources`, `fetchSystemHealth` |
| `src/agritwin_gh/frontend/src/pages/DetailedInsights.jsx` | Wired to live API: `fetchDiseaseRisks`, `fetchGrowthIntel`, `fetchActuatorState`, `fetchStageImages`, `fetchDiseaseScans` |
| `src/agritwin_gh/frontend/src/pages/ManualOverride.jsx` | Wired to live API: `postDtOverride`, `deleteDtOverride`; field-level error state; reads `response.mode` for live/override badge |

---

## 7. Existing MPC Modules Reused

No MPC or DT logic was cloned. FastAPI is a thin import wrapper only.

| Existing module | Consumed by | Key class / function |
|----------------|-------------|---------------------|
| `mpc.dt_loop.DTLoop` | `LoopService` | Generator ticked per background step |
| `mpc.dt_input_provider.CSVInputProvider` | `LoopService` (via `start_loop`) | 3 AI models initialised concurrently |
| `mpc.dt_engine.DigitalTwinEngine` | `DTLoop` (unchanged) | ARX physics step |
| `mpc.dt_runtime_prep.build_fused_state` | `DTLoop` → `build_mpc_solver` | FusedState for MPC |
| `mpc.mpc_solver.MPCRunner` | `ControlService` | One MPC solve when applying presets |
| `mpc.realtime_core.STAGE_DURATION_HOURS` | `DTLoop` (auto-advance) | Stage transition timing |
| `mpc.constants.GROWTH_STAGES` | `DTLoop`, `RuntimeStore` | Stage ordering |
| `mpc.disturbance.WeatherDisturbanceForecast` | `CSVInputProvider` | 24h weather forecast |
| `models.disease.TomatoDiseaseClassifier` | `RuntimeStore` (via `IntelligenceService`) | Disease CNN inference |
| `models.growth.TomatoGrowthStageClassifier` | `RuntimeStore` (via `IntelligenceService`) | Growth CNN inference |
| `utils.database.DatabaseManager` | `get_db_session()` + `MediaService` | SQLAlchemy sessions |
| `utils.minio_client.MinIOClient` | `MediaService` | Presigned image URLs |
| `config.settings` | `lifespan.py` | Poll interval, DB URL, MinIO credentials |

---

## 8. API Endpoint Reference

All paths are relative to `http://localhost:8000`.  
Interactive docs: `http://localhost:8000/docs`

| Method | Path | Request | Response schema | Backend source |
|--------|------|---------|-----------------|----------------|
| `GET` | `/api/dt/state` | — | `DTStateResponse` | `RuntimeStore.as_dt_state_response()` |
| `POST` | `/api/dt/override` | `{"param": str, "value": any}` | `DtOverrideResponse` | `ControlService.set_override()` |
| `POST` | `/api/dt/override/sim` | `{"stage": str, "day_in_stage": int, ...}` | `DtOverrideResponse` | `ControlService.simulate_override()` |
| `DELETE` | `/api/dt/override` | — | `DtOverrideResponse` | `ControlService.clear_override()` |
| `POST` | `/api/dt/preset/{preset_id}` | — | `DtPresetResponse` | `ControlService.apply_preset()` |
| `GET` | `/api/actuators/state` | — | `ActuatorStateResponse` | `RuntimeStore.as_actuator_state_response()` |
| `POST` | `/api/actuators/set` | `{"actuators": [{id, level, on}]}` | `ActuatorSetResponse` | `ControlService.set_actuator()` |
| `GET` | `/api/weather/current` | — | `WeatherResponse` | `RuntimeStore.as_weather_response()` |
| `GET` | `/api/intelligence/disease` | — | `DiseaseRisksResponse` | `RuntimeStore.as_disease_risks_response()` |
| `GET` | `/api/intelligence/growth` | — | `GrowthIntelResponse` | `RuntimeStore.as_growth_intel_response()` |
| `GET` | `/api/resources/monthly` | — | `ResourcesResponse` | `RuntimeStore.as_resources_response()` |
| `GET` | `/api/media/latest` | — | `LatestMediaResponse` | `MediaService.get_latest()` |
| `GET` | `/api/media/stage-images` | — | `StageImagesResponse` | `MediaService.get_stage_images()` |
| `GET` | `/api/media/disease-scans` | — | `DiseaseImagesResponse` | `MediaService.get_disease_scans()` |
| `GET` | `/api/system/health` | — | `SystemHealthResponse` | `SystemService.get_health()` |

### Named presets (`POST /api/dt/preset/{preset_id}`)

| `preset_id` | Environment configured |
|-------------|----------------------|
| `high-growth` | Optimal temperature + CO₂ for vegetative growth |
| `water-save` | Reduced irrigation targets |
| `night-mode` | Low light, reduced HVAC |
| `disease-alert` | Low humidity, high airflow to slow pathogen spread |
| `harvest-ready` | Pre-harvest ripening conditions |

Invalid preset ID → HTTP 422.

---

## 9. Runtime State Management

### 9.1 Data Flow

```
CSVInputProvider (startup — 3 AI models in parallel)
    │
    ▼
DTLoop generator created (auto_advance_stage=True, total_steps=25632)
    │
    ▼  every 5 min (wall-clock), driven by LoopService.run_background_loop()
LoopService.run_one_step()
    │
    ├─► next(DTLoop.run())     ← 5-min physics + MPC (every 15 min) + CNN (every 30 min)
    │       yields DTLoopStepResult
    │
    ├─► _phase() logging       ← structured log to logs/dt_loop_YYYYMMDD.log
    │
    └─► RuntimeStore.update_from_step_result(result)    ← fast locked write
            │
            ▼
     FastAPI GET /api/* request (any time)
            │
            ▼
     RuntimeStore.as_*_response()   ← pure projection, no computation
            │
            ▼
     Pydantic schema → JSON → React frontend   (total round-trip < 1 ms)
```

> **Why it's fast:** The DT physics engine, MPC solver, and AI models all run
> in the background loop. HTTP handlers only read already-computed values from
> `RuntimeStore` — no blocking on any model inference or optimisation.

### 9.2 Thread Safety

`RuntimeStore` uses `threading.RLock` around:
- `update_from_step_result()` — write path (background loop)
- every `as_*_response()` — read path (HTTP handler)

Concurrent HTTP requests cannot observe a partially written state.

### 9.3 Override Isolation

`OverrideConfig` is stored **separately** from `LatestState` in `RuntimeStore`.
Applying an override never mutates the simulated greenhouse state. Only the
projection layer (`as_dt_state_response()`) merges the two: if
`OverrideConfig.active is True`, override values replace MPC values in the
JSON response.

This means:
- The DT loop continues running unchanged in the background during an override.
- Clearing the override (`DELETE /api/dt/override`) instantly reverts to the
  loop's latest physics output at zero cost.

---

## 10. Live Mode vs Override Mode

The `mode` field in `GET /api/dt/state` has two values:

### `"live"` — MPC-autonomous control

- DT loop runs independently; all sensor/actuator fields reflect MPC output.
- No operator-set values are stored.
- React `ManualOverride` page shows **Live / Current** pill.

### `"override"` — Operator control

- Triggered by `POST /api/dt/override/sim` (stage / day in stage / date / hour)
  or `POST /api/actuators/set` (explicit actuator levels).
- `OverrideConfig` is stored alongside `LatestState`.
- `GET /api/dt/state` immediately reflects override values — no wait for next loop tick.
- Actuators set via `POST /api/actuators/set` show `status = "MANUAL"`.
- React `ManualOverride` page shows **Manual Override** pill + orange warning banner.
- Cleared by `DELETE /api/dt/override` → mode reverts to `"live"`.

**State machine:**

```
live ──POST /api/dt/override/sim──► override
live ──POST /api/actuators/set────► override
override ──DELETE /api/dt/override► live
```

**Simulate without committing** — `POST /api/dt/override/sim` returns the
projected state without calling `set_override()`, so the operator can preview
the effect before applying it.

---

## 11. Manual Override Mechanism

| Action | Endpoint | Result |
|--------|----------|--------|
| Define crop stage / day in stage | `POST /api/dt/override/sim` | `mode = "override"`, override blended into `GET /api/dt/state` |
| Set specific actuator levels | `POST /api/actuators/set` | `mode = "override"`, actuator status → `MANUAL` |
| Apply a named environmental preset | `POST /api/dt/preset/{preset_id}` | `mode = "override"`, preset levels applied immediately |
| Clear all overrides | `DELETE /api/dt/override` | `mode = "live"`, actuator status → `auto` |

The frontend reads `response.mode` from these responses to update the UI badge
without an extra `GET /api/dt/state` round-trip.

---

## 12. 3D Integration Fields

Every `GET /api/dt/state` response already contains 3D scene metadata at zero
extra compute cost. The current React dashboard does not consume it. A future
Unity or WebGL layer can subscribe without any API change.

### Compact top-level fields

```json
{
  "time_of_day":           "morning",
  "current_growth_stage":  "flowering",
  "next_growth_stage":     "unripe",
  "actuator_visual_state": {
    "on_off":  {"fan": true,  "led": false, "co2": true, "heater": false, "vent": true, "fogger": false, "irrigation": false},
    "levels":  {"fan": 75.0, "led": 0.0,  "co2": 55.0, "heater": 0.0,  "vent": 60.0, "fogger": 0.0,  "irrigation": 0.0},
    "mode":    "auto"
  }
}
```

### Full `scene_context` block

```json
{
  "scene_context": {
    "actuator_states_on_off":  {"fan": true, ...},
    "actuator_levels":         {"fan": 75.0, ...},
    "current_growth_stage":    "flowering",
    "next_growth_stage":       "unripe",
    "current_timestamp":       "2026-04-05T09:15:00",
    "time_of_day":             "morning"
  }
}
```

### `ThreeDPayload` block

```json
{
  "three_d": {
    "plant_health_score":  0.84,
    "disease_risk_score":  0.16,
    "growth_stage_label":  "flowering",
    "growth_stage_index":  3,
    "alert_count":         0
  }
}
```

**Time-of-day buckets:**

| Bucket | Hours |
|--------|-------|
| `morning` | 06:00 – 11:59 |
| `afternoon` | 12:00 – 17:59 |
| `evening` | 18:00 – 20:59 |
| `night` | 21:00 – 05:59 |

**3D rendering hints:**

- Swap the plant mesh when `current_growth_stage` changes.
- Animate actuator props (fan RPM, LED glow, CO₂ cloud) from `actuator_levels`.
- Set ambient lighting and sky-box from `time_of_day`.
- Use `next_growth_stage` for transition animations.
- Drive health shader intensity from `three_d.plant_health_score`.

`GET /api/actuators/state` can also be polled directly for actuator levels
at higher frequency without fetching the full DT state.

---

## 13. Environment Variables

### Backend (`.env` in repo root)

```powershell
cp .env.example .env   # create from template
```

| Variable | Default | Description |
|----------|---------|-------------|
| `AGRITWIN_BACKGROUND_LOOP` | `1` | `1` = DT loop advances automatically every interval |
| `AGRITWIN_GROWTH_STAGE` | `"seedling"` | Starting crop stage (`seedling` → `ripe`) |
| `AGRITWIN_TOTAL_STEPS` | `25632` | Total simulation steps (25632 = 89 days at 5 min/step) |
| `AGRITWIN_AUTO_ADVANCE_STAGE` | `1` | `1` = loop auto-advances through all 6 growth stages |
| `AGRITWIN_STEP_INTERVAL_SEC` | `300.0` | Wall-clock seconds between DT steps |
| `AGRITWIN_DAYS_ELAPSED` | `0.0` | Days already elapsed within the starting stage |
| `AGRITWIN_NO_FRONTEND` | `0` | `1` = skip serving static frontend files |
| `TESTING` | `0` | `1` = lifespan skips loop + DB (used by pytest) |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` | — | PostgreSQL credentials (only needed for `MediaService`) |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint for image retrieval |

### Frontend (`src/agritwin_gh/frontend/.env.local`)

```
VITE_API_BASE_URL=http://localhost:8000
```

Consumed by `api.js` as `import.meta.env.VITE_API_BASE_URL`. Change for
staging/production deployments.

---

## 14. Running the System

### Prerequisites

```powershell
# From repo root
.venv\Scripts\Activate.ps1
python -c "import fastapi, uvicorn; print('OK')"
```

### Backend — development (auto-reload)

```powershell
$env:PYTHONPATH = "src"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or via the entry point:

```powershell
python main.py
```

After start, browse to:
- **Swagger UI:** `http://localhost:8000/docs`
- **Quick check:** `http://localhost:8000/api/dt/state`

### Frontend — development

```powershell
cd src\agritwin_gh\frontend
npm install        # first time only
npm run dev        # → http://localhost:5173
```

Production build:

```powershell
npm run build      # output → src/agritwin_gh/frontend/dist/
```

### Running both together

**Terminal 1 — backend:**

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend:**

```powershell
cd src\agritwin_gh\frontend
npm run dev
```

Open `http://localhost:5173` — CORS is pre-configured.

### Common launch variants

**24-hour demo in flowering stage (no stage advance):**

```powershell
$env:AGRITWIN_GROWTH_STAGE       = "flowering"
$env:AGRITWIN_TOTAL_STEPS        = "288"
$env:AGRITWIN_AUTO_ADVANCE_STAGE = "0"
python main.py
```

**Rapid development (5-second steps, no frontend):**

```powershell
$env:AGRITWIN_STEP_INTERVAL_SEC = "5"
$env:AGRITWIN_NO_FRONTEND       = "1"
python main.py
```

**API-driven step mode (no automatic advance):**

```powershell
$env:AGRITWIN_BACKGROUND_LOOP = "0"
python main.py
# Then push each step via: POST http://localhost:8000/api/loop/step
```

---

## 15. Testing & Verification

### Automated smoke tests (no running server required)

```powershell
$env:PYTHONPATH = "src"
pytest tests/test_api_smoke.py -v
```

Uses FastAPI's in-process `TestClient`. `TESTING=1` is set automatically
so the lifespan skips the DB loop — no PostgreSQL needed.

**Coverage:**

| Endpoint | What is tested |
|----------|---------------|
| `GET /api/dt/state` | HTTP 200, all fields present, `mode` field, 3D blocks |
| `GET /api/actuators/state` | HTTP 200, 7 actuators, level in [0,100], status field |
| `GET /api/weather/current` | HTTP 200, `current` + `forecast` list shape |
| `GET /api/intelligence/disease` | HTTP 200, `composite_risk` in [0,1], pathogens list |
| `GET /api/intelligence/growth` | HTTP 200, stage fields, `stage_history` |
| `GET /api/system/health` | HTTP 200, `rows` list, bool `ok` |
| `POST /api/dt/override` | HTTP 200, `ok=True`, echoed fields |
| `POST /api/dt/override/sim` | HTTP 200, `mode = "override"`, stage reflected in state |
| `DELETE /api/dt/override` | HTTP 200, `mode = "live"` after clear |
| `POST /api/actuators/set` | HTTP 200, unknown ID skipped, level clamped, `MANUAL` status |
| `POST /api/dt/preset/{id}` | All 5 presets → 200; invalid preset → 422 |

### Live endpoint sampler (requires running server)

```powershell
python main.py                           # terminal 1
python scripts/sample_api_responses.py  # terminal 2
```

Hits all endpoints, pretty-prints JSON excerpts, exits 1 if any fail.

```powershell
# Override URL if needed:
$env:AGRITWIN_API_URL = "http://localhost:8000"
python scripts/sample_api_responses.py
```

### Manual PowerShell checks

```powershell
# Full state snapshot
Invoke-RestMethod http://localhost:8000/api/dt/state | ConvertTo-Json -Depth 4

# Apply a sim override
$body = '{"stage":"flowering","day_in_stage":7,"start_date":"2026-04-04","start_hour":9}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/dt/override/sim `
    -ContentType "application/json" -Body $body

# Confirm mode changed
(Invoke-RestMethod http://localhost:8000/api/dt/state).mode   # → "override"

# Clear override
Invoke-RestMethod -Method Delete -Uri http://localhost:8000/api/dt/override

# Confirm reverted
(Invoke-RestMethod http://localhost:8000/api/dt/state).mode   # → "live"

# Apply preset
Invoke-RestMethod -Method Post http://localhost:8000/api/dt/preset/disease-alert

# Check actuator override
$body = '[{"id":"fan","level":90,"on":true}]'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/actuators/set `
    -ContentType "application/json" -Body $body

# System health
Invoke-RestMethod http://localhost:8000/api/system/health | ConvertTo-Json -Depth 3
```

---

## 16. Design Decisions & Constraints

| Decision | Rationale |
|----------|-----------|
| FastAPI app factory (`create_app()`) | Each test gets a fresh app instance — no singleton pollution between test classes |
| `RuntimeStore` as in-process singleton | HTTP handlers never block on MPC solve; sub-millisecond reads; correct for single-worker deployment |
| Schemas separate from MPC dataclasses | API surface can evolve independently; avoids coupling FastAPI to internal physics types |
| Lazy imports inside service methods | Prevents circular imports in the 34-file MPC module graph |
| `SceneContext` + `ThreeDPayload` in every DT response | Zero-cost 3D readiness; no future schema version needed |
| No changes to any file in `mpc/` | Existing smoke tests, evaluation scripts, and CLI runner all remain valid |
| Background loop **enabled** by default | `python main.py` runs the full 89-day simulation immediately; set `AGRITWIN_BACKGROUND_LOOP=0` to disable |
| Single worker process | `RuntimeStore` is in-process; multi-worker deployment needs Redis (Phase 3+) |
| `OverrideConfig` never mutates `LatestState` | Override and simulation are isolated; no state corruption; instant clear |
| `DashboardService` is the only injected read service into routes | Thin services (`ActuatorService`, etc.) exist only for the `services/__init__.py` public surface, not for direct injection |
| Tamil Nadu INR tariffs as energy/water cost fallback | Matches constants in `docs/MPC_COMPLETE_GUIDE.md` §18 |

---

## 17. Known Limitations

| Area | Limitation | Resolution |
|------|-----------|------------|
| **Multi-worker** | `RuntimeStore` is in-process; multiple Uvicorn workers have divergent state | Phase 3: replace with Redis-backed shared state |
| **Media endpoints** | `GET /api/media/*` returns placeholder images until MinIO is configured | Configure MinIO per [Image Storage Setup](IMAGE_STORAGE_SETUP.md) |
| **Disease / growth ML** | `IntelligenceService` returns synthetic vectors until Keras model paths are set | Set paths in `config/settings.yaml`; train with `notebooks/tomato_disease_classifier_train.ipynb` |
| **Weather forecast** | `GET /api/weather/current` returns a synthetic diurnal forecast until wired to a real weather API | Set `WEATHER_API_KEY` in `.env` |
| **Override persistence** | `OverrideConfig` is RAM-only; server restart clears all overrides | Accepted by design for Phase 1; DB persistence is Phase 3+ |
| **Frontend actuator sliders** | `ManualOverride` exposes on/off toggles only (no level sliders); level defaults to 100 when toggled on | UI iteration planned |
| **CORS in production** | `_CORS_ORIGINS` in `api/app.py` is hardcoded to `localhost` | Add production domain before any public deployment |

---

## 18. Future Work

| Phase | What it adds |
|-------|-------------|
| **Phase 2 — Live DB loop** | Wire `DTService` (current placeholder) to PostgreSQL sensor reads; replace synthetic CSV loop with a `RealtimeLoop`. `dt_service.py` is the ready stub. |
| **Phase 3 — RealtimeLoop + MinIO** | Replace `LoopService` with `RealtimeLoop` reading from live sensor stream; write images to MinIO for `MediaService` to serve with presigned URLs. |
| **Phase 4 — ML classifiers on live data** | Run `TomatoDiseaseClassifier` and `TomatoGrowthStageClassifier` on every loop tick; commit predictions to `RuntimeStore.disease_snapshot` and `growth_snapshot`. |
| **Phase 5 — 3D WebSocket** | Add `WS /api/ws/state` emitting `DTStateResponse` diffs at ~1 Hz. All required data fields already exist in the HTTP response. No API schema changes needed. |
| **Phase 6 — Redis + multi-worker** | Replace in-process `RuntimeStore` with Redis pub/sub so multiple Uvicorn workers share live state. |

---

*For the DT loop cadence, per-step log format, and AI model refresh mechanics see [DT_LOOP_STREAMING_GUIDE.md](DT_LOOP_STREAMING_GUIDE.md).*  
*For MPC cost function, constraints, and setpoints see [MPC_COMPLETE_GUIDE.md](MPC_COMPLETE_GUIDE.md).*  
*For the React dashboard pages, components, and data-binding contract see [FRONTEND_UI_REFERENCE.md](FRONTEND_UI_REFERENCE.md).*  
*For PostgreSQL schema and database setup see [POSTGRESQL_QUICKSTART.md](POSTGRESQL_QUICKSTART.md).*  
*For MinIO image pipeline setup see [IMAGE_STORAGE_SETUP.md](IMAGE_STORAGE_SETUP.md).*
