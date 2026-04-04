# AgriTwin-GH — FastAPI Integration Plan

> **Status:** Architecture scaffold complete — implementation pending.  
> **Date created:** April 2026  
> **Authors:** Copilot-assisted design pass over existing codebase

---

## Table of Contents

1. [Overview](#1-overview)
2. [What Was Found in the Repo](#2-what-was-found-in-the-repo)
3. [Integration Architecture](#3-integration-architecture)
4. [Canonical Modules Reused](#4-canonical-modules-reused)
5. [File Tree — New Files Created](#5-file-tree--new-files-created)
6. [Layer-by-Layer Reference](#6-layer-by-layer-reference)  
   6.1 [Schema Layer](#61-schema-layer)  
   6.2 [API Layer (Route Handlers)](#62-api-layer-route-handlers)  
   6.3 [Service Layer](#63-service-layer)  
   6.4 [Core / Runtime Layer](#64-core--runtime-layer)
7. [API Endpoint Inventory](#7-api-endpoint-inventory)
8. [3D Environment Integration Fields](#8-3d-environment-integration-fields)
9. [RuntimeStore — State Bridge](#9-runtimestore--state-bridge)
10. [How to Add a New Dependency (FastAPI + Uvicorn)](#10-how-to-add-fastapi--uvicorn)
11. [Implementation Phases](#11-implementation-phases)
12. [Design Decisions & Constraints](#12-design-decisions--constraints)

---

## 1. Overview

AgriTwin-GH already contained a complete MPC + Digital-Twin engine
(`src/agritwin_gh/mpc/`) and a React frontend (`src/agritwin_gh/frontend/`).
The frontend's `api.js` was a stub layer that returned mock data.

This plan describes the FastAPI integration that makes every `api.js` function
call a real backend endpoint, sourcing data from the live MPC/DT engine.

**Key constraint:** the existing MPC/DT engine files are **not moved or
modified**. FastAPI wraps them; it does not replace them.

---

## 2. What Was Found in the Repo

| Location | Content | Status |
|----------|---------|--------|
| `src/agritwin_gh/api/__init__.py` | Empty shell | Pre-planned |
| `src/agritwin_gh/services/__init__.py` | Empty shell | Pre-planned |
| `src/agritwin_gh/core/__init__.py` | Empty shell | Pre-planned |
| `src/agritwin_gh/mpc/` | 34 files — full MPC + DT engine | Production-ready |
| `src/agritwin_gh/models/` | Disease + growth inference models | Available |
| `src/agritwin_gh/utils/database.py` | `DatabaseManager` with SQLAlchemy | Available |
| `src/agritwin_gh/frontend/src/services/api.js` | All 14 endpoint stubs | Stubbed |
| `main.py` | `print("Hello from agritwin-gh!")` | Replaced |
| `pyproject.toml` | No `fastapi` or `uvicorn` | **Must add** |

---

## 3. Integration Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        REACT FRONTEND (Vite, port 5173)                  │
│  src/agritwin_gh/frontend/src/services/api.js                            │
│  → 14 fetch() calls to http://localhost:8000/api/*                        │
└───────────────────────────────────┬──────────────────────────────────────┘
                                    │ HTTP + JSON
┌───────────────────────────────────▼──────────────────────────────────────┐
│                     FASTAPI APP  (Uvicorn, port 8000)                    │
│  main.py  →  agritwin_gh.api.app.create_app()                            │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐  │
│  │                    ROUTE HANDLERS  (thin)                          │  │
│  │  agritwin_gh/api/routes/                                           │  │
│  │    dt.py · actuators.py · weather.py · intelligence.py            │  │
│  │    resources.py · media.py · system.py                            │  │
│  │                                                                    │  │
│  │  Depend on → RuntimeStore (via Depends)                           │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │                   SERVICE LAYER  (business logic)                  │  │
│  │  agritwin_gh/services/                                             │  │
│  │    DTService           · ActuatorService  · WeatherService         │  │
│  │    IntelligenceService · ResourceService  · MediaService           │  │
│  │                                                                    │  │
│  │  Wraps ──► MPC/DT core modules (no copying, import only)          │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
│                             │                                            │
│  ┌──────────────────────────▼─────────────────────────────────────────┐  │
│  │                  RUNTIME STORE  (in-process singleton)             │  │
│  │  agritwin_gh/core/runtime_store.py  →  RuntimeStore               │  │
│  │                                                                    │  │
│  │  Holds last DT step result; updated by the background loop.       │  │
│  │  Route handlers read from it; never block on MPC solve.           │  │
│  └──────────────────────────┬─────────────────────────────────────────┘  │
└───────────────────────────────────┼──────────────────────────────────────┘
                                    │ import only (no HTTP)
┌───────────────────────────────────▼──────────────────────────────────────┐
│                   MPC / DT ENGINE  (unchanged)                           │
│  src/agritwin_gh/mpc/                                                    │
│    realtime_core.py  ← production DB-backed loop                        │
│    dt_loop.py        ← synthetic simulation loop                        │
│    mpc_solver.py     ← SLSQP optimiser                                  │
│    state_fusion.py   ← assembles FusedState                             │
│    image_streamer.py ← image_metadata DB queries                        │
│    dt_engine.py      ← ARX physics step                                 │
│    constants.py      ← single source of truth for labels                │
│    ...                                                                   │
│                                                                          │
│  src/agritwin_gh/models/                                                 │
│    disease_inference.py      ← Keras disease classifier                 │
│    growth_stage_inference.py ← Keras growth-stage classifier            │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Canonical Modules Reused

| Service | Reused MPC/core module | Key class / function |
|---------|----------------------|---------------------|
| `DTService` | `mpc/realtime_core.py` | `RealtimeLoop`, `RealtimeLoopConfig` |
| `DTService` (synthetic) | `mpc/dt_loop.py` | `DTLoop` |
| `DTService` | `mpc/dt_runtime_prep.py` | `prepare_initial_state`, `build_mpc_solver` |
| `ActuatorService` | `mpc/state.py` | `ActuatorState` |
| `ActuatorService` | `mpc/constants.py` | `CONTROL_VARIABLES` |
| `ActuatorService` | `mpc/mpc_solver.py` | `MPCSolution.u_optimal` |
| `WeatherService` | `mpc/disturbance.py` | `WeatherDisturbanceForecast` |
| `WeatherService` | `mpc/dt_runtime_prep.py` | `prepare_weather_sequence` |
| `IntelligenceService` | `mpc/disease_penalty.py` | `DiseaseRiskPenalty` |
| `IntelligenceService` | `mpc/growth_weights.py` | `GrowthStageWeights` |
| `IntelligenceService` | `models/disease_inference.py` | `predict_image` |
| `IntelligenceService` | `models/growth_stage_inference.py` | `predict_growth_stage` |
| `ResourceService` | `mpc/config.py` | `MPCConfig` (tariff constants) |
| `MediaService` | `mpc/image_streamer.py` | `ImageStreamer` |
| `MediaService` | `config/minio_config.py` | MinIO client factory |
| All services | `utils/database.py` | `DatabaseManager.get_session()` |
| All schemas | `mpc/constants.py` | `GROWTH_STAGES`, `DISEASE_CATEGORIES` |

---

## 5. File Tree — New Files Created

```
src/agritwin_gh/
│
├── schemas/                         ← NEW: Pydantic response/request models
│   ├── __init__.py
│   ├── dt_schemas.py                → DTStateResponse, SceneContext, CropInfo …
│   ├── actuator_schemas.py          → ActuatorStateResponse, ActuatorSetRequest …
│   ├── weather_schemas.py           → WeatherResponse, ForecastEntry …
│   ├── intelligence_schemas.py      → DiseaseRisksResponse, GrowthIntelResponse …
│   ├── resource_schemas.py          → ResourcesResponse, MonthlyCost …
│   ├── media_schemas.py             → LatestMediaResponse, ImageEntry …
│   └── system_schemas.py            → SystemHealthResponse, HealthRow
│
├── api/                             ← EXTENDED (was empty __init__.py only)
│   ├── __init__.py                  (unchanged)
│   ├── app.py                       ← NEW: FastAPI app factory + CORS + routers
│   ├── dependencies.py              ← NEW: DB session, RuntimeStore DI providers
│   └── routes/                      ← NEW: one module per resource group
│       ├── __init__.py
│       ├── dt.py                    → GET /api/dt/state, POST /api/dt/override …
│       ├── actuators.py             → GET/POST /api/actuators/*
│       ├── weather.py               → GET /api/weather/current
│       ├── intelligence.py          → GET /api/intelligence/disease|growth
│       ├── resources.py             → GET /api/resources/monthly
│       ├── media.py                 → GET /api/media/latest|stage-images|disease-scans
│       └── system.py                → GET /api/system/health
│
├── services/                        ← EXTENDED (was empty __init__.py only)
│   ├── __init__.py                  (unchanged)
│   ├── dt_service.py                ← NEW: DTService — manages DT/MPC loop lifecycle
│   ├── actuator_service.py          ← NEW: ActuatorService — actuator state + overrides
│   ├── weather_service.py           ← NEW: WeatherService — forecast adapter
│   ├── intelligence_service.py      ← NEW: IntelligenceService — disease + growth models
│   ├── resource_service.py          ← NEW: ResourceService — energy/water accounting
│   └── media_service.py             ← NEW: MediaService — MinIO presigned URLs
│
├── core/                            ← EXTENDED (was empty __init__.py only)
│   ├── __init__.py                  (unchanged)
│   ├── runtime_store.py             ← NEW: RuntimeStore — thread-safe in-memory state
│   └── lifespan.py                  ← NEW: FastAPI lifespan startup/shutdown hooks
│
└── mpc/                             ← UNCHANGED — all 34 files untouched
    └── (no changes)

main.py                              ← UPDATED: now a Uvicorn entry point
```

---

## 6. Layer-by-Layer Reference

### 6.1 Schema Layer

`src/agritwin_gh/schemas/`

* Pure Pydantic v2 `BaseModel` classes — no business logic.
* Deliberately decoupled from the MPC dataclasses so the API surface can
  evolve independently.
* All schemas use **snake_case** field names; the React frontend uses camelCase
  — the `api.js` stub can rename fields during the integration swap.

### 6.2 API Layer (Route Handlers)

`src/agritwin_gh/api/routes/`

* Each route module is a thin adapter: validate input → call service →
  return schema.
* No database calls, no MPC logic.
* All route handlers are `async` for non-blocking I/O.
* All routers are mounted under `/api` so the frontend `BASE_URL` stays
  `http://localhost:8000` with no path prefix.

### 6.3 Service Layer

`src/agritwin_gh/services/`

* Contains all business logic; each service imports and wraps existing MPC
  modules without copying any code.
* Services are **plain Python classes** (not FastAPI dependencies directly)
  so they can be unit-tested independently of FastAPI.
* Lazy imports inside methods prevent circular-import issues common in large
  FastAPI projects.

### 6.4 Core / Runtime Layer

`src/agritwin_gh/core/`

| File | Responsibility |
|------|---------------|
| `runtime_store.py` | Thread-safe in-memory state bridge between background DT loop and HTTP handlers |
| `lifespan.py` | FastAPI lifespan hooks — startup init, optional background loop start, shutdown cleanup |

---

## 7. API Endpoint Inventory

All paths are relative to `http://localhost:8000`.

| Method | Path | Schema | `api.js` function | Implementation source |
|--------|------|--------|------------------|-----------------------|
| GET | `/api/dt/state` | `DTStateResponse` | `getDtState` | `DTService.get_current_state()` |
| POST | `/api/dt/override` | `DtOverrideResponse` | `postDtOverride` | `DTService.apply_override()` |
| POST | `/api/dt/preset/{id}` | `DtPresetResponse` | `postDtPreset` | `DTService.apply_preset()` |
| GET | `/api/actuators/state` | `ActuatorStateResponse` | `getActuatorState` | `ActuatorService.get_state()` |
| POST | `/api/actuators/set` | `ActuatorSetResponse` | `postActuatorSet` | `ActuatorService.apply_overrides()` |
| GET | `/api/weather/current` | `WeatherResponse` | `getWeatherCurrent` | `WeatherService.get_current()` |
| GET | `/api/intelligence/disease` | `DiseaseRisksResponse` | `getDiseaseRisks` | `IntelligenceService.get_disease_risks()` |
| GET | `/api/intelligence/growth` | `GrowthIntelResponse` | `getGrowthIntel` | `IntelligenceService.get_growth_intel()` |
| GET | `/api/resources/monthly` | `ResourcesResponse` | `getMonthlyResources` | `ResourceService.get_monthly()` |
| GET | `/api/media/latest` | `LatestMediaResponse` | `getLatestMedia` | `MediaService.get_latest()` |
| GET | `/api/media/stage-images` | `StageImagesResponse` | `getStageImages` | `MediaService.get_stage_images()` |
| GET | `/api/media/disease-scans` | `DiseaseImagesResponse` | `getDiseaseImages` | `MediaService.get_disease_images()` |
| GET | `/api/system/health` | `SystemHealthResponse` | `getSystemHealth` | `RuntimeStore.as_system_health_response()` |

---

## 8. 3D Environment Integration Fields

The `SceneContext` block is included in **every** `GET /api/dt/state` response.
It is computed from `RuntimeStore` at zero extra cost (no DB query).  The
current React dashboard ignores it, but a future 3D environment layer can
consume it without any API change.

```python
class SceneContext(BaseModel):
    actuator_states_on_off: dict[str, bool]  # fan, vent, irrigation, heater, led, co2, fogger
    actuator_levels:        dict[str, float] # same keys, 0–100
    current_growth_stage:   str              # e.g. "flowering"
    next_growth_stage:      str | None       # e.g. "unripe"  (None at last stage)
    current_timestamp:      str              # ISO-8601
    time_of_day:            str              # "morning" | "afternoon" | "evening" | "night"
```

**Time-of-day buckets:**

| Bucket | Hours (UTC / local TZ) |
|--------|-----------------------|
| morning | 06:00 – 11:59 |
| afternoon | 12:00 – 17:59 |
| evening | 18:00 – 20:59 |
| night | 21:00 – 05:59 |

These fields also exist on `ActuatorEntry.on_off` and `ActuatorEntry.level`
so the 3D layer can alternatively call `GET /api/actuators/state`.

---

## 9. RuntimeStore — State Bridge

`RuntimeStore` is the key design element that prevents the HTTP handlers from
ever blocking on the MPC solver.

```
Background DT loop  (every 5 min)
    DTService.tick()
        → MPCSolver.solve()  [up to 200 ms]
        → DigitalTwinEngine.step()
        → store.update_from_step_result(result)   ← fast write, locked

HTTP handler  (any time)
    GET /api/dt/state
        → store.as_dt_state_response()   ← fast read, locked
        → return 200 JSON  [< 1 ms]
```

The store is **shared state within one OS process**.  For a multi-worker
deployment, use Redis or a persistent DB to share state across workers
(out of scope for this phase).

---

## 10. How to Add FastAPI + Uvicorn

FastAPI and Uvicorn are not yet in `pyproject.toml`.  Add them before
starting the implementation phase:

```toml
# pyproject.toml — add to [project].dependencies
"fastapi>=0.111.0",
"uvicorn[standard]>=0.29.0",
"pydantic>=2.7.0",
"python-multipart>=0.0.9",   # required by FastAPI for form data
```

Or with uv:

```powershell
cd e:\AgriTwin-GH
uv add fastapi "uvicorn[standard]" pydantic python-multipart
```

Verify the server starts with:

```powershell
$env:PYTHONPATH = "e:\AgriTwin-GH\src"
python main.py
# → Uvicorn running on http://0.0.0.0:8000
# → Browse http://localhost:8000/docs for interactive Swagger UI
```

---

## 11. Implementation Phases

### Phase 1 — Dependency wiring (no business logic)

**Goal:** Server starts, all 13 endpoints return valid (empty/zero-state)
JSON.  Frontend `api.js` stubs can be swapped and the UI renders without
errors.

Tasks:
1. `uv add fastapi "uvicorn[standard]" pydantic python-multipart`
2. Wire `DTService.setup()` to initialise `RuntimeStore` with a synthetic
   `DTLoop` in `lifespan.py`.
3. Implement `RuntimeStore.update_from_step_result()` to unpack
   `DTLoopStepResult` fields.
4. Uncomment `fetch()` calls in `api.js` and remove mock imports.

### Phase 2 — Live DT loop in background

**Goal:** `RuntimeStore` is updated every 5 minutes by a background
`DTLoop`.  `GET /api/dt/state` always returns the latest real state.

Tasks:
1. Implement `DTService.run_background_loop()` as an `asyncio` task.
2. Start it in `lifespan.py` via `asyncio.create_task()`.
3. Wire `ActuatorService.update_from_mpc_solution()` inside `DTService.tick()`.
4. Validate `GET /api/dt/state` updates between poll intervals in the dashboard.

### Phase 3 — Database-backed production mode

**Goal:** `DTService` uses `RealtimeLoop` + `MPCInputPreparation` to read
real sensor data from PostgreSQL.

Tasks:
1. Implement `RealtimeLoop`-backed `DTService.setup()` / `tick()`.
2. Wire `ImageStreamer` → `MediaService._presign()` → MinIO URLs.
3. Enable `AGRITWIN_BACKGROUND_LOOP=1` in production.

### Phase 4 — Intelligence model integration

**Goal:** Disease and growth classifiers return real predictions.

Tasks:
1. Load `disease_inference` and `growth_stage_inference` lazily in
   `IntelligenceService`.
2. Integrate with the image cache from `MediaService`.
3. Propagate classifier outputs to `RuntimeStore.disease_risk` /
   `growth_stage_index`.

### Phase 5 — 3D environment (future)

**Goal:** A 3D scene layer reads `SceneContext` from `GET /api/dt/state`
and animates actuators / growth-stage visuals in real time.

No API changes required — `SceneContext` is already in every response.

---

## 12. Design Decisions & Constraints

| Decision | Rationale |
|----------|-----------|
| FastAPI app factory (`create_app()`) | Makes testing easier — each test gets a fresh app |
| `RuntimeStore` as singleton | Avoids blocking HTTP handlers on MPC solve; survives multiple requests |
| Schemas separate from MPC dataclasses | API surface can evolve independently; avoids tight coupling |
| Lazy imports in service methods | Prevents circular imports in the large existing module graph |
| `SceneContext` in every DT state response | Zero-cost 3D readiness; no future schema change needed |
| No changes to `mpc/` files | Preserves all existing tests, smoke tests, and evaluation scripts |
| Background loop disabled by default | Allows running `python main.py` without a live DB on any laptop |
| Single worker process for now | `RuntimeStore` is in-process; multi-worker needs Redis (Phase 3+) |
| Tamil Nadu INR tariffs as fallback | Matches constants in `docs/MPC_COMPLETE_GUIDE.md` §18 |
