# AgriTwin-GH — FastAPI + Frontend Integration Runbook

> **Status:** Complete — all endpoints implemented and smoke-tested.  
> **Last updated:** April 2026  
> **Covers:** Backend startup, frontend startup, env variables, data flow,
> live vs override modes, 3D integration fields, and known limitations.

---

## Table of Contents

1. [Where Things Live](#1-where-things-live)  
2. [Environment Variables](#2-environment-variables)  
3. [Running the Backend](#3-running-the-backend)  
4. [Running the Frontend](#4-running-the-frontend)  
5. [Running Both Together](#5-running-both-together)  
6. [Data Flow](#6-data-flow)  
7. [Live Mode vs Override Mode](#7-live-mode-vs-override-mode)  
8. [Endpoint Reference](#8-endpoint-reference)  
9. [3D Integration Fields (Future)](#9-3d-integration-fields-future)  
10. [Testing & Verification](#10-testing--verification)  
11. [Known Limitations](#11-known-limitations)

---

## 1. Where Things Live

| Component | Path |
|-----------|------|
| FastAPI entry point | `main.py` (repo root) |
| App factory | `src/agritwin_gh/api/app.py` |
| Route handlers | `src/agritwin_gh/api/routes/*.py` |
| Service layer | `src/agritwin_gh/services/*.py` |
| In-memory state bridge | `src/agritwin_gh/core/runtime_store.py` |
| Lifespan startup/shutdown | `src/agritwin_gh/core/lifespan.py` |
| Pydantic response schemas | `src/agritwin_gh/schemas/*.py` |
| React frontend | `src/agritwin_gh/frontend/` |
| Frontend API layer | `src/agritwin_gh/frontend/src/services/api.js` |
| Frontend env file | `src/agritwin_gh/frontend/.env.local` |
| Backend env file | `.env` (copy from `.env.example`) |

---

## 2. Environment Variables

### Backend (`.env` in repo root)

Copy the template first:

```powershell
cp .env.example .env
```

Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `AGRITWIN_BACKGROUND_LOOP` | `0` | Set to `1` to start the DT simulation loop at startup. Requires a live DB. |
| `AGRITWIN_TOTAL_STEPS` | `288` | Number of simulated steps (288 × 5 min = 24 h). |
| `AGRITWIN_STEP_INTERVAL_SEC` | `300.0` | Wall-clock seconds between background DT steps. |
| `TESTING` | `0` | Set to `1` in CI/pytest to prevent the lifespan hook from starting the loop. |
| `DB_USER` / `DB_PASSWORD` / `DB_NAME` | — | PostgreSQL credentials (needed only when `BACKGROUND_LOOP=1`). |
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint (for media endpoints). |

### Frontend (`src/agritwin_gh/frontend/.env.local`)

Already created with:

```
VITE_API_BASE_URL=http://localhost:8000
```

Change this for staging/production deployments.  The value is consumed by
`api.js` as `import.meta.env.VITE_API_BASE_URL`.

---

## 3. Running the Backend

### Prerequisites

```powershell
# From repo root — activate the venv
.venv\Scripts\Activate.ps1

# Verify FastAPI + Uvicorn are installed
python -c "import fastapi, uvicorn; print('OK')"
```

### Start (development, auto-reload on code changes)

```powershell
$env:PYTHONPATH = "src"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Alternatively, run the entry point directly:

```powershell
$env:PYTHONPATH = "src"
python main.py
```

### Verify

Browse to:

- **Swagger UI:** `http://localhost:8000/docs`
- **Quick smoke check:** `http://localhost:8000/api/dt/state`

### With background DT loop enabled

```powershell
$env:AGRITWIN_BACKGROUND_LOOP = "1"
$env:AGRITWIN_TOTAL_STEPS     = "288"
uvicorn main:app --host 0.0.0.0 --port 8000
```

`RuntimeStore` will update every 5 minutes (simulated).  The dashboard charts
will reflect evolving state between browser refreshes.

> **Note:** `BACKGROUND_LOOP=1` requires PostgreSQL credentials in `.env`.
> Without a live DB it will log a warning and skip the loop automatically.

---

## 4. Running the Frontend

```powershell
cd src\agritwin_gh\frontend
npm install          # first time only
npm run dev          # Vite dev server on http://localhost:5173
```

For a production build:

```powershell
npm run build        # outputs to src/agritwin_gh/frontend/dist/
```

---

## 5. Running Both Together

Open two terminals:

**Terminal 1 — backend**

```powershell
.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 — frontend**

```powershell
cd src\agritwin_gh\frontend
npm run dev
```

Then open `http://localhost:5173` in a browser.

CORS is pre-configured to allow `localhost:5173` (Vite dev) and
`localhost:4173` (Vite preview) → `localhost:8000` (FastAPI).

---

## 6. Data Flow

```
React (port 5173)
   └─ api.js fetch()  ──GET /api/dt/state──────────────────────►┐
                       POST /api/dt/override/sim                 │
                       POST /api/actuators/set                   │
                       DELETE /api/dt/override                   │
                                                                 │
FastAPI (port 8000) ◄────────────────────────────────────────────┘
   ├─ /api/routes/*.py        (thin adapters, validate → call service)
   ├─ /services/*.py          (business logic, wraps MPC modules)
   └─ /core/runtime_store.py  ← single in-memory state bridge
        ↑
       Background DT loop (optional, every 5 min)
        DTService.tick()
          → MPCSolver.solve()
          → DigitalTwinEngine.step()
          → store.update_from_step_result(result)
```

**Request lifecycle for `GET /api/dt/state`:**

1. FastAPI invokes `DashboardService.get_dt_state()`
2. `DashboardService` calls `RuntimeStore.as_dt_state_response()`
3. `RuntimeStore` reads the latest `LatestState` snapshot under a `threading.RLock`
4. If an override is active (`LatestState.mode == "override"`), the response
   blends in the sim stage / day-in-stage values from `OverrideConfig`
5. Response is serialised to JSON and returned — total latency < 1 ms

---

## 7. Live Mode vs Override Mode

The `mode` field in `GET /api/dt/state` has two values:

### `"live"` — MPC-autonomous control

- The DT loop runs independently, driven by sensor data (or synthetic inputs).
- The `crop`, `sensors`, and `actuators` fields reflect the MPC's latest output.
- No operator overrides are stored.
- The React `ManualOverride` page shows the **Live / Current** pill selected.

### `"override"` — Operator control

- Triggered by `POST /api/dt/override/sim` (stage / day / date / hour) or
  `POST /api/actuators/set` (actuator levels).
- The `OverrideConfig` is stored in `RuntimeStore` alongside the base state.
- `GET /api/dt/state` blends the override values into `CropInfo` so the
  response immediately reflects what the operator set, without waiting for the
  next DT loop tick.
- Actuators set via `POST /api/actuators/set` are reflected immediately in
  `GET /api/actuators/state` — levels and `status = "MANUAL"`.
- The React `ManualOverride` page shows the **Manual Override** pill and
  displays an orange warning banner.
- Cleared by `DELETE /api/dt/override`, which resets both `mode → "live"` and
  `ActuatorSnapshot.mode → "auto"`.

**State machine:**

```
live ──POST /api/dt/override/sim──────── ► override
live ──POST /api/actuators/set────────── ► override
override ──DELETE /api/dt/override────── ► live
```

**Design constraint:** the base `LatestState` is **never mutated** by
an override.  `OverrideConfig` is a separate field that the response
projection layer reads to blend values.  This prevents override corruption
of the persisted simulation state.

---

## 8. Endpoint Reference

All paths are relative to `http://localhost:8000`.

| Method | Path | Schema | Description |
|--------|------|--------|-------------|
| GET | `/api/dt/state` | `DTStateResponse` | Full DT snapshot: crop, health, sensors, growth, mode, 3D fields |
| POST | `/api/dt/override` | `DtOverrideResponse` | Single parameter override (`param`, `value`) |
| POST | `/api/dt/override/sim` | `DtOverrideResponse` | Sim stage / day / date / hour override |
| DELETE | `/api/dt/override` | `DtOverrideResponse` | Clear all overrides → live mode |
| POST | `/api/dt/preset/{id}` | `DtPresetResponse` | Named preset: `day-cycle` / `night-cycle` / `emergency-flush` |
| GET | `/api/actuators/state` | `ActuatorStateResponse` | All 7 actuators with level, on/off, status |
| POST | `/api/actuators/set` | `ActuatorSetResponse` | Batch actuator level override |
| GET | `/api/weather/current` | `WeatherResponse` | Outdoor conditions + forecast list |
| GET | `/api/intelligence/disease` | `DiseaseRisksResponse` | Composite risk + per-pathogen entries |
| GET | `/api/intelligence/growth` | `GrowthIntelResponse` | Stage, transition probability, stage history |
| GET | `/api/resources/monthly` | `ResourcesResponse` | Energy (kWh) + water (L) usage and INR cost |
| GET | `/api/media/latest` | `LatestMediaResponse` | Latest stage + leaf camera images |
| GET | `/api/media/stage-images` | `StageImagesResponse` | Stage image gallery |
| GET | `/api/media/disease-scans` | `DiseaseImagesResponse` | Disease scan gallery |
| GET | `/api/system/health` | `SystemHealthResponse` | 6-row subsystem health check |

Interactive Swagger docs: `http://localhost:8000/docs`

---

## 9. 3D Integration Fields (Future)

Every `GET /api/dt/state` response already carries two 3D-ready blocks at
zero extra query cost.  The current React dashboard does not consume them.
A future Unity or WebGL layer can read them without any API change:

### Top-level compact fields

```json
{
  "time_of_day": "morning",
  "current_growth_stage": "flowering",
  "next_growth_stage": "unripe",
  "actuator_visual_state": {
    "on_off":  {"fan": true, "led": false, "co2": true, ...},
    "levels":  {"fan": 75.0, "led": 0.0,  "co2": 55.0, ...},
    "mode": ""
  }
}
```

### Full `scene_context` block

```json
{
  "scene_context": {
    "actuator_states_on_off": {"fan": true, ...},
    "actuator_levels":        {"fan": 75.0, ...},
    "current_growth_stage":   "flowering",
    "next_growth_stage":      "unripe",
    "current_timestamp":      "2026-04-04T09:15:00",
    "time_of_day":            "morning"
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

---

## 10. Testing & Verification

### Automated smoke tests (no running server required)

```powershell
# From repo root with venv active
$env:PYTHONPATH = "src"
pytest tests/test_api_smoke.py -v
```

The tests use FastAPI's in-process `TestClient`, so no Uvicorn process is
needed.  `TESTING=1` is set automatically so the lifespan hook skips the DB
loop.

**Coverage:**

| Endpoint | Tests |
|----------|-------|
| `GET /api/dt/state` | HTTP 200, all fields, `mode` field, 3D blocks |
| `GET /api/actuators/state` | HTTP 200, 7 actuators, level bounds, status field |
| `GET /api/weather/current` | HTTP 200, current + forecast shape |
| `GET /api/intelligence/disease` | HTTP 200, composite_risk range, pathogens list |
| `GET /api/intelligence/growth` | HTTP 200, stage fields, stage_history |
| `GET /api/system/health` | HTTP 200, rows list, bool ok field |
| `POST /api/dt/override` | HTTP 200, ok=True, echo fields |
| `POST /api/dt/override/sim` | HTTP 200, mode → override, stage reflected in state |
| `DELETE /api/dt/override` | HTTP 200, mode → live after clear |
| `POST /api/actuators/set` | HTTP 200, unknown ID skipped, level clamped, MANUAL status |
| `POST /api/dt/preset/{id}` | all three presets, invalid preset → 4xx |

### Live endpoint sampler (requires running server)

```powershell
python main.py                           # in one terminal
python scripts/sample_api_responses.py  # in another
```

Hits all key endpoints, pretty-prints JSON excerpts, and exits 1 if any fail.
Override the URL:

```powershell
$env:AGRITWIN_API_URL = "http://localhost:8000"
python scripts/sample_api_responses.py
```

### Manual curl / browser checks

```powershell
# Basic state check
Invoke-RestMethod http://localhost:8000/api/dt/state | ConvertTo-Json -Depth 4

# Sim override
$body = '{"stage":"flowering","day_in_stage":7,"start_date":"2026-04-04","start_hour":9}'
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/dt/override/sim `
    -ContentType "application/json" -Body $body

# Confirm mode is now override
(Invoke-RestMethod http://localhost:8000/api/dt/state).mode

# Reset
Invoke-RestMethod -Method Delete -Uri http://localhost:8000/api/dt/override
```

---

## 11. Known Limitations

| Area | Limitation | Planned fix |
|------|-----------|-------------|
| **Multi-worker** | `RuntimeStore` is in-process; only one Uvicorn worker is supported. Multiple workers would have divergent state. | Phase 3: replace with Redis-backed shared state. |
| **Background loop without DB** | If `AGRITWIN_BACKGROUND_LOOP=1` but no PostgreSQL is reachable, the loop is silently skipped. State in `RuntimeStore` remains at defaults (zero-state). | Connect to PostgreSQL as per `docs/POSTGRESQL_QUICKSTART.md`. |
| **Media endpoints** | `GET /api/media/latest` and the gallery endpoints return placeholder data until MinIO is configured and images are uploaded. | Configure MinIO per `docs/IMAGE_STORAGE_SETUP.md`. |
| **Disease / growth ML models** | The `IntelligenceService` returns zero-state disease risks and stage history until the Keras models are loaded. Model paths must be set in `config/settings.yaml`. | Train classifiers with `notebooks/tomato_disease_classifier_train.ipynb`. |
| **Weather forecast** | `GET /api/weather/current` returns a synthetic diurnal forecast until `WeatherService` is wired to a live weather API. | Set `WEATHER_API_KEY` in `.env`. |
| **Override persistence** | `OverrideConfig` lives in `RuntimeStore` (RAM only). A server restart clears all overrides. | Accepted by design for Phase 1; DB persistence is a Phase 3+ concern. |
| **Frontend actuator level sliders** | The `ManualOverride` page currently exposes on/off toggles only (no level sliders). Level defaults to 100 when active. | UI iteration planned. |
| **CORS production** | `_CORS_ORIGINS` in `api/app.py` is hardcoded to `localhost`. | Add production domain before deployment. |
