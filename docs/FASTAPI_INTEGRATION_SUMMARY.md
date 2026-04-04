# FastAPI Integration Summary

This document records everything that was created or modified during the FastAPI
backend integration.  It is the canonical reference for:

- what files exist and why,
- which existing modules are reused rather than replaced,
- what the full endpoint surface looks like,
- how runtime state flows through the system,
- how manual override works, and
- what 3D-ready data is already exposed in responses.

---

## 1. What Was Added

### 1.1 Schema layer — `src/agritwin_gh/schemas/`

| File | Purpose |
|------|---------|
| `__init__.py` | Re-exports every public schema; import from here to avoid deep coupling. |
| `enums.py` | `SystemStatus`, `AlertSeverity`, `GrowthStage`, `ActuatorMode` — zero external imports. |
| `dt_schemas.py` | `DTStateResponse`, `SceneContext`, `ActuatorVisualState`, `ThreeDPayload`, override request/response schemas. |
| `actuator_schemas.py` | `ActuatorStateResponse`, `ActuatorSetRequest`, `ActuatorSetResponse`, `ActuatorSnapshot`. |
| `intelligence_schemas.py` | `DiseaseRisksResponse`, `DiseaseRisk`, `GrowthIntelResponse`. |
| `weather_schemas.py` | `WeatherResponse`, `WeatherForecastPoint`. |
| `resource_schemas.py` | `ResourcesResponse`, `ResourceMetric`. |
| `media_schemas.py` | `LatestMediaResponse`, `StageImagesResponse`, `DiseaseImagesResponse`, `ImageMeta`. |
| `system_schemas.py` | `SystemHealthResponse`, `HealthRow`. |

### 1.2 API layer — `src/agritwin_gh/api/`

| File | Purpose |
|------|---------|
| `app.py` | `create_app()` factory — builds the ASGI app, registers CORS, mounts all 7 routers under `/api`. |
| `dependencies.py` | FastAPI dependency providers: `get_runtime_store`, `get_dashboard_service`, `get_control_service`, `get_media_service`, `get_system_service`, `get_db_session`. |
| `routes/dt.py` | `GET /api/dt/state`, `POST /api/dt/override`, `POST /api/dt/override/sim`, `DELETE /api/dt/override`, `POST /api/dt/preset/{preset_id}`. |
| `routes/actuators.py` | `GET /api/actuators/state`, `POST /api/actuators/set`. |
| `routes/weather.py` | `GET /api/weather/current`. |
| `routes/intelligence.py` | `GET /api/intelligence/disease`, `GET /api/intelligence/growth`. |
| `routes/resources.py` | `GET /api/resources/monthly`. |
| `routes/media.py` | `GET /api/media/latest`, `GET /api/media/stage-images`, `GET /api/media/disease-scans`. |
| `routes/system.py` | `GET /api/system/health`. |

### 1.3 Service layer — `src/agritwin_gh/services/`

| File | Active? | Purpose |
|------|---------|---------|
| `dashboard_service.py` | ✅ Active | Read facade over `RuntimeStore`; one call per endpoint category. |
| `control_service.py` | ✅ Active | Write facade: applies overrides, presets, and individual actuator levels to `RuntimeStore`. |
| `loop_service.py` | ✅ Active | Async background loop that ticks `GreenhouseDTLoop` and commits `LatestState`. |
| `system_service.py` | ✅ Active | Derives `SystemHealthResponse` from staleness thresholds in `RuntimeStore`. |
| `media_service.py` | ✅ Active | Queries PostgreSQL / MinIO for image metadata; gracefully degrades to empty lists. |
| `actuator_service.py` | ✅ Active | Thin projection service; used by `services/__init__.py` public surface. |
| `weather_service.py` | ✅ Active | Thin projection service; used by `services/__init__.py` public surface. |
| `intelligence_service.py` | ✅ Active | Thin projection service; used by `services/__init__.py` public surface. |
| `resource_service.py` | ✅ Active | Thin projection service; used by `services/__init__.py` public surface. |
| `dt_service.py` | 🔲 Placeholder | Skeleton for Phase 2 (live DB loop).  Not wired to any route yet. |

### 1.4 Core layer — `src/agritwin_gh/core/`

| File | Purpose |
|------|---------|
| `runtime_store.py` | Thread-safe singleton (~1400 lines).  Owns `LatestState`, `OverrideConfig`, and all `as_*_response()` projection methods.  The single source of truth for every `GET` endpoint. |
| `lifespan.py` | FastAPI lifespan context manager — starts `LoopService` on startup, shuts it down on teardown.  Owns the `RuntimeStore` singleton.  Has `TESTING` guard to skip the loop during unit tests. |

### 1.5 Tests and scripts

| File | Purpose |
|------|---------|
| `tests/test_api_smoke.py` | pytest smoke suite — one class per endpoint group, all 15 paths covered, mock-free (runs against a `TestClient`). |
| `scripts/sample_api_responses.py` | stdlib-only sampler that prints pretty-printed JSON for every endpoint; useful for ad-hoc manual testing without pytest. |

### 1.6 Documentation

| File | Purpose |
|------|---------|
| `docs/FASTAPI_FRONTEND_INTEGRATION_RUNBOOK.md` | Step-by-step runbook: environment setup, starting backend + Vite dev server, all 15 `curl` one-liners, troubleshooting guide. |
| `docs/FASTAPI_INTEGRATION_SUMMARY.md` | This file. |

---

## 2. What Was Modified

| File | Change |
|------|--------|
| `main.py` | Updated entry point to call `create_app()` from `agritwin_gh.api.app` via `uvicorn`. |
| `src/agritwin_gh/frontend/src/services/api.js` | Fully rewritten — 14 real-fetch functions replacing the original mock stubs, plus `deleteDtOverride()`. |
| `src/agritwin_gh/frontend/src/pages/HomeDashboard.jsx` | Wired to live API (`fetchDtState`, `fetchWeather`, `fetchResources`, `fetchSystemHealth`). |
| `src/agritwin_gh/frontend/src/pages/DetailedInsights.jsx` | Wired to live API (`fetchDiseaseRisks`, `fetchGrowthIntel`, `fetchActuatorState`, `fetchStageImages`, `fetchDiseaseScans`). |
| `src/agritwin_gh/frontend/src/pages/ManualOverride.jsx` | Wired to live API (`postDtOverride`, `deleteDtOverride`); includes field-level error state; reads `response.mode` to reflect active/cleared override state. |

---

## 3. Existing Modules Reused (Not Replaced)

The FastAPI layer is a thin read/write facade.  No MPC or DT logic was cloned.

| Existing module | Consumed by |
|----------------|-------------|
| `mpc.dt_loop.GreenhouseDTLoop` | `LoopService` — ticks the simulation on each background iteration. |
| `mpc.mpc_runner.MPCRunner` | `ControlService` — executes one MPC solve when applying actuator presets. |
| `models.disease.TomatoDiseaseClassifier` | `IntelligenceService` (via `RuntimeStore.as_disease_risks_response()`). |
| `models.growth.TomatoGrowthStageClassifier` | `IntelligenceService` (via `RuntimeStore.as_growth_intel_response()`). |
| `utils.database.DatabaseManager` | `get_db_session()` dependency and `MediaService`. |
| `utils.minio_client.MinIOClient` | `MediaService`. |
| `config.settings` | `lifespan.py` — reads loop interval, DB URL, MinIO credentials. |

---

## 4. API Endpoints

All routes are prefixed with `/api`.

| Method | Path | Response schema | Backend source |
|--------|------|-----------------|----------------|
| `GET` | `/api/dt/state` | `DTStateResponse` | `RuntimeStore.as_dt_state_response()` |
| `POST` | `/api/dt/override` | `DtOverrideResponse` | `ControlService.set_override()` |
| `POST` | `/api/dt/override/sim` | `DtOverrideResponse` | `ControlService.simulate_override()` |
| `DELETE` | `/api/dt/override` | `DtOverrideResponse` | `ControlService.clear_override()` |
| `POST` | `/api/dt/preset/{preset_id}` | `DtPresetResponse` | `ControlService.apply_preset()` |
| `GET` | `/api/actuators/state` | `ActuatorStateResponse` | `RuntimeStore.as_actuator_state_response()` |
| `POST` | `/api/actuators/set` | `ActuatorSetResponse` | `ControlService.set_actuator()` |
| `GET` | `/api/weather/current` | `WeatherResponse` | `RuntimeStore.as_weather_response()` |
| `GET` | `/api/intelligence/disease` | `DiseaseRisksResponse` | `RuntimeStore.as_disease_risks_response()` |
| `GET` | `/api/intelligence/growth` | `GrowthIntelResponse` | `RuntimeStore.as_growth_intel_response()` |
| `GET` | `/api/resources/monthly` | `ResourcesResponse` | `RuntimeStore.as_resources_response()` |
| `GET` | `/api/media/latest` | `LatestMediaResponse` | `MediaService.get_latest()` |
| `GET` | `/api/media/stage-images` | `StageImagesResponse` | `MediaService.get_stage_images()` |
| `GET` | `/api/media/disease-scans` | `DiseaseImagesResponse` | `MediaService.get_disease_scans()` |
| `GET` | `/api/system/health` | `SystemHealthResponse` | `SystemService.get_health()` |

---

## 5. Runtime State Management

### Data flow

```
GreenhouseDTLoop.step()
    │
    ▼  (every N seconds, configured via settings.loop_interval_s)
LoopService._tick()
    │
    ├─► RuntimeStore.commit(state: GreenhouseState)
    │       atomically replaces LatestState
    │       computes all derived scalars (alerts, risks, resources …)
    │
    └─► RuntimeStore ready for reads
            │
            ▼
     FastAPI GET request
            │
            ▼
     as_*_response()        ← pure projection, no DB call
            │
            ▼
     Pydantic response → JSON → frontend
```

### Thread safety

`RuntimeStore` uses a `threading.Lock` around `commit()` and around every `as_*_response()` call that reads the shared `LatestState`.  Concurrent HTTP requests cannot observe a partial write.

### Override isolation

`OverrideConfig` is stored separately from `LatestState`.  Applying an override never mutates the last simulated greenhouse state — it only sets a flag and target levels.  Reading `as_dt_state_response()` merges the two: if `OverrideConfig.active` is `True`, the returned `mode` field is `"override"` and actuator levels reflect the override targets rather than the MPC solution.

---

## 6. Manual Override Mechanism

| Action | API call | Runtime effect |
|--------|----------|----------------|
| Apply operator override | `POST /api/dt/override` body: `OverrideRequest` | `RuntimeStore.set_override(config)` → `mode = "override"`, actuator targets set immediately |
| Apply a named preset | `POST /api/dt/preset/{preset_id}` | `ControlService.apply_preset()` → resolves preset levels → `set_override()` |
| Simulate (no commit) | `POST /api/dt/override/sim` | `ControlService.simulate_override()` → returns projected state, does **not** call `set_override()` |
| Clear override | `DELETE /api/dt/override` | `RuntimeStore.clear_override()` → `mode = "live"`, override targets discarded |

The frontend reads `response.mode` from these responses to update the UI state badge without an extra `GET /api/dt/state` round-trip.

Named presets available out-of-the-box (defined in `ControlService.PRESETS`):

| `preset_id` | Description |
|-------------|-------------|
| `high-growth` | Optimal temperature and CO₂ for vegetative growth |
| `water-save` | Reduced irrigation targets |
| `night-mode` | Low light, reduced HVAC |
| `disease-alert` | Low humidity, high airflow to slow spread |
| `harvest-ready` | Pre-harvest ripening conditions |

---

## 7. 3D-Ready Data Already Exposed

The `GET /api/dt/state` response (`DTStateResponse`) already carries all the
geometry and animation metadata a future 3D scene will need.  No API change
is required to add 3D rendering.

### Schemas in `DTStateResponse`

| Field / Schema | 3D usage |
|----------------|----------|
| `scene_context: SceneContext` | Viewport environment: `time_of_day` bucket (`"morning"` / `"afternoon"` / `"evening"` / `"night"`), `weather_condition`, `alert_severity`. Drives sky-dome, ambient light intensity, and fog density. |
| `actuators: list[ActuatorVisualState]` | Per-actuator visual state: `level` (0–100), `mode` (`"auto"` / `"manual"`), `is_active`. Drives actuator mesh animation and indicator colours. |
| `three_d: ThreeDPayload` | Convenience bundle: `plant_health_score`, `disease_risk_score`, `growth_stage_label`, `growth_stage_index`, `alert_count`. Drives plant-mesh health shader and growth-stage pose. |
| `time_of_day: str` | Top-level convenience alias for `scene_context.time_of_day`. |
| `current_growth_stage: str` | Current growth stage label (e.g. `"flowering"`). |
| `next_growth_stage: str \| null` | Next stage label; `null` at `"harvesting"`. Used for stage-transition animations. |

The `ThreeDPayload.get_3d_payload()` helper (on `RuntimeStore`) returns this
bundle without requiring callers to traverse the full response tree.

---

## 8. Architectural Constraints Respected

The audit confirmed that all of the following constraints hold across every
created file:

| Constraint | Status |
|-----------|--------|
| No MPC / DT logic was copied or reimplemented | ✅ All services import from `mpc/` and `models/` |
| `DTService` placeholder is clearly labeled as Phase 2 | ✅ Not wired to any route |
| `LoopService` is the active loop driver; `DTService` is not | ✅ Confirmed |
| No `IntelligenceService`, `ActuatorService`, `WeatherService`, `ResourceService` injected directly into routes | ✅ Routes inject `DashboardService`; thin services exist for `services/__init__.py` surface only |
| 3D data exposed but 3D rendering not implemented | ✅ `SceneContext` / `ThreeDPayload` populated; no WebGL/WebSocket code added |
| Frontend remains focused on current Dashboard / Insights / Override UI | ✅ No new pages or 3D canvas added |
| Backend is the single source of truth | ✅ All reads go through `RuntimeStore`; no client-side state computation |
| `time_of_day` bucket computed in two places by design | ✅ `RuntimeStore._compute_time_of_day()` (private, for state commits) and `DashboardService.time_of_day_from_ts()` (public, for one-off queries) — different audiences |

---

## 9. Future Work (Not Implemented)

| Phase | What it adds |
|-------|-------------|
| **Phase 2 — Live DB loop** | Wire `DTService` to a persistent database query; replace the background-tick synthetic state with real sensor reads. `dt_service.py` is the stub. |
| **Phase 3 — RealtimeLoop + MinIO** | Replace `LoopService` with a `RealtimeLoop` that reads from the live sensor stream and writes images to MinIO for `MediaService` to serve. |
| **Phase 4 — ML classifiers on live data** | Run `TomatoDiseaseClassifier` and `TomatoGrowthStageClassifier` on every loop tick and commit predictions to `RuntimeStore`. Currently the classifiers are invoked with synthetic feature vectors. |
| **Phase 5 — 3D WebSocket** | Add a `WS /api/ws/state` WebSocket endpoint that pushes `DTStateResponse` diffs at ~1 Hz.  The 3D scene subscribes directly.  All data fields are already present in the HTTP response. |

---

## 10. Quick-Start Reference

### Start the backend

```bash
# from project root
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend dev server

```bash
cd src/agritwin_gh/frontend
npm run dev          # Vite proxies /api → http://localhost:8000
```

### Run the smoke tests

```bash
pytest tests/test_api_smoke.py -v
```

### Sample all endpoints (no pytest)

```bash
python scripts/sample_api_responses.py
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGRITWIN_ENV` | `development` | Selects `settings.yaml` overlay |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend fetch base |
| `DATABASE_URL` | — | PostgreSQL DSN for `MediaService` |
| `MINIO_ENDPOINT` | — | MinIO host for image retrieval |
| `DT_LOOP_INTERVAL_S` | `5` | Seconds between `LoopService` ticks |

See `docs/FASTAPI_FRONTEND_INTEGRATION_RUNBOOK.md` for full environment setup
and `curl` one-liners for every endpoint.
