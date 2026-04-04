/**
 * AgriTwin-GH — FastAPI Integration Stub Layer
 *
 * Swap `import.meta.env.VITE_API_BASE_URL` in .env.local to point at
 * the running FastAPI server. Until then, all functions resolve to mock data.
 *
 * API surface mirrors:
 *   main.py (AgriTwin-GH FastAPI app)
 *   src/agritwin_gh/digital_twin/dt_core.py  — DT state / loop
 *   src/agritwin_gh/control/mpc_controller.py — actuator overrides
 *   src/agritwin_gh/models/               — disease / growth predictions
 *
 * ── Integration checklist ────────────────────────────────────────────────────
 *   1. Set VITE_API_BASE_URL in .env.local
 *   2. Remove `import * as mock from …` at the top
 *   3. Uncomment each `const res = await fetch(…)` block
 *   4. Remove the `return mock.*` fallback line in that function
 *   5. Add error handling (401 → refresh token, 503 → show toast)
 * ────────────────────────────────────────────────────────────────────────────
 */

import * as mock from '../data/mockData.js';

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ── helpers ──────────────────────────────────────────────────────────────────

/**
 * GET wrapper with JSON parsing and error surfacing.
 * @param {string} path  e.g. '/api/dt/state'
 */
async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { 'Content-Type': 'application/json' },
  });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status} ${res.statusText}`);
  return res.json();
}

/**
 * POST wrapper. Body is serialised to JSON.
 * @param {string} path
 * @param {object} body
 */
async function post(path, body = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status} ${res.statusText}`);
  return res.json();
}

// ── Digital-Twin state ────────────────────────────────────────────────────────

/**
 * GET /api/dt/state
 * Returns the current digital-twin snapshot: crop, health, sensors, etc.
 */
export async function getDtState() {
  // TODO: const data = await get('/api/dt/state');  return data;
  return Promise.resolve({
    crop:    mock.CROP,
    health:  mock.CROP_HEALTH,
    sensors: mock.SENSOR_METRICS,
    growth:  mock.GROWTH_INTEL,
  });
}

/**
 * POST /api/dt/override
 * Inject a manual parameter override into the DT loop.
 * @param {{ param: string, value: number }} params
 */
export async function postDtOverride(params) {
  console.info('[api stub] POST /api/dt/override', params);
  // TODO: return post('/api/dt/override', params);
  return Promise.resolve({ ok: true });
}

/**
 * POST /api/dt/preset/:id
 * Apply a named quick-action preset (day-cycle, night-cycle, emergency-flush).
 * @param {string} presetId
 */
export async function postDtPreset(presetId) {
  console.info('[api stub] POST /api/dt/preset/' + presetId);
  // TODO: return post(`/api/dt/preset/${presetId}`);
  return Promise.resolve({ ok: true, preset: presetId });
}

// ── Actuators ─────────────────────────────────────────────────────────────────

/**
 * GET /api/actuators/state
 * Returns current actuator states and levels.
 */
export async function getActuatorState() {
  // TODO: const data = await get('/api/actuators/state');  return data;
  return Promise.resolve(mock.ACTUATOR_STATE);
}

/**
 * POST /api/actuators/set
 * Apply batch actuator level overrides.
 * @param {{ id: string, level: number }[]} actuators
 */
export async function postActuatorSet(actuators) {
  console.info('[api stub] POST /api/actuators/set', actuators);
  // TODO: return post('/api/actuators/set', { actuators });
  return Promise.resolve({ ok: true });
}

// ── Weather ───────────────────────────────────────────────────────────────────

/**
 * GET /api/weather/current
 * Returns outdoor current conditions + forecast + alert status.
 */
export async function getWeatherCurrent() {
  // TODO: const data = await get('/api/weather/current');  return data;
  return Promise.resolve({
    status:   mock.WEATHER_STATUS,
    current:  mock.OUTDOOR_CURRENT,
    forecast: mock.OUTDOOR_FORECAST,
  });
}

// ── Intelligence ──────────────────────────────────────────────────────────────

/**
 * GET /api/intelligence/disease
 * Returns disease risk scores and pathogen probabilities.
 */
export async function getDiseaseRisks() {
  // TODO: const data = await get('/api/intelligence/disease');  return data;
  return Promise.resolve(mock.DISEASE_RISKS);
}

/**
 * GET /api/intelligence/growth
 * Returns growth stage progress and transition predictions.
 */
export async function getGrowthIntel() {
  // TODO: const data = await get('/api/intelligence/growth');  return data;
  return Promise.resolve(mock.GROWTH_INTEL);
}

// ── Resources ─────────────────────────────────────────────────────────────────

/**
 * GET /api/resources/monthly
 * Returns monthly resource consumption and cost summary.
 */
export async function getMonthlyResources() {
  // TODO: const data = await get('/api/resources/monthly');  return data;
  return Promise.resolve({
    resources: mock.MONTHLY_RESOURCES,
    cost:      mock.MONTHLY_COST,
  });
}

// ── Media ─────────────────────────────────────────────────────────────────────

/**
 * GET /api/media/latest
 * Returns latest stage and leaf camera image URLs.
 */
export async function getLatestMedia() {
  // TODO: const data = await get('/api/media/latest');  return data;
  return Promise.resolve(mock.CROP_IMAGES);
}

/**
 * GET /api/media/stage-images
 * Returns stage-by-stage image gallery entries.
 */
export async function getStageImages() {
  // TODO: const data = await get('/api/media/stage-images');  return data;
  return Promise.resolve(mock.GROWTH_STAGE_IMAGES);
}

/**
 * GET /api/media/disease-scans
 * Returns disease scan image gallery entries.
 */
export async function getDiseaseImages() {
  // TODO: const data = await get('/api/media/disease-scans');  return data;
  return Promise.resolve(mock.DISEASE_IMAGES);
}

// ── System health ─────────────────────────────────────────────────────────────

/**
 * GET /api/system/health
 * Returns sensor array and pipeline health status rows.
 */
export async function getSystemHealth() {
  // TODO: const data = await get('/api/system/health');  return data;
  return Promise.resolve(mock.HEALTH_ROWS);
}
