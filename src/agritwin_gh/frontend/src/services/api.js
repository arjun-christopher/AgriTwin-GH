/**
 * AgriTwin-GH — FastAPI Integration Layer
 *
 * All functions call the real FastAPI backend at VITE_API_BASE_URL.
 * Field translations (snake_case → camelCase) happen here so pages
 * receive normalised data matching mock shapes.
 *
 * API surface:
 *   GET  /api/dt/state
 *   POST /api/dt/override
 *   POST /api/dt/override/sim
 *   POST /api/dt/preset/:id
 *   GET  /api/actuators/state
 *   POST /api/actuators/set
 *   GET  /api/weather/current
 *   GET  /api/intelligence/disease
 *   GET  /api/intelligence/growth
 *   GET  /api/resources/monthly
 *   GET  /api/media/latest
 *   GET  /api/media/stage-images
 *   GET  /api/media/disease-scans
 *   GET  /api/system/health
 */

const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000';

// ── helpers ──────────────────────────────────────────────────────────────────

async function get(path) {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`GET ${path} → ${res.status} ${res.statusText}`);
  return res.json();
}

async function post(path, body = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`POST ${path} → ${res.status} ${res.statusText}`);
  return res.json();
}

async function del(path) {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: 'DELETE',
    headers: { Accept: 'application/json' },
  });
  if (!res.ok) throw new Error(`DELETE ${path} → ${res.status} ${res.statusText}`);
  return res.json();
}

// ── Digital-Twin state ────────────────────────────────────────────────────────

/** GET /api/dt/state — full DT snapshot */
export async function getDtState() {
  try {
    const d = await get('/api/dt/state');
    return {
      crop: {
        current:       d.crop.current,
        currentIndex:  d.crop.current_index,
        currentPct:    d.crop.progress_pct,
        daysInStage:   d.crop.days_in_stage,
        stageDuration: d.crop.stage_duration_days,
        next:          d.crop.next_stage,
        nextInDays:    d.crop.next_in_days,
        stages:        d.crop.stages,
      },
      health: {
        status:     d.health.status,
        confidence: d.health.confidence,
        detail:     d.health.detail,
        scannedAgo: d.health.scanned_ago,
      },
      sensors: d.sensors.map(s => ({
        key:        s.key,
        iconKey:    s.icon_key,
        label:      s.label,
        value:      s.value,
        unit:       s.unit,
        trend:      s.trend,
        trendLabel: s.trend_label,
        trendClass: s.trend_color,
        rangeMin:   s.range_min,
        rangeMax:   s.range_max,
        optimal:    s.optimal_range,
        status:     s.status,
        delta:      s.delta ?? 0,
      })),
      growth: {
        hoursToNextStage:  d.growth.hours_to_next_stage,
        transitionProb24h: d.growth.transition_prob_24h,
        stageHistory: (d.growth.stage_history ?? []).map(h => ({
          stage:      h.stage,
          daysUsed:   h.days_used,
          daysTarget: h.days_target,
          complete:   h.complete,
        })),
      },
      // Operational mode: 'live' | 'override'
      mode: d.mode ?? 'live',
    };
  } catch (e) {
    console.error('[api] getDtState:', e);
    throw e;
  }
}

/** POST /api/dt/override — inject a parameter override */
export async function postDtOverride(params) {
  try {
    return await post('/api/dt/override', params);
  } catch (e) {
    console.error('[api] postDtOverride:', e);
    throw e;
  }
}

/** POST /api/dt/override/sim — set simulation stage / date params */
export async function postDtSimOverride(body) {
  try {
    return await post('/api/dt/override/sim', body);
  } catch (e) {
    console.error('[api] postDtSimOverride:', e);
    throw e;
  }
}

/** POST /api/dt/preset/:id — apply a named quick-action preset */
export async function postDtPreset(presetId) {
  try {
    return await post(`/api/dt/preset/${presetId}`);
  } catch (e) {
    console.error('[api] postDtPreset:', e);
    throw e;
  }
}

/** DELETE /api/dt/override — clear override and return to live mode */
export async function deleteDtOverride() {
  try {
    return await del('/api/dt/override');
  } catch (e) {
    console.error('[api] deleteDtOverride:', e);
    throw e;
  }
}

// ── Actuators ─────────────────────────────────────────────────────────────────

/** GET /api/actuators/state — current actuator states and levels */
export async function getActuatorState() {
  try {
    const d = await get('/api/actuators/state');
    return (d.actuators ?? []).map(a => ({
      id:      a.id,
      iconKey: a.icon_key,
      label:   a.label,
      status:  a.status,
      active:  a.active,
      level:   a.level,
      value:   a.value_display,
      color:   a.color,
    }));
  } catch (e) {
    console.error('[api] getActuatorState:', e);
    throw e;
  }
}

/** POST /api/actuators/set — batch actuator level overrides */
export async function postActuatorSet(actuators) {
  try {
    return await post('/api/actuators/set', { actuators });
  } catch (e) {
    console.error('[api] postActuatorSet:', e);
    throw e;
  }
}

// ── Weather ───────────────────────────────────────────────────────────────────

/** GET /api/weather/current — outdoor conditions + forecast */
export async function getWeather() {
  try {
    const d = await get('/api/weather/current');
    return {
      status: {
        status:    d.status.status,
        condition: d.status.condition,
        detail:    d.status.detail,
        forecast:  d.status.forecast_note,
      },
      current: {
        temp:       d.current.temp,
        humidity:   d.current.humidity,
        windSpeed:  d.current.wind_speed,
        windDir:    d.current.wind_dir,
        pressure:   d.current.pressure,
        uvIndex:    d.current.uv_index,
        dewPoint:   d.current.dew_point,
        solarRad:   d.current.solar_rad,
        condition:  d.current.condition,
        visibility: d.current.visibility,
      },
      forecast: (d.forecast ?? []).map(f => ({
        time:      f.time,
        high:      f.high,
        low:       f.low,
        humidity:  f.humidity,
        condition: f.condition,
        iconKey:   f.icon_key,
      })),
      // 24-hr-ahead model forecast: {temp_external, humidity_external, solar_radiation, windspeed, conditions}
      forecast24h: d.forecast_24h ?? {},
    };
  } catch (e) {
    console.error('[api] getWeather:', e);
    throw e;
  }
}

// ── Intelligence ──────────────────────────────────────────────────────────────

/** GET /api/intelligence/disease — disease risk scores */
export async function getDiseaseRisks() {
  try {
    const d = await get('/api/intelligence/disease');
    return (d.pathogens ?? []).map(p => ({
      name:     p.name,
      pathogen: p.pathogen,
      risk24h:  p.risk_24h,
      severity: p.severity,
    }));
  } catch (e) {
    console.error('[api] getDiseaseRisks:', e);
    throw e;
  }
}

/** GET /api/intelligence/growth — growth stage progress and transition predictions */
export async function getGrowthIntel() {
  try {
    const d = await get('/api/intelligence/growth');
    return {
      hoursToNextStage:  d.hours_to_next_stage,
      transitionProb24h: d.transition_prob_24h,
      stageHistory: (d.stage_history ?? []).map(h => ({
        stage:      h.stage,
        daysUsed:   h.days_used,
        daysTarget: h.days_target,
        complete:   h.complete,
      })),
    };
  } catch (e) {
    console.error('[api] getGrowthIntel:', e);
    throw e;
  }
}

// ── Resources ─────────────────────────────────────────────────────────────────

/** GET /api/resources/monthly — monthly resource consumption and cost */
export async function getMonthlyResources() {
  try {
    const d = await get('/api/resources/monthly');
    return {
      resources: (d.resources ?? []).map(r => ({
        label: r.label,
        used:  r.used,
        unit:  r.unit,
      })),
      cost: {
        month:  d.cost.month,
        energy: d.cost.energy_inr,
        water:  d.cost.water_inr,
        total:  d.cost.total_inr,
      },
      actuators: (d.actuators ?? []).map(a => ({
        key:       a.key,
        label:     a.label,
        energyKwh: a.energy_kwh,
        waterL:    a.water_l,
        costInr:   a.cost_inr,
      })),
    };
  } catch (e) {
    console.error('[api] getMonthlyResources:', e);
    throw e;
  }
}

// ── Media ─────────────────────────────────────────────────────────────────────

/** GET /api/media/latest — latest stage and leaf camera images */
export async function getLatestMedia() {
  try {
    const d = await get('/api/media/latest');
    return {
      stage: { src: d.stage.src, alt: d.stage.alt, badge: d.stage.badge, location: d.stage.location, captured: d.stage.captured },
      leaf:  { src: d.leaf.src,  alt: d.leaf.alt,  badge: d.leaf.badge,  location: d.leaf.location,  captured: d.leaf.captured  },
    };
  } catch (e) {
    console.error('[api] getLatestMedia:', e);
    throw e;
  }
}

/** GET /api/media/stage-images — stage image gallery */
export async function getStageImages() {
  try {
    const d = await get('/api/media/stage-images');
    return (d.images ?? []).map(i => ({
      src:         i.src,
      alt:         i.alt,
      capturedAgo: i.captured_ago,
      stage:       i.stage,
      confidence:  i.confidence,
    }));
  } catch (e) {
    console.error('[api] getStageImages:', e);
    throw e;
  }
}

/** GET /api/media/disease-scans — disease scan image gallery */
export async function getDiseaseImages() {
  try {
    const d = await get('/api/media/disease-scans');
    return (d.images ?? []).map(i => ({
      src:            i.src,
      alt:            i.alt,
      capturedAgo:    i.captured_ago,
      classification: i.classification,
      risk:           i.risk,
      confidence:     i.confidence,
    }));
  } catch (e) {
    console.error('[api] getDiseaseImages:', e);
    throw e;
  }
}

// ── System health ─────────────────────────────────────────────────────────────

/** GET /api/system/health — sensor array and pipeline health rows */
export async function getSystemHealth() {
  try {
    const d = await get('/api/system/health');
    return (d.rows ?? []).map(r => ({
      label:  r.label,
      status: r.status,
      ok:     r.ok,
    }));
  } catch (e) {
    console.error('[api] getSystemHealth:', e);
    throw e;
  }
}
