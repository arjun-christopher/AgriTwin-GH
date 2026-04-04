/**
 * AgriTwin-GH — Centralised Mock Data
 *
 * Single source of truth for all static frontend data.
 * When FastAPI integration is added, replace each export with a call
 * to the corresponding function in /src/services/api.js.
 *
 * No lucide-react imports here — icon names are stored as strings.
 * Component files contain the lookup map: `const ICON = { Fan, Droplets, … }`.
 *
 * ── API integration map ────────────────────────────────────────────────
 *   CROP              →  GET  /api/dt/state          (.crop)
 *   CROP_HEALTH       →  GET  /api/dt/state          (.health)
 *   WEATHER_STATUS    →  GET  /api/weather/current   (.status)
 *   OUTDOOR_CURRENT   →  GET  /api/weather/current   (.current)
 *   OUTDOOR_FORECAST  →  GET  /api/weather/current   (.forecast)
 *   ACTUATOR_STATE    →  GET  /api/actuators/state
 *   MONTHLY_RESOURCES →  GET  /api/resources/monthly (.resources)
 *   MONTHLY_COST      →  GET  /api/resources/monthly (.cost)
 *   CROP_IMAGES       →  GET  /api/media/latest
 *   SENSOR_METRICS    →  GET  /api/sensors/summary
 *   INDOOR_SENSORS    →  GET  /api/sensors/detail
 *   HEALTH_ROWS       →  GET  /api/system/health
 *   GROWTH_INTEL      →  GET  /api/intelligence/growth
 *   DISEASE_RISKS     →  GET  /api/intelligence/disease
 *   DISEASE_IMAGES    →  GET  /api/media/disease-scans
 *   GROWTH_STAGE_IMAGES → GET /api/media/stage-images
 */

// ── Crop stage ────────────────────────────────────────────────────────────────
export const CROP = {
  current:       'Flowering',
  currentIndex:  3,              // 0-based into stages[]
  currentPct:    95,             // % through the current stage
  daysInStage:   12,
  stageDuration: 14,
  next:          'Fruiting',
  nextInDays:    2,
  stages: ['Germination', 'Seedling', 'Vegetative', 'Flowering', 'Fruiting', 'Harvest'],
};

/** Max expected days for each stage — used for slider clamping. */
export const STAGE_MAX_DAYS = {
  Germination: 10,
  Seedling:    28,
  Vegetative:  42,
  Flowering:   21,
  Fruiting:    35,
  Harvest:     14,
};

// ── Crop health ───────────────────────────────────────────────────────────────
// status: 'Healthy' | 'Warning' | 'Risk'
export const CROP_HEALTH = {
  status:     'Healthy',
  confidence: 99.5,
  detail:     'No anomalies detected. Leaf scan returned clear.',
  scannedAgo: '2 mins ago',
};

// ── Weather / outdoor conditions ──────────────────────────────────────────────
export const WEATHER_STATUS = {
  status:    'Warning',
  condition: 'High External Humidity',
  detail:    'Ext. humidity at 89%. Risk of condensation — ventilation recommended.',
  forecast:  'Expected to clear in ~3 hrs',
};

export const OUTDOOR_CURRENT = {
  temp: 18.4, humidity: 89, windSpeed: 12, windDir: 'NNE',
  pressure: 1013, uvIndex: 3, dewPoint: 16.8, condition: 'Overcast', visibility: 8.2,
};

export const OUTDOOR_FORECAST = [
  { time: '03:00', high: 17, low: 15, humidity: 88, condition: 'Overcast',      iconKey: 'Cloud'        },
  { time: '06:00', high: 16, low: 14, humidity: 85, condition: 'Light Rain',    iconKey: 'CloudDrizzle' },
  { time: '09:00', high: 18, low: 16, humidity: 82, condition: 'Partly Cloudy', iconKey: 'CloudSun'     },
  { time: '12:00', high: 22, low: 19, humidity: 74, condition: 'Partly Cloudy', iconKey: 'CloudSun'     },
  { time: '15:00', high: 24, low: 21, humidity: 68, condition: 'Sunny',         iconKey: 'Sun'          },
  { time: '18:00', high: 21, low: 18, humidity: 72, condition: 'Clear Evening', iconKey: 'Moon'         },
];

// ── Actuator states ───────────────────────────────────────────────────────────
// color: 'primary' | 'secondary' | 'warning' | 'danger' | 'neutral'
// iconKey: lucide icon name resolved by each page's ICON_MAP constant
export const ACTUATOR_STATE = [
  { id: 'fan',       iconKey: 'Fan',         label: 'Ventilation Fan',  status: 'ON',   active: true,  level: 75, color: 'primary'   },
  { id: 'irrigation',iconKey: 'Droplets',    label: 'Irrigation',       status: 'ON',   active: true,  level: 45, color: 'secondary' },
  { id: 'lights',    iconKey: 'Sun',         label: 'Grow Lights',      status: 'OFF',  active: false, level: 0,  color: 'neutral'   },
  { id: 'co2',       iconKey: 'Wind',        label: 'CO₂ Injection',    status: 'ON',   active: true,  level: 55, color: 'warning'   },
  { id: 'heating',   iconKey: 'Thermometer', label: 'Heating',          status: 'ON',   active: true,  level: 60, color: 'danger'    },
  { id: 'nutrients', iconKey: 'FlaskConical',label: 'Nutrient Dosing',  status: 'OFF',  active: false, level: 30, color: 'primary'   },
];

// ── Monthly resource budget ───────────────────────────────────────────────────
// barClass retained for any components that still use it from this file
export const MONTHLY_RESOURCES = [
  { label: 'Water',  used: 1840, unit: 'L'   },
  { label: 'Energy', used: 312,  unit: 'kWh' },
];

export const MONTHLY_COST = {
  month:  'April 2026',
  energy: 43.68,
  water:  7.36,
  total:  51.04,
};

// ── Latest camera images ──────────────────────────────────────────────────────
export const CROP_IMAGES = {
  stage: {
    src:      'https://picsum.photos/seed/tomato-flower/480/280',
    alt:      'Tomato plant at flowering stage',
    badge:    'Day 12',
    location: 'GH-04 · Camera 2',
    captured: '1 hr ago',
  },
  leaf: {
    src:      'https://picsum.photos/seed/leaf-healthy-scan/480/280',
    alt:      'Leaf scan — GH-04 Sector B',
    badge:    '99.5% Conf.',
    location: 'Sector B · Leaf #7',
    captured: '2 mins ago',
  },
};

// ── Summary sensor metrics (header chips) ────────────────────────────────────
// iconKey resolved by ICON_MAP in consuming component
export const SENSOR_METRICS = [
  { iconKey: 'Thermometer', label: 'Temperature', value: '24.2', unit: '°C',  trend: 'up',     trendLabel: '+0.4°',  trendColor: 'text-warning'             },
  { iconKey: 'Droplets',    label: 'Humidity',    value: '65.8', unit: '%',   trend: 'down',   trendLabel: '−1.2%',  trendColor: 'text-danger'              },
  { iconKey: 'Wind',        label: 'CO₂ Level',   value: '812',  unit: 'ppm', trend: 'stable', trendLabel: 'Stable', trendColor: 'text-on-surface-variant'  },
  { iconKey: 'Sun',         label: 'Light',        value: '12k',  unit: 'lux', trend: 'up',     trendLabel: 'Peak',   trendColor: 'text-primary'             },
];

// ── System health rows ────────────────────────────────────────────────────────
export const HEALTH_ROWS = [
  { label: 'Sensor Array',  status: 'Nominal', ok: true  },
  { label: 'Network Link',  status: 'Strong',  ok: true  },
  { label: 'Data Pipeline', status: 'Active',  ok: true  },
  { label: 'Calibration',   status: 'Due: 6d', ok: false },
];

// ── Growth intelligence ───────────────────────────────────────────────────────
export const GROWTH_INTEL = {
  hoursToNextStage:  47,
  transitionProb24h: 18,
  stageHistory: [
    { stage: 'Germination', daysUsed: 7,  daysTarget: 7,  complete: true  },
    { stage: 'Seedling',    daysUsed: 21, daysTarget: 21, complete: true  },
    { stage: 'Vegetative',  daysUsed: 35, daysTarget: 35, complete: true  },
    { stage: 'Flowering',   daysUsed: 12, daysTarget: 14, complete: false },
  ],
};

// ── Disease risks ─────────────────────────────────────────────────────────────
// severity: 'High' | 'Medium' | 'Low'
export const DISEASE_RISKS = [
  { name: 'Late Blight',        pathogen: 'P. infestans',   risk24h: 31, severity: 'High'   },
  { name: 'Gray Mold',          pathogen: 'B. cinerea',     risk24h: 22, severity: 'Medium' },
  { name: 'Early Blight',       pathogen: 'A. solani',      risk24h: 8,  severity: 'Low'    },
  { name: 'Powdery Mildew',     pathogen: 'L. taurica',     risk24h: 5,  severity: 'Low'    },
  { name: 'Septoria Leaf Spot', pathogen: 'S. lycopersici', risk24h: 3,  severity: 'Low'    },
];

// ── Indoor sensor detail (DetailedInsights page) ──────────────────────────────
// rangeMin/rangeMax: expected normal operating range for the progress bar
export const INDOOR_SENSORS = [
  { iconKey: 'Thermometer', label: 'Temperature', value: 24.2,  unit: '°C',    status: 'ok', rangeMin: 22,    rangeMax: 26,    optimal: '22 – 26°C'   },
  { iconKey: 'Droplets',    label: 'Humidity',    value: 65.8,  unit: '%',     status: 'ok', rangeMin: 60,    rangeMax: 70,    optimal: '60 – 70%'    },
  { iconKey: 'Wind',        label: 'CO₂',         value: 812,   unit: 'ppm',   status: 'ok', rangeMin: 800,   rangeMax: 1000,  optimal: '800 – 1000'  },
  { iconKey: 'Sun',         label: 'Light',        value: 12000, unit: 'lux',   status: 'ok', rangeMin: 10000, rangeMax: 15000, optimal: '10k – 15k'   },
  { iconKey: 'Gauge',       label: 'VPD',          value: 1.02,  unit: 'kPa',   status: 'ok', rangeMin: 0.8,   rangeMax: 1.2,   optimal: '0.8 – 1.2'   },
  { iconKey: 'FlaskConical',label: 'pH',           value: 6.2,   unit: 'pH',    status: 'ok', rangeMin: 5.8,   rangeMax: 6.5,   optimal: '5.8 – 6.5'   },
  { iconKey: 'Zap',         label: 'EC',           value: 2.4,   unit: 'mS/cm', status: 'ok', rangeMin: 2.0,   rangeMax: 3.0,   optimal: '2.0 – 3.0'   },
];

// ── Stage camera images ───────────────────────────────────────────────────────
export const GROWTH_STAGE_IMAGES = [
  { src: 'https://picsum.photos/seed/tomato-germ-4/400/260',   alt: 'Germination stage',         stage: 'Germination', dayLabel: 'Day 7',   confidence: 99.1 },
  { src: 'https://picsum.photos/seed/tomato-seedl-4/400/260',  alt: 'Seedling stage',            stage: 'Seedling',    dayLabel: 'Day 28',  confidence: 98.7 },
  { src: 'https://picsum.photos/seed/tomato-veget-4/400/260',  alt: 'Vegetative stage',          stage: 'Vegetative',  dayLabel: 'Day 63',  confidence: 99.3 },
  { src: 'https://picsum.photos/seed/tomato-flower-4/400/260', alt: 'Flowering stage — current', stage: 'Flowering ★', dayLabel: 'Day 75',  confidence: 99.5 },
  { src: 'https://picsum.photos/seed/tomato-fruit-4/400/260',  alt: 'Fruiting — projected',      stage: 'Fruiting',    dayLabel: 'Day ~77', confidence: null  },
];

// ── Disease scan images ───────────────────────────────────────────────────────
export const DISEASE_IMAGES = [
  { src: 'https://picsum.photos/seed/disease-lb-4/400/260',  alt: 'Late Blight scan',        disease: 'Late Blight',        risk: 'High',   confidence: 92.4 },
  { src: 'https://picsum.photos/seed/disease-gm-4/400/260',  alt: 'Gray Mold scan',          disease: 'Gray Mold',          risk: 'Medium', confidence: 94.8 },
  { src: 'https://picsum.photos/seed/disease-eb-4/400/260',  alt: 'Early Blight scan',       disease: 'Early Blight',       risk: 'Low',    confidence: 96.2 },
  { src: 'https://picsum.photos/seed/disease-pm-4/400/260',  alt: 'Powdery Mildew scan',     disease: 'Powdery Mildew',     risk: 'Low',    confidence: 97.1 },
  { src: 'https://picsum.photos/seed/disease-sep-4/400/260', alt: 'Septoria Leaf Spot scan', disease: 'Septoria Leaf Spot', risk: 'Low',    confidence: 98.0 },
];

// ── Quick action presets ──────────────────────────────────────────────────────
// Used by HomeDashboard right panel and future ControlPanel page.
// Each preset maps to a future POST /api/dt/preset/{id} call.
export const QUICK_PRESETS = [
  {
    id:   'day-cycle',
    label:'Day Cycle',
    desc: 'Lights on · Fan max · CO₂ boost',
    iconKey: 'Sun',
  },
  {
    id:   'night-cycle',
    label:'Night Cycle',
    desc: 'Lights off · Fan low · Temp hold',
    iconKey: 'Moon',
  },
  {
    id:   'emergency-flush',
    label:'Emergency Flush',
    desc: 'Flush irrigation · Open vents',
    iconKey: 'AlertTriangle',
  },
];
