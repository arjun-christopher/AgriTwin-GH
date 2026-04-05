import { useState, useEffect } from 'react';
import {
  Thermometer,
  Droplets,
  Wind,
  Sun,
  Flame,
  CloudDrizzle,
  ShieldCheck,
  AlertTriangle,
  CheckCircle2,
  Fan,
  Leaf,
  Activity,
  Zap,
  Clock,
  Cloud,
  Gauge,
  FlaskConical,
  Camera,
  ScanSearch,
  Sprout,
} from 'lucide-react';

import StatusCard from '../components/ui/StatusCard';
import {
  getDtState,
  getActuatorState,
  getDiseaseRisks,
  getWeather,
  getLatestMedia,
  getMonthlyResources,
} from '../services/api.js';

const SENSOR_ICON_MAP = {
  Thermometer, Droplets, Wind, Sun, Gauge, Leaf, ShieldCheck, FlaskConical,
  Activity, CloudDrizzle, AlertTriangle, Sprout,
};

const ACTUATOR_ICON_MAP = {
  Fan, Wind, Droplets, Flame, Sun, Leaf, CloudDrizzle, Thermometer,
};

const SENSOR_2DP_KEYS = new Set(['vpd', 'leaf_wetness', 'disease_risk_score']);

function fmtSensorVal(key, v) {
  if (key === 'light_intensity') return v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${Math.round(v)}`;
  if (key === 'co2') return `${Math.round(v)}`;
  return v.toFixed(SENSOR_2DP_KEYS.has(key) ? 2 : 1);
}

/* ════════════════════════════════════════════════════════════════════════
   STATIC THEME MAPS  — all values are literal class strings so Tailwind v4
   content scanner can detect them at build time. Never construct dynamically.
════════════════════════════════════════════════════════════════════════ */

const RISK_THEME = {
  Low:    { bar: 'bg-primary', text: 'text-primary', pill: 'bg-primary/10 text-primary border border-primary/20'  },
  Medium: { bar: 'bg-warning', text: 'text-warning', pill: 'bg-warning/10 text-warning border border-warning/20'  },
  High:   { bar: 'bg-danger',  text: 'text-danger',  pill: 'bg-danger/10  text-danger  border border-danger/20'   },
};


/* ════════════════════════════════════════════════════════════════════════
   MOCK DATA  — static. Replace each constant with an API call later.
════════════════════════════════════════════════════════════════════════ */

// ── Crop Stage ──────────────────────────────────────────────────────────
// Canonical stages from src/agritwin_gh/mpc/constants.py GROWTH_STAGES tuple
// Durations from docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md
const CROP = {
  current: 'Flowering', currentIndex: 3, currentPct: 80,
  daysInStage: 12, stageDuration: 15,
  next: 'Unripe', nextInDays: 5,
  stages: ['Seedling', 'Early Veg.', 'Flw. Init.', 'Flowering', 'Unripe', 'Ripe'],
};

// ── Growth Intelligence ─────────────────────────────────────────────────
// Stage durations from docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md
const GROWTH_INTEL = {
  hoursToNextStage: 47,
  transitionProb24h: 18,
  stageHistory: [
    { stage: 'Seedling',          daysUsed: 14, daysTarget: 14, complete: true  },
    { stage: 'Early Vegetative',  daysUsed: 21, daysTarget: 21, complete: true  },
    { stage: 'Flw. Initiation',   daysUsed: 15, daysTarget: 15, complete: true  },
    { stage: 'Flowering',         daysUsed: 12, daysTarget: 15, complete: false },
  ],
};

// ── Health Status ────────────────────────────────────────────────────────────
const CROP_HEALTH = {
  status: 'Healthy',
  detail: 'No anomalies detected. Leaf scan returned clear.',
  scannedAgo: '2 mins ago',
};

// ── Actuators — from MPC ActuatorState (constants.py CONTROL_VARIABLES) ─────
const ACTUATORS = [
  { icon: Fan,         label: 'Fan Speed',    active: true  },
  { icon: Wind,        label: 'Vent Opening', active: true  },
  { icon: Droplets,    label: 'Irrigation',   active: true  },
  { icon: Thermometer, label: 'Heater',       active: true  },
  { icon: Sun,         label: 'LED Intensity',active: false },
  { icon: FlaskConical,label: 'CO₂ Valve',    active: true  },
  { icon: Droplets,    label: 'Fogger',       active: false },
];

// ── Key Indoor Metrics — from MPC GreenhouseState (STATE_VARIABLES) ────────
// Excludes growth_stage_index (displayed separately as stage tracker)
const SUMMARY_METRICS = [
  { icon: Thermometer,  label: 'Indoor Temp',    value: '24.2', unit: '°C',    note: 'MPC state' },
  { icon: Droplets,     label: 'Humidity',       value: '65.8', unit: '%',     note: 'MPC state' },
  { icon: Droplets,     label: 'Soil Moisture',  value: '62.0', unit: '%',     note: 'MPC state' },
  { icon: Wind,         label: 'CO₂',            value: '812',  unit: 'ppm',   note: 'MPC state' },
  { icon: Sun,          label: 'Light Intensity',value: '12k',  unit: 'lux',   note: 'MPC state' },
  { icon: Gauge,        label: 'VPD',            value: '1.02', unit: 'kPa',   note: 'MPC state' },
  { icon: Leaf,         label: 'Leaf Wetness',   value: '0.12', unit: 'proxy', note: 'MPC state' },
  { icon: ShieldCheck,  label: 'Disease Risk',   value: '0.12', unit: 'score', note: 'MPC state' },
];

// ── Disease Risks — canonical DISEASE_CATEGORIES from constants.py ──────────
const DISEASE_RISKS = [
  { name: 'Late Blight',    pathogen: 'P. infestans', risk24h: 31, severity: 'High'   },
  { name: 'Early Blight',   pathogen: 'A. solani',    risk24h: 8,  severity: 'Low'    },
  { name: 'Powdery Mildew', pathogen: 'L. taurica',   risk24h: 5,  severity: 'Low'    },
  { name: 'Spider Mites',   pathogen: 'T. urticae',   risk24h: 16, severity: 'Medium' },
  { name: 'Leaf Mold',      pathogen: 'P. fulva',     risk24h: 3,  severity: 'Low'    },
];

// ── Outdoor Weather — current + 24hr forecast values ───────────────────────
const OUTDOOR_CURRENT = {
  temp:          { now: 18.4,   forecast: 21.0  },
  humidity:      { now: 89,     forecast: 74    },
  windspeed:     { now: 12,     forecast: 8     },
  solarradiation:{ now: 210.5,  forecast: 380.0 },
  conditions:    { now: 'Overcast', forecast: 'Partly Cloudy' },
};

// ── Indoor Sensor Full Detail ───────────────────────────────────────────
const INDOOR_SENSORS = [
  { icon: Thermometer, label: 'Indoor Temp',    value: 24.2,  unit: '°C',    status: 'ok', rangeMin: 22,    rangeMax: 26,    optimal: '22 – 26°C'  },
  { icon: Droplets,    label: 'Humidity',       value: 65.8,  unit: '%',     status: 'ok', rangeMin: 60,    rangeMax: 70,    optimal: '60 – 70%'   },
  { icon: Droplets,    label: 'Soil Moisture',  value: 62.0,  unit: '%',     status: 'ok', rangeMin: 50,    rangeMax: 75,    optimal: '50 – 75%'   },
  { icon: Wind,        label: 'CO₂',            value: 812,   unit: 'ppm',   status: 'ok', rangeMin: 800,   rangeMax: 1000,  optimal: '800 – 1000' },
  { icon: Sun,         label: 'Light Intensity',value: 12000, unit: 'lux',   status: 'ok', rangeMin: 10000, rangeMax: 15000, optimal: '10k – 15k'  },
  { icon: Gauge,       label: 'VPD',            value: 1.02,  unit: 'kPa',   status: 'ok', rangeMin: 0.8,   rangeMax: 1.2,   optimal: '0.8 – 1.2'  },
  { icon: Droplets,    label: 'Dew Point',      value: 16.8,  unit: '°C',    status: 'ok', rangeMin: 14,    rangeMax: 18,    optimal: '14 – 18°C'  },
  { icon: Leaf,        label: 'Leaf Wetness',   value: 0.12,  unit: 'proxy', status: 'ok', rangeMin: 0,     rangeMax: 0.5,   optimal: '0.0 – 0.3'  },
  { icon: ShieldCheck, label: 'Disease Risk',   value: 0.12,  unit: 'score', status: 'ok', rangeMin: 0,     rangeMax: 1.0,   optimal: '< 0.3'      },
];

// ── Latest media fallbacks (until API populates)
const DEFAULT_IMAGES = {
  stage: { src: 'https://picsum.photos/seed/tomato-stage/480/280', alt: 'Crop stage image', badge: 'Stage', location: 'Camera', captured: 'just now' },
  leaf:  { src: 'https://picsum.photos/seed/tomato-leaf/480/280',  alt: 'Leaf scan image',  badge: 'Scan',  location: 'Camera', captured: 'just now' },
};

// ── Monthly Resources fallbacks
const DEFAULT_RESOURCES = [
  { label: 'Water',  used: 0, unit: 'L'   },
  { label: 'Energy', used: 0, unit: 'kWh' },
];
const DEFAULT_COST = { month: '—', energy: 0, water: 0, total: 0 };

// ── Historical Charts (kept from original page) ─────────────────────────


/* ════════════════════════════════════════════════════════════════════════
   SUB-COMPONENTS
════════════════════════════════════════════════════════════════════════ */

/** Consistent section heading with icon badge. */
function SectionHeader({ icon: Icon, eyebrow, title, subtitle }) {
  return (
    <div className="flex items-start gap-4 mb-6">
      <div className="shrink-0 mt-0.5 p-2.5 rounded-xl bg-primary/10 text-primary">
        <Icon size={16} />
      </div>
      <div>
        <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-0.5">
          {eyebrow}
        </p>
        <h2 className="text-2xl font-headline font-bold text-on-surface leading-none">{title}</h2>
        {subtitle && (
          <p className="text-xs text-on-surface-variant font-light mt-1">{subtitle}</p>
        )}
      </div>
    </div>
  );
}

/** Compact metric display chip for summary row. */
function MetricChip({ icon: Icon, label, value, unit, note }) {
  return (
    <div className="bg-surface-high rounded-xl p-4 border border-outline-variant/10 flex flex-col gap-1">
      <div className="flex items-center gap-1.5 text-on-surface-variant">
        <Icon size={11} />
        <span className="text-[8px] uppercase tracking-widest">{label}</span>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-xl font-headline font-bold text-on-surface">{value}</span>
        <span className="text-[10px] text-on-surface-variant">{unit}</span>
      </div>
      {note && <p className="text-[8px] text-on-surface-variant opacity-60">{note}</p>}
    </div>
  );
}

/** Actuator status pill chip — shows label, value, and ON/OFF status. */
const CHIP_ON   = 'bg-primary/10 text-primary border-primary/20';
const CHIP_OFF  = 'bg-surface-highest text-on-surface-variant border-outline-variant/20';
function ActuatorChip({ icon: Icon, label, active }) {
  return (
    <div className={`flex items-center gap-2 px-3.5 py-2 rounded-full border text-[10px] font-bold ${active ? CHIP_ON : CHIP_OFF}`}>
      <Icon size={11} />
      <span>{label}</span>
      <span className="opacity-50 font-normal">· {active ? 'ON' : 'OFF'}</span>
    </div>
  );
}

/**
 * CropStageTrack — horizontal stage dot + connecting-line progression.
 * Copied from HomeDashboard for visual consistency.
 */
function CropStageTrack({ stages, currentIndex, currentPct, allComplete = false }) {
  return (
    <div className="relative flex items-start justify-between w-full pt-1">
      <div className="absolute top-3.5 left-4 right-4 h-px bg-outline-variant/20" />
      <div
        className="absolute top-3.5 left-4 right-4 h-px bg-primary transition-all duration-700 origin-left"
        style={{
          transform: allComplete ? 'scaleX(1)' : `scaleX(${currentIndex === 0 ? 0 : (currentIndex / (stages.length - 1)).toFixed(4)})`,
        }}
      />
      {stages.map((stage, i) => {
        const done    = allComplete ? true : i < currentIndex;
        const current = allComplete ? false : i === currentIndex;
        return (
          <div key={stage} className="relative z-10 flex flex-col items-center gap-2 w-14">
            <div className={[
              'w-7 h-7 rounded-full flex items-center justify-center border-2 transition-all duration-300 shrink-0',
              done    ? 'bg-primary border-primary'
                      : current
                      ? 'bg-primary/15 border-primary ring-4 ring-primary/15'
                      : 'bg-surface-high border-outline-variant/30',
            ].join(' ')}>
              {done    && <CheckCircle2 size={12} className="text-on-primary" />}
              {current && <span className="text-[8px] font-black text-primary">{currentPct}%</span>}
            </div>
            <span className={[
              'text-[8px] text-center leading-tight',
              current ? 'text-primary font-bold'
                      : done ? 'text-on-surface-variant opacity-70'
                             : 'text-on-surface-variant opacity-35',
            ].join(' ')}>
              {stage}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/** Large spotlight stat used for hours-to-next and transition probability.
 *  Pass `days` to show an additional "/ N days" annotation next to hours.
 */
const SPOTLIGHT_ACCENT = {
  primary:   { bg: 'bg-primary/5 border-primary/15',    value: 'text-primary'   },
  secondary: { bg: 'bg-secondary/5 border-secondary/15',value: 'text-secondary' },
  warning:   { bg: 'bg-warning/5 border-warning/15',    value: 'text-warning'   },
};
function GrowthSpotlight({ label, value, unit, sub, accent = 'primary', days }) {
  const a = SPOTLIGHT_ACCENT[accent] ?? SPOTLIGHT_ACCENT.primary;
  return (
    <div className={`rounded-xl p-6 border flex flex-col justify-center ${a.bg}`}>
      <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-3">{label}</p>
      <div className="flex items-baseline gap-2">
        <span className={`text-4xl font-headline font-bold ${a.value}`}>{value}</span>
        <span className="text-sm text-on-surface-variant font-light">{unit}</span>
        {days != null && (
          <span className="text-xs text-on-surface-variant opacity-60">
            / {days} day{days !== 1 ? 's' : ''}
          </span>
        )}
      </div>
      {sub && <p className="text-[10px] text-on-surface-variant mt-2 leading-snug">{sub}</p>}
    </div>
  );
}

/** Single row in stage history comparison. */
function StageHistoryRow({ stage, daysUsed, daysTarget, complete }) {
  const pct = Math.min(100, Math.round((daysUsed / daysTarget) * 100));
  const onTime = daysUsed <= daysTarget;
  return (
    <div className="flex items-center gap-3">
      <span className="w-24 shrink-0 text-[9px] text-on-surface-variant">{stage}</span>
      <div className="flex-1 h-1.5 bg-surface-highest rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${onTime ? 'bg-primary' : 'bg-warning'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="w-24 shrink-0 flex items-center justify-end gap-1.5">
        <span className="text-[9px] text-on-surface-variant">{daysUsed}d / {daysTarget}d</span>
        {complete && onTime  && <CheckCircle2 size={10} className="text-primary" />}
        {complete && !onTime && <AlertTriangle size={10} className="text-warning" />}
        {!complete           && <Clock size={10} className="text-on-surface-variant opacity-40" />}
      </div>
    </div>
  );
}

/** Disease risk card with risk bar + severity badge. */
function DiseaseRiskCard({ name, pathogen, risk24h, severity }) {
  const t = RISK_THEME[severity] ?? RISK_THEME.Low;
  return (
    <div className="bg-surface-high rounded-xl p-4 border border-outline-variant/10 flex flex-col gap-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <p className="text-[11px] font-bold text-on-surface leading-tight">{name}</p>
          <p className="text-[8px] text-on-surface-variant opacity-60 italic mt-0.5">{pathogen}</p>
        </div>
        <span className={`shrink-0 text-[8px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full ${t.pill}`}>
          {severity}
        </span>
      </div>
      <div>
        <div className="flex justify-between text-[9px] mb-1.5">
          <span className="text-on-surface-variant">24hr risk</span>
          <span className={`font-bold ${t.text}`}>{risk24h}%</span>
        </div>
        <div className="h-1.5 bg-surface-highest rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${t.bar}`}
            style={{ width: `${Math.min(100, risk24h)}%` }}
          />
        </div>
      </div>
    </div>
  );
}

/** Polished image frame for both visual galleries. */
function GalleryFrame({ src, alt, label, badge, sub, riskBadge }) {
  return (
    <div className="relative rounded-xl overflow-hidden bg-surface-deep border border-outline-variant/10 group cursor-default">
      <img
        src={src}
        alt={alt}
        className="w-full aspect-video object-cover group-hover:scale-105 transition-transform duration-500"
        loading="lazy"
      />
      {/* Bottom overlay */}
      <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/75 to-transparent px-3 py-2.5">
        <p className="text-[10px] font-bold text-white truncate leading-tight">{label}</p>
        {sub && <p className="text-[8px] text-white/60 mt-0.5">{sub}</p>}
      </div>
      {/* Right badge (day / confidence) */}
      {badge && (
        <span className="absolute top-2 right-2 text-[8px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-black/50 text-white border border-white/10">
          {badge}
        </span>
      )}
      {/* Left badge (risk level) */}
      {riskBadge && (
        <span className={`absolute top-2 left-2 text-[8px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full ${riskBadge.theme.pill}`}>
          {riskBadge.label}
        </span>
      )}
    </div>
  );
}

/** Single outdoor condition metric tile — shows current + 24hr forecast. */
function OutdoorMetric({ icon: Icon, label, value, forecast }) {
  return (
    <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10">
      <div className="flex items-center gap-1.5 mb-2 text-on-surface-variant">
        <Icon size={12} />
        <span className="text-[8px] uppercase tracking-widest">{label}</span>
      </div>
      <p className="text-2xl font-headline font-bold text-on-surface leading-none">{value}</p>
      {forecast && (
        <div className="flex items-center gap-1 mt-1.5">
          <span className="text-[8px] text-on-surface-variant opacity-50">24 hr →</span>
          <span className="text-[9px] font-semibold text-on-surface-variant">{forecast}</span>
        </div>
      )}
    </div>
  );
}

/** Indoor sensor card — current value + DT step delta. */
function SensorCard({ icon: Icon, label, value, unit, delta }) {
  const display = value >= 1000 ? `${(value / 1000).toFixed(0)}k` : value;
  const hasDelta = delta != null && delta !== 0;
  const deltaColor = hasDelta
    ? (delta > 0 ? 'text-primary' : 'text-warning')
    : 'text-on-surface-variant';
  const deltaStr = hasDelta
    ? `${delta > 0 ? '+' : ''}${Math.abs(delta) < 1 ? delta.toFixed(2) : delta.toFixed(1)}`
    : null;
  return (
    <div className="bg-surface-high rounded-xl p-4 border border-outline-variant/10">
      <div className="flex items-center gap-1.5 mb-3 text-on-surface-variant">
        <Icon size={12} />
        <span className="text-[9px] uppercase tracking-widest">{label}</span>
      </div>
      <div className="flex items-baseline gap-1.5">
        <span className="text-2xl font-headline font-bold text-on-surface">{display}</span>
        <span className="text-xs text-on-surface-variant">{unit}</span>
      </div>
      {deltaStr && (
        <p className={`text-[9px] font-semibold mt-1 ${deltaColor}`}>
          Δ {deltaStr} {unit}
        </p>
      )}
    </div>
  );
}


/* ════════════════════════════════════════════════════════════════════════
   PAGE
════════════════════════════════════════════════════════════════════════ */
function DetailedInsights() {
  const [crop, setCrop]               = useState(CROP);
  const [health, setHealth]           = useState(CROP_HEALTH);
  const [growthIntel, setGrowthIntel] = useState(GROWTH_INTEL);
  const [actuators, setActuators]     = useState(ACTUATORS);
  const [diseaseRisks, setDiseaseRisks]     = useState(DISEASE_RISKS);
  const [weather, setWeather]         = useState(null);
  const [latestImages, setLatestImages]   = useState(DEFAULT_IMAGES);
  const [resources, setResources]     = useState(DEFAULT_RESOURCES);
  const [cost, setCost]               = useState(DEFAULT_COST);
  const [actuatorCosts, setActuatorCosts] = useState([]);
  const [indoorSensors, setIndoorSensors] = useState(INDOOR_SENSORS);

  useEffect(() => {
    // Polled every 10 s — keeps indoor sensors and actuators live
    function fetchDtLive() {
      getDtState()
        .then(d => {
          if (d?.crop)   setCrop(d.crop);
          if (d?.health) setHealth(d.health);
          if (d?.growth) setGrowthIntel(d.growth);
          if (d?.sensors?.length) {
            setIndoorSensors(d.sensors.map(s => ({
              icon:  SENSOR_ICON_MAP[s.iconKey] ?? Thermometer,
              label: s.label,
              value: s.value,
              unit:  s.unit,
              delta: s.delta ?? 0,
            })));
          }
        })
        .catch(() => {});

      getActuatorState()
        .then(acts => {
          if (acts?.length) {
            setActuators(acts.map(a => ({ ...a, icon: ACTUATOR_ICON_MAP[a.iconKey] ?? Wind })));
          }
        })
        .catch(() => {});

      getMonthlyResources()
        .then(d => {
          if (d?.resources) setResources(d.resources);
          if (d?.cost)      setCost(d.cost);
          if (d?.actuators) setActuatorCosts(d.actuators);
        })
        .catch(() => {});
    }

    fetchDtLive();
    const timer = setInterval(fetchDtLive, 10_000);

    // One-time fetches (static / infrequently changing data)
    getDiseaseRisks()
      .then(d => { if (d?.length) setDiseaseRisks(d); })
      .catch(() => {});

    getWeather()
      .then(d => { if (d) setWeather(d); })
      .catch(() => {});

    getLatestMedia()
      .then(d => {
        if (d) setLatestImages(prev => ({
          stage: d.stage?.src ? d.stage : prev.stage,
          leaf:  d.leaf?.src  ? d.leaf  : prev.leaf,
        }));
      })
      .catch(() => {});

    return () => clearInterval(timer);
  }, []);

  // Construct nested {now, forecast} shape from flat weather API response.
  // forecast24h comes from the weather-forecast model (cadence_info weather_24h_ahead).
  const outdoorMetrics = weather ? {
    temp:          { now: weather.current.temp,      forecast: weather.forecast24h?.temp_external      ?? weather.current.temp      },
    humidity:      { now: weather.current.humidity,  forecast: weather.forecast24h?.humidity_external  ?? weather.current.humidity  },
    windspeed:     { now: weather.current.windSpeed, forecast: weather.forecast24h?.windspeed          ?? weather.current.windSpeed },
    solarradiation:{ now: weather.current.solarRad,  forecast: weather.forecast24h?.solar_radiation    ?? weather.current.solarRad  },
    conditions:    { now: weather.current.condition, forecast: weather.forecast24h?.conditions         ?? weather.current.condition },
  } : OUTDOOR_CURRENT;

  return (
    <div className="py-6 flex flex-col gap-12">

      {/* ── Page header ──────────────────────────────────────────────── */}
      <header>
        <h1 className="text-5xl font-headline font-bold tracking-tighter">
          Detailed{' '}
          <span className="text-primary italic">Insights</span>
        </h1>
      </header>

      {/* ════════════════════════════════════════════════════════════════
          § 1 — GROWTH INTELLIGENCE
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Leaf}
          eyebrow="Growth Analysis"
          title="Growth Intelligence"
          subtitle="Stage progression, predicted time to transition, and historical comparison"
        />

        {/* Stage progress card */}
        <div className="bg-surface-low rounded-xl p-6 mb-5 relative overflow-hidden">
          <div className="pointer-events-none absolute top-0 right-0 w-56 h-56 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16" />
          <div className="relative z-10">
            <div className="flex items-start justify-between gap-4 mb-6">
              <div>
                <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-1">
                  Current Stage
                </p>
                <div className="flex items-baseline gap-3">
                  <span className="text-3xl font-headline font-bold text-primary leading-none">
                    {crop.current}
                  </span>
                  <span className="text-xs font-light text-on-surface-variant">
                    Day {crop.daysInStage} of {crop.stageDuration}
                  </span>
                </div>
              </div>
              <div className="shrink-0 text-right">
                <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-0.5">
                  {crop.current === 'Ripe' ? 'Action' : 'Next Stage'}
                </p>
                <p className="text-xl font-headline font-bold text-on-surface leading-none">
                  {crop.current === 'Ripe' ? 'Harvest' : (crop.next ?? '—')}
                </p>
                {crop.current !== 'Ripe' && (
                  <p className="text-[9px] text-on-surface-variant opacity-70 mt-0.5">
                    in {crop.nextInDays} days
                  </p>
                )}
              </div>
            </div>
            <div className="px-2">
              <CropStageTrack
                stages={crop.stages}
                currentIndex={crop.currentIndex}
                currentPct={crop.currentPct}
                allComplete={crop.current === 'Ripe'}
              />
            </div>
            {crop.current !== 'Ripe' && (
              <div className="mt-6 pt-4 border-t border-outline-variant/10">
                <div className="flex justify-between text-[9px] uppercase tracking-widest opacity-55 mb-2">
                  <span>Stage completion</span>
                  <span>{crop.currentPct}%</span>
                </div>
                <div className="h-1.5 bg-surface-highest rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-700"
                    style={{ width: `${crop.currentPct}%` }}
                  />
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Predicted time to next stage / Harvest Ready */}
        <div className="mb-5">
          {crop.current === 'Ripe' ? (
            <div className="rounded-xl p-6 border flex flex-col justify-center bg-primary/5 border-primary/20">
              <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-3">Status</p>
              <div className="flex items-center gap-3">
                <CheckCircle2 size={28} className="text-primary shrink-0" />
                <span className="text-4xl font-headline font-bold text-primary">Harvest Ready</span>
              </div>
              <p className="text-[10px] text-on-surface-variant mt-2 leading-snug">
                Crop has reached the final growth stage. Ready for harvest.
              </p>
            </div>
          ) : (
            <GrowthSpotlight
              label="Predicted Time to Next Stage"
              value={growthIntel.hoursToNextStage != null ? growthIntel.hoursToNextStage.toFixed(1) : '—'}
              unit="hrs"
              days={growthIntel.hoursToNextStage != null ? Math.ceil(growthIntel.hoursToNextStage / 24) : null}
              accent="primary"
            />
          )}
        </div>

        {/* Latest crop stage image — CNN classifier result */}
        <div className="bg-surface-deep rounded-xl overflow-hidden border border-outline-variant/10 flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-outline-variant/10">
            <div className="flex items-center gap-2">
              <Camera size={13} className="text-primary" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                Crop Stage Image
              </span>
            </div>
            <span className="text-[9px] text-on-surface-variant opacity-60">
              {latestImages.stage.captured}
            </span>
          </div>
          <div className="relative overflow-hidden">
            <img
              src={latestImages.stage.src}
              alt={latestImages.stage.alt}
              className="w-full aspect-video object-cover"
              loading="lazy"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/70 to-transparent px-4 py-3">
              <p className="text-[10px] font-bold text-white leading-none">{crop.current} Stage</p>
              <p className="text-[8px] text-white/70 mt-0.5">{latestImages.stage.location}</p>
            </div>
            <span className="absolute top-3 right-3 text-[8px] font-bold uppercase tracking-widest px-2 py-1 rounded-full bg-primary/20 text-primary border border-primary/25">
              {latestImages.stage.badge}
            </span>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 2 — DISEASE INTELLIGENCE
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={ShieldCheck}
          eyebrow="Pathogen Monitoring"
          title="Disease Intelligence"
          subtitle="24-hour risk prediction across all monitored pathogens · current leaf scan"
        />

        {/* Crop health banner — overall plant status */}
        <div className="mb-5">
          <StatusCard
            label="Crop Health"
            status={health.status}
            detail={health.detail}
            meta={health.scannedAgo}
            variant="crop"
          />
        </div>

        {/* Disease risk cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3 mb-5">
          {diseaseRisks.map((d) => (
            <DiseaseRiskCard key={d.name} {...d} />
          ))}
        </div>

        {/* Latest leaf / disease scan image — CNN classifier result */}
        <div className="bg-surface-deep rounded-xl overflow-hidden border border-outline-variant/10 flex flex-col">
          <div className="flex items-center justify-between px-4 py-3 border-b border-outline-variant/10">
            <div className="flex items-center gap-2">
              <ScanSearch size={13} className="text-primary" />
              <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                Leaf / Disease Scan
              </span>
            </div>
            <span className="text-[9px] text-on-surface-variant opacity-60">
              {latestImages.leaf.captured}
            </span>
          </div>
          <div className="relative overflow-hidden">
            <img
              src={latestImages.leaf.src}
              alt={latestImages.leaf.alt}
              className="w-full aspect-video object-cover"
              loading="lazy"
            />
            <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/70 to-transparent px-4 py-3">
              <p className="text-[10px] font-bold text-white leading-none">{latestImages.leaf.alt || 'Leaf Scan'}</p>
              <p className="text-[8px] text-white/70 mt-0.5">{latestImages.leaf.location}</p>
            </div>
            <span className="absolute top-3 right-3 text-[8px] font-bold uppercase tracking-widest px-2 py-1 rounded-full bg-primary/20 text-primary border border-primary/25">
              {latestImages.leaf.badge}
            </span>
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 3 — OUTDOOR WEATHER
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Cloud}
          eyebrow="External Environment"
          title="Outdoor Weather"
          subtitle="Live readings from the outdoor weather station"
        />

        {/* Current + 24hr forecast for each parameter */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <OutdoorMetric icon={Thermometer} label="Temperature"     value={`${outdoorMetrics.temp.now}°C`}                  forecast={`${outdoorMetrics.temp.forecast}°C`}              />
          <OutdoorMetric icon={Droplets}    label="Ext. Humidity"   value={`${outdoorMetrics.humidity.now}%`}               forecast={`${outdoorMetrics.humidity.forecast}%`}           />
          <OutdoorMetric icon={Wind}        label="Wind Speed"      value={`${outdoorMetrics.windspeed.now} km/h`}          forecast={`${outdoorMetrics.windspeed.forecast} km/h`}      />
          <OutdoorMetric icon={Sun}         label="Solar Radiation" value={`${outdoorMetrics.solarradiation.now} W/m²`}     forecast={`${outdoorMetrics.solarradiation.forecast} W/m²`} />
          <OutdoorMetric icon={Cloud}       label="Conditions"      value={outdoorMetrics.conditions.now}                    forecast={outdoorMetrics.conditions.forecast}               />
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 4 — INDOOR CONDITIONS
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Thermometer}
          eyebrow="Greenhouse Sensor Array"
          title="Indoor Conditions"
          subtitle="Live readings from greenhouse sensors"
        />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {indoorSensors.map((s) => (
            <SensorCard key={s.label} {...s} />
          ))}
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 5 — ACTUATOR STATUS
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Zap}
          eyebrow="Control Systems"
          title="Actuator Status"
          subtitle="Current MPC-driven actuator states across all 7 control channels"
        />
        <div className="bg-surface-low rounded-xl p-5 border border-outline-variant/10">
          <div className="flex flex-wrap gap-3">
            {actuators.map((a) => (
              <ActuatorChip key={a.label} {...a} />
            ))}
          </div>
        </div>
      </section>

      {/* ══════════════════════════════════════════════════════════════
          § 6 — MONTHLY RESOURCES & COST
      ══════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Zap}
          eyebrow="Resource Accounting"
          title="Monthly Resources & Cost"
          subtitle={`Operational consumption and cost for ${cost.month} · Tamil Nadu tariff rates`}
        />

        {/* Top row: Monthly Resources + Monthly Cost — same layout as dashboard */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">

          {/* Monthly resource usage */}
          <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10 flex flex-col gap-4">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                  Monthly Resources
                </h3>
                <p className="text-[9px] text-on-surface-variant mt-0.5">Monthly consumption</p>
              </div>
              <span className="text-[9px] text-on-surface-variant opacity-55 shrink-0">{cost.month}</span>
            </div>
            <div className="flex flex-col gap-3.5">
              {resources.map(({ label, used, unit }) => (
                <div key={label} className="flex items-center justify-between">
                  <span className="text-[9px] uppercase tracking-widest text-on-surface-variant opacity-70">{label}</span>
                  <span className="text-sm font-headline font-bold text-on-surface">
                    {used} <span className="text-xs font-light text-on-surface-variant">{unit}</span>
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Monthly cost breakdown */}
          <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10 flex flex-col gap-4">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                  Monthly Cost
                </h3>
                <p className="text-[9px] text-on-surface-variant mt-0.5">Operational expenditure</p>
              </div>
              <span className="text-[9px] text-on-surface-variant opacity-55 shrink-0">{cost.month}</span>
            </div>
            <div className="flex flex-col gap-2 flex-1">
              <div className="flex items-center justify-between">
                <span className="text-xs text-on-surface-variant">Energy</span>
                <span className="text-sm font-medium text-on-surface-variant">₹{cost.energy.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-on-surface-variant">Water</span>
                <span className="text-sm font-medium text-on-surface-variant">₹{cost.water.toFixed(2)}</span>
              </div>
              <div className="flex items-center justify-between pt-3 border-t border-outline-variant/15 mt-1">
                <span className="text-sm font-semibold text-on-surface">Total</span>
                <span className="text-2xl font-headline font-bold text-on-surface">₹{cost.total.toFixed(2)}</span>
              </div>
            </div>
            <div className="pt-3 border-t border-outline-variant/10 flex items-center justify-between">
              <span className="text-[9px] uppercase tracking-widest text-on-surface-variant opacity-55">
                Daily avg. cost
              </span>
              <span className="text-sm font-headline font-bold text-on-surface">
                ₹{cost.total > 0 ? (cost.total / new Date().getDate()).toFixed(2) : '0.00'}
              </span>
            </div>
          </div>
        </div>

        {/* Per-actuator resource breakdown */}
        {actuatorCosts.length > 0 && (
          <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10">
            <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface mb-4">
              Actuator Resource Breakdown
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
              {actuatorCosts.map((a) => (
                <div
                  key={a.key}
                  className="bg-surface-low rounded-xl p-4 border border-outline-variant/10 flex flex-col gap-2"
                >
                  <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">{a.label}</p>
                  <div className="flex items-center justify-between">
                    <span className="text-[9px] text-on-surface-variant">Energy</span>
                    <span className="text-xs font-medium text-on-surface">{a.energyKwh.toFixed(4)} kWh</span>
                  </div>
                  {a.waterL > 0 && (
                    <div className="flex items-center justify-between">
                      <span className="text-[9px] text-on-surface-variant">Water</span>
                      <span className="text-xs font-medium text-on-surface">{a.waterL.toFixed(1)} L</span>
                    </div>
                  )}
                  <div className="flex items-center justify-between pt-2 border-t border-outline-variant/10">
                    <span className="text-[9px] text-on-surface-variant">Cost</span>
                    <span className="text-sm font-headline font-bold text-primary">₹{a.costInr.toFixed(2)}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

    </div>
  );
}

export default DetailedInsights;
