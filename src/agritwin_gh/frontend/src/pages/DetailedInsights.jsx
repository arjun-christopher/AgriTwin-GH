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
} from 'lucide-react';

import StatusCard from '../components/ui/StatusCard';
import {
  getDtState,
  getActuatorState,
  getDiseaseRisks,
  getWeather,
  getStageImages,
  getDiseaseImages,
} from '../services/api.js';

const SENSOR_ICON_MAP = {
  Thermometer, Droplets, Wind, Sun, Gauge, Leaf, ShieldCheck, FlaskConical,
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
  { icon: Fan,         label: 'Fan Speed',    status: 'ON',  active: true,  value: '75%'  },
  { icon: Wind,        label: 'Vent Opening', status: 'ON',  active: true,  value: '45%'  },
  { icon: Droplets,    label: 'Irrigation',   status: 'ON',  active: true,  value: '45 L' },
  { icon: Thermometer, label: 'Heater',       status: 'ON',  active: true,  value: '60%'  },
  { icon: Sun,         label: 'LED Intensity',status: 'OFF', active: false, value: '0%'   },
  { icon: FlaskConical,label: 'CO₂ Valve',    status: 'ON',  active: true,  value: '55%'  },
  { icon: Droplets,    label: 'Fogger',       status: 'OFF', active: false, value: '0%'   },
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

// ── Recent Crop Images — rolling 30-min cadence, max 5 frames ───────────────
const RECENT_CROP_IMAGES = [
  { src: 'https://picsum.photos/seed/crop-r1/400/260', alt: 'Recent crop image 1', capturedAgo: '30 min ago',  stage: 'Flowering', confidence: 99.5 },
  { src: 'https://picsum.photos/seed/crop-r2/400/260', alt: 'Recent crop image 2', capturedAgo: '60 min ago',  stage: 'Flowering', confidence: 98.2 },
  { src: 'https://picsum.photos/seed/crop-r3/400/260', alt: 'Recent crop image 3', capturedAgo: '90 min ago',  stage: 'Flowering', confidence: 97.8 },
  { src: 'https://picsum.photos/seed/crop-r4/400/260', alt: 'Recent crop image 4', capturedAgo: '120 min ago', stage: 'Flowering', confidence: 99.1 },
  { src: 'https://picsum.photos/seed/crop-r5/400/260', alt: 'Recent crop image 5', capturedAgo: '150 min ago', stage: 'Flowering', confidence: 98.6 },
];

// ── Recent Leaf Scans — rolling 30-min cadence, max 5 frames ────────────────
// Classification labels from canonical DISEASE_CATEGORIES (constants.py)
const RECENT_LEAF_IMAGES = [
  { src: 'https://picsum.photos/seed/leaf-scan-1/400/260', alt: 'Leaf scan 1', capturedAgo: '30 min ago',  classification: 'Late Blight',    risk: 'High',   confidence: 91.4 },
  { src: 'https://picsum.photos/seed/leaf-scan-2/400/260', alt: 'Leaf scan 2', capturedAgo: '60 min ago',  classification: 'Healthy Leaves', risk: 'Low',    confidence: 97.8 },
  { src: 'https://picsum.photos/seed/leaf-scan-3/400/260', alt: 'Leaf scan 3', capturedAgo: '90 min ago',  classification: 'Powdery Mildew', risk: 'Low',    confidence: 62.3 },
  { src: 'https://picsum.photos/seed/leaf-scan-4/400/260', alt: 'Leaf scan 4', capturedAgo: '120 min ago', classification: 'Healthy Leaves', risk: 'Low',    confidence: 94.1 },
  { src: 'https://picsum.photos/seed/leaf-scan-5/400/260', alt: 'Leaf scan 5', capturedAgo: '150 min ago', classification: 'Spider Mites',   risk: 'Medium', confidence: 76.5 },
];

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
function ActuatorChip({ icon: Icon, label, status, active, value }) {
  return (
    <div className={`flex items-center gap-2 px-3.5 py-2 rounded-full border text-[10px] font-bold ${active ? CHIP_ON : CHIP_OFF}`}>
      <Icon size={11} />
      <span>{label}</span>
      {value && <span className="font-mono font-normal opacity-80">{value}</span>}
      <span className="opacity-50 font-normal">· {status}</span>
    </div>
  );
}

/**
 * CropStageTrack — horizontal stage dot + connecting-line progression.
 * Copied from HomeDashboard for visual consistency.
 */
function CropStageTrack({ stages, currentIndex, currentPct }) {
  return (
    <div className="relative flex items-start justify-between w-full pt-1">
      <div className="absolute top-3.5 left-4 right-4 h-px bg-outline-variant/20" />
      <div
        className="absolute top-3.5 left-4 h-px bg-primary transition-all duration-700"
        style={{
          width: currentIndex === 0
            ? '0%'
            : `${((currentIndex / (stages.length - 1)) * 100).toFixed(1)}%`,
        }}
      />
      {stages.map((stage, i) => {
        const done    = i < currentIndex;
        const current = i === currentIndex;
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

/** Indoor sensor card — current value only. */
function SensorCard({ icon: Icon, label, value, unit }) {
  const display = value >= 1000 ? `${(value / 1000).toFixed(0)}k` : value;
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
  const [summaryMetrics, setSummaryMetrics] = useState(SUMMARY_METRICS);
  const [diseaseRisks, setDiseaseRisks]     = useState(DISEASE_RISKS);
  const [weather, setWeather]         = useState(null);
  const [stageImages, setStageImages] = useState(RECENT_CROP_IMAGES);
  const [leafImages, setLeafImages]   = useState(RECENT_LEAF_IMAGES);

  useEffect(() => {
    getDtState()
      .then(d => {
        if (d?.crop)   setCrop(d.crop);
        if (d?.health) setHealth(d.health);
        if (d?.growth) setGrowthIntel(d.growth);
        if (d?.sensors?.length) {
          setSummaryMetrics(d.sensors.map(s => ({
            icon:  SENSOR_ICON_MAP[s.iconKey] ?? Thermometer,
            label: s.label,
            value: fmtSensorVal(s.key, s.value),
            unit:  s.unit,
            note:  'MPC state',
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

    getDiseaseRisks()
      .then(d => { if (d?.length) setDiseaseRisks(d); })
      .catch(() => {});

    getWeather()
      .then(d => { if (d) setWeather(d); })
      .catch(() => {});

    getStageImages()
      .then(d => { if (d?.length) setStageImages(d); })
      .catch(() => {});

    getDiseaseImages()
      .then(d => { if (d?.length) setLeafImages(d); })
      .catch(() => {});
  }, []);

  // Construct nested {now, forecast} shape from flat weather API response
  const outdoorMetrics = weather ? {
    temp:          { now: weather.current.temp,      forecast: weather.forecast.at(-1)?.high      ?? weather.current.temp      },
    humidity:      { now: weather.current.humidity,  forecast: weather.forecast.at(-1)?.humidity  ?? weather.current.humidity  },
    windspeed:     { now: weather.current.windSpeed, forecast: weather.forecast.at(-1)?.low       ?? weather.current.windSpeed },
    solarradiation:{ now: weather.current.solarRad,  forecast: weather.forecast.at(-1)?.high      ?? 0                         },
    conditions:    { now: weather.current.condition, forecast: weather.forecast.at(-1)?.condition ?? ''                        },
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
          § 1 — SUMMARY SNAPSHOT
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Activity}
          eyebrow="Real-time Overview"
          title="Summary Snapshot"
          subtitle="Current operational status across all subsystems"
        />

        {/* Crop health */}
        <div className="mb-5">
          <StatusCard
            label="Crop Health"
            status={health.status}
            detail={health.detail}
            meta={health.scannedAgo}
            variant="crop"
          />
        </div>

        {/* Metric chips — all MPC GreenhouseState variables */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
          {summaryMetrics.map((m) => (
            <MetricChip key={m.label} icon={m.icon} label={m.label} value={m.value} unit={m.unit} note={m.note} />
          ))}
        </div>

        {/* Actuator chips — all 7 MPC ActuatorState variables */}
        <div className="flex flex-wrap gap-2">
          {actuators.map((a) => (
            <ActuatorChip key={a.label} {...a} />
          ))}
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 2 — GROWTH INTELLIGENCE
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Leaf}
          eyebrow="Growth Analysis"
          title="Growth Intelligence"
          subtitle="Stage progression, transition probability, and historical comparison"
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
                  Next Stage
                </p>
                <p className="text-xl font-headline font-bold text-on-surface leading-none">{crop.next}</p>
                <p className="text-[9px] text-on-surface-variant opacity-70 mt-0.5">
                  in {crop.nextInDays} days
                </p>
              </div>
            </div>
            <div className="px-2">
              <CropStageTrack
                stages={crop.stages}
                currentIndex={crop.currentIndex}
                currentPct={crop.currentPct}
              />
            </div>
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
          </div>
        </div>

        {/* Hours to next stage + Transition probability */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-5">
          <GrowthSpotlight
            label="Estimated Time to Next Stage"
            value={growthIntel.hoursToNextStage}
            unit="hrs"
            days={Math.ceil(growthIntel.hoursToNextStage / 24)}
            accent="primary"
          />
          <GrowthSpotlight
            label="Transition Probability (24hr)"
            value={`${growthIntel.transitionProb24h}%`}
            unit="likelihood"
            accent="secondary"
          />
        </div>

        {/* Stage history comparison */}
        <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10 mb-5">
          <div className="flex items-center justify-between mb-4">
            <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
              Stage History
            </p>
            <span className="text-[9px] text-on-surface-variant opacity-50">
              Duration vs. target · days
            </span>
          </div>
          <div className="flex flex-col gap-3.5">
            {growthIntel.stageHistory.map((row) => (
              <StageHistoryRow key={row.stage} {...row} />
            ))}
          </div>
          <p className="mt-4 text-[8px] text-on-surface-variant opacity-50 leading-relaxed">
            Green bars indicate on-schedule completion. Orange indicates overrun vs. target duration. Clock icon = in progress.
          </p>
        </div>

        {/* Recent crop image gallery — rolling 30-min captures, max 5 */}
        <div className="bg-surface-low rounded-xl p-5 border border-outline-variant/10">
          <div className="flex items-center gap-2 mb-4">
            <Camera size={13} className="text-primary" />
            <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
              Recent Crop Images
            </p>
            <span className="ml-auto text-[9px] text-on-surface-variant opacity-50">5 captures</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {stageImages.map((img) => (
              <GalleryFrame
                key={img.capturedAgo}
                src={img.src}
                alt={img.alt}
                label={img.stage}
                badge={img.capturedAgo}
                sub={`${img.confidence}% conf.`}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 3 — DISEASE INTELLIGENCE
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Leaf}
          eyebrow="Pathogen Monitoring"
          title="Disease Intelligence"
          subtitle="24-hour risk prediction across all monitored pathogens · current leaf scan"
        />

        {/* Disease risk cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3 mb-5">
          {diseaseRisks.map((d) => (
            <DiseaseRiskCard key={d.name} {...d} />
          ))}
        </div>

        {/* Recent leaf scan gallery — rolling 30-min captures, max 5 */}
        <div className="bg-surface-low rounded-xl p-5 border border-outline-variant/10">
          <div className="flex items-center gap-2 mb-4">
            <ScanSearch size={13} className="text-primary" />
            <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
              Recent Leaf Scans
            </p>
            <span className="ml-auto text-[9px] text-on-surface-variant opacity-50">5 captures</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {leafImages.map((img) => (
              <GalleryFrame
                key={img.capturedAgo}
                src={img.src}
                alt={img.alt}
                label={img.classification}
                badge={img.capturedAgo}
                sub={`${img.confidence}% conf.`}
              />
            ))}
          </div>
        </div>
      </section>

      {/* ════════════════════════════════════════════════════════════════
          § 4 — OUTDOOR WEATHER
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
          § 5 — INDOOR CONDITIONS
      ════════════════════════════════════════════════════════════════ */}
      <section>
        <SectionHeader
          icon={Thermometer}
          eyebrow="Greenhouse Sensor Array"
          title="Indoor Conditions"
          subtitle="Live readings from greenhouse sensors"
        />
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
          {INDOOR_SENSORS.map((s) => (
            <SensorCard key={s.label} {...s} />
          ))}
        </div>
      </section>

    </div>
  );
}

export default DetailedInsights;
