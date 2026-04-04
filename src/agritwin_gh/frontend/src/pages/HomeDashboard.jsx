import {
  Thermometer,
  Droplets,
  Wind,
  Sun,
  ArrowUp,
  ArrowDown,
  TrendingUp,
  ArrowRight,
  BarChart2,
  Sliders,
  Fan,
  CheckCircle2,
  Camera,
  ScanSearch,
  Box,
  ExternalLink,
} from 'lucide-react';

import PanelCard   from '../components/ui/PanelCard';
import StatusCard  from '../components/ui/StatusCard';

/* ══════════════════════════════════════════════════════════════════════════
   MOCK DATA
   All values are static. Replace each constant with an API call later.
══════════════════════════════════════════════════════════════════════════ */

// ── 1 & 2. Crop stage data ──────────────────────────────────────────────
const CROP = {
  current:       'Flowering',
  currentIndex:  3,
  currentPct:    80,
  daysInStage:   12,
  stageDuration: 15,
  next:          'Unripe',
  nextInDays:    5,
  stages: ['Seedling', 'Early Veg.', 'Flw. Init.', 'Flowering', 'Unripe', 'Ripe'],
};

// ── 6. Crop health indicator ─────────────────────────────────────────────
// status: 'Healthy' | 'Warning' | 'Risk'
const CROP_HEALTH = {
  status:     'Healthy',
  detail:     'No anomalies detected. Leaf scan returned clear.',
  scannedAgo: '2 mins ago',
};

// ── 7. Weather / external conditions ─────────────────────────────────────
// status: 'Healthy' | 'Warning' | 'Risk'
const WEATHER = {
  status:    'Warning',
  condition: 'High External Humidity',
  detail:    'Ext. humidity at 89%. Risk of condensation — ventilation recommended.',
  forecast:  'Expected to clear in ~3 hrs',
};

// ── 3. Actuator state — from MPC ActuatorState (constants.py CONTROL_VARIABLES) ────
const ACTUATORS = [
  { icon: Fan,         label: 'Fan Speed',    status: 'ON',  active: true,  color: 'primary'   },
  { icon: Wind,        label: 'Vent Opening', status: 'ON',  active: true,  color: 'primary'   },
  { icon: Droplets,    label: 'Irrigation',   status: 'ON',  active: true,  color: 'secondary' },
  { icon: Thermometer, label: 'Heater',       status: 'ON',  active: true,  color: 'warning'   },
  { icon: Sun,         label: 'LED Intensity',status: 'OFF', active: false, color: 'neutral'   },
  { icon: Wind,        label: 'CO₂ Valve',    status: 'ON',  active: true,  color: 'warning'   },
  { icon: Droplets,    label: 'Fogger',       status: 'OFF', active: false, color: 'neutral'   },
];

// ── 4. Monthly resource usage ─────────────────────────────────────────────
const MONTHLY_RESOURCES = [
  { label: 'Water',  used: 1840, unit: 'L'   },
  { label: 'Energy', used: 312,  unit: 'kWh' },
];

// ── 5. Monthly resource cost ──────────────────────────────────────────────
const MONTHLY_COST = {
  month:  'April 2026',
  energy: 43.68,
  water:  7.36,
  total:  51.04,   // kept explicit so it can override computed value if needed
};

// ── 6. Crop image frames ──────────────────────────────────────────────────
const CROP_IMAGES = {
  stage: {
    src:       'https://picsum.photos/seed/tomato-flower/480/280',
    alt:       'Tomato plant at flowering stage',
    badge:     'Day 12',
    location:  'GH-04 · Camera 2',
    captured:  '1 hr ago',
  },
  leaf: {
    src:       'https://picsum.photos/seed/leaf-healthy-scan/480/280',
    alt:       'Leaf scan — GH-04 Sector B',
    badge:     '99.5% Conf.',
    location:  'Sector B · Leaf #7',
    captured:  '2 mins ago',
  },
};

// ── Growth intelligence (used by left panel stage bars) ─────────────────────
// Canonical stages + durations from docs/TOMATO_GROWTH_STAGE_CLASSIFICATION.md
const GROWTH_INTEL = {
  stageHistory: [
    { stage: 'Seedl.',   daysUsed: 14, daysTarget: 14, complete: true  },
    { stage: 'E.Veg.',   daysUsed: 21, daysTarget: 21, complete: true  },
    { stage: 'Fl.Init.', daysUsed: 15, daysTarget: 15, complete: true  },
    { stage: 'Flower',   daysUsed: 12, daysTarget: 15, complete: false },
  ],
};

// ── Left/right panel data (unchanged) ────────────────────────────────────
const METRICS = [
  { icon: Thermometer, label: 'Temperature', value: '24.2', unit: '°C',  trend: 'up',     trendLabel: '+0.4°', trendClass: 'text-warning'            },
  { icon: Droplets,    label: 'Humidity',    value: '65.8', unit: '%',   trend: 'down',   trendLabel: '−1.2%', trendClass: 'text-danger'             },
  { icon: Wind,        label: 'CO₂ Level',   value: '812',  unit: 'ppm', trend: 'stable', trendLabel: 'Stable', trendClass: 'text-on-surface-variant' },
  { icon: Sun,         label: 'Light',       value: '12k',  unit: 'lux', trend: 'up',     trendLabel: 'Peak',   trendClass: 'text-primary'           },
];

const HEALTH_ROWS = [
  { label: 'Sensor Array',  status: 'Nominal', ok: true  },
  { label: 'Network Link',  status: 'Strong',  ok: true  },
  { label: 'Data Pipeline', status: 'Active',  ok: true  },
  { label: 'Calibration',   status: 'Due: 6d', ok: false },
];

/* ── Local helpers ───────────────────────────────────────────────────────── */

/** Thin shimmer-style placeholder for sections awaiting real content. */
function PlaceholderBlock({ label, height = 'h-20' }) {
  return (
    <div
      className={`${height} rounded-lg bg-surface-highest/40 border border-outline-variant/10 flex items-center justify-center`}
    >
      <span className="text-[9px] uppercase tracking-widest text-on-surface-variant opacity-40">
        {label}
      </span>
    </div>
  );
}

function TrendIcon({ trend }) {
  if (trend === 'up')   return <ArrowUp size={11} />;
  if (trend === 'down') return <ArrowDown size={11} />;
  return <TrendingUp size={11} />;
}

function MetricCard({ icon: Icon, label, value, unit, trend, trendLabel, trendClass }) {
  return (
    <div>
      <div className="flex items-center gap-1.5 mb-2 text-on-surface-variant">
        <Icon size={13} />
        <span className="text-[9px] uppercase tracking-widest">{label}</span>
      </div>
      <div className="text-3xl font-headline font-bold leading-none">
        {value}
        <span className="text-sm font-light opacity-50 ml-0.5">{unit}</span>
      </div>
      <div className={`flex items-center gap-1 text-[9px] mt-1.5 ${trendClass}`}>
        <TrendIcon trend={trend} />
        {trendLabel}
      </div>
    </div>
  );
}

/** Single actuator row (used in right panel preview). */
function ActuatorRow({ icon: Icon, label, status, active, color }) {
  const pill = {
    primary:   'text-primary bg-primary/10 border-primary/20',
    secondary: 'text-secondary bg-secondary/10 border-secondary/20',
    warning:   'text-warning bg-warning/10 border-warning/20',
    neutral:   'text-on-surface-variant bg-surface-highest border-outline-variant/20',
  };
  const pillClass = active ? (pill[color] ?? pill.neutral) : pill.neutral;
  const iconClass = active ? (pill[color] ?? pill.neutral) : 'bg-surface-highest text-on-surface-variant';
  return (
    <div className="flex items-center justify-between">
      <div className="flex items-center gap-2.5">
        <span className={`p-1.5 rounded-lg ${iconClass}`}><Icon size={13} /></span>
        <span className="text-xs text-on-surface">{label}</span>
      </div>
      <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${pillClass}`}>
        {status}
      </span>
    </div>
  );
}

/* ── Centre panel sub-components ─────────────────────────────────────────── */

/**
 * CropStageTrack — horizontal stage dot + connecting-line progression track.
 * Filled (primary) for passed stages, pulsing ring for current, muted for future.
 */
function CropStageTrack({ stages, currentIndex, currentPct }) {
  return (
    <div className="relative flex items-start justify-between w-full pt-1">
      {/* Background connector line */}
      <div className="absolute top-3.5 left-4 right-4 h-px bg-outline-variant/20" />

      {/* Filled connector up to current stage */}
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
            {/* Stage dot */}
            <div
              className={[
                'w-7 h-7 rounded-full flex items-center justify-center border-2 transition-all duration-300 shrink-0',
                done    ? 'bg-primary border-primary'
                        : current
                        ? 'bg-primary/15 border-primary ring-4 ring-primary/15'
                        : 'bg-surface-high border-outline-variant/30',
              ].join(' ')}
            >
              {done    && <CheckCircle2 size={12} className="text-on-primary" />}
              {current && <span className="text-[8px] font-black text-primary">{currentPct}%</span>}
            </div>

            {/* Stage label */}
            <span
              className={[
                'text-[8px] text-center leading-tight',
                current ? 'text-primary font-bold'
                        : done
                        ? 'text-on-surface-variant opacity-70'
                        : 'text-on-surface-variant opacity-35',
              ].join(' ')}
            >
              {stage}
            </span>
          </div>
        );
      })}
    </div>
  );
}

/**
 * ActuatorTile — coloured tile card for the 2×2 actuator grid.
 * Background tint + pill colour driven by status/active state.
 */
function ActuatorTile({ icon: Icon, label, status, active, color }) {
  /* All class strings must be literal (no dynamic template literals)
     so Tailwind v4 sees them at build time.                          */
  const TILE = {
    primary:   { root: 'bg-primary/10 border-primary/20',     icon: 'bg-primary/15 text-primary',             pill: 'text-primary'   },
    secondary: { root: 'bg-secondary/10 border-secondary/20', icon: 'bg-secondary/15 text-secondary',         pill: 'text-secondary' },
    warning:   { root: 'bg-warning/10 border-warning/20',     icon: 'bg-warning/15 text-warning',             pill: 'text-warning'   },
    neutral:   { root: 'bg-surface-high border-outline-variant/15', icon: 'bg-surface-highest text-on-surface-variant', pill: 'text-on-surface-variant' },
  };
  const t = active ? (TILE[color] ?? TILE.neutral) : TILE.neutral;
  return (
    <div className={`rounded-xl border p-4 flex flex-col gap-2.5 ${t.root}`}>
      <div className="flex items-center justify-between">
        <span className={`p-1.5 rounded-lg ${t.icon}`}>
          <Icon size={14} />
        </span>
        <span className={`text-[8px] font-black uppercase tracking-widest ${t.pill}`}>
          {status}
        </span>
      </div>
      <span className="text-[10px] font-medium text-on-surface leading-tight">{label}</span>
    </div>
  );
}

/** Single cost line for the monthly cost breakdown. */
function CostRow({ label, amount, emphasis = false }) {
  return (
    <div
      className={[
        'flex items-center justify-between',
        emphasis ? 'pt-3 border-t border-outline-variant/15 mt-1' : '',
      ].join(' ')}
    >
      <span
        className={
          emphasis
            ? 'text-sm font-semibold text-on-surface'
            : 'text-xs text-on-surface-variant'
        }
      >
        {label}
      </span>
      <span
        className={
          emphasis
            ? 'text-2xl font-headline font-bold text-on-surface'
            : 'text-sm font-medium text-on-surface-variant'
        }
      >
        ₹{amount.toFixed(2)}
      </span>
    </div>
  );
}

/* ── Page ────────────────────────────────────────────────────────────────── */

/**
 * HomeDashboard — 3-column layout:
 *   Left  (xl:w-72)  — Detailed Insights preview
 *   Center (flex-1)  — Current Dashboard (primary content)
 *   Right  (xl:w-72) — Manual Override preview
 *
 * Responsive stacking order on mobile: center → left → right.
 * Accepts `navigate` prop from AppRouter for panel CTAs.
 */
function HomeDashboard({ navigate }) {
  return (
    <div className="py-6">

      {/* ── Compact page header ──────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-5xl font-headline font-bold tracking-tighter text-on-surface leading-tight">
            Greenhouse{' '}
            <span className="text-primary italic">Dashboard</span>
          </h1>
        </div>

        {/* Status pills row */}
        <div className="flex items-center gap-3 shrink-0">
          {/* Growth stage */}
          <div className="glass-panel flex items-center gap-4 px-5 py-3 rounded-xl">
            <div className="text-right">
              <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-0.5">
                Growth Stage
              </p>
              <p className="text-lg font-headline font-bold text-primary leading-none">Flowering</p>
            </div>
            <div className="relative w-10 h-10 shrink-0">
              <div className="w-10 h-10 rounded-full border-2 border-primary/20 border-t-primary animate-spin-slow" />
              <span className="absolute inset-0 flex items-center justify-center text-[9px] font-bold">
                95%
              </span>
            </div>
          </div>

          {/* Health badge */}
          <div className="flex items-center gap-2 px-3 py-2 rounded-full bg-primary/10 border border-primary/20">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse" />
            <span className="text-[9px] font-bold uppercase tracking-widest text-primary">Healthy</span>
          </div>
        </div>
      </div>

      {/* ══════════════════════════════════════════════════════════════════
          3-COLUMN LAYOUT
          xl+: left(w-72) | center(flex-1) | right(w-72) side by side
          <xl: stacked — center first (order-1), then left, then right
      ══════════════════════════════════════════════════════════════════ */}
      <div className="flex flex-col xl:flex-row gap-5 items-start">

        {/* ──────────────────────────────────────────────────────────────
            LEFT PANEL — Detailed Insights preview
        ─────────────────────────────────────────────────────────────── */}
        <aside className="w-full xl:w-72 2xl:w-80 shrink-0 order-2 xl:order-1 flex flex-col gap-4">

          {/* Insights overview panel */}
          <PanelCard
            title="Detailed Insights"
            subtitle="Env. trends · growth · risk score"
            icon={BarChart2}
            badge="Preview"
            onNavigate={() => navigate?.('insights')}
            navLabel="Open Full Insights"
          >
            {/* Growth progression — mini stage history bars */}
            <div>
              <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-2">
                Growth Progression
              </p>
              <div className="flex gap-1.5">
                {GROWTH_INTEL.stageHistory.map(({ stage, daysUsed, daysTarget, complete }) => {
                  const pct = Math.round((daysUsed / daysTarget) * 100);
                  return (
                  <div key={stage} className="flex-1 flex flex-col gap-1">
                    <div className="h-12 rounded-lg overflow-hidden bg-surface-highest/40 flex flex-col justify-end">
                      <div
                        className={`w-full rounded-t-sm transition-all duration-500 ${complete ? 'bg-primary' : 'bg-primary/50'}`}
                        style={{ height: `${pct}%` }}
                      />
                    </div>
                    <span className="text-[7px] uppercase tracking-wide text-on-surface-variant/60 text-center leading-tight">
                      {stage}
                    </span>
                  </div>
                  );
                })}
              </div>
            </div>

            {/* Risk score — disease severity chips */}
            <div>
              <p className="text-[9px] uppercase tracking-widest text-on-surface-variant mb-2">
                Disease Risk
              </p>
              <div className="flex flex-col gap-1.5">
                {[
                  { label: 'Late Blight',  sev: 'High',   chipClass: 'bg-danger/10 border-danger/25 text-danger'   },
                  { label: 'Gray Mold',    sev: 'Medium', chipClass: 'bg-warning/10 border-warning/25 text-warning' },
                  { label: 'Early Blight', sev: 'Low',    chipClass: 'bg-primary/10 border-primary/20 text-primary' },
                ].map(({ label, sev, chipClass }) => (
                  <div key={label} className="flex items-center justify-between px-2 py-1 rounded-lg bg-surface-highest/30 border border-outline-variant/10">
                    <span className="text-[9px] text-on-surface-variant">{label}</span>
                    <span className={`text-[8px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded-full border ${chipClass}`}>
                      {sev}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </PanelCard>

        </aside>

        {/* ════════════════════════════════════════════════════════════
            CENTER PANEL — Current Dashboard
            Cards (top → bottom):
              ① Crop Stage Progress (current + next)
              ② Crop Health  |  Weather Status  (2-col)
              ③ Actuator State grid (2×2)
              ④ Monthly Resources  |  Monthly Cost  (2-col)
        ════════════════════════════════════════════════════════════ */}
        <section className="flex-1 min-w-0 order-1 xl:order-2 flex flex-col gap-5">

          {/* ── ① CROP STAGE PROGRESS ─────────────────────────────── */}
          <div className="bg-surface-low rounded-xl p-6 relative overflow-hidden">
            {/* Ambient glow blob */}
            <div className="pointer-events-none absolute top-0 right-0 w-48 h-48 bg-primary/5 rounded-full blur-3xl -mr-16 -mt-16" />

            <div className="relative z-10">

              {/* Header row: current stage name + next stage info */}
              <div className="flex items-start justify-between gap-4 mb-6">
                <div>
                  <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-1">
                    Current Crop Stage
                  </p>
                  <div className="flex items-baseline gap-3">
                    <span className="text-3xl font-headline font-bold text-primary leading-none">
                      {CROP.current}
                    </span>
                    <span className="text-xs font-light text-on-surface-variant">
                      Day {CROP.daysInStage} of {CROP.stageDuration}
                    </span>
                  </div>
                </div>

                {/* Next stage */}
                <div className="shrink-0 text-right">
                  <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-0.5">
                    Next Stage
                  </p>
                  <p className="text-xl font-headline font-bold text-on-surface leading-none">
                    {CROP.next}
                  </p>
                  <p className="text-[9px] text-on-surface-variant opacity-70 mt-0.5">
                    in {CROP.nextInDays} days
                  </p>
                </div>
              </div>

              {/* Stage progress track */}
              <div className="px-2">
                <CropStageTrack
                  stages={CROP.stages}
                  currentIndex={CROP.currentIndex}
                  currentPct={CROP.currentPct}
                />
              </div>

              {/* Stage completion progress bar */}
              <div className="mt-6 pt-4 border-t border-outline-variant/10">
                <div className="flex justify-between text-[9px] uppercase tracking-widest opacity-55 mb-2">
                  <span>Stage completion</span>
                  <span>{CROP.currentPct}%</span>
                </div>
                <div className="h-1.5 bg-surface-highest rounded-full overflow-hidden">
                  <div
                    className="h-full bg-primary rounded-full transition-all duration-700"
                    style={{ width: `${CROP.currentPct}%` }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* ── ② CROP HEALTH ─────────────────────────────────── */}
          <StatusCard
            label="Crop Health"
            status={CROP_HEALTH.status}
            detail={CROP_HEALTH.detail}
            meta={CROP_HEALTH.scannedAgo}
            variant="crop"
          />

          {/* ── ③ ACTUATOR STATE — 2×2 tile grid ─────────────────── */}
          <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                  Actuator State
                </h3>
                <p className="text-[9px] text-on-surface-variant mt-0.5">
                  Current operating conditions · auto-refreshes
                </p>
              </div>
              <span className="text-[9px] font-bold uppercase tracking-widest px-2.5 py-1 rounded-full bg-surface-highest border border-outline-variant/20 text-on-surface-variant">
                Live
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
              {ACTUATORS.map((a) => (
                <ActuatorTile key={a.label} {...a} />
              ))}
            </div>
          </div>

          {/* ── ④ MONTHLY RESOURCES + MONTHLY COST ───────────────── */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">

            {/* 4. Monthly resource usage — consumption summary */}
            <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10 flex flex-col gap-4">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                    Monthly Resources
                  </h3>
                  <p className="text-[9px] text-on-surface-variant mt-0.5">Monthly consumption</p>
                </div>
                <span className="text-[9px] text-on-surface-variant opacity-55 shrink-0">
                  {MONTHLY_COST.month}
                </span>
              </div>

              <div className="flex flex-col gap-3.5">
                {MONTHLY_RESOURCES.map(({ label, used, unit }) => (
                  <div key={label} className="flex items-center justify-between">
                    <span className="text-[9px] uppercase tracking-widest text-on-surface-variant opacity-70">
                      {label}
                    </span>
                    <span className="text-sm font-headline font-bold text-on-surface">
                      {used} <span className="text-xs font-light text-on-surface-variant">{unit}</span>
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* 5. Monthly resource cost — energy + water + total */}
            <div className="bg-surface-high rounded-xl p-5 border border-outline-variant/10 flex flex-col gap-4">
              <div className="flex items-start justify-between">
                <div>
                  <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                    Monthly Cost
                  </h3>
                  <p className="text-[9px] text-on-surface-variant mt-0.5">Operational expenditure</p>
                </div>
                <span className="text-[9px] text-on-surface-variant opacity-55 shrink-0">
                  {MONTHLY_COST.month}
                </span>
              </div>

              <div className="flex flex-col gap-2 flex-1">
                <CostRow label="Energy" amount={MONTHLY_COST.energy} />
                <CostRow label="Water"  amount={MONTHLY_COST.water}  />
                <CostRow label="Total"  amount={MONTHLY_COST.total}  emphasis />
              </div>

              {/* Daily average */}
              <div className="pt-3 border-t border-outline-variant/10 flex items-center justify-between">
                <span className="text-[9px] uppercase tracking-widest text-on-surface-variant opacity-55">
                  Daily avg. cost
                </span>
                <span className="text-sm font-headline font-bold text-on-surface">
                  ₹{(MONTHLY_COST.total / new Date(2026, 3, 0).getDate()).toFixed(2)}
                </span>
              </div>
            </div>

          </div>

          {/* ── ⑤ CROP IMAGE FRAMES ───────────────────────────────── */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">

            {/* Card A — Current Crop Stage Image */}
            <div className="bg-surface-deep rounded-xl overflow-hidden border border-outline-variant/10 flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-outline-variant/10">
                <div className="flex items-center gap-2">
                  <Camera size={13} className="text-primary" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                    Crop Stage Image
                  </span>
                </div>
                <span className="text-[9px] text-on-surface-variant opacity-60">
                  {CROP_IMAGES.stage.captured}
                </span>
              </div>
              <div className="relative overflow-hidden">
                <img
                  src={CROP_IMAGES.stage.src}
                  alt={CROP_IMAGES.stage.alt}
                  className="w-full aspect-video object-cover"
                  loading="lazy"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/70 to-transparent px-4 py-3">
                  <p className="text-[10px] font-bold text-white leading-none">
                    {CROP.current} Stage
                  </p>
                  <p className="text-[8px] text-white/70 mt-0.5">{CROP_IMAGES.stage.location}</p>
                </div>
                <span className="absolute top-3 right-3 text-[8px] font-bold uppercase tracking-widest px-2 py-1 rounded-full bg-primary/20 text-primary border border-primary/25">
                  {CROP_IMAGES.stage.badge}
                </span>
              </div>
            </div>

            {/* Card B — Leaf / Disease Status Image */}
            <div className="bg-surface-deep rounded-xl overflow-hidden border border-outline-variant/10 flex flex-col">
              <div className="flex items-center justify-between px-4 py-3 border-b border-outline-variant/10">
                <div className="flex items-center gap-2">
                  <ScanSearch size={13} className="text-primary" />
                  <span className="text-[10px] font-bold uppercase tracking-widest text-on-surface">
                    Leaf / Disease Scan
                  </span>
                </div>
                <span className="text-[9px] text-on-surface-variant opacity-60">
                  {CROP_IMAGES.leaf.captured}
                </span>
              </div>
              <div className="relative overflow-hidden">
                <img
                  src={CROP_IMAGES.leaf.src}
                  alt={CROP_IMAGES.leaf.alt}
                  className="w-full aspect-video object-cover"
                  loading="lazy"
                />
                <div className="absolute bottom-0 left-0 right-0 bg-linear-to-t from-black/70 to-transparent px-4 py-3">
                  <p className="text-[10px] font-bold text-white leading-none">No Anomalies Detected</p>
                  <p className="text-[8px] text-white/70 mt-0.5">{CROP_IMAGES.leaf.location}</p>
                </div>
                <span className="absolute top-3 right-3 text-[8px] font-bold uppercase tracking-widest px-2 py-1 rounded-full bg-primary/20 text-primary border border-primary/25">
                  {CROP_IMAGES.leaf.badge}
                </span>
              </div>
            </div>

          </div>

        </section>

        {/* ──────────────────────────────────────────────────────────────
            RIGHT PANEL — Manual Override preview
        ─────────────────────────────────────────────────────────────── */}
        <aside className="w-full xl:w-72 2xl:w-80 shrink-0 order-3 flex flex-col gap-4">

          {/* Actuator status panel */}
          <PanelCard
            title="Manual Override"
            subtitle="Actuator status · controls"
            icon={Sliders}
            badge="Configure"
            onNavigate={() => navigate?.('override')}
            navLabel="Open Full Config"
          >
            <div className="flex flex-col gap-3">
              {ACTUATORS.map((a) => (
                <ActuatorRow key={a.label} {...a} />
              ))}
            </div>

            </PanelCard>

        </aside>

      </div>

      {/* ══════════════════════════════════════════════════════════════════
          3D GREENHOUSE ENVIRONMENT — full-width entry card
      ══════════════════════════════════════════════════════════════════ */}
      <div className="mt-5">
        <div
          role="button"
          tabIndex={0}
          className="group relative overflow-hidden cursor-pointer rounded-xl border border-outline-variant/15 bg-surface-deep hover:border-primary/30 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:ring-offset-2 focus-visible:ring-offset-background transition-all duration-300"
          onClick={() => window.open('/greenhouse-3d', '_blank')}
          onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && window.open('/greenhouse-3d', '_blank')}
        >
          {/* Hover glow overlay */}
          <div className="pointer-events-none absolute inset-0 bg-linear-to-br from-primary/5 via-transparent to-primary/3 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

          {/* Wireframe grid SVG decoration */}
          <svg
            className="pointer-events-none absolute inset-0 w-full h-full text-on-surface-variant opacity-8 group-hover:opacity-15 transition-opacity duration-500"
            viewBox="0 0 900 180"
            preserveAspectRatio="xMidYMid slice"
            aria-hidden="true"
          >
            {/* Converging vertical perspective lines */}
            {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => (
              <line key={`v${i}`} x1={50 + i * 80} y1={0} x2={450 + (i - 5) * 30} y2={180} stroke="currentColor" strokeWidth="0.6" />
            ))}
            {/* Horizontal receding lines */}
            {[0, 1, 2, 3, 4].map((i) => (
              <line key={`h${i}`} x1={50 - i * 10} y1={i * 40} x2={850 + i * 10} y2={i * 40} stroke="currentColor" strokeWidth="0.5" />
            ))}
            {/* Schematic greenhouse box */}
            <rect x="380" y="25" width="140" height="100" rx="3" fill="none" stroke="currentColor" strokeWidth="1.2" />
            <line x1="380" y1="25" x2="350" y2="5"  stroke="currentColor" strokeWidth="0.9" />
            <line x1="520" y1="25" x2="550" y2="5"  stroke="currentColor" strokeWidth="0.9" />
            <line x1="350" y1="5"  x2="550" y2="5"  stroke="currentColor" strokeWidth="0.9" />
            <line x1="350" y1="5"  x2="350" y2="105" stroke="currentColor" strokeWidth="0.9" />
            <line x1="550" y1="5"  x2="550" y2="105" stroke="currentColor" strokeWidth="0.9" />
            <line x1="350" y1="105" x2="380" y2="125" stroke="currentColor" strokeWidth="0.9" />
            <line x1="550" y1="105" x2="520" y2="125" stroke="currentColor" strokeWidth="0.9" />
            {/* Roof ridge line */}
            <line x1="450" y1="5"   x2="450" y2="25"  stroke="currentColor" strokeWidth="0.6" strokeDasharray="3 3" />
            {/* Cross-hatch on top face */}
            <line x1="350" y1="5"  x2="520" y2="25"  stroke="currentColor" strokeWidth="0.4" strokeDasharray="4 4" />
            <line x1="550" y1="5"  x2="380" y2="25"  stroke="currentColor" strokeWidth="0.4" strokeDasharray="4 4" />
          </svg>

          {/* Content row */}
          <div className="relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5 px-8 py-7">
            <div className="flex items-center gap-5">
              {/* Icon badge */}
              <div className="shrink-0 p-4 rounded-2xl bg-primary/10 border border-primary/20 text-primary group-hover:bg-primary/15 transition-colors duration-300">
                <Box size={26} />
              </div>

              {/* Label + description */}
              <div>
                <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant mb-1">
                  Immersive Simulation
                </p>
                <h2 className="text-xl font-headline font-bold text-on-surface leading-tight">
                  3D Greenhouse Environment
                </h2>
                <p className="text-xs text-on-surface-variant mt-1 max-w-lg leading-relaxed">
                  Explore your digital twin in a real-time 3D environment. Monitor sensors, inspect
                  actuators, and walk through the virtual greenhouse spatially.
                </p>
              </div>
            </div>

            {/* Launch CTA */}
            <div className="shrink-0 flex items-center gap-2 text-[9px] font-bold uppercase tracking-widest px-5 py-2.5 rounded-full border border-primary/25 bg-primary/10 text-primary group-hover:bg-primary/20 group-hover:border-primary/40 transition-all duration-300">
              Launch Environment
              <ExternalLink size={11} />
            </div>
          </div>

          {/* Bottom shimmer accent line */}
          <div className="h-px bg-linear-to-r from-primary/0 via-primary/35 to-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
        </div>
      </div>

    </div>
  );
}

export default HomeDashboard;
