import { useState, useEffect } from 'react';
import {
  Wind,
  Droplets,
  Thermometer,
  Sun,
  AlertTriangle,
  SlidersHorizontal,
  RotateCcw,
  CheckCircle2,
  Fan,
  Zap,
  Clock,
  Calendar,
  Sprout,
  FlaskConical,
  Activity,
  Radio,
  RadioTower,
  ChevronDown,
} from 'lucide-react';

/* -----------------------------------------------------------------
   CONSTANTS
----------------------------------------------------------------- */

// Canonical stages — matches DetailedInsights / constants.py GROWTH_STAGES
const GROWTH_STAGES = ['Seedling', 'Early Vegetative', 'Flowering Initiation', 'Flowering', 'Unripe', 'Ripe'];

// Max days per stage — from TOMATO_GROWTH_STAGE_CLASSIFICATION.md
const STAGE_MAX_DAYS = {
  'Seedling':              14,
  'Early Vegetative':      21,
  'Flowering Initiation':  15,
  'Flowering':             15,
  'Unripe':                14,
  'Ripe':                  14,
};

// Actuators — matches DetailedInsights / MPC ActuatorState (constants.py CONTROL_VARIABLES)
const ACTUATOR_DEFS = [
  {
    id: 'fan',
    icon: Fan,
    label: 'Fan Speed',
    accent: { text: 'text-secondary', bg: 'bg-secondary/10', ring: 'ring-secondary/20' },
  },
  {
    id: 'vent',
    icon: Wind,
    label: 'Vent Opening',
    accent: { text: 'text-secondary', bg: 'bg-secondary/10', ring: 'ring-secondary/20' },
  },
  {
    id: 'irrigation',
    icon: Droplets,
    label: 'Irrigation',
    accent: { text: 'text-primary', bg: 'bg-primary/10', ring: 'ring-primary/20' },
  },
  {
    id: 'heater',
    icon: Thermometer,
    label: 'Heater',
    accent: { text: 'text-danger', bg: 'bg-danger/10', ring: 'ring-danger/20' },
  },
  {
    id: 'led',
    icon: Sun,
    label: 'LED Intensity',
    accent: { text: 'text-warning', bg: 'bg-warning/10', ring: 'ring-warning/20' },
  },
  {
    id: 'co2',
    icon: FlaskConical,
    label: 'CO\u2082 Valve',
    accent: { text: 'text-secondary', bg: 'bg-secondary/10', ring: 'ring-secondary/20' },
  },
  {
    id: 'fogger',
    icon: Droplets,
    label: 'Fogger',
    accent: { text: 'text-primary', bg: 'bg-primary/10', ring: 'ring-primary/20' },
  },
];

const INITIAL_ACTUATORS = {
  fan:       { active: true,  level: 75 },
  vent:      { active: true,  level: 45 },
  irrigation:{ active: true,  level: 45 },
  heater:    { active: true,  level: 60 },
  led:       { active: false, level: 0  },
  co2:       { active: true,  level: 55 },
  fogger:    { active: false, level: 0  },
};

const HOURS = Array.from({ length: 24 }, (_, i) => i);

/* -----------------------------------------------------------------
   HELPERS
----------------------------------------------------------------- */

function getLiveDate() {
  const now = new Date();
  return now.toISOString().slice(0, 10);
}
function getLiveHour() {
  return new Date().getHours();
}

/* -----------------------------------------------------------------
   SUB-COMPONENTS
----------------------------------------------------------------- */

function Toggle({ active, onToggle, disabled = false }) {
  return (
    <button
      type="button"
      onClick={onToggle}
      disabled={disabled}
      aria-pressed={active}
      aria-label={active ? 'Deactivate' : 'Activate'}
      className={`relative w-11 h-6 rounded-full transition-all duration-300 shrink-0
        ${active ? 'bg-primary' : 'bg-surface-highest'}
        ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer'}
      `}
    >
      <span
        className={`absolute top-1 w-4 h-4 rounded-full bg-white shadow transition-all duration-300
          ${active ? 'left-6' : 'left-1'}
        `}
      />
    </button>
  );
}

function ModePill({ mode, setMode }) {
  return (
    <div className="inline-flex rounded-full bg-surface-highest border border-outline-variant/20 p-0.5">
      {['live', 'override'].map((m) => (
        <button
          key={m}
          type="button"
          onClick={() => setMode(m)}
          className={`px-5 py-1.5 rounded-full text-[10px] font-bold uppercase tracking-widest transition-all duration-200
            ${mode === m
              ? m === 'live'
                ? 'bg-primary text-on-primary shadow'
                : 'bg-warning text-black shadow'
              : 'text-on-surface-variant hover:text-on-surface'
            }`}
        >
          {m === 'live' ? 'Live / Current' : 'Manual Override'}
        </button>
      ))}
    </div>
  );
}

function Field({ label, hint, children }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">
        {label}
      </label>
      {children}
      {hint && <p className="text-[8px] text-on-surface-variant opacity-50 leading-snug">{hint}</p>}
    </div>
  );
}

function Select({ value, onChange, disabled, children }) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={`w-full appearance-none bg-surface-high text-on-surface text-sm font-medium
          border border-outline-variant/20 rounded-xl px-4 py-3 pr-10
          focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/40
          transition-all duration-200
          ${disabled ? 'opacity-40 cursor-not-allowed' : 'cursor-pointer hover:border-outline-variant/40'}
        `}
      >
        {children}
      </select>
      <ChevronDown
        size={14}
        className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-on-surface-variant"
      />
    </div>
  );
}

function NumberInput({ value, onChange, min, max, disabled, suffix }) {
  return (
    <div className="relative flex items-center">
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Number(e.target.value))}
        disabled={disabled}
        className={`w-full bg-surface-high text-on-surface text-sm font-medium
          border border-outline-variant/20 rounded-xl px-4 py-3
          ${suffix ? 'pr-14' : ''}
          focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/40
          transition-all duration-200
          ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
        `}
      />
      {suffix && (
        <span className="pointer-events-none absolute right-4 text-[10px] text-on-surface-variant opacity-60">
          {suffix}
        </span>
      )}
    </div>
  );
}

function DateInput({ value, onChange, disabled }) {
  return (
    <input
      type="date"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      disabled={disabled}
      className={`w-full bg-surface-high text-on-surface text-sm font-medium
        border border-outline-variant/20 rounded-xl px-4 py-3
        focus:outline-none focus:ring-2 focus:ring-primary/30 focus:border-primary/40
        transition-all duration-200
        ${disabled ? 'opacity-40 cursor-not-allowed' : ''}
      `}
    />
  );
}

function HourPicker({ value, onChange, disabled }) {
  return (
    <div className={`grid grid-cols-6 gap-1 ${disabled ? 'opacity-40 pointer-events-none' : ''}`}>
      {HOURS.map((h) => (
        <button
          key={h}
          type="button"
          onClick={() => onChange(h)}
          disabled={disabled}
          className={`py-1.5 rounded-lg text-[10px] font-bold transition-all duration-150
            ${value === h
              ? 'bg-primary text-on-primary shadow'
              : 'bg-surface-high text-on-surface-variant hover:bg-surface-highest hover:text-on-surface'
            }`}
        >
          {String(h).padStart(2, '0')}
        </button>
      ))}
    </div>
  );
}

/** Single actuator card - on/off toggle only. */
function ActuatorCard({ def, state, onToggle, disabled }) {
  const { icon: Icon, label, accent } = def;
  const { active } = state;
  const isOff = !active || disabled;

  return (
    <div
      className={`bg-surface-high rounded-xl p-5 border border-outline-variant/10 flex items-center justify-between gap-4 transition-all duration-300
        ${isOff ? 'opacity-55' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <div className={`p-2.5 rounded-xl ${accent.bg}`}>
          <Icon size={16} className={accent.text} />
        </div>
        <div>
          <p className="text-sm font-headline font-bold text-on-surface leading-tight">{label}</p>
          <div className="flex items-center gap-1.5 mt-0.5">
            <span className={`w-1.5 h-1.5 rounded-full ${active && !disabled ? 'bg-primary animate-pulse' : 'bg-surface-highest'}`} />
            <span className="text-[9px] text-on-surface-variant">
              {disabled ? 'Locked - live mode' : active ? 'Active' : 'Inactive'}
            </span>
          </div>
        </div>
      </div>
      <Toggle active={active} onToggle={onToggle} disabled={disabled} />
    </div>
  );
}

/* -----------------------------------------------------------------
   PAGE
----------------------------------------------------------------- */
function ManualOverride() {
  const [mode, setMode] = useState('live');
  const isOverride = mode === 'override';

  const [liveDate, setLiveDate] = useState(getLiveDate);
  const [liveHour, setLiveHour] = useState(getLiveHour);
  useEffect(() => {
    const id = setInterval(() => {
      setLiveDate(getLiveDate());
      setLiveHour(getLiveHour());
    }, 60_000);
    return () => clearInterval(id);
  }, []);

  const [overrideDate, setOverrideDate] = useState(getLiveDate);
  const [overrideHour, setOverrideHour] = useState(getLiveHour);
  const [stage, setStage]               = useState('Flowering');
  const [daysInStage, setDaysInStage]   = useState(12);

  const [actuators, setActuators] = useState(INITIAL_ACTUATORS);
  const [applied, setApplied]     = useState(false);
  const [pendingReset, setPendingReset] = useState(false);

  function toggleActuator(id) {
    setActuators((prev) => ({ ...prev, [id]: { ...prev[id], active: !prev[id].active } }));
    setApplied(false);
  }
  function handleApply() { setApplied(true); }
  function handleReset() {
    const allOff = Object.fromEntries(
      Object.keys(INITIAL_ACTUATORS).map((id) => [id, { ...INITIAL_ACTUATORS[id], active: false }])
    );
    setActuators(allOff);
    setApplied(false);
    setPendingReset(false);
  }

  const displayDate  = isOverride ? overrideDate : liveDate;
  const displayHour  = isOverride ? overrideHour : liveHour;
  const displayStage = isOverride ? stage : 'Flowering';
  const maxDays      = STAGE_MAX_DAYS[stage] ?? 30;

  return (
    <div className="py-6 flex flex-col gap-8">

      {/* Page header */}
      <header className="flex flex-col xl:flex-row xl:items-end justify-between gap-5">
        <h1 className="text-5xl font-headline font-bold tracking-tighter">
          Manual{' '}
          <span className={isOverride ? 'text-warning italic' : 'text-primary italic'}>Override</span>
        </h1>
        <div className="shrink-0">
          <ModePill mode={mode} setMode={(m) => { setMode(m); setApplied(false); }} />
        </div>
      </header>

      {/* Mode status banner */}
      {isOverride ? (
        <div className="flex items-start gap-4 bg-warning/8 border border-warning/20 rounded-xl px-6 py-4">
          <AlertTriangle size={18} className="text-warning shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-bold text-warning uppercase tracking-widest">Manual Override Active</p>
            <p className="text-xs text-on-surface-variant mt-0.5 leading-relaxed">
              Simulation inputs are now editable. Actuator states below are fully controllable.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-4 bg-primary/8 border border-primary/15 rounded-xl px-6 py-4">
          <RadioTower size={18} className="text-primary shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-bold text-primary uppercase tracking-widest">Live Mode - Read Only</p>
            <p className="text-xs text-on-surface-variant mt-0.5 leading-relaxed">
              Displaying current real-time parameters. Switch to{' '}
              <strong className="text-on-surface font-medium">Manual Override</strong> to edit simulation inputs
              and control actuators directly.
            </p>
          </div>
        </div>
      )}

      {/* SECTION A - SIMULATION PARAMETERS */}
      <section>
        <div className="flex items-center gap-3 mb-5">
          <div className="p-2 rounded-xl bg-surface-high border border-outline-variant/15">
            <Activity size={14} className="text-on-surface-variant" />
          </div>
          <div>
            <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">Section A</p>
            <h2 className="text-xl font-headline font-bold text-on-surface leading-none">Simulation Parameters</h2>
          </div>
          {!isOverride && (
            <span className="ml-auto text-[9px] font-bold uppercase tracking-widest px-3 py-1 rounded-full bg-surface-high border border-outline-variant/20 text-on-surface-variant">
              Locked &middot; Live
            </span>
          )}
        </div>

        <div className="bg-surface-low rounded-xl border border-outline-variant/10 divide-y divide-outline-variant/10">

          {/* Date + Hour */}
          <div className="p-6">
            <div className="flex items-center gap-2 mb-5">
              <Calendar size={13} className="text-on-surface-variant" />
              <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface">Date &amp; Time</p>
              <span className="ml-2 text-[9px] text-on-surface-variant opacity-50">
                {isOverride
                  ? 'Set the simulation reference date and hour'
                  : `Live - ${new Date().toLocaleDateString('en-GB', { weekday:'long', day:'numeric', month:'long', year:'numeric' })}`}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Field
                label="Simulation Date"
                hint={isOverride ? 'YYYY-MM-DD - used as the reference date for DT loop injection.' : 'Read from system clock - updates every minute.'}
              >
                <DateInput value={displayDate} onChange={setOverrideDate} disabled={!isOverride} />
              </Field>

              <Field
                label="Hour of Day (0 - 23)"
                hint={isOverride ? 'Select the hour of day for this simulation run. Minutes are not configurable.' : `Current hour: ${String(displayHour).padStart(2,'0')}:00`}
              >
                <div className="bg-surface-high rounded-xl p-3 border border-outline-variant/20">
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <Clock size={12} className="text-on-surface-variant" />
                      <span className="text-[9px] uppercase tracking-widest text-on-surface-variant">Selected hour</span>
                    </div>
                    <span className="text-xl font-headline font-bold text-primary">
                      {String(displayHour).padStart(2, '0')}:00
                    </span>
                  </div>
                  <HourPicker value={displayHour} onChange={setOverrideHour} disabled={!isOverride} />
                </div>
              </Field>
            </div>
          </div>

          {/* Growth Stage */}
          <div className="p-6">
            <div className="flex items-center gap-2 mb-5">
              <Sprout size={13} className="text-on-surface-variant" />
              <p className="text-[10px] font-bold uppercase tracking-widest text-on-surface">Crop Stage</p>
              <span className="ml-2 text-[9px] text-on-surface-variant opacity-50">
                {isOverride ? 'Inject a specific growth stage into the DT' : 'Derived from live DT state'}
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Field
                label="Growth Stage"
                hint={isOverride ? 'Select the stage to inject - overrides crop model output.' : 'Automatically tracked by the digital twin.'}
              >
                {isOverride ? (
                  <Select value={stage} onChange={setStage} disabled={false}>
                    {GROWTH_STAGES.map((s) => <option key={s} value={s}>{s}</option>)}
                  </Select>
                ) : (
                  <div className="bg-surface-high rounded-xl px-4 py-3 border border-outline-variant/20 text-sm font-medium text-on-surface opacity-60 cursor-not-allowed">
                    {displayStage}
                  </div>
                )}
              </Field>

              <Field
                label="Days in Current Stage"
                hint={`0 - ${maxDays} days - Max for ${isOverride ? stage : displayStage}.`}
              >
                <NumberInput
                  value={isOverride ? daysInStage : 12}
                  onChange={(v) => setDaysInStage(Math.min(maxDays, Math.max(0, v)))}
                  min={0}
                  max={maxDays}
                  disabled={!isOverride}
                  suffix="days"
                />
                {isOverride && (
                  <input
                    type="range"
                    min={0}
                    max={maxDays}
                    value={daysInStage}
                    onChange={(e) => setDaysInStage(Number(e.target.value))}
                    className="w-full h-1 rounded-full appearance-none bg-surface-highest mt-1 cursor-pointer"
                  />
                )}
              </Field>
            </div>

            {isOverride && (
              <div className="mt-4 flex flex-wrap gap-2">
                {GROWTH_STAGES.map((s, i) => {
                  const stageIdx = GROWTH_STAGES.indexOf(stage);
                  const done    = i < stageIdx;
                  const current = i === stageIdx;
                  return (
                    <button
                      key={s}
                      type="button"
                      onClick={() => { setStage(s); setDaysInStage(0); }}
                      className={`px-3 py-1.5 rounded-full text-[9px] font-bold uppercase tracking-widest transition-all duration-200
                        ${current
                          ? 'bg-primary text-on-primary shadow'
                          : done
                          ? 'bg-surface-highest text-on-surface-variant border border-outline-variant/30 line-through opacity-50'
                          : 'bg-surface-high text-on-surface-variant border border-outline-variant/20 hover:border-primary/30 hover:text-on-surface'
                        }`}
                    >
                      {s}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Parameter summary */}
          <div className="px-6 py-4 bg-surface-deep rounded-b-xl">
            <div className="flex flex-wrap gap-5 items-center">
              <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">
                {isOverride ? 'Override Preview' : 'Live Reading'}
              </p>
              <div className="flex flex-wrap gap-3">
                {[
                  { icon: Calendar, label: 'Date',  value: displayDate },
                  { icon: Clock,    label: 'Hour',  value: `${String(displayHour).padStart(2,'0')}:00` },
                  { icon: Sprout,   label: 'Stage', value: isOverride ? stage : displayStage },
                  { icon: Activity, label: 'Days',  value: `${isOverride ? daysInStage : 12} days` },
                ].map(({ icon: I, label, value }) => (
                  <div key={label} className="flex items-center gap-1.5 bg-surface-high rounded-full px-3 py-1.5 border border-outline-variant/15">
                    <I size={10} className="text-on-surface-variant" />
                    <span className="text-[9px] text-on-surface-variant">{label}:</span>
                    <span className="text-[9px] font-bold text-on-surface">{value}</span>
                  </div>
                ))}
                {isOverride && (
                  <span className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-widest text-warning">
                    <Zap size={10} />
                    Override active
                  </span>
                )}
                {!isOverride && (
                  <span className="flex items-center gap-1.5 text-[9px] font-bold uppercase tracking-widest text-primary">
                    <Radio size={10} />
                    Live sync
                  </span>
                )}
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* SECTION B - ACTUATOR CONTROLS */}
      <section>
        <div className="flex items-center gap-3 mb-5">
          <div className="p-2 rounded-xl bg-surface-high border border-outline-variant/15">
            <SlidersHorizontal size={14} className="text-on-surface-variant" />
          </div>
          <div>
            <p className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">Section B</p>
            <h2 className="text-xl font-headline font-bold text-on-surface leading-none">Actuator Controls</h2>
          </div>
          {!isOverride && (
            <span className="ml-auto text-[9px] font-bold uppercase tracking-widest px-3 py-1 rounded-full bg-surface-high border border-outline-variant/20 text-on-surface-variant">
              Locked &middot; Live
            </span>
          )}
          {isOverride && (
            <span className="ml-auto text-[9px] font-bold uppercase tracking-widest px-3 py-1 rounded-full bg-warning/10 border border-warning/20 text-warning">
              Editable
            </span>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4 mb-5">
          {ACTUATOR_DEFS.map((def) => (
            <ActuatorCard
              key={def.id}
              def={def}
              state={actuators[def.id]}
              onToggle={() => toggleActuator(def.id)}
              disabled={!isOverride}
            />
          ))}
        </div>

        {/* Apply / Reset action bar */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 glass-panel rounded-xl px-6 py-5">
          <div className="text-sm text-on-surface-variant">
            {!isOverride ? (
              <span className="flex items-center gap-2 opacity-50">
                <Clock size={14} />
                Switch to Manual Override to control actuators.
              </span>
            ) : applied ? (
              <span className="flex items-center gap-2 text-primary font-medium">
                <CheckCircle2 size={15} />
                Settings applied.
              </span>
            ) : (
              <span className="opacity-60">Unsaved changes - apply to confirm.</span>
            )}
          </div>
          <div className="flex gap-3">
            <button
              type="button"
              onClick={() => setPendingReset(true)}
              disabled={!isOverride}
              className="flex items-center gap-2 bg-surface-high border border-outline-variant/20 text-on-surface-variant font-headline font-bold text-sm px-5 py-2.5 rounded-full transition-all hover:bg-surface-highest hover:text-on-surface active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <RotateCcw size={14} />
              Reset
            </button>
            <button
              type="button"
              onClick={handleApply}
              disabled={!isOverride || applied}
              className={`font-headline font-bold text-sm px-6 py-2.5 rounded-full transition-all hover:opacity-90 active:scale-95 disabled:opacity-40 disabled:cursor-not-allowed ${isOverride ? 'bg-warning text-black' : 'bg-primary text-on-primary'}`}
            >
              Apply All Changes
            </button>
          </div>
        </div>

        {/* Reset confirmation */}
        {pendingReset && (
          <div className="mt-3 flex items-center justify-between gap-4 bg-danger/8 border border-danger/20 rounded-xl px-5 py-4">
            <p className="text-sm text-on-surface">
              Turn off all actuators and reset to defaults?
            </p>
            <div className="flex gap-2 shrink-0">
              <button
                type="button"
                onClick={() => setPendingReset(false)}
                className="px-4 py-1.5 rounded-full text-xs font-bold bg-surface-high border border-outline-variant/20 text-on-surface-variant hover:bg-surface-highest"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={handleReset}
                className="px-4 py-1.5 rounded-full text-xs font-bold bg-danger text-white hover:opacity-90"
              >
                Confirm Reset
              </button>
            </div>
          </div>
        )}
      </section>

    </div>
  );
}

export default ManualOverride;
