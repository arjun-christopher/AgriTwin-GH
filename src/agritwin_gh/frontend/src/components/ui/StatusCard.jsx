import { ShieldCheck, AlertTriangle, ShieldX, CloudSun, CloudRain, Zap } from 'lucide-react';

/**
 * StatusCard — high-visibility Healthy / Warning / Risk status tile.
 *
 * Props:
 *   label      {string}      Section label (e.g. "Crop Health", "Weather Status")
 *   status     {string}      'Healthy' | 'Warning' | 'Risk'
 *   condition  {string}      Short condition summary line (optional)
 *   detail     {string}      One-sentence description (optional)
 *   meta       {string}      Footer meta text, e.g. confidence or forecast (optional)
 *   icon       {LucideIcon}  Override the default status icon (optional)
 *   variant    {string}      'crop' | 'weather' — picks domain-aware icons
 *   className  {string}      Extra classes on the root div
 */

/* ── Status → visual theme ──────────────────────────────────────────────── *
 * All classes are literal strings so Tailwind v4 can include them at build.
 * ────────────────────────────────────────────────────────────────────────── */
const THEME = {
  Healthy: {
    root:      'bg-primary/10 border-primary/25',
    text:      'text-primary',
    iconBg:    'bg-primary/15',
    pulse:     'bg-primary',
    cropIcon:  ShieldCheck,
    weatherIcon: CloudSun,
  },
  Warning: {
    root:      'bg-warning/10 border-warning/25',
    text:      'text-warning',
    iconBg:    'bg-warning/15',
    pulse:     'bg-warning',
    cropIcon:  AlertTriangle,
    weatherIcon: CloudRain,
  },
  Risk: {
    root:      'bg-danger/10 border-danger/25',
    text:      'text-danger',
    iconBg:    'bg-danger/15',
    pulse:     'bg-danger',
    cropIcon:  ShieldX,
    weatherIcon: Zap,
  },
};

function StatusCard({
  label,
  status = 'Healthy',
  condition,
  detail,
  meta,
  icon: IconOverride,
  variant = 'crop',
  className = '',
}) {
  const cfg = THEME[status] ?? THEME.Healthy;
  const DefaultIcon = variant === 'weather' ? cfg.weatherIcon : cfg.cropIcon;
  const Icon = IconOverride ?? DefaultIcon;

  return (
    <div className={`rounded-xl border ${cfg.root} p-5 flex flex-col gap-3 ${className}`}>

      {/* ─── Top row: label + live pulse ─────────────────────────── */}
      <div className="flex items-center justify-between">
        <span className="text-[9px] font-bold uppercase tracking-widest text-on-surface-variant">
          {label}
        </span>
        <span className={`w-2 h-2 rounded-full ${cfg.pulse} animate-pulse`} />
      </div>

      {/* ─── Status headline ─────────────────────────────────────── */}
      <div className="flex items-center gap-3">
        <div className={`shrink-0 p-2.5 rounded-xl ${cfg.iconBg} ${cfg.text}`}>
          <Icon size={20} />
        </div>
        <div className="min-w-0">
          <span className={`text-2xl font-headline font-bold leading-none ${cfg.text}`}>
            {status}
          </span>
          {condition && (
            <p className="text-[9px] text-on-surface-variant mt-1 leading-snug truncate">
              {condition}
            </p>
          )}
        </div>
      </div>

      {/* ─── Detail body ─────────────────────────────────────────── */}
      {detail && (
        <p className="text-[10px] text-on-surface-variant leading-snug">
          {detail}
        </p>
      )}

      {/* ─── Meta footer ─────────────────────────────────────────── */}
      {meta && (
        <p className={`text-[9px] font-medium ${cfg.text} opacity-75`}>
          {meta}
        </p>
      )}
    </div>
  );
}

export default StatusCard;
