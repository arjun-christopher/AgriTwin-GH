import { ArrowRight } from 'lucide-react';

/**
 * PanelCard — reusable panel wrapper for the 3-column home layout.
 *
 * Props:
 *   title        {string}      Panel header label (omit to hide header)
 *   subtitle     {string}      Secondary text below title
 *   icon         {LucideIcon}  Small icon in the header
 *   badge        {string}      Pill label next to the title
 *   accent       {boolean}     Left-edge primary-colour accent border
 *   onNavigate   {fn}          CTA handler in the footer (omit to hide footer)
 *   navLabel     {string}      CTA button text (default "View more")
 *   className    {string}      Extra classes on the root element
 *   children     {ReactNode}   Panel body content
 */
function PanelCard({
  title,
  subtitle,
  icon: Icon,
  badge,
  accent = false,
  onNavigate,
  navLabel = 'View more',
  className = '',
  children,
}) {
  return (
    <div
      className={[
        'glass-panel rounded-xl flex flex-col overflow-hidden',
        accent ? 'border-l-4 border-primary' : '',
        className,
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {/* ─── Header ──────────────────────────────────────────────────── */}
      {title && (
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-outline-variant/10 shrink-0">
          <div className="flex items-center gap-2.5 min-w-0">
            {Icon && (
              <span className="shrink-0 p-1.5 rounded-lg bg-primary/10 text-primary">
                <Icon size={13} />
              </span>
            )}
            <div className="min-w-0">
              <h3 className="text-[10px] font-bold uppercase tracking-widest text-on-surface leading-none">
                {title}
              </h3>
              {subtitle && (
                <p className="text-[9px] text-on-surface-variant mt-1 leading-none truncate">
                  {subtitle}
                </p>
              )}
            </div>
          </div>

          {badge && (
            <span className="shrink-0 ml-3 text-[9px] font-bold uppercase tracking-widest px-2 py-0.5 rounded-full bg-surface-highest border border-outline-variant/20 text-on-surface-variant">
              {badge}
            </span>
          )}
        </div>
      )}

      {/* ─── Content ─────────────────────────────────────────────────── */}
      <div className="flex-1 p-5 flex flex-col gap-3">
        {children}
      </div>

      {/* ─── Footer CTA ──────────────────────────────────────────────── */}
      {onNavigate && (
        <div className="px-5 py-3 border-t border-outline-variant/10 shrink-0">
          <button
            onClick={onNavigate}
            className="w-full flex items-center justify-center gap-1.5 py-2 rounded-lg text-[10px] font-bold uppercase tracking-widest text-primary hover:bg-primary/10 active:scale-95 transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
          >
            {navLabel}
            <ArrowRight size={11} />
          </button>
        </div>
      )}
    </div>
  );
}

export default PanelCard;
