/**
 * AgriTwin-GH Design Token Constants
 *
 * Single source of truth for colours, spacing, shadows, radii, and
 * transitions used across the frontend.  Tailwind @theme tokens in
 * index.css MIRROR these values — update both together when changing
 * the design language.
 */

/* ─── Colours ───────────────────────────────────────────────────────────── */
export const COLORS = {
  // Surface scale
  background:       '#0b1326',
  surface:          '#171f33',
  surfaceLow:       '#131b2e',
  surfaceDeep:      '#060e20',
  surfaceHigh:      '#222a3d',
  surfaceHighest:   '#2d3449',

  // Brand accents
  primary:           '#4be277',
  primaryContainer:  '#22c55e',
  secondary:         '#89ceff',
  secondaryDim:      '#5db8f5',

  // Semantic
  danger:   '#f87171',
  warning:  '#fbbf24',
  success:  '#4be277',

  // Text
  onSurface:        '#dae2fd',
  onSurfaceVariant: '#bccbb9',
  onPrimary:        '#003915',

  // Border
  outlineVariant: '#3d4a3d',
};

/* ─── Typography ────────────────────────────────────────────────────────── */
export const FONTS = {
  sans:     '"Inter", ui-sans-serif, system-ui, sans-serif',
  headline: '"Space Grotesk", sans-serif',
  mono:     '"JetBrains Mono", "Fira Code", ui-monospace, monospace',
};

/* ─── Border radii ──────────────────────────────────────────────────────── */
export const RADII = {
  sm:   '0.375rem',   // 6 px
  md:   '0.5rem',     // 8 px
  lg:   '0.75rem',    // 12 px
  xl:   '1rem',       // 16 px
  '2xl': '1.25rem',   // 20 px
  '3xl': '1.5rem',    // 24 px
  full: '9999px',
};

/* ─── Shadows ───────────────────────────────────────────────────────────── */
export const SHADOWS = {
  card:      '0 4px 24px rgba(0,0,0,0.45), 0 1px 4px rgba(0,0,0,0.25)',
  glass:     '0 8px 32px rgba(0,0,0,0.55)',
  inset:     'inset 0 1px 0 rgba(255,255,255,0.04)',
  glowGreen: '0 0 40px rgba(75,226,119,0.20), 0 0 80px rgba(75,226,119,0.08)',
  glowBlue:  '0 0 40px rgba(137,206,255,0.20), 0 0 80px rgba(137,206,255,0.08)',
};

/* ─── Transitions ───────────────────────────────────────────────────────── */
export const TRANSITIONS = {
  fast:   '150ms cubic-bezier(0.4, 0, 0.2, 1)',
  normal: '250ms cubic-bezier(0.4, 0, 0.2, 1)',
  slow:   '400ms cubic-bezier(0.4, 0, 0.2, 1)',
  spring: '400ms cubic-bezier(0.34, 1.56, 0.64, 1)',
};

/* ─── Layout constants ──────────────────────────────────────────────────── */
export const LAYOUT = {
  navbarHeight:  '4rem',    // 64 px  — fixed top navbar
  sidebarWidth:  '5rem',    // 80 px  — fixed left sidebar (xl+)
  pagePadX:      '2rem',    // 32 px  — horizontal page padding
  sectionGap:    '3rem',    // 48 px  — between major page sections
  cardPad:       '1.5rem',  // 24 px  — inner card padding
};

/* ─── Reusable Tailwind class strings ───────────────────────────────────── */
export const CLS = {
  // Containers
  glassPanel:    'glass-panel rounded-xl',
  surfaceCard:   'bg-surface-low rounded-xl shadow-card',
  surfaceHigh:   'bg-surface-high rounded-xl',

  // Text roles
  sectionLabel:  'text-[10px] uppercase tracking-widest text-on-surface-variant',
  heading:       'font-headline font-bold tracking-tight text-on-surface',
  mono:          'font-mono text-xs text-on-surface-variant',

  // Action buttons
  primaryBtn:    'bg-primary text-on-primary font-headline font-bold px-6 py-3 rounded-full transition-all hover:bg-primary-container active:scale-95',
  ghostBtn:      'bg-primary/10 border border-primary/20 text-primary font-headline font-bold px-6 py-3 rounded-full transition-all hover:bg-primary/20 active:scale-95',
  ghostDangerBtn:'bg-danger/10 border border-danger/20 text-danger font-headline font-bold px-6 py-3 rounded-full transition-all hover:bg-danger/20 active:scale-95',
  iconBtn:       'p-2 rounded-full text-on-surface-variant hover:text-primary hover:bg-surface-highest transition-all',

  // Status badges
  onlineBadge:   'flex items-center gap-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20',
  warningBadge:  'flex items-center gap-1.5 px-3 py-1 rounded-full bg-warning/10 border border-warning/20',
  dangerBadge:   'flex items-center gap-1.5 px-3 py-1 rounded-full bg-danger/10 border border-danger/20',
};

export default { COLORS, FONTS, RADII, SHADOWS, TRANSITIONS, LAYOUT, CLS };
