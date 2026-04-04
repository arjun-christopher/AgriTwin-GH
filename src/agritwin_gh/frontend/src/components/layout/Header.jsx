/**
 * Header — fixed top bar.
 *
 * Left  : Sprout logo icon + "AgriTwin-GH" wordmark
 * Centre: (empty — content flows via flex justify-between)
 * Right : live date · live time · theme toggle button
 *
 * The live clock updates every second via setInterval.
 * All display strings are pure JS — no external date library.
 */

import { useEffect, useState } from 'react';
import { Sprout, Moon, Sun } from 'lucide-react';
import { useTheme } from '../../context/ThemeContext';

/* ── Date/time formatting helpers ─────────────────────────────────────────── */

const DAY_NAMES  = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTH_NAMES = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

function pad(n) {
  return String(n).padStart(2, '0');
}

/** Format a Date → e.g.  "Sat 04 Apr 2026" */
function formatDate(d) {
  return `${DAY_NAMES[d.getDay()]} ${pad(d.getDate())} ${MONTH_NAMES[d.getMonth()]} ${d.getFullYear()}`;
}

/** Format a Date → e.g.  "14:32:07 UTC" */
function formatTime(d) {
  return `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}:${pad(d.getUTCSeconds())} UTC`;
}

/* ── Component ────────────────────────────────────────────────────────────── */

function Header({ onLogoClick }) {
  const { isDark, toggleTheme } = useTheme();

  const [now, setNow] = useState(() => new Date());

  /* Tick every second */
  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 1000);
    return () => clearInterval(id);
  }, []);

  return (
    <header className="fixed top-0 left-0 w-full z-50 flex items-center justify-between px-6 md:px-8 h-14 bg-surface-deep/80 backdrop-blur-3xl border-b border-outline-variant/15">

      {/* ── Brand ── */}
      <button
        onClick={onLogoClick}
        className="flex items-center gap-2.5 font-headline font-bold text-xl tracking-tighter text-primary hover:opacity-80 transition-opacity duration-200 select-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 focus-visible:rounded-lg"
        aria-label="Go to dashboard"
      >
        <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-primary/10 border border-primary/20">
          <Sprout size={17} strokeWidth={2.2} />
        </div>
        <span>Agri<span className="italic">Twin</span><span className="text-on-surface-variant/60 font-light">-GH</span></span>
      </button>

      {/* ── Right slot: datetime + theme toggle ── */}
      <div className="flex items-center gap-4">

        {/* Date + time — stacked */}
        <div className="hidden sm:flex flex-col items-end leading-tight select-none">
          <span className="text-[11px] font-mono text-on-surface-variant opacity-70 tracking-wide">
            {formatDate(now)}
          </span>
          <span className="text-[11px] font-mono text-primary opacity-80 tracking-wider tabular-nums">
            {formatTime(now)}
          </span>
        </div>

        {/* Mobile: show just the time inline */}
        <span className="sm:hidden text-[11px] font-mono text-primary tabular-nums select-none">
          {formatTime(now)}
        </span>

        {/* Divider */}
        <div className="hidden sm:block w-px h-5 bg-outline-variant/30" />

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          aria-label={isDark ? 'Switch to light theme' : 'Switch to dark theme'}
          className="group relative p-2 rounded-full text-on-surface-variant hover:text-primary hover:bg-surface-highest transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40"
        >
          {/* Sun icon fades in for dark mode (clicking switches to light) */}
          <span
            className={`absolute inset-0 flex items-center justify-center transition-all duration-300 ${
              isDark ? 'opacity-100 scale-100' : 'opacity-0 scale-75'
            }`}
            aria-hidden={!isDark}
          >
            <Sun size={17} />
          </span>
          {/* Moon icon fades in for light mode */}
          <span
            className={`flex items-center justify-center transition-all duration-300 ${
              isDark ? 'opacity-0 scale-75' : 'opacity-100 scale-100'
            }`}
            aria-hidden={isDark}
          >
            <Moon size={17} />
          </span>
        </button>
      </div>
    </header>
  );
}

export default Header;
