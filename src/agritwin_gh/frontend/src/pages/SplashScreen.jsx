import { useEffect, useState } from 'react';
import { Sprout } from 'lucide-react';

/* ─────────────────────────────────────────────────────────────────────────────
   TIMING CONFIG — adjust these values to change the feel of the intro.
   All values are in milliseconds.
───────────────────────────────────────────────────────────────────────────── */
const TIMING = {
  /* Total time the intro plays before auto-navigating to the app */
  introDuration: 4200,

  /* Fade-out overlay duration — this overlap is subtracted from introDuration */
  exitFadeDuration: 500,

  /* Stagger delays for each content band appearing on screen */
  delay: {
    bg:      0,      // ambient background glows
    logo:    280,    // logo mark icon
    heading: 820,    // "AgriTwin-GH" title
    tagline: 1260,   // tagline text  
  },
};

/* ── Helpers ─────────────────────────────────────────────────────────────── */
/** Returns an inline style object that starts a CSS animation after `delay` ms. */
function delayed(delayMs) {
  return { animationDelay: `${delayMs}ms` };
}

/* ── Component ───────────────────────────────────────────────────────────── */
function SplashScreen({ onComplete }) {
  const [exiting, setExiting] = useState(false);

  /* Launch the exit sequence then fire onComplete */
  function startExit() {
    setExiting(true);
    setTimeout(onComplete, TIMING.exitFadeDuration);
  }

  /* Auto-advance after introDuration */
  useEffect(() => {
    const timer = setTimeout(
      startExit,
      TIMING.introDuration - TIMING.exitFadeDuration,
    );
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  /* ── Render ── */
  return (
    <div
      /* Root — handles the exit cross-fade via CSS transition */
      className="fixed inset-0 z-100 bg-background flex items-center justify-center overflow-hidden px-6"
      style={{
        transition: `opacity ${TIMING.exitFadeDuration}ms ease-in-out, transform ${TIMING.exitFadeDuration}ms ease-in-out`,
        opacity:    exiting ? 0 : 1,
        transform:  exiting ? 'scale(1.015)' : 'scale(1)',
      }}
    >

      {/* ── Ambient background layer ── */}
      <div
        className="splash-bg pointer-events-none absolute inset-0 overflow-hidden"
        style={delayed(TIMING.delay.bg)}
      >
        {/* Primary green bloom — top-center */}
        <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-175 h-175 rounded-full bg-primary/6 blur-[130px]" />
        {/* Secondary blue bloom — bottom-right */}
        <div className="absolute -bottom-16 -right-16 w-125 h-125 rounded-full bg-secondary/5 blur-[110px]" />
        {/* Tertiary green — bottom-left corner accent */}
        <div className="absolute bottom-0 -left-24 w-96 h-96 rounded-full bg-primary/4 blur-[100px]" />

        {/* Fine grid overlay — currentColor inherits from CSS variable so it adapts to the active theme */}
        <svg
          className="absolute inset-0 w-full h-full opacity-[0.035]"
          style={{ color: 'var(--color-primary)' }}
          xmlns="http://www.w3.org/2000/svg"
          aria-hidden="true"
        >
          <defs>
            <pattern id="splash-grid" width="52" height="52" patternUnits="userSpaceOnUse">
              <path d="M 52 0 L 0 0 0 52" fill="none" stroke="currentColor" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width="100%" height="100%" fill="url(#splash-grid)" />
        </svg>

        {/* Corner tick marks — futuristic framing detail */}
        {[
          'top-6 left-6 border-t border-l',
          'top-6 right-6 border-t border-r',
          'bottom-6 left-6 border-b border-l',
          'bottom-6 right-6 border-b border-r',
        ].map((cls) => (
          <div
            key={cls}
            className={`absolute w-8 h-8 border-primary/25 ${cls}`}
          />
        ))}
      </div>

      {/* ── Centre content stack ── */}
      <div className="relative z-10 flex flex-col items-center text-center gap-7 max-w-2xl w-full mx-auto">

        {/* Logo mark */}
        <div
          className="splash-logo flex flex-col items-center gap-3"
          style={delayed(TIMING.delay.logo)}
        >
          {/* Icon box with glow ring and scan-line sweep */}
          <div className="relative">
            <div className="splash-glow-ring w-20 h-20 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center">
              <Sprout size={36} className="text-primary" strokeWidth={1.5} />
            </div>
            {/* Horizontal scan line */}
            <div
              className="splash-scan-line absolute left-0 w-full overflow-hidden pointer-events-none"
              style={{ height: '2px' }}
              aria-hidden="true"
            >
              <div className="h-full w-full bg-linear-to-r from-transparent via-primary/70 to-transparent" />
            </div>
          </div>


        </div>

        {/* Main title — converging letter-spacing reveal */}
        <h1
          className="splash-title font-headline font-bold text-on-surface leading-none select-none"
          style={{
            ...delayed(TIMING.delay.heading),
            fontSize: 'clamp(3.5rem, 10vw, 6.5rem)',
          }}
        >
          Agri<span className="text-primary italic">Twin</span>
          <span className="text-on-surface-variant/70">‑GH</span>
        </h1>

        {/* Tagline */}
        <p
          className="splash-tagline text-on-surface-variant font-light text-base md:text-lg leading-relaxed max-w-md mx-auto"
          style={delayed(TIMING.delay.tagline)}
        >
          Ecological intelligence for precision greenhouse horticulture.
        </p>



        {/* Progress bar — thin bar fills over introDuration */}
        <div className="w-48 h-px bg-surface-highest rounded-full overflow-hidden mt-1">
          <div
            className="h-full bg-primary rounded-full"
            style={{
              animation: `progress-fill ${TIMING.introDuration - TIMING.exitFadeDuration}ms linear forwards`,
            }}
          />
        </div>


      </div>



      {/* ── Decorative spinning ring — bottom right corner ── */}
      <div
        className="pointer-events-none absolute bottom-10 right-10 w-20 h-20 rounded-full border-2 border-primary/10 border-t-primary/35 animate-spin-slow"
        aria-hidden="true"
      />
    </div>
  );
}

export default SplashScreen;

