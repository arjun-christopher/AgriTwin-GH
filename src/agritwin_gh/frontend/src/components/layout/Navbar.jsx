/**
 * Navbar — secondary navigation bar, sits directly below Header.
 *
 * Left  : page navigation links (Home · Detailed Insights · Manual Override)
 * Right : unit identifier + online status pulse badge
 *
 * Active link is highlighted with a bottom border + primary colour.
 * Style matches the sample UI's nav language.
 */

import { LayoutDashboard, TrendingUp, SlidersHorizontal } from 'lucide-react';

const NAV_LINKS = [
  { label: 'Dashboard',        page: 'dashboard', icon: LayoutDashboard  },
  { label: 'Detailed Insights',page: 'insights',  icon: TrendingUp       },
  { label: 'Manual Override',  page: 'override',  icon: SlidersHorizontal},
];

function Navbar({ currentPage, navigate }) {
  return (
    <nav
      className="fixed top-14 left-0 w-full z-40 flex items-center justify-between px-6 md:px-8 h-11 bg-surface-highest/50 backdrop-blur-2xl border-b border-outline-variant/10"
      aria-label="Primary navigation"
    >
      {/* ── Nav links ── */}
      <div className="flex items-center gap-0.5 h-full">
        {NAV_LINKS.map(({ label, page, icon: Icon }) => {
          const active = currentPage === page;
          return (
            <button
              key={page}
              onClick={() => navigate(page)}
              className={`
                relative flex items-center gap-1.5 h-full px-3 text-[11px] font-headline uppercase tracking-widest
                transition-colors duration-200 whitespace-nowrap
                focus-visible:outline-none focus-visible:bg-surface-high
                ${active
                  ? 'text-primary'
                  : 'text-on-surface-variant hover:text-on-surface'
                }
              `}
              aria-current={active ? 'page' : undefined}
            >
              <Icon size={13} strokeWidth={active ? 2.3 : 1.8} className="shrink-0" />
              <span className="hidden sm:inline">{label}</span>

              {/* Active underline — slides in from below */}
              <span
                className={`
                  absolute bottom-0 left-2 right-2 h-0.5 rounded-full bg-primary
                  transition-all duration-250
                  ${active ? 'opacity-100 scale-x-100' : 'opacity-0 scale-x-0'}
                `}
                style={{ transformOrigin: 'left center' }}
              />
            </button>
          );
        })}
      </div>


    </nav>
  );
}

export default Navbar;

