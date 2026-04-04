import {
  LayoutDashboard,
  TrendingUp,
  SlidersHorizontal,
} from 'lucide-react';

const ITEMS = [
  { icon: LayoutDashboard,   label: 'Dashboard', page: 'dashboard' },
  { icon: TrendingUp,        label: 'Insights',  page: 'insights'  },
  { icon: SlidersHorizontal, label: 'Override',  page: 'override'  },
];

function Sidebar({ currentPage, navigate }) {
  return (
    <aside className="fixed left-0 top-25 h-[calc(100vh-100px)] hidden xl:flex flex-col items-center py-6 gap-4 bg-surface-low/80 backdrop-blur-xl w-20 z-40 border-r border-outline-variant/10">
      {ITEMS.map(({ icon: Icon, label, page }) => {
        const active = currentPage === page && label !== 'Crops';
        return (
          <div key={label} className="group relative">
            <button
              onClick={() => navigate(page)}
              className={`p-3 rounded-xl flex items-center justify-center transition-all duration-200 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary/40 ${
                active
                  ? 'bg-primary/15 text-primary'
                  : 'text-on-surface-variant hover:bg-surface-highest hover:text-on-surface'
              }`}
              aria-label={label}
            >
              <Icon size={22} />
            </button>

            {/* Tooltip */}
            <span className="pointer-events-none absolute left-18 top-1/2 -translate-y-1/2 bg-surface-highest text-on-surface text-[10px] px-2 py-1 rounded whitespace-nowrap opacity-0 group-hover:opacity-100 transition-opacity duration-150 shadow-card">
              {label}
            </span>
          </div>
        );
      })}
    </aside>
  );
}

export default Sidebar;
