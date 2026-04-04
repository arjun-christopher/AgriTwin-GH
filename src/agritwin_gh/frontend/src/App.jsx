import { useState } from 'react';

import { ThemeProvider } from './context/ThemeContext';
import AppShell from './components/layout/AppShell';
import SplashScreen from './pages/SplashScreen';
import HomeDashboard from './pages/HomeDashboard';
import DetailedInsights from './pages/DetailedInsights';
import ManualOverride from './pages/ManualOverride';

/** Map page keys to their components. */
const PAGE_MAP = {
  dashboard: HomeDashboard,
  insights:  DetailedInsights,
  override:  ManualOverride,
};

function AppRouter() {
  const [page, setPage] = useState('splash');

  // Splash intro — sits above everything, auto-advances.
  if (page === 'splash') {
    return <SplashScreen onComplete={() => setPage('dashboard')} />;
  }

  const PageComponent = PAGE_MAP[page] ?? HomeDashboard;

  return (
    // animate-app-enter fades the shell in after the splash exits.
    <div className="animate-app-enter">
      <AppShell currentPage={page} navigate={setPage}>
        <PageComponent navigate={setPage} />
      </AppShell>
    </div>
  );
}

/**
 * App — root component.
 * ThemeProvider wraps the entire tree so Header (and any future component)
 * can call useTheme() to read or toggle the current theme.
 */
function App() {
  return (
    <ThemeProvider>
      <AppRouter />
    </ThemeProvider>
  );
}

export default App;



