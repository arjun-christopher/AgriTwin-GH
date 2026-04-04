/**
 * ThemeContext — lightweight theme management.
 *
 * Architecture:
 *  - 'dark'  = default; CSS variables defined in @theme (index.css)
 *  - 'light' = [data-theme="light"] block in index.css overrides those vars
 *
 * The provider applies `data-theme` on <html> so every CSS custom property
 * override takes effect app-wide, including inside portals and modals.
 *
 * Usage:
 *   const { theme, toggleTheme, isDark } = useTheme();
 */

import { createContext, useContext, useEffect, useState } from 'react';

const ThemeContext = createContext(null);

export function ThemeProvider({ children }) {
  // Initialise from localStorage. The attribute is applied synchronously here
  // (before the first paint) to prevent a flash of the wrong theme on reload.
  const [theme, setTheme] = useState(() => {
    let saved = 'dark';
    try {
      saved = localStorage.getItem('agritwin-theme') ?? 'dark';
    } catch {
      // ignore — storage unavailable in some environments
    }
    // Set immediately so CSS variable overrides are active before first paint.
    document.documentElement.setAttribute('data-theme', saved);
    return saved;
  });

  // Sync data-theme attribute on <html> whenever theme changes.
  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute('data-theme', theme);
    try {
      localStorage.setItem('agritwin-theme', theme);
    } catch {
      // ignore — storage unavailable in some environments
    }
  }, [theme]);

  function toggleTheme() {
    setTheme((prev) => (prev === 'dark' ? 'light' : 'dark'));
  }

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, isDark: theme === 'dark' }}>
      {children}
    </ThemeContext.Provider>
  );
}

/** Consume theme anywhere in the tree. */
export function useTheme() {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used inside <ThemeProvider>');
  return ctx;
}
