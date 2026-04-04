import Header from './Header';
import Navbar from './Navbar';

/**
 * AppShell — persistent layout wrapper for all in-app pages.
 *
 * Fixed layers (z-order, top → bottom of stacking):
 *   z-50  Header  (h-14 = 56px)
 *   z-40  Navbar  (h-11 = 44px)  — top-14
 *
 * Main content is padded to clear both bars:
 *   pt-[100px]  (56 + 44)  — clears Header + Navbar
 */
function AppShell({ children, currentPage, navigate }) {
  return (
    <div className="min-h-screen bg-background text-on-surface font-sans transition-colors duration-300">

      <Header onLogoClick={() => navigate('dashboard')} />
      <Navbar currentPage={currentPage} navigate={navigate} />

      {/* Main scroll area — offset for Header (56px) + Navbar (44px) */}
      <main className="min-h-screen pt-25 px-6 md:px-8 pb-16">
        {children}
      </main>
    </div>
  );
}

export default AppShell;

