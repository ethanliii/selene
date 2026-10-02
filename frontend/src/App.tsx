/** App shell: persistent classification/demo banner (with the LIVE/MOCK endpoint summary) + router. */
import { useEffect } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import { api } from './api/client';
import { ArchitecturePage } from './pages/ArchitecturePage';
import { CoveragePage } from './pages/CoveragePage';
import { OpsPage } from './pages/OpsPage';
import { LiveStatus } from './panels/LiveStatus';

function Banner() {
  return (
    <div className="banner" role="note">
      <span>UNCLASSIFIED // DEMONSTRATION — ALL OBJECTS AND EVENTS MARKED SIMULATED ARE NOTIONAL</span>
      <LiveStatus />
    </div>
  );
}

export default function App() {
  // Probe the backend once so the banner reflects mock mode even on placeholder pages.
  useEffect(() => {
    api.health().catch(() => undefined);
  }, []);
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <div className="app">
        <Banner />
        <Routes>
          <Route path="/ops" element={<OpsPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
          <Route path="/coverage" element={<CoveragePage />} />
          <Route path="*" element={<Navigate to="/ops" replace />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}
