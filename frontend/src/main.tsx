import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import './styles/theme.css';
import { useSelene } from './store/useSelene';

// Dev-only handle for headless QA scripts (select an object, move the cursor) — never shipped in the build.
if (import.meta.env.DEV) (window as unknown as { __selene?: typeof useSelene }).__selene = useSelene;

ReactDOM.createRoot(document.getElementById('root') as HTMLElement).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
