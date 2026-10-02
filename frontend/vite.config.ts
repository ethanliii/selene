import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// SELENE frontend. In dev, /api is proxied to the FastAPI backend (uvicorn :8000).
// In production, `make build` copies dist/ into backend/selene/api/static so uvicorn serves both.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
      // FastAPI's OpenAPI document: the client reads it once to learn which optional routes exist.
      '/openapi.json': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
});
