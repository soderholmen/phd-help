/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Dev proxy to the FastAPI server (uvicorn's port is operator-chosen;
// PHD_BACKEND overrides, e.g. http://localhost:8777).
const backend = process.env.PHD_BACKEND ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/health": backend,
      "/sections": backend,
      "/corpus": backend,
      "/ca.pem": backend,
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
  test: {
    environment: "jsdom",
    setupFiles: "./vitest.setup.ts",
    globals: true, // RTL auto-cleanup registers against the global afterEach
  },
});
