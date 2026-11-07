import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// Разработка: npm run dev → http://127.0.0.1:5173, запросы /api проксируются на бэкенд (python -m backend).
// Сборка: npm run build → frontend/dist, его раздаёт бэкенд на http://127.0.0.1:8000.
const backend = process.env.PHOTONIKA_BACKEND || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: { "/api": { target: backend, changeOrigin: true } },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
