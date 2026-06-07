import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/topics": "http://localhost:8000",
      "/match-post-topics": "http://localhost:8000",
      "/topic-response": "http://localhost:8000",
      "/health": "http://localhost:8000",
    },
  },
});
