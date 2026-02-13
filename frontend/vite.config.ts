import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/a2a': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/upload': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/files': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/database': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/clear': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    }
  }
})
