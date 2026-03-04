import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    open: true,
    proxy: {
      '/api/ask': 'http://127.0.0.1:8001',
      '/api/followup': 'http://127.0.0.1:8002',
      '/api/updatestatus': 'http://127.0.0.1:8003',
      '/health': {
        target: 'http://127.0.0.1:8001',
      },
    },
  },
})
