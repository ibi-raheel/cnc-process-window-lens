import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Served from ibiraheel.com/demos/cnc-process-window-lens/. Every asset and data
// fetch goes through import.meta.env.BASE_URL, so this one line moves the demo.
export default defineConfig({
  base: '/demos/cnc-process-window-lens/',
  plugins: [react()],
})
