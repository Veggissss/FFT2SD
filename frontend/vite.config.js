import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    // https://github.com/solidjs/vite-plugin-solid/issues/125#issuecomment-1975771849
    alias: [
      { find: "msw/node", replacement: "/node_modules/msw/lib/native/index.mjs" }
    ],
  }
})
