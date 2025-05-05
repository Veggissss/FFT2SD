import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// eslint-disable-next-line no-undef
const baseUrl = process.env.VITE_API_BASE_URL ?? '';
let basePath = "/";
if (baseUrl.includes("github.io")) {
  basePath = `/${baseUrl.split("/").pop()}/`;
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  base: basePath,
  resolve: {
    // https://github.com/solidjs/vite-plugin-solid/issues/125#issuecomment-1975771849
    alias: [
      { find: "msw/node", replacement: "/node_modules/msw/lib/native/index.mjs" }
    ],
  }
})
