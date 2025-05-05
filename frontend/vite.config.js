/* eslint-disable no-undef */
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default ({ mode }) => {
  // https://stackoverflow.com/questions/66389043/how-can-i-use-vite-env-variables-in-vite-config-js
  process.env = { ...process.env, ...loadEnv(mode, process.cwd()) };
  const baseUrl = process.env.VITE_BASE_URL ?? '';
  let basePath = "/";
  if (baseUrl.includes("github.io")) {
    basePath = `/${baseUrl.split("/").pop()}/`;
  }
  return defineConfig({

    plugins: [react()],
    base: basePath,
    resolve: {
      // https://github.com/solidjs/vite-plugin-solid/issues/125#issuecomment-1975771849
      alias: [
        { find: "msw/node", replacement: "/node_modules/msw/lib/native/index.mjs" }
      ],
    }
  })
}

