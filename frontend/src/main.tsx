import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import { server } from "./mocks/server";
import App from './App';

const baseUrl = import.meta.env.VITE_API_BASE_URL ?? '';
if (baseUrl.includes("github.io")) {
  server.listen();
}

const rootElement = document.getElementById('root')
if (!rootElement) throw new Error('Failed to find the root element')
createRoot(rootElement).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
