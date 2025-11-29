// Centralized backend configuration for HM Fit.
// Reads Vite/Tauri env vars when available; falls back to sensible defaults
// so the app works across machines without code changes.

const host = (import.meta?.env?.VITE_BACKEND_HOST || "127.0.0.1").trim();
const port = (import.meta?.env?.VITE_BACKEND_PORT || "8001").trim();

export const BACKEND_BASE_URL = `http://${host}:${port}`;
export const WS_BASE_URL = `ws://${host}:${port}`;

export function describeBackendTarget() {
  return `${host}:${port}`;
}
