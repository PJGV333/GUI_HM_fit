# HM Fit - Prototipo Tauri + FastAPI (Fase 1)

Esta fase agrega un esqueleto mínimo para probar un backend FastAPI y un frontend Tauri sin modificar la GUI wxPython existente.

## Backend FastAPI

1. **Crear/activar entorno Python** (ejemplos):
   ```bash
   # con venv
   python -m venv .venv
   source .venv/bin/activate

   # o con conda
   conda create -n hmfit-fastapi python=3.11
   conda activate hmfit-fastapi
   ```
2. **Instalar dependencias**:
   ```bash
   pip install -r backend_fastapi/requirements.txt
   ```
3. **Arrancar el servidor** (puerto 8000):
   ```bash
   python -m backend_fastapi.main
   ```
   Endpoints disponibles:
   - `GET /health` → `{ "status": "ok" }`
   - `POST /dummy_fit` → `{ "sum_y": <suma de y>, "n_points": <len(y)> }`

Estos endpoints están listos para conectarse más adelante con la lógica de:
- `Simulation_controls.py` (simulación)
- `Methods.py`, `NR_conc_algoritm.py`, `LM_conc_algoritm.py` (ajustes)
- `errors.py`, `core_ad_probe.py`, etc.

## Frontend Tauri (proyecto `hmfit_tauri/`)

1. **Dependencias de sistema**: asegúrate de tener toolchain de Rust y Node instalados (en CachyOS/Arch):
   ```bash
   sudo pacman -S --needed rustup nodejs npm
   rustup default stable
   ```
   Instala el CLI de Tauri si aún no está disponible:
   ```bash
   npm install -g @tauri-apps/cli
   ```
2. **Instalar dependencias del proyecto**:
   ```bash
   cd hmfit_tauri
   npm install
   ```
3. **Ejecutar en modo desarrollo** (asumiendo FastAPI activo en `http://127.0.0.1:8000`):
   ```bash
   npm run tauri dev
   ```
   En la ventana verás un botón **“Probar conexión”** que hace `fetch` a `/health` y muestra la respuesta.

> Nota: En esta fase el frontend no inicia el proceso de Python; levanta el backend en otra terminal antes de abrir Tauri.

## Resumen rápido de comandos

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r backend_fastapi/requirements.txt
python -m backend_fastapi.main

# Frontend (en otra terminal)
cd hmfit_tauri
npm install
npm run tauri dev
```
