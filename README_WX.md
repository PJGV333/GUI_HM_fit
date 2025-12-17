# HM Fit (wxPython) — estado experimental

Objetivo: portar el prototipo Tauri/FastAPI a una GUI 100% Python usando wxPython.

## Ejecutar

1. Instala dependencias científicas:
   - `pip install -r backend_fastapi/requirements.txt`
2. Instala wxPython:
   - `pip install -r requirements_wx.txt`
3. Lanza la GUI desde la raíz del repo:
   - `python -m hmfit_wx`

## Notas

- Linux/Windows: el diseño evita `WebView` y usa imágenes PNG (matplotlib) para máxima compatibilidad.
- La ejecución corre en un hilo (no bloquea la UI) y el progreso se muestra en el panel de log.
- Tauri (`hmfit_tauri/`) se mantiene como referencia temporal mientras se completa la paridad de UI.

