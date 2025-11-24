"""Minimal FastAPI backend for HM Fit prototype.

En esta fase los endpoints solo sirven como “stubs” para probar
la comunicación con la GUI Tauri. Más adelante aquí llamaremos
a Simulation_controls.py, Methods.py, etc.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
import io


class DummyFitRequest(BaseModel):
    """Ejemplo sencillo para probar el backend."""
    x: list[float]
    y: list[float]


class SpectroscopySetup(BaseModel):
    """Datos básicos del formulario de Spectroscopy (fase 1)."""

    spectra_sheet: str | None = None
    conc_sheet: str | None = None
    column_names: list[str] = []
    receptor_label: str | None = None
    guest_label: str | None = None
    efa_enabled: bool = False
    efa_eigenvalues: int = 0
    n_components: int = 0
    n_species: int = 0
    non_abs_species: list[str] = []


from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HM Fit FastAPI prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción restringir a ["http://localhost:5173", "tauri://localhost"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    """Health check endpoint usado por el botón 'Probar backend'."""
    return {"status": "ok"}


@app.post("/dummy_fit")
def dummy_fit(req: DummyFitRequest):
    """Endpoint de prueba original."""
    return {"sum_y": sum(req.y), "n_points": len(req.y)}


@app.post("/spectroscopy/preview")
def spectroscopy_preview(setup: SpectroscopySetup):
    """Por ahora solo eco de los datos que manda la GUI."""
    return {
        "message": "Spectroscopy setup recibido (stub).",
        "setup": setup,
    }


@app.post("/list_sheets")
async def list_sheets(file: UploadFile = File(...)):
    """Recibe un archivo Excel y devuelve la lista de hojas."""
    try:
        contents = await file.read()
        # Usamos pandas para leer solo los nombres de las hojas sin cargar todo el archivo
        xl = pd.ExcelFile(io.BytesIO(contents))
        return {"sheets": xl.sheet_names}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "backend_fastapi.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
