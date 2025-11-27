"""FastAPI backend for HM Fit - Full version with JAX processing.

Includes:
- WebSocket for progress streaming
- Direct file processing (no sessions) to avoid hangs
- JAX-based spectroscopy processing
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import uvicorn
import pandas as pd
import io
import os
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Optional
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HM Fit FastAPI prototype")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "hmfit_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# WebSocket connections
active_connections: list[WebSocket] = []

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/list_sheets")
async def list_sheets(file: UploadFile = File(...)):
    """Receive Excel file and return list of sheet names."""
    try:
        contents = await file.read()
        xl = pd.ExcelFile(io.BytesIO(contents))
        return {"sheets": xl.sheet_names}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")

@app.post("/list_columns")
async def list_columns(file: UploadFile = File(...), sheet_name: str = Form(...)):
    """Receive Excel file and sheet name, return column names."""
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents), sheet_name=sheet_name, nrows=0)
        return {"columns": list(df.columns)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading columns: {str(e)}")

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """WebSocket endpoint for streaming progress updates."""
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        active_connections.remove(websocket)

async def broadcast_progress(message: str):
    """Broadcast progress message to all connected WebSocket clients."""
    for connection in active_connections:
        try:
            await connection.send_text(json.dumps({"type": "progress", "message": message}))
        except:
            pass

@app.post("/process_spectroscopy")
async def process_spectroscopy(
    file: UploadFile = File(...),
    spectra_sheet: str = Form(...),
    conc_sheet: str = Form(...),
    column_names: str = Form(...),  # JSON string
    receptor_label: str = Form(""),
    guest_label: str = Form(""),
    efa_enabled: str = Form("false"),
    efa_eigenvalues: str = Form("0"),
    modelo: str = Form("[]"),  # JSON string
    non_abs_species: str = Form("[]"),  # JSON string
    algorithm: str = Form("Newton-Raphson"),
    model_settings: str = Form("Free"),
    optimizer: str = Form("powell"),
    initial_k: str = Form("[]"),  # JSON string
    bounds: str = Form("[]")  # JSON string
):
    """Process spectroscopy data with progress streaming."""
    try:
        import json
        
        # Parse JSON strings
        column_names_list = json.loads(column_names)
        modelo_list = json.loads(modelo)
        non_abs_species_list = json.loads(non_abs_species)
        initial_k_list = json.loads(initial_k)
        bounds_list = json.loads(bounds)
        efa_enabled_bool = efa_enabled.lower() == "true"
        efa_eigenvalues_int = int(efa_eigenvalues)
        
        # Save file temporarily
        temp_path = UPLOAD_DIR / f"temp_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Import processor
        # NOTE: We assume dependencies (JAX, etc.) are installed now
        import sys
        # Add current directory to path to find spectroscopy_processor.py
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.append(current_dir)
            
        from spectroscopy_processor import process_spectroscopy_data, set_progress_callback
        
        # Get current event loop
        loop = asyncio.get_running_loop()
        
        # Callback wrapper that schedules the async broadcast
        def progress_callback_wrapper(message: str):
            asyncio.create_task(broadcast_progress(message))

        # Wrapper function to run in thread pool
        def run_processing_in_thread():
            # Set callback with loop for thread-safety
            set_progress_callback(progress_callback_wrapper, loop)
            
            return process_spectroscopy_data(
                file_path=str(temp_path),
                spectra_sheet=spectra_sheet,
                conc_sheet=conc_sheet,
                column_names=column_names_list,
                receptor_label=receptor_label if receptor_label else None,
                guest_label=guest_label if guest_label else None,
                efa_enabled=efa_enabled_bool,
                efa_eigenvalues=efa_eigenvalues_int,
                modelo=modelo_list,
                non_abs_species=non_abs_species_list,
                algorithm=algorithm,
                model_settings=model_settings,
                optimizer=optimizer,
                initial_k=initial_k_list,
                bounds=bounds_list
            )
        
        # Send initial message
        await broadcast_progress("Iniciando procesamiento de datos (en hilo separado)...")
        
        # Run in executor (thread pool) to avoid blocking event loop
        results = await loop.run_in_executor(None, run_processing_in_thread)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass
        
        # Send completion message
        await broadcast_progress("Procesamiento completado!")
        
        return results
        
    except Exception as e:
        error_msg = f"Error en procesamiento: {str(e)}"
        await broadcast_progress(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)

if __name__ == "__main__":
    uvicorn.run(
        "backend_fastapi.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
