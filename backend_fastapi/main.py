"""FastAPI backend for HM Fit.

Includes:
- WebSocket for progress streaming.
- Spectroscopy processing using the refactored business logic.
- Lightweight NMR endpoints so the Tauri frontend can be wired without
  reimplementing scientific code.
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
from fastapi.responses import StreamingResponse

from backend_fastapi import nmr_processor
from backend_fastapi.nmr_processor import set_progress_callback
from backend_fastapi.config import (
    BACKEND_HOST,
    BACKEND_PORT,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_HEADERS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_ORIGIN_REGEX,
    CORS_ALLOW_ORIGINS,
    HM_FIT_ENV,
)

app = FastAPI(title="HM Fit FastAPI prototype")
app.state.hm_fit_env = HM_FIT_ENV


def _log_cors_settings():
    print(
        "[HM Fit] CORS settings => env=", HM_FIT_ENV,
        "allow_origins=", CORS_ALLOW_ORIGINS,
        "allow_origin_regex=", CORS_ALLOW_ORIGIN_REGEX,
    )


# CORS configuration: in dev allow localhost variants (and a regex for any port);
# in prod tighten to Tauri origins. See backend_fastapi/config.py for details.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_origin_regex=CORS_ALLOW_ORIGIN_REGEX,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)
_log_cors_settings()

# Temporary directory for uploaded files
UPLOAD_DIR = Path(tempfile.gettempdir()) / "hmfit_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# WebSocket connections
active_connections: list[WebSocket] = []


class ExportRequest(BaseModel):
    constants: list[dict] = []
    statistics: dict = {}
    results_text: str | None = None
    export_data: dict | None = None

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


@app.post("/nmr/list_sheets")
async def nmr_list_sheets(file: UploadFile = File(...)):
    """Expose workbook sheet names for the NMR flow (same as spectroscopy)."""
    try:
        contents = await file.read()
        return {"sheets": nmr_processor.list_sheets_from_bytes(contents)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading Excel file: {str(e)}")


@app.post("/nmr/list_columns")
async def nmr_list_columns(file: UploadFile = File(...), sheet_name: str = Form(...)):
    """Expose sheet columns for the NMR flow."""
    try:
        contents = await file.read()
        columns = nmr_processor.list_columns_from_bytes(contents, sheet_name)
        return {"columns": columns}
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
        
        try:
            # Run in executor (thread pool) to avoid blocking event loop
            results = await loop.run_in_executor(None, run_processing_in_thread)
        except ValueError as ve:
            # Diferential_evolution bounds u otros errores validados: devolver 400
            await broadcast_progress(str(ve))
            raise HTTPException(status_code=400, detail=str(ve))
        finally:
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


@app.post("/process_nmr")
async def process_nmr(
    file: UploadFile = File(...),
    spectra_sheet: str = Form(...),
    conc_sheet: str = Form(...),
    column_names: str = Form(...),  # JSON string
    signals_sheet: str = Form(""),
    signal_names: str = Form("[]"), # JSON string
    receptor_label: str = Form(""),
    guest_label: str = Form(""),
    modelo: str = Form("[]"),  # JSON string
    non_abs_species: str = Form("[]"),  # JSON string
    algorithm: str = Form("Newton-Raphson"),
    model_settings: str = Form("Free"),
    optimizer: str = Form("powell"),
    initial_k: str = Form("[]"),  # JSON string
    bounds: str = Form("[]")  # JSON string
):
    """
    Process NMR titration data.
    """
    try:
        import json
        
        # Parse JSON strings
        column_names_list = json.loads(column_names)
        signal_names_list = json.loads(signal_names)
        modelo_list = json.loads(modelo)
        non_abs_species_list = json.loads(non_abs_species)
        initial_k_list = json.loads(initial_k)
        bounds_list = json.loads(bounds)
        
        # Save file temporarily
        temp_path = UPLOAD_DIR / f"temp_nmr_{int(time.time())}_{file.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        loop = asyncio.get_running_loop()

        def progress_callback_wrapper(message: str):
            asyncio.create_task(broadcast_progress(message))

        def run_processing_in_thread():
            # Set callback with loop for thread-safety
            set_progress_callback(progress_callback_wrapper, loop)
            return nmr_processor.process_nmr_data(
                file_path=str(temp_path),
                spectra_sheet=spectra_sheet, # This is the sheet with chemical shifts
                conc_sheet=conc_sheet,
                column_names=column_names_list,
                signal_names=signal_names_list,
                receptor_label=receptor_label,
                guest_label=guest_label,
                model_matrix=modelo_list,
                k_initial=initial_k_list,
                k_bounds=bounds_list,
                algorithm=algorithm,
                optimizer=optimizer,
                model_settings=model_settings,
                non_absorbent_species=non_abs_species_list
            )

        await broadcast_progress("Iniciando procesamiento NMR (en hilo separado)...")

        try:
            results = await loop.run_in_executor(None, run_processing_in_thread)
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        if "error" in results:
            await broadcast_progress(results["error"])
            raise HTTPException(status_code=400, detail=results["error"])

        await broadcast_progress("Procesamiento NMR completado!")
        return results

    except Exception as e:
        error_msg = f"Error en procesamiento NMR: {str(e)}"
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/export_results_xlsx")
async def export_results_xlsx(payload: ExportRequest):
    """
    Construye un archivo XLSX con las constantes y estadísticas enviadas desde el frontend.
    Sigue la idea del guardado en wx: hojas separadas para constants/stats y el reporte plano.
    """
    try:
        constants = payload.constants or []
        statistics = payload.statistics or {}
        results_text = payload.results_text or ""
        export_data = payload.export_data or {}

        buffer = io.BytesIO()

        with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
            def as_dataframe(name: str, data, *, allow_none: bool = True) -> pd.DataFrame | None:
                if data is None:
                    return None if allow_none else pd.DataFrame()
                if isinstance(data, pd.DataFrame):
                    return data
                if isinstance(data, pd.Series):
                    return data.to_frame()
                if isinstance(data, dict):
                    return pd.DataFrame(list(data.items()), columns=["key", "value"])
                return pd.DataFrame(data)

            def write_sheet(name: str, df: pd.DataFrame) -> None:
                df.to_excel(writer, sheet_name=name, index=True)

            is_nmr_export = any(
                key in export_data
                for key in (
                    "Chemical_Shifts",
                    "Calculated_Chemical_Shifts",
                    "signal_names",
                )
            )

            if is_nmr_export:
                import numpy as np

                modelo = as_dataframe("Model", export_data.get("modelo"), allow_none=False)
                Co = as_dataframe("Absorbent_species", export_data.get("Co"), allow_none=False)
                C = as_dataframe("All_species", export_data.get("C"), allow_none=False)
                C_T = as_dataframe("Tot_con_comp", export_data.get("C_T"), allow_none=False)
                dq = as_dataframe("Chemical_Shifts", export_data.get("Chemical_Shifts"), allow_none=False)
                dq_cal = as_dataframe(
                    "Calculated_Chemical_Shifts",
                    export_data.get("Calculated_Chemical_Shifts"),
                    allow_none=False,
                )

                k_vals = export_data.get("k") or []
                percK = export_data.get("percK") or []
                k_nombres = [f"K{i+1}" for i in range(len(k_vals))]
                k = pd.DataFrame(
                    {
                        "Constants": list(k_vals),
                        "Error (%)": list(percK)[: len(k_vals)],
                    },
                    index=k_nombres,
                )

                k_ini_vals = export_data.get("k_ini") or []
                k_ini_nombres = [f"K{i+1}" for i in range(len(k_ini_vals))]
                k_ini = pd.DataFrame({"Constants": list(k_ini_vals)}, index=k_ini_nombres)

                stats_table = export_data.get("stats_table")
                if stats_table:
                    stats = pd.DataFrame(stats_table, columns=["metric", "Stats"]).set_index("metric")
                else:
                    stats = pd.DataFrame(list(statistics.items()), columns=["metric", "Stats"]).set_index("metric")
                if "covfit" in stats.index:
                    stats.loc["covfit", "Stats"] = str(stats.loc["covfit", "Stats"])

                coef = as_dataframe("Coefficients", export_data.get("Coefficients"), allow_none=True)
                if coef is None or coef.empty:
                    # Reconstruir coeficientes SOLO para exportación (no afecta cálculo ni plots).
                    try:
                        dq_arr = np.asarray(export_data.get("Chemical_Shifts"), dtype=float)
                        C_arr = np.asarray(export_data.get("C"), dtype=float)
                        C_T_arr = np.asarray(export_data.get("C_T"), dtype=float)
                        column_names = list(export_data.get("column_names") or [])
                        signal_names = list(export_data.get("signal_names") or [])

                        C_T_df = pd.DataFrame(C_T_arr, columns=column_names) if column_names else pd.DataFrame(C_T_arr)
                        D_cols, _ = nmr_processor.build_D_cols(C_T_df, column_names, signal_names, default_idx=0)

                        coef_mat = np.full((C_arr.shape[1], dq_arr.shape[1]), np.nan, dtype=float)
                        Xbase = np.asarray(C_arr, float)
                        finite_rows = np.isfinite(Xbase).all(axis=1)
                        nonzero_rows = np.linalg.norm(Xbase, axis=1) > 0.0
                        goodX = finite_rows & nonzero_rows
                        mask = np.isfinite(dq_arr) & np.isfinite(D_cols) & (np.abs(D_cols) > 0)

                        for j in range(dq_arr.shape[1]):
                            D = D_cols[:, j]
                            mj = mask[:, j] & goodX & np.isfinite(D) & (np.abs(D) > 0)
                            if int(mj.sum()) < 2:
                                continue
                            Xj = Xbase[mj, :] / D[mj][:, None]
                            y = dq_arr[mj, j]
                            coef_vec, *_ = np.linalg.lstsq(Xj, y, rcond=1e-10)
                            coef_mat[:, j] = coef_vec

                        coef = pd.DataFrame(coef_mat)
                    except Exception as e:
                        print("DEBUG coef reconstruction failed:", e)
                        coef = pd.DataFrame()

                # Logs solicitados (solo backend)
                print("DEBUG coef shape:", getattr(coef, "shape", None))
                print("DEBUG coef head:\n", coef.head() if hasattr(coef, "head") else coef)

                # Forzar a DataFrame numérico si vienen tipos raros
                if not isinstance(coef, pd.DataFrame):
                    coef = pd.DataFrame(np.asarray(coef))

                if coef.shape[1] > 0 and all(isinstance(c, (int, np.integer)) for c in coef.columns):
                    coef.columns = [f"coef_{i}" for i in range(1, coef.shape[1] + 1)]

                sheets = {
                    "Model": modelo,
                    "Absorbent_species": Co,
                    "All_species": C,
                    "Tot_con_comp": C_T,
                    "Chemical_Shifts": dq,
                    "Calculated_Chemical_Shifts": dq_cal,
                    "Coefficients": coef,
                    "K_calculated": k,
                    "Init_guess_K": k_ini,
                    "Stats": stats,
                }

                for name, df in sheets.items():
                    if df is None:
                        df = pd.DataFrame()
                    write_sheet(name, df)
            else:
                if export_data:
                    modelo = as_dataframe("Model", export_data.get("modelo"))
                    if modelo is not None:
                        modelo.to_excel(writer, sheet_name="Model", index=False)

                    C = as_dataframe("Absorbent_species", export_data.get("C"))
                    if C is not None:
                        C.to_excel(writer, sheet_name="Absorbent_species", index=False)

                    Co = as_dataframe("All_species", export_data.get("Co"))
                    if Co is not None:
                        Co.to_excel(writer, sheet_name="All_species", index=False)

                    C_T = as_dataframe("Tot_con_comp", export_data.get("C_T"))
                    if C_T is not None:
                        C_T.to_excel(writer, sheet_name="Tot_con_comp", index=False)

                    A = export_data.get("A")
                    nm = export_data.get("A_index") or export_data.get("nm")
                    if A is not None:
                        dfA = as_dataframe("Molar_Absortivities", A, allow_none=True)
                        if dfA is not None:
                            if nm:
                                dfA.index = nm
                            dfA.to_excel(writer, sheet_name="Molar_Absortivities", index_label='nm' if nm else None)

                    Y = export_data.get("Y")
                    if Y is not None:
                        dfY = as_dataframe("Y_observed", Y, allow_none=True)
                        if dfY is not None:
                            if nm:
                                dfY.index = nm
                            dfY.to_excel(writer, sheet_name="Y_observed", index_label='nm' if nm else None)

                    yfit = export_data.get("yfit")
                    if yfit is not None:
                        dfPhi = as_dataframe("Y_calculated", yfit, allow_none=True)
                        if dfPhi is not None:
                            if nm:
                                dfPhi.index = nm
                            dfPhi.to_excel(writer, sheet_name="Y_calculated", index_label='nm' if nm else None)

                    k_vals = export_data.get("k") or []
                    percK = export_data.get("percK") or []
                    if k_vals:
                        names = [f"K{i+1}" for i in range(len(k_vals))]
                        dfk = pd.DataFrame({"log10K": k_vals, "percK(%)": percK[:len(k_vals)]}, index=names)
                        dfk.to_excel(writer, sheet_name="K_calculated")

                    k_ini = export_data.get("k_ini") or []
                    if k_ini:
                        names_ini = [f"k{i+1}" for i in range(len(k_ini))]
                        dfin = pd.DataFrame({"init_guess": k_ini}, index=names_ini)
                        dfin.to_excel(writer, sheet_name="Init_guess_K")

                if statistics:
                    pd.DataFrame(list(statistics.items()), columns=["metric", "value"]).to_excel(
                        writer, sheet_name="Statistics", index=False
                    )

        buffer.seek(0)
        data = buffer.getvalue()
        headers = {
            "Content-Disposition": 'attachment; filename="hmfit_results.xlsx"'
        }
        return StreamingResponse(
            io.BytesIO(data),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers=headers,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo generar el XLSX: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "backend_fastapi.main:app",
        host=BACKEND_HOST,
        port=BACKEND_PORT,
        reload=True,
    )
