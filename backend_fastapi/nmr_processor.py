"""
Minimal NMR backend hooks exposed to the Tauri frontend.

The logic here is intentionally lightweight: it mirrors the workbook
loading flow of ``NMR_controls.py`` (file → sheet selection → column
selection) and returns structured placeholders that the frontend can
consume without reimplementing scientific details.  The numerical core
from ``NMR_controls.py`` can be integrated later behind the same
function signatures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import base64
import io

import pandas as pd


def list_sheets_from_bytes(file_bytes: bytes) -> List[str]:
    """Return available sheet names from an Excel workbook."""
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return xl.sheet_names


def list_columns_from_bytes(file_bytes: bytes, sheet_name: str) -> List[str]:
    """Return column headers for a given sheet."""
    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet_name, nrows=0)
    return list(df.columns)


def summarize_nmr_inputs(
    *,
    file_path: str,
    spectra_sheet: str,
    conc_sheet: str,
    column_names: List[str],
    signals_sheet: Optional[str] = None,
    receptor_label: str | None = None,
    guest_label: str | None = None,
) -> Dict:
    """
    Lightweight placeholder that mirrors the wxPython workflow:
    - Reads the selected sheets.
    - Echoes back the chosen columns and simple sizes.

    This allows the frontend to be wired without duplicating scientific
    logic.  The full NMR fit can later replace this function while
    keeping the same shape of the response payload.
    """
    spectra_df = pd.read_excel(file_path, spectra_sheet, header=0, index_col=0)
    conc_df = pd.read_excel(file_path, conc_sheet, header=0)

    selected_columns = conc_df[column_names]

    # Optional chemical shift sheet: present in wx GUI
    signals = None
    if signals_sheet:
        signals = pd.read_excel(file_path, signals_sheet, header=0, index_col=0)

    payload = {
        "success": True,
        "n_points": int(len(spectra_df.index)),
        "n_traces": int(spectra_df.shape[1]),
        "n_concentrations": int(selected_columns.shape[1]),
        "columns": list(selected_columns.columns),
        "receptor_label": receptor_label or "",
        "guest_label": guest_label or "",
        "signals_sheet": signals_sheet or "",
    }

    if signals is not None:
        payload["n_signals"] = int(signals.shape[0])
        payload["signal_axis"] = list(map(str, signals.index))

    return payload
