from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _write_minimal_nmr_xlsx(path: Path) -> None:
    pytest.importorskip("openpyxl")
    from openpyxl import Workbook

    # Simple 1:1 binding model (H + G <-> HG) with a single H signal.
    log10K = 3.0
    K = 10**log10K

    H_tot = 1.0e-3
    G_tot = np.array([0.0, 0.5e-3, 1.0e-3, 1.5e-3, 2.0e-3], dtype=float)
    H_arr = np.full_like(G_tot, H_tot, dtype=float)

    # Solve 1:1 equilibrium: K x^2 - (K(H+G)+1)x + KHG = 0
    a = K
    b = -(K * (H_arr + G_tot) + 1.0)
    c = K * H_arr * G_tot
    disc = np.maximum(b * b - 4.0 * a * c, 0.0)
    x = (-b - np.sqrt(disc)) / (2.0 * a)  # smaller root

    delta_H = 1.0
    delta_HG = 2.0
    delta_obs = delta_H + (x / H_arr) * (delta_HG - delta_H)

    wb = Workbook()
    ws_conc = wb.active
    ws_conc.title = "Conc"
    ws_conc.append(["H (M)", "G (M)"])
    for h, g in zip(H_arr.tolist(), G_tot.tolist()):
        ws_conc.append([float(h), float(g)])

    ws_nmr = wb.create_sheet("NMR")
    ws_nmr.append(["sig (H)"])
    for d in delta_obs.tolist():
        ws_nmr.append([float(d)])

    wb.save(str(path))


def test_nmr_regression_minimal_xlsx():
    root = _repo_root()
    sys.path.insert(0, str(root))

    from backend_fastapi.nmr_processor import process_nmr_data

    with tempfile.TemporaryDirectory() as td:
        book_path = Path(td) / "mini_nmr.xlsx"
        _write_minimal_nmr_xlsx(book_path)

        result = process_nmr_data(
            file_path=str(book_path),
            spectra_sheet="NMR",
            conc_sheet="Conc",
            column_names=["H (M)", "G (M)"],
            signal_names=["sig (H)"],
            receptor_label="H (M)",
            guest_label="G (M)",
            model_matrix=[
                [1, 0],  # H
                [0, 1],  # G
                [1, 1],  # HG
            ],
            k_initial=[3.0],
            k_bounds=[{"min": 0.0, "max": 10.0}],
            algorithm="Newton-Raphson",
            optimizer="powell",
            model_settings="Free",
            non_absorbent_species=[],
            k_fixed=[True],
        )

        assert result.get("success") is True, result.get("error")
