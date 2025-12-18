import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _stats_table_to_dict(stats_table) -> dict:
    out = {}
    for row in stats_table or []:
        if not isinstance(row, (list, tuple)) or len(row) < 2:
            continue
        out[str(row[0])] = row[1]
    return out


def _write_minimal_nmr_workbook(path: Path) -> None:
    # Simple 1:1 binding model (H + G <-> HG) with a single H signal.
    log10K = 3.0
    K = 10 ** log10K

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

    conc_df = pd.DataFrame({"H (M)": H_arr, "G (M)": G_tot})
    shifts_df = pd.DataFrame({"sig (H)": delta_obs})

    # Use ODS to avoid depending on `openpyxl` in the local environment.
    with pd.ExcelWriter(path, engine="odf") as writer:
        conc_df.to_excel(writer, sheet_name="Conc", index=False)
        shifts_df.to_excel(writer, sheet_name="NMR", index=False)


def test_core_wrapper_matches_backend_direct_nmr():
    # Add project root to path (tests can be run from anywhere).
    root = _repo_root()
    sys.path.insert(0, str(root))

    from backend_fastapi.nmr_processor import process_nmr_data
    from hmfit_core import run_nmr

    with tempfile.TemporaryDirectory() as td:
        book_path = Path(td) / "mini_nmr.ods"
        _write_minimal_nmr_workbook(book_path)

        direct = process_nmr_data(
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
        assert direct.get("success") is True, direct.get("error")

        gui_cfg = {
            "file_path": str(book_path),
            "nmr_sheet": "NMR",
            "conc_sheet": "Conc",
            "column_names": ["H (M)", "G (M)"],
            "signal_names": ["sig (H)"],
            "receptor_label": "H (M)",
            "guest_label": "G (M)",
            "modelo": [
                [1, 0],  # H
                [0, 1],  # G
                [1, 1],  # HG
            ],
            "initial_k": [3.0],
            "bounds": [(0.0, 10.0)],
            "fixed_mask": [True],
            "algorithm": "Newton-Raphson",
            "optimizer": "powell",
            "model_settings": "Free",
            "non_abs_species": [],
        }
        via_core = run_nmr(gui_cfg)
        assert via_core.get("success") is True, via_core.get("error")

        k_direct = np.asarray(direct["export_data"]["k"], dtype=float)
        k_core = np.asarray(via_core["export_data"]["k"], dtype=float)
        assert np.allclose(k_direct, k_core, atol=1e-12, rtol=0.0)
        assert bool(direct["export_data"]["fixed_mask"][0]) is True
        assert bool(via_core["export_data"]["fixed_mask"][0]) is True

        stats_direct = _stats_table_to_dict(direct["export_data"].get("stats_table"))
        stats_core = _stats_table_to_dict(via_core["export_data"].get("stats_table"))
        assert abs(float(stats_direct["RMS"]) - float(stats_core["RMS"])) < 1e-12


def test_gui_panels_do_not_import_legacy_math_modules():
    root = _repo_root()

    spectro_path = root / "hmfit_wx_legacy" / "Spectroscopy_controls.py"
    nmr_path = root / "hmfit_wx_legacy" / "NMR_controls.py"

    spectro_src = spectro_path.read_text(encoding="utf-8")
    nmr_src = nmr_path.read_text(encoding="utf-8")

    forbidden_imports = (
        "from NR_conc_algoritm",
        "from LM_conc_algoritm",
        "from errors import",
        "from scipy",
    )
    for token in forbidden_imports:
        assert token not in spectro_src, f"{spectro_path} imports legacy math: {token}"
        assert token not in nmr_src, f"{nmr_path} imports legacy math: {token}"

    assert "from hmfit_core import run_spectroscopy" in spectro_src
    assert "from hmfit_core import run_nmr" in nmr_src
