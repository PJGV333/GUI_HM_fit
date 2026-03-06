import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scipy")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmfit_core import run_spectroscopy
from hmfit_core.solvers import NewtonRaphson


def _matrix_by_lambda(values, n_lambda: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise AssertionError(f"Expected a 2D matrix, got shape={arr.shape}")
    if arr.shape[0] == n_lambda:
        return arr
    if arr.shape[1] == n_lambda:
        return arr.T
    raise AssertionError(f"Matrix shape {arr.shape} is incompatible with n_lambda={n_lambda}")


def _build_identity_dataset(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    wavelengths = np.linspace(240.0, 560.0, 33, dtype=float)
    n_points = 12
    h_tot = np.full(n_points, 1.0e-3, dtype=float)
    g_tot = np.linspace(0.0, 2.0e-3, n_points, dtype=float)
    c_tot = np.column_stack([h_tot, g_tot])

    modelo_solver = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float)
    solver = NewtonRaphson(c_tot, modelo_solver, nas=[], model_sett="Free")
    c_abs, _ = solver.concentraciones(np.array([3.2], dtype=float))

    eps = np.zeros((3, wavelengths.size), dtype=float)
    eps[0, :] = 1.3e3 * np.exp(-((wavelengths - 270.0) ** 2) / (2.0 * 22.0**2))
    eps[1, :] = 0.9e3 * np.exp(-((wavelengths - 320.0) ** 2) / (2.0 * 26.0**2))
    eps[2, :] = 1.5e3 * np.exp(-((wavelengths - 380.0) ** 2) / (2.0 * 30.0**2))

    y = c_abs @ eps
    y += rng.normal(scale=2.0e-5, size=y.shape)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(y.T, index=wavelengths).to_excel(writer, sheet_name="Spectra")
        pd.DataFrame({"H": h_tot, "G": g_tot}).to_excel(writer, sheet_name="Conc", index=False)

    return {"wavelengths": wavelengths}


def _build_shared_group_dataset(path: Path) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(17)
    wavelengths = np.linspace(250.0, 500.0, 36, dtype=float)
    n_points = 14
    h_tot = np.full(n_points, 1.0e-3, dtype=float)
    g_tot = np.linspace(0.0, 3.0e-3, n_points, dtype=float)
    c_tot = np.column_stack([h_tot, g_tot])

    modelo_solver = np.array([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0]], dtype=float)
    solver = NewtonRaphson(c_tot, modelo_solver, nas=[1], model_sett="Step by step")
    c_abs, _ = solver.concentraciones(np.array([5.2, 2.9], dtype=float))

    eps_h = 1.2e3 * np.exp(-((wavelengths - 290.0) ** 2) / (2.0 * 18.0**2))
    eps_complex = 9.0e2 * np.exp(-((wavelengths - 360.0) ** 2) / (2.0 * 28.0**2))
    eps_species = np.vstack([eps_h, eps_complex, eps_complex])

    y = c_abs @ eps_species
    y += rng.normal(scale=2.0e-5, size=y.shape)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        pd.DataFrame(y.T, index=wavelengths).to_excel(writer, sheet_name="Spectra")
        pd.DataFrame({"H": h_tot, "G": g_tot}).to_excel(writer, sheet_name="Conc", index=False)

    return {"wavelengths": wavelengths, "eps_h": eps_h, "eps_complex": eps_complex}


def test_absgrp_identity_matches_legacy_fit(tmp_path: Path) -> None:
    xlsx = tmp_path / "identity_absgrp.xlsx"
    meta = _build_identity_dataset(xlsx)

    base_cfg = {
        "file_path": str(xlsx),
        "spectra_sheet": "Spectra",
        "conc_sheet": "Conc",
        "column_names": ["H", "G"],
        "receptor_label": "H",
        "guest_label": "G",
        "efa_enabled": False,
        "efa_eigenvalues": 0,
        "modelo": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "non_abs_species": [],
        "species_names": ["H", "G", "HG"],
        "algorithm": "Newton-Raphson",
        "model_settings": "Free",
        "optimizer": "powell",
        "initial_k": [3.0],
        "bounds": [(0.0, 6.0)],
        "channels_raw": "All",
        "channels_mode": "all",
        "channels_resolved": [],
    }

    out_default = run_spectroscopy(dict(base_cfg))
    out_identity = run_spectroscopy(
        dict(base_cfg, abs_groups={"A": ["H"], "B": ["G"], "C": ["HG"]})
    )

    assert out_default.get("success", False), out_default.get("error")
    assert out_identity.get("success", False), out_identity.get("error")

    assert np.isclose(
        float(out_default["statistics"]["RMS"]),
        float(out_identity["statistics"]["RMS"]),
        rtol=1e-6,
        atol=1e-9,
    )
    assert np.allclose(
        np.asarray(out_default["export_data"]["k"], dtype=float),
        np.asarray(out_identity["export_data"]["k"], dtype=float),
        rtol=1e-4,
        atol=1e-5,
    )

    n_lambda = len(meta["wavelengths"])
    a_default = _matrix_by_lambda(out_default["export_data"]["A"], n_lambda)
    a_identity = _matrix_by_lambda(out_identity["export_data"]["A"], n_lambda)
    assert a_default.shape == a_identity.shape == (n_lambda, 3)
    assert np.allclose(a_default, a_identity, rtol=1e-5, atol=1e-6)


def test_absgrp_shared_groups_reduce_optical_components(tmp_path: Path) -> None:
    xlsx = tmp_path / "shared_absgrp.xlsx"
    meta = _build_shared_group_dataset(xlsx)

    base_cfg = {
        "file_path": str(xlsx),
        "spectra_sheet": "Spectra",
        "conc_sheet": "Conc",
        "column_names": ["H", "G"],
        "receptor_label": "H",
        "guest_label": "G",
        "efa_enabled": True,
        "efa_eigenvalues": 2,
        "modelo": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 2.0]],
        "non_abs_species": [1],
        "species_names": ["H", "G", "HG", "HG2"],
        "algorithm": "Newton-Raphson",
        "model_settings": "Step by step",
        "optimizer": "powell",
        "initial_k": [5.0, 3.0],
        "bounds": [(0.0, 8.0), (0.0, 6.0)],
        "channels_raw": "All",
        "channels_mode": "all",
        "channels_resolved": [],
        "eps_solver_mode": "nnls_hard",
    }

    out_no_groups = run_spectroscopy(dict(base_cfg))
    out_grouped = run_spectroscopy(
        dict(base_cfg, abs_groups={"1": ["H"], "2": ["HG", "HG2"]})
    )

    assert out_no_groups.get("success", False), out_no_groups.get("error")
    assert out_grouped.get("success", False), out_grouped.get("error")

    export_grouped = out_grouped["export_data"]
    n_lambda = len(meta["wavelengths"])
    a_grouped = _matrix_by_lambda(export_grouped["A"], n_lambda)
    a_species = _matrix_by_lambda(export_grouped["A_species"], n_lambda)
    a_no_groups = _matrix_by_lambda(out_no_groups["export_data"]["A"], n_lambda)

    assert a_no_groups.shape[1] == 3
    assert a_grouped.shape == (n_lambda, 2)
    assert a_species.shape == (n_lambda, 3)
    assert export_grouped["abs_group_labels"] == ["absgrp1", "absgrp2"]
    assert export_grouped["abs_group_members"] == [["H"], ["HG", "HG2"]]
    assert export_grouped["abs_group_count"] == 2
    assert export_grouped["grouped_absorptivities_active"] is True
    assert "Absorptivity groups: 2" in out_grouped["results_text"]

    assert np.allclose(a_species[:, 1], a_species[:, 2], rtol=0.0, atol=1e-10)
    assert np.allclose(a_grouped[:, 0], meta["eps_h"], rtol=0.20, atol=20.0)
    assert np.allclose(a_grouped[:, 1], meta["eps_complex"], rtol=0.20, atol=20.0)

    rms_no_groups = float(out_no_groups["statistics"]["RMS"])
    rms_grouped = float(out_grouped["statistics"]["RMS"])
    assert rms_grouped <= (1.05 * rms_no_groups + 1e-8)
