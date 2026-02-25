import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


ROOT = _repo_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmfit_core import run_spectroscopy
from hmfit_core.solvers import NewtonRaphson
from hmfit_core.processors.spectroscopy_processor import compute_spectral_weights


def _build_dataset(path: Path, *, seed: int, tail_noise: float = 0.0) -> None:
    rng = np.random.default_rng(seed)

    wavelengths = np.linspace(240.0, 600.0, 31, dtype=float)
    m = 10
    h_tot = np.full(m, 1.0e-3, dtype=float)
    g_tot = np.linspace(0.0, 2.0e-3, m, dtype=float)
    c_tot = np.column_stack([h_tot, g_tot])

    modelo = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], dtype=float)
    solver = NewtonRaphson(c_tot, modelo, nas=[], model_sett="Free")
    c_abs, _ = solver.concentraciones(np.array([3.0], dtype=float))

    wl = wavelengths
    eps = np.zeros((3, wl.size), dtype=float)
    eps[0, :] = 1.2e3 * np.exp(-((wl - 265.0) ** 2) / (2.0 * 22.0**2))
    eps[1, :] = 0.8e3 * np.exp(-((wl - 320.0) ** 2) / (2.0 * 26.0**2))
    eps[2, :] = 1.6e3 * np.exp(-((wl - 350.0) ** 2) / (2.0 * 30.0**2))

    y = c_abs @ eps
    y += np.linspace(0.020, -0.010, m, dtype=float)[:, None]
    y += rng.normal(scale=2.0e-4, size=y.shape)

    tail = wl >= 450.0
    if tail_noise > 0.0:
        y[:, tail] += rng.normal(scale=tail_noise, size=(m, int(np.count_nonzero(tail))))

    spec_df = pd.DataFrame(y.T, index=wl, columns=[f"p{i+1}" for i in range(m)])
    conc_df = pd.DataFrame({"H": h_tot, "G": g_tot})

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        spec_df.to_excel(writer, sheet_name="Spectra", index=True)
        conc_df.to_excel(writer, sheet_name="Conc", index=False)


def _base_config(file_path: Path) -> dict:
    return {
        "file_path": str(file_path),
        "spectra_sheet": "Spectra",
        "conc_sheet": "Conc",
        "column_names": ["H", "G"],
        "receptor_label": "H",
        "guest_label": "G",
        "efa_enabled": False,
        "efa_eigenvalues": 0,
        "modelo": [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "non_abs_species": [],
        "algorithm": "Newton-Raphson",
        "model_settings": "Free",
        "optimizer": "powell",
        "initial_k": [3.0],
        "bounds": [(0.0, 6.0)],
        "channels_raw": "All",
        "channels_mode": "all",
        "channels_resolved": [],
        "show_stability_diagnostics": False,
        "multi_start_runs": 1,
    }


def _as_species_by_lambda(a_export, n_lambda: int) -> np.ndarray:
    arr = np.asarray(a_export, dtype=float)
    if arr.ndim != 2:
        raise AssertionError(f"Expected 2D absorptivity matrix, got shape={arr.shape}")
    if arr.shape[0] == n_lambda:
        return arr.T
    return arr


@pytest.mark.parametrize(
    "seed,tail_noise",
    [
        (1, 0.0),      # titration buena
        (2, 0.0),      # repetición
        (3, 1.5e-3),   # cola más plana/ruidosa
    ],
)
def test_spectro_golden_regression_soft_bound(seed: int, tail_noise: float, tmp_path: Path) -> None:
    xlsx = tmp_path / f"golden_{seed}.xlsx"
    _build_dataset(xlsx, seed=seed, tail_noise=tail_noise)

    cfg_default = _base_config(xlsx)
    out_default = run_spectroscopy(cfg_default)
    assert out_default.get("success", False), out_default.get("error")

    cfg_enh = dict(cfg_default)
    cfg_enh.update(
        {
            "baseline_mode": "range",
            "baseline_start": 450.0,
            "baseline_end": 600.0,
            "baseline_auto_quantile": 0.20,
            "weighting_mode": "std",
            "weighting_power": 1.0,
            "weighting_normalize": True,
            "eps_solver_mode": "soft_bound",
            "mu": 1e-2,
            "delta_mode": "relative",
            "delta_rel": 0.005,
            "alpha_smooth": 0.02,
        }
    )
    out_enh = run_spectroscopy(cfg_enh)
    assert out_enh.get("success", False), out_enh.get("error")

    rms_default = float(out_default["statistics"]["RMS"])
    rms_enh = float(out_enh["statistics"]["RMS"])
    assert np.isfinite(rms_default) and np.isfinite(rms_enh)
    assert rms_enh <= (1.5 * rms_default)

    k_default = float(out_default["export_data"]["k"][0])
    k_enh = float(out_enh["export_data"]["k"][0])
    assert abs(k_enh - k_default) < 0.6

    export = out_enh["export_data"]
    lb = np.asarray(export.get("lower_bound"), dtype=float).ravel()
    assert lb.size > 0
    n_lambda = len(export.get("nm") or [])
    a_solver = _as_species_by_lambda(export.get("A_solver"), n_lambda=n_lambda)
    assert a_solver.shape[0] == lb.size
    min_per_species = np.min(a_solver, axis=1)
    assert np.all(min_per_species >= (lb - 1e-8))


def test_baseline_range_fallback_when_channels_missing(tmp_path: Path) -> None:
    xlsx = tmp_path / "baseline_fallback.xlsx"
    _build_dataset(xlsx, seed=7, tail_noise=0.0)

    cfg = _base_config(xlsx)
    all_nm = np.linspace(240.0, 600.0, 31, dtype=float).tolist()
    cfg.update(
        {
            "channels_mode": "custom",
            "channels_resolved": [nm for nm in all_nm if nm < 430.0],
            "baseline_mode": "range",
            "baseline_start": 450.0,
            "baseline_end": 600.0,
            "baseline_auto_quantile": 0.20,
        }
    )
    out = run_spectroscopy(cfg)
    assert out.get("success", False), out.get("error")

    meta = out["export_data"]["baseline_meta"]
    assert str(meta.get("mode")) == "auto"
    assert str(meta.get("requested_mode")) == "range"
    warnings = [str(w).lower() for w in (out.get("warnings") or [])]
    assert any("falling back to auto baseline" in w for w in warnings)


@pytest.mark.parametrize("mode", ["std", "max"])
def test_weighting_degenerate_profile_fallback_to_uniform(mode: str) -> None:
    y = np.zeros((8, 12), dtype=float)
    w, meta = compute_spectral_weights(y, mode=mode, normalize=True, power=1.0)
    assert w.shape == (12,)
    assert np.allclose(w, 1.0, atol=0.0, rtol=0.0)
    assert "warning" in meta
