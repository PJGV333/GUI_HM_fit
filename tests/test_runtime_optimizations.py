from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from hmfit_core import run_spectroscopy
from hmfit_core.processors import nmr_processor as nmr_mod
from hmfit_core.processors import spectroscopy_processor as spectro_mod
from hmfit_core.solvers import NewtonRaphson


def _build_spectro_dataset(path: Path, seed: int = 17) -> None:
    rng = np.random.default_rng(seed)
    wavelengths = np.linspace(240.0, 600.0, 27, dtype=float)
    m = 9
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
    y += np.linspace(0.015, -0.005, m, dtype=float)[:, None]
    y += rng.normal(scale=2.0e-4, size=y.shape)

    spec_df = pd.DataFrame(y.T, index=wl, columns=[f"p{i+1}" for i in range(m)])
    conc_df = pd.DataFrame({"H": h_tot, "G": g_tot})
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        spec_df.to_excel(writer, sheet_name="Spectra", index=True)
        conc_df.to_excel(writer, sheet_name="Conc", index=False)


def _base_spectro_config(file_path: Path) -> dict:
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


def test_spectro_callback_path_avoids_duplicate_objective(monkeypatch):
    class _DummySolver:
        def concentraciones(self, k_vec):
            k_arr = np.asarray(k_vec, dtype=float).ravel()
            c = np.tile(k_arr[:1], (3, 1))
            return c, np.zeros_like(c)

    def _fake_solve_abs(C, YT, **kwargs):
        del kwargs
        return np.ones((C.shape[1], YT.shape[1]), dtype=float), None

    def _fake_minimize(fun, x0, method=None, bounds=None, callback=None):
        del method, bounds
        x = np.asarray(x0, dtype=float).ravel()
        val = float(fun(x))
        if callback is not None:
            callback(x)

        class _Res:
            pass

        res = _Res()
        res.x = x
        res.fun = val
        res.success = True
        res.message = "ok"
        res.nfev = 1
        return res

    monkeypatch.setattr(spectro_mod, "_build_equilibrium_solver", lambda *a, **k: _DummySolver())
    monkeypatch.setattr(spectro_mod, "_solve_absorptivities", _fake_solve_abs)
    monkeypatch.setattr(spectro_mod.optimize, "minimize", _fake_minimize)

    out = spectro_mod._run_spectro_single_start(
        optimizer="powell",
        algorithm="Newton-Raphson",
        c_t_array=np.array([[1.0], [1.0], [1.0]], dtype=float),
        modelo=np.array([[1.0]], dtype=float),
        non_abs_species=[],
        model_settings="Free",
        start_k=np.array([1.0], dtype=float),
        processed_bounds=[(0.0, 2.0)],
        seed=None,
        y_transposed=np.array([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]], dtype=float),
        weights_row=np.ones((1, 2), dtype=float),
        eps_solver_mode="soft_penalty",
        eps_mu=1e-2,
        delta_mode="off",
        delta_rel=0.01,
        alpha_smooth=0.0,
        smooth_matrix=None,
    )

    assert out["ok"] is True
    assert out["objective_evaluations"] == 1
    assert out["nfev"] == 1


def test_spectro_render_graphs_schema_compatible(tmp_path: Path):
    book = tmp_path / "spectro_render.xlsx"
    _build_spectro_dataset(book)
    cfg = _base_spectro_config(book)

    out_render = run_spectroscopy(dict(cfg, render_graphs=True, render_quality="preview"))
    out_no_render = run_spectroscopy(dict(cfg, render_graphs=False, render_quality="preview"))

    assert out_render.get("success") is True, out_render
    assert out_no_render.get("success") is True, out_no_render

    for key in ("constants", "statistics", "graphs", "legacy_graphs", "plotData", "export_data", "optimizer_result"):
        assert key in out_render
        assert key in out_no_render

    assert set(out_render["graphs"].keys()) == set(out_no_render["graphs"].keys())
    assert out_no_render["graphs"]["fit"] == ""
    assert out_no_render["graphs"]["concentrations"] == ""
    assert out_no_render["graphs"]["absorptivities"] == ""

    k_render = [c["log10K"] for c in out_render["constants"]]
    k_no_render = [c["log10K"] for c in out_no_render["constants"]]
    assert np.allclose(k_render, k_no_render, atol=1e-10, rtol=0.0)


def test_nmr_parallel_multistart_selects_best(monkeypatch):
    conc = pd.DataFrame({"H": [1.0, 1.1, 1.2, 1.3], "G": [0.5, 0.6, 0.7, 0.8]})
    shifts = pd.DataFrame({"sig1": [0.1, 0.1, 0.1, 0.1]})

    def _fake_read_excel(file_path, sheet_name=None, header=0, *args, **kwargs):
        del file_path, header, args, kwargs
        if sheet_name == "nmr":
            return shifts.copy()
        if sheet_name == "conc":
            return conc.copy()
        raise ValueError(sheet_name)

    class _DummyAlgo:
        def __init__(self, c_t_df, modelo, nas, model_settings):
            del modelo, nas, model_settings
            self._n_i = int(getattr(c_t_df, "shape", [0])[0])

        def concentraciones(self, k_full):
            k_full = np.asarray(k_full, dtype=float).ravel()
            c = np.tile(k_full[:1], (self._n_i, 1))
            return c, np.zeros_like(c)

    def _fake_project(dq_block, c_block, d_cols, mask_block, *args, **kwargs):
        del d_cols, mask_block, args, kwargs
        dq_block = np.asarray(dq_block, dtype=float)
        c_block = np.asarray(c_block, dtype=float)
        return np.tile(c_block[:, :1], (1, dq_block.shape[1]))

    def _fake_errors(*args, **kwargs):
        del args, kwargs
        return {
            "SE_log10K": np.array([0.1], dtype=float),
            "percK": np.array([1.0], dtype=float),
            "covfit": 1.0,
            "rms": 0.01,
            "residuals_vec": np.array([0.01, -0.01], dtype=float),
            "stability_diag": {},
        }

    class _ImmediateFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _FakeExecutor:
        def __init__(self, max_workers=None):
            del max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return False

        def submit(self, fn, payload):
            return _ImmediateFuture(fn(payload))

    def _fake_wait(pending, timeout=None, return_when=None):
        del timeout, return_when
        if not pending:
            return set(), set()
        one = next(iter(pending))
        rest = set(pending)
        rest.remove(one)
        return {one}, rest

    def _fake_worker(payload):
        seed = int(payload.get("seed") or 0)
        mapping = {
            11: (0.4, [1.1]),
            22: (0.1, [2.2]),
            33: (0.3, [3.3]),
        }
        run_rms, run_full = mapping.get(seed, (1.0, [0.0]))
        return {
            "ok": True,
            "error": "",
            "objective_evaluations": 7,
            "nfev": 3,
            "run_rms": float(run_rms),
            "run_full": list(run_full),
            "success": True,
            "message": "ok",
        }

    monkeypatch.setattr(nmr_mod.pd, "read_excel", _fake_read_excel)
    monkeypatch.setattr(nmr_mod, "NewtonRaphson", _DummyAlgo)
    monkeypatch.setattr(nmr_mod, "LevenbergMarquardt", _DummyAlgo)
    monkeypatch.setattr(nmr_mod, "project_coeffs_block_onp_frac", _fake_project)
    monkeypatch.setattr(nmr_mod, "compute_errors_nmr_varpro", _fake_errors)
    monkeypatch.setattr(nmr_mod, "generate_figure_base64", lambda *a, **k: "img")
    monkeypatch.setattr(nmr_mod, "generate_figure2_base64", lambda *a, **k: "img2")
    monkeypatch.setattr(nmr_mod, "ProcessPoolExecutor", _FakeExecutor)
    monkeypatch.setattr(nmr_mod, "wait", _fake_wait)
    monkeypatch.setattr(nmr_mod, "_run_nmr_single_start", _fake_worker)

    out = nmr_mod.process_nmr_data(
        file_path="dummy.xlsx",
        spectra_sheet="nmr",
        conc_sheet="conc",
        column_names=["H", "G"],
        signal_names=["sig1"],
        receptor_label="H",
        guest_label="G",
        model_matrix=[[1, 0], [0, 1], [1, 1]],
        k_initial=[1.0],
        k_bounds=[{"min": -2, "max": 5}],
        algorithm="Newton-Raphson",
        optimizer="powell",
        model_settings="Free",
        non_absorbent_species=[],
        k_fixed=[False],
        multi_start_runs=3,
        multi_start_seeds=[11, 22, 33],
        multi_start_parallel=True,
        multi_start_max_workers=2,
    )

    assert out.get("success") is True, out
    k_best = float(out["export_data"]["k"][0])
    assert abs(k_best - 2.2) < 1e-12
    assert int(out["objective_evaluations"]) >= 21
