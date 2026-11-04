"""Production-backed simulation adapters; currently UV-Vis 1:1 analytical only.

Other techniques and uncertainty methods are deliberately rejected until they
have independent tests; this module is not the requested complete campaign.
"""
from __future__ import annotations

import contextlib
import io
import math
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.stats import t


def run_one(config: dict, replicate: int, seed: int) -> dict:
    if config["scenario"] != "uvvis_11_iid" or config["method"] != "analytic":
        raise ValueError("Only uvvis_11_iid / analytic is implemented")
    estimator = str(config.get("estimator", "soft_penalty"))
    if estimator not in ("soft_penalty", "ols", "nnls"):
        raise ValueError(f"unknown estimator {estimator!r}")
    from hmfit_core import run_spectroscopy
    from hmfit_core.solvers import NewtonRaphson
    from hmfit_core.utils.errors import compute_errors_spectro_varpro
    from tests.phase1_cases import _spectro_cfg, _spectro_dataset

    model = [[1, 0], [0, 1], [1, 1]]
    truth = 3.0
    row = {"scenario": config["scenario"], "replicate": replicate, "seed": seed,
           "method": "analytic", "estimator": estimator,
           "parameter": "log10K", "truth": truth,
           "estimate": None, "se": None, "ci_low": None, "ci_high": None,
           "dof": None, "success": False, "error": "", "bootstrap_failed": None,
           "nfev": None, "seconds": None}
    start = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix="hmfit-validation-") as tmp:
            path = Path(tmp) / "spectra.xlsx"
            _spectro_dataset(path, [truth], model, n_points=int(config["points"]),
                             noise=float(config["noise"]), seed=seed)
            cfg = _spectro_cfg(path, model, [2.5], [(0.0, 6.0)])
            cfg["render_graphs"] = False
            # "ols" uses the production soft-penalty path with mu=0 (no penalty
            # term): equivalence with the unconstrained projector is locked by
            # tests/test_uvvis_bias_reference.py.
            if estimator == "nnls":
                cfg["eps_solver_mode"] = "nnls_hard"
            elif estimator == "ols":
                cfg["mu"] = 0.0
            with contextlib.redirect_stdout(io.StringIO()):
                output = run_spectroscopy(cfg)
            if not output.get("success") or not output["optimizer_result"]["success"]:
                raise RuntimeError("production optimizer reported non-convergence")
            ed = output["export_data"]
            theta = np.asarray(ed["k"], dtype=float)
            se = np.asarray(ed["SE_log10K"], dtype=float)
            if theta.shape != (1,) or se.shape != (1,) or not np.isfinite(theta[0]) or not np.isfinite(se[0]):
                raise RuntimeError("non-finite production estimate or standard error")
            solver = NewtonRaphson(np.asarray(ed["C_T"], float), np.asarray(ed["modelo"], float),
                                   nas=[], model_sett="Free")
            metrics = compute_errors_spectro_varpro(theta, solver, np.asarray(ed["Y"], float),
                                                     np.asarray(ed["modelo"], float), [],
                                                     weights=np.asarray(ed["weights"], float))
            dof = int(metrics["dof"])
            if dof <= 0:
                raise RuntimeError("non-positive residual degrees of freedom")
            q = float(t.ppf(0.975, dof))
            row.update(estimate=float(theta[0]), se=float(se[0]), dof=dof,
                       ci_low=float(theta[0] - q * se[0]), ci_high=float(theta[0] + q * se[0]),
                       nfev=int(output["optimizer_result"]["nfev"]), success=True)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = time.perf_counter() - start
    return row
