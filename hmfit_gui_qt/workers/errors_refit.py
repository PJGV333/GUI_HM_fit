from __future__ import annotations

from typing import Any

import numpy as np


def refit_nmr_from_context(
    ctx: dict[str, Any],
    data_star,
    theta0,
    *,
    max_iter: int = 30,
    tol: float = 1e-8,
):
    try:
        from scipy import optimize
        from scipy.optimize import differential_evolution

        from hmfit_core.processors.nmr_processor import build_D_cols, project_coeffs_block_onp_frac
        from hmfit_core.processors.spectroscopy_processor import _build_bounds_list
        from hmfit_core.solvers import LevenbergMarquardt, NewtonRaphson

        dq_star = np.asarray(data_star, dtype=float)
        dq_obs_raw = ctx.get("dq")
        dq_obs = np.asarray(dq_obs_raw if dq_obs_raw is not None else [], dtype=float)
        c_t_raw = ctx.get("C_T")
        C_T = np.asarray(c_t_raw if c_t_raw is not None else [], dtype=float)
        modelo_raw = ctx.get("modelo_solver")
        modelo = np.asarray(modelo_raw if modelo_raw is not None else [], dtype=float)
        nas_raw = ctx.get("non_abs_species")
        nas = list(nas_raw) if nas_raw is not None else []
        algorithm = str(ctx.get("algorithm") or "Newton-Raphson")
        model_settings = str(ctx.get("model_settings") or "Free")
        optimizer = str(ctx.get("optimizer") or "powell")
        bounds_raw = ctx.get("bounds")
        bounds_raw = list(bounds_raw) if bounds_raw is not None else []
        column_names_raw = ctx.get("column_names")
        column_names = list(column_names_raw) if column_names_raw is not None else []
        signal_names_raw = ctx.get("signal_names")
        signal_names = list(signal_names_raw) if signal_names_raw is not None else []
        parent_idx_raw = ctx.get("parent_idx")
        parent_idx = list(parent_idx_raw) if parent_idx_raw is not None else []

        if algorithm == "Newton-Raphson":
            res = NewtonRaphson(C_T, modelo, nas, model_settings)
        elif algorithm == "Levenberg-Marquardt":
            res = LevenbergMarquardt(C_T, modelo, nas, model_settings)
        else:
            return np.asarray(theta0, dtype=float), False, {"error": f"Unknown algorithm: {algorithm}"}

        D_cols = ctx.get("D_cols")
        if D_cols is None or len(np.asarray(D_cols).shape) == 0:
            if not column_names or not signal_names:
                return np.asarray(theta0, dtype=float), False, {"error": "Missing D_cols and signal metadata."}
            D_cols, parent_idx = build_D_cols(C_T, column_names, signal_names, default_idx=0)
        D_cols = np.asarray(D_cols, dtype=float)
        if not parent_idx:
            parent_idx_ctx = ctx.get("parent_idx")
            parent_idx = list(parent_idx_ctx) if parent_idx_ctx is not None else []

        mask = ctx.get("mask")
        if mask is None:
            base = dq_obs if dq_obs.size else dq_star
            mask = np.isfinite(base) & np.isfinite(D_cols) & (np.abs(D_cols) > 0)
        mask = np.asarray(mask, dtype=bool)

        theta0 = np.asarray(theta0, dtype=float).ravel()
        p0_full = theta0.copy()
        fixed_mask_raw = ctx.get("fixed_mask")
        fixed_mask = (
            np.asarray(fixed_mask_raw, dtype=bool)
            if fixed_mask_raw is not None
            else np.zeros(p0_full.size, dtype=bool)
        )
        bnds = _build_bounds_list(bounds_raw)
        if len(bnds) < p0_full.size:
            bnds.extend([(-np.inf, np.inf)] * (p0_full.size - len(bnds)))
        if len(bnds) > p0_full.size:
            bnds = bnds[: p0_full.size]
        for i, (lb, ub) in enumerate(bnds):
            if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
                fixed_mask[i] = True
        free_idx = np.where(~fixed_mask)[0]

        modelo_abs = modelo
        if nas:
            modelo_abs = np.delete(modelo, nas, axis=0)

        def pack(theta_free: np.ndarray) -> np.ndarray:
            k_full = p0_full.copy()
            if free_idx.size:
                k_full[free_idx] = np.asarray(theta_free, dtype=float).ravel()
            return k_full

        def f_m(theta_free: np.ndarray) -> float:
            try:
                k_curr_full = pack(theta_free)
                C = res.concentraciones(k_curr_full)[0]
                dq_cal = project_coeffs_block_onp_frac(
                    dq_star,
                    C,
                    D_cols,
                    mask,
                    modelo=modelo_abs,
                    parent_idx=parent_idx,
                    rcond=1e-10,
                    ridge=1e-8,
                )
                diff = dq_star - dq_cal
                valid_residuals = mask & np.isfinite(dq_cal)
                r = diff[valid_residuals].ravel()
                if (r.size <= np.asarray(theta_free).ravel().size) or (not np.isfinite(r).all()):
                    return 1e9
                return float(np.sqrt(np.mean(r * r)))
            except Exception:
                return 1e9

        n_iter = 0
        success = True
        if free_idx.size == 0:
            theta_star = p0_full.copy()
        else:
            k_free0 = p0_full[free_idx]
            bounds_free = [bnds[i] for i in free_idx]
            if optimizer == "differential_evolution":
                opt_res = differential_evolution(
                    f_m,
                    bounds_free,
                    x0=k_free0,
                    strategy="best1bin",
                    maxiter=int(max_iter),
                    popsize=15,
                    tol=float(tol),
                    mutation=(0.5, 1),
                    recombination=0.7,
                    init="latinhypercube",
                )
            else:
                options = {"maxiter": int(max_iter)}
                if tol is not None:
                    options.update(
                        {
                            "xtol": float(tol),
                            "ftol": float(tol),
                            "gtol": float(tol),
                            "xatol": float(tol),
                            "fatol": float(tol),
                        }
                    )
                opt_res = optimize.minimize(
                    f_m, k_free0, method=optimizer, bounds=bounds_free, options=options
                )
            theta_star = pack(opt_res.x)
            success = bool(getattr(opt_res, "success", True))
            n_iter = int(getattr(opt_res, "nit", 0) or 0)

        final_rms = f_m(theta_star[free_idx] if free_idx.size else np.array([]))
        theta_star = np.asarray(theta_star, dtype=float).ravel()
        ok = bool(np.isfinite(final_rms) and np.all(np.isfinite(theta_star)))
        info = {"n_iter": n_iter, "final_rms": final_rms, "success": success}
        return theta_star, ok, info
    except Exception as exc:
        return np.asarray(theta0, dtype=float), False, {"error": str(exc)}


def refit_spectro_from_context(
    ctx: dict[str, Any],
    data_star,
    theta0,
    *,
    max_iter: int = 30,
    tol: float = 1e-8,
):
    try:
        from scipy import optimize
        from scipy.optimize import differential_evolution

        from hmfit_core.processors.spectroscopy_processor import _build_bounds_list
        from hmfit_core.solvers import LevenbergMarquardt, NewtonRaphson
        from hmfit_core.utils.nnls_utils import solve_A_nnls_pgd

        Y_star = np.asarray(data_star, dtype=float)
        c_t_raw = ctx.get("C_T")
        C_T = np.asarray(c_t_raw if c_t_raw is not None else [], dtype=float)
        modelo_raw = ctx.get("modelo_solver")
        modelo = np.asarray(modelo_raw if modelo_raw is not None else [], dtype=float)
        nas_raw = ctx.get("non_abs_species")
        nas = list(nas_raw) if nas_raw is not None else []
        algorithm = str(ctx.get("algorithm") or "Newton-Raphson")
        model_settings = str(ctx.get("model_settings") or "Free")
        optimizer = str(ctx.get("optimizer") or "powell")
        bounds_raw = ctx.get("bounds")
        bounds_raw = list(bounds_raw) if bounds_raw is not None else []

        if algorithm == "Newton-Raphson":
            res = NewtonRaphson(C_T, modelo, nas, model_settings)
        elif algorithm == "Levenberg-Marquardt":
            res = LevenbergMarquardt(C_T, modelo, nas, model_settings)
        else:
            return np.asarray(theta0, dtype=float), False, {"error": f"Unknown algorithm: {algorithm}"}

        theta0 = np.asarray(theta0, dtype=float).ravel()
        p0_full = theta0.copy()
        fixed_mask_raw = ctx.get("fixed_mask")
        fixed_mask = (
            np.asarray(fixed_mask_raw, dtype=bool)
            if fixed_mask_raw is not None
            else np.zeros(p0_full.size, dtype=bool)
        )
        bnds = _build_bounds_list(bounds_raw)
        if len(bnds) < p0_full.size:
            bnds.extend([(-np.inf, np.inf)] * (p0_full.size - len(bnds)))
        if len(bnds) > p0_full.size:
            bnds = bnds[: p0_full.size]
        for i, (lb, ub) in enumerate(bnds):
            if np.isfinite(lb) and np.isfinite(ub) and lb == ub:
                fixed_mask[i] = True
        free_idx = np.where(~fixed_mask)[0]

        def pack(theta_free: np.ndarray) -> np.ndarray:
            k_full = p0_full.copy()
            if free_idx.size:
                k_full[free_idx] = np.asarray(theta_free, dtype=float).ravel()
            return k_full

        def f_m(theta_free: np.ndarray) -> float:
            try:
                k_curr_full = pack(theta_free)
                C = res.concentraciones(k_curr_full)[0]
                A = solve_A_nnls_pgd(C, Y_star.T, ridge=0.0, max_iters=300)
                r = C @ A - Y_star.T
                rms = float(np.sqrt(np.mean(np.square(r))))
                if np.isnan(rms) or np.isinf(rms):
                    return 1e50
                return rms
            except Exception:
                return 1e50

        n_iter = 0
        success = True
        if free_idx.size == 0:
            theta_star = p0_full.copy()
        else:
            k_free0 = p0_full[free_idx]
            bounds_free = [bnds[i] for i in free_idx]
            if optimizer == "differential_evolution":
                opt_res = differential_evolution(
                    f_m,
                    bounds_free,
                    x0=k_free0,
                    strategy="best1bin",
                    maxiter=int(max_iter),
                    popsize=15,
                    tol=float(tol),
                    mutation=(0.5, 1),
                    recombination=0.7,
                    init="latinhypercube",
                )
            else:
                options = {"maxiter": int(max_iter)}
                if tol is not None:
                    options.update(
                        {
                            "xtol": float(tol),
                            "ftol": float(tol),
                            "gtol": float(tol),
                            "xatol": float(tol),
                            "fatol": float(tol),
                        }
                    )
                opt_res = optimize.minimize(
                    f_m, k_free0, method=optimizer, bounds=bounds_free, options=options
                )
            theta_star = pack(opt_res.x)
            success = bool(getattr(opt_res, "success", True))
            n_iter = int(getattr(opt_res, "nit", 0) or 0)

        final_rms = f_m(theta_star[free_idx] if free_idx.size else np.array([]))
        theta_star = np.asarray(theta_star, dtype=float).ravel()
        ok = bool(np.isfinite(final_rms) and np.all(np.isfinite(theta_star)))
        info = {"n_iter": n_iter, "final_rms": final_rms, "success": success}
        return theta_star, ok, info
    except Exception as exc:
        return np.asarray(theta0, dtype=float), False, {"error": str(exc)}
