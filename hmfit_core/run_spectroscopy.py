# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, Optional

ProgressCallback = Optional[Callable[[str], None]]


def _normalize_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dict-like mapping or a dataclass instance")


def run_spectroscopy(config: Mapping[str, Any] | Any, progress_cb: ProgressCallback = None) -> dict[str, Any]:
    """
    Run the spectroscopy workflow using `hmfit_core.processors.spectroscopy_processor`.

    This directly calls `process_spectroscopy_data(...)`.
    """
    from hmfit_core.processors.spectroscopy_processor import process_spectroscopy_data, set_progress_callback

    cfg = _normalize_config(config)
    set_progress_callback(progress_cb, loop=None)

    multi_runs = int(cfg.get("multi_start_runs", 1) or 1)
    multi_seeds = cfg.get("multi_start_seeds")

    return process_spectroscopy_data(
        file_path=str(cfg["file_path"]),
        spectra_sheet=str(cfg["spectra_sheet"]),
        conc_sheet=str(cfg["conc_sheet"]),
        column_names=list(cfg["column_names"]),
        receptor_label=cfg.get("receptor_label") or None,
        guest_label=cfg.get("guest_label") or None,
        efa_enabled=bool(cfg.get("efa_enabled", False)),
        efa_eigenvalues=int(cfg.get("efa_eigenvalues", 0) or 0),
        modelo=cfg.get("modelo") or [],
        non_abs_species=list(cfg.get("non_abs_species") or []),
        algorithm=str(cfg.get("algorithm") or "Newton-Raphson"),
        model_settings=str(cfg.get("model_settings") or "Free"),
        optimizer=str(cfg.get("optimizer") or "powell"),
        initial_k=list(cfg.get("initial_k") or []),
        bounds=list(cfg.get("bounds") or []),
        channels_raw=str(cfg.get("channels_raw") or "All"),
        channels_mode=str(cfg.get("channels_mode") or "all"),
        channels_resolved=list(cfg.get("channels_resolved") or []),
        show_stability_diagnostics=bool(cfg.get("show_stability_diagnostics", False)),
        multi_start_runs=multi_runs,
        multi_start_seeds=multi_seeds,
        baseline_mode=str(cfg.get("baseline_mode") or "off"),
        baseline_start=cfg.get("baseline_start", 450.0),
        baseline_end=cfg.get("baseline_end", 600.0),
        baseline_auto_quantile=cfg.get("baseline_auto_quantile", 0.20),
        baseline_apply_per_spectrum=bool(cfg.get("baseline_apply_per_spectrum", True)),
        weighting_mode=str(cfg.get("weighting_mode") or "none"),
        weighting_eps=cfg.get("weighting_eps", 1e-12),
        weighting_power=cfg.get("weighting_power", 1.0),
        weighting_normalize=bool(cfg.get("weighting_normalize", True)),
        eps_solver_mode=str(cfg.get("eps_solver_mode") or "soft_penalty"),
        eps_mu=cfg.get("mu", 1e-2),
        delta_mode=str(cfg.get("delta_mode") or "off"),
        delta_rel=cfg.get("delta_rel", 0.01),
        alpha_smooth=cfg.get("alpha_smooth", 0.0),
    )
