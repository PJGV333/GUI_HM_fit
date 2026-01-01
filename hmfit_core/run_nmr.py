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


def run_nmr(config: Mapping[str, Any] | Any, progress_cb: ProgressCallback = None) -> dict[str, Any]:
    """
    Run the NMR workflow using `hmfit_core.processors.nmr_processor`.

    This directly calls `process_nmr_data(...)`.
    """
    from hmfit_core.processors.nmr_processor import process_nmr_data, set_progress_callback

    cfg = _normalize_config(config)
    set_progress_callback(progress_cb, loop=None)

    bounds_raw = list(cfg.get("bounds") or [])
    k_bounds: list[dict[str, float]] = []
    for b in bounds_raw:
        if isinstance(b, dict):
            k_bounds.append({"min": b.get("min"), "max": b.get("max")})
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            k_bounds.append({"min": b[0], "max": b[1]})
        else:
            k_bounds.append({"min": None, "max": None})

    stoich_map = cfg.get("stoichiometry_map") or cfg.get("stoichiometry")
    if progress_cb is not None:
        progress_cb(f"[DEBUG] stoichiometry_map present? {stoich_map is not None}")
        if stoich_map is not None:
            try:
                import numpy as np

                arr = np.asarray(stoich_map)
                min_val = "NA"
                max_val = "NA"
                if arr.size:
                    try:
                        min_val = float(np.nanmin(arr))
                        max_val = float(np.nanmax(arr))
                    except Exception:
                        min_val = "NA"
                        max_val = "NA"
                progress_cb(
                    f"[DEBUG] stoichiometry_map raw shape={arr.shape}, min={min_val}, max={max_val}"
                )
            except Exception as exc:
                progress_cb(f"[DEBUG] stoichiometry_map raw shape=<error: {exc}>")

    return process_nmr_data(
        file_path=str(cfg["file_path"]),
        spectra_sheet=str(cfg["nmr_sheet"]),
        conc_sheet=str(cfg["conc_sheet"]),
        column_names=list(cfg["column_names"]),
        signal_names=list(cfg["signal_names"]),
        receptor_label=str(cfg.get("receptor_label") or ""),
        guest_label=str(cfg.get("guest_label") or ""),
        model_matrix=cfg.get("modelo") or [],
        k_initial=list(cfg.get("initial_k") or []),
        k_bounds=k_bounds,
        algorithm=str(cfg.get("algorithm") or "Newton-Raphson"),
        optimizer=str(cfg.get("optimizer") or "powell"),
        model_settings=str(cfg.get("model_settings") or "Free"),
        non_absorbent_species=list(cfg.get("non_abs_species") or []),
        k_fixed=list((cfg.get("k_fixed") if cfg.get("k_fixed") is not None else cfg.get("fixed_mask")) or []) or None,
        stoichiometry_map=stoich_map,
        show_stability_diagnostics=bool(cfg.get("show_stability_diagnostics", False)),
    )
