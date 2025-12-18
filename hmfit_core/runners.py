from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

ProgressCallback = Optional[Callable[[str], None]]


def run_spectroscopy(config: Dict[str, Any], progress_cb: ProgressCallback = None) -> Dict[str, Any]:
    """
    Run the spectroscopy workflow using `backend_fastapi.spectroscopy_processor` as a pure library.
    """
    from backend_fastapi.spectroscopy_processor import process_spectroscopy_data, set_progress_callback

    set_progress_callback(progress_cb, loop=None)

    return process_spectroscopy_data(
        file_path=str(config["file_path"]),
        spectra_sheet=str(config["spectra_sheet"]),
        conc_sheet=str(config["conc_sheet"]),
        column_names=list(config["column_names"]),
        receptor_label=config.get("receptor_label") or None,
        guest_label=config.get("guest_label") or None,
        efa_enabled=bool(config.get("efa_enabled", False)),
        efa_eigenvalues=int(config.get("efa_eigenvalues", 0) or 0),
        modelo=config.get("modelo") or [],
        non_abs_species=list(config.get("non_abs_species") or []),
        algorithm=str(config.get("algorithm") or "Newton-Raphson"),
        model_settings=str(config.get("model_settings") or "Free"),
        optimizer=str(config.get("optimizer") or "powell"),
        initial_k=list(config.get("initial_k") or []),
        bounds=list(config.get("bounds") or []),
        channels_raw=str(config.get("channels_raw") or "All"),
        channels_mode=str(config.get("channels_mode") or "all"),
        channels_resolved=list(config.get("channels_resolved") or []),
    )


def run_nmr(config: Dict[str, Any], progress_cb: ProgressCallback = None) -> Dict[str, Any]:
    """
    Run the NMR workflow using `backend_fastapi.nmr_processor` as a pure library.
    """
    from backend_fastapi.nmr_processor import process_nmr_data, set_progress_callback

    set_progress_callback(progress_cb, loop=None)

    bounds_raw = list(config.get("bounds") or [])
    k_bounds: List[Dict[str, float]] = []
    for b in bounds_raw:
        if isinstance(b, dict):
            k_bounds.append({"min": b.get("min"), "max": b.get("max")})
        elif isinstance(b, (list, tuple)) and len(b) >= 2:
            k_bounds.append({"min": b[0], "max": b[1]})
        else:
            k_bounds.append({"min": None, "max": None})

    return process_nmr_data(
        file_path=str(config["file_path"]),
        spectra_sheet=str(config["nmr_sheet"]),
        conc_sheet=str(config["conc_sheet"]),
        column_names=list(config["column_names"]),
        signal_names=list(config["signal_names"]),
        receptor_label=str(config.get("receptor_label") or ""),
        guest_label=str(config.get("guest_label") or ""),
        model_matrix=config.get("modelo") or [],
        k_initial=list(config.get("initial_k") or []),
        k_bounds=k_bounds,
        algorithm=str(config.get("algorithm") or "Newton-Raphson"),
        optimizer=str(config.get("optimizer") or "powell"),
        model_settings=str(config.get("model_settings") or "Free"),
        non_absorbent_species=list(config.get("non_abs_species") or []),
        k_fixed=list(config.get("k_fixed") or []) or None,
    )

