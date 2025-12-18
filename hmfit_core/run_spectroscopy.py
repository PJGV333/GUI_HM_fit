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
    Run the spectroscopy workflow using `backend_fastapi.spectroscopy_processor` as a pure library.

    This must NOT start a FastAPI server; it directly calls `process_spectroscopy_data(...)`.
    """
    from backend_fastapi.spectroscopy_processor import process_spectroscopy_data, set_progress_callback

    cfg = _normalize_config(config)
    set_progress_callback(progress_cb, loop=None)

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
    )

