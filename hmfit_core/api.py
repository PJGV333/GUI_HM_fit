from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Mapping, Optional

ProgressCallback = Optional[Callable[[str], None]]
CancelCallback = Optional[Callable[[], bool]]


class FitCancelled(RuntimeError):
    pass


def _normalize_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, Mapping):
        return dict(config)
    raise TypeError("config must be a dict-like mapping or a dataclass instance")


def _normalize_nmr_config(config: Mapping[str, Any] | Any) -> dict[str, Any]:
    """
    Accepted config schema (compatible with wx + Tauri/FastAPI naming):

    Required:
    - file_path: str
    - conc_sheet: str
    - column_names: list[str]
    - signal_names: list[str]
    - modelo: list[list[float]]

    Sheet key (either):
    - nmr_sheet: str (wx / hmfit_core)
    - spectra_sheet: str (FastAPI endpoint naming; sheet with chemical shifts)
    - signals_sheet: str (legacy FastAPI form field, alias of spectra_sheet)

    Optional:
    - receptor_label: str
    - guest_label: str
    - non_abs_species: list[int]
    - algorithm: str
    - model_settings: str
    - optimizer: str
    - initial_k: list[float]
    - bounds: list[[min,max]] | list[{"min":..,"max":..}]
    - k_fixed / fixed_mask / fixed: list[bool]
    """
    cfg = _normalize_config(config)

    if "nmr_sheet" not in cfg:
        if "spectra_sheet" in cfg:
            cfg["nmr_sheet"] = cfg["spectra_sheet"]
        elif "signals_sheet" in cfg:
            cfg["nmr_sheet"] = cfg["signals_sheet"]

    if "k_fixed" not in cfg and "fixed_mask" not in cfg and "fixed" in cfg:
        cfg["k_fixed"] = cfg["fixed"]

    return cfg


def _wrap_progress(progress_cb: ProgressCallback, cancel: CancelCallback) -> ProgressCallback:
    if cancel is None:
        return progress_cb

    def _cb(msg: str) -> None:
        if cancel():
            raise FitCancelled("Fit cancelled.")
        if progress_cb is not None:
            progress_cb(msg)
        else:
            print(str(msg).rstrip())

    return _cb


def run_spectroscopy_fit(
    config: Mapping[str, Any] | Any,
    progress_cb: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> dict[str, Any]:
    from hmfit_core.run_spectroscopy import run_spectroscopy

    return run_spectroscopy(_normalize_config(config), progress_cb=_wrap_progress(progress_cb, cancel))


def run_nmr_fit(
    config: Mapping[str, Any] | Any,
    progress_cb: ProgressCallback = None,
    cancel: CancelCallback = None,
) -> dict[str, Any]:
    from hmfit_core.run_nmr import run_nmr

    return run_nmr(_normalize_nmr_config(config), progress_cb=_wrap_progress(progress_cb, cancel))
