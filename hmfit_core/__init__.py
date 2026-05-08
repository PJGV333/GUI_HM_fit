"""Core services for HM Fit (no GUI, no FastAPI server)."""

from .exports import write_results_xlsx
from .run_nmr import run_nmr
from .run_spectroscopy import run_spectroscopy


def run_acid_base(*args, **kwargs):
    from .run_acid_base import run_acid_base as _run_acid_base

    return _run_acid_base(*args, **kwargs)

__all__ = [
    "run_acid_base",
    "run_spectroscopy",
    "run_nmr",
    "write_results_xlsx",
]
