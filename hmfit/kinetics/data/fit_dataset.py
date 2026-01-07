"""Dataset definitions for kinetics fits and GUI usage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .dataset import KineticsDataset


TechniqueType = Literal["spec_full", "spec_channels", "nmr_integrals", "nmr_full"]


@dataclass
class KineticsFitDataset(KineticsDataset):
    """Dataset with observed signals and metadata for fitting."""

    D: np.ndarray | None = None
    x: np.ndarray | None = None
    channel_labels: list[str] = field(default_factory=list)
    technique: TechniqueType = "spec_channels"
    time_unit: str = "s"
    x_unit: str = ""
    signal_unit: str = ""
    name: str = "Dataset"
    source_path: str | None = None
    loader_kind: str | None = None
    loader_delimiter: str | None = None
    loader_header: bool = True
    loader_transpose: bool = False
    loader_time_col: int | str = 0
    loader_sheet: str | None = None
    channel_indices: list[int] = field(default_factory=list)
    fit_C: np.ndarray | None = None
    fit_A: np.ndarray | None = None
    fit_D_hat: np.ndarray | None = None
    fit_residuals: np.ndarray | None = None
