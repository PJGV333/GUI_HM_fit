"""Dataset definitions for kinetics ODE integration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np


@dataclass
class KineticsDataset:
    """Minimal dataset/context for ODE integration."""

    t: np.ndarray
    y0: Mapping[str, float] | np.ndarray | None
    fixed_conc: Mapping[str, float] = field(default_factory=dict)
    temperature: float | Callable[[float], float] = 298.15
    name: str | None = None
    D: np.ndarray | None = None
    sigma: np.ndarray | None = None
    weights: np.ndarray | None = None
