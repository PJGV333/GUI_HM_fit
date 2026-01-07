"""Dataset definitions for kinetics ODE integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class KineticsDataset:
    """Minimal dataset/context for ODE integration."""

    t: np.ndarray
    y0: Mapping[str, float] | np.ndarray
    fixed_conc: Mapping[str, float]
    temperature: float | Callable[[float], float] = 298.15
    name: str | None = None
