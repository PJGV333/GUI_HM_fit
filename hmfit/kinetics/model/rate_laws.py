"""Rate law implementations."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def mass_action_rate(
    reactants: Mapping[str, int],
    y: np.ndarray,
    fixed_conc: Mapping[str, float],
    idx_map: Mapping[str, int],
    k: float,
) -> float:
    """Compute a mass-action rate for a single reaction."""
    rate = float(k)
    for species, coeff in reactants.items():
        if species in idx_map:
            conc = y[idx_map[species]]
        else:
            if species not in fixed_conc:
                raise ValueError(f"Missing fixed concentration for '{species}'.")
            conc = fixed_conc[species]
        rate *= conc ** coeff
    return rate
