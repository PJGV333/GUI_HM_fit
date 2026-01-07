"""Synthetic dataset generator for kinetics models."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import numpy as np

from ..data.dataset import KineticsDataset
from ..mechanism_editor.parser import parse_mechanism
from ..model.kinetics_model import KineticsModel, KineticsContext


def generate_dataset(
    mechanism_text: str,
    k_params: Mapping[str, float],
    A_true: np.ndarray,
    t_grid: np.ndarray,
    y0: Mapping[str, float] | np.ndarray,
    *,
    noise_level: float = 0.0,
    temperatures: float | Sequence[float] = 298.15,
    fixed_conc: Mapping[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> KineticsDataset | list[KineticsDataset]:
    """Generate synthetic datasets from a mechanism and parameters."""
    if noise_level < 0:
        raise ValueError("noise_level must be non-negative.")

    fixed_conc = fixed_conc or {}
    A_true = np.asarray(A_true, dtype=float)
    if A_true.ndim != 2:
        raise ValueError("A_true must be a 2D array.")

    temperature_list, is_sequence = _coerce_temperatures(temperatures)
    mechanism = parse_mechanism(mechanism_text)
    model = KineticsModel(mechanism)

    datasets: list[KineticsDataset] = []
    for temperature in temperature_list:
        context = KineticsContext(fixed_conc=fixed_conc, temperature=temperature)
        C = model.solve_concentrations(t_grid, y0, k_params, context)

        if C.shape[1] != A_true.shape[0]:
            raise ValueError("A_true rows must match number of dynamic species.")

        D = C @ A_true
        sigma = None
        if noise_level > 0:
            rng = rng or np.random.default_rng()
            D = D + rng.normal(0.0, noise_level, size=D.shape)
            sigma = np.full(D.shape, noise_level, dtype=float)

        datasets.append(
            KineticsDataset(
                t=t_grid,
                y0=y0,
                fixed_conc=fixed_conc,
                temperature=temperature,
                D=D,
                sigma=sigma,
            )
        )

    if is_sequence:
        return datasets
    return datasets[0]


def _coerce_temperatures(
    temperatures: float | Sequence[float],
) -> tuple[list[float], bool]:
    if isinstance(temperatures, (list, tuple, np.ndarray)):
        return [float(temp) for temp in temperatures], True
    return [float(temperatures)], False
