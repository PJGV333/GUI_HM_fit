from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def coerce_param_transform(
    transform: object,
    *,
    expected_rows: int | None = None,
) -> np.ndarray | None:
    if transform is None:
        return None
    arr = np.asarray(transform, dtype=float)
    if arr.size == 0:
        return None
    if arr.ndim != 2:
        raise ValueError(f"Parameter transform must be 2D, got shape={arr.shape}.")
    if expected_rows is not None and int(arr.shape[0]) != int(expected_rows):
        raise ValueError(
            f"Parameter transform row mismatch: expected {int(expected_rows)}, got {int(arr.shape[0])}."
        )
    if arr.shape[1] <= 0:
        raise ValueError("Parameter transform must contain at least one parameter column.")
    return arr.astype(float, copy=False)


def expand_solver_params(params: object, transform: object | None = None) -> np.ndarray:
    theta = np.asarray(params, dtype=float).ravel()
    matrix = coerce_param_transform(transform)
    if matrix is None:
        return theta
    if theta.size != matrix.shape[1]:
        raise ValueError(
            f"Parameter vector size mismatch: expected {matrix.shape[1]}, got {theta.size}."
        )
    return np.asarray(matrix @ theta, dtype=float).ravel()


@dataclass
class SolverParamTransformWrapper:
    solver: Any
    transform: np.ndarray

    def __post_init__(self) -> None:
        self.transform = coerce_param_transform(self.transform)
        if self.transform is None:
            raise ValueError("SolverParamTransformWrapper requires a non-empty transform.")

    def solver_params(self, params: object) -> np.ndarray:
        return expand_solver_params(params, self.transform)

    def concentraciones(self, params: object):
        return self.solver.concentraciones(self.solver_params(params))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.solver, name)
