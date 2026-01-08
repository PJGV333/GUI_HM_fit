"""Objective function for global kinetics fits."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from ..data.dataset import KineticsDataset
from ..model.kinetics_model import KineticsModel
from ..observation.linear_matrix import prepare_weights
from .variable_projection import solve_A_ls, solve_A_nnls


@dataclass
class GlobalKineticsObjective:
    model: KineticsModel
    datasets: Sequence[KineticsDataset]
    param_names: Sequence[str] | None = None
    nnls: bool = False
    log_params: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.log_params = set(self.log_params)

    def predict_dataset(
        self, params: Mapping[str, float] | np.ndarray, dataset: KineticsDataset
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        params_dict = self._coerce_params(params)
        if dataset.D is None:
            raise ValueError("Dataset is missing observed data D.")

        try:
            C = self.model.solve_concentrations(
                dataset.t, dataset.y0, params_dict, dataset
            )
        except RuntimeError as exc:
            name = dataset.name or "dataset"
            raise RuntimeError(f"ODE solver failed for {name}: {exc}") from exc
        weights = _dataset_weights(dataset)
        if self.nnls:
            A, D_hat = solve_A_nnls(C, dataset.D, weights=weights)
        else:
            A, D_hat = solve_A_ls(C, dataset.D, weights=weights)
        return C, A, D_hat

    def residuals(self, params: Mapping[str, float] | np.ndarray) -> np.ndarray:
        params_dict = self._coerce_params(params)
        residuals: list[np.ndarray] = []

        for dataset in self.datasets:
            if dataset.D is None:
                raise ValueError("Dataset is missing observed data D.")

            try:
                C = self.model.solve_concentrations(
                    dataset.t, dataset.y0, params_dict, dataset
                )
            except RuntimeError as exc:
                name = dataset.name or "dataset"
                raise RuntimeError(f"ODE solver failed for {name}: {exc}") from exc
            weights = _dataset_weights(dataset)
            if self.nnls:
                A, D_hat = solve_A_nnls(C, dataset.D, weights=weights)
            else:
                A, D_hat = solve_A_ls(C, dataset.D, weights=weights)
            resid = dataset.D - D_hat

            if weights is not None:
                resid = resid * weights

            residuals.append(resid.ravel())

        if not residuals:
            return np.array([], dtype=float)

        return np.concatenate(residuals)

    def _coerce_params(self, params: Mapping[str, float] | np.ndarray) -> dict[str, float]:
        if isinstance(params, Mapping):
            return dict(params)

        if self.param_names is None:
            raise ValueError("Parameter names must be provided for array params.")

        values = np.asarray(params, dtype=float).reshape(-1)
        if values.shape[0] != len(self.param_names):
            raise ValueError("Parameter vector length does not match param_names.")
        params_dict: dict[str, float] = {}
        for name, value in zip(self.param_names, values, strict=True):
            if name in self.log_params:
                params_dict[name] = float(10.0**value)
            else:
                params_dict[name] = float(value)
        return params_dict


def _dataset_weights(dataset: KineticsDataset) -> np.ndarray | None:
    if dataset.weights is not None and dataset.sigma is not None:
        raise ValueError("Provide either weights or sigma, not both.")

    if dataset.weights is not None:
        weights = dataset.weights
    elif dataset.sigma is not None:
        if np.any(np.asarray(dataset.sigma) <= 0):
            raise ValueError("Sigma values must be positive.")
        weights = 1.0 / np.asarray(dataset.sigma)
    else:
        return None

    return prepare_weights(weights, dataset.D.shape[0], dataset.D.shape[1])
