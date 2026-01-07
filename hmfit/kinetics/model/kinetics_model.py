"""Kinetics model construction and ODE integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.integrate import solve_ivp

from ..data.dataset import KineticsDataset
from ..mechanism_editor.ast import MechanismAST
from .rate_laws import mass_action_rate
from .reactions import Reaction, expand_reactions
from .temperature import ParamResolver, TemperatureType


@dataclass(frozen=True)
class KineticsContext:
    fixed_conc: Mapping[str, float]
    temperature: TemperatureType


class KineticsModel:
    """Builds and solves a kinetic ODE model from a MechanismAST."""

    def __init__(
        self, mechanism: MechanismAST, param_resolver: ParamResolver | None = None
    ) -> None:
        self.mechanism = mechanism
        self.dynamic_species = [
            species for species in mechanism.species if species not in mechanism.fixed
        ]
        self.fixed_species = set(mechanism.fixed)
        self._validate_species_lists()

        self.reactions = expand_reactions(mechanism)
        self._validate_reaction_species()

        self.species_index = {
            species: index for index, species in enumerate(self.dynamic_species)
        }
        self.S = build_stoichiometric_matrix(self.dynamic_species, self.reactions)
        self.param_resolver = param_resolver or ParamResolver(mechanism.temp_models)

    def rhs(
        self,
        t: float,
        y: np.ndarray,
        params: Mapping[str, float],
        fixed_conc: Mapping[str, float],
        temperature: TemperatureType,
    ) -> np.ndarray:
        temp_value = temperature(t) if callable(temperature) else temperature
        k_values = self.param_resolver.resolve(params, temp_value)

        rates = np.empty(len(self.reactions), dtype=float)
        for idx, reaction in enumerate(self.reactions):
            if reaction.k_name not in k_values:
                raise KeyError(f"Missing kinetic parameter '{reaction.k_name}'.")
            rates[idx] = mass_action_rate(
                reaction.reactants,
                y,
                fixed_conc,
                self.species_index,
                k_values[reaction.k_name],
            )

        return self.S @ rates

    def solve_concentrations(
        self,
        t_grid: np.ndarray,
        y0: Mapping[str, float] | np.ndarray | Sequence[float] | None,
        params: Mapping[str, float],
        context: KineticsContext | KineticsDataset | None = None,
        *,
        method: str = "BDF",
        rtol: float = 1e-8,
        atol: float = 1e-12,
    ) -> np.ndarray:
        if y0 is None:
            if context is None or not hasattr(context, "y0"):
                raise ValueError("Initial conditions y0 are required.")
            y0 = context.y0

        y0_vec = _coerce_y0(y0, self.dynamic_species)
        fixed_conc = _extract_fixed_conc(context)
        temperature = _extract_temperature(context)

        sol = solve_ivp(
            lambda t, y: self.rhs(t, y, params, fixed_conc, temperature),
            t_span=(float(t_grid[0]), float(t_grid[-1])),
            y0=y0_vec,
            t_eval=t_grid,
            method=method,
            rtol=rtol,
            atol=atol,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        return sol.y.T

    def _validate_species_lists(self) -> None:
        species_set = set(self.mechanism.species)
        missing_fixed = self.fixed_species - species_set
        if missing_fixed:
            missing = ", ".join(sorted(missing_fixed))
            raise ValueError(f"Fixed species not declared in species list: {missing}")

    def _validate_reaction_species(self) -> None:
        species_set = set(self.mechanism.species)
        for reaction in self.mechanism.reactions:
            for species in list(reaction.reactants) + list(reaction.products):
                if species not in species_set:
                    raise ValueError(
                        f"Reaction uses undeclared species '{species}'."
                    )


def build_stoichiometric_matrix(
    dynamic_species: Sequence[str], reactions: Sequence[Reaction]
) -> np.ndarray:
    """Build the stoichiometric matrix for dynamic species."""
    idx_map = {species: index for index, species in enumerate(dynamic_species)}
    matrix = np.zeros((len(dynamic_species), len(reactions)), dtype=float)
    for j, reaction in enumerate(reactions):
        for species, coeff in reaction.products.items():
            if species in idx_map:
                matrix[idx_map[species], j] += coeff
        for species, coeff in reaction.reactants.items():
            if species in idx_map:
                matrix[idx_map[species], j] -= coeff
    return matrix


def _coerce_y0(
    y0: Mapping[str, float] | np.ndarray | Sequence[float],
    dynamic_species: Sequence[str],
) -> np.ndarray:
    if isinstance(y0, Mapping):
        missing = [species for species in dynamic_species if species not in y0]
        if missing:
            missing_str = ", ".join(missing)
            raise KeyError(f"Missing initial conditions for: {missing_str}")
        return np.array([float(y0[species]) for species in dynamic_species], dtype=float)

    arr = np.asarray(y0, dtype=float).reshape(-1)
    if arr.shape[0] != len(dynamic_species):
        raise ValueError(
            "Initial condition array length does not match dynamic species."
        )
    return arr


def _extract_fixed_conc(
    context: KineticsContext | KineticsDataset | None,
) -> Mapping[str, float]:
    if context is None:
        return {}
    fixed_conc = getattr(context, "fixed_conc", {})
    return fixed_conc if fixed_conc is not None else {}


def _extract_temperature(
    context: KineticsContext | KineticsDataset | None,
) -> TemperatureType:
    if context is None:
        return 298.15
    temperature = getattr(context, "temperature", 298.15)
    return temperature if temperature is not None else 298.15
