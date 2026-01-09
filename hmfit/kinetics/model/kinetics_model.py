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
        temp_value = float(temp_value)
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
        t_grid = _validate_time_grid(t_grid)
        if y0 is None:
            if context is None:
                raise ValueError("Initial conditions y0 are required.")
            y0 = getattr(context, "y0", None)
            if y0 is None:
                raise ValueError("Initial conditions y0 are required.")

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

    def solve_concentrations_jax(
        self,
        t_grid: np.ndarray,
        y0: Mapping[str, float] | np.ndarray | Sequence[float] | None,
        params: Mapping[str, float],
        context: KineticsContext | KineticsDataset | None = None,
    ):
        try:
            import jax.numpy as jnp
            from jax import lax
        except Exception as exc:
            raise RuntimeError("JAX backend is not available.") from exc

        t_np = _validate_time_grid(t_grid)
        if y0 is None:
            if context is None:
                raise ValueError("Initial conditions y0 are required.")
            y0 = getattr(context, "y0", None)
            if y0 is None:
                raise ValueError("Initial conditions y0 are required.")

        y0_vec = _coerce_y0(y0, self.dynamic_species)
        fixed_conc = _extract_fixed_conc(context)
        temperature = _extract_temperature(context)
        if callable(temperature):
            raise ValueError("Callable temperature is not supported for JAX backend.")

        missing_fixed = [sp for sp in self.fixed_species if sp not in fixed_conc]
        if missing_fixed:
            missing_str = ", ".join(sorted(missing_fixed))
            raise ValueError(f"Missing fixed concentrations for: {missing_str}")

        t = jnp.asarray(t_np, dtype=float)
        y0_j = jnp.asarray(y0_vec, dtype=float)
        S = jnp.asarray(self.S, dtype=float)
        fixed_conc = {str(k): float(v) for k, v in (fixed_conc or {}).items()}

        def resolve_params(temp_val):
            if not self.param_resolver.temp_models:
                return params
            resolved = dict(params)
            for k_name, model in self.param_resolver.temp_models.items():
                if model.kind == "arrhenius":
                    A_key = model.params.get("A")
                    Ea_key = model.params.get("Ea")
                    if A_key is None or Ea_key is None:
                        raise ValueError(
                            f"Arrhenius model for '{k_name}' requires A and Ea."
                        )
                    resolved[k_name] = resolved[A_key] * jnp.exp(
                        -resolved[Ea_key] / (float(temp_val) * 8.314462618)
                    )
                elif model.kind == "eyring":
                    dH_key = model.params.get("dH")
                    dS_key = model.params.get("dS")
                    if dH_key is None or dS_key is None:
                        raise ValueError(
                            f"Eyring model for '{k_name}' requires dH and dS."
                        )
                    temp = float(temp_val)
                    resolved[k_name] = (
                        1.380649e-23
                        * temp
                        / 6.62607015e-34
                        * jnp.exp(resolved[dS_key] / 8.314462618)
                        * jnp.exp(-resolved[dH_key] / (8.314462618 * temp))
                    )
                else:
                    raise ValueError(f"Unknown temperature model '{model.kind}'.")
            return resolved

        def rhs(t_val, y_val):
            k_values = resolve_params(temperature)
            rates = []
            for reaction in self.reactions:
                if reaction.k_name not in k_values:
                    raise KeyError(f"Missing kinetic parameter '{reaction.k_name}'.")
                rate = k_values[reaction.k_name]
                for species, coeff in reaction.reactants.items():
                    if species in self.species_index:
                        conc = y_val[self.species_index[species]]
                    else:
                        conc = fixed_conc.get(species)
                        if conc is None:
                            raise ValueError(
                                f"Missing fixed concentration for '{species}'."
                            )
                    rate = rate * conc ** coeff
                rates.append(rate)
            rates = jnp.stack(rates) if rates else jnp.zeros((0,), dtype=float)
            return S @ rates

        if t.shape[0] < 2:
            return jnp.asarray(y0_j).reshape(1, -1)

        def rk4_step(y_val, t_pair):
            t0, t1 = t_pair
            dt = t1 - t0
            k1 = rhs(t0, y_val)
            k2 = rhs(t0 + 0.5 * dt, y_val + 0.5 * dt * k1)
            k3 = rhs(t0 + 0.5 * dt, y_val + 0.5 * dt * k2)
            k4 = rhs(t1, y_val + dt * k3)
            y_next = y_val + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            return y_next, y_next

        ys, _ = lax.scan(rk4_step, y0_j, (t[:-1], t[1:]))
        return jnp.vstack([y0_j, ys])

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


def _validate_time_grid(t_grid: np.ndarray | Sequence[float]) -> np.ndarray:
    t = np.asarray(t_grid, dtype=float).reshape(-1)
    if t.size < 2:
        return t
    if not np.all(np.diff(t) > 0):
        raise ValueError("Time values must be strictly increasing.")
    return t
