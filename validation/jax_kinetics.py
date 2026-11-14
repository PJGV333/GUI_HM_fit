"""Historical JAX kinetics experiments; not used by production fits."""

from hmfit.kinetics.model.kinetics_model import (
    _coerce_y0, _extract_fixed_conc, _extract_temperature, _validate_time_grid,
)


def jax_solve_linear_matrix(C, D, weights=None):
    """Project with the same relative rank cutoff as the SciPy observation model."""
    import jax.numpy as jnp
    from hmfit.kinetics.observation.linear_matrix import RANK_RCOND

    if weights is None:
        return jnp.linalg.pinv(C, rtol=RANK_RCOND) @ D
    if weights.ndim == 1 or weights.shape[1] == 1:
        w = weights.reshape(-1, 1)
        return jnp.linalg.pinv(C * w, rtol=RANK_RCOND) @ (D * w)
    return jnp.concatenate([
        jnp.linalg.pinv(C * weights[:, j:j + 1], rtol=RANK_RCOND)
        @ (D[:, j:j + 1] * weights[:, j:j + 1])
        for j in range(D.shape[1])
    ], axis=1)


def solve_concentrations_jax(self, t_grid, y0, params, context=None):
    """Historical fixed-step RK4 integration; ``self`` is a KineticsModel."""
    try:
        import jax
        jax.config.update("jax_enable_x64", True)
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
    eyring_factors = self._eyring_unit_factors(context)

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
                    * eyring_factors[k_name]
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

    _, ys = lax.scan(rk4_step, y0_j, (t[:-1], t[1:]))
    return jnp.vstack([y0_j, ys])
