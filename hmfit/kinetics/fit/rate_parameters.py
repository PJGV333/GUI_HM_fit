"""Kinetics rate initialization, dimensional validation and unit metadata."""
from __future__ import annotations

import numpy as np

from hmfit.kinetics.model.kinetics_model import KineticsModel


def _shared_rate_order_error(model: KineticsModel) -> str | None:
    """Reject shared rate constants whose reactions require incompatible units."""
    groups: dict[str, list] = {}
    for reaction in model.reactions:
        groups.setdefault(reaction.k_name, []).append(reaction)
    for name, reactions in groups.items():
        orders = {sum(reaction.reactants.values()) for reaction in reactions}
        if len(orders) < 2:
            continue

        def side(species):
            return " + ".join(f"{n} {s}" if n != 1 else s for s, n in species.items()) or "∅"

        descriptions = []
        for reaction in reactions:
            order = sum(reaction.reactants.values())
            units = "s⁻¹" if order == 1 else f"M^{1 - order} s⁻¹"
            descriptions.append(f"{side(reaction.reactants)} -> {side(reaction.products)} (order {order}, {units})")
        return f"Rate constant {name} is shared by reactions of different orders: " + "; ".join(descriptions)
    return None



def _rate_parameter_units(model, datasets):
    from hmfit.kinetics.data.units import canonical_time_unit, canonical_concentration_unit
    times = {canonical_time_unit(getattr(ds, "time_unit", "")) for ds in datasets}
    concs = {canonical_concentration_unit(getattr(ds, "concentration_unit", "")) for ds in datasets}
    t = next(iter(times)) if len(times) == 1 and None not in times else "time unit unspecified"
    c = next(iter(concs)) if len(concs) == 1 and None not in concs else "concentration unit unspecified"
    units = {}
    for r in model.reactions:
        order = sum(r.reactants.values())
        prefix = "" if order == 1 else (f"{c}⁻¹·" if order == 2 else f"{c}^({1-order})·")
        unit = f"{prefix}{t}⁻¹"
        temp = model.mechanism.temp_models.get(r.k_name)
        if temp is None:
            units[r.k_name] = unit
        elif temp.kind == "arrhenius":
            units[temp.params["A"]] = unit
            units[temp.params["Ea"]] = "J·mol⁻¹"
        else:
            units[temp.params["dH"]] = "J·mol⁻¹"
            units[temp.params["dS"]] = "J·mol⁻¹·K⁻¹"
    return units


def _eyring_dataset_warnings(model, datasets):
    from hmfit.kinetics.data.units import eyring_unit_factors
    warnings = []
    for reaction in model.reactions:
        temp = model.mechanism.temp_models.get(reaction.k_name)
        if temp and temp.kind == "eyring":
            for dataset in datasets:
                _, messages = eyring_unit_factors(dataset, sum(reaction.reactants.values()))
                warnings.extend(messages)
    return list(dict.fromkeys(warnings))



def _automatic_rate_parameters(model, datasets, user_values=None):
    """Scale rates to the experiment, in its numeric units (no dataset conversion).

    t_total is the largest last-minus-first time. For each distinct rate in first
    declaration order, k0 = 1/(t_total*c_typical**(n-1))/3**index, where n is
    reaction order and c_typical is the largest positive starting concentration of
    any reactant in any reaction using that rate. For intermediates use the median
    of all positive initial concentrations. Shared rates use the fastest channel,
    independently of reaction declaration order. Thermodynamic parameters are
    obtained by inverting the actual rate laws at the median dataset temperature.
    """
    from hmfit.kinetics.model.temperature import arrhenius, eyring, R_GAS
    from hmfit.kinetics.data.units import eyring_unit_factors
    user_values = user_values or {}
    spans = [float(ds.t[-1] - ds.t[0]) for ds in datasets if len(ds.t) >= 2]
    if not spans or not np.isfinite(max(spans)) or max(spans) <= 0:
        return {}
    duration = max(spans)
    concentrations = []
    per_dataset = []
    for ds in datasets:
        initial = dict(ds.y0) if isinstance(ds.y0, dict) else dict(zip(model.dynamic_species, ds.y0 if ds.y0 is not None else []))
        initial.update(dict(ds.fixed_conc or {}))
        per_dataset.append(initial)
        concentrations.extend(float(v) for v in initial.values() if np.isfinite(v) and v > 0)
    groups = {}
    for reaction in model.reactions:
        groups.setdefault(reaction.k_name, []).append(reaction)
    temperatures = [float(ds.temperature) for ds in datasets if not callable(ds.temperature) and np.isfinite(ds.temperature) and ds.temperature > 0]
    tref = float(np.median(temperatures)) if temperatures else None
    proposals = {}
    for idx, (name, reactions) in enumerate(groups.items()):
        order = sum(reactions[0].reactants.values())
        candidates = [float(initial.get(s, 0)) for initial in per_dataset for reaction in reactions
                      for s in reaction.reactants if np.isfinite(initial.get(s, 0)) and initial.get(s, 0) > 0]
        typical = max(candidates) if candidates else (float(np.median(concentrations)) if concentrations else None)
        if order > 1 and typical is None:
            continue
        k0 = 1.0 / (duration * (typical ** (order - 1) if order > 1 else 1.0) * 3.0 ** idx)
        temp = model.mechanism.temp_models.get(name)
        if temp is None:
            proposals[name] = k0
        elif tref is not None and temp.kind == "arrhenius":
            A, Ea = temp.params["A"], temp.params["Ea"]
            ea = user_values.get(Ea, 0.0)
            proposals[Ea] = 0.0
            proposals[A] = k0 / arrhenius(1.0, ea, tref)
        elif tref is not None and temp.kind == "eyring":
            H, S = temp.params["dH"], temp.params["dS"]
            h = user_values.get(H, 0.0)
            try:
                factor, _ = eyring_unit_factors(datasets[0], order)
                proposals[H] = 0.0
                proposals[S] = R_GAS * np.log(k0 / (factor * eyring(h, 0., tref)))
            except ValueError:
                continue
    return proposals

