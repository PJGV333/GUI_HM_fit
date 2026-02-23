# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import Dict, Mapping

from .chemical_graph import ChemicalGraph, parse_formula_tokens


def _ionic_product_from_components(
    species_name: str, current_concentrations: Mapping[str, float]
) -> float:
    ionic_product = 1.0
    for token, power in parse_formula_tokens(species_name).items():
        concentration = float(current_concentrations.get(token, 0.0))
        if concentration < 0.0:
            raise ValueError(
                f"Concentration for component {token!r} must be >= 0. Got {concentration}."
            )
        ionic_product *= concentration**power
    return ionic_product


def update_phase_states(
    graph: ChemicalGraph,
    current_concentrations: Mapping[str, float],
    solubility_products: Mapping[str, float],
) -> Dict[str, bool]:
    """
    Actualiza el estado de fase de especies según Kps:
      - True  -> precipitada (is_solid=True)
      - False -> disuelta    (is_solid=False)

    Retorna un diccionario {species_name: is_solid} solo de las especies evaluadas.
    """

    updated: Dict[str, bool] = {}
    for species_name, kps_raw in solubility_products.items():
        species = graph.get_species(species_name)
        if species is None:
            raise KeyError(f"Species {species_name!r} not found in graph.")

        kps = float(kps_raw)
        if kps <= 0.0:
            raise ValueError(f"Kps for {species_name!r} must be > 0. Got {kps}.")

        ionic_product = _ionic_product_from_components(
            species_name=species_name,
            current_concentrations=current_concentrations,
        )
        is_solid = ionic_product > kps
        graph.set_species_solid(species_name, is_solid=is_solid)
        updated[species_name] = is_solid

    return updated
