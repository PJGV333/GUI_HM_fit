# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np

from ..solvers.nr_conc import NewtonRaphson
from .chemical_graph import ChemicalGraph, create_solver_inputs_from_graph


def calculate_free_species(
    graph: ChemicalGraph, total_concentrations: Mapping[str, float]
) -> Dict[str, float]:
    payload = create_solver_inputs_from_graph(graph)

    components = payload.get("components", [])
    complexes = payload.get("complexes", [])
    solver_inputs = payload["solver_inputs"]

    if isinstance(components, Mapping):
        component_names = list(components.get("names") or [])
    else:
        component_names = [str(name) for name in (components or [])]

    if isinstance(complexes, Mapping):
        complex_names = list(complexes.get("names") or [])
    else:
        complex_names = [str(name) for name in (complexes or [])]

    missing = [name for name in component_names if name not in total_concentrations]
    if missing:
        raise KeyError(
            f"Missing total concentrations for components: {', '.join(sorted(missing))}"
        )

    ctot = np.asarray(
        [[float(total_concentrations[name]) for name in component_names]], dtype=float
    )
    model = np.asarray(solver_inputs["modelo"], dtype=float)
    nas = np.asarray(solver_inputs["nas"], dtype=int)
    if isinstance(complexes, Mapping):
        k_source = complexes.get("log_beta")
    else:
        k_source = payload.get("complex_log_beta")
    if k_source is None:
        k_source = solver_inputs.get("k")
    k_log = np.asarray(k_source, dtype=float)

    solver = NewtonRaphson(ctot=ctot, modelo=model, nas=nas, model_sett="Free")
    _, c_calculada = solver.concentraciones(k_log)

    ordered_names = component_names + complex_names
    if c_calculada.shape[1] != len(ordered_names):
        raise ValueError(
            "Solver output size does not match graph species. "
            f"Expected {len(ordered_names)}, got {c_calculada.shape[1]}."
        )

    return {
        species_name: float(c_calculada[0, idx])
        for idx, species_name in enumerate(ordered_names)
    }
