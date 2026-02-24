# SPDX-License-Identifier: GPL-3.0-or-later

from .nodes import Node, ComponentNode, SpeciesNode
from .edges import StoichiometricEdge
from .hmgraph import HMGraph
from .chemical_graph import (
    ChemicalGraph,
    ReactionEdge,
    SpeciesNode as ChemicalSpeciesNode,
    parse_formula_tokens,
    create_solver_inputs_from_graph,
)
from .phase_manager import update_phase_states


def calculate_free_species(*args, **kwargs):
    # Lazy import to avoid circular dependency with hmfit_core.solvers.
    from .graph_solver_bridge import calculate_free_species as _calculate_free_species

    return _calculate_free_species(*args, **kwargs)

__all__ = [
    "Node",
    "ComponentNode",
    "SpeciesNode",
    "StoichiometricEdge",
    "HMGraph",
    "ChemicalSpeciesNode",
    "ReactionEdge",
    "ChemicalGraph",
    "parse_formula_tokens",
    "create_solver_inputs_from_graph",
    "calculate_free_species",
    "update_phase_states",
]
