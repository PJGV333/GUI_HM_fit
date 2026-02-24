"""Adapters from kinetic graph structures to native kinetics models."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .kinetics_graph import KineticGraph, KineticReactionEdge
from .kinetics_model import KineticsModel
from .reactions import Reaction
from ..mechanism_editor.ast import MechanismAST, ReactionAST

_INT_TOL = 1e-9


def _as_int_stoich(stoich: Dict[str, float]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for species, coeff in stoich.items():
        rounded = int(round(coeff))
        if abs(coeff - rounded) > _INT_TOL:
            raise ValueError(
                f"Kinetics stoichiometry for {species!r} must be integer-like; got {coeff}."
            )
        out[species] = rounded
    return out


def _edge_to_native_reactions(
    edge: KineticReactionEdge,
) -> Tuple[ReactionAST, List[Reaction], Dict[str, float]]:
    reactants = _as_int_stoich({s.name: c for s, c in edge.reactants.items()})
    products = _as_int_stoich({s.name: c for s, c in edge.products.items()})

    if edge.k_forward is None:
        raise ValueError(f"Edge {edge.id!r} is missing k_forward.")

    kf_name = f"{edge.id}_kf"
    kr_name = f"{edge.id}_kb" if edge.k_backward is not None else None

    reaction_ast = ReactionAST(
        reactants=reactants,
        products=products,
        k_forward=kf_name,
        k_reverse=kr_name,
    )

    native_reactions: List[Reaction] = [Reaction(reactants, products, kf_name)]
    params: Dict[str, float] = {kf_name: float(edge.k_forward)}

    if edge.k_backward is not None:
        native_reactions.append(Reaction(products, reactants, kr_name))  # type: ignore[arg-type]
        params[kr_name] = float(edge.k_backward)  # type: ignore[index]

    return reaction_ast, native_reactions, params


def build_kinetics_model_from_graph(graph: KineticGraph) -> KineticsModel:
    active_species = [node.name for node in graph.get_active_species()]

    reaction_asts: List[ReactionAST] = []
    native_reactions: List[Reaction] = []
    param_defaults: Dict[str, float] = {}

    for edge in graph.reactions:
        if not isinstance(edge, KineticReactionEdge):
            raise TypeError(
                f"Expected KineticReactionEdge, got {type(edge).__name__}."
            )
        reaction_ast, native_block, params = _edge_to_native_reactions(edge)
        reaction_asts.append(reaction_ast)
        native_reactions.extend(native_block)
        param_defaults.update(params)

    mechanism = MechanismAST(
        species=active_species,
        fixed=set(),
        reactions=reaction_asts,
        temp_models={},
    )

    model = KineticsModel(mechanism)
    # Bridge helper: convenient parameter defaults generated from graph rates.
    model.default_params = param_defaults  # type: ignore[attr-defined]
    # Bridge helper: explicit native reaction list produced during adaptation.
    model.native_reactions_from_graph = native_reactions  # type: ignore[attr-defined]
    return model
