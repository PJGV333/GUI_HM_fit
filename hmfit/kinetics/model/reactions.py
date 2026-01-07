"""Reaction definitions and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..mechanism_editor.ast import MechanismAST, ReactionAST


@dataclass(frozen=True)
class Reaction:
    reactants: dict[str, int]
    products: dict[str, int]
    k_name: str


def expand_reactions(mechanism: MechanismAST) -> List[Reaction]:
    """Expand reversible reactions into separate forward/reverse entries."""
    expanded: List[Reaction] = []
    for reaction in mechanism.reactions:
        expanded.extend(_expand_reaction(reaction))
    return expanded


def _expand_reaction(reaction: ReactionAST) -> List[Reaction]:
    expanded = [Reaction(reaction.reactants, reaction.products, reaction.k_forward)]
    if reaction.k_reverse:
        expanded.append(Reaction(reaction.products, reaction.reactants, reaction.k_reverse))
    return expanded
