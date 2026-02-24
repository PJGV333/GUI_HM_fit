"""Kinetic graph structures built on top of the core chemical graph."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

from hmfit_core.graph.chemical_graph import ChemicalGraph, ReactionEdge

_LOG_TOL = 1e-9


@dataclass(slots=True)
class KineticReactionEdge(ReactionEdge):
    k_forward: Optional[float] = None
    k_backward: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()

        if self.k_forward is not None:
            self.k_forward = float(self.k_forward)
            if self.k_forward <= 0.0:
                raise ValueError("k_forward must be > 0.")

        if self.k_backward is not None:
            self.k_backward = float(self.k_backward)
            if self.k_backward <= 0.0:
                raise ValueError("k_backward must be > 0.")

        if self.is_reversible:
            expected_log_beta = math.log10(self.k_forward / self.k_backward)  # type: ignore[operator]
            if abs(self.log_beta - expected_log_beta) > _LOG_TOL:
                raise ValueError(
                    "Inconsistent reversible edge: log_beta must equal "
                    "log10(k_forward / k_backward)."
                )

    @property
    def is_reversible(self) -> bool:
        return self.k_forward is not None and self.k_backward is not None


class KineticGraph(ChemicalGraph):
    """ChemicalGraph extension with kinetic rate constants."""

    def add_kinetic_reaction(
        self,
        reaction_str: str,
        k_forward: float,
        k_backward: Optional[float] = None,
    ) -> KineticReactionEdge:
        text = reaction_str.strip()
        if "<=>" in text:
            left, right = text.split("<=>", maxsplit=1)
            reversible_arrow = True
        elif "<->" in text:
            left, right = text.split("<->", maxsplit=1)
            reversible_arrow = True
        elif "->" in text:
            left, right = text.split("->", maxsplit=1)
            reversible_arrow = False
        else:
            raise ValueError(
                "Reaction must contain one of: '->', '<->', '<=>'."
            )

        if reversible_arrow and k_backward is None:
            raise ValueError(
                "Reversible reactions require k_backward."
            )

        reactants = self._parse_side(left)
        products = self._parse_side(right)

        if k_forward <= 0.0:
            raise ValueError("k_forward must be > 0.")

        if k_backward is not None:
            if k_backward <= 0.0:
                raise ValueError("k_backward must be > 0.")
            log_beta = math.log10(float(k_forward) / float(k_backward))
        else:
            log_beta = 0.0

        edge = KineticReactionEdge(
            id=self._next_reaction_id(),
            reactants=reactants,
            products=products,
            log_beta=log_beta,
            k_forward=float(k_forward),
            k_backward=float(k_backward) if k_backward is not None else None,
        )
        self.add_reaction(edge)
        return edge
