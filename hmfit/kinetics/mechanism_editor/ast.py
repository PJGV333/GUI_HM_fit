"""AST types for the kinetics mechanism editor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Literal


@dataclass(frozen=True)
class TempModelAST:
    kind: Literal["arrhenius", "eyring"]
    params: Dict[str, str]


@dataclass(frozen=True)
class ReactionAST:
    reactants: Dict[str, int]
    products: Dict[str, int]
    k_forward: str
    k_reverse: Optional[str] = None


@dataclass(frozen=True)
class MechanismAST:
    species: List[str]
    fixed: Set[str]
    reactions: List[ReactionAST]
    temp_models: Dict[str, TempModelAST]
