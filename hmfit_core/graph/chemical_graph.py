# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*)")
_TERM_RE = re.compile(r"^\s*(?:(\d+(?:\.\d+)?)\s*)?([A-Za-z][A-Za-z0-9_]*)\s*$")
_EPS = 1e-12


def parse_formula_tokens(formula: str) -> Dict[str, float]:
    """Best-effort parser for simple formulas (e.g., H2O, ML2)."""
    clean = formula.strip()
    if not clean:
        raise ValueError("Species formula/name cannot be empty.")

    matches = list(_TOKEN_RE.finditer(clean))
    if not matches:
        return {clean: 1.0}

    covered = "".join(m.group(0) for m in matches)
    if covered != clean:
        # Fallback for non-formula names: treat full name as one pseudo-component.
        return {clean: 1.0}

    out: Dict[str, float] = {}
    for match in matches:
        token = match.group(1)
        count = float(match.group(2)) if match.group(2) else 1.0
        out[token] = out.get(token, 0.0) + count
    return out


def _merge_delta(
    delta: Dict[str, float], side: Mapping[SpeciesNode, float], sign: float
) -> None:
    for species, coeff in side.items():
        for token, count in parse_formula_tokens(species.name).items():
            delta[token] = delta.get(token, 0.0) + (sign * coeff * count)


@dataclass(slots=True, unsafe_hash=True)
class SpeciesNode:
    name: str
    charge: int = 0
    is_solid: bool = field(default=False, compare=False, hash=False)
    initial_concentration: float = field(default=0.0, compare=False, hash=False)

    def __post_init__(self) -> None:
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("SpeciesNode.name cannot be empty.")
        if self.initial_concentration < 0.0:
            raise ValueError("SpeciesNode.initial_concentration must be >= 0.")
        self.name = clean_name


@dataclass(slots=True)
class ReactionEdge:
    id: str
    reactants: Dict[SpeciesNode, float]
    products: Dict[SpeciesNode, float]
    log_beta: float

    def __post_init__(self) -> None:
        clean_id = self.id.strip()
        if not clean_id:
            raise ValueError("ReactionEdge.id cannot be empty.")
        self.id = clean_id
        self.reactants = self._validate_side(self.reactants, "reactants")
        self.products = self._validate_side(self.products, "products")
        self.log_beta = float(self.log_beta)

    @staticmethod
    def _validate_side(
        side: Mapping[SpeciesNode, float], side_name: str
    ) -> Dict[SpeciesNode, float]:
        out_by_name: Dict[str, Tuple[SpeciesNode, float]] = {}
        for species, coeff in side.items():
            value = float(coeff)
            if value <= 0.0:
                raise ValueError(
                    f"ReactionEdge.{side_name} coefficients must be > 0. Got {value} for {species.name!r}."
                )
            existing = out_by_name.get(species.name)
            if existing is None:
                out_by_name[species.name] = (species, value)
                continue
            prev_species, prev_coeff = existing
            if prev_species.charge != species.charge:
                raise ValueError(
                    f"ReactionEdge.{side_name} contains duplicated species {species.name!r} with different charge."
                )
            out_by_name[species.name] = (prev_species, prev_coeff + value)

        return {species: coeff for species, coeff in out_by_name.values()}

    @property
    def charge_balance(self) -> float:
        reactant_charge = sum(s.charge * c for s, c in self.reactants.items())
        product_charge = sum(s.charge * c for s, c in self.products.items())
        return product_charge - reactant_charge

    @property
    def is_charge_balanced(self) -> bool:
        return abs(self.charge_balance) <= _EPS

    @property
    def mass_balance_delta(self) -> Dict[str, float]:
        delta: Dict[str, float] = {}
        _merge_delta(delta, self.products, +1.0)
        _merge_delta(delta, self.reactants, -1.0)
        return {token: value for token, value in delta.items() if abs(value) > _EPS}

    @property
    def is_mass_balanced(self) -> bool:
        return not self.mass_balance_delta


class ChemicalGraph:
    def __init__(self) -> None:
        self._species_by_name: Dict[str, SpeciesNode] = {}
        self._reactions: List[ReactionEdge] = []

    @property
    def species(self) -> Tuple[SpeciesNode, ...]:
        return tuple(self._species_by_name.values())

    @property
    def reactions(self) -> Tuple[ReactionEdge, ...]:
        return tuple(self._reactions)

    def get_species(self, name: str) -> SpeciesNode | None:
        return self._species_by_name.get(name)

    def set_species_solid(self, name: str, is_solid: bool) -> None:
        species = self._species_by_name.get(name)
        if species is None:
            raise KeyError(f"Unknown species {name!r}.")
        species.is_solid = bool(is_solid)

    def add_species(self, node: SpeciesNode) -> None:
        existing = self._species_by_name.get(node.name)
        if existing is None:
            self._species_by_name[node.name] = node
            return

        if existing.charge != node.charge:
            raise ValueError(
                f"Species {node.name!r} already exists with different charge ({existing.charge} != {node.charge})."
            )
        if existing.is_solid != node.is_solid:
            raise ValueError(
                f"Species {node.name!r} already exists with different is_solid flag."
            )
        if abs(existing.initial_concentration - node.initial_concentration) > _EPS:
            raise ValueError(
                f"Species {node.name!r} already exists with different initial concentration."
            )

    def add_reaction(self, edge: ReactionEdge) -> None:
        if any(existing.id == edge.id for existing in self._reactions):
            raise ValueError(f"Reaction id {edge.id!r} already exists.")

        for species in self._iter_species(edge.reactants, edge.products):
            self.add_species(species)

        self._reactions.append(edge)

    def get_active_species(self) -> List[SpeciesNode]:
        return [species for species in self._species_by_name.values() if not species.is_solid]

    def to_stoichiometric_matrix(self) -> Tuple[List[SpeciesNode], np.ndarray]:
        species_list = self.get_active_species()
        n_reactions = len(self._reactions)
        n_species = len(species_list)
        matrix = np.zeros((n_reactions, n_species), dtype=float)

        index_by_name = {species.name: idx for idx, species in enumerate(species_list)}

        for row_idx, edge in enumerate(self._reactions):
            for species, coeff in edge.reactants.items():
                col_idx = index_by_name.get(species.name)
                if col_idx is not None:
                    matrix[row_idx, col_idx] -= coeff
            for species, coeff in edge.products.items():
                col_idx = index_by_name.get(species.name)
                if col_idx is not None:
                    matrix[row_idx, col_idx] += coeff

        return species_list, matrix

    def add_reaction_from_string(self, reaction_str: str, log_beta: float) -> ReactionEdge:
        if "<=>" not in reaction_str:
            raise ValueError(
                "Reaction must use '<=>' as separator, e.g. 'M + 2 L <=> ML2'."
            )
        left, right = reaction_str.split("<=>", maxsplit=1)
        reactants = self._parse_side(left)
        products = self._parse_side(right)
        edge = ReactionEdge(
            id=self._next_reaction_id(),
            reactants=reactants,
            products=products,
            log_beta=log_beta,
        )
        self.add_reaction(edge)
        return edge

    def validate_thermodynamic_cycles(self, tolerance: float = 1e-6) -> None:
        if tolerance <= 0.0:
            raise ValueError("tolerance must be > 0.")

        adjacency = self._build_adjacency()

        for start in sorted(adjacency.keys()):
            visited = {start}
            path_nodes: List[str] = [start]
            path_edges: List[str] = []

            def dfs(current: str, acc_weight: float) -> None:
                for nxt, weight, edge_tag in adjacency.get(current, []):
                    # Evitar el backtracking inmediato para reducir ciclos triviales.
                    if len(path_nodes) >= 2 and nxt == path_nodes[-2]:
                        continue

                    if nxt == start and len(path_nodes) >= 2:
                        loop_weight = acc_weight + weight
                        if abs(loop_weight) > tolerance:
                            cycle_path = " -> ".join(path_nodes + [start])
                            edge_path = ", ".join(path_edges + [edge_tag])
                            raise ValueError(
                                "Thermodynamic inconsistency detected in cycle "
                                f"{cycle_path} (delta_log_beta={loop_weight:.6g}; edges={edge_path})."
                            )
                        continue

                    if nxt in visited:
                        continue

                    visited.add(nxt)
                    path_nodes.append(nxt)
                    path_edges.append(edge_tag)
                    dfs(nxt, acc_weight + weight)
                    path_edges.pop()
                    path_nodes.pop()
                    visited.remove(nxt)

            dfs(start, 0.0)

    def _build_adjacency(self) -> Dict[str, List[Tuple[str, float, str]]]:
        adjacency: Dict[str, List[Tuple[str, float, str]]] = {
            name: [] for name in self._species_by_name.keys()
        }
        for edge in self._reactions:
            reactant_names = [species.name for species in edge.reactants.keys()]
            product_names = [species.name for species in edge.products.keys()]
            if not reactant_names or not product_names:
                continue

            for src in reactant_names:
                for dst in product_names:
                    adjacency.setdefault(src, []).append(
                        (dst, float(edge.log_beta), f"{edge.id}:fwd")
                    )
                    adjacency.setdefault(dst, []).append(
                        (src, -float(edge.log_beta), f"{edge.id}:rev")
                    )
        return adjacency

    def _next_reaction_id(self) -> str:
        next_idx = len(self._reactions) + 1
        while any(edge.id == f"R{next_idx}" for edge in self._reactions):
            next_idx += 1
        return f"R{next_idx}"

    def _parse_side(self, side: str) -> Dict[SpeciesNode, float]:
        out_by_name: Dict[str, float] = {}
        tokens = [chunk.strip() for chunk in side.split("+") if chunk.strip()]
        if not tokens:
            return {}

        for token in tokens:
            match = _TERM_RE.match(token)
            if match is None:
                raise ValueError(f"Invalid reaction token: {token!r}")
            coeff = float(match.group(1)) if match.group(1) else 1.0
            name = match.group(2)
            out_by_name[name] = out_by_name.get(name, 0.0) + coeff

        out: Dict[SpeciesNode, float] = {}
        for name, coeff in out_by_name.items():
            species = self._species_by_name.get(name)
            if species is None:
                species = SpeciesNode(name=name)
                self._species_by_name[name] = species
            out[species] = coeff
        return out

    @staticmethod
    def _iter_species(
        reactants: Mapping[SpeciesNode, float], products: Mapping[SpeciesNode, float]
    ) -> Iterable[SpeciesNode]:
        yield from reactants.keys()
        yield from products.keys()


def _infer_component_indices(matrix: np.ndarray) -> np.ndarray:
    consumed = np.any(matrix < -_EPS, axis=0)
    produced = np.any(matrix > _EPS, axis=0)
    components = np.flatnonzero(consumed & ~produced)
    if components.size > 0:
        return components
    # Fallback conservador si el sistema no es estrictamente de formacion.
    return np.flatnonzero(consumed)


def _infer_complex_indices(matrix: np.ndarray, component_idx: np.ndarray) -> np.ndarray:
    produced = np.flatnonzero(np.any(matrix > _EPS, axis=0))
    component_set = set(component_idx.tolist())
    return np.asarray(
        [idx for idx in produced.tolist() if idx not in component_set], dtype=int
    )


def _build_model_matrix(
    matrix: np.ndarray, component_idx: np.ndarray, complex_idx: np.ndarray
) -> np.ndarray:
    n_comp = int(component_idx.size)
    n_complex = int(complex_idx.size)
    model = np.zeros((n_comp, n_comp + n_complex), dtype=float)
    model[:, :n_comp] = np.eye(n_comp, dtype=float)

    if n_comp == 0 or n_complex == 0:
        return model

    for j, species_col in enumerate(complex_idx.tolist()):
        candidates: List[np.ndarray] = []
        for row in range(matrix.shape[0]):
            prod_coeff = matrix[row, species_col]
            if prod_coeff <= _EPS:
                continue
            coeffs = -matrix[row, component_idx] / prod_coeff
            coeffs = np.where(coeffs > 0.0, coeffs, 0.0)
            if np.any(coeffs > _EPS):
                candidates.append(coeffs)

        if candidates:
            model[:, n_comp + j] = np.mean(np.vstack(candidates), axis=0)

    return model


def _infer_complex_log_beta(
    reactions: Tuple[ReactionEdge, ...], matrix: np.ndarray, complex_idx: np.ndarray
) -> np.ndarray:
    out = np.zeros(complex_idx.size, dtype=float)
    for j, col in enumerate(complex_idx.tolist()):
        for row, edge in enumerate(reactions):
            coeff = matrix[row, col]
            if coeff > _EPS:
                out[j] = float(edge.log_beta) / coeff
                break
    return out


def create_solver_inputs_from_graph(graph: ChemicalGraph) -> Dict[str, Any]:
    species_list, matrix = graph.to_stoichiometric_matrix()
    total_concentrations = np.asarray(
        [species.initial_concentration for species in species_list], dtype=float
    )
    edge_log_beta = np.asarray([edge.log_beta for edge in graph.reactions], dtype=float)

    component_idx = _infer_component_indices(matrix)
    complex_idx = _infer_complex_indices(matrix, component_idx)

    component_species = [species_list[idx] for idx in component_idx.tolist()]
    complex_species = [species_list[idx] for idx in complex_idx.tolist()]

    model_matrix = _build_model_matrix(matrix, component_idx, complex_idx)
    complex_log_beta = _infer_complex_log_beta(graph.reactions, matrix, complex_idx)
    ctot = total_concentrations[component_idx][np.newaxis, :]

    return {
        "species": species_list,
        "stoichiometric_matrix": matrix,
        "total_concentrations": total_concentrations,
        "edge_log_beta": edge_log_beta,
        "components": {
            "species": component_species,
            "names": [species.name for species in component_species],
            "indices": component_idx,
            "total_concentrations": total_concentrations[component_idx],
        },
        "complexes": {
            "species": complex_species,
            "names": [species.name for species in complex_species],
            "indices": complex_idx,
            "log_beta": complex_log_beta,
        },
        "solver_inputs": {
            "ctot": ctot,
            "modelo": model_matrix,
            "nas": np.asarray([], dtype=int),
            "k": complex_log_beta,
            "model_sett": "Free",
        },
    }
