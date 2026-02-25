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

    @staticmethod
    def _reaction_side_sort_key(side: Mapping[SpeciesNode, float]) -> str:
        parts: List[str] = []
        for species, coeff in sorted(side.items(), key=lambda item: item[0].name):
            value = float(coeff)
            if abs(value - 1.0) <= _EPS:
                parts.append(species.name)
            elif float(value).is_integer():
                parts.append(f"{int(value)} {species.name}")
            else:
                parts.append(f"{value:g} {species.name}")
        return " + ".join(parts)

    @classmethod
    def _reaction_sort_key(cls, edge: ReactionEdge) -> tuple[str, str]:
        reaction_text = (
            f"{cls._reaction_side_sort_key(edge.reactants)} <=> "
            f"{cls._reaction_side_sort_key(edge.products)}"
        )
        return reaction_text, edge.id

    def get_sorted_elements(
        self,
    ) -> Tuple[List[SpeciesNode], List[SpeciesNode], List[ReactionEdge]]:
        active_species = self.get_active_species()
        produced_names = {
            species.name
            for edge in self._reactions
            for species in edge.products.keys()
            if not species.is_solid
        }

        components = sorted(
            [species for species in active_species if species.name not in produced_names],
            key=lambda species: species.name,
        )
        complexes = sorted(
            [species for species in active_species if species.name in produced_names],
            key=lambda species: species.name,
        )
        sorted_reactions = sorted(self._reactions, key=self._reaction_sort_key)
        return components, complexes, sorted_reactions

    def resolve_global_pathways(self) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Flatten stepwise equilibria into global equilibria relative to base components.
        """
        active_names = {species.name for species in self.get_active_species()}

        forming_reactions: Dict[str, ReactionEdge] = {}
        for reaction in self._reactions:
            for product in reaction.products.keys():
                product_name = product.name
                if product_name not in active_names:
                    continue
                prev = forming_reactions.get(product_name)
                if prev is not None and prev is not reaction:
                    raise ValueError(f"Múltiples rutas de formación para {product_name!r}.")
                forming_reactions[product_name] = reaction

        global_stoich: Dict[str, Dict[str, float]] = {}
        global_log_beta: Dict[str, float] = {}
        path: set[str] = set()

        def get_global(species_name: str) -> Tuple[Dict[str, float], float]:
            if species_name in global_stoich:
                return global_stoich[species_name], global_log_beta[species_name]

            if species_name in path:
                raise ValueError(f"Ciclo detectado en rutas de formación para {species_name!r}.")

            path.add(species_name)
            try:
                reaction = forming_reactions.get(species_name)
                if reaction is None:
                    stoich = {species_name: 1.0}
                    beta = 0.0
                else:
                    total_stoich: Dict[str, float] = {}
                    total_beta = float(reaction.log_beta)
                    for reactant, coeff in reaction.reactants.items():
                        reactant_stoich, reactant_beta = get_global(reactant.name)
                        coeff_f = float(coeff)
                        total_beta += reactant_beta * coeff_f
                        for comp_name, comp_coeff in reactant_stoich.items():
                            total_stoich[comp_name] = (
                                total_stoich.get(comp_name, 0.0) + (float(comp_coeff) * coeff_f)
                            )

                    product_coeff = None
                    for product, coeff in reaction.products.items():
                        if product.name == species_name:
                            product_coeff = float(coeff)
                            break
                    if product_coeff is None or abs(product_coeff) <= _EPS:
                        raise ValueError(
                            f"No product coefficient found for species {species_name!r}."
                        )

                    inv_coeff = 1.0 / product_coeff
                    total_beta *= inv_coeff
                    stoich = {
                        comp_name: float(comp_coeff) * inv_coeff
                        for comp_name, comp_coeff in total_stoich.items()
                    }
                    beta = float(total_beta)

                global_stoich[species_name] = stoich
                global_log_beta[species_name] = float(beta)
                return stoich, float(beta)
            finally:
                path.discard(species_name)

        for species in self.get_active_species():
            get_global(species.name)

        return global_stoich, global_log_beta

    def to_stoichiometric_matrix(self) -> Tuple[List[SpeciesNode], np.ndarray]:
        components, complexes, sorted_reactions = self.get_sorted_elements()
        species_list = components + complexes
        n_reactions = len(sorted_reactions)
        n_species = len(species_list)
        matrix = np.zeros((n_reactions, n_species), dtype=float)

        index_by_name = {species.name: idx for idx, species in enumerate(species_list)}

        for row_idx, edge in enumerate(sorted_reactions):
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
        if not self._reactions or len(self._species_by_name) <= 1:
            return

        species_names = list(self._species_by_name.keys())
        reference = species_names[0]
        unknown_names = [name for name in species_names if name != reference]
        if not unknown_names:
            return

        unknown_index = {name: idx for idx, name in enumerate(unknown_names)}
        rows: List[np.ndarray] = []
        rhs: List[float] = []
        edge_ids: List[str] = []

        for edge in self._reactions:
            row = np.zeros(len(unknown_names), dtype=float)
            for species, coeff in edge.products.items():
                idx = unknown_index.get(species.name)
                if idx is not None:
                    row[idx] += float(coeff)
            for species, coeff in edge.reactants.items():
                idx = unknown_index.get(species.name)
                if idx is not None:
                    row[idx] -= float(coeff)

            if np.any(np.abs(row) > _EPS):
                rows.append(row)
                rhs.append(float(edge.log_beta))
                edge_ids.append(edge.id)

        if not rows:
            return

        a_mat = np.vstack(rows)
        b_vec = np.asarray(rhs, dtype=float)
        solution, _, _, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
        residual = a_mat @ solution - b_vec
        max_residual = float(np.max(np.abs(residual)))
        if max_residual > tolerance:
            worst_idx = int(np.argmax(np.abs(residual)))
            raise ValueError(
                "Thermodynamic inconsistency detected in cycle constraints "
                f"(edge={edge_ids[worst_idx]!r}, residual={residual[worst_idx]:.6g}, "
                f"tolerance={tolerance:.6g})."
            )

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


def _formula_based_component_vector(
    species_name: str, component_names: List[str]
) -> np.ndarray | None:
    token_counts = parse_formula_tokens(species_name)
    if any(token not in component_names for token in token_counts.keys()):
        return None

    vec = np.asarray([token_counts.get(name, 0.0) for name in component_names], dtype=float)
    if not np.any(vec > _EPS):
        return None
    return vec


def _build_model_matrix(
    species_list: List[SpeciesNode],
    matrix: np.ndarray,
    component_idx: np.ndarray,
    complex_idx: np.ndarray,
) -> np.ndarray:
    n_comp = int(component_idx.size)
    n_complex = int(complex_idx.size)
    model = np.zeros((n_comp, n_comp + n_complex), dtype=float)
    model[:, :n_comp] = np.eye(n_comp, dtype=float)

    if n_comp == 0 or n_complex == 0:
        return model

    component_names = [species_list[idx].name for idx in component_idx.tolist()]
    for j, species_col in enumerate(complex_idx.tolist()):
        species_name = species_list[species_col].name
        formula_vec = _formula_based_component_vector(species_name, component_names)
        if formula_vec is not None:
            model[:, n_comp + j] = formula_vec
            continue

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


def _infer_complex_log_beta_first_hit(
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


def _infer_complex_log_beta(
    reactions: Tuple[ReactionEdge, ...],
    matrix: np.ndarray,
    component_names: List[str],
    complex_names: List[str],
    complex_idx: np.ndarray,
) -> np.ndarray:
    n_complex = len(complex_names)
    if n_complex == 0:
        return np.asarray([], dtype=float)

    complex_index = {name: idx for idx, name in enumerate(complex_names)}
    active_names = set(component_names) | set(complex_names)

    rows: List[np.ndarray] = []
    rhs: List[float] = []
    for edge in reactions:
        row = np.zeros(n_complex, dtype=float)
        has_unknown = False

        for species, coeff in edge.products.items():
            species_name = species.name
            if species_name not in active_names:
                continue
            idx = complex_index.get(species_name)
            if idx is not None:
                row[idx] += float(coeff)
                has_unknown = True

        for species, coeff in edge.reactants.items():
            species_name = species.name
            if species_name not in active_names:
                continue
            idx = complex_index.get(species_name)
            if idx is not None:
                row[idx] -= float(coeff)
                has_unknown = True

        if has_unknown:
            rows.append(row)
            rhs.append(float(edge.log_beta))

    if not rows:
        return np.zeros(n_complex, dtype=float)

    a_mat = np.vstack(rows)
    b_vec = np.asarray(rhs, dtype=float)

    solution, _, rank, _ = np.linalg.lstsq(a_mat, b_vec, rcond=None)
    if rank < n_complex:
        return _infer_complex_log_beta_first_hit(
            reactions,
            matrix,
            complex_idx,
        )

    residual = np.max(np.abs(a_mat @ solution - b_vec))
    if residual > 1e-6:
        return _infer_complex_log_beta_first_hit(
            reactions,
            matrix,
            complex_idx,
        )

    return np.asarray(solution, dtype=float)


def create_solver_inputs_from_graph(graph: ChemicalGraph) -> Dict[str, Any]:
    component_species, complex_species, sorted_reactions = graph.get_sorted_elements()
    species_list, matrix = graph.to_stoichiometric_matrix()
    total_concentrations = np.asarray(
        [species.initial_concentration for species in species_list], dtype=float
    )
    edge_log_beta = np.asarray([edge.log_beta for edge in sorted_reactions], dtype=float)

    global_stoich, global_log_beta = graph.resolve_global_pathways()
    component_names = [species.name for species in component_species]
    complex_names = [species.name for species in complex_species]
    n_components = len(component_species)
    component_idx = np.arange(n_components, dtype=int)
    complex_idx = np.arange(n_components, len(species_list), dtype=int)
    n_complex = len(complex_species)

    model_matrix = np.zeros((n_components, n_components + n_complex), dtype=float)
    if n_components > 0:
        model_matrix[:, :n_components] = np.eye(n_components, dtype=float)

    for j, complex_name in enumerate(complex_names):
        stoich = global_stoich.get(complex_name, {})
        for i, component_name in enumerate(component_names):
            model_matrix[i, n_components + j] = float(stoich.get(component_name, 0.0))

    complex_log_beta = np.asarray(
        [float(global_log_beta.get(complex_name, 0.0)) for complex_name in complex_names],
        dtype=float,
    )
    ctot = np.asarray(
        [[float(species.initial_concentration) for species in component_species]], dtype=float
    )

    return {
        "species": species_list,
        "reactions": sorted_reactions,
        "stoichiometric_matrix": matrix,
        "total_concentrations": total_concentrations,
        "edge_log_beta": edge_log_beta,
        "components": component_names,
        "complexes": complex_names,
        "components_meta": {
            "species": component_species,
            "names": component_names,
            "indices": component_idx,
            "total_concentrations": total_concentrations[component_idx],
        },
        "complexes_meta": {
            "species": complex_species,
            "names": complex_names,
            "indices": complex_idx,
            "log_beta": complex_log_beta,
        },
        "complex_log_beta": complex_log_beta,
        "solver_inputs": {
            "ctot": ctot,
            "modelo": model_matrix,
            "nas": np.asarray([], dtype=int),
            "k": complex_log_beta,
            "model_sett": "Free",
        },
    }
