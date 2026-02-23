# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np

from .edges import StoichiometricEdge
from .nodes import ComponentNode, SpeciesNode


@dataclass
class Compilation:
    """
    Resultado de compilar el grafo a matrices para el solver.
    - modelo: (n_comp, nspec) con columnas [componentes libres] + [especies complejas]
    - nas: indices de especies NO observables (para excluir en señal)
    - component_keys: orden final de componentes (filas)
    - species_keys: orden final de especies (columnas)
    """

    modelo: np.ndarray
    nas: np.ndarray
    component_keys: List[str]
    species_keys: List[str]


class HMGraph:
    """
    Grafo minimo y determinista (sin dependencia externa).
    - Componentes: nodos base (filas)
    - Especies: nodos complejos (columnas adicionales)
    - Aristas estequiometricas: componente -> especie con coeficiente entero

    NOTA:
      Las especies libres de los componentes se generan automaticamente al compilar,
      para respetar el contrato actual de los solvers (primer bloque identidad).
    """

    def __init__(self):
        self.components: Dict[str, ComponentNode] = {}
        self.species: Dict[str, SpeciesNode] = {}  # solo especies complejas
        self.edges: Dict[Tuple[str, str], StoichiometricEdge] = {}

        self._revision: int = 0
        self._last_compilation: Optional[Compilation] = None
        self._last_solution: Optional[np.ndarray] = None

    # ---------- utilidades ----------
    @property
    def revision(self) -> int:
        return self._revision

    def _touch(self) -> None:
        self._revision += 1

    # ---------- creacion de nodos ----------
    def add_component(
        self, key: str, name: Optional[str] = None, *, observable: bool = True, order: float = 0.0
    ) -> ComponentNode:
        if key in self.components:
            raise ValueError(f"Component '{key}' already exists.")
        node = ComponentNode(key=key, name=name or key, observable=observable, order=order)
        self.components[key] = node
        self._touch()
        return node

    def add_species(
        self,
        key: str,
        name: Optional[str] = None,
        *,
        stoich: Optional[Dict[str, int]] = None,
        observable: bool = True,
        order: float = 0.0,
    ) -> SpeciesNode:
        if key in self.species:
            raise ValueError(f"Species '{key}' already exists.")
        node = SpeciesNode(key=key, name=name or key, stoich=stoich or {}, observable=observable, order=order)
        self.species[key] = node
        self._touch()
        return node

    # ---------- aristas ----------
    def connect(self, component_key: str, species_key: str, coefficient: int) -> None:
        if coefficient < 0:
            raise ValueError("Stoichiometric coefficient must be >= 0.")
        if component_key not in self.components:
            raise KeyError(f"Unknown component '{component_key}'.")
        if species_key not in self.species:
            raise KeyError(f"Unknown species '{species_key}'.")

        edge = StoichiometricEdge(
            component_key=component_key, species_key=species_key, coefficient=int(coefficient)
        )
        self.edges[(component_key, species_key)] = edge
        # Mantener tambien la estequiometria en el nodo de especie (fuente unica: edges)
        self.species[species_key].stoich[component_key] = int(coefficient)
        self._touch()

    # ---------- activar/desactivar ----------
    def set_active(self, key: str, active: bool) -> None:
        if key in self.components:
            self.components[key].active = bool(active)
        elif key in self.species:
            self.species[key].active = bool(active)
        else:
            raise KeyError(f"Unknown node '{key}'.")
        self._touch()

    def set_precipitated(self, species_key: str, precipitated: bool) -> None:
        if species_key not in self.species:
            raise KeyError(f"Unknown species '{species_key}'.")
        self.species[species_key].precipitated = bool(precipitated)
        self._touch()

    def set_observable(self, key: str, observable: bool) -> None:
        if key in self.components:
            self.components[key].observable = bool(observable)
        elif key in self.species:
            self.species[key].observable = bool(observable)
        else:
            raise KeyError(f"Unknown node '{key}'.")
        self._touch()

    # ---------- validacion basica ----------
    def validate(self) -> None:
        # Coeficientes enteros y no negativos; componentes presentes
        for sp in self.species.values():
            for ck, coef in sp.stoich.items():
                if ck not in self.components:
                    raise ValueError(f"Species '{sp.key}' references unknown component '{ck}'.")
                if int(coef) != coef or coef < 0:
                    raise ValueError(f"Invalid stoichiometry: {sp.key}[{ck}] = {coef}")

    # ---------- compilacion (puente a algebra matricial) ----------
    def compile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compila el grafo a (modelo, nas) siguiendo el contrato de los solvers:
          - modelo: (n_comp, nspec) con columnas: [componentes libres] + [complejos]
          - nas: indices de especies no observables dentro del orden compilado
        """
        self.validate()

        # Orden determinista: primero por `order`; en empates se conserva
        # el orden de insercion del diccionario (sort estable de Python).
        comp_keys = sorted(
            [k for k, n in self.components.items() if n.active],
            key=lambda k: self.components[k].order,
        )
        # Excluir especies inactivas o precipitados (por diseno en esta primera version)
        complex_keys = sorted(
            [k for k, n in self.species.items() if n.active and (not n.precipitated)],
            key=lambda k: self.species[k].order,
        )

        n_comp = len(comp_keys)
        species_keys = comp_keys + complex_keys
        nspec = len(species_keys)

        mtx = np.zeros((n_comp, nspec), dtype=float)

        # Bloque identidad para componentes libres
        for i, ck in enumerate(comp_keys):
            mtx[i, i] = 1.0

        # Complejos
        comp_index = {ck: i for i, ck in enumerate(comp_keys)}
        for j, sk in enumerate(complex_keys, start=n_comp):
            sto = self.species[sk].stoich
            for ck, coef in sto.items():
                if ck in comp_index:
                    mtx[comp_index[ck], j] = float(coef)

        # nas: especies NO observables (incluye componentes si asi lo marcas)
        nas_list: List[int] = []
        for j, key in enumerate(species_keys):
            if key in self.components:
                if not self.components[key].observable:
                    nas_list.append(j)
            else:
                if not self.species[key].observable:
                    nas_list.append(j)

        comp = Compilation(
            modelo=mtx,
            nas=np.asarray(nas_list, dtype=int),
            component_keys=comp_keys,
            species_keys=species_keys,
        )
        self._last_compilation = comp
        return comp.modelo, comp.nas

    # ---------- actualizacion post-solver ----------
    def set_last_solution(self, c_calculada: np.ndarray) -> None:
        """
        Guarda la ultima solucion del solver.
        c_calculada: (n_reacciones, nspec) en el orden de la ultima compilacion.
        """
        self._last_solution = np.asarray(c_calculada, dtype=float)

    def last_solution(self) -> Optional[np.ndarray]:
        return self._last_solution

    def last_compilation(self) -> Optional[Compilation]:
        return self._last_compilation
