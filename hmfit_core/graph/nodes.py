# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Node:
    """
    Nodo base. Se usa para componentes y especies.

    - active: si el nodo participa en el modelo actual
    - precipitated: si la especie quedó fuera de solución (se excluye al compilar)
    - observable: si contribuye a la señal (si False -> va a 'nas')
    """

    key: str
    name: Optional[str] = None
    active: bool = True
    precipitated: bool = False
    observable: bool = True
    order: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComponentNode(Node):
    """
    Un componente químico (filas de la matriz).
    Por diseño, también existirá como especie libre al compilar (columnas identidad).
    """

    # Totales por corrida (opcional; si lo usas, guardas aquí un vector por corrida)
    ctot: Optional[Any] = None


@dataclass
class SpeciesNode(Node):
    """
    Una especie compleja (columnas posteriores a las de componentes).
    La estequiometría se expresa como un dict: {component_key: coef_int}
    """

    stoich: Dict[str, int] = field(default_factory=dict)
    # Para mapear parámetros (logK/logBeta) sin obligar todavía a una convención
    logK: Optional[float] = None
