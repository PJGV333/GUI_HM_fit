# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StoichiometricEdge:
    """
    Arista componente -> especie con un coeficiente entero positivo.
    Mantenerla explícita facilita:
      - validación (coeficientes no negativos)
      - inspección y debug
      - futuras extensiones (p.ej., etiquetas, fuentes de datos, etc.)
    """

    component_key: str
    species_key: str
    coefficient: int
