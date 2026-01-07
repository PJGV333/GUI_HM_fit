"""Temperature-dependent parameter resolution (MVP hook)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import math

from ..mechanism_editor.ast import TempModelAST


TemperatureType = float | Callable[[float], float]

R_GAS = 8.314462618  # J / (mol K)
K_BOLTZ = 1.380649e-23  # J / K
H_PLANCK = 6.62607015e-34  # J s


def arrhenius(A: float, Ea: float, temperature: float) -> float:
    """Arrhenius rate constant."""
    return float(A) * math.exp(-float(Ea) / (R_GAS * float(temperature)))


def eyring(dH: float, dS: float, temperature: float) -> float:
    """Eyring rate constant."""
    temp = float(temperature)
    return (K_BOLTZ * temp / H_PLANCK) * math.exp(float(dS) / R_GAS) * math.exp(
        -float(dH) / (R_GAS * temp)
    )


@dataclass(frozen=True)
class ParamResolver:
    """Resolve base parameters into kinetic rates for a given temperature."""

    temp_models: Mapping[str, TempModelAST] | None = None

    def resolve(
        self, params: Mapping[str, float], temperature: TemperatureType
    ) -> dict[str, float]:
        resolved = dict(params)
        if not self.temp_models:
            return resolved

        temp = float(temperature)
        for k_name, model in self.temp_models.items():
            if model.kind == "arrhenius":
                A_key = model.params.get("A")
                Ea_key = model.params.get("Ea")
                if A_key is None or Ea_key is None:
                    raise ValueError(
                        f"Arrhenius model for '{k_name}' requires A and Ea."
                    )
                if A_key not in params:
                    raise KeyError(f"Missing Arrhenius parameter '{A_key}'.")
                if Ea_key not in params:
                    raise KeyError(f"Missing Arrhenius parameter '{Ea_key}'.")
                resolved[k_name] = arrhenius(params[A_key], params[Ea_key], temp)
            elif model.kind == "eyring":
                dH_key = model.params.get("dH")
                dS_key = model.params.get("dS")
                if dH_key is None or dS_key is None:
                    raise ValueError(f"Eyring model for '{k_name}' requires dH and dS.")
                if dH_key not in params:
                    raise KeyError(f"Missing Eyring parameter '{dH_key}'.")
                if dS_key not in params:
                    raise KeyError(f"Missing Eyring parameter '{dS_key}'.")
                resolved[k_name] = eyring(params[dH_key], params[dS_key], temp)
            else:
                raise ValueError(f"Unknown temperature model '{model.kind}'.")

        return resolved
