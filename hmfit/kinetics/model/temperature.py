"""Temperature-dependent parameter resolution (MVP hook)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

from ..mechanism_editor.ast import TempModelAST


TemperatureType = float | Callable[[float], float]


@dataclass(frozen=True)
class ParamResolver:
    """Resolve base parameters into kinetic rates for a given temperature."""

    temp_models: Mapping[str, TempModelAST] | None = None

    def resolve(
        self, params: Mapping[str, float], temperature: TemperatureType
    ) -> dict[str, float]:
        if self.temp_models:
            raise NotImplementedError(
                "Temperature-dependent parameters are not implemented yet."
            )
        return dict(params)
