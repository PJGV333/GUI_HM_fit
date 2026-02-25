# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import re
from typing import Any, Tuple

from hmfit_core.graph.chemical_graph import ChemicalGraph, create_solver_inputs_from_graph


def _parse_log_beta(raw_value: str) -> float:
    text = str(raw_value or "").strip()
    if not text:
        raise ValueError("Missing equilibrium constant after ';'.")

    if "=" in text:
        _, rhs = text.split("=", 1)
        text = rhs.strip()
        if not text:
            raise ValueError("Missing numeric value after '='.")

    try:
        return float(text)
    except Exception as exc:
        raise ValueError(f"Invalid equilibrium constant {raw_value!r}.") from exc


def parse_multiline_equilibria(text_block: str) -> Tuple[ChemicalGraph, dict[str, Any]]:
    """
    Parse a multiline equilibrium block into a ChemicalGraph and solver payload.

    Expected line format:
      `equation ; constant`
    Example:
      `H + G <=> HG ; 4.5`
      `H + G <=> HG ; logB=4.5`
    """

    graph = ChemicalGraph()
    valid_count = 0
    non_abs_set: set[str] = set()

    for line_no, raw_line in enumerate(str(text_block or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            line_to_parse = line
            if "@na" in line.lower():
                parts = re.split(r"@na", line, maxsplit=1, flags=re.IGNORECASE)
                line_to_parse = parts[0].strip()
                non_abs_raw = parts[1] if len(parts) > 1 else ""
                for token in non_abs_raw.split(","):
                    species = str(token or "").strip()
                    if species:
                        non_abs_set.add(species)

            if ";" not in line_to_parse:
                raise ValueError("Expected ';' delimiter between equation and constant.")
            reaction_text, const_text = line_to_parse.split(";", 1)
            reaction = reaction_text.strip()
            if not reaction:
                raise ValueError("Missing reaction equation before ';'.")
            log_beta = _parse_log_beta(const_text)
            graph.add_reaction_from_string(reaction, log_beta=log_beta)
            valid_count += 1
        except Exception as exc:
            raise ValueError(f"Line {line_no}: {exc}") from exc

    if valid_count == 0:
        raise ValueError("No valid equilibrium lines found.")

    solver_inputs = create_solver_inputs_from_graph(graph)
    solver_inputs["non_abs_species"] = sorted(non_abs_set)
    return graph, solver_inputs
