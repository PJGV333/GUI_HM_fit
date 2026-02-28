# SPDX-License-Identifier: GPL-3.0-or-later

from __future__ import annotations

import copy
import hashlib
import re
from collections import OrderedDict
from typing import Any, Tuple

from hmfit_core.graph.chemical_graph import ChemicalGraph, create_solver_inputs_from_graph

_PARSE_CACHE_MAXSIZE = 64
_PARSE_CACHE: "OrderedDict[str, tuple[ChemicalGraph, dict[str, Any]]]" = OrderedDict()
_PARSE_CACHE_STATS = {"hits": 0, "misses": 0}


def normalize_equilibria_text(text_block: str) -> str:
    lines: list[str] = []
    for raw_line in str(text_block or "").splitlines():
        line = str(raw_line or "").strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"\s+", " ", line)
        line = re.sub(r"\s*;\s*", " ; ", line)
        line = re.sub(r"\s*<=>\s*", " <=> ", line)
        line = re.sub(r"\s*\+\s*", " + ", line)
        line = re.sub(r"\s*=\s*", "=", line)
        line = re.sub(r"\s*(?i:@na)\s*", " @na ", line)
        line = re.sub(r"\s+", " ", line).strip()
        lines.append(line)
    return "\n".join(lines)


def clear_parse_cache() -> None:
    _PARSE_CACHE.clear()
    _PARSE_CACHE_STATS["hits"] = 0
    _PARSE_CACHE_STATS["misses"] = 0


def get_parse_cache_stats() -> dict[str, int]:
    return {
        "hits": int(_PARSE_CACHE_STATS["hits"]),
        "misses": int(_PARSE_CACHE_STATS["misses"]),
        "size": int(len(_PARSE_CACHE)),
        "maxsize": int(_PARSE_CACHE_MAXSIZE),
    }


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

    normalized = normalize_equilibria_text(text_block)
    cache_key = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
    cached = _PARSE_CACHE.get(cache_key)
    if cached is not None:
        _PARSE_CACHE_STATS["hits"] += 1
        _PARSE_CACHE.move_to_end(cache_key)
        graph_cached, solver_cached = cached
        return copy.deepcopy(graph_cached), copy.deepcopy(solver_cached)
    _PARSE_CACHE_STATS["misses"] += 1

    graph = ChemicalGraph()
    valid_count = 0
    non_abs_set: set[str] = set()

    for line_no, raw_line in enumerate(str(text_block or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            # Case 1: dedicated non-absorbent line.
            if line.strip().lower().startswith("@na"):
                parts = re.split(r"(?i)@na", line, maxsplit=1)
                non_abs_raw = parts[1] if len(parts) > 1 else ""
                for token in non_abs_raw.split(","):
                    species = str(token or "").strip()
                    if species:
                        non_abs_set.add(species)
                continue

            # Case 2: equation line, optionally with inline @NA declaration.
            parts = re.split(r"(?i)@na", line, maxsplit=1)
            line_to_parse = parts[0].strip()
            inline_non_abs: list[str] = []
            has_inline_na = len(parts) > 1
            if has_inline_na:
                non_abs_raw = parts[1]
                inline_non_abs = [
                    str(token or "").strip()
                    for token in non_abs_raw.split(",")
                    if str(token or "").strip()
                ]

            if ";" not in line_to_parse:
                raise ValueError("Expected ';' delimiter between equation and constant.")
            reaction_text, const_text = line_to_parse.split(";", 1)
            reaction = reaction_text.strip()
            if not reaction:
                raise ValueError("Missing reaction equation before ';'.")
            log_beta = _parse_log_beta(const_text)
            edge = graph.add_reaction_from_string(reaction, log_beta=log_beta)

            # Inline @NA semantics:
            # - '@na' (without names): mark reaction products as non-absorbent.
            # - '@na a, b': keep explicit names as provided.
            if has_inline_na:
                if inline_non_abs:
                    for species_name in inline_non_abs:
                        non_abs_set.add(species_name)
                else:
                    for species in edge.products.keys():
                        non_abs_set.add(species.name)
            valid_count += 1
        except Exception as exc:
            raise ValueError(f"Line {line_no}: {exc}") from exc

    if valid_count == 0:
        raise ValueError("No valid equilibrium lines found.")

    solver_inputs = create_solver_inputs_from_graph(graph)
    solver_inputs["non_abs_species"] = list(non_abs_set)
    _PARSE_CACHE[cache_key] = (copy.deepcopy(graph), copy.deepcopy(solver_inputs))
    _PARSE_CACHE.move_to_end(cache_key)
    while len(_PARSE_CACHE) > _PARSE_CACHE_MAXSIZE:
        _PARSE_CACHE.popitem(last=False)
    return graph, solver_inputs
