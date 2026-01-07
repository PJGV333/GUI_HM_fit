"""Simple parser for kinetics mechanism definitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, List, Optional

from .ast import MechanismAST, ReactionAST, TempModelAST


IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TERM_RE = re.compile(r"^\s*(\d+)?\s*([A-Za-z_][A-Za-z0-9_]*)\s*$")
MODEL_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$")


class MechanismParseError(ValueError):
    """Raised when a mechanism definition cannot be parsed."""


@dataclass
class _TempBlock:
    kind: str


def parse_file(path: str | Path) -> MechanismAST:
    """Parse a mechanism file from disk."""
    text = Path(path).read_text(encoding="utf-8")
    return parse_mechanism(text)


def parse_mechanism(text: str) -> MechanismAST:
    """Parse a mechanism definition into an AST."""
    species: List[str] = []
    fixed: set[str] = set()
    reactions: List[ReactionAST] = []
    temp_models: Dict[str, TempModelAST] = {}
    temp_block: Optional[_TempBlock] = None

    for line_num, raw_line in enumerate(text.splitlines(), start=1):
        line = _strip_comment(raw_line).strip()
        if not line:
            continue

        if line.lower().startswith("species:"):
            temp_block = None
            species = _parse_list(line.split(":", 1)[1], line_num)
            continue

        if line.lower().startswith("fixed:"):
            temp_block = None
            fixed.update(_parse_list(line.split(":", 1)[1], line_num))
            continue

        if line.lower().startswith("arrhenius:"):
            temp_block = _parse_temp_block("arrhenius", line, line_num, temp_models)
            continue

        if line.lower().startswith("eyring:"):
            temp_block = _parse_temp_block("eyring", line, line_num, temp_models)
            continue

        if "->" in line:
            temp_block = None
            reactions.append(_parse_reaction(line, line_num))
            continue

        if temp_block is not None:
            _parse_temp_model_line(line, temp_block.kind, line_num, temp_models)
            continue

        raise MechanismParseError(f"Unrecognized line {line_num}: {raw_line}")

    return MechanismAST(
        species=species,
        fixed=fixed,
        reactions=reactions,
        temp_models=temp_models,
    )


def _strip_comment(line: str) -> str:
    if "#" not in line:
        return line
    return line.split("#", 1)[0]


def _parse_list(raw: str, line_num: int) -> List[str]:
    entries = [item.strip() for item in raw.split(",")]
    entries = [item for item in entries if item]
    if not entries:
        raise MechanismParseError(f"Expected at least one entry on line {line_num}.")
    for entry in entries:
        if not IDENT_RE.match(entry):
            raise MechanismParseError(f"Invalid identifier '{entry}' on line {line_num}.")
    return entries


def _parse_reaction(line: str, line_num: int) -> ReactionAST:
    if ";" not in line:
        raise MechanismParseError(f"Missing ';' separator on line {line_num}.")
    lhs_rhs, params_raw = line.split(";", 1)
    params = [item.strip() for item in params_raw.split(",") if item.strip()]

    if "<->" in lhs_rhs:
        if lhs_rhs.count("<->") != 1:
            raise MechanismParseError(f"Invalid reversible arrow on line {line_num}.")
        lhs, rhs = lhs_rhs.split("<->")
        if len(params) != 2:
            raise MechanismParseError(
                f"Reversible reaction expects two parameters on line {line_num}."
            )
        k_forward, k_reverse = params
    elif "->" in lhs_rhs:
        if lhs_rhs.count("->") != 1:
            raise MechanismParseError(f"Invalid arrow on line {line_num}.")
        lhs, rhs = lhs_rhs.split("->")
        if len(params) != 1:
            raise MechanismParseError(
                f"Irreversible reaction expects one parameter on line {line_num}."
            )
        k_forward = params[0]
        k_reverse = None
    else:
        raise MechanismParseError(f"Missing reaction arrow on line {line_num}.")

    _validate_identifiers(params, line_num)

    reactants = _parse_side(lhs, line_num)
    products = _parse_side(rhs, line_num)

    return ReactionAST(
        reactants=reactants,
        products=products,
        k_forward=k_forward,
        k_reverse=k_reverse,
    )


def _parse_side(side: str, line_num: int) -> Dict[str, int]:
    side = side.strip()
    if not side:
        raise MechanismParseError(f"Empty reaction side on line {line_num}.")

    species: Dict[str, int] = {}
    for part in side.split("+"):
        match = TERM_RE.match(part)
        if not match:
            raise MechanismParseError(
                f"Invalid species term '{part.strip()}' on line {line_num}."
            )
        coeff = int(match.group(1) or 1)
        name = match.group(2)
        species[name] = species.get(name, 0) + coeff
    return species


def _parse_temp_block(
    kind: str,
    line: str,
    line_num: int,
    temp_models: Dict[str, TempModelAST],
) -> Optional[_TempBlock]:
    _, rest = line.split(":", 1)
    rest = rest.strip()
    if not rest:
        return _TempBlock(kind=kind)
    _parse_temp_model_line(rest, kind, line_num, temp_models)
    return None


def _parse_temp_model_line(
    line: str,
    kind: str,
    line_num: int,
    temp_models: Dict[str, TempModelAST],
) -> None:
    match = MODEL_RE.match(line)
    if not match:
        raise MechanismParseError(
            f"Invalid temperature model line {line_num}: {line}"
        )
    k_name, params_raw = match.groups()
    if k_name in temp_models:
        raise MechanismParseError(
            f"Duplicate temperature model for '{k_name}' on line {line_num}."
        )
    if not IDENT_RE.match(k_name):
        raise MechanismParseError(
            f"Invalid parameter name '{k_name}' on line {line_num}."
        )

    params = _parse_param_pairs(params_raw, line_num)
    temp_models[k_name] = TempModelAST(kind=kind, params=params)


def _parse_param_pairs(raw: str, line_num: int) -> Dict[str, str]:
    params: Dict[str, str] = {}
    parts = [item.strip() for item in raw.split(",") if item.strip()]
    if not parts:
        raise MechanismParseError(f"Expected parameters on line {line_num}.")
    for part in parts:
        if "=" not in part:
            raise MechanismParseError(
                f"Expected key=value pair '{part}' on line {line_num}."
            )
        key, value = [item.strip() for item in part.split("=", 1)]
        if not key or not value:
            raise MechanismParseError(
                f"Invalid key=value pair '{part}' on line {line_num}."
            )
        _validate_identifiers([key, value], line_num)
        params[key] = value
    return params


def _validate_identifiers(items: List[str], line_num: int) -> None:
    for item in items:
        if not IDENT_RE.match(item):
            raise MechanismParseError(
                f"Invalid identifier '{item}' on line {line_num}."
            )
