from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence
import re

from hmfit_core.acid_base import (
    AcidBaseComponent,
    AcidBaseSpecies,
    AcidBaseSystem,
    log_beta_to_pka,
    pka_to_log_beta,
)

_PROTON_ALIASES = {"h", "h+", "proton"}
_CHARGE_TOKEN_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+?)(?:\((?P<charge>[+-]?\d+)\))?$")
_SIDE_TERM_RE = re.compile(r"^\s*(?:(?P<coef>\d+)\s*)?(?P<token>[A-Za-z0-9_()+\-]+)\s*$")
_CONST_RE = re.compile(
    r"(?P<kind>pka|logb|log_beta|logbeta)\s*=\s*(?P<value>[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)",
    flags=re.IGNORECASE,
)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(float(value))
    except Exception:
        return default


def normalize_constant_mode(mode: Any) -> str:
    text = str(mode or "pKa").strip().lower()
    if text in {"logb", "log_beta", "logbeta", "beta"}:
        return "log_beta"
    return "pKa"


def is_proton_component_name(name: Any) -> bool:
    text = str(name or "").strip().lower()
    return text in _PROTON_ALIASES


def proton_component_name() -> str:
    return "H"


def _normalize_component_role(comp: Mapping[str, Any], *, fallback: str = "analyte") -> str:
    role = str(comp.get("role") or "").strip().lower()
    if role:
        if role in {"imposed ph", "imposed pH", "imposed_ph"}:
            return "imposed pH"
        return role
    if bool(comp.get("is_proton")) or is_proton_component_name(comp.get("name")):
        return "proton"
    return fallback


def _default_species_name(component_name: str, h_count: int) -> str:
    if h_count <= 0:
        return component_name
    prefix = "H" if h_count == 1 else f"H{h_count}"
    return f"{prefix}{component_name}"


def _component_lookup(model: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for comp in list(model.get("components") or []):
        name = str(comp.get("name") or "").strip()
        if name:
            lookup[name] = dict(comp)
    return lookup


def _primary_component_name(model: Mapping[str, Any]) -> str:
    components = [dict(comp) for comp in list(model.get("components") or [])]
    for comp in components:
        if str(comp.get("role") or "").strip().lower() == "analyte" and not bool(comp.get("is_proton")):
            name = str(comp.get("name") or "").strip()
            if name:
                return name
    for comp in components:
        name = str(comp.get("name") or "").strip()
        if name and not bool(comp.get("is_proton")) and not is_proton_component_name(name):
            return name
    return "L"


def _build_ladder_species(
    component_name: str,
    *,
    base_charge: int,
    log_beta: Sequence[float],
    observable: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "name": component_name,
            "charge": int(base_charge),
            "h_count": 0,
            "include": True,
            "observable": bool(observable),
            "fixed": True,
            "non_observable": False,
            "parent_component": component_name,
            "log_beta": 0.0,
        }
    ]
    for h_count, value in enumerate(list(log_beta or []), start=1):
        rows.append(
            {
                "name": _default_species_name(component_name, h_count),
                "charge": int(base_charge + h_count),
                "h_count": h_count,
                "include": True,
                "observable": bool(observable),
                "fixed": False,
                "non_observable": False,
                "parent_component": component_name,
                "log_beta": float(value),
            }
        )
    return rows


def _build_matrix(component_names: Sequence[str], species_rows: Sequence[Mapping[str, Any]]) -> list[list[int]]:
    proton_name = proton_component_name()
    matrix = [[0 for _ in range(len(species_rows))] for _ in range(len(component_names))]
    for col, sp in enumerate(species_rows):
        parent = str(sp.get("parent_component") or "").strip()
        h_count = int(_safe_int(sp.get("h_count"), 0) or 0)
        for row, comp_name in enumerate(component_names):
            if comp_name == parent:
                matrix[row][col] = 1
            elif comp_name == proton_name:
                matrix[row][col] = h_count
    return matrix


def _make_template(
    template_id: str,
    *,
    component_name: str,
    pka: Sequence[float],
    base_charge: int,
    analytical_concentration: float,
) -> dict[str, Any]:
    log_beta = pka_to_log_beta([float(v) for v in pka])
    species_rows = _build_ladder_species(
        component_name,
        base_charge=int(base_charge),
        log_beta=log_beta,
    )
    components = [
        {
            "name": component_name,
            "role": "analyte",
            "analytical_concentration": float(analytical_concentration),
            "charge": int(base_charge),
            "is_proton": False,
            "is_titrant": False,
            "is_background": False,
            "fixed_concentration": False,
            "implicit": False,
        },
        {
            "name": proton_component_name(),
            "role": "proton",
            "analytical_concentration": None,
            "charge": 1,
            "is_proton": True,
            "is_titrant": False,
            "is_background": False,
            "fixed_concentration": True,
            "implicit": True,
        },
    ]
    component_names = [str(comp["name"]) for comp in components]
    species_names = [str(sp["name"]) for sp in species_rows]
    return {
        "template_id": template_id,
        "definition_mode": "matrix",
        "constant_mode": "pKa",
        "components": components,
        "species": species_rows,
        "stoichiometric_matrix": _build_matrix(component_names, species_rows),
        "component_names": component_names,
        "species_names": species_names,
        "pka": [float(v) for v in pka],
        "log_beta": [float(v) for v in log_beta],
        "equations_text": "",
    }


def build_acid_base_template(
    template_id: str,
    *,
    analytical_concentration: float = 1.0e-3,
) -> dict[str, Any]:
    key = str(template_id or "").strip().lower()
    if key == "simple_monoprotic":
        return _make_template(
            "simple_monoprotic",
            component_name="L",
            pka=[5.20],
            base_charge=-1,
            analytical_concentration=analytical_concentration,
        )
    if key == "diprotic_ligand":
        return _make_template(
            "diprotic_ligand",
            component_name="L",
            pka=[4.50, 8.90],
            base_charge=-2,
            analytical_concentration=analytical_concentration,
        )
    if key == "triprotic_ligand":
        return _make_template(
            "triprotic_ligand",
            component_name="L",
            pka=[2.10, 6.70, 10.20],
            base_charge=-3,
            analytical_concentration=analytical_concentration,
        )
    if key == "multiple_components":
        model = _make_template(
            "multiple_components",
            component_name="L1",
            pka=[5.20],
            base_charge=-1,
            analytical_concentration=analytical_concentration,
        )
        components = list(model["components"])
        species = list(model["species"])
        components.insert(
            1,
            {
                "name": "L2",
                "role": "analyte",
                "analytical_concentration": float(analytical_concentration),
                "charge": -1,
                "is_proton": False,
                "is_titrant": False,
                "is_background": False,
                "fixed_concentration": False,
                "implicit": False,
            },
        )
        species.extend(
            _build_ladder_species(
                "L2",
                base_charge=-1,
                log_beta=pka_to_log_beta([6.80]),
            )
        )
        component_names = [str(comp["name"]) for comp in components]
        model["components"] = components
        model["species"] = species
        model["component_names"] = component_names
        model["species_names"] = [str(sp["name"]) for sp in species]
        model["stoichiometric_matrix"] = _build_matrix(component_names, species)
        return model
    if key in {"custom_acid_base_system", "custom"}:
        model = _make_template(
            "custom_acid_base_system",
            component_name="L",
            pka=[5.20],
            base_charge=-1,
            analytical_concentration=analytical_concentration,
        )
        model["template_id"] = "custom_acid_base_system"
        return model
    raise ValueError(f"Unknown acid-base template: {template_id}")


def _parse_equation_token(token: str) -> tuple[str, int | None]:
    text = str(token or "").strip()
    if not text:
        raise ValueError("Empty species token in equation editor.")
    if is_proton_component_name(text):
        return proton_component_name(), 1 if text.strip().lower() == "h+" else None
    match = _CHARGE_TOKEN_RE.match(text)
    if match is None:
        raise ValueError(f"Could not parse species token {text!r}.")
    name = str(match.group("name") or "").strip()
    charge_raw = match.group("charge")
    charge = None if charge_raw in (None, "") else int(charge_raw)
    return name, charge


def _parse_equation_side(side: str) -> list[tuple[int, str, int | None]]:
    terms: list[tuple[int, str, int | None]] = []
    for raw_term in re.split(r"\s+\+\s+", str(side or "").strip()):
        term = raw_term.strip()
        if not term:
            continue
        match = _SIDE_TERM_RE.match(term)
        if match is None:
            raise ValueError(f"Could not parse equation term {term!r}.")
        coef = int(match.group("coef") or 1)
        token = str(match.group("token") or "").strip()
        name, charge = _parse_equation_token(token)
        terms.append((coef, name, charge))
    if not terms:
        raise ValueError(f"Equation side {side!r} is empty.")
    return terms


def acid_base_model_from_equations(
    equations_text: str,
    *,
    analytical_concentration: float = 1.0e-3,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    constants_kind: str | None = None
    base_component_name = "L"
    base_charge = 0
    known_h_count: dict[str, int] = {}
    known_charge: dict[str, int] = {}
    cumulative_log_beta: dict[int, float] = {}

    for raw_line in str(equations_text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ";" not in line:
            raise ValueError("Each equation line must include '; pKa=...' or '; logB=...'.")
        eqn_text, meta_text = [part.strip() for part in line.split(";", 1)]
        if "<=>" in eqn_text:
            lhs_text, rhs_text = [part.strip() for part in eqn_text.split("<=>", 1)]
        elif "⇌" in eqn_text:
            lhs_text, rhs_text = [part.strip() for part in eqn_text.split("⇌", 1)]
        else:
            raise ValueError("Equation editor expects '<=>'.")

        lhs_terms = _parse_equation_side(lhs_text)
        rhs_terms = _parse_equation_side(rhs_text)
        const_match = _CONST_RE.search(meta_text)
        if const_match is None:
            raise ValueError(f"Could not parse constant in line {line!r}.")
        kind = str(const_match.group("kind") or "").strip().lower()
        value = float(const_match.group("value"))
        current_mode = "log_beta" if kind.startswith("log") else "pKa"
        if constants_kind is None:
            constants_kind = current_mode
        elif constants_kind != current_mode:
            raise ValueError("Do not mix pKa and logB definitions in the same equation block.")

        proton_lhs = sum(coef for coef, name, _charge in lhs_terms if is_proton_component_name(name))
        proton_rhs = sum(coef for coef, name, _charge in rhs_terms if is_proton_component_name(name))
        delta_h = proton_lhs - proton_rhs
        if delta_h <= 0:
            raise ValueError("Each acid-base equation must add one or more protons on the left side.")

        reactants = [(coef, name, charge) for coef, name, charge in lhs_terms if not is_proton_component_name(name)]
        products = [(coef, name, charge) for coef, name, charge in rhs_terms if not is_proton_component_name(name)]
        if len(products) != 1 or products[0][0] != 1:
            raise ValueError("Each equation must produce exactly one acid-base species.")
        if len(reactants) != 1 or reactants[0][0] != 1:
            raise ValueError("Each equation must consume exactly one non-proton species.")

        _coef_r, reactant_name, reactant_charge = reactants[0]
        _coef_p, product_name, product_charge = products[0]
        if not known_h_count:
            base_component_name = reactant_name
            known_h_count[reactant_name] = 0
            known_charge[reactant_name] = 0 if reactant_charge is None else int(reactant_charge)
        reactant_h = int(known_h_count.get(reactant_name, 0))
        product_h = reactant_h + delta_h
        known_h_count[reactant_name] = reactant_h
        known_h_count[product_name] = product_h

        if reactant_charge is not None:
            known_charge[reactant_name] = int(reactant_charge)
        if product_charge is not None:
            known_charge[product_name] = int(product_charge)
        elif reactant_name in known_charge:
            known_charge[product_name] = int(known_charge[reactant_name]) + delta_h

        if current_mode == "pKa":
            prev = cumulative_log_beta.get(reactant_h, 0.0)
            cumulative = prev + float(value)
        else:
            cumulative = float(value)
        cumulative_log_beta[product_h] = cumulative

    if not known_h_count:
        raise ValueError("Equation editor did not contain any acid-base equations.")

    max_h = max(known_h_count.values())
    species_rows: list[dict[str, Any]] = []
    for h_count in range(max_h + 1):
        matching = [name for name, value in known_h_count.items() if value == h_count]
        if not matching:
            matching = [_default_species_name(base_component_name, h_count)]
        species_name = matching[0]
        base_charge = int(known_charge.get(base_component_name, 0))
        charge = int(known_charge.get(species_name, base_charge + h_count))
        species_rows.append(
            {
                "name": species_name,
                "charge": charge,
                "h_count": h_count,
                "include": True,
                "observable": True,
                "fixed": h_count == 0,
                "non_observable": False,
                "parent_component": base_component_name,
                "log_beta": 0.0 if h_count == 0 else float(cumulative_log_beta[h_count]),
            }
        )

    log_beta = [float(cumulative_log_beta[h_count]) for h_count in range(1, max_h + 1)]
    pka = log_beta_to_pka(log_beta)
    components = [
        {
            "name": base_component_name,
            "role": "analyte",
            "analytical_concentration": float(analytical_concentration),
            "charge": int(known_charge.get(base_component_name, 0)),
            "is_proton": False,
            "is_titrant": False,
            "is_background": False,
            "fixed_concentration": False,
            "implicit": False,
        },
        {
            "name": proton_component_name(),
            "role": "proton",
            "analytical_concentration": None,
            "charge": 1,
            "is_proton": True,
            "is_titrant": False,
            "is_background": False,
            "fixed_concentration": True,
            "implicit": True,
        },
    ]
    component_names = [str(comp["name"]) for comp in components]
    return {
        "template_id": "equation_editor",
        "definition_mode": "equations",
        "constant_mode": constants_kind or "pKa",
        "equations_text": str(equations_text or "").strip(),
        "components": components,
        "species": species_rows,
        "stoichiometric_matrix": _build_matrix(component_names, species_rows),
        "component_names": component_names,
        "species_names": [str(sp["name"]) for sp in species_rows],
        "pka": [float(v) for v in pka],
        "log_beta": [float(v) for v in log_beta],
    }


def _looks_like_legacy_model(model_def: Mapping[str, Any]) -> bool:
    if model_def.get("definition_mode") or model_def.get("stoichiometric_matrix"):
        return False
    components = list(model_def.get("components") or [])
    if not components:
        return False
    first = dict(components[0])
    return any(key in first for key in {"base_charge", "n_steps", "use_log_beta"})


def _legacy_model_to_canonical(model_def: Mapping[str, Any]) -> dict[str, Any]:
    components_in = [dict(item) for item in list(model_def.get("components") or [])]
    species_in = [dict(item) for item in list(model_def.get("species") or []) if bool(dict(item).get("include", True))]
    components: list[dict[str, Any]] = []
    for idx, comp in enumerate(components_in):
        name = str(comp.get("name") or f"L{idx + 1}").strip()
        components.append(
            {
                "name": name,
                "role": _normalize_component_role(comp, fallback="analyte" if idx == 0 else "spectator"),
                "analytical_concentration": _safe_float(comp.get("analytical_concentration"), 0.0),
                "charge": int(_safe_int(comp.get("base_charge"), 0) or 0),
                "is_proton": False,
                "is_titrant": str(comp.get("role") or "").strip().lower() == "titrant",
                "is_background": str(comp.get("role") or "").strip().lower() in {"background", "spectator"},
                "fixed_concentration": bool(comp.get("fixed_concentration", False)),
                "implicit": False,
            }
        )
    has_proton = any(is_proton_component_name(comp.get("name")) or bool(comp.get("is_proton")) for comp in components)
    if not has_proton:
        components.append(
            {
                "name": proton_component_name(),
                "role": "proton",
                "analytical_concentration": None,
                "charge": 1,
                "is_proton": True,
                "is_titrant": False,
                "is_background": False,
                "fixed_concentration": True,
                "implicit": True,
            }
        )

    species: list[dict[str, Any]] = []
    for sp in species_in:
        parent = str(sp.get("parent_component") or sp.get("component") or "").strip()
        species.append(
            {
                "name": str(sp.get("name") or "").strip(),
                "charge": _safe_int(sp.get("charge"), 0),
                "h_count": int(_safe_int(sp.get("h_count"), 0) or 0),
                "include": bool(sp.get("include", True)),
                "observable": True,
                "fixed": bool(sp.get("fixed", False)),
                "non_observable": False,
                "parent_component": parent,
                "log_beta": _safe_float(sp.get("log_beta"), 0.0),
            }
        )

    primary = components[0]["name"] if components else "L"
    log_beta = [
        float(sp.get("log_beta") or 0.0)
        for sp in sorted(
            [sp for sp in species if str(sp.get("parent_component") or "") == primary and int(sp.get("h_count") or 0) > 0],
            key=lambda item: int(item.get("h_count") or 0),
        )
    ]
    return {
        "template_id": str(model_def.get("model_type") or "legacy"),
        "definition_mode": "matrix",
        "constant_mode": "log_beta" if bool(components_in and components_in[0].get("use_log_beta", False)) else "pKa",
        "components": components,
        "species": species,
        "component_names": [str(comp["name"]) for comp in components],
        "species_names": [str(sp["name"]) for sp in species],
        "stoichiometric_matrix": _build_matrix([str(comp["name"]) for comp in components], species),
        "pka": log_beta_to_pka(log_beta) if log_beta else [],
        "log_beta": log_beta,
        "equations_text": "",
    }


def acid_base_model_from_simple_config(
    *,
    component_name: str = "L",
    pka: Sequence[float] | None = None,
    analytical_concentration: float = 1.0e-3,
    base_charge: int = -1,
) -> dict[str, Any]:
    pka_values = [float(v) for v in list(pka or [5.0])]
    model = _make_template(
        "legacy_simple_config",
        component_name=str(component_name or "L"),
        pka=pka_values,
        base_charge=int(base_charge),
        analytical_concentration=float(analytical_concentration),
    )
    return model


def canonicalize_acid_base_model(
    model_def: Mapping[str, Any] | None,
    *,
    fallback_component_name: str = "L",
    fallback_pka: Sequence[float] | None = None,
    fallback_concentration: float = 1.0e-3,
    fallback_base_charge: int = -1,
) -> dict[str, Any]:
    if not isinstance(model_def, Mapping) or not model_def:
        return acid_base_model_from_simple_config(
            component_name=fallback_component_name,
            pka=fallback_pka,
            analytical_concentration=fallback_concentration,
            base_charge=fallback_base_charge,
        )

    if _looks_like_legacy_model(model_def):
        raw = _legacy_model_to_canonical(model_def)
    else:
        raw = deepcopy(dict(model_def))

    components_in = [dict(item) for item in list(raw.get("components") or [])]
    species_in = [dict(item) for item in list(raw.get("species") or [])]
    if not components_in:
        return acid_base_model_from_simple_config(
            component_name=fallback_component_name,
            pka=fallback_pka,
            analytical_concentration=fallback_concentration,
            base_charge=fallback_base_charge,
        )

    components: list[dict[str, Any]] = []
    for idx, comp in enumerate(components_in):
        name = str(comp.get("name") or f"C{idx + 1}").strip()
        role = _normalize_component_role(comp, fallback="analyte" if idx == 0 else "spectator")
        is_proton = bool(comp.get("is_proton")) or is_proton_component_name(name) or role == "proton"
        components.append(
            {
                "name": name,
                "role": "proton" if is_proton else role,
                "analytical_concentration": _safe_float(comp.get("analytical_concentration"), None),
                "charge": int(_safe_int(comp.get("charge", comp.get("base_charge")), 1 if is_proton else 0) or 0),
                "is_proton": bool(is_proton),
                "is_titrant": bool(comp.get("is_titrant", False)) or role == "titrant",
                "is_background": bool(comp.get("is_background", False)) or role in {"background", "spectator"},
                "fixed_concentration": bool(comp.get("fixed_concentration", False)),
                "implicit": bool(comp.get("implicit", False)) or is_proton,
            }
        )
    if not any(bool(comp.get("is_proton")) for comp in components):
        components.append(
            {
                "name": proton_component_name(),
                "role": "proton",
                "analytical_concentration": None,
                "charge": 1,
                "is_proton": True,
                "is_titrant": False,
                "is_background": False,
                "fixed_concentration": True,
                "implicit": True,
            }
        )
    component_names = [str(comp["name"]) for comp in components]
    proton_row = next(
        (idx for idx, comp in enumerate(components) if bool(comp.get("is_proton"))),
        None,
    )
    matrix_in = list(raw.get("stoichiometric_matrix") or [])
    matrix: list[list[int]] = [[0 for _ in range(len(species_in))] for _ in range(len(component_names))]
    for row_idx in range(min(len(matrix), len(matrix_in))):
        row_values = list(matrix_in[row_idx] or [])
        for col_idx in range(min(len(species_in), len(row_values))):
            matrix[row_idx][col_idx] = int(_safe_int(row_values[col_idx], 0) or 0)

    primary_name = _primary_component_name({"components": components})
    comp_lookup = {str(comp["name"]): dict(comp) for comp in components}
    species: list[dict[str, Any]] = []
    for idx, sp in enumerate(species_in):
        name = str(sp.get("name") or raw.get("species_names", [])[idx] if idx < len(list(raw.get("species_names") or [])) else "").strip()
        parent = str(
            sp.get("parent_component")
            or sp.get("component")
            or ""
        ).strip()
        if not parent:
            for row_idx, comp_name in enumerate(component_names):
                if proton_row is not None and row_idx == proton_row:
                    continue
                if idx < len(matrix[row_idx]) and int(matrix[row_idx][idx]) > 0:
                    parent = comp_name
                    break
        if not parent:
            parent = primary_name
        h_count = _safe_int(sp.get("h_count"), None)
        if h_count is None and proton_row is not None and idx < len(matrix[proton_row]):
            h_count = int(matrix[proton_row][idx])
        h_count = int(h_count or 0)
        if not name:
            name = _default_species_name(parent, h_count)
        charge = _safe_int(sp.get("charge"), None)
        if charge is None:
            base_charge = int(_safe_int(comp_lookup.get(parent, {}).get("charge"), 0) or 0)
            if proton_row is not None:
                charge = base_charge + h_count
            else:
                charge = base_charge
        log_beta = _safe_float(sp.get("log_beta"), None)
        if log_beta is None:
            if h_count == 0:
                log_beta = 0.0
            else:
                default_log_beta = list(raw.get("log_beta") or [])
                if h_count - 1 < len(default_log_beta):
                    log_beta = float(default_log_beta[h_count - 1])
        species.append(
            {
                "name": name,
                "charge": int(charge),
                "h_count": h_count,
                "include": bool(sp.get("include", True)),
                "observable": bool(sp.get("observable", not bool(sp.get("non_observable", False)))),
                "fixed": bool(sp.get("fixed", h_count == 0)),
                "non_observable": bool(sp.get("non_observable", False)) or not bool(sp.get("observable", True)),
                "parent_component": parent,
                "log_beta": 0.0 if h_count == 0 else (None if log_beta is None else float(log_beta)),
            }
        )

    if not species:
        template = acid_base_model_from_simple_config(
            component_name=fallback_component_name,
            pka=fallback_pka,
            analytical_concentration=fallback_concentration,
            base_charge=fallback_base_charge,
        )
        template["components"] = components
        return canonicalize_acid_base_model(template)

    species.sort(key=lambda row: (str(row.get("parent_component") or ""), int(row.get("h_count") or 0), str(row.get("name") or "")))
    species_names = [str(sp["name"]) for sp in species]
    if not any(any(value != 0 for value in row) for row in matrix):
        matrix = _build_matrix(component_names, species)

    blocks = acid_base_constant_blocks(
        {
            "components": components,
            "species": species,
        }
    )
    primary_block = next((block for block in blocks if block["component_name"] == primary_name), blocks[0] if blocks else None)
    pka = list(raw.get("pka") or [])
    log_beta = list(raw.get("log_beta") or [])
    if primary_block is not None:
        if not log_beta:
            log_beta = [float(v) for v in primary_block["log_beta"]]
        if not pka:
            pka = [float(v) for v in primary_block["pka"]]

    mode = normalize_constant_mode(raw.get("constant_mode"))
    if mode == "pKa" and not pka and log_beta:
        pka = log_beta_to_pka([float(v) for v in log_beta])
    if mode == "log_beta" and not log_beta and pka:
        log_beta = pka_to_log_beta([float(v) for v in pka])

    return {
        "template_id": str(raw.get("template_id") or raw.get("model_type") or "custom_acid_base_system"),
        "definition_mode": str(raw.get("definition_mode") or "matrix"),
        "constant_mode": mode,
        "equations_text": str(raw.get("equations_text") or ""),
        "components": components,
        "species": species,
        "stoichiometric_matrix": matrix,
        "component_names": component_names,
        "species_names": species_names,
        "pka": [float(v) for v in pka],
        "log_beta": [float(v) for v in log_beta],
    }


def acid_base_constant_blocks(model_def: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = dict(model_def or {})
    components = [dict(comp) for comp in list(model.get("components") or [])]
    species = [dict(sp) for sp in list(model.get("species") or []) if bool(dict(sp).get("include", True))]
    blocks: list[dict[str, Any]] = []
    for comp in components:
        name = str(comp.get("name") or "").strip()
        if not name or bool(comp.get("is_proton")) or is_proton_component_name(name):
            continue
        rows = sorted(
            [sp for sp in species if str(sp.get("parent_component") or "") == name],
            key=lambda item: int(item.get("h_count") or 0),
        )
        if not rows:
            continue
        log_beta = [
            float(sp.get("log_beta") or 0.0)
            for sp in rows
            if int(sp.get("h_count") or 0) > 0
        ]
        blocks.append(
            {
                "component_name": name,
                "species": rows,
                "log_beta": log_beta,
                "pka": log_beta_to_pka(log_beta) if log_beta else [],
            }
        )
    return blocks


def parameter_rows_from_model(model_def: Mapping[str, Any]) -> list[dict[str, Any]]:
    model = canonicalize_acid_base_model(model_def)
    constant_mode = normalize_constant_mode(model.get("constant_mode"))
    blocks = acid_base_constant_blocks(model)
    component_count = len(blocks)
    rows: list[dict[str, Any]] = []
    for block in blocks:
        prefix = "" if component_count <= 1 else f"{block['component_name']}_"
        values = block["pka"] if constant_mode == "pKa" else block["log_beta"]
        kind = "pKa" if constant_mode == "pKa" else "log_beta"
        lower = -5.0 if kind == "pKa" else -50.0
        upper = 25.0 if kind == "pKa" else 50.0
        protonated_species = [sp for sp in block["species"] if int(sp.get("h_count") or 0) > 0]
        for idx, value in enumerate(values, start=1):
            linked_species = ""
            if idx - 1 < len(protonated_species):
                linked_species = str(protonated_species[idx - 1].get("name") or "")
            rows.append(
                {
                    "parameter": f"{prefix}{kind}{idx}",
                    "type": kind,
                    "initial_value": float(value),
                    "min": float(lower),
                    "max": float(upper),
                    "fixed": False,
                    "linked_species": linked_species,
                    "description": f"{kind} for component {block['component_name']}",
                }
            )
    return rows


def validate_acid_base_model(
    model_def: Mapping[str, Any],
    *,
    analysis_kind: str = "",
    experimental_pH: Sequence[float] | None = None,
    require_charges: bool = False,
) -> tuple[list[str], list[str]]:
    model = canonicalize_acid_base_model(model_def)
    errors: list[str] = []
    warnings: list[str] = []
    species = [dict(sp) for sp in list(model.get("species") or []) if bool(dict(sp).get("include", True))]
    if not species:
        errors.append("Define at least one included species.")
        return errors, warnings

    for sp in species:
        name = str(sp.get("name") or "").strip()
        if not name:
            errors.append("Each included species must have a name.")
        if sp.get("charge") in (None, "") and require_charges:
            errors.append(f"Species {name or '<unnamed>'} needs a charge.")
        if int(sp.get("h_count") or 0) > 0 and sp.get("h_count") in (None, ""):
            errors.append(f"Species {name or '<unnamed>'} needs h_count.")
        if int(sp.get("h_count") or 0) == 0 and float(sp.get("log_beta") or 0.0) not in {0.0}:
            warnings.append(f"Species {name} has h_count=0 but non-zero log_beta.")

    for block in acid_base_constant_blocks(model):
        counts = [int(sp.get("h_count") or 0) for sp in block["species"]]
        if counts != list(range(max(counts) + 1)):
            warnings.append(
                f"Component {block['component_name']} uses non-consecutive h_count values {counts}."
            )
        if len(block["log_beta"]) != max(counts):
            errors.append(
                f"Component {block['component_name']} has missing constants for one or more protonation steps."
            )

    if analysis_kind.strip().lower() == "potentiometry":
        for sp in species:
            if sp.get("charge") in (None, ""):
                errors.append("Potentiometry requires charge information for every included species.")
                break

    pH_values = [float(v) for v in list(experimental_pH or []) if _safe_float(v) is not None]
    if pH_values:
        ph_min = min(pH_values)
        ph_max = max(pH_values)
        spacing = 0.0
        unique = sorted(set(round(value, 8) for value in pH_values))
        if len(unique) >= 2:
            spacing = min(abs(unique[idx + 1] - unique[idx]) for idx in range(len(unique) - 1))
        for block in acid_base_constant_blocks(model):
            for idx, value in enumerate(block["pka"], start=1):
                if value < ph_min - 1.0 or value > ph_max + 1.0:
                    warnings.append(
                        f"{block['component_name']} pKa{idx}={value:.3f} lies well outside the experimental pH range."
                    )
            if spacing > 0.0:
                for left, right in zip(block["pka"], block["pka"][1:]):
                    if abs(right - left) < max(2.0 * spacing, 0.25):
                        warnings.append(
                            f"{block['component_name']} has closely spaced pKa values ({left:.3f}, {right:.3f}) relative to pH sampling."
                        )
    return errors, warnings


def acid_base_system_from_model(
    model_def: Mapping[str, Any],
    *,
    temperature: float = 298.15,
    ionic_strength: float | None = None,
    kw: float = 1.0e-14,
) -> AcidBaseSystem:
    model = canonicalize_acid_base_model(model_def)
    comp_lookup = _component_lookup(model)
    components: list[AcidBaseComponent] = []
    for block in acid_base_constant_blocks(model):
        comp_meta = dict(comp_lookup.get(block["component_name"]) or {})
        base_charge = int(_safe_int(comp_meta.get("charge"), 0) or 0)
        concentration = _safe_float(comp_meta.get("analytical_concentration"), 0.0) or 0.0
        species: list[AcidBaseSpecies] = []
        for row in block["species"]:
            h_count = int(row.get("h_count") or 0)
            charge = int(_safe_int(row.get("charge"), base_charge + h_count) or 0)
            species.append(
                AcidBaseSpecies(
                    name=str(row.get("name") or _default_species_name(block["component_name"], h_count)),
                    charge=charge,
                    h_count=h_count,
                    log_beta=None if h_count == 0 else float(row.get("log_beta") or 0.0),
                    fixed=bool(row.get("fixed", h_count == 0)),
                )
            )
        if not species:
            raise ValueError(f"Component {block['component_name']!r} has no included species.")
        components.append(
            AcidBaseComponent(
                name=block["component_name"],
                analytical_concentration=float(concentration),
                species=species,
            )
        )
    if not components:
        raise ValueError("No non-proton acid-base components were defined.")
    return AcidBaseSystem(
        components=components,
        temperature=float(temperature),
        ionic_strength=None if ionic_strength in (None, "") else float(ionic_strength),
        kw=float(kw),
    )


def serializable_model_definition_from_system(
    cfg_model: Mapping[str, Any] | None,
    *,
    system: AcidBaseSystem,
) -> dict[str, Any]:
    if isinstance(cfg_model, Mapping) and cfg_model:
        return canonicalize_acid_base_model(cfg_model)
    first = system.components[0]
    return acid_base_model_from_simple_config(
        component_name=str(first.name),
        pka=log_beta_to_pka([float(sp.log_beta) for sp in sorted(first.species, key=lambda sp: int(sp.h_count)) if int(sp.h_count) > 0 and sp.log_beta is not None]),
        analytical_concentration=float(first.analytical_concentration),
        base_charge=int(sorted(first.species, key=lambda sp: int(sp.h_count))[0].charge),
    )


__all__ = [
    "acid_base_constant_blocks",
    "acid_base_model_from_equations",
    "acid_base_model_from_simple_config",
    "acid_base_system_from_model",
    "build_acid_base_template",
    "canonicalize_acid_base_model",
    "is_proton_component_name",
    "normalize_constant_mode",
    "parameter_rows_from_model",
    "proton_component_name",
    "serializable_model_definition_from_system",
    "validate_acid_base_model",
]
