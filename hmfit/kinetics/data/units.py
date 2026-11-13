"""Metadata-only unit checks: the kinetics integrator does not convert datasets."""
from __future__ import annotations

_TIME = {
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
}
_CONCENTRATION = {
    "M": "M", "mol/L": "M", "mol L-1": "M",
    "mM": "mM", "mmol/L": "mM", "mmol L-1": "mM",
    "uM": "µM", "µM": "µM", "μM": "µM", "umol/L": "µM", "µmol/L": "µM",
    "nM": "nM", "nmol/L": "nM",
}


def canonical_time_unit(value: str | None) -> str | None:
    return _TIME.get(str(value or "").strip().lower())


def canonical_concentration_unit(value: str | None) -> str | None:
    return _CONCENTRATION.get(str(value or "").strip())


def eyring_unit_factors(dataset, order: int) -> tuple[float, list[str]]:
    """Convert an Eyring rate in M and seconds to the dataset's numeric units."""
    raw_t = str(getattr(dataset, "time_unit", "") or "").strip()
    raw_c = str(getattr(dataset, "concentration_unit", "") or "").strip()
    t = canonical_time_unit(raw_t)
    c = canonical_concentration_unit(raw_c)
    if raw_t and t is None:
        raise ValueError(f"Unrecognized time unit for Eyring: {raw_t!r}.")
    if raw_c and c is None:
        raise ValueError(f"Unrecognized concentration unit for Eyring: {raw_c!r}.")
    warnings = []
    if not raw_t:
        warnings.append("Eyring: empty time unit; assuming s.")
    if not raw_c:
        warnings.append("Eyring: empty concentration unit; assuming M. If the data are in mM, "
                        "bimolecular ΔS‡ is shifted by about 57 J·mol⁻¹·K⁻¹; set the correct unit.")
    seconds = {"s": 1., "ms": .001, "min": 60., "h": 3600.}[t or "s"]
    molar = {"M": 1., "mM": 1e-3, "µM": 1e-6, "nM": 1e-9}[c or "M"]
    return seconds * molar ** (order - 1), warnings


def validate_dataset_units(datasets) -> list[str]:
    """Reject incompatible known units and warn on unknowns in global fits."""
    if len(datasets) < 2:
        return []
    warnings = []
    for label, attr, canonical in (
        ("time", "time_unit", canonical_time_unit),
        ("concentration", "concentration_unit", canonical_concentration_unit),
    ):
        units = [canonical(getattr(ds, attr, "")) for ds in datasets]
        known = set(u for u in units if u is not None)
        if len(known) > 1:
            raise ValueError(f"Global fit has incompatible {label} units: {', '.join(sorted(known))}. No automatic conversion is performed.")
        if any(u is None for u in units):
            warnings.append(f"Some datasets have empty or unrecognized {label} units; compatibility cannot be verified.")
    return warnings
