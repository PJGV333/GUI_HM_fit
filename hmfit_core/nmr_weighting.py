"""Versioned NMR weighting configuration and legacy compatibility."""
from typing import Any, Mapping

WEIGHTS_SCHEMA_VERSION = 1


def resolve_nmr_weights_mode(config: Mapping[str, Any]) -> str:
    """Return the explicitly selected mode, or IRLS for unversioned legacy input."""
    has_version = "weights_schema_version" in config
    if has_version:
        version = config["weights_schema_version"]
        if type(version) is not int or version != WEIGHTS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported NMR weights schema version: {version!r}.")
    if "weights_mode" not in config:
        if has_version:
            raise ValueError("Versioned NMR weights configuration requires weights_mode.")
        return "irls"
    mode = config["weights_mode"]
    if not isinstance(mode, str) or mode not in {"unit", "irls"}:
        raise ValueError(f"Unsupported NMR weights mode: {mode!r}; choose unit or irls explicitly.")
    return mode
