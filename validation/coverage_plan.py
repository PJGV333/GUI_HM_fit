"""Which pre-registered cells feed which Phase 2 decision (pre-registration amendment 2).

    python -m validation.coverage_plan table                 # tier x decision map (markdown)
    python -m validation.coverage_plan write R1 OUT.json     # sub-grid to run now
    python -m validation.coverage_plan write R2 OUT.json     # after the one-step optimisation
    python -m validation.coverage_plan write FR-<tech> OUT.json  # conditional full-refit passes

Cells are taken verbatim from validation/configs/coverage/primary_grid.json (no cell is
changed); this module only selects them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

GRID = Path("validation/configs/coverage/primary_grid.json")
REALISTIC = {"iid", "hetero", "iid1", "iid5"}


def cells():
    return json.loads(GRID.read_text(encoding="utf-8"))["cells"]


def is_fr(c):
    return c["id"].endswith("~fr")


def purpose(c) -> str:
    """Decision a cell feeds: '2.3', '2.4', '2.3+2.4' or 'deferred' (reported, no decision)."""
    if c["tier"] == 9:
        return "deferred"  # diagnostic, no coverage expectation
    if c["noise"] not in REALISTIC:
        return "deferred"  # correlated noise: reported separately, not in any decision rule
    est = c.get("estimator", "soft_penalty")
    if c["technique"] == "uvvis" and c["noise"] == "iid" and c["n_points"] in (8, 12, 15) and est != "soft_penalty":
        return "2.3"
    if c["technique"] == "uvvis" and c["noise"] == "iid" and c["n_points"] in (8, 12, 15):
        return "2.3+2.4"
    return "2.4"


def stage(c) -> str:
    p = purpose(c)
    if p == "deferred":
        return "deferred"
    if is_fr(c) or (c["technique"] in ("pot", "kin") and set(c["methods"]) <= {"fr_row", "fr_el"}):
        tech = c["technique"]
        return f"FR-{'2.3' if p == '2.3' else tech}"
    if c["tier"] == 4:
        return "R2"
    return "R1"


def table() -> str:
    rows = {}
    for c in cells():
        key = (c["tier"], purpose(c), stage(c))
        rows[key] = rows.get(key, 0) + 1
    lines = ["| tier | decisión | etapa de ejecución | celdas |", "|---:|---|---|---:|"]
    for (t, p, s), n in sorted(rows.items()):
        lines.append(f"| {t} | {p} | {s} | {n} |")
    return "\n".join(lines)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[0] == "table":
        print(table())
    elif argv[0] == "write":
        sel = [c for c in cells() if stage(c) == argv[1]]
        Path(argv[2]).write_text(json.dumps({"cells": sel, "selection": argv[1]}, indent=1), encoding="utf-8")
        print(f"{len(sel)} cells ({argv[1]}) -> {argv[2]}")


if __name__ == "__main__":
    main()
