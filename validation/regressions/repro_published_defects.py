"""Reproduce the Phase 2 defects on a source tree (released tag, main, or this branch).

    git archive v0.2.0 | tar -x -C /tmp/v020
    QT_QPA_PLATFORM=offscreen python3.11 validation/regressions/repro_published_defects.py /tmp/v020

Prints one line per defect: REPRODUCED / not reproduced / not applicable (code absent).
Used for the "Defectos en versiones publicadas" section of docs/STABILITY_PHASE2_REPORT.md.
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback

root = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, root)
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

results: list[tuple[str, str, str]] = []


def check(name):
    def deco(fn):
        try:
            status, detail = fn()
        except ModuleNotFoundError as exc:
            status, detail = "not applicable", f"code absent ({exc.name})"
        except Exception as exc:  # unexpected: show it
            status, detail = "ERROR", f"{type(exc).__name__}: {exc}"[:200]
            traceback.print_exc()
        results.append((name, status, detail))
        return fn
    return deco


@check("D1 kinetics graph edge on Python < 3.14")
def _d1():
    from hmfit.kinetics.model.kinetics_graph import KineticGraph

    try:
        KineticGraph().add_kinetic_reaction("A -> B", k_forward=1.0)
    except TypeError as exc:
        return "REPRODUCED", f"TypeError: {exc}"
    return "not reproduced", f"Python {sys.version.split()[0]}"


@check("D2 non-numeric value/bound in the spectroscopy/NMR parameter table")
def _d2():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QTableWidgetItem

    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    QApplication.instance() or QApplication([])
    w = ModelOptPlotsWidget()
    w.params_table.blockSignals(True)
    w.params_table.setRowCount(1)
    for c, text in enumerate(("K1", "3,5", "0", "8x")):  # decimal comma and a typo
        w.params_table.setItem(0, c, QTableWidgetItem(text))
    item = QTableWidgetItem()
    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
    item.setCheckState(Qt.CheckState.Unchecked)
    w.params_table.setItem(0, 4, item)
    if hasattr(w, "validate_parameter_table"):
        try:
            w.validate_parameter_table()
        except ValueError as exc:
            return "not reproduced", "rejected: " + str(exc).splitlines()[1][:90]
    k, bounds, _ = w.get_optimization()
    return "REPRODUCED", f"initial value '3,5' -> {k[0]}, upper bound '8x' -> {bounds[0][1]} (silently unbounded)"


@check("D3 potentiometry with a fixed pKa")
def _d3():
    from hmfit_core.acid_base import make_simple_acid_base_system
    from hmfit_core.acid_base_errors import compute_errors_acid_base_from_context
    from hmfit_core.potentiometry import PotentiometryExperiment, simulate_pH_titration
    from hmfit_core.run_acid_base import run_acid_base

    system = make_simple_acid_base_system(analytical_concentration=1e-2, pka=[8.0, 4.0], base_charge=-2)
    vol = np.linspace(0.0, 25.0, 40)
    exp = PotentiometryExperiment(initial_volume=10.0, titrant_volumes=vol, analyte_concentration=1e-2,
                                  titrant_concentration=1e-2, titrant_type="base")
    pH = simulate_pH_titration(exp, system) + np.random.default_rng(1).normal(0.0, 0.005, vol.size)
    path = os.path.join(tempfile.mkdtemp(), "pot.csv")
    pd.DataFrame({"volume_mL": vol, "pH": pH}).to_csv(path, index=False)
    res = run_acid_base({"file_path": path, "data_type": "potentiometry", "pka_initial": "8.3, 4.0",
                         "pka_fit_mask": [True, False], "analyte_concentration": 1e-2,
                         "titrant_concentration": 1e-2, "initial_volume": 10.0, "titrant_type": "base",
                         "base_charge": -2}, progress_cb=lambda _m: None)
    try:
        compute_errors_acid_base_from_context(res["errors_context"], {"method": "analytic"})
    except IndexError as exc:
        return "REPRODUCED", f"IndexError: {exc}; fit pKa1 = {res['export_data']['pka_table'][0]['value']:.4f} (fit itself fine)"
    return "not reproduced", "error table produced"


@check("D4 NMR error analysis with fixed + non-identifiable parameters (shape mismatch)")
def _d4():
    import subprocess

    here = os.path.dirname(os.path.abspath(__file__))
    matrix = os.path.join(here, "..", "..", "tests", "test_phase2_invariants_matrix.py")
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "test_matrix_copy.py")
    with open(matrix, encoding="utf-8") as fh, open(dst, "w", encoding="utf-8") as out:
        out.write(fh.read())
    ids = [f"{dst}::test_uncertainty_conventions[nmr-{m}-nonidentifiable]"
           for m in ("analytic", "bootstrap_linear", "bootstrap_onestep", "bootstrap_full_refit_audit")]
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider", "-o", "addopts=",
                           "--rootdir", root, *ids], cwd=root, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": root})
    shape = sum(1 for line in proc.stdout.splitlines() if "ValueError: shape mismatch" in line and not line.startswith("E "))
    return ("REPRODUCED" if shape else "not reproduced"), f"{shape} of 4 NMR methods raise 'shape mismatch'"


for name, status, detail in results:
    print(f"{status:16s} | {name} | {detail}")
print(f"(tree: {root}; Python {sys.version.split()[0]})")
