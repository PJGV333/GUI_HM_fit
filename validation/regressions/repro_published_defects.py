"""Which confirmed defects ([Defecto comprobado] in docs/RELIABILITY_REVIEW.md,
docs/STABILITY_PHASE1_REPORT.md and docs/STABILITY_PHASE2_REPORT.md) exist in a source tree.

    git archive v0.2.0 | tar -x -C /tmp/v020
    QT_QPA_PLATFORM=offscreen python3.11 validation/regressions/repro_published_defects.py /tmp/v020 [--json OUT]

Two kinds of checks, both run against the *target tree's* code:

* **probes** (functions below): direct reproductions through the GUI widgets or the core API;
* **regression tests**: the test that pins each fix, copied from this branch into a temporary
  directory. Top-level imports are wrapped in ``try/except`` so that a module missing in an
  old tree only affects the tests that use it.

Outcome per defect: ``REPRODUCED`` (the test/probe fails on its assertion, or the documented
error is raised), ``absent`` (passes), ``fix API absent`` (the test cannot call the function it
needs: the fix is not there, but the behaviour was *not* verified), ``not applicable`` (the
module itself does not exist in that tree). The "effect" column says whether the defect changes
reported numbers silently or only makes something fail loudly.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
BRANCH_TESTS = os.path.normpath(os.path.join(HERE, "..", "..", "tests"))

# id, source, module, description, effect, regression tests (file::name)
REGISTRY = [
    ("RR1", "RELIABILITY_REVIEW #1", "Kinetics/spectro/NMR bootstrap one-step",
     "LM step damped with unscaled lambda*I (SD ~7000x too small for k in M^-1 s^-1)", "numbers",
     ["test_uncertainty_reliability.py::test_lm_damped_step_is_scale_invariant"]),
    ("RR2", "RELIABILITY_REVIEW #2", "UV-Vis absorptivity solver",
     "soft_penalty (300 gradient iterations) does not converge at molar scale: biased log K", "numbers",
     ["test_uncertainty_reliability.py::test_soft_penalty_absorptivity_solver_is_exact_at_molar_scale"]),
    ("RR3", "RELIABILITY_REVIEW #3", "dof (kinetics, acid-base)",
     "degrees of freedom do not subtract the linear parameters removed by variable projection", "numbers",
     ["test_uncertainty_reliability.py::test_kinetics_dof_counts_linear_spectra",
      "test_uncertainty_reliability.py::test_acid_base_nmr_dof_subtracts_limiting_shifts"]),
    ("RR4", "RELIABILITY_REVIEW #4", "Analytic CI (all)", "z = 1.96 instead of Student t with the residual dof",
     "numbers", ["test_uncertainty_reliability.py::test_t_quantile_and_leverage"]),
    ("RR5", "RELIABILITY_REVIEW #5", "Kinetics fit", "fixing any parameter raises ValueError (bounds)", "crash",
     ["test_uncertainty_reliability.py::test_kinetics_fixed_parameter_does_not_crash_and_has_zero_se"]),
    ("RR7", "RELIABILITY_REVIEW #7", "Spectro/NMR bootstrap, 1 parameter (1:1)",
     "np.corrcoef returns a scalar: every bootstrap of a 1:1 model fails", "crash",
     ["test_uncertainty_reliability.py::test_bootstrap_correlation_for_single_parameter"]),
    ("RR11", "RELIABILITY_REVIEW #11", "Chemical graphs", "consistent thermodynamic cycles rejected", "crash",
     ["test_multicomponent_system.py::test_thermodynamic_cycle_consistency"]),
    ("P1-1", "Fase 1 #1", "Kinetics bounds", "inverted bounds silently fix the parameter", "numbers",
     ["test_phase1_invariants.py::test_inverted_bounds_raise_instead_of_fixing"]),
    ("P1-2", "Fase 1 #2", "Kinetics fixed by mask", "mask-fixed parameter moved to its lower bound", "numbers",
     ["test_phase1_invariants.py::test_mask_fixed_kinetic_parameter_keeps_its_exact_value"]),
    ("P1-4", "Fase 1 #4", "Kinetics/spectro identifiability",
     "zero Jacobian column read as 'fixed' -> SE = 0", "numbers",
     ["test_phase1_invariants.py::test_unidentifiable_kinetic_parameter_is_nan_not_zero"]),
    ("P1-5", "Fase 1 #5", "Linearized/one-step bootstrap",
     "collinear or insensitive parameters get a finite SD or 0", "numbers",
     ["test_phase1_invariants.py::test_linearized_bootstrap_never_moves_absent_or_null_directions"]),
    ("P1-6", "Fase 1 #6", "Spectroscopy errors with fixed log beta",
     "analytic/one-step ignore fixed_mask: fixed SE != 0 and inflated free SE", "numbers",
     ["test_phase1_invariants.py::test_spectro_fixed_mask_analytic_and_one_step"]),
    ("P1-7", "Fase 1 #7", "Acid-base NMR/spectroscopy", "pka_fit_mask and pka_bounds ignored (fixed pKa refitted)",
     "numbers", ["test_phase1_invariants.py::test_acid_base_fixed_pka_is_respected_in_nmr_and_spectroscopy"]),
    ("P1-8", "Fase 1 #8", "Acid-base context", "`x or []` on NumPy arrays raises; linear count silently 0",
     "crash", ["test_phase1_invariants.py::test_acid_base_context_accepts_numpy_arrays"]),
    ("P1-10", "Fase 1 #10", "dof <= 0 (all)", "dof <= 0 truncated to 1 without warning", "numbers",
     ["test_phase1_invariants.py::test_acid_base_dof_nonpositive_reports_nan",
      "test_phase1_invariants.py::test_linearized_covariance_reference_and_degenerate_cases"]),
    ("P1-x", "Fase 1 §2.1", "Kinetics fixed/free invariants", "all-fixed / mixed fixed parameters", "numbers",
     ["test_phase1_invariants.py::test_all_fixed_none_fixed_and_mixed",
      "test_phase1_invariants.py::test_initial_value_outside_bounds_is_rejected"]),
    ("P2-refit", "Fase 2 (Anexo A §4)", "Spectroscopy full-refit",
     "optimizer non-convergence / solver failure accepted as a valid replicate", "numbers",
     ["test_phase2_completion_critical.py::test_refit_rejects_optimizer_nonconvergence",
      "test_phase2_completion_critical.py::test_full_refit_counts_failures_and_rejects_nonfinite_success"]),
    ("P2-pgd", "Fase 2 (Anexo A §4)", "solve_A_nnls_pgd", "constant per-wavelength multipliers ignored", "numbers",
     ["test_phase2_completion_critical.py::test_wavelength_constant_multiplier_applies_to_penalized_objective"]),
    ("P2-nmr12", "Fase 2 (Anexo A §7)", "NMR 1:2", "optimizer failure flag ignored: echo reported as success",
     "numbers", ["test_nmr_12_gap.py::test_nmr_unknown_optimizer_fails_honestly"]),
    ("P2-pot", "Fase 2 #2 (D3)", "Potentiometry", "fixed pKa crashes the error analysis", "crash",
     ["test_potentiometry_fixed_pka.py::test_fixed_pka_reaches_export_sheets"]),
    ("P2-conv", "Fase 2 #3 (D5)", "All error engines",
     "uncertainty conventions (fixed corr 0/1, finite SE with dof <= 0, ...): see the 105-case matrix", "numbers",
     []),
]


# ----------------------------------------------------------------------------- probes
def probe_d1():
    from hmfit.kinetics.model.kinetics_graph import KineticGraph

    try:
        KineticGraph().add_kinetic_reaction("A -> B", k_forward=1.0)
    except TypeError as exc:
        return "REPRODUCED", f"TypeError: {exc}"
    return "absent", f"Python {sys.version.split()[0]}"


def _binding_widget(rows):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QTableWidgetItem

    from hmfit_gui_qt.widgets.model_opt_plots import ModelOptPlotsWidget

    QApplication.instance() or QApplication([])
    w = ModelOptPlotsWidget()
    w.params_table.blockSignals(True)
    w.params_table.setRowCount(len(rows))
    for r, texts in enumerate(rows):
        for c, text in enumerate(texts):
            w.params_table.setItem(r, c, QTableWidgetItem(text))
        item = QTableWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(Qt.CheckState.Unchecked)
        w.params_table.setItem(r, 4, item)
    return w


def _binding_case(texts):
    w = _binding_widget([texts])
    if hasattr(w, "validate_parameter_table"):
        try:
            w.validate_parameter_table()
        except ValueError as exc:
            return "absent", "rejected: " + str(exc).splitlines()[1][:80]
    k, bounds, _ = w.get_optimization()
    return k[0], bounds[0]


def probe_d2_typo():
    out = _binding_case(("K1", "3x", "0", "8x"))
    if out[0] == "absent":
        return out
    return "REPRODUCED", f"value '3x' -> {out[0]}, upper bound '8x' -> {out[1][1]}"


def probe_d2_comma():
    out = _binding_case(("K1", "0,5", "0,1", "8"))
    if out[0] == "absent":
        return out
    return "REPRODUCED", f"value '0,5' -> {out[0]}, lower bound '0,1' -> {out[1][0]}"


def probe_d2_kinetics_comma():
    from PySide6.QtWidgets import QApplication, QTableWidgetItem

    from hmfit_gui_qt.widgets.kinetics_model_opt_plots import KineticsModelOptPlotsWidget

    QApplication.instance() or QApplication([])
    w = KineticsModelOptPlotsWidget()
    w.set_params(["k1"])
    for col, text in zip((1, 2, 3), ("0,5", "0,1", "10")):
        w.params_table.setItem(0, col, QTableWidgetItem(text))
    if hasattr(w, "validate_parameter_table"):
        issues = w.validate_parameter_table()
        if issues:
            return "absent", "rejected: " + issues[0][:80]
    st = w.get_params_state()
    return "REPRODUCED", f"value '0,5' -> {st.values[0]}, lower bound '0,1' -> {st.bounds[0][0]}"


def probe_d3():
    import numpy as np
    import pandas as pd

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
        return "REPRODUCED", f"IndexError: {exc}; fitted pKa1 = {res['export_data']['pka_table'][0]['value']:.4f}"
    return "absent", "error table produced"


def probe_d4(root):
    """NMR with a constant that does not affect the data (invariants matrix cases)."""
    tmp = tempfile.mkdtemp()
    dst = os.path.join(tmp, "test_matrix_copy.py")
    shutil.copy(os.path.join(BRANCH_TESTS, "test_phase2_invariants_matrix.py"), dst)
    ids = [f"{dst}::test_uncertainty_conventions[nmr-{m}-nonidentifiable]"
           for m in ("analytic", "bootstrap_linear", "bootstrap_onestep", "bootstrap_full_refit_audit")]
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--tb=line", "-p", "no:cacheprovider", "-o",
                           "addopts=", "--rootdir", root, *ids], cwd=root, capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": root})
    lines = proc.stdout.splitlines()
    crash = sum(1 for ln in lines if "ValueError: shape mismatch" in ln and not ln.startswith("E "))
    se0 = sum(1 for ln in lines if "SE=0.0 (must be NaN)" in ln and not ln.startswith("E "))
    if crash or se0:
        return "REPRODUCED", f"{crash} of 4 methods raise 'shape mismatch'; {se0} report SE = 0 (reads as 'fixed')"
    return "absent", "4 of 4 methods follow the convention"


PROBES = [
    ("P2-D1", "Fase 2 #1", "Kinetics graph API (Python < 3.14)", "super() in a slots dataclass", "crash",
     probe_d1),
    ("P2-D2a", "Fase 2 #5", "Spectro/NMR parameter table", "typo in value/bound replaced silently", "numbers",
     probe_d2_typo),
    ("P2-D2b", "Fase 2 #5", "Spectro/NMR parameter table", "decimal comma ('0,5') replaced silently", "numbers",
     probe_d2_comma),
    ("P2-D2c", "Fase 2 #5", "Kinetics parameter table", "decimal comma ('0,5') replaced silently", "numbers",
     probe_d2_kinetics_comma),
    ("P2-D3", "Fase 2 #2", "Potentiometry", "fixed pKa crashes the error analysis", "crash", probe_d3),
    ("P2-D4", "Fase 2 #4", "NMR errors", "non-identifiable constant: shape mismatch or SE = 0", "crash/numbers",
     "d4"),
]


# ----------------------------------------------------------------------------- tests
def _guard_imports(src: str) -> str:
    tree = ast.parse(src)
    body = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not (
            isinstance(node, ast.ImportFrom) and node.module == "__future__"
        ):
            body.append(ast.Try(body=[node], handlers=[ast.ExceptHandler(
                type=ast.Name("Exception", ast.Load()), name=None, body=[ast.Pass()])], orelse=[], finalbody=[]))
        else:
            body.append(node)
    tree.body = body
    return ast.unparse(ast.fix_missing_locations(tree))


API_RE = re.compile(r"NameError|ImportError|ModuleNotFoundError|AttributeError|unexpected keyword|"
                    r"got an unexpected|missing \d+ required|takes \d+ positional|has no attribute")


def run_tests(root: str) -> dict[str, tuple[str, str]]:
    nodes = sorted({t for *_, tests in REGISTRY for t in tests})
    files = sorted({n.split("::")[0] for n in nodes})
    tmp = tempfile.mkdtemp(prefix="repro_tests_")
    for f in files + ["phase1_cases.py"]:
        src = open(os.path.join(BRANCH_TESTS, f), encoding="utf-8").read()
        with open(os.path.join(tmp, f), "w", encoding="utf-8") as fh:
            fh.write(_guard_imports(src) if f.startswith("test_") else src)
    junit = os.path.join(tmp, "junit.xml")
    subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "--rootdir",
                    tmp, f"--junitxml={junit}", *[os.path.join(tmp, n) for n in nodes]],
                   cwd=root, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": root})
    out = {}
    for tc in ET.parse(junit).iter("testcase"):
        key = f"{tc.get('file') or tc.get('classname').split('.')[-1] + '.py'}::{tc.get('name')}"
        key = os.path.basename(key.split("::")[0]) + "::" + tc.get("name")
        f = tc.find("failure")
        f = f if f is not None else tc.find("error")
        if f is None:
            out[key] = ("absent", "test passes")
            continue
        msg = (f.get("message") or "").splitlines()[0][:160]
        out[key] = ("fix API absent", msg) if API_RE.search(msg) else ("REPRODUCED", msg)
    return out


def module_exists(root, rel):
    return os.path.exists(os.path.join(root, rel))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default=".")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    root = os.path.abspath(a.root)
    sys.path.insert(0, root)
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    rows = []
    for did, source, module, desc, effect, fn in PROBES:
        try:
            status, detail = probe_d4(root) if fn == "d4" else fn()
        except ModuleNotFoundError as exc:
            status, detail = "not applicable", f"module absent ({exc.name})"
        except Exception as exc:
            status, detail = "ERROR", f"{type(exc).__name__}: {exc}"[:200]
            traceback.print_exc()
        rows.append({"id": did, "source": source, "module": module, "defect": desc, "effect": effect,
                     "status": status, "detail": detail})
    tests = run_tests(root)
    has_ab = module_exists(root, "hmfit_core/acid_base_errors.py")
    for did, source, module, desc, effect, nodes in REGISTRY:
        if not nodes:
            continue
        res = []
        for n in nodes:
            hits = [v for k, v in tests.items() if k == n or k.startswith(n + "[")]
            if not hits:
                res.append(("not collected", ""))
            elif any(h[0] == "REPRODUCED" for h in hits):
                res.append(next(h for h in hits if h[0] == "REPRODUCED"))
            elif all(h[0] == "absent" for h in hits):
                res.append(hits[0])
            else:
                res.append(next(h for h in hits if h[0] != "absent"))
        statuses = [r[0] for r in res]
        if "REPRODUCED" in statuses:
            status = "REPRODUCED"
        elif all(s == "absent" for s in statuses):
            status = "absent"
        else:
            status = "fix API absent"
        if not has_ab and ("acid" in module.lower() or "potentiometry" in module.lower()):
            status = "not applicable"
        detail = " | ".join(f"{n.split('::')[1]}: {s} ({m[:90]})" for n, (s, m) in zip(nodes, res))
        rows.append({"id": did, "source": source, "module": module, "defect": desc, "effect": effect,
                     "status": status, "detail": detail})
    for r in rows:
        print(f"{r['status']:15s} | {r['id']:8s} | {r['effect']:13s} | {r['module']} | {r['detail'][:150]}")
    print(f"(tree: {root}; Python {sys.version.split()[0]})")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump({"tree": root, "python": sys.version.split()[0], "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
