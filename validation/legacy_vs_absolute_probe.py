"""Probe: legacy vs absolute weight-regularization conventions (Stage H audit).

For one pilot dataset and three weightings, fits log K under each convention
via the production profiled objective and reports the fitted value plus both
objective values on a common grid — quantifying the documented convention
difference without touching production code.
"""
import json
import numpy as np

from hmfit_core.processors import spectroscopy_processor as sp
from hmfit_core.solvers import NewtonRaphson
from validation.bias_experiment import argmin_1d, dataset

scen = {"points": 8, "noise": 2e-4, "truth": 3.0, "g_max": 2e-3}
d = dataset(777, scen)
c, Yh = d["c"], d["y"]              # (m, nw)
Y = Yh                              # _solve_spectral_model consumes (m, nw)
M = d["M"]
m, nw = Yh.shape

def make_obj(mode, weights_row):
    solver = NewtonRaphson(c, M, nas=[], model_sett="Free")

    def f(th):
        C = solver.concentraciones(np.asarray([float(th)]))[0]
        fit = sp._solve_spectral_model(
            C, Y, eps_solver_mode="soft_penalty", mu=1e-2, alpha_smooth=0.0,
            weights_row=weights_row, weight_regularization_mode=mode)
        return sp._spectral_profile_objective(fit, Y, weights_row, 1e-2, 0.0, None, mode)
    return f

weights_cases = {
    "unit": np.ones((m, nw)),
    "wl_ramp_0.5_2.0": np.tile(np.linspace(0.5, 2.0, nw), (m, 1)),
    "point_ramp_0.5_2.0": np.tile(np.linspace(0.5, 2.0, m)[:, None], (1, nw)),
}
report = []
for wname, w in weights_cases.items():
    row = {"weights": wname}
    ks, fs = {}, {}
    for mode in ("legacy", "absolute"):
        fs[mode] = make_obj(mode, w)
        ks[mode] = argmin_1d(fs[mode], 2.4, 3.6)[0]
    for mode in ("legacy", "absolute"):
        other = "absolute" if mode == "legacy" else "legacy"
        row[mode] = {"k": ks[mode], "obj_at_own_k": fs[mode](ks[mode]),
                     "obj_at_other_k": fs[mode](ks[other])}
    row["dk_legacy_minus_absolute"] = ks["legacy"] - ks["absolute"]
    report.append(row)
print(json.dumps(report, indent=1))
