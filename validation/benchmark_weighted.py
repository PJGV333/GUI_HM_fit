"""Bounded benchmark of the weighted semismooth Newton solver.

Run each case in a fresh process so ru_maxrss is a process-local peak:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m validation.benchmark_weighted --species 3 --wavelengths 100
The synthetic problem is a numerical stress test, not a coverage experiment.
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np

from hmfit_core.utils.nnls_utils import WeightedLinearSolveError, _solve_weighted_penalty_newton


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--species", type=int, required=True)
    parser.add_argument("--wavelengths", type=int, required=True)
    parser.add_argument("--points", type=int, default=12)
    parser.add_argument("--seed", type=int, default=701)
    args = parser.parse_args()
    s, nw, m = args.species, args.wavelengths, args.points
    size = s * nw
    result = {"species": s, "wavelengths": nw, "points": m, "seed": args.seed,
              "hessian_elements": size * size, "hessian_bytes_float64": 8 * size * size,
              "alpha": 1e-6, "mu": 1e-2, "status": "not_started"}
    if size > 4000:
        result["status"] = "skipped_size_limit"
        print(json.dumps(result, sort_keys=True))
        return
    rng = np.random.default_rng(args.seed)
    C = rng.uniform(0.0001, 0.001, size=(m, s))
    true_A = rng.uniform(100.0, 1000.0, size=(s, nw))
    Y = C @ true_A + rng.normal(0, 0.0002, size=(m, nw))
    W = rng.uniform(0.5, 2.0, size=(m, nw))
    D = np.diff(np.eye(nw), axis=0)
    L = D.T @ D
    start = time.perf_counter()
    try:
        A = np.asarray(_solve_weighted_penalty_newton(C, Y, W, 0.0, result["mu"], None,
                                                      result["alpha"], L, 300), dtype=float)
        residual = C @ A - Y
        grad = C.T @ ((W * W) * residual) + result["mu"] * np.minimum(A, 0.0) + result["alpha"] * A @ L
        result.update(status="converged", gradient_inf=float(np.max(np.abs(grad))),
                      objective=float(0.5 * (np.sum((W * residual) ** 2)
                                            + result["mu"] * np.sum(np.minimum(A, 0.0) ** 2)
                                            + result["alpha"] * np.sum((A @ L) * A))))
    except WeightedLinearSolveError as exc:
        result.update(status="solver_failure", error=str(exc))
    result["seconds"] = time.perf_counter() - start
    result["max_rss_kib_linux"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result["iterations"] = None  # current solver does not expose its iteration count
    result["reference_gap"] = None  # independent reference is covered only by small regression tests
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
