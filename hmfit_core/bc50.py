# SPDX-License-Identifier: GPL-3.0-or-later
"""Median binding concentration BC50 and intrinsic BC50^0 (Vacca, Roelens).

Definitions (R09a = Chem. Eur. J. 2009, 15, 2635, text and SI appendix eqs A1-A30):

* A is the *analysed* reagent (the one whose affinity is assessed), B the *bound analyte*
  and C an optional ancillary reagent (three-reagent systems).
* x_A is the fraction of A bound in species that contain both A and B.
* BC50(x_A) is the total concentration of A at which 50 % of B is bound (eq A8:
  bound B = unbound B), at bound fraction x_A of A (eqs A10-A15).
* The third reagent is set either by a fixed free concentration ``[C]`` (default
  1 mol/L) or by electroneutrality, T_C = T_K, with K the counterion of C (A or B).
* BC50^0 = lim_{x_A -> 0} BC50 (eqs A16', A20): with [B] -> 0 only the species
  mononuclear in B contribute.

Known limitation of the descriptor (R09a, note 18): species *multinuclear in B* (with
A present, e.g. A B2 C2) enter BC50 with an x_A-dependent weight and are ignored by
BC50^0; when no species mononuclear in B contains A, BC50^0 is indefinite. This module
never returns such a value silently: it adds a warning, and returns NaN for an
indefinite BC50^0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass
class BC50Result:
    x: np.ndarray                      # bound fraction of A
    bc50: np.ndarray                   # total of A (mol/L)
    free: dict[str, np.ndarray]        # free concentration of each reagent
    totals: dict[str, np.ndarray]      # total of each reagent
    species: list[str]
    fractions: np.ndarray              # (len(x), n_species): a_s [S] / T_A
    concentrations: np.ndarray | None = None   # (len(x), n_species), mol/L; at x = 0 with [B] -> 0
    warnings: list[str] = field(default_factory=list)
    converged: np.ndarray | None = None


def _species_conc(logc, stoich, logbeta):
    return np.exp(np.clip(np.log(10.0) * logbeta + stoich @ logc, -700.0, 700.0))


def bc50_curve(
    reagents: Sequence[str],
    stoich: Sequence[Sequence[int]],
    log_beta: Sequence[float],
    analysed: str,
    bound: str,
    ancillary: str | None = None,
    *,
    x: Sequence[float] | None = None,
    condition: str = "fixed_free",
    free_ancillary: float = 1.0,
    counterion: str | None = None,
    species_names: Sequence[str] | None = None,
) -> BC50Result:
    """BC50 along ``x`` (default 0, 0.02, ..., 0.98).

    ``stoich``/``log_beta`` describe the complexes only (free reagents are implicit),
    with columns in the order of ``reagents``. ``condition`` is ``"fixed_free"``
    ([C] = ``free_ancillary``) or ``"electroneutral"`` (T_C = T_counterion). Two-reagent
    systems pass ``ancillary=None``. ``x = 0`` gives BC50^0.
    """
    reagents = list(reagents)
    nr = len(reagents)
    S = np.vstack([np.eye(nr, dtype=float), np.asarray(stoich, dtype=float).reshape(-1, nr)])
    lb = np.concatenate([np.zeros(nr), np.asarray(log_beta, dtype=float)])
    names = list(reagents) + list(species_names or [f"S{i + 1}" for i in range(S.shape[0] - nr)])
    ia, ib = reagents.index(analysed), reagents.index(bound)
    ic = reagents.index(ancillary) if ancillary is not None else None
    if ic is None and nr != 2:
        raise ValueError("two-reagent BC50 needs exactly two reagents; pass ancillary for three")
    if ic is not None:
        if condition not in ("fixed_free", "electroneutral"):
            raise ValueError("condition must be 'fixed_free' or 'electroneutral'")
        if condition == "electroneutral":
            if counterion not in (analysed, bound):
                raise ValueError("electroneutral condition needs counterion = analysed or bound reagent")
            ik = reagents.index(counterion)
    a, b = S[:, ia], S[:, ib]
    is_bound = (a > 0) & (b > 0)
    warnings: list[str] = []
    multi_b = is_bound & (b > 1)
    mono_b = is_bound & (b == 1)
    if multi_b.any():
        warnings.append(
            "species multinuclear in the bound analyte present ("
            + ", ".join(np.asarray(names)[multi_b]) + "): their weight in BC50 depends on x_A "
            "and on T_B, and BC50^0 ignores them (R09a, note 18)")
    xs = np.asarray(x if x is not None else np.round(np.arange(0.0, 0.99, 0.02), 2), dtype=float)
    if np.any((xs < 0) | (xs >= 1)):
        raise ValueError("x must lie in [0, 1)")

    n = xs.size
    free = {r: np.full(n, np.nan) for r in reagents}
    tot = {r: np.full(n, np.nan) for r in reagents}
    frac = np.full((n, S.shape[0]), np.nan)
    concs = np.full((n, S.shape[0]), np.nan)
    bc = np.full(n, np.nan)
    conv = np.zeros(n, dtype=bool)

    def totals_of(c):
        return S.T @ c

    # ---------------------------------------------------------------- x = 0 (BC50^0)
    def solve_x0():
        if not mono_b.any():
            warnings.append("BC50^0 is indefinite: no species mononuclear in the bound analyte "
                            "contains the analysed reagent (R09a, note 18); returned NaN")
            return None
        electro = ic is not None and condition == "electroneutral"
        c_zero = electro and ik == ib          # T_C = T_B -> 0 when x_A -> 0, so [C] -> 0
        c_unknown = electro and ik == ia       # T_C = T_A with B absent

        def terms(u):
            """beta [A]^a [C]^c for every species: the concentration of species without B,
            and the concentration divided by [B] for species with b = 1 (eq A19 -> A20)."""
            e = np.log(10.0) * lb + S[:, ia] * u[0]
            if ic is not None:
                if c_zero:
                    e = np.where(S[:, ic] > 0, -np.inf, e)
                else:
                    e = e + S[:, ic] * (u[1] if c_unknown else np.log(free_ancillary))
            return np.exp(np.clip(e, -745.0, 700.0))

        def g(u):
            t = terms(u)
            res = [t[mono_b].sum() / t[(b == 1) & (a == 0)].sum() - 1.0]   # eq A20 (free B -> 1)
            if c_unknown:
                c0 = t * (b == 0)
                tc, ta = (S[:, ic] * c0).sum(), (S[:, ia] * c0).sum()
                res.append((tc - ta) / (tc + ta))
            return np.asarray(res)

        nunk = 2 if c_unknown else 1
        best = None
        for v in (1e-3, 1e-6, 1e-1, 1e-9):
            sol = least_squares(g, np.full(nunk, np.log(v)), xtol=1e-15, ftol=1e-15, gtol=1e-15,
                                max_nfev=4000)
            if best is None or sol.cost < best.cost:
                best = sol
        t = terms(best.x)
        c0 = t * (b == 0)                       # species present at [B] = 0
        lc = np.full(nr, -np.inf)
        lc[ia] = best.x[0]
        if ic is not None and not c_zero:
            lc[ic] = best.x[1] if c_unknown else np.log(free_ancillary)
        return lc, c0, bool(best.cost < 1e-20)

    # ---------------------------------------------------------------- x > 0
    def residual(u, alpha):
        lc = np.empty(nr)
        lc[ia], lc[ib] = u[0], u[1]
        k = 2
        if ic is not None:
            if condition == "fixed_free":
                lc[ic] = np.log(free_ancillary)
            else:
                lc[ic] = u[k]
                k += 1
        c = _species_conc(lc, S, lb)
        bB, uB = (b * c)[is_bound].sum(), (b * c)[~is_bound & (b > 0)].sum()
        bA, uA = (a * c)[is_bound].sum(), (a * c)[~is_bound & (a > 0)].sum()
        res = [(bB - uB) / (bB + uB), (bA - alpha * uA) / (bA + alpha * uA)]
        if ic is not None and condition == "electroneutral":
            t = totals_of(c)
            res.append((t[ic] - t[ik]) / (t[ic] + t[ik]))
        return np.asarray(res), lc, c

    prev = None
    for i in np.argsort(xs):
        xv = xs[i]
        if xv == 0.0:
            out = solve_x0()
            if out is None:
                continue
            lc, c0, ok = out
            ta = (a * c0).sum()
            bc[i] = ta
            for j, r in enumerate(reagents):
                free[r][i] = np.exp(lc[j]) if np.isfinite(lc[j]) else 0.0
                tot[r][i] = (S[:, j] * c0).sum()
            frac[i] = a * c0 / ta
            concs[i] = c0
            conv[i] = ok
            continue
        alpha = xv / (1.0 - xv)
        nunk = 2 + (1 if (ic is not None and condition == "electroneutral") else 0)
        starts = [prev] if prev is not None else []
        starts += [np.full(nunk, np.log(v)) for v in (1e-3, 1e-5, 1e-2, 1e-7)]
        best = None
        for st in starts:
            sol = least_squares(lambda u: residual(u, alpha)[0], st, xtol=1e-15, ftol=1e-15,
                                gtol=1e-15, max_nfev=4000)
            if best is None or sol.cost < best.cost:
                best = sol
            if best.cost < 1e-24:
                break
        _, lc, c = residual(best.x, alpha)
        prev = best.x
        t = totals_of(c)
        bc[i] = t[ia]
        for j, r in enumerate(reagents):
            free[r][i] = np.exp(lc[j])
            tot[r][i] = t[j]
        frac[i] = a * c / t[ia]
        concs[i] = c
        conv[i] = best.cost < 1e-20
    if not conv[np.isfinite(bc)].all():
        warnings.append("BC50 equations not solved to tolerance at x = "
                        + ", ".join(f"{v:g}" for v in xs[np.isfinite(bc) & ~conv]))
    return BC50Result(x=xs, bc50=bc, free=free, totals=tot, species=names, fractions=frac,
                      concentrations=concs, warnings=warnings, converged=conv)
