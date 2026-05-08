# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import brentq, least_squares

from hmfit_core.acid_base import (
    AcidBaseSystem,
    AcidBaseFitResult,
    SpeciesDistributionResult,
    clone_system,
    clone_system_with_pka,
    component_charges,
    component_log_beta,
    component_species_names,
    distribution_fractions_from_pH,
    system_pka_values,
    system_log_beta_values,
    update_system_from_parameter_vector,
    _build_fit_result,
)


@dataclass
class PotentiometryExperiment:
    initial_volume: float
    titrant_volumes: np.ndarray
    measured_pH: np.ndarray | None = None
    measured_emf: np.ndarray | None = None
    analyte_concentration: float | None = None
    titrant_concentration: float | None = None
    titrant_type: str = "base"
    temperature: float = 298.15
    electrode_e0: float | None = None
    electrode_slope: float | None = None
    ph_from_emf_intercept: float | None = None
    ph_from_emf_slope: float | None = None
    volume_offset: float = 0.0


def _as_1d(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def compute_diluted_totals(
    initial_volume: float,
    titrant_volumes: np.ndarray,
    initial_concentrations: dict[str, float],
    titrant_concentrations: dict[str, float],
) -> dict[str, np.ndarray]:
    """
    Compute analytical concentrations after each titrant addition.

    C_i,total = (C_i,0 V0 + C_i,titrant Vadd) / (V0 + Vadd)

    Volumes only need to be internally consistent, for example all in mL.
    """
    v0 = float(initial_volume)
    vadd = _as_1d(titrant_volumes, name="titrant_volumes")
    vtot = v0 + vadd
    if v0 <= 0.0 or np.any(vtot <= 0.0):
        raise ValueError("Volumes must be positive after titrant addition.")
    keys = set(initial_concentrations or {}) | set(titrant_concentrations or {})
    out: dict[str, np.ndarray] = {}
    for key in sorted(keys):
        c0 = float((initial_concentrations or {}).get(key, 0.0) or 0.0)
        ct = float((titrant_concentrations or {}).get(key, 0.0) or 0.0)
        out[str(key)] = (c0 * v0 + ct * vadd) / vtot
    return out


def electrode_emf_from_pH(
    pH: np.ndarray,
    *,
    electrode_e0: float | None,
    electrode_slope: float | None,
) -> np.ndarray:
    e0 = 0.0 if electrode_e0 is None else float(electrode_e0)
    slope = -59.16 if electrode_slope is None else float(electrode_slope)
    return e0 + slope * np.asarray(pH, dtype=float)


def electrode_pH_from_emf(
    emf: np.ndarray,
    *,
    electrode_e0: float | None = None,
    electrode_slope: float | None = None,
    ph_from_emf_intercept: float | None = None,
    ph_from_emf_slope: float | None = None,
) -> np.ndarray:
    E = np.asarray(emf, dtype=float)
    if ph_from_emf_intercept is not None or ph_from_emf_slope is not None:
        a = 0.0 if ph_from_emf_intercept is None else float(ph_from_emf_intercept)
        b = 1.0 if ph_from_emf_slope is None else float(ph_from_emf_slope)
        return a + b * E
    e0 = 0.0 if electrode_e0 is None else float(electrode_e0)
    slope = -59.16 if electrode_slope is None else float(electrode_slope)
    if abs(slope) <= 1e-300:
        raise ValueError("electrode_slope must be non-zero.")
    return (E - e0) / slope


def observed_pH(experiment: PotentiometryExperiment) -> np.ndarray | None:
    if experiment.measured_pH is not None:
        return np.asarray(experiment.measured_pH, dtype=float).reshape(-1)
    if experiment.measured_emf is None:
        return None
    return electrode_pH_from_emf(
        experiment.measured_emf,
        electrode_e0=experiment.electrode_e0,
        electrode_slope=experiment.electrode_slope,
        ph_from_emf_intercept=experiment.ph_from_emf_intercept,
        ph_from_emf_slope=experiment.ph_from_emf_slope,
    ).reshape(-1)


def _component_totals_for_experiment(
    experiment: PotentiometryExperiment,
    system: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, np.ndarray]:
    opts = dict(fit_options or {})
    initial: dict[str, float] = {}
    titrant: dict[str, float] = {}
    for idx, comp in enumerate(system.components):
        value = float(comp.analytical_concentration)
        if idx == 0 and experiment.analyte_concentration is not None:
            value = float(experiment.analyte_concentration)
        initial[str(comp.name)] = value
        titrant[str(comp.name)] = 0.0
    initial.update({str(k): float(v) for k, v in (opts.get("initial_concentrations") or {}).items()})
    titrant.update({str(k): float(v) for k, v in (opts.get("titrant_concentrations") or {}).items()})
    volumes = _effective_titrant_volumes(experiment, opts)
    return compute_diluted_totals(
        float(experiment.initial_volume),
        volumes,
        initial,
        titrant,
    )


def _effective_titrant_volumes(
    experiment: PotentiometryExperiment,
    fit_options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    opts = dict(fit_options or {})
    offset = float(opts.get("volume_offset", experiment.volume_offset or 0.0) or 0.0)
    return _as_1d(experiment.titrant_volumes, name="titrant_volumes") + offset


def _strong_ion_charge(
    experiment: PotentiometryExperiment,
    fit_options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    opts = dict(fit_options or {})
    volumes = _effective_titrant_volumes(experiment, opts)
    v0 = float(experiment.initial_volume)
    vtot = v0 + volumes
    if np.any(vtot <= 0.0):
        raise ValueError("Volumes must be positive after titrant addition.")

    explicit = opts.get("strong_ion_charge")
    if explicit is not None:
        arr = np.asarray(explicit, dtype=float)
        if arr.ndim == 0:
            return np.full(volumes.shape, float(arr), dtype=float)
        arr = arr.reshape(-1)
        if arr.size != volumes.size:
            raise ValueError("strong_ion_charge length must match titration points.")
        return arr

    charge = np.zeros_like(volumes, dtype=float)
    initial_strong = float(opts.get("initial_strong_charge", 0.0) or 0.0)
    titrant_strong = opts.get("titrant_strong_charge")
    if titrant_strong is None:
        ct = experiment.titrant_concentration
        if ct is None:
            ct = opts.get("titrant_concentration", 0.0)
        ct = float(ct or 0.0)
        titrant_type = str(experiment.titrant_type or "base").strip().lower()
        # Strong base contributes spectator cation; strong acid contributes
        # spectator anion. H+ and OH- are included explicitly in charge balance.
        titrant_strong = ct if titrant_type == "base" else -ct
    titrant_strong = float(titrant_strong or 0.0)
    charge += (initial_strong * v0 + titrant_strong * volumes) / vtot

    strong_charges = opts.get("strong_ion_charges") or {}
    if strong_charges:
        totals = compute_diluted_totals(
            v0,
            volumes,
            {str(k): float(v) for k, v in (opts.get("initial_strong_ions") or {}).items()},
            {str(k): float(v) for k, v in (opts.get("titrant_strong_ions") or {}).items()},
        )
        for ion, conc in totals.items():
            charge += float(strong_charges.get(ion, 0.0) or 0.0) * conc
    return charge


def charge_balance_at_pH(
    pH: float,
    system: AcidBaseSystem,
    analytical_totals: Mapping[str, float],
    *,
    strong_ion_charge: float = 0.0,
) -> float:
    h = 10.0 ** (-float(pH))
    oh = float(system.kw) / max(h, 1e-300)
    charge = h - oh + float(strong_ion_charge)
    for comp in system.components:
        total = float(analytical_totals.get(str(comp.name), comp.analytical_concentration) or 0.0)
        fractions = distribution_fractions_from_pH(np.asarray([pH], dtype=float), component_log_beta(comp))[0]
        charges = component_charges(comp)
        charge += total * float(np.dot(fractions, charges))
    return float(charge)


def solve_pH_for_totals(
    system: AcidBaseSystem,
    analytical_totals: Mapping[str, float],
    *,
    strong_ion_charge: float = 0.0,
    pH_bounds: tuple[float, float] = (-2.0, 16.0),
    previous_pH: float | None = None,
) -> float:
    lo, hi = float(pH_bounds[0]), float(pH_bounds[1])
    if lo >= hi:
        raise ValueError("pH_bounds must be increasing.")

    def f(ph: float) -> float:
        return charge_balance_at_pH(
            ph,
            system,
            analytical_totals,
            strong_ion_charge=strong_ion_charge,
        )

    flo = f(lo)
    fhi = f(hi)
    if np.isfinite(flo) and abs(flo) < 1e-14:
        return lo
    if np.isfinite(fhi) and abs(fhi) < 1e-14:
        return hi
    if np.isfinite(flo) and np.isfinite(fhi) and flo * fhi < 0.0:
        return float(brentq(f, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=200))

    grid = np.linspace(lo, hi, 361)
    vals = np.asarray([f(x) for x in grid], dtype=float)
    candidates: list[tuple[float, int]] = []
    for i in range(grid.size - 1):
        a = vals[i]
        b = vals[i + 1]
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0:
            return float(grid[i])
        if a * b < 0.0:
            midpoint = 0.5 * (grid[i] + grid[i + 1])
            distance = abs(midpoint - float(previous_pH)) if previous_pH is not None else abs(a) + abs(b)
            candidates.append((distance, i))
    if candidates:
        _, i = min(candidates, key=lambda item: item[0])
        return float(brentq(f, grid[i], grid[i + 1], xtol=1e-12, rtol=1e-12, maxiter=200))

    x0 = 7.0 if previous_pH is None else float(np.clip(previous_pH, lo, hi))
    opt = least_squares(lambda x: np.asarray([f(float(x[0]))]), np.asarray([x0]), bounds=([lo], [hi]))
    return float(opt.x[0])


def simulate_pH_titration(
    experiment: PotentiometryExperiment,
    system: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None = None,
) -> np.ndarray:
    opts = dict(fit_options or {})
    totals = _component_totals_for_experiment(experiment, system, opts)
    strong = _strong_ion_charge(experiment, opts)
    volumes = _effective_titrant_volumes(experiment, opts)
    if strong.size != volumes.size:
        raise ValueError("Strong-ion charge vector length mismatch.")
    bounds = tuple(opts.get("pH_bounds", (-2.0, 16.0)))
    pH = np.empty(volumes.size, dtype=float)
    prev: float | None = None
    for i in range(volumes.size):
        point_totals = {name: float(vals[i]) for name, vals in totals.items()}
        pH[i] = solve_pH_for_totals(
            system,
            point_totals,
            strong_ion_charge=float(strong[i]),
            pH_bounds=(float(bounds[0]), float(bounds[1])),
            previous_pH=prev,
        )
        prev = float(pH[i])
    return pH


def simulate_species_vs_volume(
    experiment: PotentiometryExperiment,
    system: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None = None,
) -> SpeciesDistributionResult:
    opts = dict(fit_options or {})
    volumes = _effective_titrant_volumes(experiment, opts)
    pH = simulate_pH_titration(experiment, system, opts)
    totals = _component_totals_for_experiment(experiment, system, opts)
    names: list[str] = []
    conc_blocks: list[np.ndarray] = []
    frac_blocks: list[np.ndarray] = []
    for comp in system.components:
        frac = distribution_fractions_from_pH(pH, component_log_beta(comp))
        total = np.asarray(totals[str(comp.name)], dtype=float).reshape(-1)
        conc = frac * total[:, None]
        names.extend(component_species_names(comp))
        frac_blocks.append(frac)
        conc_blocks.append(conc)
    fractions = np.concatenate(frac_blocks, axis=1) if frac_blocks else np.empty((pH.size, 0))
    concentrations = np.concatenate(conc_blocks, axis=1) if conc_blocks else np.empty((pH.size, 0))
    return SpeciesDistributionResult(
        x=volumes,
        x_label="Titrant volume",
        species_names=names,
        concentrations=concentrations,
        fractions=fractions,
        pH=pH,
    )


def _apply_potentiometry_local_params(
    params: np.ndarray,
    consumed: np.ndarray,
    experiment: PotentiometryExperiment,
    system: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None,
) -> tuple[PotentiometryExperiment, AcidBaseSystem, dict[str, Any]]:
    opts = dict(fit_options or {})
    names = [str(n) for n in (opts.get("parameter_names") or [])]
    if names and len(names) != params.size:
        raise ValueError("parameter_names length must match params length.")
    exp = replace(experiment)
    sys = clone_system(system)
    local_opts = dict(opts)
    if not names:
        return exp, sys, local_opts

    for idx, name_raw in enumerate(names):
        if consumed[idx]:
            continue
        name = name_raw.strip().lower().replace(" ", "_").replace("-", "_")
        value = float(params[idx])
        if name in {"analyte_concentration", "c0", "concentration"}:
            exp.analyte_concentration = value
            if sys.components:
                sys.components[0].analytical_concentration = value
        elif name in {"titrant_concentration", "ctitrant", "ct"}:
            exp.titrant_concentration = value
        elif name in {"initial_volume", "v0"}:
            exp.initial_volume = value
        elif name in {"blank_volume", "volume_offset", "vblank"}:
            exp.volume_offset = value
            local_opts["volume_offset"] = value
        elif name in {"electrode_e0", "e0"}:
            exp.electrode_e0 = value
        elif name in {"electrode_slope", "slope"}:
            exp.electrode_slope = value
        elif name == "kw":
            sys.kw = value
    return exp, sys, local_opts


def potentiometry_residuals(
    params: np.ndarray,
    experiment: PotentiometryExperiment,
    system_template: AcidBaseSystem,
    fit_options: dict,
) -> np.ndarray:
    p = np.asarray(params, dtype=float).reshape(-1)
    system, consumed = update_system_from_parameter_vector(p, system_template, fit_options)
    exp, system, opts = _apply_potentiometry_local_params(
        p,
        consumed,
        experiment,
        system,
        fit_options,
    )
    calc_pH = simulate_pH_titration(exp, system, opts)
    fit_signal = str(opts.get("fit_signal", "auto")).strip().lower()
    use_emf = fit_signal in {"emf", "e", "potential"} or (
        fit_signal == "auto" and exp.measured_pH is None and exp.measured_emf is not None
    )
    if use_emf:
        if exp.measured_emf is None:
            raise ValueError("EMF residual requested but measured_emf is missing.")
        obs = np.asarray(exp.measured_emf, dtype=float).reshape(-1)
        calc = electrode_emf_from_pH(
            calc_pH,
            electrode_e0=exp.electrode_e0,
            electrode_slope=exp.electrode_slope,
        )
        sigma = float(opts.get("sigma_E", opts.get("sigma_emf", 1.0)) or 1.0)
    else:
        obs = observed_pH(exp)
        if obs is None:
            raise ValueError("No measured pH or EMF data are available.")
        calc = calc_pH
        sigma = float(opts.get("sigma_pH", 1.0) or 1.0)
    if obs.size != calc.size:
        raise ValueError("Observed and calculated potentiometry vectors have different lengths.")
    residual = (obs - calc) / max(sigma, 1e-300)
    return residual[np.isfinite(residual)].reshape(-1)


def fit_potentiometry(
    experiment: PotentiometryExperiment,
    system_template: AcidBaseSystem,
    initial_pka: Sequence[float] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> AcidBaseFitResult:
    opts = dict(fit_options or {})
    parameter_names_opt = opts.get("parameter_names")
    if parameter_names_opt:
        parameter_names = [str(name) for name in parameter_names_opt]
        if opts.get("initial_params") is not None:
            x0 = _as_1d(opts.get("initial_params"), name="initial_params")
        else:
            pka0 = np.asarray(
                system_pka_values(system_template) if initial_pka is None else initial_pka,
                dtype=float,
            ).reshape(-1)
            x_values: list[float] = []
            for name_raw in parameter_names:
                name = name_raw.strip().lower().replace(" ", "_").replace("-", "_")
                if name.startswith("pka"):
                    digits = "".join(ch for ch in name if ch.isdigit())
                    idx = int(digits or "1") - 1
                    x_values.append(float(pka0[idx]))
                elif name in {"electrode_e0", "e0"}:
                    x_values.append(0.0 if experiment.electrode_e0 is None else float(experiment.electrode_e0))
                elif name in {"electrode_slope", "slope"}:
                    x_values.append(
                        -59.16 if experiment.electrode_slope is None else float(experiment.electrode_slope)
                    )
                elif name in {"analyte_concentration", "c0", "concentration"}:
                    value = experiment.analyte_concentration
                    if value is None and system_template.components:
                        value = system_template.components[0].analytical_concentration
                    x_values.append(float(value or 0.0))
                elif name in {"titrant_concentration", "ctitrant", "ct"}:
                    x_values.append(float(experiment.titrant_concentration or 0.0))
                elif name in {"blank_volume", "volume_offset", "vblank"}:
                    x_values.append(float(experiment.volume_offset or 0.0))
                elif name == "kw":
                    x_values.append(float(system_template.kw))
                else:
                    x_values.append(0.0)
            x0 = np.asarray(x_values, dtype=float)
        if x0.size != len(parameter_names):
            raise ValueError("initial_params length must match parameter_names.")
    else:
        if initial_pka is None:
            x0 = np.asarray(system_pka_values(system_template), dtype=float)
        else:
            x0 = _as_1d(initial_pka, name="initial_pka")
        parameter_names = [f"pKa{i+1}" for i in range(x0.size)]
    bounds = opts.get("bounds", (-5.0, 25.0))

    def residual(theta: np.ndarray) -> np.ndarray:
        return potentiometry_residuals(theta, experiment, system_template, opts)

    opt = least_squares(residual, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    fitted_system, _ = update_system_from_parameter_vector(opt.x, system_template, opts)
    return _build_fit_result(
        opt,
        fitted_system,
        residual(opt.x),
        parameter_names,
        bounds=bounds,
        fixed_mask=opts.get("fixed_mask"),
        parameter_space="pka",
    )


__all__ = [
    "PotentiometryExperiment",
    "compute_diluted_totals",
    "electrode_emf_from_pH",
    "electrode_pH_from_emf",
    "observed_pH",
    "charge_balance_at_pH",
    "solve_pH_for_totals",
    "simulate_pH_titration",
    "simulate_species_vs_volume",
    "potentiometry_residuals",
    "fit_potentiometry",
]
