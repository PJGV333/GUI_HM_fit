# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Mapping, Sequence
import re

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    from hmfit_core.potentiometry import PotentiometryExperiment


@dataclass
class AcidBaseSpecies:
    name: str
    charge: int
    h_count: int
    log_beta: float | None = None
    fixed: bool = False


@dataclass
class AcidBaseComponent:
    name: str
    analytical_concentration: float
    species: list[AcidBaseSpecies]


@dataclass
class AcidBaseSystem:
    components: list[AcidBaseComponent]
    temperature: float = 298.15
    ionic_strength: float | None = None
    kw: float = 1e-14


@dataclass
class SpeciesDistributionResult:
    x: np.ndarray
    x_label: str
    species_names: list[str]
    concentrations: np.ndarray
    fractions: np.ndarray
    pH: np.ndarray | None = None


@dataclass
class SpectroscopicAcidBaseDataset:
    pH: np.ndarray
    signal: np.ndarray
    wavelengths: np.ndarray | None = None
    technique: str = "uvvis"


@dataclass
class NMRAcidBaseDataset:
    pH: np.ndarray
    shifts: np.ndarray
    nuclei_labels: list[str]
    exchange_regime: str = "fast"


@dataclass
class GlobalAcidBaseFit:
    system_template: AcidBaseSystem
    potentiometry_experiments: list["PotentiometryExperiment"] = field(default_factory=list)
    spectroscopy_datasets: list[SpectroscopicAcidBaseDataset] = field(default_factory=list)
    nmr_datasets: list[NMRAcidBaseDataset] = field(default_factory=list)
    shared_parameters: list[str] = field(default_factory=list)
    local_parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class AcidBaseFitResult:
    success: bool
    message: str
    fitted_pka: list[float]
    fitted_log_beta: list[float]
    parameter_table: pd.DataFrame
    covariance_matrix: np.ndarray | None
    correlation_matrix: np.ndarray | None
    residuals: np.ndarray
    chi_square: float
    reduced_chi_square: float | None
    aic: float | None
    bic: float | None


_PKA_RE = re.compile(r"^pka(?:[_\-\s]*|\[)?(\d+)\]?$", re.IGNORECASE)
_LOGBETA_RE = re.compile(
    r"^(?:log[_\-\s]*beta|logbeta|beta)(?:[_\-\s]*|\[)?(\d+)\]?$",
    re.IGNORECASE,
)


def _as_1d_float(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def pka_to_log_beta(pka_values: list[float]) -> list[float]:
    """
    Convert stepwise pKa values to cumulative protonation constants.

    Convention used by HM Fit acid-base tools:

        L + n H+ <=> H_nL
        beta_n = [H_nL] / ([L] [H+]**n)

    with log_beta_0 = 0. Therefore:

        pKa_n = log_beta_n - log_beta_(n-1)
        log_beta_n = pKa_1 + ... + pKa_n

    The returned list contains log_beta_1, ..., log_beta_n.
    """
    pka = _as_1d_float(pka_values, name="pka_values")
    return np.cumsum(pka).astype(float).tolist()


def log_beta_to_pka(log_beta_values: list[float]) -> list[float]:
    """
    Convert cumulative protonation constants to stepwise pKa values.

    See :func:`pka_to_log_beta` for the chemical convention. The input list
    contains log_beta_1, ..., log_beta_n and log_beta_0 is assumed to be 0.
    """
    log_beta = _as_1d_float(log_beta_values, name="log_beta_values")
    prev = np.concatenate(([0.0], log_beta[:-1]))
    return (log_beta - prev).astype(float).tolist()


def distribution_fractions_from_pH(
    pH: np.ndarray,
    log_beta: list[float],
) -> np.ndarray:
    """
    Return species fractions for L, HL, H2L, ..., HnL at imposed pH.

    For cumulative protonation constants beta_i and beta_0 = 1:

        alpha_i = beta_i [H+]**i / sum_j(beta_j [H+]**j)

    The output has shape ``n_points x n_species``. Each row sums to 1 within
    numerical precision.
    """
    ph_arr = np.asarray(pH, dtype=float).reshape(-1)
    if not np.all(np.isfinite(ph_arr)):
        raise ValueError("pH contains non-finite values.")

    lb = _as_1d_float(log_beta, name="log_beta")
    log_beta_full = np.concatenate(([0.0], lb))
    h_counts = np.arange(log_beta_full.size, dtype=float)
    log_terms = log_beta_full[None, :] - ph_arr[:, None] * h_counts[None, :]
    log_terms -= np.max(log_terms, axis=1, keepdims=True)
    terms = np.power(10.0, np.clip(log_terms, -300.0, 300.0))
    denom = np.sum(terms, axis=1, keepdims=True)
    if np.any(denom <= 0.0):
        raise ValueError("Invalid distribution denominator.")
    return terms / denom


def component_species_sorted(component: AcidBaseComponent) -> list[AcidBaseSpecies]:
    species = sorted(list(component.species or []), key=lambda sp: int(sp.h_count))
    if not species:
        raise ValueError(f"Component {component.name!r} has no acid-base species.")
    counts = [int(sp.h_count) for sp in species]
    expected = list(range(max(counts) + 1))
    if counts != expected:
        raise ValueError(
            f"Species for component {component.name!r} must use consecutive h_count values "
            f"starting at 0; got {counts}."
        )
    return species


def component_log_beta(component: AcidBaseComponent) -> list[float]:
    species = component_species_sorted(component)
    vals: list[float] = []
    for sp in species[1:]:
        if sp.log_beta is None:
            raise ValueError(
                f"Species {sp.name!r} needs log_beta for h_count={sp.h_count}."
            )
        vals.append(float(sp.log_beta))
    return vals


def component_charges(component: AcidBaseComponent) -> np.ndarray:
    return np.asarray([int(sp.charge) for sp in component_species_sorted(component)], dtype=float)


def component_species_names(component: AcidBaseComponent) -> list[str]:
    return [str(sp.name) for sp in component_species_sorted(component)]


def system_pka_values(system: AcidBaseSystem, *, component_index: int = 0) -> list[float]:
    return log_beta_to_pka(component_log_beta(system.components[int(component_index)]))


def system_log_beta_values(system: AcidBaseSystem, *, component_index: int = 0) -> list[float]:
    return component_log_beta(system.components[int(component_index)])


def clone_system(system: AcidBaseSystem) -> AcidBaseSystem:
    return AcidBaseSystem(
        components=[
            AcidBaseComponent(
                name=str(comp.name),
                analytical_concentration=float(comp.analytical_concentration),
                species=[
                    AcidBaseSpecies(
                        name=str(sp.name),
                        charge=int(sp.charge),
                        h_count=int(sp.h_count),
                        log_beta=(None if sp.log_beta is None else float(sp.log_beta)),
                        fixed=bool(sp.fixed),
                    )
                    for sp in comp.species
                ],
            )
            for comp in system.components
        ],
        temperature=float(system.temperature),
        ionic_strength=(
            None if system.ionic_strength is None else float(system.ionic_strength)
        ),
        kw=float(system.kw),
    )


def clone_system_with_log_beta(
    system: AcidBaseSystem,
    log_beta: Sequence[float],
    *,
    component_index: int = 0,
) -> AcidBaseSystem:
    out = clone_system(system)
    comp = out.components[int(component_index)]
    lb = _as_1d_float(log_beta, name="log_beta")
    species = component_species_sorted(comp)
    if lb.size != len(species) - 1:
        raise ValueError(
            f"Expected {len(species) - 1} log_beta values for {comp.name!r}, got {lb.size}."
        )
    by_h = {int(sp.h_count): sp for sp in comp.species}
    for h_count, value in enumerate(lb.tolist(), start=1):
        by_h[h_count].log_beta = float(value)
    return out


def clone_system_with_pka(
    system: AcidBaseSystem,
    pka: Sequence[float],
    *,
    component_index: int = 0,
) -> AcidBaseSystem:
    return clone_system_with_log_beta(
        system,
        pka_to_log_beta([float(v) for v in pka]),
        component_index=component_index,
    )


def make_simple_acid_base_system(
    *,
    name: str = "L",
    analytical_concentration: float = 1.0e-3,
    pka: Sequence[float] | None = None,
    log_beta: Sequence[float] | None = None,
    base_charge: int = -1,
    temperature: float = 298.15,
    ionic_strength: float | None = None,
    kw: float = 1e-14,
) -> AcidBaseSystem:
    if log_beta is None:
        if pka is None:
            pka = [5.0]
        log_beta = pka_to_log_beta([float(v) for v in pka])
    lb = _as_1d_float(log_beta, name="log_beta")
    species = [
        AcidBaseSpecies(name=str(name), charge=int(base_charge), h_count=0, log_beta=None)
    ]
    for h_count, value in enumerate(lb.tolist(), start=1):
        prefix = "H" if h_count == 1 else f"H{h_count}"
        species.append(
            AcidBaseSpecies(
                name=f"{prefix}{name}",
                charge=int(base_charge) + h_count,
                h_count=h_count,
                log_beta=float(value),
            )
        )
    return AcidBaseSystem(
        components=[
            AcidBaseComponent(
                name=str(name),
                analytical_concentration=float(analytical_concentration),
                species=species,
            )
        ],
        temperature=float(temperature),
        ionic_strength=ionic_strength,
        kw=float(kw),
    )


def build_solver_inputs_for_component(
    component: AcidBaseComponent,
    *,
    proton_name: str = "H",
    exclude_free_proton: bool = True,
) -> dict[str, Any]:
    """
    Build HM Fit mass-balance solver inputs for L + nH <=> HnL.

    The existing concentration solvers use components as rows and species as
    columns. This helper exposes acid-base chemistry in that same convention.
    Potentiometric pH fitting still needs an outer electroneutrality equation,
    because the current mass-balance solver does not directly solve charge
    balance.
    """
    species = component_species_sorted(component)
    component_names = [str(component.name), str(proton_name)]
    species_names = [str(component.name), str(proton_name)]
    model_cols = [[1.0, 0.0], [0.0, 1.0]]
    log_beta = []
    for sp in species[1:]:
        species_names.append(str(sp.name))
        model_cols.append([1.0, float(sp.h_count)])
        log_beta.append(float(sp.log_beta))
    modelo = np.asarray(model_cols, dtype=float).T
    nas = [1] if exclude_free_proton else []
    return {
        "component_names": component_names,
        "species_names": species_names,
        "modelo": modelo,
        "nas": nas,
        "log_beta": np.asarray(log_beta, dtype=float),
    }


def simulate_species_vs_pH(
    system: AcidBaseSystem,
    pH_min: float = 0.0,
    pH_max: float = 14.0,
    n_points: int = 500,
) -> SpeciesDistributionResult:
    ph = np.linspace(float(pH_min), float(pH_max), int(n_points))
    names: list[str] = []
    concentrations_blocks: list[np.ndarray] = []
    fractions_blocks: list[np.ndarray] = []
    for comp in system.components:
        frac = distribution_fractions_from_pH(ph, component_log_beta(comp))
        conc = frac * float(comp.analytical_concentration)
        names.extend(component_species_names(comp))
        fractions_blocks.append(frac)
        concentrations_blocks.append(conc)
    fractions = np.concatenate(fractions_blocks, axis=1) if fractions_blocks else np.empty((ph.size, 0))
    concentrations = (
        np.concatenate(concentrations_blocks, axis=1)
        if concentrations_blocks
        else np.empty((ph.size, 0))
    )
    return SpeciesDistributionResult(
        x=ph,
        x_label="pH",
        species_names=names,
        concentrations=concentrations,
        fractions=fractions,
        pH=ph,
    )


def _match_index(pattern: re.Pattern[str], name: str) -> int | None:
    match = pattern.match(str(name).strip())
    if not match:
        return None
    idx = int(match.group(1)) - 1
    if idx < 0:
        raise ValueError(f"Parameter index in {name!r} must be one-based.")
    return idx


def update_system_from_parameter_vector(
    params: Sequence[float] | np.ndarray,
    system_template: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None = None,
) -> tuple[AcidBaseSystem, np.ndarray]:
    """
    Apply pKa/log_beta parameters to a system template.

    If ``fit_options["parameter_names"]`` is absent, the leading parameters are
    interpreted as pKa_1..pKa_n for the selected component. The returned boolean
    mask marks which entries of ``params`` were consumed as chemical parameters.
    """
    opts = dict(fit_options or {})
    p = np.asarray(params, dtype=float).reshape(-1)
    consumed = np.zeros(p.size, dtype=bool)
    component_index = int(opts.get("component_index", 0) or 0)
    base_log_beta = np.asarray(
        component_log_beta(system_template.components[component_index]), dtype=float
    )
    pka = np.asarray(log_beta_to_pka(base_log_beta.tolist()), dtype=float)
    log_beta = base_log_beta.copy()
    pka_touched = False
    log_beta_touched = False

    names_raw = opts.get("parameter_names")
    names = [str(n) for n in names_raw] if names_raw is not None else []
    if names:
        if len(names) != p.size:
            raise ValueError("parameter_names length must match params length.")
        for i, name in enumerate(names):
            pka_idx = _match_index(_PKA_RE, name)
            if pka_idx is not None:
                if pka_idx >= pka.size:
                    raise ValueError(f"{name!r} is outside the pKa vector.")
                pka[pka_idx] = float(p[i])
                consumed[i] = True
                pka_touched = True
                continue
            lb_idx = _match_index(_LOGBETA_RE, name)
            if lb_idx is not None:
                if lb_idx >= log_beta.size:
                    raise ValueError(f"{name!r} is outside the log_beta vector.")
                log_beta[lb_idx] = float(p[i])
                consumed[i] = True
                log_beta_touched = True
        if pka_touched and not log_beta_touched:
            log_beta = np.asarray(pka_to_log_beta(pka.tolist()), dtype=float)
    else:
        n = min(int(p.size), int(pka.size))
        if n:
            pka[:n] = p[:n]
            consumed[:n] = True
            log_beta = np.asarray(pka_to_log_beta(pka.tolist()), dtype=float)

    if pka_touched and log_beta_touched:
        # Explicit log_beta entries are allowed to override pKa-derived entries.
        pass

    return (
        clone_system_with_log_beta(
            system_template,
            log_beta,
            component_index=component_index,
        ),
        consumed,
    )


def _normalise_observed_matrix(values: np.ndarray | Sequence[float], *, name: str) -> tuple[np.ndarray, bool]:
    arr = np.asarray(values, dtype=float)
    was_1d = arr.ndim == 1
    if was_1d:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")
    return arr, was_1d


def _sigma_array(sigma: Any, shape: tuple[int, int]) -> np.ndarray:
    if sigma is None:
        return np.ones(shape, dtype=float)
    arr = np.asarray(sigma, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, max(float(arr), 1e-300), dtype=float)
    if arr.shape == shape:
        out = arr.astype(float, copy=True)
    elif arr.size == shape[0]:
        out = np.broadcast_to(arr.reshape(-1, 1), shape).astype(float, copy=True)
    elif arr.size == shape[1]:
        out = np.broadcast_to(arr.reshape(1, -1), shape).astype(float, copy=True)
    else:
        raise ValueError(f"sigma shape {arr.shape} is incompatible with {shape}.")
    out[~np.isfinite(out) | (out <= 0.0)] = 1.0
    return out


def _linear_observable_prediction(
    fractions: np.ndarray,
    observed: np.ndarray,
    *,
    baseline: bool = False,
    sigma: Any = None,
    pure_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    F = np.asarray(fractions, dtype=float)
    Y = np.asarray(observed, dtype=float)
    if F.ndim != 2 or Y.ndim != 2 or F.shape[0] != Y.shape[0]:
        raise ValueError("fractions and observed data have incompatible shapes.")
    X = F
    if baseline:
        X = np.column_stack([F, np.ones(F.shape[0], dtype=float)])

    if pure_values is None:
        coeff = np.full((X.shape[1], Y.shape[1]), np.nan, dtype=float)
        calc = np.full_like(Y, np.nan, dtype=float)
        weights = _sigma_array(sigma, Y.shape)
        for j in range(Y.shape[1]):
            mask = np.isfinite(Y[:, j]) & np.isfinite(X).all(axis=1) & np.isfinite(weights[:, j])
            if int(mask.sum()) < X.shape[1]:
                mask = np.isfinite(Y[:, j]) & np.isfinite(X).all(axis=1)
            if not np.any(mask):
                continue
            Xj = X[mask, :]
            yj = Y[mask, j]
            wj = 1.0 / np.maximum(weights[mask, j], 1e-300)
            Xw = Xj * wj[:, None]
            yw = yj * wj
            coeff[:, j], *_ = np.linalg.lstsq(Xw, yw, rcond=1e-12)
            calc[:, j] = X @ coeff[:, j]
    else:
        coeff = np.asarray(pure_values, dtype=float)
        if coeff.ndim == 1:
            coeff = coeff.reshape(-1, 1)
        if coeff.shape[0] != X.shape[1]:
            raise ValueError(
                f"pure_values has {coeff.shape[0]} rows, expected {X.shape[1]}."
            )
        calc = X @ coeff
    residual = Y - calc
    weights = _sigma_array(sigma, Y.shape)
    residual = residual / weights
    return calc, coeff, residual


def predict_spectroscopy_acid_base(
    dataset: SpectroscopicAcidBaseDataset,
    system: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    opts = dict(fit_options or {})
    pH = np.asarray(dataset.pH, dtype=float).reshape(-1)
    Y, was_1d = _normalise_observed_matrix(dataset.signal, name="signal")
    if Y.shape[0] != pH.size:
        raise ValueError("dataset.pH and dataset.signal have incompatible lengths.")
    comp = system.components[int(opts.get("component_index", 0) or 0)]
    fractions = distribution_fractions_from_pH(pH, component_log_beta(comp))
    calc, coeff, residual = _linear_observable_prediction(
        fractions,
        Y,
        baseline=bool(opts.get("baseline", False)),
        sigma=opts.get("sigma"),
        pure_values=opts.get("species_signals"),
    )
    return {
        "pH": pH,
        "fractions": fractions,
        "calculated": calc[:, 0] if was_1d else calc,
        "coefficients": coeff[:, 0] if was_1d and coeff.shape[1] == 1 else coeff,
        "residuals": residual[:, 0] if was_1d else residual,
        "species_names": component_species_names(comp),
    }


def spectroscopy_acid_base_residuals(
    params: np.ndarray,
    dataset: SpectroscopicAcidBaseDataset,
    system_template: AcidBaseSystem,
    fit_options: dict,
) -> np.ndarray:
    system, _consumed = update_system_from_parameter_vector(
        params,
        system_template,
        fit_options,
    )
    pred = predict_spectroscopy_acid_base(dataset, system, fit_options)
    residual = np.asarray(pred["residuals"], dtype=float)
    return residual[np.isfinite(residual)].reshape(-1)


def predict_nmr_acid_base(
    dataset: NMRAcidBaseDataset,
    system: AcidBaseSystem,
    fit_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    opts = dict(fit_options or {})
    if str(dataset.exchange_regime or "fast").lower() != "fast":
        raise NotImplementedError("Only fast-exchange NMR is implemented in v1.")
    pH = np.asarray(dataset.pH, dtype=float).reshape(-1)
    Y, was_1d = _normalise_observed_matrix(dataset.shifts, name="shifts")
    if Y.shape[0] != pH.size:
        raise ValueError("dataset.pH and dataset.shifts have incompatible lengths.")
    comp = system.components[int(opts.get("component_index", 0) or 0)]
    fractions = distribution_fractions_from_pH(pH, component_log_beta(comp))
    calc, coeff, residual = _linear_observable_prediction(
        fractions,
        Y,
        baseline=bool(opts.get("baseline", False)),
        sigma=opts.get("sigma"),
        pure_values=opts.get("species_shifts"),
    )
    return {
        "pH": pH,
        "fractions": fractions,
        "calculated": calc[:, 0] if was_1d else calc,
        "coefficients": coeff[:, 0] if was_1d and coeff.shape[1] == 1 else coeff,
        "residuals": residual[:, 0] if was_1d else residual,
        "species_names": component_species_names(comp),
        "nuclei_labels": list(dataset.nuclei_labels or []),
    }


def nmr_acid_base_residuals(
    params: np.ndarray,
    dataset: NMRAcidBaseDataset,
    system_template: AcidBaseSystem,
    fit_options: dict,
) -> np.ndarray:
    system, _consumed = update_system_from_parameter_vector(
        params,
        system_template,
        fit_options,
    )
    pred = predict_nmr_acid_base(dataset, system, fit_options)
    residual = np.asarray(pred["residuals"], dtype=float)
    return residual[np.isfinite(residual)].reshape(-1)


def _covariance_and_correlation(opt_result: Any, residuals: np.ndarray, n_params: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    jac = getattr(opt_result, "jac", None)
    if jac is None or n_params <= 0:
        return None, None
    J = np.asarray(jac, dtype=float)
    if J.ndim != 2 or J.shape[1] != n_params:
        return None, None
    dof = int(residuals.size) - int(n_params)
    if dof <= 0:
        scale = 1.0
    else:
        scale = float(np.sum(residuals**2)) / float(dof)
    try:
        cov = np.linalg.pinv(J.T @ J, rcond=1e-12) * scale
    except np.linalg.LinAlgError:
        return None, None
    diag = np.diag(cov)
    corr = np.full_like(cov, np.nan, dtype=float)
    good = diag > 0.0
    if np.any(good):
        denom = np.sqrt(np.outer(diag, diag))
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = cov / denom
        corr[~np.isfinite(corr)] = np.nan
    return cov, corr


def _information_criteria(chi_square: float, n_obs: int, n_params: int) -> tuple[float | None, float | None]:
    if n_obs <= 0:
        return None, None
    rss_per_obs = max(float(chi_square) / float(n_obs), 1e-300)
    aic = float(n_obs) * float(np.log(rss_per_obs)) + 2.0 * float(n_params)
    bic = float(n_obs) * float(np.log(rss_per_obs)) + float(n_params) * float(np.log(n_obs))
    return aic, bic


def _build_fit_result(
    opt_result: Any,
    system: AcidBaseSystem,
    residuals: np.ndarray,
    parameter_names: Sequence[str],
) -> AcidBaseFitResult:
    pka = system_pka_values(system)
    log_beta = system_log_beta_values(system)
    residuals = np.asarray(residuals, dtype=float).reshape(-1)
    n_params = len(parameter_names)
    chi = float(np.sum(residuals**2))
    dof = int(residuals.size) - int(n_params)
    red = (chi / float(dof)) if dof > 0 else None
    aic, bic = _information_criteria(chi, int(residuals.size), int(n_params))
    cov, corr = _covariance_and_correlation(opt_result, residuals, n_params)
    values = np.asarray(getattr(opt_result, "x", []), dtype=float).reshape(-1)
    if values.size != n_params:
        values = np.asarray(pka[:n_params], dtype=float)
    stderr = None
    if cov is not None and cov.shape[0] == n_params:
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    table = pd.DataFrame(
        {
            "parameter": list(parameter_names),
            "value": values.tolist(),
            "stderr": (
                stderr.tolist()
                if stderr is not None
                else [np.nan for _ in range(n_params)]
            ),
            "fixed": [False for _ in range(n_params)],
        }
    )
    return AcidBaseFitResult(
        success=bool(getattr(opt_result, "success", False)),
        message=str(getattr(opt_result, "message", "")),
        fitted_pka=[float(v) for v in pka],
        fitted_log_beta=[float(v) for v in log_beta],
        parameter_table=table,
        covariance_matrix=cov,
        correlation_matrix=corr,
        residuals=residuals,
        chi_square=chi,
        reduced_chi_square=red,
        aic=aic,
        bic=bic,
    )


def _default_pka_start(
    system_template: AcidBaseSystem,
    initial_pka: Sequence[float] | None,
) -> np.ndarray:
    if initial_pka is None:
        return np.asarray(system_pka_values(system_template), dtype=float)
    arr = _as_1d_float(initial_pka, name="initial_pka")
    expected = len(system_pka_values(system_template))
    if arr.size != expected:
        raise ValueError(f"Expected {expected} initial pKa values, got {arr.size}.")
    return arr


def fit_spectroscopy_acid_base(
    dataset: SpectroscopicAcidBaseDataset,
    system_template: AcidBaseSystem,
    initial_pka: Sequence[float] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> AcidBaseFitResult:
    opts = dict(fit_options or {})
    x0 = _default_pka_start(system_template, initial_pka)
    bounds = opts.get("bounds", (-5.0, 25.0))
    parameter_names = [f"pKa{i+1}" for i in range(x0.size)]

    def residual(theta: np.ndarray) -> np.ndarray:
        return spectroscopy_acid_base_residuals(theta, dataset, system_template, opts)

    opt = least_squares(residual, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    fitted_system = clone_system_with_pka(system_template, opt.x)
    return _build_fit_result(opt, fitted_system, residual(opt.x), parameter_names)


def fit_nmr_acid_base(
    dataset: NMRAcidBaseDataset,
    system_template: AcidBaseSystem,
    initial_pka: Sequence[float] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> AcidBaseFitResult:
    opts = dict(fit_options or {})
    x0 = _default_pka_start(system_template, initial_pka)
    bounds = opts.get("bounds", (-5.0, 25.0))
    parameter_names = [f"pKa{i+1}" for i in range(x0.size)]

    def residual(theta: np.ndarray) -> np.ndarray:
        return nmr_acid_base_residuals(theta, dataset, system_template, opts)

    opt = least_squares(residual, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    fitted_system = clone_system_with_pka(system_template, opt.x)
    return _build_fit_result(opt, fitted_system, residual(opt.x), parameter_names)


def global_acid_base_residuals(
    params: np.ndarray,
    global_fit: GlobalAcidBaseFit,
) -> np.ndarray:
    shared_names = list(global_fit.shared_parameters or [])
    opts: dict[str, Any] = {}
    if shared_names:
        opts["parameter_names"] = shared_names
    system, _consumed = update_system_from_parameter_vector(
        params,
        global_fit.system_template,
        opts,
    )
    residual_blocks: list[np.ndarray] = []

    if global_fit.potentiometry_experiments:
        from hmfit_core.potentiometry import potentiometry_residuals

        pot_opts = dict((global_fit.local_parameters or {}).get("potentiometry") or {})
        for idx, experiment in enumerate(global_fit.potentiometry_experiments):
            local_opts = dict(pot_opts)
            local_opts.update((global_fit.local_parameters or {}).get(f"potentiometry_{idx}") or {})
            residual_blocks.append(
                potentiometry_residuals(np.asarray([], dtype=float), experiment, system, local_opts)
            )

    spec_opts = dict((global_fit.local_parameters or {}).get("spectroscopy") or {})
    for idx, dataset in enumerate(global_fit.spectroscopy_datasets or []):
        local_opts = dict(spec_opts)
        local_opts.update((global_fit.local_parameters or {}).get(f"spectroscopy_{idx}") or {})
        residual_blocks.append(
            spectroscopy_acid_base_residuals(np.asarray([], dtype=float), dataset, system, local_opts)
        )

    nmr_opts = dict((global_fit.local_parameters or {}).get("nmr") or {})
    for idx, dataset in enumerate(global_fit.nmr_datasets or []):
        local_opts = dict(nmr_opts)
        local_opts.update((global_fit.local_parameters or {}).get(f"nmr_{idx}") or {})
        residual_blocks.append(
            nmr_acid_base_residuals(np.asarray([], dtype=float), dataset, system, local_opts)
        )

    if not residual_blocks:
        return np.asarray([], dtype=float)
    return np.concatenate([np.asarray(block, dtype=float).reshape(-1) for block in residual_blocks])


def fit_global_acid_base(
    global_fit: GlobalAcidBaseFit,
    initial_pka: Sequence[float] | None = None,
    fit_options: Mapping[str, Any] | None = None,
) -> AcidBaseFitResult:
    opts = dict(fit_options or {})
    x0 = _default_pka_start(global_fit.system_template, initial_pka)
    parameter_names = list(global_fit.shared_parameters or [f"pKa{i+1}" for i in range(x0.size)])
    bounds = opts.get("bounds", (-5.0, 25.0))

    def residual(theta: np.ndarray) -> np.ndarray:
        return global_acid_base_residuals(theta, global_fit)

    opt = least_squares(residual, x0, bounds=bounds, xtol=1e-10, ftol=1e-10, gtol=1e-10)
    fitted_system = clone_system_with_pka(global_fit.system_template, opt.x)
    return _build_fit_result(opt, fitted_system, residual(opt.x), parameter_names)


__all__ = [
    "AcidBaseSpecies",
    "AcidBaseComponent",
    "AcidBaseSystem",
    "SpeciesDistributionResult",
    "SpectroscopicAcidBaseDataset",
    "NMRAcidBaseDataset",
    "GlobalAcidBaseFit",
    "AcidBaseFitResult",
    "pka_to_log_beta",
    "log_beta_to_pka",
    "distribution_fractions_from_pH",
    "component_log_beta",
    "component_charges",
    "component_species_names",
    "system_pka_values",
    "system_log_beta_values",
    "clone_system",
    "clone_system_with_log_beta",
    "clone_system_with_pka",
    "make_simple_acid_base_system",
    "build_solver_inputs_for_component",
    "simulate_species_vs_pH",
    "update_system_from_parameter_vector",
    "predict_spectroscopy_acid_base",
    "spectroscopy_acid_base_residuals",
    "fit_spectroscopy_acid_base",
    "predict_nmr_acid_base",
    "nmr_acid_base_residuals",
    "fit_nmr_acid_base",
    "global_acid_base_residuals",
    "fit_global_acid_base",
]
