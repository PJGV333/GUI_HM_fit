# Validation harness — provisional

This is **not** the Phase 2 confirmatory campaign. It currently implements one
production-backed scenario only: UV-Vis 1:1, i.i.d. normal response noise and
analytic uncertainty, using the existing spectroscopy processor and Phase 1
synthetic generator. It uses a two-sided Student-t interval based on the
production VarPro residual degrees of freedom. The fit uses the existing default
soft-penalty absorptivity estimator; analytic uncertainty remains unconstrained,
so its coverage is a research question rather than a guaranteed property.
Other techniques, noise models and bootstrap methods are rejected, not simulated.

For a bounded exploratory pilot (never label its coverage confirmatory):

```sh
QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  .venv/bin/python -m validation.run_campaign validation/configs/uvvis_11_pilot.json "$TMPDIR/hmfit-uvvis-pilot"
.venv/bin/python -m validation.analyze "$TMPDIR/hmfit-uvvis-pilot/results.csv"
```

The manifest includes the complete configuration, Git commit, SHA-256 of tracked
Python sources plus `validation/*.py`, dependency versions and backend. Each
replicate has a seed derived from the base seed and replicate index, a
checksummed, atomically replaced checkpoint and a row with convergence, estimate,
SE, interval, runtime and fit evaluation count. `results.csv` is sorted by index.
Resume by re-running the same command; mismatched code, configuration or versions
and corrupted checkpoints are refused. A source change detected while the run is
active stops checkpointing. `workers` uses spawned processes, with BLAS thread
environment set to one before their creation. Run only **one coordinator per
output directory**: cross-coordinator locking is not yet implemented.

The summary reports fit failure fraction and, conditional on successful fits,
bias, RMSE, Monte Carlo SD, mean reported SE, SE/SD, interval widths, coverage,
coverage Monte Carlo standard error and a Wilson interval. It does not yet report
Monte Carlo errors for every metric. Raw data stay outside the repository; do not
commit large run directories. Unsupported confirmatory cells remain incomplete.

## Paired-estimator bias experiments (`validation/bias_experiment.py`)

Reproducible Monte-Carlo comparison of the three absorptivity projectors (free
OLS, production soft penalty at given mu, hard NNLS) on identical datasets, with
an independent closed-form equilibrium reference and exact scalar argmin
(grid + bounded golden-section), so optimizer noise cannot contaminate paired
differences. Checkpointed per replicate (JSONL) and resumable.

```sh
QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m validation.bias_experiment validation/configs/bias/pilot40.json "$TMPDIR/bias-pilot40"
```

Scenario definitions live in `validation/configs/bias/` (21 diagnostic cells:
noise levels, mu sweep, design sizes, K values, absorptivity regimes). Curated
summaries and the pilot paired rows are committed under
`validation/results/bias/` (small; full raw campaign directories remain out of
repo). Figures are regenerable with `python -m validation.make_bias_figures`
into `docs/figures/uvvis_bias/`. Mechanism evidence (expected profiled RSS vs
theta): `python -m validation.mechanism_profile`; analytic-SE calibration:
`python -m validation.se_calibration`. Interpretation and conclusions:
`docs/UVVIS_BIAS_INVESTIGATION.md`.

## Uncertainty-method comparison (`validation/uncertainty_experiment.py`)

Runs the full Stage F zoo (analytic, HC1, linearized wild per element / per
spectrum, one-step wild per spectrum / per wavelength, and three full-refit
bootstraps) on every replicate of a scenario, with percentile **and** basic
(reversed-percentile) intervals. The production-backed campaign adapter
(`run_campaign.py`) additionally accepts `estimator`: `soft_penalty` (default),
`ols` (production path with mu=0) or `nnls` (hard NNLS). Noise models `iid`,
`hetero` (per-point ramp) and `ar1` (per-spectrum AR(1)) are available in
`bias_experiment.dataset`. Frozen intermediate designs:
`validation/configs/unc/`; curated results land in
`validation/results/unc/`. Method definitions, design freeze and acceptance
criteria: `docs/STATISTICAL_VALIDATION.md`; numeric tables:
`docs/MONTE_CARLO_RESULTS.md`.

## Stage G technique campaigns (`nmr_experiment.py`, `ph_experiment.py`, `kinetics_experiment.py`)

Monte Carlo harnesses over the production paths for the other techniques:

- `nmr_experiment.py` — 1H NMR 1:1 shifts through `process_nmr_data`
  (configs `validation/configs/nmr/`, results `validation/results/nmr/`).
- `ph_experiment.py` — acid-base equilibria: NMR shift titrations and pH
  potentiometry through `run_acid_base` + production analytic errors
  (`validation/configs/ph/`, `validation/results/ph/`).
- `kinetics_experiment.py` — first order, consecutive, reversible and
  second-order mechanisms through `hmfit.kinetics.fit_global`
  (`validation/configs/kin/`, `validation/results/kin/`).

All three are resumable (per-replicate JSONL), record a manifest (config
sha256, commit, versions) and use frozen `SeedSequence([base_seed, r])`
seeds. Findings (including the structural non-identifiability of reversible
A<->B with free spectra): `docs/MONTE_CARLO_RESULTS.md` sections 7-9.
