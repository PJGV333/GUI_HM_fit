# Changelog

All notable user-visible changes. Versions follow the GitHub releases
(`v0.1.0-beta.1`, `v0.1.0-beta.2`, `v0.2.0`). Details, evidence and test references:
`docs/STABILITY_PHASE1_REPORT.md`, `docs/STABILITY_PHASE2_REPORT.md`.

## [Unreleased]

### Changed
- Spectroscopy: `explicit_lm` is an opt-in optimizer with convergence diagnostics and
  active-bound uncertainty warnings. Powell remains the default for new and saved
  configurations without an optimizer. Depending on the start and convergence, LM
  may produce a different optimum; see `tests/test_phase31_spectro_lm.py` and the
  phase 3 report. The absorptivity estimator is unchanged.
- Kinetics: automatic initial rate estimates now use the experiment duration and reactant
  concentrations in the data's units; rate constants and Arrhenius A use Log10 by default.
  Existing user entries and saved settings are retained. Initial guesses can change fitted
  results when the previous start reached another minimum. The parameter table and fit report
  display the dataset units.
- Kinetics: a mechanism sharing a rate constant across reactions of different orders is
  rejected during validation because its units are inconsistent. Saved configurations still
  open and display the validation error; such fits can no longer run.
- **Acid-base: default initial pKa guesses are now a descending ladder** (diprotic 7, 5;
  triprotic 9, 7, 5; previously 5, 7, 9). In HM Fit's convention pKa1 = log β1 is the most
  basic site, so a non-cooperative ligand has pKa1 > pKa2 > …; the old ascending defaults
  started fits in the inverted (cooperative) ordering. Approved by the author (Phase 2).
- Acid-base: every pKa label names its equilibrium, e.g. `pKa1 (HL ⇌ L + H⁺)` (pKa table,
  fit report, error table). The equation-editor examples now use pKa1 > pKa2.
- Potentiometry reports and the new XLSX `Fit_summary` sheet show pKa equilibria and analytic
  standard errors (or `fixed`), signal-unit RMS, point count, degrees of freedom and residual
  autocorrelation before additional statistics. With σ = 1, chi-square is the SSE. The first
  zero titrant volume is now described as normal; fitted numbers are unchanged.
- Acid-base exports: `Equilibrium` column in the `pKa` sheet and a `pKa_convention` sheet.
- Kinetics: the "Algorithm" combo offers only the method that actually runs
  ("Trust-region reflective (bounded)"); saved configurations with "multistart" keep their
  10 starts.
- Error tables: headers name the real parameter (pKa, log β or log10K); undefined values are
  shown as "—" with the reason in the tooltip; exports keep NaN plus a `Note` column.

### Fixed
- Explicit LM utility: a parameter whose lower and upper bounds coincide is excluded
  from the projected scaled gradient. Previously it could report a misleadingly
  large final gradient even after optimizing all free parameters. The utility is
  not yet connected to the spectroscopy or NMR fit paths; default fits are unchanged.
- Acid-base: a warning is shown (values are never reordered) when entered or fitted pKa are in
  inverted order.
- Spectroscopy/NMR: a non-numeric initial value or bound (e.g. `3,5` with a decimal comma) is
  rejected with a readable message; it used to become 1.0 (value) or "no limit" (bound) silently.
- Potentiometry: fixing a pKa no longer breaks the error analysis; the fixed pKa is reported as
  fixed (SE 0).
- Uncertainty conventions in all error engines: SE = 0 only for fixed parameters; NaN for
  non-identifiable parameters, ν ≤ 0 or failures; correlations involving fixed or
  non-identifiable parameters are NaN (were 0 or 1). NMR error analysis no longer fails with a
  shape mismatch when a constant is not identifiable.
- Kinetics graph API: `KineticGraph.add_kinetic_reaction` failed on Python < 3.14.

### Fixed (Phase 3, Eyring units)
- Eyring rates now account for recognized time and concentration units, including the 1 M
  standard state. Previously, ΔS‡ was shifted by +R·ln 60 ≈ 34 J·mol⁻¹·K⁻¹ for
  times in minutes and −R·ln 1000 ≈ −57 J·mol⁻¹·K⁻¹ for bimolecular reactions with
  concentrations in mM; ΔH‡ does not change from these unit factors. Empty units retain
  the previous s/M assumption with a warning; unrecognized units are rejected.

### Fixed (Phase 3, kinetics dataset units)
- Global kinetics fits reject incompatible recognized time or concentration units; no numerical
  conversion is performed. Unknown or empty units generate a warning. Older projects still open.
  Dataset metadata now has an optional concentration unit; previously it was not recorded.
  Previously accepted mixed-unit fits may no longer run.

### Fixed (Phase 3, kinetics JAX)
- The experimental JAX integrator returned only the initial and final concentrations instead
  of one row per sample; it now returns the complete RK4 trajectory. JAX fits that previously
  failed on ordinary grids can now run, and their results change.
- JAX linear projection now uses the same rank cutoff as SciPy; fitted constants and errors
  may change for rank-deficient concentration matrices.
- JAX is labeled experimental, with a warning that it uses one RK4 step per sampling interval
  without error control; SciPy remains the reference and default.

### Fixed (kinetics and acid-base review, November 2026 — details in `docs/REVIEW_KINETICS_POTENTIOMETRY.md`)
- **Kinetics: mechanisms with conservation laws no longer return the initial guess as a fit.**
  With more dynamic species than independent reactions + 1 (e.g. `A + B -> C -> D` at unequal
  concentrations) the projection kept or dropped a round-off direction erratically; the
  optimiser stopped at the starting values ("Fit complete", k1 = 20.000 for a true 30). The
  projection and the count of linear parameters now use a rank cutoff of 1e-10.
  **Fitted constants and standard errors of such mechanisms change.**
- **Kinetics: the result no longer depends on the units of the signal.** With small signals
  (concentrations in M, weak absorbances) the fit could stop early or at the initial values.
- Kinetics: a fit that did not converge or did not move from the initial values is reported as
  not reliable instead of "Fit complete".
- **Kinetics (GUI): a global fit keeps the concentrations of each dataset.** Pressing Process
  copied the concentrations of the displayed dataset to all datasets.
- **Kinetics (GUI): initial and fixed concentrations are no longer rounded to 6 decimals**
  (2.5e-7 M became 0); scientific notation is accepted.
- Kinetics: Multi-start with the default bounds (min 0, no max) aborted with "Initial value
  outside its bounds"; random starts are now drawn in decades around the initial value, or
  log-uniformly between wide positive bounds, and a failed random start is discarded and
  counted. Seeds therefore give different starting points than before.
- Acid-base (GUI): the Data type selector is visible again and is guessed from the column names;
  spectroscopy-vs-pH and NMR-vs-pH analyses could not be selected in the interface.

### Added
- Kinetics: warning when exchanging two rate constants gives the same fit (slow-fast ambiguity
  of consecutive steps).
- Kinetics and acid-base: lag-1 autocorrelation of the residuals in the report, with a warning
  when it shows unexplained structure (all uncertainty estimates are too small in that case).
- Kinetics (GUI): mechanism examples, concentrations used per dataset in the report, left panel
  sized to its content (also in Acid-base), readable fit-summary table.

### Infrastructure
- CI (`tests` workflow) on Linux (GUI offscreen) and Windows (non-GUI), Python 3.11 and 3.14;
  release builds require the test job; exact tested versions in `constraints.txt`; minimum
  Python 3.11.
