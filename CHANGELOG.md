# Changelog

All notable user-visible changes. Versions follow the GitHub releases
(`v0.1.0-beta.1`, `v0.1.0-beta.2`, `v0.2.0`). Details, evidence and test references:
`docs/STABILITY_PHASE1_REPORT.md`, `docs/STABILITY_PHASE2_REPORT.md`.

## [Unreleased]

### Changed
- **Acid-base: default initial pKa guesses are now a descending ladder** (diprotic 7, 5;
  triprotic 9, 7, 5; previously 5, 7, 9). In HM Fit's convention pKa1 = log β1 is the most
  basic site, so a non-cooperative ligand has pKa1 > pKa2 > …; the old ascending defaults
  started fits in the inverted (cooperative) ordering. Approved by the author (Phase 2).
- Acid-base: every pKa label names its equilibrium, e.g. `pKa1 (HL ⇌ L + H⁺)` (pKa table,
  fit report, error table). The equation-editor examples now use pKa1 > pKa2.
- Acid-base exports: `Equilibrium` column in the `pKa` sheet and a `pKa_convention` sheet.
- Kinetics: the "Algorithm" combo offers only the method that actually runs
  ("Trust-region reflective (bounded)"); saved configurations with "multistart" keep their
  10 starts.
- Error tables: headers name the real parameter (pKa, log β or log10K); undefined values are
  shown as "—" with the reason in the tooltip; exports keep NaN plus a `Note` column.

### Fixed
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

### Infrastructure
- CI (`tests` workflow) on Linux (GUI offscreen) and Windows (non-GUI), Python 3.11 and 3.14;
  release builds require the test job; exact tested versions in `constraints.txt`; minimum
  Python 3.11.
