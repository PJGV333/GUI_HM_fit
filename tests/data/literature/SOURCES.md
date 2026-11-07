# Literature benchmark — sources

The numbers in this folder were extracted from the Supporting Information (SI) of the two
articles below with `validation/literature_extract.py` (two independent passes, pdfplumber and
pdftotext, compared automatically). **The data belong to their authors and are used here only
for the internal validation of HM fit.** The PDFs themselves are not part of the repository.

- **[R09a]** S. Roelens, A. Vacca, C. Venturi, "Binding of Ionic Species: A General Approach To
  Measuring Binding Constants and Assessing Affinities", *Chem. Eur. J.* **2009**, *15*,
  2635–2644. DOI: [10.1002/chem.200802298](https://doi.org/10.1002/chem.200802298).
  SI file: `chem_200802298_sm_miscellaneous_information.pdf`.
- **[R09b]** S. Roelens, A. Vacca, O. Francesconi, C. Venturi, "Ion-Pair Binding: Is Binding Both
  Binding Better?", *Chem. Eur. J.* **2009**, *15*, 8296–8302. DOI:
  [10.1002/chem.200900342](https://doi.org/10.1002/chem.200900342).
  SI file: `chem_200900342_sm_miscellaneous_information.pdf`.

Reference results were computed by the authors with HypNMR (versions 16.11.2006 and 19.06.2007,
as printed on each results page) and the "BC50 Calculator".

## Cases

| Case | Source | SI pages (data; results) | Article table | Content |
|---|---|---|---|---|
| `r09a-r1-11-me` | R09a | S4; S5 | — | Receptor 1 + TMAC, 1:1 two-reagent model (R, G), Me only |
| `r09a-r1-11-3n` | R09a | S4; S6 | — | Same data, 1:1 model with 3 nuclei (deliberately poor fit) |
| `r09a-r1-3sp` | R09a | S7; S8 | — | Three reagents (R, Q, X), RQX, RX, QX; TMAC 0.131 / 1.06 mM |
| `r09a-r1-5sp` | R09a | S7; S9 | Table 1 | Same data; RQX, RX, R2, R2X, QX (final model of receptor 1) |
| `r09a-qx-dil` | R09a | S10; S11 | — | TMAC dilution, Q + X ⇌ QX, Me |
| `r09a-p2-058` | R09a | S12; S13–S14 | Table 2 | Receptor 2 (P), TMAC 0.15 / 0.58 mM, 7 nuclei, 6 species, β(P2) fixed |
| `r09a-p2-106` | R09a | S17; S18–S19 | — | Receptor 2 (P), TMAC 0.15 / 1.06 mM |
| `r09b-r-qx` | R09b | S9; S10 | Table 1 | Repeats `r09a-r1-5sp` (data and results reported from R09a) |
| `r09b-b-qx` | R09b | S13; S14 | Table 2 | Receptor B, Me only, β(QX) fixed, 2 titrations (45 values / 50 spectra) |
| `r09b-d-qx` | R09b | S15; S16–S17 | Table 3 | Receptor D, 6 nuclei, β(QX) and β(D2) fixed, DQ shifts fixed |
| `r09b-spec-r` | R09b | S20 | (Tables 1) | BC50 (X vs R) and species distribution, T_Q = T_X |
| `r09b-spec-b` | R09b | S21 | (Table 2) | BC50 (Q vs B) and species distribution, T_X = T_Q |
| `r09b-spec-d-x` | R09b | S22 | (Table 3) | BC50 (X vs D) and species distribution, T_Q = T_X |
| `r09b-spec-d-q` | R09b | S23 | (Table 3) | BC50 (Q vs D) and species distribution, T_X = T_Q |

Each `case.json` repeats its reference, DOI and SI pages. Plots in the SI pages were not used.

## Files

- `<case>/data.csv`: one row per spectrum: `dataset` (titration index), total concentration of
  each reagent `T_<reagent>` in mol/L, and one column per nucleus in ppm, **as printed** (the
  printed strings are kept so that the printed precision is known). Missing values are empty.
- `<case>/case.json`: source, reagents, species (refined or fixed, with value), nuclei with
  their reagent and the status of each limiting shift (refined, fixed with value, absent),
  printed counts, the HypNMR degrees-of-freedom identity and the reference results
  (log β ± SD, σ, RMS weighted residual, χ², limiting shifts ± error, correlations × 1000).
- `<case>/baseline.json`: HM fit's own results (internal regression baseline; see
  `docs/LITERATURE_BENCHMARK.md`).
- `<spec case>/table.csv`: speciation table as printed.
