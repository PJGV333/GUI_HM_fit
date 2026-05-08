# Acid-base and potentiometric analysis

HM Fit includes an initial acid-base / potentiometry workflow built on the same
core conventions used by the existing equilibrium modules: chemical constants
are cumulative log10 formation/protonation constants, and pKa values are a
special case of those constants.

## Accepted data

The PySide6 tab imports CSV files.

Potentiometry can use direct pH:

```csv
volume_mL,pH
0.000,3.214
0.100,3.256
0.200,3.301
```

or electrode potential in mV:

```csv
volume_mL,E_mV
0.000,245.2
0.100,243.8
0.200,242.1
```

Spectroscopy at one wavelength:

```csv
pH,signal
2.00,0.143
2.50,0.151
3.00,0.174
```

Spectral matrices use one pH column followed by wavelength columns:

```csv
pH,300,301,302,303,304
2.00,0.11,0.12,0.13,0.15,0.16
2.50,0.12,0.13,0.15,0.17,0.18
```

Fast-exchange proton NMR:

```csv
pH,H1,H2,H3
2.00,8.12,7.45,6.98
2.50,8.08,7.42,6.95
3.00,8.01,7.38,6.90
```

## pKa and log_beta convention

For cumulative protonation:

```text
L + nH+ <=> H_nL
beta_n = [H_nL] / ([L] [H+]^n)
```

HM Fit assumes `log_beta_0 = 0`. Stepwise pKa values are:

```text
pKa_n = log_beta_n - log_beta_(n-1)
```

Therefore `pKa = [4.0, 7.0]` corresponds to
`log_beta = [4.0, 11.0]`.

## Species diagrams

For a component with species `L, HL, H2L, ...`, fractions at imposed pH are:

```text
alpha_i = beta_i [H+]^i / sum_j(beta_j [H+]^j)
```

Rows in the distribution table sum to 1 for each component. The GUI plots
fraction vs pH and, for potentiometry, fraction vs titrant volume.

## Potentiometric fitting

The v1 potentiometry model simulates pH by solving electroneutrality at each
titration point. Dilution is included:

```text
C_i,total = (C_i,0 V0 + C_i,titrant Vadd) / (V0 + Vadd)
```

The default monoprotonic acid model uses `L-` and neutral `HL`, titrated with a
strong base that contributes a spectator cation. A strong acid titrant
contributes a spectator anion. The ideal electrode model is:

```text
E = E0 + S pH
```

`E0` and `S` can be fixed or fitted for EMF data.

## Spectroscopy fitting

At each pH, the observed signal is a linear combination of species fractions:

```text
S_obs(pH) = sum_i alpha_i(pH) S_i
```

For matrices, each wavelength is solved as an independent linear observable.
The species signals are solved by linear least squares for each trial pKa
(variable projection).

## Proton NMR fitting

The initial NMR implementation assumes fast exchange:

```text
delta_obs(pH) = sum_i alpha_i(pH) delta_i
```

Multiple proton labels are fitted simultaneously with shared pKa values and
local limiting shifts.

## Global multimodal fits

The core module exposes residual combiners for potentiometry, spectroscopy, and
NMR datasets. Chemical parameters such as pKa/log_beta can be shared, while
instrumental parameters such as electrode constants, optical species signals,
and limiting NMR shifts remain local to each technique.

## Export

Excel export includes pKa, log_beta, fitted/local parameters, experimental vs
calculated data, residuals, species distributions, and covariance/correlation
matrices when available. CSV export from the GUI writes the main
experimental-vs-calculated table.

## Current limitations

This v1 implementation assumes ideal solutions or constant ionic strength.
Activities are approximated by concentrations. Activity corrections should be
added as a later extension.

Other explicit limitations:

- Fixed or ignored ionic strength.
- Ideal electrode with optional fitted slope.
- Linear spectroscopy in species fractions.
- Proton NMR fast-exchange regime only.
- No automatic atmospheric CO2 correction.
- No advanced competitive salt binding unless the general HM Fit equilibrium
  solver is used through a future acid-base graph wrapper.

Planned extensions include Debye-Huckel, Davies, SIT, variable ionic strength,
Gran calibration, alkalinity analysis, complex polyprotic systems,
host-guest/protonation coupling, and tighter coupling to the existing UV-Vis,
fluorescence, and NMR observation models.
