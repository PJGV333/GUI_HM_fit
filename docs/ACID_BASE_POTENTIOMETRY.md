# Acid-base and potentiometric analysis

HM Fit includes an initial acid-base / potentiometry workflow built on the same
core conventions used by the existing equilibrium modules: chemical constants
are cumulative log10 formation/protonation constants, and pKa values are a
special case of those constants.

## Accepted data

The PySide6 tab imports CSV/TXT files and Excel workbooks (`.xlsx`).
For Excel files, select the worksheet in the **DATA / INPUT** section before
fitting.

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

## GUI model definition

The Acid-base / Potentiometry tab follows the same workflow style as the
Spectroscopy and NMR tabs:

1. Use **DATA / INPUT** to choose the dataset type and import a CSV, TXT, or
   XLSX file.
2. Use **Model** to define the acid-base system.
3. Use **Optimization** to review fitted/local parameters, including `pKw`.
4. Use **Plots** to select which plot pages should be rendered in the right
   panel.
5. Use **Errors** to inspect the parameter table and covariance/correlation
   matrices when available.

The **Model** tab contains a **Components** table. Each component defines a
base form such as `L`, its total concentration, base charge, number of
protonation steps, and initial pKa or `log_beta` values. The pKa field accepts
comma-separated or semicolon-separated values, for example `4.5, 8.9` or
`4.5; 8.9`.

For a ligand `L` with base charge `z` and `n` protonation steps, HM Fit
generates:

```text
L, HL, H2L, ..., HnL
```

with charges:

```text
z, z + 1, z + 2, ..., z + n
```

and cumulative constants:

```text
log_beta_n = pKa_1 + pKa_2 + ... + pKa_n
```

Examples:

- Monoprotic `HL/L-`: base charge `-1`, steps `1`, pKa `5.20`.
- Diprotic `H2L/L2-`: base charge `-2`, steps `2`, pKa `4.50, 8.90`.
- Triprotic `H3L/L3-`: base charge `-3`, steps `3`, pKa `2.10, 6.70, 10.20`.

The generated **Species** table shows species name, parent component,
`h_count`, charge, cumulative `log_beta`, and stepwise pKa. It is automatically
generated from the component table for simple, polyprotic, and multiple
component models. In **Custom species table** mode, the species table can be
edited directly.

The current GUI model types are:

- Simple monoprotic acid/base.
- Polyprotic ligand.
- Multiple acid-base components.
- Custom species table.
- Coupled acid-base / host-guest model, shown as a future/experimental option.

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

For potentiometric data, HM Fit uses the volume column as the independent
variable, applies dilution and strong ion contributions, and solves pH by
electroneutrality with water autoionization. The **Optimization** tab includes
`pKw`, with default `14.0000`, and internally uses:

```text
Kw = 10^(-pKw)
```

`pKw` affects potentiometric electroneutrality calculations. For spectroscopy
and NMR acid-base datasets, measured pH is imposed directly, so `pKw` is
accepted in the configuration but does not affect species fractions in v1.

The **Potentiometric titration model** group contains the initial volume,
titrant concentration, titrant type, optional manual strong ion charge,
volume-offset/blank correction, pH bounds, and a small custom titrant table for
future non-strong-acid/base titrants.

## Spectroscopy fitting

At each pH, the observed signal is a linear combination of species fractions:

```text
S_obs(pH) = sum_i alpha_i(pH) S_i
```

For matrices, each wavelength is solved as an independent linear observable.
The species signals are solved by linear least squares for each trial pKa
(variable projection).

Spectroscopy uses measured pH as an imposed independent variable and does not
solve electroneutrality in v1.

## Proton NMR fitting

The initial NMR implementation assumes fast exchange:

```text
delta_obs(pH) = sum_i alpha_i(pH) delta_i
```

Multiple proton labels are fitted simultaneously with shared pKa values and
local limiting shifts.

NMR uses measured pH as an imposed independent variable and assumes fast
exchange in v1:

```text
delta_obs = sum_i alpha_i delta_i
```

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
