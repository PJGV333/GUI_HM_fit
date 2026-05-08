# Acid-base and Potentiometry

The Acid-base / Potentiometry module now follows the same model-definition
philosophy as the Spectroscopy and NMR modules:

1. Import data in **DATA / INPUT**.
2. Define the chemical model in **Model**.
3. Review parameter guesses and local settings in **Optimization**.
4. Select output plots in **Plots**.
5. Compute uncertainty diagnostics in **Errors**.

The model can be defined in two ways:

- **Matriz estequiometrica**
- **Editor de ecuaciones**

Both modes populate the same canonical internal representation,
`cfg["acid_base_model"]`, which is then consumed by `run_acid_base`.

## Accepted data

The tab accepts `CSV`, `TXT`, and `XLSX`.

Potentiometry:

```csv
volume_mL,pH
0.000,3.214
0.100,3.256
0.200,3.301
```

or EMF:

```csv
volume_mL,E_mV
0.000,245.2
0.100,243.8
0.200,242.1
```

Spectroscopy:

```csv
pH,signal
2.00,0.143
2.50,0.151
3.00,0.174
```

NMR:

```csv
pH,H1,H2,H3
2.00,8.12,7.45,6.98
2.50,8.08,7.42,6.95
3.00,8.01,7.38,6.90
```

## Matrix workflow

In **Matriz estequiometrica** you define:

- Number of components
- Number of species
- Component metadata
- Species metadata
- Stoichiometric matrix

### Components table

The components table includes:

- `Component name`
- `Role`
- `Analytical concentration`
- `Charge`
- `Is proton`
- `Is titrant`
- `Is background/spectator`
- `Fixed concentration`

The proton component `H` is special.

- In spectroscopy and NMR, pH is imposed by the dataset, so `H` is conceptual.
- In potentiometry, `H` participates through electroneutrality and `Kw`, not as a normal conserved analytical component.

### Species table

The species table includes:

- `Species name`
- `Charge`
- `h_count`
- `Include`
- `Observable`
- `Fixed`
- `Non-observable / non-absorbing`
- `Parent component or group`

### Monoprotic example

Components:

- `L`
- `H`

Species:

- `L`
- `HL`

Stoichiometric matrix:

```text
        L   HL
L       1   1
H       0   1
```

Typical metadata:

- `L`: charge `-1`, `h_count = 0`, `log_beta = 0`
- `HL`: charge `0`, `h_count = 1`, `log_beta = pKa1`

### Diprotic example

Components:

- `L`
- `H`

Species:

- `L`
- `HL`
- `H2L`

Stoichiometric matrix:

```text
        L   HL   H2L
L       1   1    1
H       0   1    2
```

Typical metadata:

- `L`: charge `-2`, `h_count = 0`, `log_beta = 0`
- `HL`: charge `-1`, `h_count = 1`, `log_beta = pKa1`
- `H2L`: charge `0`, `h_count = 2`, `log_beta = pKa1 + pKa2`

## Equation editor workflow

The equation editor accepts acid-base equations and converts them into the same
component/species/matrix model.

### Example 1

```text
L + H <=> HL ; pKa=5.20
```

### Example 2

```text
L + H <=> HL ; pKa=4.50
HL + H <=> H2L ; pKa=8.90
```

### Example 3

```text
L + H <=> HL ; logB=4.50
L + 2H <=> H2L ; logB=13.40
```

The parser recognizes:

- `H`, `H+`, `h`, `proton`
- `pKa`, `pka`
- `logB`, `log_beta`, `logbeta`
- Integer stoichiometric coefficients
- Explicit charges such as `L(-2)` or `H(1)`

If absolute charges are omitted, the parser uses a consistent relative ladder
that can be corrected in the species table before fitting potentiometric data.

## pKa and log_beta

For cumulative protonation:

```text
L + nH+ <=> H_nL
beta_n = [H_nL] / ([L][H+]^n)
```

HM Fit uses:

```text
log_beta_0 = 0
```

and the relation:

```text
log_beta_1 = pKa1
log_beta_2 = pKa1 + pKa2
log_beta_3 = pKa1 + pKa2 + pKa3
```

Conversely:

```text
pKa1 = log_beta1
pKa2 = log_beta2 - log_beta1
pKa3 = log_beta3 - log_beta2
```

The GUI shows both conventions so the user can verify the ladder.

## Templates

The **Load template** control provides presets:

- `Simple monoprotic acid/base`
- `Diprotic ligand`
- `Triprotic ligand`
- `Multiple acid-base components`
- `Custom acid-base system`
- `Coupled acid-base / host-guest model (future)`

The presets are convenience tools only. They no longer define the fitting
engine; the fitting engine is always the canonical matrix/equation model.

## Optimization tab

The **Optimization** tab is generated from the model definition.

For a diprotic system in `pKa` mode it creates:

- `pKa1`
- `pKa2`

For `log_beta` mode it creates:

- `log_beta1`
- `log_beta2`

Each parameter row includes:

- `Initial value`
- `Min`
- `Max`
- `Fixed`
- `Linked species`
- `Description`

Potentiometry also keeps local settings such as:

- `electrode_e0`
- `electrode_slope`
- `analyte concentration`
- `titrant concentration`
- `volume offset`
- `pKw`

## Why H is special

The proton row exists in the model editor because it is chemically meaningful
for the stoichiometric matrix and for `h_count`.

Internally:

- Spectroscopy/NMR use the measured pH directly.
- Potentiometry solves pH from electroneutrality, dilution, strong-ion
  contribution, species charges, and water autoionization.

That is why `H` appears in the editor but is not treated as a normal conserved
analytical component in the standard potentiometric workflow.

## pKw and when it matters

`pKw` is entered in **Optimization** under **Water autoionization**.

```text
Kw = 10^(-pKw)
```

- In potentiometry, `pKw` affects electroneutrality and therefore calculated pH.
- In spectroscopy and NMR v1, pH is imposed by the dataset, so `pKw` does not
  alter species fractions.

Default:

```text
pKw = 14.0000
```

## Validation

Before fitting, the module validates the acid-base model and warns about common
issues such as:

- Missing species names
- Missing charges
- Missing `h_count`
- Non-consecutive protonation ladders
- pKa values far outside the experimental pH range
- Two pKa values too close relative to the pH sampling

Potentiometry requires valid charge information for all included species.

## Backward compatibility

Legacy simple configurations still work:

- `component_name`
- `pka_initial`
- `analyte_concentration`
- `base_charge`

If `cfg["acid_base_model"]` is missing, `run_acid_base` still builds a simple
monoprotic or polyprotic system from those old fields.
