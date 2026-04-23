# HM fit (PySide6 Version)

Pedro Jancarlo Gomez Vega*, José Octavio Juárez Sánchez, Ramón Moreno Corral, Felipe Medrano Valenzuela, David Octavio Corona Martínez, and Karen L. Ochoa Lara*

Grupo de Química Supramolecular, Universidad de Sonora

`* Corresponding authors`

HM fit is a PySide6 desktop application for the quantitative analysis and fitting of host-guest and related chemical models from spectroscopy, nuclear magnetic resonance (NMR), and kinetics data. The current graphical interface is organized into three modules, `Spectroscopy`, `NMR`, and `Kinetics`, which distinguish equilibrium-style analysis of spectroscopic and chemical-shift measurements from mechanism-based kinetic modeling while providing a unified environment for data handling, numerical fitting, diagnostics, plotting, and export of results.

In the `Spectroscopy` module, `.xlsx` datasets can be imported with explicit selection of spectra and concentration sheets, configurable channel selection, EFA-assisted workflows, baseline correction, weighting options, and fitted plots with result export. The `NMR` module supports `.xlsx` import with concentration-sheet and chemical-shift-sheet selection, signal selection and assignment, model definition, numerical fitting, and export of the resulting analysis. The `Kinetics` module provides a dataset import wizard, editable reaction- and mechanism-based kinetic models, global fitting, ODE-based concentration profiles, fit diagnostics, and XLSX export of fitted results.

## Architecture

The project is organized into three main components:
- `hmfit_core`: Core numerical routines for equilibrium-model fitting, plotting payloads, exports, and shared utilities.
- `hmfit_gui_qt`: The PySide6 graphical user interface, including the `Spectroscopy`, `NMR`, and `Kinetics` modules.
- `hmfit.kinetics`: Kinetics-specific data handling, mechanism parsing, model construction, ODE solving, fitting, diagnostics, and reporting utilities.

## Installation

```bash
python -m pip install -r requirements_qt.txt
```

## Running

```bash
python -m hmfit_gui_qt
```

## Releases and updates

HM fit ships GitHub-based release artifacts for:

- Linux `AppImage`
- Linux `.flatpak`
- Windows portable `.exe`
- Windows installer `setup.exe`

The GUI updater supports two user-selectable channels:

- `stable`
- `beta`

See [`build_all.md`](build_all.md) for the GitHub Actions release flow and local packaging commands.

## Requirements

- Python 3.9+
- PySide6
- NumPy, SciPy, Pandas
- Matplotlib, Plotly

## License

GPL-3.0-or-later
