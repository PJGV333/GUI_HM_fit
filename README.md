# HM fit (PySide6 Version)

**Authors**  
Pedro Jancarlo Gomez Vega\*, José Octavio Juárez Sánchez, Ramón Moreno Corral, Felipe Medrano Valenzuela, David Octavio Corona Martínez y Karen L. Ochoa Lara\*

**Affiliation**  
Grupo de Química Supramolecular, Universidad de Sonora

\* Corresponding authors

HM fit is a software platform with a graphical user interface for the quantitative analysis and fitting of host-guest binding models from spectroscopic and nuclear magnetic resonance (NMR) data. It supports the study of complexation equilibria by integrating model-based fitting workflows with an accessible PySide6 interface for scientific data analysis.

## Architecture

The project is divided into two main components:
- `hmfit_core`: The mathematical engine, organized into `solvers`, `utils`, and `processors`.
- `hmfit_gui_qt`: The PySide6-based graphical user interface.

## Installation

```bash
python -m pip install -r requirements_qt.txt
```

## Running

```bash
python -m hmfit_gui_qt
```

## Releases and updates

HM fit now ships GitHub-based release artifacts for:

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
