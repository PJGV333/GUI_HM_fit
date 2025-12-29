# HM fit (PySide6 Version)

A GUI application for Host-Guest binding model fitting using spectroscopy and NMR data.

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

## Requirements

- Python 3.9+
- PySide6
- NumPy, SciPy, Pandas
- Matplotlib, Plotly
