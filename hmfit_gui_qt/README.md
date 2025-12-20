# HM Fit (PySide6 GUI)

## Run

1) Install dependencies (from repo root):

```bash
python -m pip install -r requirements_qt.txt
```

2) Start the Qt GUI:

```bash
python -m hmfit_gui_qt
```

## Notes

- Inputs are currently **XLSX only**.
- Fitting calls into `hmfit_core` (no math duplicated in the GUI).
- If the app hard-crashes (SIGSEGV), check `hmfit_crash.log` (written to the current working directory).
