from __future__ import annotations

from hmfit_gui_qt.__main__ import main as gui_main

# Force PyInstaller to include Qt and GUI modules.
import hmfit_gui_qt.main  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(gui_main())
