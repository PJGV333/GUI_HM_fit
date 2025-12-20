from __future__ import annotations


def main() -> int:
    import sys

    # Ensure a Qt-compatible backend for the GUI, even if the environment defaults to TkAgg/etc.
    # Note: the core pipelines force `Agg` for headless figure generation.
    try:
        import matplotlib

        matplotlib.use("QtAgg")
    except Exception:
        pass

    from PySide6.QtWidgets import QApplication

    from hmfit_gui_qt.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
