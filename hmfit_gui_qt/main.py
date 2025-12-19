from __future__ import annotations


def main() -> int:
    import sys

    from PySide6.QtWidgets import QApplication

    from hmfit_gui_qt.main_window import MainWindow

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

