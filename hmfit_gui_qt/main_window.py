from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from hmfit_gui_qt.tabs.nmr_tab import NMRTab
from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("HM Fit")

        tabs = QTabWidget(self)
        tabs.addTab(SpectroscopyTab(parent=tabs), "Spectroscopy")
        tabs.addTab(NMRTab(parent=tabs), "NMR")

        self.setCentralWidget(tabs)
        self.resize(1200, 800)

