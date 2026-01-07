# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from PySide6.QtWidgets import QMainWindow, QTabWidget

from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab
from hmfit_gui_qt.tabs.nmr_tab import NMRTab
from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle("HM Fit")

        tabs = QTabWidget(self)
        tabs.addTab(SpectroscopyTab(parent=tabs), "Spectroscopy")
        tabs.addTab(NMRTab(parent=tabs), "NMR")
        tabs.addTab(KineticsTab(parent=tabs), "Kinetics")

        self.setCentralWidget(tabs)
        self.resize(1200, 800)
