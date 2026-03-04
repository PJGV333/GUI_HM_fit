# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTabWidget

from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab
from hmfit_gui_qt.tabs.nmr_tab import NMRTab
from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab
from hmfit_gui_qt.version import GITHUB_OWNER, GITHUB_REPO, VERSION
from hmfit_gui_qt.workers.updater_worker import AssetDownloadWorker, ReleaseCheckWorker


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self._github_owner = str(os.getenv("HMFIT_GITHUB_OWNER", GITHUB_OWNER)).strip()
        self._github_repo = str(os.getenv("HMFIT_GITHUB_REPO", GITHUB_REPO)).strip()
        self._update_check_worker: ReleaseCheckWorker | None = None
        self._download_worker: AssetDownloadWorker | None = None
        self._pending_release: dict[str, Any] | None = None

        self.setWindowTitle(f"HM Fit {VERSION}")

        tabs = QTabWidget(self)
        tabs.addTab(SpectroscopyTab(parent=tabs), "Spectroscopy")
        tabs.addTab(NMRTab(parent=tabs), "NMR")
        tabs.addTab(KineticsTab(parent=tabs), "Kinetics")

        self.setCentralWidget(tabs)
        self.resize(1200, 800)
        self._build_menu()
        QTimer.singleShot(1200, self.check_for_updates_on_startup)

    def _build_menu(self) -> None:
        help_menu = self.menuBar().addMenu("Ayuda")

        check_updates_action = QAction("Buscar actualizaciones...", self)
        check_updates_action.triggered.connect(self.check_for_updates_manual)
        help_menu.addAction(check_updates_action)

        open_releases_action = QAction("Abrir página de releases", self)
        open_releases_action.triggered.connect(self._open_releases_page)
        help_menu.addAction(open_releases_action)

        show_version_action = QAction(f"Versión actual: {VERSION}", self)
        show_version_action.setEnabled(False)
        help_menu.addAction(show_version_action)

    def _is_repo_configured(self) -> bool:
        owner = self._github_owner.strip()
        repo = self._github_repo.strip()
        if not owner or not repo:
            return False
        if owner.upper() == "USUARIO" or repo.upper() == "REPO":
            return False
        return True

    def _open_releases_page(self) -> None:
        if not self._is_repo_configured():
            QMessageBox.information(
                self,
                "Actualizador no configurado",
                "Configura HMFIT_GITHUB_OWNER y HMFIT_GITHUB_REPO para habilitar el actualizador.",
            )
            return
        QDesktopServices.openUrl(QUrl(f"https://github.com/{self._github_owner}/{self._github_repo}/releases"))

    def check_for_updates_on_startup(self) -> None:
        if not self._is_repo_configured():
            return
        self._start_update_check(silent=True)

    def check_for_updates_manual(self) -> None:
        if not self._is_repo_configured():
            QMessageBox.information(
                self,
                "Actualizador no configurado",
                (
                    "No se configuró el repositorio GitHub.\n\n"
                    "Define estas variables de entorno antes de ejecutar HM Fit:\n"
                    "  HMFIT_GITHUB_OWNER=<usuario>\n"
                    "  HMFIT_GITHUB_REPO=<repo>"
                ),
            )
            return
        self._start_update_check(silent=False)

    def _start_update_check(self, *, silent: bool) -> None:
        if self._update_check_worker is not None:
            if not silent:
                QMessageBox.information(self, "Actualización", "Ya hay una verificación de actualización en curso.")
            return

        worker = ReleaseCheckWorker(
            owner=self._github_owner,
            repo=self._github_repo,
            current_version=VERSION,
            parent=self,
        )
        self._update_check_worker = worker
        worker.update_available.connect(self._on_update_available)
        worker.up_to_date.connect(lambda msg, silent=silent: self._on_update_up_to_date(msg, silent=silent))
        worker.unsupported_platform.connect(lambda msg, silent=silent: self._on_update_error(msg, silent=silent))
        worker.error.connect(lambda msg, silent=silent: self._on_update_error(msg, silent=silent))
        worker.finished.connect(self._on_update_check_finished)
        worker.start()

    def _on_update_check_finished(self) -> None:
        self._update_check_worker = None

    def _on_update_up_to_date(self, message: str, *, silent: bool) -> None:
        if silent:
            return
        QMessageBox.information(self, "Actualización", message)

    def _on_update_error(self, message: str, *, silent: bool) -> None:
        if silent:
            return
        QMessageBox.warning(self, "Actualización", message)

    def _on_update_available(self, release_payload: dict[str, Any]) -> None:
        self._pending_release = dict(release_payload)

        release_name = str(release_payload.get("release_name") or release_payload.get("tag_name") or "Nueva versión")
        tag_name = str(release_payload.get("tag_name") or "")
        published_at = str(release_payload.get("published_at") or "")
        asset_name = str(release_payload.get("asset_name") or "asset")
        asset_size = int(release_payload.get("asset_size") or 0)
        size_mb = asset_size / (1024 * 1024) if asset_size > 0 else 0.0

        details = [
            f"Versión actual: {VERSION}",
            f"Versión disponible: {tag_name}",
            f"Release: {release_name}",
            f"Archivo: {asset_name}",
        ]
        if size_mb > 0:
            details.append(f"Tamaño aprox.: {size_mb:.2f} MB")
        if published_at:
            details.append(f"Publicado: {published_at}")

        text = "Hay una actualización disponible.\n\n" + "\n".join(details) + "\n\n¿Deseas descargarla ahora?"
        answer = QMessageBox.question(
            self,
            "Actualización disponible",
            text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            return

        self._start_update_download(release_payload)

    def _start_update_download(self, release_payload: dict[str, Any]) -> None:
        if self._download_worker is not None:
            QMessageBox.information(self, "Actualización", "Ya hay una descarga en curso.")
            return

        download_url = str(release_payload.get("asset_url") or "").strip()
        asset_name = str(release_payload.get("asset_name") or "").strip()
        if not download_url or not asset_name:
            QMessageBox.warning(self, "Actualización", "No se encontró URL de descarga para el release.")
            return

        worker = AssetDownloadWorker(download_url=download_url, file_name=asset_name, parent=self)
        self._download_worker = worker
        worker.progress.connect(self._on_download_progress)
        worker.completed.connect(self._on_download_completed)
        worker.error.connect(self._on_download_error)
        worker.finished.connect(self._on_download_finished)
        self.statusBar().showMessage("Descargando actualización...")
        worker.start()

    def _on_download_finished(self) -> None:
        self._download_worker = None
        self.statusBar().clearMessage()

    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            percent = (downloaded / total) * 100.0
            self.statusBar().showMessage(f"Descargando actualización... {percent:.1f}%")
        else:
            mb = downloaded / (1024 * 1024)
            self.statusBar().showMessage(f"Descargando actualización... {mb:.2f} MB")

    def _on_download_error(self, message: str) -> None:
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "Actualización", f"No se pudo descargar la actualización:\n{message}")

    def _on_download_completed(self, file_path: str) -> None:
        path = Path(file_path)
        system_name = platform.system().lower()

        if "windows" in system_name:
            self._prompt_windows_installer(path)
            return
        if "linux" in system_name:
            self._show_linux_appimage_instructions(path)
            return

        QMessageBox.information(
            self,
            "Actualización descargada",
            f"Archivo descargado en:\n{path}\n\nInstálalo manualmente para continuar.",
        )

    def _prompt_windows_installer(self, installer_path: Path) -> None:
        answer = QMessageBox.question(
            self,
            "Instalador listo",
            (
                f"Se descargó el instalador:\n{installer_path}\n\n"
                "¿Deseas ejecutarlo ahora? HM Fit se cerrará para completar la instalación."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            QMessageBox.information(self, "Actualización", f"Puedes ejecutar manualmente:\n{installer_path}")
            return

        try:
            subprocess.Popen([str(installer_path)], cwd=str(installer_path.parent))
        except Exception as exc:
            QMessageBox.critical(self, "Actualización", f"No se pudo iniciar el instalador:\n{exc}")
            return

        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _show_linux_appimage_instructions(self, appimage_path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(appimage_path.parent)))
        QMessageBox.information(
            self,
            "AppImage descargada",
            (
                f"Se descargó la nueva AppImage en:\n{appimage_path}\n\n"
                "Pasos sugeridos:\n"
                "1) Mueve el archivo al directorio donde guardas HM Fit.\n"
                "2) Dale permisos de ejecución: chmod +x <archivo>.AppImage\n"
                "3) Reemplaza la AppImage anterior y vuelve a abrir HM Fit."
            ),
        )
