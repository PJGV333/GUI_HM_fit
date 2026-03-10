# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QTimer, QUrl
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import QApplication, QInputDialog, QMainWindow, QMessageBox, QTabWidget

from hmfit_gui_qt.tabs.kinetics_tab import KineticsTab
from hmfit_gui_qt.tabs.nmr_tab import NMRTab
from hmfit_gui_qt.tabs.spectroscopy_tab import SpectroscopyTab
from hmfit_gui_qt.update_utils import (
    PACKAGE_KIND_APPIMAGE,
    PACKAGE_KIND_FLATPAK,
    PACKAGE_KIND_UNKNOWN,
    PACKAGE_KIND_WINDOWS_INSTALLER,
    PACKAGE_KIND_WINDOWS_PORTABLE,
    channel_display_name,
    default_update_channel_for_version,
    detect_runtime_context,
    normalize_update_channel,
)
from hmfit_gui_qt.version import FLATPAK_APP_ID, GITHUB_OWNER, GITHUB_REPO, RELEASES_URL, VERSION
from hmfit_gui_qt.workers.updater_worker import AssetDownloadWorker, FlatpakUpdateWorker, ReleaseCheckWorker

SETTINGS_UPDATE_CHANNEL_KEY = "updates/channel"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self._settings = QSettings()
        self._github_owner = GITHUB_OWNER
        self._github_repo = GITHUB_REPO
        self._runtime_context = detect_runtime_context()
        default_channel = default_update_channel_for_version(VERSION)
        saved_channel = self._settings.value(SETTINGS_UPDATE_CHANNEL_KEY, default_channel, type=str)
        self._update_channel = normalize_update_channel(
            os.getenv("HMFIT_UPDATE_CHANNEL") or saved_channel,
            fallback=default_channel,
        )

        self._update_check_worker: ReleaseCheckWorker | None = None
        self._download_worker: AssetDownloadWorker | None = None
        self._flatpak_worker: FlatpakUpdateWorker | None = None
        self._pending_release: dict[str, Any] | None = None
        self._channel_status_action: QAction | None = None
        self._distribution_status_action: QAction | None = None

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

        channel_action = QAction("Canal de actualizaciones...", self)
        channel_action.triggered.connect(self._choose_update_channel)
        help_menu.addAction(channel_action)

        open_releases_action = QAction("Abrir página de releases", self)
        open_releases_action.triggered.connect(self._open_releases_page)
        help_menu.addAction(open_releases_action)

        help_menu.addSeparator()

        self._channel_status_action = QAction("", self)
        self._channel_status_action.setEnabled(False)
        help_menu.addAction(self._channel_status_action)

        self._distribution_status_action = QAction("", self)
        self._distribution_status_action.setEnabled(False)
        help_menu.addAction(self._distribution_status_action)

        show_version_action = QAction(f"Versión actual: {VERSION}", self)
        show_version_action.setEnabled(False)
        help_menu.addAction(show_version_action)

        self._refresh_update_status_actions()

    def _refresh_update_status_actions(self) -> None:
        if self._channel_status_action is not None:
            self._channel_status_action.setText(f"Canal actual: {channel_display_name(self._update_channel)}")
        if self._distribution_status_action is not None:
            self._distribution_status_action.setText(f"Instalación: {self._distribution_label()}")

    def _distribution_label(self) -> str:
        if self._runtime_context.package_kind == PACKAGE_KIND_FLATPAK:
            branch = str(self._runtime_context.flatpak_branch or "").strip()
            if branch:
                return f"Flatpak ({branch})"
            return "Flatpak"
        if self._runtime_context.package_kind == PACKAGE_KIND_APPIMAGE:
            return "AppImage"
        if self._runtime_context.package_kind == PACKAGE_KIND_WINDOWS_INSTALLER:
            return "Windows instalada"
        if self._runtime_context.package_kind == PACKAGE_KIND_WINDOWS_PORTABLE:
            return "Windows portable"
        if self._runtime_context.package_kind == PACKAGE_KIND_UNKNOWN:
            return "No detectada"
        return self._runtime_context.package_kind

    def _is_repo_configured(self) -> bool:
        return bool(self._github_owner.strip() and self._github_repo.strip())

    def _open_releases_page(self) -> None:
        if not self._is_repo_configured():
            QMessageBox.information(self, "Actualizador", "El repositorio GitHub de HM Fit no está configurado.")
            return

        url = RELEASES_URL
        if self._github_owner != GITHUB_OWNER or self._github_repo != GITHUB_REPO:
            url = f"https://github.com/{self._github_owner}/{self._github_repo}/releases"
        QDesktopServices.openUrl(QUrl(url))

    def check_for_updates_on_startup(self) -> None:
        if not self._is_repo_configured():
            return
        self._start_update_check(silent=True)

    def check_for_updates_manual(self) -> None:
        if not self._is_repo_configured():
            QMessageBox.information(self, "Actualizador", "El repositorio GitHub de HM Fit no está configurado.")
            return
        self._start_update_check(silent=False)

    def _choose_update_channel(self) -> None:
        options = [
            "Estable: solo releases finales",
            "Beta: incluye prereleases y betas",
        ]
        current_index = 1 if self._update_channel == "beta" else 0
        choice, accepted = QInputDialog.getItem(
            self,
            "Canal de actualizaciones",
            "Selecciona el canal que prefieres para HM Fit:",
            options,
            current_index,
            False,
        )
        if not accepted:
            return

        new_channel = "beta" if str(choice).startswith("Beta") else "stable"
        if new_channel == self._update_channel:
            return

        self._update_channel = new_channel
        self._settings.setValue(SETTINGS_UPDATE_CHANNEL_KEY, new_channel)
        self._refresh_update_status_actions()

        QMessageBox.information(
            self,
            "Canal guardado",
            f"HM Fit usará el canal {channel_display_name(new_channel)} en la próxima verificación.",
        )

    def _start_update_check(self, *, silent: bool) -> None:
        if self._update_check_worker is not None:
            if not silent:
                QMessageBox.information(self, "Actualización", "Ya hay una verificación de actualización en curso.")
            return

        worker = ReleaseCheckWorker(
            owner=self._github_owner,
            repo=self._github_repo,
            current_version=VERSION,
            channel=self._update_channel,
            current_platform=self._runtime_context.platform,
            package_kind=self._runtime_context.package_kind,
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

    def _update_action_text(self) -> str:
        package_kind = self._runtime_context.package_kind
        if package_kind == PACKAGE_KIND_FLATPAK:
            return "¿Deseas actualizar ahora la instalación Flatpak?"
        if package_kind == PACKAGE_KIND_WINDOWS_INSTALLER:
            return "¿Deseas descargar e instalar la actualización ahora?"
        if package_kind == PACKAGE_KIND_WINDOWS_PORTABLE:
            return "¿Deseas descargar la actualización portable ahora?"
        if package_kind == PACKAGE_KIND_APPIMAGE:
            return "¿Deseas descargar y reemplazar la AppImage actual ahora?"
        return "¿Deseas descargar la actualización ahora?"

    def _on_update_available(self, release_payload: dict[str, Any]) -> None:
        self._pending_release = dict(release_payload)

        release_name = str(release_payload.get("release_name") or release_payload.get("tag_name") or "Nueva versión")
        tag_name = str(release_payload.get("tag_name") or "")
        published_at = str(release_payload.get("published_at") or "")
        asset_name = str(release_payload.get("asset_name") or "").strip()
        asset_size = int(release_payload.get("asset_size") or 0)
        size_mb = asset_size / (1024 * 1024) if asset_size > 0 else 0.0

        details = [
            f"Canal: {channel_display_name(str(release_payload.get('channel') or self._update_channel))}",
            f"Instalación detectada: {self._distribution_label()}",
            f"Versión actual: {VERSION}",
            f"Versión disponible: {tag_name}",
            f"Release: {release_name}",
        ]
        if asset_name:
            details.append(f"Archivo: {asset_name}")
        if size_mb > 0:
            details.append(f"Tamaño aprox.: {size_mb:.2f} MB")
        if published_at:
            details.append(f"Publicado: {published_at}")

        text = "Hay una actualización disponible.\n\n" + "\n".join(details) + "\n\n" + self._update_action_text()
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
            QMessageBox.warning(self, "Actualización", "No se encontró un archivo compatible para esta instalación.")
            return

        worker = AssetDownloadWorker(download_url=download_url, file_name=asset_name, parent=self)
        self._download_worker = worker
        worker.progress.connect(self._on_download_progress)
        worker.completed.connect(self._on_download_completed)
        worker.error.connect(self._on_download_error)
        worker.finished.connect(self._on_download_finished)
        self.statusBar().showMessage("Descargando actualización...")
        worker.start()

    def _start_flatpak_update(self, bundle_path: Path) -> None:
        if self._flatpak_worker is not None:
            QMessageBox.information(self, "Actualización", "Ya hay una actualización Flatpak en curso.")
            return

        worker = FlatpakUpdateWorker(bundle_path=str(bundle_path), app_id=FLATPAK_APP_ID, parent=self)
        self._flatpak_worker = worker
        worker.completed.connect(self._on_flatpak_update_completed)
        worker.error.connect(self._on_flatpak_update_error)
        worker.finished.connect(self._on_flatpak_update_finished)
        self.statusBar().showMessage("Instalando actualización Flatpak...")
        worker.start()

    def _on_flatpak_update_finished(self) -> None:
        self._flatpak_worker = None
        self.statusBar().clearMessage()

    def _on_flatpak_update_completed(self, payload: dict[str, Any]) -> None:
        self._runtime_context = detect_runtime_context()
        self._refresh_update_status_actions()
        answer = QMessageBox.question(
            self,
            "Flatpak actualizado",
            (
                "La instalación Flatpak se actualizó correctamente.\n\n"
                "Reinicia HM Fit para cargar la nueva versión.\n"
                "¿Deseas cerrar la aplicación ahora?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer == QMessageBox.Yes:
            app = QApplication.instance()
            if app is not None:
                app.quit()

    def _on_flatpak_update_error(self, message: str) -> None:
        self.statusBar().clearMessage()
        QMessageBox.warning(self, "Flatpak", f"No se pudo actualizar la instalación Flatpak:\n{message}")

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

        if self._runtime_context.package_kind == PACKAGE_KIND_WINDOWS_INSTALLER:
            self._prompt_windows_installer(path)
            return
        if self._runtime_context.package_kind == PACKAGE_KIND_WINDOWS_PORTABLE:
            self._prompt_windows_portable_replace(path)
            return
        if self._runtime_context.package_kind == PACKAGE_KIND_FLATPAK:
            self._prompt_flatpak_install(path)
            return
        if self._runtime_context.package_kind == PACKAGE_KIND_APPIMAGE:
            self._prompt_appimage_replace(path)
            return

        self._show_download_location(path)

    def _show_download_location(self, path: Path) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
        QMessageBox.information(
            self,
            "Actualización descargada",
            f"Archivo descargado en:\n{path}\n\nÁbrelo manualmente para completar la actualización.",
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
            self._show_download_location(installer_path)
            return

        try:
            subprocess.Popen([str(installer_path)], cwd=str(installer_path.parent))
        except Exception as exc:
            QMessageBox.critical(self, "Actualización", f"No se pudo iniciar el instalador:\n{exc}")
            return

        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _prompt_windows_portable_replace(self, portable_path: Path) -> None:
        current_exe = self._runtime_context.executable_path
        if not current_exe.exists() or current_exe.suffix.lower() != ".exe" or not os.access(current_exe.parent, os.W_OK):
            self._show_download_location(portable_path)
            return

        answer = QMessageBox.question(
            self,
            "Actualización portable",
            (
                f"Se descargó una nueva versión en:\n{portable_path}\n\n"
                "¿Deseas reemplazar el ejecutable actual y reiniciar HM Fit ahora?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self._show_download_location(portable_path)
            return

        helper_path = portable_path.parent / "hmfit_apply_windows_update.cmd"
        new_target = current_exe.with_name(f"{current_exe.name}.new")
        script = (
            "@echo off\r\n"
            "setlocal\r\n"
            "ping 127.0.0.1 -n 3 >nul\r\n"
            f'copy /Y "{portable_path}" "{new_target}" >nul\r\n'
            f'move /Y "{new_target}" "{current_exe}" >nul\r\n'
            f'start "" "{current_exe}"\r\n'
            f'del /Q "{portable_path}" >nul 2>nul\r\n'
            "del /Q \"%~f0\" >nul 2>nul\r\n"
        )
        helper_path.write_text(script, encoding="utf-8")

        creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        try:
            subprocess.Popen(
                ["cmd.exe", "/c", str(helper_path)],
                cwd=str(helper_path.parent),
                creationflags=creationflags,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Actualización", f"No se pudo preparar el reemplazo automático:\n{exc}")
            return

        app = QApplication.instance()
        if app is not None:
            app.quit()

    def _prompt_flatpak_install(self, bundle_path: Path) -> None:
        answer = QMessageBox.question(
            self,
            "Bundle Flatpak lista",
            (
                f"Se descargó el bundle Flatpak en:\n{bundle_path}\n\n"
                "¿Deseas instalar la actualización ahora sobre la instalación actual?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self._show_download_location(bundle_path)
            return

        self._start_flatpak_update(bundle_path)

    def _prompt_appimage_replace(self, appimage_path: Path) -> None:
        current_appimage = self._runtime_context.appimage_path
        if current_appimage is None or not current_appimage.exists() or not os.access(current_appimage.parent, os.W_OK):
            self._show_download_location(appimage_path)
            return

        answer = QMessageBox.question(
            self,
            "AppImage lista",
            (
                f"Se descargó una nueva AppImage en:\n{appimage_path}\n\n"
                "¿Deseas reemplazar la AppImage actual y reiniciar HM Fit ahora?"
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )
        if answer != QMessageBox.Yes:
            self._show_download_location(appimage_path)
            return

        helper_path = appimage_path.parent / "hmfit_apply_appimage_update.sh"
        backup_path = current_appimage.with_name(f"{current_appimage.name}.previous")
        script = (
            "#!/bin/sh\n"
            "set -eu\n"
            "NEW_FILE=\"$1\"\n"
            "TARGET_FILE=\"$2\"\n"
            "BACKUP_FILE=\"$3\"\n"
            "sleep 1\n"
            "TMP_FILE=\"${TARGET_FILE}.new\"\n"
            "cp \"$NEW_FILE\" \"$TMP_FILE\"\n"
            "chmod +x \"$TMP_FILE\"\n"
            "if [ -e \"$TARGET_FILE\" ]; then\n"
            "  mv -f \"$TARGET_FILE\" \"$BACKUP_FILE\" || true\n"
            "fi\n"
            "mv -f \"$TMP_FILE\" \"$TARGET_FILE\"\n"
            "chmod +x \"$TARGET_FILE\"\n"
            "rm -f \"$NEW_FILE\"\n"
            "nohup \"$TARGET_FILE\" >/dev/null 2>&1 &\n"
            "rm -f \"$0\"\n"
        )
        helper_path.write_text(script, encoding="utf-8")
        helper_path.chmod(helper_path.stat().st_mode | 0o111)

        try:
            subprocess.Popen(
                ["/bin/sh", str(helper_path), str(appimage_path), str(current_appimage), str(backup_path)],
                cwd=str(helper_path.parent),
                start_new_session=True,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Actualización", f"No se pudo preparar el reemplazo automático:\n{exc}")
            return

        app = QApplication.instance()
        if app is not None:
            app.quit()
