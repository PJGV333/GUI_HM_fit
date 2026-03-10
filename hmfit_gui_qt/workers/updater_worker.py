from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from PySide6.QtCore import QObject, QThread, Signal, Slot

from hmfit_gui_qt.update_utils import (
    PACKAGE_KIND_FLATPAK,
    is_newer_version,
    normalize_update_channel,
    select_release_asset,
    select_release_for_channel,
)
from hmfit_gui_qt.version import FLATPAK_APP_ID, GITHUB_OWNER, GITHUB_REPO, VERSION

GITHUB_RELEASES_API = "https://api.github.com/repos/{owner}/{repo}/releases?per_page=30"
REQUEST_TIMEOUT_S = 20
CHUNK_SIZE = 256 * 1024


def _safe_filename(name: str) -> str:
    base = Path(str(name or "").strip()).name
    if not base:
        return "hmfit_update.bin"
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base).strip("._")
    return safe or "hmfit_update.bin"


def _http_json(url: str) -> Any:
    request = urlrequest.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "HMFit-Updater/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="GET",
    )
    with urlrequest.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        status = int(getattr(response, "status", 200))
        raw = response.read()
        if status < 200 or status >= 300:
            raise RuntimeError(f"GitHub API returned HTTP {status}.")

    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from GitHub API.") from exc


def _format_http_error(exc: urlerror.HTTPError) -> str:
    msg = f"GitHub API error: HTTP {exc.code}."
    try:
        body = exc.read().decode("utf-8", errors="ignore")
    except Exception:
        body = ""
    if body:
        try:
            parsed = json.loads(body)
            api_msg = str(parsed.get("message") or "").strip()
            if api_msg:
                msg = f"{msg} {api_msg}"
        except Exception:
            pass
    if exc.code == 403:
        msg = f"{msg} Rate limit reached. Try again later."
    return msg


def _format_process_error(command: list[str], proc: subprocess.CompletedProcess[str]) -> str:
    stdout = str(proc.stdout or "").strip()
    stderr = str(proc.stderr or "").strip()
    parts = [f"Command failed ({proc.returncode}): {' '.join(command)}"]
    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(stderr)
    return "\n\n".join(parts)


class _ThreadedWorker(QObject):
    finished = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(None)
        self._thread = QThread(parent)
        self.moveToThread(self._thread)
        self._thread.started.connect(self.run)
        self.finished.connect(self._thread.quit)
        self.finished.connect(self.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)

    def start(self) -> None:
        self._thread.start()

    @Slot()
    def run(self) -> None:
        raise NotImplementedError


class ReleaseCheckWorker(_ThreadedWorker):
    update_available = Signal(object)
    up_to_date = Signal(str)
    unsupported_platform = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        *,
        owner: str = GITHUB_OWNER,
        repo: str = GITHUB_REPO,
        current_version: str = VERSION,
        channel: str,
        current_platform: str | None,
        package_kind: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._owner = str(owner or "").strip()
        self._repo = str(repo or "").strip()
        self._current_version = str(current_version or "0.0.0").strip()
        self._channel = normalize_update_channel(channel)
        self._current_platform = str(current_platform or "").strip() or None
        self._package_kind = str(package_kind or "").strip()

    @Slot()
    def run(self) -> None:
        try:
            if not self._owner or not self._repo:
                raise RuntimeError("Missing GitHub owner/repository configuration.")

            if self._current_platform is None:
                self.unsupported_platform.emit("Auto-update is only supported on Windows and Linux.")
                return

            api_url = GITHUB_RELEASES_API.format(owner=self._owner, repo=self._repo)
            releases = _http_json(api_url)
            if not isinstance(releases, list):
                raise RuntimeError("Unexpected GitHub API payload while listing releases.")

            release = select_release_for_channel(releases, self._channel)
            if release is None:
                raise RuntimeError(
                    f"No published releases were found for the {self._channel} channel in {self._owner}/{self._repo}."
                )

            tag_name = str(release.get("tag_name") or "").strip()
            if not tag_name:
                raise RuntimeError("Release tag not found in GitHub response.")

            if not is_newer_version(tag_name, self._current_version):
                self.up_to_date.emit(
                    f"HM Fit ya está actualizado en el canal {self._channel} ({self._current_version})."
                )
                return

            assets = release.get("assets")
            if not isinstance(assets, list):
                assets = []

            selected_asset = select_release_asset(assets, self._current_platform, self._package_kind)
            if self._package_kind != PACKAGE_KIND_FLATPAK and selected_asset is None:
                raise RuntimeError(
                    f"New version found ({tag_name}) but no compatible asset for {self._package_kind or self._current_platform} was published."
                )

            payload = {
                "channel": self._channel,
                "current_version": self._current_version,
                "tag_name": tag_name,
                "release_name": str(release.get("name") or tag_name),
                "release_url": str(release.get("html_url") or ""),
                "published_at": str(release.get("published_at") or ""),
                "body": str(release.get("body") or ""),
                "asset_name": str((selected_asset or {}).get("name") or ""),
                "asset_url": str((selected_asset or {}).get("browser_download_url") or ""),
                "asset_size": int((selected_asset or {}).get("size") or 0),
                "platform": self._current_platform,
                "package_kind": self._package_kind,
                "prerelease": bool(release.get("prerelease")),
            }
            self.update_available.emit(payload)
        except urlerror.HTTPError as exc:
            self.error.emit(_format_http_error(exc))
        except urlerror.URLError as exc:
            self.error.emit(f"Network error while checking updates: {exc.reason}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class AssetDownloadWorker(_ThreadedWorker):
    progress = Signal(int, int)  # downloaded, total (bytes), total can be -1
    completed = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        *,
        download_url: str,
        file_name: str,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._download_url = str(download_url or "").strip()
        self._file_name = _safe_filename(file_name)

    @Slot()
    def run(self) -> None:
        try:
            if not self._download_url:
                raise RuntimeError("Empty download URL.")

            tmp_dir = Path(tempfile.mkdtemp(prefix="hmfit_update_"))
            output_path = tmp_dir / self._file_name

            request = urlrequest.Request(
                self._download_url,
                headers={"User-Agent": "HMFit-Updater/1.0"},
                method="GET",
            )
            with urlrequest.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
                status = int(getattr(response, "status", 200))
                if status < 200 or status >= 300:
                    raise RuntimeError(f"Download failed with HTTP {status}.")

                content_length = str(response.headers.get("Content-Length") or "").strip()
                total_size = int(content_length) if content_length.isdigit() else -1
                downloaded = 0
                with open(output_path, "wb") as out_file:
                    while True:
                        chunk = response.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        self.progress.emit(downloaded, total_size)

            if output_path.suffix.lower() in {".appimage", ".sh"}:
                output_path.chmod(output_path.stat().st_mode | 0o111)

            self.completed.emit(str(output_path))
        except urlerror.HTTPError as exc:
            self.error.emit(_format_http_error(exc))
        except urlerror.URLError as exc:
            self.error.emit(f"Network error while downloading update: {exc.reason}")
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class FlatpakUpdateWorker(_ThreadedWorker):
    completed = Signal(object)
    error = Signal(str)

    def __init__(
        self,
        *,
        bundle_path: str,
        app_id: str = FLATPAK_APP_ID,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._bundle_path = Path(str(bundle_path or "").strip()).expanduser()
        self._app_id = str(app_id or FLATPAK_APP_ID).strip()

    def _run_host(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        if shutil.which("flatpak-spawn") is None:
            raise RuntimeError("flatpak-spawn is not available in this Flatpak runtime.")

        proc = subprocess.run(
            ["flatpak-spawn", "--host", *command],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(_format_process_error(["flatpak-spawn", "--host", *command], proc))
        return proc

    def _installation_scope(self) -> list[str]:
        try:
            proc = self._run_host(["flatpak", "info", "--show-installation", self._app_id])
        except Exception:
            return ["--user"]

        installation = str(proc.stdout or "").strip()
        if not installation or installation == "user":
            return ["--user"]
        return [f"--installation={installation}"]

    @Slot()
    def run(self) -> None:
        try:
            if not self._app_id:
                raise RuntimeError("Flatpak app-id is not configured.")
            if not self._bundle_path.is_file():
                raise RuntimeError(f"Flatpak bundle not found: {self._bundle_path}")

            scope = self._installation_scope()
            self._run_host(["flatpak", *scope, "install", "-y", "--or-update", "--bundle", str(self._bundle_path)])

            self.completed.emit(
                {
                    "app_id": self._app_id,
                    "bundle_path": str(self._bundle_path),
                }
            )
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()
