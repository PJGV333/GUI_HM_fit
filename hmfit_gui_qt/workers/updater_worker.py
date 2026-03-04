from __future__ import annotations

import json
import platform
import re
import tempfile
from pathlib import Path
from typing import Any
from urllib import error as urlerror
from urllib import request as urlrequest

from PySide6.QtCore import QObject, QThread, Signal, Slot

from hmfit_gui_qt.version import GITHUB_OWNER, GITHUB_REPO, VERSION

GITHUB_RELEASE_LATEST_API = "https://api.github.com/repos/{owner}/{repo}/releases/latest"
REQUEST_TIMEOUT_S = 20
CHUNK_SIZE = 256 * 1024

_PRERELEASE_ORDER = {
    "dev": 0,
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "pre": 3,
    "rc": 3,
}


def _normalize_tag(version: str) -> str:
    clean = str(version or "").strip()
    clean = clean.split("+", 1)[0]
    clean = clean.lstrip("vV")
    return clean


def _split_main_and_prerelease(version: str) -> tuple[str, str]:
    clean = _normalize_tag(version)
    if "-" in clean:
        return clean.split("-", 1)[0], clean.split("-", 1)[1]

    # Also supports compact formats like 1.2.3rc1.
    compact = re.match(r"^(\d+(?:\.\d+)*)([A-Za-z].*)$", clean)
    if compact:
        return compact.group(1), compact.group(2)
    return clean, ""


def _prerelease_key(prerelease: str) -> tuple[int, int, int]:
    text = str(prerelease or "").strip().lower()
    text = text.lstrip("._-")
    if not text:
        # Stable release is always newer than pre-release for same X.Y.Z.
        return (1, 0, 0)

    match = re.match(r"([a-z]+)?[-_.]?(\d*)", text)
    if not match:
        return (0, 0, 0)

    label = str(match.group(1) or "").lower()
    rank = _PRERELEASE_ORDER.get(label, 0)
    number_txt = str(match.group(2) or "0")
    try:
        number = int(number_txt)
    except ValueError:
        number = 0
    return (0, rank, number)


def version_key(version: str) -> tuple[int, int, int, int, int, int]:
    main, prerelease = _split_main_and_prerelease(version)
    numbers = [int(part) for part in re.findall(r"\d+", main)]
    if not numbers:
        numbers = [0]
    numbers = (numbers + [0, 0, 0])[:3]
    pre_key = _prerelease_key(prerelease)
    return (numbers[0], numbers[1], numbers[2], pre_key[0], pre_key[1], pre_key[2])


def is_newer_version(candidate_version: str, current_version: str) -> bool:
    return version_key(candidate_version) > version_key(current_version)


def detect_platform() -> str | None:
    system_name = platform.system().lower()
    if "windows" in system_name:
        return "windows"
    if "linux" in system_name:
        return "linux"
    return None


def select_release_asset(assets: list[dict[str, Any]], current_platform: str) -> dict[str, Any] | None:
    valid_assets: list[dict[str, Any]] = []
    for asset in assets:
        if not isinstance(asset, dict):
            continue
        name = str(asset.get("name") or "").strip()
        url = str(asset.get("browser_download_url") or "").strip()
        if not name or not url:
            continue
        valid_assets.append(asset)

    if not valid_assets:
        return None

    if current_platform == "windows":
        exe_assets = [a for a in valid_assets if str(a.get("name") or "").lower().endswith(".exe")]
        if not exe_assets:
            return None
        exe_assets.sort(
            key=lambda asset: (
                0 if ("setup" in str(asset.get("name") or "").lower() or "installer" in str(asset.get("name") or "").lower()) else 1,
                0 if ("x64" in str(asset.get("name") or "").lower() or "win64" in str(asset.get("name") or "").lower()) else 1,
                len(str(asset.get("name") or "")),
            )
        )
        return exe_assets[0]

    if current_platform == "linux":
        appimage_assets = [a for a in valid_assets if str(a.get("name") or "").lower().endswith(".appimage")]
        if not appimage_assets:
            return None
        appimage_assets.sort(
            key=lambda asset: (
                0 if ("x86_64" in str(asset.get("name") or "").lower() or "amd64" in str(asset.get("name") or "").lower()) else 1,
                len(str(asset.get("name") or "")),
            )
        )
        return appimage_assets[0]

    return None


def _safe_filename(name: str) -> str:
    base = Path(str(name or "").strip()).name
    if not base:
        return "hmfit_update.bin"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", base).strip("._")
    return safe or "hmfit_update.bin"


def _http_json(url: str) -> dict[str, Any]:
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
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Invalid JSON from GitHub API.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected payload received from GitHub API.")
    return payload


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
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._owner = str(owner or "").strip()
        self._repo = str(repo or "").strip()
        self._current_version = str(current_version or "0.0.0").strip()

    @Slot()
    def run(self) -> None:
        try:
            if not self._owner or not self._repo:
                raise RuntimeError("Missing GitHub owner/repository configuration.")
            if self._owner.upper() == "USUARIO" or self._repo.upper() == "REPO":
                raise RuntimeError(
                    "Updater not configured: define real values for HMFIT_GITHUB_OWNER/HMFIT_GITHUB_REPO."
                )

            current_platform = detect_platform()
            if current_platform is None:
                self.unsupported_platform.emit("Auto-update is only supported on Windows and Linux.")
                return

            api_url = GITHUB_RELEASE_LATEST_API.format(owner=self._owner, repo=self._repo)
            release = _http_json(api_url)
            tag_name = str(release.get("tag_name") or "").strip()
            if not tag_name:
                raise RuntimeError("Release tag not found in GitHub response.")

            if not is_newer_version(tag_name, self._current_version):
                self.up_to_date.emit(f"HM Fit is up to date ({self._current_version}).")
                return

            assets = release.get("assets")
            if not isinstance(assets, list):
                assets = []
            selected_asset = select_release_asset(assets, current_platform)
            if selected_asset is None:
                raise RuntimeError(
                    f"New version found ({tag_name}) but no compatible asset for {current_platform} was published."
                )

            payload = {
                "current_version": self._current_version,
                "tag_name": tag_name,
                "release_name": str(release.get("name") or tag_name),
                "release_url": str(release.get("html_url") or ""),
                "published_at": str(release.get("published_at") or ""),
                "body": str(release.get("body") or ""),
                "asset_name": str(selected_asset.get("name") or ""),
                "asset_url": str(selected_asset.get("browser_download_url") or ""),
                "asset_size": int(selected_asset.get("size") or 0),
                "platform": current_platform,
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

            if platform.system().lower().startswith("linux") and output_path.suffix.lower() == ".appimage":
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
