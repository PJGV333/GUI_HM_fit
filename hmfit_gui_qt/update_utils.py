from __future__ import annotations

import configparser
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

UPDATE_CHANNEL_STABLE = "stable"
UPDATE_CHANNEL_BETA = "beta"
UPDATE_CHANNELS = (UPDATE_CHANNEL_STABLE, UPDATE_CHANNEL_BETA)

PACKAGE_KIND_APPIMAGE = "appimage"
PACKAGE_KIND_FLATPAK = "flatpak"
PACKAGE_KIND_WINDOWS_INSTALLER = "windows-installer"
PACKAGE_KIND_WINDOWS_PORTABLE = "windows-portable"
PACKAGE_KIND_UNKNOWN = "unknown"

_PRERELEASE_ORDER = {
    "dev": 0,
    "a": 1,
    "alpha": 1,
    "b": 2,
    "beta": 2,
    "pre": 3,
    "rc": 3,
}


def normalize_update_channel(channel: str | None, *, fallback: str = UPDATE_CHANNEL_STABLE) -> str:
    value = str(channel or "").strip().lower()
    if value in UPDATE_CHANNELS:
        return value
    return fallback


def channel_display_name(channel: str) -> str:
    normalized = normalize_update_channel(channel)
    return "Beta" if normalized == UPDATE_CHANNEL_BETA else "Estable"


def _normalize_tag(version: str) -> str:
    clean = str(version or "").strip()
    clean = clean.split("+", 1)[0]
    clean = clean.lstrip("vV")
    return clean


def _split_main_and_prerelease(version: str) -> tuple[str, str]:
    clean = _normalize_tag(version)
    if "-" in clean:
        return clean.split("-", 1)[0], clean.split("-", 1)[1]

    compact = re.match(r"^(\d+(?:\.\d+)*)([A-Za-z].*)$", clean)
    if compact:
        return compact.group(1), compact.group(2)
    return clean, ""


def _prerelease_key(prerelease: str) -> tuple[int, int, int]:
    text = str(prerelease or "").strip().lower()
    text = text.lstrip("._-")
    if not text:
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


def is_prerelease_version(version: str) -> bool:
    return bool(_split_main_and_prerelease(version)[1])


def default_update_channel_for_version(version: str) -> str:
    if is_prerelease_version(version):
        return UPDATE_CHANNEL_BETA
    return UPDATE_CHANNEL_STABLE


def is_newer_version(candidate_version: str, current_version: str) -> bool:
    return version_key(candidate_version) > version_key(current_version)


def detect_platform() -> str | None:
    system_name = platform.system().lower()
    if "windows" in system_name:
        return "windows"
    if "linux" in system_name:
        return "linux"
    return None


def _resolve_executable_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    argv0 = Path(sys.argv[0] if sys.argv else ".")
    if argv0.exists():
        return argv0.resolve()
    return Path.cwd()


def _read_flatpak_info() -> dict[str, str]:
    info_path = Path("/.flatpak-info")
    if not info_path.is_file():
        return {}

    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read(info_path, encoding="utf-8")
    except Exception:
        return {}

    data: dict[str, str] = {}
    for section in parser.sections():
        for key, value in parser.items(section):
            data[f"{section.lower()}.{key.lower()}"] = str(value)
    return data


@dataclass(frozen=True)
class RuntimeContext:
    platform: str | None
    package_kind: str
    executable_path: Path
    appimage_path: Path | None = None
    flatpak_id: str = ""
    flatpak_branch: str = ""

    @property
    def is_flatpak(self) -> bool:
        return self.package_kind == PACKAGE_KIND_FLATPAK

    @property
    def is_appimage(self) -> bool:
        return self.package_kind == PACKAGE_KIND_APPIMAGE

    @property
    def is_windows_installer(self) -> bool:
        return self.package_kind == PACKAGE_KIND_WINDOWS_INSTALLER

    @property
    def is_windows_portable(self) -> bool:
        return self.package_kind == PACKAGE_KIND_WINDOWS_PORTABLE


def detect_runtime_context() -> RuntimeContext:
    current_platform = detect_platform()
    executable_path = _resolve_executable_path()

    if os.getenv("FLATPAK_ID") or Path("/.flatpak-info").is_file():
        flatpak_info = _read_flatpak_info()
        flatpak_id = str(flatpak_info.get("application.name") or os.getenv("FLATPAK_ID") or "").strip()
        flatpak_branch = str(flatpak_info.get("instance.branch") or flatpak_info.get("application.branch") or "").strip()
        return RuntimeContext(
            platform=current_platform,
            package_kind=PACKAGE_KIND_FLATPAK,
            executable_path=executable_path,
            flatpak_id=flatpak_id,
            flatpak_branch=flatpak_branch,
        )

    appimage_env = str(os.getenv("APPIMAGE") or "").strip()
    if appimage_env:
        appimage_path = Path(appimage_env).expanduser()
        try:
            appimage_path = appimage_path.resolve()
        except Exception:
            pass
        return RuntimeContext(
            platform=current_platform,
            package_kind=PACKAGE_KIND_APPIMAGE,
            executable_path=executable_path,
            appimage_path=appimage_path,
        )

    if current_platform == "windows":
        exe_dir = executable_path.parent
        has_inno_uninstaller = any(exe_dir.glob("unins*.exe"))
        package_kind = PACKAGE_KIND_WINDOWS_INSTALLER if has_inno_uninstaller else PACKAGE_KIND_WINDOWS_PORTABLE
        return RuntimeContext(
            platform=current_platform,
            package_kind=package_kind,
            executable_path=executable_path,
        )

    return RuntimeContext(
        platform=current_platform,
        package_kind=PACKAGE_KIND_UNKNOWN,
        executable_path=executable_path,
    )


def select_release_for_channel(releases: list[dict[str, Any]], channel: str) -> dict[str, Any] | None:
    normalized_channel = normalize_update_channel(channel)
    candidates: list[dict[str, Any]] = []

    for release in releases:
        if not isinstance(release, dict):
            continue
        if bool(release.get("draft")):
            continue
        if normalized_channel == UPDATE_CHANNEL_STABLE and bool(release.get("prerelease")):
            continue

        tag_name = str(release.get("tag_name") or "").strip()
        if not tag_name:
            continue
        candidates.append(release)

    if not candidates:
        return None

    candidates.sort(
        key=lambda release: (
            version_key(str(release.get("tag_name") or "")),
            str(release.get("published_at") or ""),
            str(release.get("created_at") or ""),
        ),
        reverse=True,
    )
    return candidates[0]


def select_release_asset(
    assets: list[dict[str, Any]],
    current_platform: str,
    package_kind: str,
) -> dict[str, Any] | None:
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

        if package_kind == PACKAGE_KIND_WINDOWS_INSTALLER:
            installer_assets = [
                asset
                for asset in exe_assets
                if any(token in str(asset.get("name") or "").lower() for token in ("setup", "installer"))
            ]
            if not installer_assets:
                installer_assets = exe_assets
            installer_assets.sort(
                key=lambda asset: (
                    0 if ("x64" in str(asset.get("name") or "").lower() or "win64" in str(asset.get("name") or "").lower()) else 1,
                    len(str(asset.get("name") or "")),
                )
            )
            return installer_assets[0]

        portable_assets = [
            asset
            for asset in exe_assets
            if not any(token in str(asset.get("name") or "").lower() for token in ("setup", "installer"))
        ]
        if not portable_assets:
            portable_assets = exe_assets
        portable_assets.sort(
            key=lambda asset: (
                0 if "portable" in str(asset.get("name") or "").lower() else 1,
                0 if ("x64" in str(asset.get("name") or "").lower() or "win64" in str(asset.get("name") or "").lower()) else 1,
                len(str(asset.get("name") or "")),
            )
        )
        return portable_assets[0]

    if current_platform == "linux":
        if package_kind == PACKAGE_KIND_APPIMAGE:
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

        if package_kind == PACKAGE_KIND_FLATPAK:
            flatpak_assets = [a for a in valid_assets if str(a.get("name") or "").lower().endswith(".flatpak")]
            if not flatpak_assets:
                return None
            flatpak_assets.sort(key=lambda asset: len(str(asset.get("name") or "")))
            return flatpak_assets[0]

    return None
