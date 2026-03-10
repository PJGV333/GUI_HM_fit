from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hmfit_gui_qt.update_utils import (
    PACKAGE_KIND_APPIMAGE,
    PACKAGE_KIND_WINDOWS_INSTALLER,
    PACKAGE_KIND_WINDOWS_PORTABLE,
    default_update_channel_for_version,
    detect_runtime_context,
    is_newer_version,
    select_release_asset,
    select_release_for_channel,
)


def test_default_update_channel_uses_beta_for_prerelease() -> None:
    assert default_update_channel_for_version("1.2.3-beta.1") == "beta"
    assert default_update_channel_for_version("1.2.3") == "stable"


def test_is_newer_version_orders_stable_after_same_beta() -> None:
    assert is_newer_version("1.2.3", "1.2.3-beta.1")
    assert is_newer_version("1.2.4-beta.1", "1.2.3")
    assert not is_newer_version("1.2.3-beta.1", "1.2.3")


def test_select_release_for_stable_skips_prereleases() -> None:
    releases = [
        {"tag_name": "v1.0.0-beta.1", "prerelease": True, "draft": False, "published_at": "2026-03-10T09:00:00Z"},
        {"tag_name": "v0.9.0", "prerelease": False, "draft": False, "published_at": "2026-03-09T09:00:00Z"},
    ]

    selected = select_release_for_channel(releases, "stable")

    assert selected is not None
    assert selected["tag_name"] == "v0.9.0"


def test_select_release_for_beta_picks_newest_version() -> None:
    releases = [
        {"tag_name": "v1.0.0", "prerelease": False, "draft": False, "published_at": "2026-03-08T09:00:00Z"},
        {"tag_name": "v1.1.0-beta.1", "prerelease": True, "draft": False, "published_at": "2026-03-10T09:00:00Z"},
        {"tag_name": "v1.0.5", "prerelease": False, "draft": False, "published_at": "2026-03-09T09:00:00Z"},
    ]

    selected = select_release_for_channel(releases, "beta")

    assert selected is not None
    assert selected["tag_name"] == "v1.1.0-beta.1"


def test_select_release_asset_windows_installer_prefers_setup() -> None:
    assets = [
        {"name": "HMFit-1.0.0-windows-x64-portable.exe", "browser_download_url": "https://example.invalid/portable.exe"},
        {"name": "HMFit-1.0.0-windows-x64-setup.exe", "browser_download_url": "https://example.invalid/setup.exe"},
    ]

    selected = select_release_asset(assets, "windows", PACKAGE_KIND_WINDOWS_INSTALLER)

    assert selected is not None
    assert selected["name"].endswith("setup.exe")


def test_select_release_asset_windows_portable_ignores_setup() -> None:
    assets = [
        {"name": "HMFit-1.0.0-windows-x64-setup.exe", "browser_download_url": "https://example.invalid/setup.exe"},
        {"name": "HMFit-1.0.0-windows-x64-portable.exe", "browser_download_url": "https://example.invalid/portable.exe"},
    ]

    selected = select_release_asset(assets, "windows", PACKAGE_KIND_WINDOWS_PORTABLE)

    assert selected is not None
    assert selected["name"].endswith("portable.exe")


def test_detect_runtime_context_appimage(monkeypatch, tmp_path: Path) -> None:
    fake_app = tmp_path / "HMFit.AppImage"
    fake_app.write_text("binary", encoding="utf-8")
    fake_launcher = tmp_path / "launcher.py"
    fake_launcher.write_text("print('hmfit')", encoding="utf-8")

    monkeypatch.setattr("hmfit_gui_qt.update_utils.sys.argv", [str(fake_launcher)])
    monkeypatch.setattr("hmfit_gui_qt.update_utils.platform.system", lambda: "Linux")
    monkeypatch.setenv("APPIMAGE", str(fake_app))
    monkeypatch.delenv("FLATPAK_ID", raising=False)

    context = detect_runtime_context()

    assert context.package_kind == PACKAGE_KIND_APPIMAGE
    assert context.appimage_path == fake_app.resolve()
