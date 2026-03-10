# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import os

APP_NAME = "HM Fit"
APP_ORGANIZATION = "PJGV333"
APP_DOMAIN = "github.com"

# Release version used by the GUI, installers and updater.
VERSION = "0.1.0-beta.1"

# Default GitHub repository used by the updater. These values still accept
# environment overrides so forks can reuse the same code without patching it.
GITHUB_OWNER = str(os.getenv("HMFIT_GITHUB_OWNER") or "PJGV333").strip()
GITHUB_REPO = str(os.getenv("HMFIT_GITHUB_REPO") or "GUI_HM_fit").strip()
REPOSITORY_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}"
RELEASES_URL = f"{REPOSITORY_URL}/releases"

FLATPAK_APP_ID = "org.hmfit.HMFit"
FLATPAK_REMOTE_NAME = "hmfit"
FLATPAK_REPO_URL = str(
    os.getenv("HMFIT_FLATPAK_REPO_URL") or f"https://{GITHUB_OWNER.lower()}.github.io/{GITHUB_REPO}/flatpak/repo/"
).strip()
FLATPAK_REPOREF_URL = str(
    os.getenv("HMFIT_FLATPAK_REPOREF_URL")
    or f"https://{GITHUB_OWNER.lower()}.github.io/{GITHUB_REPO}/flatpak/hmfit.flatpakrepo"
).strip()
