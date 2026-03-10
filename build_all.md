# HM Fit release and updater guide

## GitHub release flow

The main release pipeline now lives in [`.github/workflows/release.yml`](.github/workflows/release.yml).

It builds these artifacts in GitHub Actions and publishes them to GitHub Releases:

- `AppImage` for Linux
- `.flatpak` bundle for Linux
- Windows portable `.exe`
- Windows installer `setup.exe`

The workflow runs on:

- `push` to tags like `v0.1.0` or `v0.1.0-beta.1`
- manual `workflow_dispatch`

Release channel is inferred from `hmfit_gui_qt/version.py`:

- versions without prerelease suffix are `stable`
- versions like `-beta.1`, `-rc1`, `-alpha.1` are `beta`

If the workflow is triggered by a tag, the tag must match `v<VERSION>`.

## In-app updater behavior

The updater now uses `PJGV333/GUI_HM_fit` by default. No environment variables are required for normal use.

The user can choose the update channel inside `Ayuda > Canal de actualizaciones...`:

- `Estable`: only final releases
- `Beta`: includes prereleases

Update behavior depends on the installed format:

- Windows installed build: downloads the installer and launches it
- Windows portable build: downloads the new portable `.exe` and can replace the current executable
- AppImage: downloads the new AppImage, replaces the current file and relaunches HM Fit
- Flatpak: downloads the `.flatpak` bundle from GitHub Releases and runs `flatpak install --or-update --bundle` on the host

## Local builds

### Common prerequisites

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_qt.txt pyinstaller
```

### Windows

- Inno Setup 6 (`iscc.exe` available in `PATH`)

Portable + installer:

```powershell
./build_windows.ps1 -Target All -PortableOneFile
```

Outputs:

- `dist\hmfit_pyside6_portable.exe`
- `dist\installer\hmfit_setup.exe`

### Linux AppImage

```bash
chmod +x scripts/build_hmfit_pyside6_appimage.sh packaging/linux/appimage/AppRun
./scripts/build_hmfit_pyside6_appimage.sh --source . --out ./dist_appimage
```

### Linux Flatpak

Install the SDK/runtime once:

```bash
flatpak install -y flathub org.kde.Platform//6.7 org.kde.Sdk//6.7
```

Build a bundle for the selected channel branch:

```bash
python3 scripts/build_hmfit_flatpak.py --source . --branch stable --dest ./dist/org.hmfit.HMFit.flatpak
```

or:

```bash
python3 scripts/build_hmfit_flatpak.py --source . --branch beta --dest ./dist/org.hmfit.HMFit-beta.flatpak
```

The Flatpak build script:

- verifies the KDE SDK is installed
- downloads wheels inside the Flatpak SDK sandbox
- builds the Flatpak offline
- exports the requested branch
- creates a `.flatpak` bundle ready to install with `flatpak install --or-update --bundle`
