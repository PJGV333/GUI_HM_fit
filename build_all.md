# HM Fit prerelease build guide

## 0) Prerequisites

### Common
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements_qt.txt pyinstaller
```

### Windows only
- Inno Setup 6 (`iscc.exe` in PATH)

### Linux only
- `appimagetool` (or `linuxdeployqt`)
- `flatpak` + `flatpak-builder`

## 1) Auto-updater configuration (GUI)

Before running HM Fit in tester machines, set the repository used by the updater:

```bash
# Linux/macOS
export HMFIT_GITHUB_OWNER="<USUARIO>"
export HMFIT_GITHUB_REPO="<REPO>"
```

```powershell
# Windows PowerShell
$env:HMFIT_GITHUB_OWNER = "<USUARIO>"
$env:HMFIT_GITHUB_REPO  = "<REPO>"
```

Current prerelease version constant is `0.9.0-beta` in `hmfit_gui_qt/version.py`.

## 2) Windows portable (.exe)

### Option A: Using existing spec (onefile)
```bash
pyinstaller --noconfirm --clean hmfit_pyside6.spec
```

Expected output:
- `dist/hmfit_pyside6.exe` (single-file executable)

### Option B: onedir (recommended for setup creation)
```bash
pyinstaller --noconfirm --clean --windowed --onedir --name hmfit_pyside6 hmfit_pyside6_entry.py
```

Expected output:
- `dist/hmfit_pyside6/` (folder with `hmfit_pyside6.exe` and runtime files)

## 3) Windows installer setup (Inno Setup)

Use the generated onedir folder (`dist/hmfit_pyside6/`):

```powershell
iscc /DMyAppVersion=0.9.0-beta /DMyDistDir=dist\hmfit_pyside6 packaging\windows\hmfit_setup.iss
```

Expected output:
- `dist/installer/hmfit_setup.exe`

## 4) Linux AppImage

### Option A: Existing script (updated to use standard AppRun + desktop)
```bash
chmod +x scripts/build_hmfit_pyside6_appimage.sh packaging/linux/appimage/AppRun
./scripts/build_hmfit_pyside6_appimage.sh --source . --out ./dist_appimage
```

Expected output:
- `dist_appimage/hmfit_pyside6-x86_64.AppImage`

### Option B: Manual with appimagetool
```bash
pyinstaller --noconfirm --clean --windowed --onefile --name hmfit_pyside6 hmfit_pyside6_entry.py

rm -rf AppDir
mkdir -p AppDir/usr/bin AppDir/usr/share/applications AppDir/usr/share/icons/hicolor/scalable/apps
cp dist/hmfit_pyside6 AppDir/usr/bin/hmfit_pyside6
cp packaging/linux/appimage/AppRun AppDir/AppRun
chmod +x AppDir/AppRun
cp packaging/linux/appimage/hmfit.desktop AppDir/usr/share/applications/hmfit.desktop
cp packaging/linux/appimage/hmfit.desktop AppDir/hmfit.desktop
cp hmfit.svg AppDir/usr/share/icons/hicolor/scalable/apps/hmfit.svg
cp hmfit.svg AppDir/hmfit.svg

ARCH=x86_64 appimagetool AppDir HMFit-x86_64.AppImage
```

### Option C: linuxdeployqt
```bash
linuxdeployqt AppDir/usr/share/applications/hmfit.desktop -appimage -unsupported-allow-new-glibc
```

## 5) Linux Flatpak

Install runtimes:
```bash
flatpak install -y flathub org.kde.Platform//6.7 org.kde.Sdk//6.7
```

Recommended build and local install:
```bash
python3 scripts/build_hmfit_flatpak.py --install
```

What the script does:
- Verifies the SDK is installed first with `flatpak info org.kde.Sdk//6.7`
- Downloads Python wheels inside `org.kde.Sdk//6.7` using `flatpak run --command=sh ... -lc ...`
- Builds the Flatpak offline from the patched manifest
- Creates a `.flatpak` bundle and installs it with `flatpak install --user --or-update --bundle`
- Does not use a persistent `hmfit-local` remote

If you used an older version of the script that created `hmfit-local`, remove the stale remote once:
```bash
flatpak remote-delete --user hmfit-local
```

Run:
```bash
flatpak run org.hmfit.HMFit
```

Check Python inside the Flatpak sandbox:
```bash
flatpak run --command=sh org.hmfit.HMFit -lc 'which python3; python3 -V'
```
