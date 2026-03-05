#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

usage() {
  cat <<'EOF'
Build HM Fit (PySide6) AppImage from an external source without dirtying the repo.

Usage:
  build_hmfit_pyside6_appimage.sh --source <path|git-url> [options]

Options:
  --ref <git-ref>       Git ref (tag/branch/commit) to checkout
  --out <dir>           Output directory for AppImage
                        Default: $HOME/BUILD_HMFIT_PYSIDE6/output
  --out-file <path>     Full output AppImage path (overrides --out)
                        Example: /path/to/HMFit.AppImage
  --build-root <dir>    Root directory for temporary build
                        Default: $HOME/BUILD_HMFIT_PYSIDE6
  --appimagetool <path> Use a specific appimagetool binary/AppImage
  --keep-build          Keep the temporary build directory
  -h, --help            Show this help

Examples:
  build_hmfit_pyside6_appimage.sh --source /path/to/GUI_HM_fit
  build_hmfit_pyside6_appimage.sh --source https://github.com/user/GUI_HM_fit.git --ref v1.2.3
EOF
}

log() { echo "[build] $*"; }
warn() { echo "[warn]  $*" >&2; }
die() { echo "[error] $*" >&2; exit 1; }

require_arg() {
  if [[ -z "${2:-}" ]]; then
    die "Missing value for $1"
  fi
}

SOURCE=""
REF=""
OUT_DIR=""
OUT_FILE=""
BUILD_ROOT=""
APPIMAGETOOL_PATH=""
KEEP_BUILD=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source)
      require_arg "$1" "${2:-}"
      SOURCE="$2"
      shift 2
      ;;
    --ref)
      require_arg "$1" "${2:-}"
      REF="$2"
      shift 2
      ;;
    --out)
      require_arg "$1" "${2:-}"
      OUT_DIR="$2"
      shift 2
      ;;
    --out-file)
      require_arg "$1" "${2:-}"
      OUT_FILE="$2"
      shift 2
      ;;
    --build-root)
      require_arg "$1" "${2:-}"
      BUILD_ROOT="$2"
      shift 2
      ;;
    --appimagetool)
      require_arg "$1" "${2:-}"
      APPIMAGETOOL_PATH="$2"
      shift 2
      ;;
    --keep-build)
      KEEP_BUILD=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "$SOURCE" ]]; then
        SOURCE="$1"
        shift
      else
        die "Unknown argument: $1"
      fi
      ;;
  esac
done

if [[ -z "$SOURCE" ]]; then
  usage
  exit 1
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found"
ARCH="$(uname -m)"
[[ "$ARCH" == "x86_64" ]] || die "Unsupported architecture: $ARCH (only x86_64 supported)"

BUILD_ROOT="${BUILD_ROOT:-$HOME/BUILD_HMFIT_PYSIDE6}"
OUT_DIR="${OUT_DIR:-$BUILD_ROOT/output}"
mkdir -p "$BUILD_ROOT" "$OUT_DIR"

BUILD_DIR="$(mktemp -d "$BUILD_ROOT/hmfit_pyside6_XXXXXX")"
WORK_DIR="$BUILD_DIR/src"

cleanup() {
  if [[ "$KEEP_BUILD" -eq 1 ]]; then
    log "Keeping build dir: $BUILD_DIR"
  else
    rm -rf "$BUILD_DIR"
  fi
}
trap cleanup EXIT

mkdir -p "$WORK_DIR"

is_git_url=0
if [[ "$SOURCE" =~ ^(https?://|git@|ssh://) ]] || [[ "$SOURCE" == *.git ]]; then
  is_git_url=1
fi

copy_source() {
  local src="$1"
  if command -v rsync >/dev/null 2>&1; then
    log "Copying source with rsync"
    rsync -a \
      --exclude '.git' \
      --exclude '.venv' \
      --exclude '.venv*' \
      --exclude 'dist' \
      --exclude 'dist_appimage' \
      --exclude 'build' \
      --exclude '__pycache__' \
      --exclude '.pytest_cache' \
      --exclude 'node_modules' \
      --exclude '.idea' \
      "$src"/ "$WORK_DIR"/
  else
    log "rsync not found; copying source with cp -a"
    cp -a "$src"/. "$WORK_DIR"/
  fi
}

if [[ "$is_git_url" -eq 1 ]]; then
  command -v git >/dev/null 2>&1 || die "git not found"
  log "Cloning source: $SOURCE"
  git clone "$SOURCE" "$WORK_DIR"
  if [[ -n "$REF" ]]; then
    log "Checking out ref: $REF"
    git -C "$WORK_DIR" checkout "$REF"
  fi
else
  [[ -d "$SOURCE" ]] || die "Source path not found: $SOURCE"
  if [[ -n "$REF" ]]; then
    command -v git >/dev/null 2>&1 || die "git not found"
    if git -C "$SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      log "Cloning local git source: $SOURCE"
      git clone "$SOURCE" "$WORK_DIR"
      log "Checking out ref: $REF"
      git -C "$WORK_DIR" checkout "$REF"
    else
      die "--ref requires a git source"
    fi
  else
    copy_source "$SOURCE"
  fi
fi

# Basic sanity
[[ -f "$WORK_DIR/requirements_qt.txt" ]] || die "requirements_qt.txt not found in source"
[[ -d "$WORK_DIR/hmfit_gui_qt" ]] || die "hmfit_gui_qt not found in source"

# If user accidentally has a venv inside hmfit/ (common name collision), drop it from the copied tree.
if [[ -f "$WORK_DIR/hmfit/pyvenv.cfg" || -d "$WORK_DIR/hmfit/bin" ]]; then
  warn "Found venv artifacts inside hmfit/ in the source tree copy. Removing them for this build."
  rm -rf "$WORK_DIR/hmfit/bin" "$WORK_DIR/hmfit/include" "$WORK_DIR/hmfit/lib" \
         "$WORK_DIR/hmfit/lib64" "$WORK_DIR/hmfit/share" "$WORK_DIR/hmfit/pyvenv.cfg" || true
fi

log "Creating venv (inside build dir, not in source)"
VENV_DIR="$BUILD_DIR/venv"
python3 -m venv --copies "$VENV_DIR"
PY="$VENV_DIR/bin/python3"

log "Installing Python dependencies"
"$PY" -m pip install --upgrade pip wheel setuptools
"$PY" -m pip install -r "$WORK_DIR/requirements_qt.txt" pyinstaller

# Write entrypoint OUTSIDE the source tree
ENTRY_DIR="$BUILD_DIR/entry"
mkdir -p "$ENTRY_DIR"
ENTRY="$ENTRY_DIR/hmfit_pyside6_entry.py"
cat > "$ENTRY" <<'PY'
from __future__ import annotations

from hmfit_gui_qt.__main__ import main as gui_main

# Force PyInstaller to include Qt and GUI modules.
import hmfit_gui_qt.main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(gui_main())
PY

# Force PyInstaller outputs OUTSIDE the source tree too
PYI_DIST="$BUILD_DIR/pyinstaller_dist"
PYI_WORK="$BUILD_DIR/pyinstaller_build"
PYI_SPEC="$BUILD_DIR/pyinstaller_spec"
mkdir -p "$PYI_DIST" "$PYI_WORK" "$PYI_SPEC"

log "Building PyInstaller binary (out-of-tree)"
pushd "$WORK_DIR" >/dev/null
"$PY" -m PyInstaller --noconfirm --clean \
  --name hmfit_pyside6 \
  --windowed \
  --onefile \
  --paths "$WORK_DIR" \
  --distpath "$PYI_DIST" \
  --workpath "$PYI_WORK" \
  --specpath "$PYI_SPEC" \
  "$ENTRY"
popd >/dev/null

BIN="$PYI_DIST/hmfit_pyside6"
[[ -x "$BIN" ]] || die "PyInstaller output not found/executable: $BIN"

APPDIR="$BUILD_DIR/AppDir"
mkdir -p "$APPDIR/usr/bin" \
  "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/scalable/apps"

cp -f "$BIN" "$APPDIR/usr/bin/hmfit_pyside6"

APPIMAGE_TEMPLATE_DIR="$WORK_DIR/packaging/linux/appimage"
if [[ -f "$APPIMAGE_TEMPLATE_DIR/AppRun" ]]; then
  cp -f "$APPIMAGE_TEMPLATE_DIR/AppRun" "$APPDIR/AppRun"
  chmod +x "$APPDIR/AppRun"
else
  ln -sf usr/bin/hmfit_pyside6 "$APPDIR/AppRun"
fi

if [[ -f "$WORK_DIR/hmfit.svg" ]]; then
  cp -f "$WORK_DIR/hmfit.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/hmfit.svg"
  cp -f "$WORK_DIR/hmfit.svg" "$APPDIR/hmfit.svg"
fi

DESKTOP_FILE="$APPDIR/usr/share/applications/hmfit_pyside6.desktop"
if [[ -f "$APPIMAGE_TEMPLATE_DIR/hmfit.desktop" ]]; then
  cp -f "$APPIMAGE_TEMPLATE_DIR/hmfit.desktop" "$DESKTOP_FILE"
else
  cat > "$DESKTOP_FILE" <<'EOF'
[Desktop Entry]
Type=Application
Name=HM Fit (PySide6)
Exec=hmfit_pyside6
Icon=hmfit
Categories=Science;Education;
Terminal=false
EOF
fi
cp -f "$DESKTOP_FILE" "$APPDIR/hmfit_pyside6.desktop"

APPIMAGETOOL=""
if [[ -n "$APPIMAGETOOL_PATH" ]]; then
  APPIMAGETOOL="$APPIMAGETOOL_PATH"
elif command -v appimagetool >/dev/null 2>&1; then
  APPIMAGETOOL="$(command -v appimagetool)"
else
  TOOL_DIR="$BUILD_DIR/tools"
  mkdir -p "$TOOL_DIR"
  APPIMAGETOOL="$TOOL_DIR/appimagetool-x86_64.AppImage"
  if [[ ! -x "$APPIMAGETOOL" ]]; then
    log "Downloading appimagetool"
    if command -v curl >/dev/null 2>&1; then
      curl -L -o "$APPIMAGETOOL" \
        "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$APPIMAGETOOL" \
        "https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    else
      die "curl or wget is required to download appimagetool"
    fi
    chmod +x "$APPIMAGETOOL"
  fi
fi

[[ -x "$APPIMAGETOOL" ]] || die "appimagetool not found or not executable: $APPIMAGETOOL"

# Output file selection
if [[ -n "$OUT_FILE" ]]; then
  mkdir -p "$(dirname "$OUT_FILE")"
  FINAL_OUT="$OUT_FILE"
else
  FINAL_OUT="$OUT_DIR/hmfit_pyside6-${ARCH}.AppImage"
fi

log "Building AppImage: $FINAL_OUT"
if [[ -r /dev/fuse && -w /dev/fuse ]]; then
  ARCH="$ARCH" "$APPIMAGETOOL" "$APPDIR" "$FINAL_OUT"
else
  APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" "$APPIMAGETOOL" "$APPDIR" "$FINAL_OUT"
fi

log "Done: $FINAL_OUT"