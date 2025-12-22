#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build HM Fit (PySide6) AppImage from an external source without dirtying the repo.

Usage:
  build_hmfit_pyside6_appimage.sh --source <path|git-url> [options]

Options:
  --ref <git-ref>       Git ref (tag/branch/commit) to checkout
  --out <dir>           Output directory for AppImage
                        Default: $HOME/BUILD_HMFIT_PYSIDE6/output
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
die() { echo "[error] $*" >&2; exit 1; }

require_arg() {
  if [[ -z "${2:-}" ]]; then
    die "Missing value for $1"
  fi
}

SOURCE=""
REF=""
OUT_DIR=""
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

if ! command -v python3 >/dev/null 2>&1; then
  die "python3 not found"
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" ]]; then
  die "Unsupported architecture: $ARCH (only x86_64 supported)"
fi

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

if [[ "$is_git_url" -eq 1 ]]; then
  if ! command -v git >/dev/null 2>&1; then
    die "git not found"
  fi
  log "Cloning source: $SOURCE"
  git clone "$SOURCE" "$WORK_DIR"
  if [[ -n "$REF" ]]; then
    log "Checking out ref: $REF"
    git -C "$WORK_DIR" checkout "$REF"
  fi
else
  if [[ ! -d "$SOURCE" ]]; then
    die "Source path not found: $SOURCE"
  fi
  if [[ -n "$REF" ]]; then
    if ! command -v git >/dev/null 2>&1; then
      die "git not found"
    fi
    if git -C "$SOURCE" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      log "Cloning local git source: $SOURCE"
      git clone "$SOURCE" "$WORK_DIR"
      log "Checking out ref: $REF"
      git -C "$WORK_DIR" checkout "$REF"
    else
      die "--ref requires a git source"
    fi
  else
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
        "$SOURCE"/ "$WORK_DIR"/
    else
      log "rsync not found; copying source with cp -a"
      cp -a "$SOURCE"/. "$WORK_DIR"/
    fi
  fi
fi

if [[ ! -f "$WORK_DIR/requirements_qt.txt" ]]; then
  die "requirements_qt.txt not found in source"
fi
if [[ ! -d "$WORK_DIR/hmfit_gui_qt" ]]; then
  die "hmfit_gui_qt not found in source"
fi

log "Creating venv"
python3 -m venv "$BUILD_DIR/.venv"
# shellcheck disable=SC1091
source "$BUILD_DIR/.venv/bin/activate"

log "Installing Python dependencies"
python -m pip install --upgrade pip wheel
python -m pip install -r "$WORK_DIR/requirements_qt.txt" pyinstaller

ENTRY="$WORK_DIR/.hmfit_pyside6_entry.py"
cat > "$ENTRY" <<'PY'
from __future__ import annotations

from hmfit_gui_qt.__main__ import main as gui_main

# Force PyInstaller to include Qt and GUI modules.
import hmfit_gui_qt.main  # noqa: F401


if __name__ == "__main__":
    raise SystemExit(gui_main())
PY

log "Building PyInstaller binary"
pushd "$WORK_DIR" >/dev/null
pyinstaller --noconfirm --clean --name hmfit_pyside6 --windowed --onefile "$ENTRY"
popd >/dev/null

BIN="$WORK_DIR/dist/hmfit_pyside6"
if [[ ! -x "$BIN" ]]; then
  die "PyInstaller output not found: $BIN"
fi

APPDIR="$BUILD_DIR/AppDir"
mkdir -p "$APPDIR/usr/bin" \
  "$APPDIR/usr/share/applications" \
  "$APPDIR/usr/share/icons/hicolor/scalable/apps"

cp -f "$BIN" "$APPDIR/usr/bin/hmfit_pyside6"
ln -sf usr/bin/hmfit_pyside6 "$APPDIR/AppRun"

if [[ -f "$WORK_DIR/hmfit.svg" ]]; then
  cp -f "$WORK_DIR/hmfit.svg" "$APPDIR/usr/share/icons/hicolor/scalable/apps/hmfit.svg"
  cp -f "$WORK_DIR/hmfit.svg" "$APPDIR/hmfit.svg"
fi

cat > "$APPDIR/usr/share/applications/hmfit_pyside6.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=HM Fit (PySide6)
Exec=hmfit_pyside6
Icon=hmfit
Categories=Science;Education;
Terminal=false
EOF

cp -f "$APPDIR/usr/share/applications/hmfit_pyside6.desktop" "$APPDIR/hmfit_pyside6.desktop"

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

if [[ ! -x "$APPIMAGETOOL" ]]; then
  die "appimagetool not found or not executable: $APPIMAGETOOL"
fi

OUT_FILE="$OUT_DIR/hmfit_pyside6-${ARCH}.AppImage"
log "Building AppImage: $OUT_FILE"
if [[ -r /dev/fuse && -w /dev/fuse ]]; then
  ARCH="$ARCH" "$APPIMAGETOOL" "$APPDIR" "$OUT_FILE"
else
  APPIMAGE_EXTRACT_AND_RUN=1 ARCH="$ARCH" "$APPIMAGETOOL" "$APPDIR" "$OUT_FILE"
fi

log "Done: $OUT_FILE"
