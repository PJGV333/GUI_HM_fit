#!/usr/bin/env bash
set -euo pipefail

APPDIR=${APPDIR:-AppDir}

if [[ ! -d "${APPDIR}" ]]; then
  echo "AppDir directory \"${APPDIR}\" not found. Please set APPDIR to your staging directory." >&2
  exit 1
fi

SYSTEM_FONT="/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
DEST_DIR="${APPDIR}/usr/share/GUI_HM_fit/assets/fonts"
FONT_DEST="${DEST_DIR}/DejaVuSansMono.ttf"

if [[ -f "${SYSTEM_FONT}" ]]; then
  install -Dm644 "${SYSTEM_FONT}" "${FONT_DEST}"
else
  echo "Warning: system font \"${SYSTEM_FONT}\" was not found. " \
       "Ensure DejaVu Sans Mono is available in the runtime." >&2
fi
