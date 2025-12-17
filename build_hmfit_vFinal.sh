#!/bin/bash

# ==============================================================================
# Script: build_hmfit_vFinal.sh
# Descripción: Automatiza la compilación de HM Fit (Tauri + FastAPI) a AppImage.
#              Crea un entorno limpio, compila el backend Python y luego Tauri.
# ==============================================================================

set -e  # Salir inmediatamente si un comando falla

# --- CONFIGURACIÓN ---
PROJECT_ROOT="/mnt/HDD_4TB/GUI_HM_fit"
BUILD_DIR="/home/ccachyavgp/Documentos/BUILD_HMfit"
OUTPUT_DIR="$PROJECT_ROOT/dist_appimage" # Carpeta final para el AppImage

# Colores para logs
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[BUILD]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# ==============================================================================
# 1. PREPARACIÓN DEL ENTORNO
# ==============================================================================
log "1. Preparando entorno de compilación..."

if [ -d "$BUILD_DIR" ]; then
    warn "Limpiando directorio de build existente: $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi
mkdir -p "$BUILD_DIR"

log "Copiando proyecto a $BUILD_DIR (excluyendo basura)..."
rsync -av --progress "$PROJECT_ROOT/" "$BUILD_DIR/" \
    --exclude '.venv' \
    --exclude 'node_modules' \
    --exclude '.git' \
    --exclude 'dist' \
    --exclude 'build' \
    --exclude '__pycache__' \
    --exclude '.idea' \
    --exclude 'src-tauri/target' \
    --exclude 'hmfit_tauri/node_modules' \
    --exclude 'hmfit_tauri/src-tauri/target'

cd "$BUILD_DIR" || error "No se pudo entrar al directorio de build"

# ==============================================================================
# 2. PARCHEO DINÁMICO DE PYTHON (BACKEND)
# ==============================================================================
log "2. Parcheando backend para ejecución congelada..."

# 2.1 Crear __init__.py
touch backend_fastapi/__init__.py

# 2.2 Inyectar lógica de rutas en main.py
MAIN_PY="backend_fastapi/main.py"
log "Modificando $MAIN_PY..."

# Bloque de código para inyectar al inicio
cat << 'EOF' > temp_header.py
import sys
import os

# --- INYECCIÓN PARA FROZEN APP ---
if getattr(sys, 'frozen', False):
    # Si estamos en un ejecutable (PyInstaller), el path base es sys._MEIPASS
    base_dir = sys._MEIPASS
    # Ajustar sys.path para que encuentre los módulos internos
    sys.path.insert(0, base_dir)
    # TAMBIÉN añadir backend_fastapi al path para imports directos (como spectroscopy_processor)
    sys.path.insert(0, os.path.join(base_dir, 'backend_fastapi'))
    
    # Cambiar al directorio de trabajo temporal para evitar problemas de rutas relativas
    os.chdir(base_dir)
# ---------------------------------

EOF

# Concatenar el header con el archivo original
cat temp_header.py "$MAIN_PY" > "${MAIN_PY}.tmp" && mv "${MAIN_PY}.tmp" "$MAIN_PY"
rm temp_header.py

# 2.3 Corregir Uvicorn (string -> app object, reload=False)
# Buscamos: uvicorn.run("backend_fastapi.main:app", ... reload=True, ...)
# Reemplazamos con: uvicorn.run(app, ... reload=False, ...)
# Nota: Usamos sed con cuidado. Asumimos la estructura del archivo original.
sed -i 's/uvicorn.run(/uvicorn.run(app, # Modified by build script/g' "$MAIN_PY"
sed -i 's/"backend_fastapi.main:app",//g' "$MAIN_PY"
sed -i 's/reload=True/reload=False/g' "$MAIN_PY"

# 2.4 Generar backend.spec
log "Generando backend.spec..."
cat << EOF > backend.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['backend_fastapi/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'backend_fastapi.spectroscopy_processor',
        'backend_fastapi.nmr_processor',
        'backend_fastapi.config',
        'backend_fastapi.errors',
        'np_backend',
        'LM_conc_algoritm',
        'NR_conc_algoritm',
        'core_ad_probe',
        'noncoop_utils',
        'utils',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='hmfit-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='hmfit-backend',
)
EOF

# ==============================================================================
# 3. COMPILACIÓN DEL BACKEND
# ==============================================================================
log "3. Compilando Backend (PyInstaller)..."

python3 -m venv .build_venv
source .build_venv/bin/activate

log "Instalando dependencias..."
pip install --upgrade pip wheel
pip install pyinstaller
pip install -r backend_fastapi/requirements.txt

log "Ejecutando PyInstaller..."
pyinstaller backend.spec --clean --noconfirm

if [ ! -d "dist/hmfit-backend" ]; then
    error "No se encontró el directorio dist/hmfit-backend. La compilación falló."
fi

log "Backend compilado exitosamente."
deactivate

# ==============================================================================
# 4. PREPARACIÓN DE TAURI (SIDECAR)
# ==============================================================================
log "4. Configurando Tauri Sidecar..."

# 4.1 Detectar arquitectura
TRIPLE=$(rustc -vV | grep "host:" | cut -d " " -f 2)
log "Arquitectura detectada: $TRIPLE"

# 4.2 Preparar directorio de binarios
BINARIES_DIR="hmfit_tauri/src-tauri/binaries"
mkdir -p "$BINARIES_DIR"

# 4.3 Copiar y renombrar el binario
# PyInstaller en modo onedir crea una carpeta. Tauri espera un ejecutable único si es sidecar,
# PERO si es onedir, necesitamos un script wrapper o empaquetarlo como onefile.
# EL USUARIO PIDIÓ "convertir tu FastAPI en un binario ejecutable".
# Si usamos onedir (COLLECT en spec), tenemos una carpeta. Tauri sidecar espera un archivo.
# CAMBIO ESTRATEGIA: Usaremos 'onefile' para simplificar el sidecar de Tauri,
# O si usamos onedir, necesitamos que el sidecar sea un script que llame al binario real.
# Para simplificar y seguir la solicitud de "ejecutable único", modificaré el spec para onefile.

log "Ajustando spec para ONEFILE (necesario para sidecar simple)..."
# Reescribimos el spec para onefile
cat << EOF > backend.spec
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['backend_fastapi/main.py'],
    pathex=['.'],
    binaries=[],
    datas=[],
    hiddenimports=[
        'backend_fastapi.spectroscopy_processor',
        'backend_fastapi.nmr_processor',
        'backend_fastapi.config',
        'backend_fastapi.errors',
        'np_backend',
        'LM_conc_algoritm',
        'NR_conc_algoritm',
        'core_ad_probe',
        'noncoop_utils',
        'utils',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='hmfit-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF

# Re-compilar como onefile
source .build_venv/bin/activate
log "Re-compilando como ONEFILE..."
pyinstaller backend.spec --clean --noconfirm
deactivate

# Copiar el binario único
cp "dist/hmfit-backend" "$BINARIES_DIR/hmfit-backend-$TRIPLE"
chmod +x "$BINARIES_DIR/hmfit-backend-$TRIPLE"

# 4.4 Parchear Cargo.toml
CARGO_TOML="hmfit_tauri/src-tauri/Cargo.toml"
log "Parcheando $CARGO_TOML..."

# Añadir features a tauri dependency
# Usamos sed para buscar la línea de tauri = { ... } y reemplazarla
sed -i 's/^tauri = { version = "1", features = \[.*\] }/tauri = { version = "1", features = [ "fs-write-file", "dialog-save", "shell-open", "process-command-api", "custom-protocol"] }/' "$CARGO_TOML"

# Añadir sección [features] si no existe
if ! grep -q "\[features\]" "$CARGO_TOML"; then
    echo -e "\n[features]\ncustom-protocol = [\"tauri/custom-protocol\"]" >> "$CARGO_TOML"
fi

# 4.5 Parchear tauri.conf.json
TAURI_CONF="hmfit_tauri/src-tauri/tauri.conf.json"
log "Parcheando $TAURI_CONF..."

# Usamos jq si está disponible, si no, sed (más arriesgado pero portable si no hay jq)
# Asumimos que jq podría no estar, así que intentamos usar python para editar el json de forma segura
python3 -c "
import json
import sys

with open('$TAURI_CONF', 'r') as f:
    data = json.load(f)

# Añadir externalBin
if 'tauri' not in data:
    data['tauri'] = {}
if 'bundle' not in data['tauri']:
    data['tauri']['bundle'] = {}

# Ensure identifier exists (Required for Tauri build)
if 'identifier' not in data['tauri']['bundle']:
    data['tauri']['bundle']['identifier'] = 'com.hmfit.app'

# Force bundle active and targets
data['tauri']['bundle']['active'] = True
data['tauri']['bundle']['targets'] = 'all'

data['tauri']['bundle']['externalBin'] = ['binaries/hmfit-backend']

# Asegurar allowlist process
if 'allowlist' not in data['tauri']:
    data['tauri']['allowlist'] = {}

data['tauri']['allowlist']['shell'] = {
    'all': False,
    'open': True,
    'execute': True,
    'sidecar': True
}

with open('$TAURI_CONF', 'w') as f:
    json.dump(data, f, indent=2)
"

# Debug: Mostrar tauri.conf.json final
log "Contenido final de tauri.conf.json:"
cat "$TAURI_CONF"

# ==============================================================================
# 4.6 GENERACIÓN DE ICONOS (FIX: Missing square icon)
# ==============================================================================
log "4.6 Verificando iconos..."
ICONS_DIR="hmfit_tauri/src-tauri/icons"
SOURCE_ICON="$ICONS_DIR/icon.png"

if [ -f "$SOURCE_ICON" ]; then
    log "Generando iconos estándar desde $SOURCE_ICON..."
    # Asegurar que existan los iconos requeridos por Tauri AppImage
    # 32x32, 128x128, 256x256 (512x512 ya suele ser el source)
    
    # Intentar usar convert (ImageMagick)
    if command -v convert &> /dev/null; then
        # Force resize to square with !
        convert "$SOURCE_ICON" -resize 32x32! "$ICONS_DIR/32x32.png"
        convert "$SOURCE_ICON" -resize 128x128! "$ICONS_DIR/128x128.png"
        convert "$SOURCE_ICON" -resize 128x128! "$ICONS_DIR/128x128@2x.png"
        convert "$SOURCE_ICON" -resize 256x256! "$ICONS_DIR/icon.icns"
        convert "$SOURCE_ICON" -resize 512x512! "$ICONS_DIR/512x512.png"
        log "Iconos generados con ImageMagick (Forzados a cuadrados)."
    else
        warn "ImageMagick (convert) no encontrado. Copiando icon.png como fallback..."
        cp "$SOURCE_ICON" "$ICONS_DIR/32x32.png"
        cp "$SOURCE_ICON" "$ICONS_DIR/128x128.png"
        cp "$SOURCE_ICON" "$ICONS_DIR/128x128@2x.png"
        cp "$SOURCE_ICON" "$ICONS_DIR/512x512.png"
    fi
    
    # Parchear tauri.conf.json para asegurar que use estos iconos
    # Aunque por defecto busca en icons/, vamos a ser explícitos
    python3 -c "
import json
with open('$TAURI_CONF', 'r') as f:
    data = json.load(f)

if 'tauri' in data and 'bundle' in data['tauri']:
    data['tauri']['bundle']['icon'] = [
        'icons/32x32.png',
        'icons/128x128.png',
        'icons/128x128@2x.png',
        'icons/512x512.png',
        'icons/icon.icns'
    ]

with open('$TAURI_CONF', 'w') as f:
    json.dump(data, f, indent=2)
"
else
    warn "No se encontró $SOURCE_ICON. La compilación podría fallar si no hay iconos."
fi

# ==============================================================================
# 4.7 PARCHEO DE MAIN.RS (FIX: Runtime error)
# ==============================================================================
log "4.7 Parcheando main.rs para usar Sidecar..."
MAIN_RS="hmfit_tauri/src-tauri/src/main.rs"

# Sobrescribir main.rs con la lógica correcta para Sidecar
cat << 'EOF' > "$MAIN_RS"
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, Mutex};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;
use tauri::{Manager, RunEvent};
use tauri::api::process::{Command, CommandEvent};

struct BackendHandle(Arc<Mutex<Option<tauri::api::process::CommandChild>>>);

const BACKEND_PORT: &str = "8001";

fn port_in_use(addr: &str) -> bool {
    let addrs_iter = addr
        .to_socket_addrs()
        .ok()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();

    for a in addrs_iter {
        if TcpStream::connect_timeout(&a, Duration::from_millis(200)).is_ok() {
            return true;
        }
    }
    false
}

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let backend_addr = format!("127.0.0.1:{}", BACKEND_PORT);
            
            if port_in_use(&backend_addr) {
                println!("[MAIN] Backend ya está activo en {} (se reutiliza).", backend_addr);
                app.manage(BackendHandle(Arc::new(Mutex::new(None))));
            } else {
                println!("[MAIN] Iniciando Sidecar en {}...", backend_addr);
                let (mut rx, child) = Command::new_sidecar("hmfit-backend")
                    .expect("failed to create `hmfit-backend` binary command")
                    .spawn()
                    .expect("Failed to spawn sidecar");

                // Leer stdout/stderr del sidecar en un hilo separado
                tauri::async_runtime::spawn(async move {
                    while let Some(event) = rx.recv().await {
                        if let CommandEvent::Stdout(line) = event {
                            println!("[BACKEND] {}", line);
                        } else if let CommandEvent::Stderr(line) = event {
                            eprintln!("[BACKEND ERR] {}", line);
                        }
                    }
                });

                app.manage(BackendHandle(Arc::new(Mutex::new(Some(child)))));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            match event {
                RunEvent::Exit | RunEvent::ExitRequested { .. } => {
                    let state = app_handle.state::<BackendHandle>();
                    let mut guard = state.0.lock().unwrap();
                    if let Some(child) = guard.take() {
                        // Matar el sidecar al salir
                        let _ = child.kill();
                    }
                }
                _ => {}
            }
        });
}
EOF

# ==============================================================================
# 5. COMPILACIÓN FINAL Y ENTREGA
# ==============================================================================
log "5. Compilando Tauri AppImage..."

cd hmfit_tauri
npm install

log "Ejecutando tauri build..."
# NO_STRIP=true es importante para algunas distros como Arch/CachyOS para evitar problemas con símbolos
env NO_STRIP=true npm run tauri build -- --bundles appimage --verbose

# 5.1 Extraer resultado
TARGET_DIR="src-tauri/target/release/bundle/appimage"
APPIMAGE=$(find "$TARGET_DIR" -name "*.AppImage" | head -n 1)

if [ -f "$APPIMAGE" ]; then
    log "AppImage creada: $APPIMAGE"
    mkdir -p "$OUTPUT_DIR"
    cp "$APPIMAGE" "$OUTPUT_DIR/"
    log "¡ÉXITO! AppImage copiada a: $OUTPUT_DIR"
    echo -e "${GREEN}Build finalizado correctamente.${NC}"
else
    error "No se encontró la AppImage generada en $TARGET_DIR"
    log "Contenido de src-tauri/target/release:"
    ls -R src-tauri/target/release
fi
