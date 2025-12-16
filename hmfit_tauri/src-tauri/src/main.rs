#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    net::{TcpStream, ToSocketAddrs},
    process::{Child, Command},
    sync::{Arc, Mutex},
    path::PathBuf,
    time::Duration,
};

use tauri::{Manager, RunEvent};

struct BackendHandle(Arc<Mutex<Option<Child>>>);

const BACKEND_PORT: &str = "8001"; // evita choque con otros procesos en 8000

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
            // Ruta absoluta a src-tauri en LA máquina donde se compiló
            let src_tauri_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            // .../GUI_HM_fit/hmfit_tauri/src-tauri
            let project_root = src_tauri_dir
                .parent().unwrap()  // hmfit_tauri
                .parent().unwrap()  // GUI_HM_fit
                .to_path_buf();

            let python_bin = project_root.join("venv").join("bin").join("python");

            // Debug: imprimir rutas para verificación
            eprintln!("Project root: {:?}", project_root);
            eprintln!("Python binary: {:?}", python_bin);
            eprintln!("Python binary exists: {}", python_bin.exists());

            // Si el puerto ya está en uso, asumimos que el backend está corriendo (otra instancia o manual)
            let backend_addr = format!("127.0.0.1:{}", BACKEND_PORT);
            if port_in_use(backend_addr.as_str()) {
                eprintln!(
                    "Backend ya está activo en {} (se reutiliza, no se lanza otro).",
                    backend_addr
                );
                app.manage(BackendHandle(Arc::new(Mutex::new(None))));
            } else {
                let child = Command::new(&python_bin)
                    .current_dir(&project_root)
                    .env("HM_BACKEND_PORT", BACKEND_PORT)
                    .args(["-m", "backend_fastapi.main"])
                    .spawn()
                    .expect(&format!(
                        "No se pudo lanzar el backend FastAPI.\nPython binary: {:?}\nProject root: {:?}",
                        python_bin, project_root
                    ));

                app.manage(BackendHandle(Arc::new(Mutex::new(Some(child)))));
            }
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            match event {
                RunEvent::Exit | RunEvent::ExitRequested { .. } => {
                    // 👇 Bloque separado para que el MutexGuard se libere
                    {
                        let state = app_handle.state::<BackendHandle>();
                        let mut guard = state.0.lock().unwrap();
                        if let Some(mut child) = guard.take() {
                            let _ = child.kill();
                        }
                        // guard se suelta aquí
                    }
                }
                _ => {}
            }
        });
}
