#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    net::{TcpStream, ToSocketAddrs},
    process::{Child, Command},
    sync::{Arc, Mutex},
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
            // Si el puerto ya está en uso, asumimos que el backend está corriendo (otra instancia o manual)
            let backend_addr = format!("127.0.0.1:{}", BACKEND_PORT);
            if port_in_use(backend_addr.as_str()) {
                eprintln!(
                    "Backend ya está activo en {} (se reutiliza, no se lanza otro).",
                    backend_addr
                );
                app.manage(BackendHandle(Arc::new(Mutex::new(None))));
            } else {
                // Para que el binario empaquetado (AppImage, .deb, etc.) sea portable, no asumimos
                // que existe una ruta fija a un venv dentro del repo.
                //
                // Si quieres que la app levante el backend automáticamente, exporta:
                // - HM_FIT_SPAWN_BACKEND=1
                // - HM_FIT_BACKEND_CMD="python3 -m backend_fastapi.main"
                //
                // (y asegúrate de que el comando y sus dependencias existan en el sistema).
                let should_spawn = std::env::var_os("HM_FIT_SPAWN_BACKEND").is_some();
                let backend_cmd = std::env::var("HM_FIT_BACKEND_CMD").ok();

                if should_spawn {
                    let Some(cmd) = backend_cmd else {
                        eprintln!(
                            "HM_FIT_SPAWN_BACKEND está activo pero falta HM_FIT_BACKEND_CMD; no se lanza backend."
                        );
                        app.manage(BackendHandle(Arc::new(Mutex::new(None))));
                        return Ok(());
                    };

                    let child = if cfg!(target_os = "windows") {
                        Command::new("cmd")
                            .args(["/C", &cmd])
                            .env("HM_BACKEND_PORT", BACKEND_PORT)
                            .spawn()
                    } else {
                        Command::new("sh")
                            .args(["-lc", &cmd])
                            .env("HM_BACKEND_PORT", BACKEND_PORT)
                            .spawn()
                    };

                    match child {
                        Ok(child) => {
                            app.manage(BackendHandle(Arc::new(Mutex::new(Some(child)))));
                        }
                        Err(err) => {
                            eprintln!("No se pudo lanzar el backend ({cmd}): {err}");
                            app.manage(BackendHandle(Arc::new(Mutex::new(None))));
                        }
                    }
                } else {
                    eprintln!(
                        "Backend no detectado en {} (y autolanzado deshabilitado); inicia el backend manualmente si lo necesitas.",
                        backend_addr
                    );
                    app.manage(BackendHandle(Arc::new(Mutex::new(None))));
                }
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
