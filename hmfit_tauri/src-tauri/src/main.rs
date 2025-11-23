#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
};

use tauri::{Manager, RunEvent};

struct BackendHandle(Arc<Mutex<Option<Child>>>);

fn main() {
    tauri::Builder::default()
        .setup(|app| {
            let project_root = "/mnt/HDD_4TB/GUI_HM_fit";
            let python_bin = "/mnt/HDD_4TB/GUI_HM_fit/.venv/bin/python";

            let child = Command::new(python_bin)
                .arg("-m")
                .arg("backend_fastapi.main")
                .current_dir(project_root)
                .stdout(Stdio::inherit())
                .stderr(Stdio::inherit())
                .spawn()
                .expect("no se pudo lanzar el backend FastAPI");

            app.manage(BackendHandle(Arc::new(Mutex::new(Some(child)))));
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
