#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::{
    process::{Child, Command},
    sync::{Arc, Mutex},
    path::PathBuf,
};

use tauri::{Manager, RunEvent};

struct BackendHandle(Arc<Mutex<Option<Child>>>);

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

            let child = Command::new(&python_bin)
                .current_dir(&project_root)
                .args(["-m", "backend_fastapi.main"])
                .spawn()
                .expect(&format!(
                    "No se pudo lanzar el backend FastAPI.\nPython binary: {:?}\nProject root: {:?}",
                    python_bin, project_root
                ));

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
