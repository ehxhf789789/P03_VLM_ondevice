mod analysis;
mod commands;
mod model;
mod state;

use state::AppState;
use tauri::Manager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|app| {
            let app_data_dir = app
                .path()
                .app_data_dir()
                .expect("Failed to get app data directory");
            let model_dir = app_data_dir.join("models");

            app.manage(AppState::new(model_dir));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_model_status,
            commands::download_model,
            commands::load_model,
            commands::analyze_image,
            commands::read_image_preview,
            commands::get_history,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
