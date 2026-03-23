// Prevents additional console window on Windows in release.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod model_manager;
mod node;
mod state;
mod wallet;

use state::AppState;
use wallet::WalletManager;

fn main() {
    let app_state = AppState::default();

    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(app_state)
        .setup(|app| {
            // Initialize wallet on startup
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| dirs::data_dir().unwrap_or_default().join("samhati"));

            let wallet_mgr = WalletManager::new(&data_dir);
            match wallet_mgr.load_or_create() {
                Ok((pubkey, secret)) => {
                    let state = app.state::<AppState>();
                    if let Ok(mut pk) = state.wallet_pubkey.write() {
                        *pk = Some(pubkey);
                    }
                    if let Ok(mut sk) = state.wallet_secret.write() {
                        *sk = Some(secret);
                    }
                }
                Err(e) => {
                    eprintln!("Warning: Failed to initialize wallet: {}", e);
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_node,
            commands::stop_node,
            commands::get_node_status,
            commands::send_chat,
            commands::get_wallet_info,
            commands::claim_rewards,
            commands::list_models,
            commands::download_model,
            commands::get_dashboard_stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running Samhati");
}
