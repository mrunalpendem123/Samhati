use std::sync::atomic::Ordering;

use tauri::State;

use crate::model_manager::{DashboardStats, DomainStat, EloPoint, ModelInfo};
use crate::node::NodeStatus;
use crate::state::AppState;
use crate::wallet::WalletInfo;

// ── Node commands ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn start_node(model_name: String, state: State<'_, AppState>) -> Result<String, String> {
    if state.node_running.load(Ordering::Relaxed) {
        return Err("Node is already running".into());
    }

    state.node_running.store(true, Ordering::Relaxed);
    *state.current_model.write().map_err(|e| e.to_string())? = Some(model_name.clone());
    *state.node_start_time.write().map_err(|e| e.to_string())? = Some(chrono::Utc::now());

    Ok(format!("Node started with model: {}", model_name))
}

#[tauri::command]
pub async fn stop_node(state: State<'_, AppState>) -> Result<(), String> {
    if !state.node_running.load(Ordering::Relaxed) {
        return Err("Node is not running".into());
    }

    state.node_running.store(false, Ordering::Relaxed);
    *state.current_model.write().map_err(|e| e.to_string())? = None;
    *state.node_start_time.write().map_err(|e| e.to_string())? = None;

    Ok(())
}

#[tauri::command]
pub async fn get_node_status(state: State<'_, AppState>) -> Result<NodeStatus, String> {
    let running = state.node_running.load(Ordering::Relaxed);
    let model = state
        .current_model
        .read()
        .map_err(|e| e.to_string())?
        .clone();
    let elo = state.elo_score.load(Ordering::Relaxed);
    let inferences = state.total_inferences_served.load(Ordering::Relaxed);
    let uptime_secs = state
        .node_start_time
        .read()
        .map_err(|e| e.to_string())?
        .map(|t| (chrono::Utc::now() - t).num_seconds());

    Ok(NodeStatus {
        running,
        model,
        elo_score: elo,
        inferences_served: inferences,
        uptime_secs,
    })
}

// ── Chat commands ──────────────────────────────────────────────────────────

#[tauri::command]
pub async fn send_chat(
    message: String,
    mode: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    if message.trim().is_empty() {
        return Err("Message cannot be empty".into());
    }

    let n = match mode.as_str() {
        "quick" => 3,
        "best" => 7,
        _ => return Err(format!("Unknown mode: {}. Use 'quick' or 'best'.", mode)),
    };

    // TODO: Route the query through the Samhati mesh network.
    // This would call into the proximity-router to find N nearest specialist
    // nodes, fan out the query, collect responses, and return the best one
    // based on the ELO-weighted selection.

    let model = state
        .current_model
        .read()
        .map_err(|e| e.to_string())?
        .clone()
        .unwrap_or_else(|| "llama-3.2-3b".to_string());

    // Simulate a response for development
    let response = format!(
        "This is a simulated response from the Samhati mesh network.\n\n\
         **Query**: {}\n\
         **Mode**: {} (N={})\n\
         **Model**: {}\n\n\
         In production, this response would come from the nearest {} inference nodes \
         in the decentralized network, with the best answer selected via ELO-weighted voting.",
        message, mode, n, model, n
    );

    // Increment inference counter when node is running
    if state.node_running.load(Ordering::Relaxed) {
        state
            .total_inferences_served
            .fetch_add(1, Ordering::Relaxed);
    }

    Ok(response)
}

// ── Wallet commands ────────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_wallet_info(state: State<'_, AppState>) -> Result<WalletInfo, String> {
    let pubkey = state
        .wallet_pubkey
        .read()
        .map_err(|e| e.to_string())?
        .clone()
        .unwrap_or_else(|| "Not initialized".to_string());

    let balance = state.smti_balance.load(Ordering::Relaxed);
    let pending = state.pending_rewards.load(Ordering::Relaxed);

    Ok(WalletInfo {
        pubkey,
        balance,
        pending_rewards: pending,
    })
}

#[tauri::command]
pub async fn claim_rewards(state: State<'_, AppState>) -> Result<u64, String> {
    let pending = state.pending_rewards.load(Ordering::Relaxed);
    if pending == 0 {
        return Err("No pending rewards to claim".into());
    }

    // TODO: Sign and submit Solana transaction
    state
        .smti_balance
        .fetch_add(pending, Ordering::Relaxed);
    state.pending_rewards.store(0, Ordering::Relaxed);

    Ok(pending)
}

// ── Model commands ─────────────────────────────────────────────────────────

#[tauri::command]
pub async fn list_models(
    app_handle: tauri::AppHandle,
) -> Result<Vec<ModelInfo>, String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| dirs::data_dir().unwrap_or_default().join("samhati"));

    let manager = crate::model_manager::ModelManager::new(&data_dir);
    Ok(manager.list_models())
}

#[tauri::command]
pub async fn download_model(
    model_id: String,
    app_handle: tauri::AppHandle,
) -> Result<(), String> {
    let data_dir = app_handle
        .path()
        .app_data_dir()
        .unwrap_or_else(|_| dirs::data_dir().unwrap_or_default().join("samhati"));

    let manager = crate::model_manager::ModelManager::new(&data_dir);
    manager.download_model(&model_id).await
}

// ── Dashboard commands ─────────────────────────────────────────────────────

#[tauri::command]
pub async fn get_dashboard_stats(state: State<'_, AppState>) -> Result<DashboardStats, String> {
    let elo = state.elo_score.load(Ordering::Relaxed);
    let inferences = state.total_inferences_served.load(Ordering::Relaxed);
    let balance = state.smti_balance.load(Ordering::Relaxed);

    let uptime_secs = state
        .node_start_time
        .read()
        .map_err(|e| e.to_string())?
        .map(|t| (chrono::Utc::now() - t).num_seconds())
        .unwrap_or(0);

    let elo_history = state
        .elo_history
        .read()
        .map_err(|e| e.to_string())?
        .iter()
        .map(|s| EloPoint {
            timestamp: s.timestamp.to_rfc3339(),
            score: s.score,
        })
        .collect();

    Ok(DashboardStats {
        earnings_today: balance / 3, // Simulated breakdown
        earnings_week: balance / 2,
        earnings_total: balance,
        inferences_served: inferences,
        uptime_secs,
        elo_score: elo,
        elo_history,
        domain_breakdown: vec![
            DomainStat {
                domain: "General".into(),
                count: inferences * 40 / 100,
                percentage: 40.0,
            },
            DomainStat {
                domain: "Code".into(),
                count: inferences * 25 / 100,
                percentage: 25.0,
            },
            DomainStat {
                domain: "Hindi".into(),
                count: inferences * 20 / 100,
                percentage: 20.0,
            },
            DomainStat {
                domain: "Math".into(),
                count: inferences * 15 / 100,
                percentage: 15.0,
            },
        ],
        network_nodes_online: 1_247,
        network_inferences_24h: 89_432,
    })
}
