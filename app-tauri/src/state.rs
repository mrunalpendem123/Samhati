use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU64};
use std::sync::RwLock;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Global application state, shared across Tauri commands via `tauri::State`.
pub struct AppState {
    pub node_running: AtomicBool,
    pub elo_score: AtomicI32,
    pub smti_balance: AtomicU64,
    pub total_inferences_served: AtomicU64,
    pub current_model: RwLock<Option<String>>,
    pub wallet_pubkey: RwLock<Option<String>>,
    pub wallet_secret: RwLock<Option<Vec<u8>>>,
    pub pending_rewards: AtomicU64,
    pub node_start_time: RwLock<Option<DateTime<Utc>>>,
    pub elo_history: RwLock<Vec<EloSnapshot>>,
    pub inference_log: RwLock<Vec<InferenceRecord>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloSnapshot {
    pub timestamp: DateTime<Utc>,
    pub score: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceRecord {
    pub timestamp: DateTime<Utc>,
    pub model: String,
    pub domain: String,
    pub latency_ms: u64,
}

impl Default for AppState {
    fn default() -> Self {
        Self {
            node_running: AtomicBool::new(false),
            elo_score: AtomicI32::new(1200),
            smti_balance: AtomicU64::new(0),
            total_inferences_served: AtomicU64::new(0),
            current_model: RwLock::new(None),
            wallet_pubkey: RwLock::new(None),
            wallet_secret: RwLock::new(None),
            pending_rewards: AtomicU64::new(0),
            node_start_time: RwLock::new(None),
            elo_history: RwLock::new(Vec::new()),
            inference_log: RwLock::new(Vec::new()),
        }
    }
}
