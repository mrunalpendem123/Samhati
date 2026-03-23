use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Status payload returned to the frontend.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeStatus {
    pub running: bool,
    pub model: Option<String>,
    pub elo_score: i32,
    pub inferences_served: u64,
    pub uptime_secs: Option<i64>,
}

/// Manages the inference node lifecycle.
pub struct NodeManager {
    running: Arc<AtomicBool>,
    start_time: Option<DateTime<Utc>>,
    model_name: Option<String>,
}

impl NodeManager {
    pub fn new(running: Arc<AtomicBool>) -> Self {
        Self {
            running,
            start_time: None,
            model_name: None,
        }
    }

    /// Start the inference node with the given model.
    pub async fn start(&mut self, model_name: String) -> Result<String, String> {
        if self.running.load(Ordering::Relaxed) {
            return Err("Node is already running".into());
        }

        // TODO: Actually spawn samhati mesh-node as a subprocess or embed it.
        // For now we simulate the start.
        self.model_name = Some(model_name.clone());
        self.start_time = Some(Utc::now());
        self.running.store(true, Ordering::Relaxed);

        Ok(format!("Node started with model: {}", model_name))
    }

    /// Stop the inference node.
    pub async fn stop(&mut self) -> Result<(), String> {
        if !self.running.load(Ordering::Relaxed) {
            return Err("Node is not running".into());
        }

        // TODO: Gracefully shut down mesh-node subprocess.
        self.running.store(false, Ordering::Relaxed);
        self.model_name = None;
        self.start_time = None;

        Ok(())
    }

    /// Get current status.
    pub fn status(&self, elo: i32, inferences: u64) -> NodeStatus {
        let uptime_secs = self.start_time.map(|t| (Utc::now() - t).num_seconds());
        NodeStatus {
            running: self.running.load(Ordering::Relaxed),
            model: self.model_name.clone(),
            elo_score: elo,
            inferences_served: inferences,
            uptime_secs,
        }
    }
}
