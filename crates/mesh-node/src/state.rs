use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::protocol::{CapabilityPayload, ModelSummary};

#[derive(Debug, Clone)]
pub struct PendingPing {
    pub peer_id: String,
    pub started: Instant,
}

#[derive(Debug, Default)]
pub struct PeerState {
    peers: HashMap<String, CapabilityPayload>,
    models: HashMap<String, Vec<ModelSummary>>,
    rtt_ms: HashMap<String, f64>,
    last_seen_ms: HashMap<String, u128>,
    failures: HashMap<String, u32>,
    cooldown_until_ms: HashMap<String, u128>,
    pending: HashMap<String, PendingPing>,
}

impl PeerState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn upsert_capability(&mut self, caps: CapabilityPayload) {
        self.peers.insert(caps.node_id.clone(), caps);
    }

    pub fn upsert_models(&mut self, node_id: String, models: Vec<ModelSummary>) {
        self.models.insert(node_id, models);
    }

    pub fn peers_snapshot(&self) -> Vec<CapabilityPayload> {
        self.peers.values().cloned().collect()
    }

    pub fn models_snapshot(&self) -> HashMap<String, Vec<ModelSummary>> {
        self.models.clone()
    }

    pub fn set_rtt(&mut self, peer_id: String, rtt_ms: f64) {
        self.rtt_ms.insert(peer_id, rtt_ms);
    }

    pub fn rtt_snapshot(&self) -> HashMap<String, f64> {
        self.rtt_ms.clone()
    }

    pub fn mark_seen(&mut self, node_id: &str, now_ms: u128) {
        self.last_seen_ms.insert(node_id.to_string(), now_ms);
    }

    pub fn last_seen_snapshot(&self) -> HashMap<String, u128> {
        self.last_seen_ms.clone()
    }

    pub fn cooldown_snapshot(&self) -> HashMap<String, u128> {
        self.cooldown_until_ms.clone()
    }

    pub fn add_pending(&mut self, nonce: String, peer_id: String, started: Instant) {
        self.pending.insert(nonce, PendingPing { peer_id, started });
    }

    pub fn take_pending(&mut self, nonce: &str) -> Option<PendingPing> {
        self.pending.remove(nonce)
    }

    #[allow(dead_code)]
    pub fn prune_pending(&mut self, timeout: Duration) {
        let now = Instant::now();
        self.pending
            .retain(|_, ping| now.duration_since(ping.started) < timeout);
    }

    pub fn take_expired_pending(&mut self, timeout: Duration) -> Vec<PendingPing> {
        let now = Instant::now();
        let mut expired = Vec::new();
        self.pending.retain(|_, ping| {
            if now.duration_since(ping.started) < timeout {
                true
            } else {
                expired.push(ping.clone());
                false
            }
        });
        expired
    }

    pub fn register_failure(
        &mut self,
        peer_id: &str,
        now_ms: u128,
        threshold: u32,
        cooldown_ms: u64,
    ) -> bool {
        let count = self.failures.entry(peer_id.to_string()).or_insert(0);
        *count += 1;
        if *count >= threshold {
            let until = now_ms.saturating_add(cooldown_ms as u128);
            let prev = self.cooldown_until_ms.insert(peer_id.to_string(), until);
            return prev.is_none();
        }
        false
    }

    pub fn clear_failure(&mut self, peer_id: &str) {
        self.failures.remove(peer_id);
        self.cooldown_until_ms.remove(peer_id);
    }

    #[allow(dead_code)]
    pub fn is_healthy(&self, peer_id: &str, now_ms: u128) -> bool {
        match self.cooldown_until_ms.get(peer_id) {
            Some(until) => now_ms >= *until,
            None => true,
        }
    }
}
