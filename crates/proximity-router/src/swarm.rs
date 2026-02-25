//! Swarm-mode peer registry for open-mesh distributed inference.
//!
//! Unlike a fixed cluster, the `SwarmRegistry` is a soft-state store:
//! - Peers register themselves by broadcasting `CapabilityPayload` gossip messages.
//! - Registrations expire after a configurable TTL if the peer stops announcing.
//! - Each peer carries EWMA reputation [0,1] updated on every RPC outcome.
//! - `best_peers_for_layers()` picks the optimal peers for any layer range,
//!   combining latency, bandwidth, reliability, and reputation.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{PeerMetrics, ProximityRouter};

// ── Primitive types ───────────────────────────────────────────────────────────

pub type PeerId = String;

/// A contiguous range of transformer layers that a peer can serve for a model.
#[derive(Debug, Clone)]
pub struct LayerRange {
    /// Model identifier (e.g. "llama-7b", "custom")
    pub model: String,
    /// First layer index this peer hosts (inclusive).
    pub layer_start: usize,
    /// One-past-the-last layer this peer hosts.
    pub layer_end: usize,
    /// Total layers in the full model.
    pub total_layers: usize,
}

// ── SwarmPeer ─────────────────────────────────────────────────────────────────

/// A single peer in the swarm, with dynamic metrics and reputation.
#[derive(Debug, Clone)]
pub struct SwarmPeer {
    pub id: PeerId,
    pub metrics: PeerMetrics,
    /// Layer ranges this peer can serve (may span multiple models).
    pub layers: Vec<LayerRange>,
    /// EWMA reputation in [0, 1].  Starts at 0.5.  Updated on each RPC.
    pub reputation: f64,
    /// Normalised load: 0 = idle, 1 = saturated.
    pub load_score: f64,
    /// Self-reported uptime seconds.
    pub uptime_secs: u64,
    pub last_seen: Instant,
}

impl SwarmPeer {
    pub fn new(id: impl Into<PeerId>, metrics: PeerMetrics) -> Self {
        Self {
            id: id.into(),
            metrics,
            layers: Vec::new(),
            reputation: 0.5,
            load_score: 0.0,
            uptime_secs: 0,
            last_seen: Instant::now(),
        }
    }

    /// True if this peer can fully serve `[layer_start, layer_end)` for `model`.
    pub fn can_serve(&self, model: &str, layer_start: usize, layer_end: usize) -> bool {
        self.layers.iter().any(|lr| {
            lr.model == model
                && lr.layer_start <= layer_start
                && lr.layer_end >= layer_end
        })
    }

    /// Composite routing score: lower is better.
    ///
    /// Blends the proximity-router base score with a reputation penalty
    /// and a load penalty so busy/unreliable peers are deprioritised.
    pub fn routing_score(&self, router: &ProximityRouter) -> f64 {
        let base = router.score_peer(&self.metrics);
        // Reputation penalty: poor reputation → +0.3 extra cost
        let rep_penalty = (1.0 - self.reputation.clamp(0.0, 1.0)) * 0.3;
        // Load penalty: saturated peer → +0.2 extra cost
        let load_penalty = self.load_score.clamp(0.0, 1.0) * 0.2;
        base + rep_penalty + load_penalty
    }
}

// ── SwarmRegistry ─────────────────────────────────────────────────────────────

/// Soft-state registry of all known swarm peers.
///
/// Access pattern: `Arc<tokio::sync::RwLock<SwarmRegistry>>` — shared between
/// the gossip receive loop (writer) and the inference pipeline (reader).
pub struct SwarmRegistry {
    peers: HashMap<PeerId, SwarmPeer>,
    peer_ttl: Duration,
    router: ProximityRouter,
}

impl SwarmRegistry {
    pub fn new(peer_ttl: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            peer_ttl,
            router: ProximityRouter::default(),
        }
    }

    /// Upsert a peer from a gossip capability announcement.
    pub fn upsert(
        &mut self,
        id: impl Into<PeerId>,
        metrics: PeerMetrics,
        layers: Vec<LayerRange>,
        load_score: f64,
        uptime_secs: u64,
    ) {
        let id = id.into();
        let peer = self.peers.entry(id.clone()).or_insert_with(|| SwarmPeer::new(id, metrics.clone()));
        peer.metrics = metrics;
        peer.layers = layers;
        peer.load_score = load_score.clamp(0.0, 1.0);
        peer.uptime_secs = uptime_secs;
        peer.last_seen = Instant::now();
    }

    /// Remove peers that have not sent an announcement within `peer_ttl`.
    /// Returns the number of peers evicted.
    pub fn expire_stale(&mut self) -> usize {
        let before = self.peers.len();
        self.peers.retain(|_, p| p.last_seen.elapsed() < self.peer_ttl);
        before - self.peers.len()
    }

    /// All peers that can serve `[layer_start, layer_end)` for `model`.
    pub fn peers_for_layers(
        &self,
        model: &str,
        layer_start: usize,
        layer_end: usize,
    ) -> Vec<&SwarmPeer> {
        self.peers
            .values()
            .filter(|p| p.can_serve(model, layer_start, layer_end))
            .collect()
    }

    /// Best `count` peers for the given layer range, ranked by routing score.
    pub fn best_peers_for_layers(
        &self,
        model: &str,
        layer_start: usize,
        layer_end: usize,
        count: usize,
    ) -> Vec<&SwarmPeer> {
        let mut candidates = self.peers_for_layers(model, layer_start, layer_end);
        candidates.sort_by(|a, b| {
            a.routing_score(&self.router)
                .partial_cmp(&b.routing_score(&self.router))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.into_iter().take(count).collect()
    }

    /// Best peer for the given layer range (or `None` if no peers available).
    pub fn best_peer_for_layers(
        &self,
        model: &str,
        layer_start: usize,
        layer_end: usize,
    ) -> Option<&SwarmPeer> {
        self.best_peers_for_layers(model, layer_start, layer_end, 1).into_iter().next()
    }

    // ── Reputation updates ────────────────────────────────────────────────────

    /// Record a successful RPC for `peer_id` with the given observed latency.
    /// Updates the EWMA reputation and RTT estimate.
    pub fn record_success(&mut self, peer_id: &str, latency_ms: f64) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            // EWMA with α = 0.15
            p.reputation = p.reputation * 0.85 + 1.0 * 0.15;
            p.metrics.rtt_ms = p.metrics.rtt_ms * 0.85 + latency_ms * 0.15;
        }
    }

    /// Record a failed RPC for `peer_id`.
    pub fn record_failure(&mut self, peer_id: &str) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            // EWMA with α = 0.15 toward 0
            p.reputation = p.reputation * 0.85;
        }
    }

    // ── Read helpers ──────────────────────────────────────────────────────────

    pub fn all_peers(&self) -> Vec<&SwarmPeer> {
        self.peers.values().collect()
    }

    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }

    pub fn get_peer(&self, id: &str) -> Option<&SwarmPeer> {
        self.peers.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeerMetrics;
    use std::time::Duration;

    fn metrics(rtt_ms: f64) -> PeerMetrics {
        PeerMetrics {
            rtt_ms,
            bandwidth_mbps: 1000.0,
            reliability: 1.0,
            gpu_capacity_score: 1.0,
            free_vram_gb: Some(24.0),
            last_seen_epoch_ms: None,
        }
    }

    #[test]
    fn registry_selects_lower_latency_peer() {
        let mut reg = SwarmRegistry::new(Duration::from_secs(60));

        reg.upsert("peer-a", metrics(10.0), vec![LayerRange { model: "llama".into(), layer_start: 0, layer_end: 16, total_layers: 32 }], 0.0, 0);
        reg.upsert("peer-b", metrics(40.0), vec![LayerRange { model: "llama".into(), layer_start: 0, layer_end: 16, total_layers: 32 }], 0.0, 0);

        let best = reg.best_peer_for_layers("llama", 0, 16).expect("should find a peer");
        assert_eq!(best.id, "peer-a");
    }

    #[test]
    fn reputation_penalises_failing_peer() {
        let mut reg = SwarmRegistry::new(Duration::from_secs(60));

        reg.upsert("peer-a", metrics(10.0), vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }], 0.0, 0);
        reg.upsert("peer-b", metrics(10.0), vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }], 0.0, 0);

        // Repeatedly fail peer-a
        for _ in 0..20 {
            reg.record_failure("peer-a");
        }
        // peer-a now has low reputation; peer-b should be selected first
        let best = reg.best_peer_for_layers("m", 0, 8).expect("peer");
        assert_eq!(best.id, "peer-b");
    }

    #[test]
    fn stale_peers_are_evicted() {
        let mut reg = SwarmRegistry::new(Duration::from_millis(1));
        reg.upsert("stale", metrics(5.0), vec![], 0.0, 0);
        std::thread::sleep(Duration::from_millis(5));
        let evicted = reg.expire_stale();
        assert_eq!(evicted, 1);
        assert_eq!(reg.peer_count(), 0);
    }
}
