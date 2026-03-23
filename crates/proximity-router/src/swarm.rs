//! Swarm-mode peer registry for open-mesh distributed inference.
//!
//! Unlike a fixed cluster, the `SwarmRegistry` is a soft-state store:
//! - Peers register themselves by broadcasting `CapabilityPayload` gossip messages.
//! - Registrations expire after a configurable TTL if the peer stops announcing.
//! - Each peer carries EWMA reputation [0,1] updated on every RPC outcome.
//! - `best_peers_for_layers()` picks the optimal peers for any layer range,
//!   combining latency, bandwidth, reliability, and reputation.
//!
//! ## Heartbeat protocol
//!
//! Peers send a lightweight heartbeat every `heartbeat_interval` (default 2s).
//! `SwarmRegistry::check_heartbeats()` counts consecutive misses; after
//! `max_misses` (default 3) a peer is marked `failed` and excluded from
//! routing.  Failed peers are automatically recovered when a new heartbeat
//! arrives.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use crate::{PeerMetrics, ProximityRouter};

// ── Primitive types ───────────────────────────────────────────────────────────

pub type PeerId = String;

/// Minimum verified layers served before a peer is considered trusted.
/// Trusted peers are eligible to serve the privacy-sensitive first 4 layers
/// (layers 0–3), which carry the highest prompt-reconstruction risk.
const TRUSTED_TIER_THRESHOLD: u64 = 10_000;

/// Default heartbeat interval (2 seconds).
const DEFAULT_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(2);

/// Default maximum consecutive heartbeat misses before marking a peer failed.
const DEFAULT_MAX_MISSES: u32 = 3;

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
    /// Total inference requests this peer has served for others.
    pub inferences_served: u64,
    /// Total inference requests this peer has consumed from the mesh.
    pub inferences_consumed: u64,
    /// blake3 hashes of weight shards this peer has cached (for P2P transfer).
    pub cached_shard_hashes: Vec<String>,

    // ── Heartbeat state ───────────────────────────────────────────────────
    /// Number of consecutive heartbeat misses.
    pub consecutive_misses: u32,
    /// Whether this peer has been marked as failed (>= max_misses).
    pub failed: bool,
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
            inferences_served: 0,
            inferences_consumed: 0,
            cached_shard_hashes: Vec::new(),
            consecutive_misses: 0,
            failed: false,
        }
    }

    /// A peer is trusted if it has served at least `TRUSTED_TIER_THRESHOLD`
    /// verified layers.  Only trusted peers may serve the privacy-sensitive
    /// first 4 layers of a model (Prompt Inference Attack mitigation).
    pub fn is_trusted(&self) -> bool {
        self.inferences_served >= TRUSTED_TIER_THRESHOLD
    }

    /// Seeding ratio: served / consumed.  Returns 1.0 if no consumption yet.
    /// Peers that serve more than they consume get a routing advantage.
    pub fn seed_ratio(&self) -> f64 {
        if self.inferences_consumed == 0 {
            return 1.0; // new peers start neutral
        }
        self.inferences_served as f64 / self.inferences_consumed as f64
    }

    /// True if this peer can fully serve `[layer_start, layer_end)` for `model`
    /// and is not currently marked as failed.
    pub fn can_serve(&self, model: &str, layer_start: usize, layer_end: usize) -> bool {
        !self.failed && self.layers.iter().any(|lr| {
            lr.model == model
                && lr.layer_start <= layer_start
                && lr.layer_end >= layer_end
        })
    }

    /// Composite routing score: lower is better.
    ///
    /// Blends the proximity-router base score with a reputation penalty,
    /// a load penalty, and a seeding ratio penalty so busy/unreliable/freeloading
    /// peers are deprioritised.
    pub fn routing_score(&self, router: &ProximityRouter) -> f64 {
        let base = router.score_peer(&self.metrics);
        // Reputation penalty: poor reputation → +0.3 extra cost
        let rep_penalty = (1.0 - self.reputation.clamp(0.0, 1.0)) * 0.3;
        // Load penalty: saturated peer → +0.2 extra cost
        let load_penalty = self.load_score.clamp(0.0, 1.0) * 0.2;
        // Seed penalty: freeloaders (seed_ratio < 1.0) get deprioritised.
        // Peers serving more than they consume get a small bonus.
        let seed_penalty = if self.seed_ratio() < 1.0 {
            (1.0 - self.seed_ratio()) * 0.25  // up to +0.25 cost for pure consumers
        } else {
            0.0
        };
        base + rep_penalty + load_penalty + seed_penalty
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
    /// Expected interval between heartbeats from each peer.
    heartbeat_interval: Duration,
    /// Number of consecutive misses before a peer is marked failed.
    max_misses: u32,
}

impl SwarmRegistry {
    pub fn new(peer_ttl: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            peer_ttl,
            router: ProximityRouter::default(),
            heartbeat_interval: DEFAULT_HEARTBEAT_INTERVAL,
            max_misses: DEFAULT_MAX_MISSES,
        }
    }

    /// Create a registry with custom heartbeat parameters.
    pub fn with_heartbeat(
        peer_ttl: Duration,
        heartbeat_interval: Duration,
        max_misses: u32,
    ) -> Self {
        Self {
            peers: HashMap::new(),
            peer_ttl,
            router: ProximityRouter::default(),
            heartbeat_interval,
            max_misses: max_misses.max(1),
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
        // A capability announcement counts as a heartbeat
        peer.consecutive_misses = 0;
        if peer.failed {
            peer.failed = false;
        }
    }

    /// Remove peers that have not sent an announcement within `peer_ttl`.
    /// Returns the number of peers evicted.
    pub fn expire_stale(&mut self) -> usize {
        let before = self.peers.len();
        self.peers.retain(|_, p| p.last_seen.elapsed() < self.peer_ttl);
        before - self.peers.len()
    }

    // ── Heartbeat protocol ────────────────────────────────────────────────

    /// Record a heartbeat from a peer, resetting its miss counter and
    /// recovering it if it was previously marked failed.
    pub fn record_heartbeat(&mut self, peer_id: &str) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            p.last_seen = Instant::now();
            p.consecutive_misses = 0;
            if p.failed {
                p.failed = false;
            }
        }
    }

    /// Check all peers for heartbeat misses.
    ///
    /// Call this periodically (e.g. every 1 second).  For each peer whose
    /// `last_seen` exceeds `heartbeat_interval × (consecutive_misses + 1)`,
    /// increments the miss counter.  When a peer reaches `max_misses`, it is
    /// marked `failed` and excluded from routing.
    ///
    /// Returns the IDs of peers that were **newly** marked as failed in this
    /// check (for triggering chain reselection).
    pub fn check_heartbeats(&mut self) -> Vec<PeerId> {
        let mut newly_failed = Vec::new();
        let interval = self.heartbeat_interval;
        let max = self.max_misses;

        for peer in self.peers.values_mut() {
            if peer.failed {
                continue; // already failed, skip
            }

            let expected_deadline = interval * (peer.consecutive_misses + 1);
            if peer.last_seen.elapsed() > expected_deadline {
                peer.consecutive_misses += 1;
                if peer.consecutive_misses >= max {
                    peer.failed = true;
                    newly_failed.push(peer.id.clone());
                }
            }
        }

        newly_failed
    }

    /// Number of peers currently marked as failed.
    pub fn failed_peer_count(&self) -> usize {
        self.peers.values().filter(|p| p.failed).count()
    }

    /// All peers currently marked as failed.
    pub fn failed_peers(&self) -> Vec<&SwarmPeer> {
        self.peers.values().filter(|p| p.failed).collect()
    }

    // ── Layer-based queries ───────────────────────────────────────────────

    /// All non-failed peers that can serve `[layer_start, layer_end)` for `model`.
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
    ///
    /// **Privacy constraint**: layers 0–3 carry the highest prompt-reconstruction
    /// risk (near-one-to-one embedding-to-token correspondence).  When
    /// `layer_start < 4`, only trusted-tier peers (>10 000 verified layers served)
    /// are eligible.  If no trusted peers are available, falls back to all peers
    /// to avoid complete service denial.
    ///
    /// Failed peers are automatically excluded via `can_serve()`.
    pub fn best_peers_for_layers(
        &self,
        model: &str,
        layer_start: usize,
        layer_end: usize,
        count: usize,
    ) -> Vec<&SwarmPeer> {
        let mut candidates = self.peers_for_layers(model, layer_start, layer_end);

        // Enforce trusted-tier for privacy-sensitive first 4 layers.
        if layer_start < 4 {
            let trusted: Vec<&SwarmPeer> = candidates
                .iter()
                .copied()
                .filter(|p| p.is_trusted())
                .collect();
            if !trusted.is_empty() {
                candidates = trusted;
            }
            // If no trusted peers exist, fall back to all candidates rather
            // than refusing service entirely (graceful degradation).
        }

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

    // ── Reputation updates ────────────────────────────────────────────────

    /// Record a successful RPC for `peer_id` with the given observed latency.
    /// Updates the EWMA reputation, RTT estimate, and increments inferences_served.
    pub fn record_success(&mut self, peer_id: &str, latency_ms: f64) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            // EWMA with α = 0.15
            p.reputation = p.reputation * 0.85 + 1.0 * 0.15;
            p.metrics.rtt_ms = p.metrics.rtt_ms * 0.85 + latency_ms * 0.15;
            p.inferences_served += 1;
        }
    }

    /// Record a failed RPC for `peer_id`.
    pub fn record_failure(&mut self, peer_id: &str) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            // EWMA with α = 0.15 toward 0
            p.reputation = p.reputation * 0.85;
        }
    }

    /// Increment the inference-consumed counter for a peer (called by the
    /// requesting side after an inference completes).
    pub fn record_consumption(&mut self, peer_id: &str) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            p.inferences_consumed += 1;
        }
    }

    /// Update the cached shard hashes for a peer (from gossip announcements).
    pub fn update_cached_shards(&mut self, peer_id: &str, hashes: Vec<String>) {
        if let Some(p) = self.peers.get_mut(peer_id) {
            p.cached_shard_hashes = hashes;
        }
    }

    /// Find peers that have a specific weight shard cached (by blake3 hash).
    pub fn peers_with_shard(&self, blake3_hex: &str) -> Vec<&SwarmPeer> {
        self.peers
            .values()
            .filter(|p| p.cached_shard_hashes.iter().any(|h| h == blake3_hex))
            .collect()
    }

    // ── Read helpers ──────────────────────────────────────────────────────

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
    fn trusted_tier_enforced_for_early_layers() {
        let mut reg = SwarmRegistry::new(Duration::from_secs(60));

        // Untrusted peer (low inferences_served)
        reg.upsert(
            "untrusted",
            metrics(5.0),
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 32 }],
            0.0,
            0,
        );

        // Trusted peer (>10,000 inferences_served)
        reg.upsert(
            "trusted",
            metrics(10.0),  // slightly higher RTT, but trusted
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 32 }],
            0.0,
            0,
        );
        // Simulate enough served inferences
        if let Some(p) = reg.peers.get_mut("trusted") {
            p.inferences_served = 15_000;
        }

        // For layers 0..4, the trusted peer should be preferred
        let best = reg.best_peers_for_layers("m", 0, 4, 1);
        assert_eq!(best.len(), 1);
        assert_eq!(best[0].id, "trusted");

        // For layers 4..8, untrusted peer with lower RTT is preferred
        let best = reg.best_peers_for_layers("m", 4, 8, 1);
        assert_eq!(best[0].id, "untrusted");
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

    // ── Heartbeat tests ──────────────────────────────────────────────────

    #[test]
    fn heartbeat_miss_marks_peer_failed() {
        let mut reg = SwarmRegistry::with_heartbeat(
            Duration::from_secs(60),
            Duration::from_millis(10),  // fast heartbeat for test
            3,
        );

        reg.upsert(
            "peer-a",
            metrics(10.0),
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }],
            0.0,
            0,
        );

        // Wait for 1 heartbeat interval and check — should get 1 miss
        std::thread::sleep(Duration::from_millis(15));
        let failed = reg.check_heartbeats();
        assert!(failed.is_empty()); // only 1 miss, need 3
        assert_eq!(reg.peers.get("peer-a").unwrap().consecutive_misses, 1);

        // Wait another interval
        std::thread::sleep(Duration::from_millis(15));
        let failed = reg.check_heartbeats();
        assert!(failed.is_empty()); // 2 misses
        assert_eq!(reg.peers.get("peer-a").unwrap().consecutive_misses, 2);

        // Wait another interval — should reach 3 misses
        std::thread::sleep(Duration::from_millis(15));
        let failed = reg.check_heartbeats();
        assert_eq!(failed.len(), 1);
        assert_eq!(failed[0], "peer-a");
        assert!(reg.peers.get("peer-a").unwrap().failed);
    }

    #[test]
    fn heartbeat_resets_on_receive() {
        let mut reg = SwarmRegistry::with_heartbeat(
            Duration::from_secs(60),
            Duration::from_millis(10),
            3,
        );

        reg.upsert(
            "peer-a",
            metrics(10.0),
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }],
            0.0,
            0,
        );

        // Accumulate 2 misses
        std::thread::sleep(Duration::from_millis(15));
        reg.check_heartbeats();
        std::thread::sleep(Duration::from_millis(15));
        reg.check_heartbeats();
        assert_eq!(reg.peers.get("peer-a").unwrap().consecutive_misses, 2);

        // Receive heartbeat — resets
        reg.record_heartbeat("peer-a");
        assert_eq!(reg.peers.get("peer-a").unwrap().consecutive_misses, 0);
    }

    #[test]
    fn failed_peer_excluded_from_routing() {
        let mut reg = SwarmRegistry::new(Duration::from_secs(60));

        reg.upsert(
            "peer-a",
            metrics(10.0),
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }],
            0.0,
            0,
        );
        reg.upsert(
            "peer-b",
            metrics(20.0),
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }],
            0.0,
            0,
        );

        // Mark peer-a as failed
        if let Some(p) = reg.peers.get_mut("peer-a") {
            p.failed = true;
        }

        // peer-a should be excluded; peer-b should be the only candidate
        let best = reg.best_peer_for_layers("m", 0, 8).expect("should find peer-b");
        assert_eq!(best.id, "peer-b");
    }

    #[test]
    fn failed_peer_recovers_on_heartbeat() {
        let mut reg = SwarmRegistry::new(Duration::from_secs(60));

        reg.upsert(
            "peer-a",
            metrics(10.0),
            vec![LayerRange { model: "m".into(), layer_start: 0, layer_end: 8, total_layers: 8 }],
            0.0,
            0,
        );

        // Mark as failed
        if let Some(p) = reg.peers.get_mut("peer-a") {
            p.failed = true;
            p.consecutive_misses = 3;
        }

        // Cannot serve while failed
        assert!(reg.best_peer_for_layers("m", 0, 8).is_none());

        // Heartbeat arrives — recovers
        reg.record_heartbeat("peer-a");
        assert!(!reg.peers.get("peer-a").unwrap().failed);
        assert!(reg.best_peer_for_layers("m", 0, 8).is_some());
    }
}
