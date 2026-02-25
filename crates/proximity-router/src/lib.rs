use std::cmp::Ordering;

pub mod swarm;
pub use swarm::{LayerRange, SwarmPeer, SwarmRegistry};

pub type PeerId = String;

#[derive(Debug, Clone)]
pub struct PeerMetrics {
    pub rtt_ms: f64,
    pub bandwidth_mbps: f64,
    pub reliability: f64,
    pub gpu_capacity_score: f64,
    pub free_vram_gb: Option<f64>,
    pub last_seen_epoch_ms: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub id: PeerId,
    pub metrics: PeerMetrics,
}

#[derive(Debug, Clone)]
pub struct RankedPeer {
    pub id: PeerId,
    pub score: f64,
    pub metrics: PeerMetrics,
}

#[derive(Debug, Clone)]
pub struct ScoreWeights {
    pub w_latency: f64,
    pub w_bandwidth: f64,
    pub w_reliability: f64,
    pub w_gpu_capacity: f64,
}

impl Default for ScoreWeights {
    fn default() -> Self {
        Self {
            w_latency: 0.55,
            w_bandwidth: 0.15,
            w_reliability: 0.20,
            w_gpu_capacity: 0.10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RouterConfig {
    pub rtt_threshold_ms: f64,
    pub min_reliability: f64,
    pub bandwidth_norm_mbps: f64,
    pub rtt_norm_ms: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            rtt_threshold_ms: 60.0,
            min_reliability: 0.85,
            bandwidth_norm_mbps: 1000.0,
            rtt_norm_ms: 50.0,
        }
    }
}

pub trait RttProbe {
    fn measure_rtt_ms(&self, peer: &PeerId) -> Option<f64>;
}

#[derive(Debug, Clone)]
pub struct ProximityRouter {
    pub config: RouterConfig,
    pub weights: ScoreWeights,
}

impl Default for ProximityRouter {
    fn default() -> Self {
        Self {
            config: RouterConfig::default(),
            weights: ScoreWeights::default(),
        }
    }
}

impl ProximityRouter {
    pub fn score_peer(&self, metrics: &PeerMetrics) -> f64 {
        let latency_score = clamp01(metrics.rtt_ms / self.config.rtt_norm_ms);

        let bandwidth_score = 1.0 - clamp01(metrics.bandwidth_mbps / self.config.bandwidth_norm_mbps);

        let reliability_score = 1.0 - clamp01(metrics.reliability);
        let gpu_score = 1.0 - clamp01(metrics.gpu_capacity_score);

        (self.weights.w_latency * latency_score)
            + (self.weights.w_bandwidth * bandwidth_score)
            + (self.weights.w_reliability * reliability_score)
            + (self.weights.w_gpu_capacity * gpu_score)
    }

    pub fn rank_peers<I>(&self, peers: I) -> Vec<RankedPeer>
    where
        I: IntoIterator<Item = PeerInfo>,
    {
        let mut ranked: Vec<RankedPeer> = peers
            .into_iter()
            .filter(|p| p.metrics.rtt_ms <= self.config.rtt_threshold_ms)
            .filter(|p| p.metrics.reliability >= self.config.min_reliability)
            .map(|p| {
                let score = self.score_peer(&p.metrics);
                RankedPeer {
                    id: p.id,
                    score,
                    metrics: p.metrics,
                }
            })
            .collect();

        ranked.sort_by(|a, b| cmp_score(a.score, b.score));
        ranked
    }
}

fn clamp01(v: f64) -> f64 {
    if v.is_nan() {
        return 1.0;
    }
    if v < 0.0 {
        0.0
    } else if v > 1.0 {
        1.0
    } else {
        v
    }
}

fn cmp_score(a: f64, b: f64) -> Ordering {
    if a.is_nan() && b.is_nan() {
        Ordering::Equal
    } else if a.is_nan() {
        Ordering::Greater
    } else if b.is_nan() {
        Ordering::Less
    } else {
        a.partial_cmp(&b).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_by_latency_when_other_scores_equal() {
        let router = ProximityRouter::default();
        let peers = vec![
            PeerInfo {
                id: "a".to_string(),
                metrics: PeerMetrics {
                    rtt_ms: 10.0,
                    bandwidth_mbps: 1000.0,
                    reliability: 1.0,
                    gpu_capacity_score: 1.0,
                    free_vram_gb: Some(24.0),
                    last_seen_epoch_ms: None,
                },
            },
            PeerInfo {
                id: "b".to_string(),
                metrics: PeerMetrics {
                    rtt_ms: 30.0,
                    bandwidth_mbps: 1000.0,
                    reliability: 1.0,
                    gpu_capacity_score: 1.0,
                    free_vram_gb: Some(24.0),
                    last_seen_epoch_ms: None,
                },
            },
        ];

        let ranked = router.rank_peers(peers);
        assert_eq!(ranked[0].id, "a");
        assert_eq!(ranked[1].id, "b");
    }
}
