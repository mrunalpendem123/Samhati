use proximity_router::{PeerId, RankedPeer};

#[derive(Debug, Clone)]
pub struct ClusterConstraints {
    pub min_nodes: usize,
    pub max_nodes: usize,
    pub min_bandwidth_mbps: Option<f64>,
}

impl Default for ClusterConstraints {
    fn default() -> Self {
        Self {
            min_nodes: 2,
            max_nodes: 6,
            min_bandwidth_mbps: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterSelection {
    pub nodes: Vec<PeerId>,
    pub required_vram_gb: f64,
    pub nodes_count: usize,
}

#[derive(Debug, Clone)]
pub enum ClusterError {
    InvalidConstraints,
    NoEligiblePeers { required_vram_gb: f64 },
    NotEnoughPeers { needed: usize, available: usize },
}

impl std::fmt::Display for ClusterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidConstraints => write!(f, "invalid cluster constraints"),
            Self::NoEligiblePeers { required_vram_gb } => {
                write!(f, "no peers meet required_vram_gb={required_vram_gb}")
            }
            Self::NotEnoughPeers { needed, available } => {
                write!(f, "needed {needed} peers, only {available} eligible")
            }
        }
    }
}

impl std::error::Error for ClusterError {}

pub fn select_cluster<F>(
    ranked: &[RankedPeer],
    constraints: &ClusterConstraints,
    required_vram_for_nodes: F,
) -> Result<ClusterSelection, ClusterError>
where
    F: Fn(usize) -> f64,
{
    if constraints.min_nodes == 0 || constraints.max_nodes < constraints.min_nodes {
        return Err(ClusterError::InvalidConstraints);
    }

    for nodes in constraints.min_nodes..=constraints.max_nodes {
        let required_vram_gb = required_vram_for_nodes(nodes);
        let mut eligible: Vec<&RankedPeer> = ranked
            .iter()
            .filter(|p| {
                if let Some(min_bw) = constraints.min_bandwidth_mbps {
                    p.metrics.bandwidth_mbps >= min_bw
                } else {
                    true
                }
            })
            .filter(|p| p.metrics.free_vram_gb.unwrap_or(0.0) >= required_vram_gb)
            .collect();

        if eligible.is_empty() {
            continue;
        }

        if eligible.len() < nodes {
            continue;
        }

        eligible.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap());
        let nodes_selected: Vec<PeerId> = eligible
            .into_iter()
            .take(nodes)
            .map(|p| p.id.clone())
            .collect();

        return Ok(ClusterSelection {
            nodes: nodes_selected,
            required_vram_gb,
            nodes_count: nodes,
        });
    }

    let required_vram_gb = required_vram_for_nodes(constraints.min_nodes);
    let available = ranked
        .iter()
        .filter(|p| p.metrics.free_vram_gb.unwrap_or(0.0) >= required_vram_gb)
        .count();

    if available == 0 {
        Err(ClusterError::NoEligiblePeers { required_vram_gb })
    } else {
        Err(ClusterError::NotEnoughPeers {
            needed: constraints.min_nodes,
            available,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximity_router::PeerMetrics;

    fn peer(id: &str, score: f64, free_vram_gb: f64) -> RankedPeer {
        RankedPeer {
            id: id.to_string(),
            score,
            metrics: PeerMetrics {
                rtt_ms: 10.0,
                bandwidth_mbps: 1000.0,
                reliability: 1.0,
                gpu_capacity_score: 1.0,
                free_vram_gb: Some(free_vram_gb),
                last_seen_epoch_ms: None,
            },
        }
    }

    #[test]
    fn selects_smallest_cluster_that_fits() {
        let ranked = vec![peer("a", 0.1, 10.0), peer("b", 0.2, 10.0), peer("c", 0.3, 5.0)];

        let constraints = ClusterConstraints {
            min_nodes: 2,
            max_nodes: 4,
            min_bandwidth_mbps: None,
        };

        let selected = select_cluster(&ranked, &constraints, |nodes| 9.0 / nodes as f64)
            .expect("should select");

        assert_eq!(selected.nodes_count, 2);
        assert_eq!(selected.nodes.len(), 2);
        assert_eq!(selected.nodes[0], "a");
    }
}
