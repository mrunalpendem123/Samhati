//! Dynamic shard planner that selects peers from the live `SwarmRegistry`.
//!
//! Unlike `RoundRobinPlanner` (which distributes layers across a caller-supplied
//! list of peer IDs), `SwarmPlanner` queries the registry for whoever is best
//! positioned to serve each layer range right now.  The result is a `ShardPlan`
//! identical in structure to the static planner's output and consumed by the
//! same `Coordinator` / `IrohDistributedExecutor` path.

use anyhow::{anyhow, Result};
use proximity_router::SwarmRegistry;

use crate::plan::{ShardPlan, ShardSpec};

// ── SwarmPlanner ──────────────────────────────────────────────────────────────

/// Builds a `ShardPlan` dynamically from a `SwarmRegistry` snapshot.
///
/// `layers_per_shard` is the target number of transformer layers per peer.
/// The last shard may receive fewer layers if `total_layers` is not divisible.
pub struct SwarmPlanner {
    pub layers_per_shard: usize,
}

impl SwarmPlanner {
    pub fn new(layers_per_shard: usize) -> Self {
        Self {
            layers_per_shard: layers_per_shard.max(1),
        }
    }

    /// Build a shard plan for `model` (a model registry name) with `total_layers`
    /// total transformer layers, querying `registry` for the best available peers.
    ///
    /// Returns `Err` if any layer range has no available peers in the registry.
    pub fn plan(
        &self,
        model: &str,
        total_layers: usize,
        registry: &SwarmRegistry,
    ) -> Result<ShardPlan> {
        if total_layers == 0 {
            return Err(anyhow!("total_layers must be > 0"));
        }

        let mut shards = Vec::new();
        let mut cursor = 0usize;

        while cursor < total_layers {
            let end = (cursor + self.layers_per_shard).min(total_layers);

            let peer = registry
                .best_peer_for_layers(model, cursor, end)
                .ok_or_else(|| {
                    anyhow!(
                        "no swarm peer can serve model={model} layers={cursor}..{end} \
                         (registry has {} peer(s))",
                        registry.peer_count()
                    )
                })?;

            shards.push(ShardSpec {
                peer_id: peer.id.clone(),
                layer_start: cursor,
                layer_end: end,
                total_layers: Some(total_layers),
            });

            cursor = end;
        }

        Ok(ShardPlan {
            model: model.to_string(),
            total_layers,
            shards,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proximity_router::{swarm::LayerRange, PeerMetrics, SwarmRegistry};
    use std::time::Duration;

    fn metrics() -> PeerMetrics {
        PeerMetrics {
            rtt_ms: 10.0,
            bandwidth_mbps: 1000.0,
            reliability: 1.0,
            gpu_capacity_score: 1.0,
            free_vram_gb: Some(24.0),
            last_seen_epoch_ms: None,
        }
    }

    #[test]
    fn plan_with_two_peers() {
        let mut reg = SwarmRegistry::new(Duration::from_secs(60));
        reg.upsert(
            "peer-a",
            metrics(),
            vec![LayerRange { model: "test".into(), layer_start: 0, layer_end: 16, total_layers: 32 }],
            0.0,
            0,
        );
        reg.upsert(
            "peer-b",
            metrics(),
            vec![LayerRange { model: "test".into(), layer_start: 16, layer_end: 32, total_layers: 32 }],
            0.0,
            0,
        );

        let planner = SwarmPlanner::new(16);
        let plan = planner.plan("test", 32, &reg).expect("should plan");
        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.shards[0].peer_id, "peer-a");
        assert_eq!(plan.shards[1].peer_id, "peer-b");
    }

    #[test]
    fn plan_fails_when_no_peer_for_range() {
        let reg = SwarmRegistry::new(Duration::from_secs(60));
        let planner = SwarmPlanner::new(8);
        assert!(planner.plan("model", 16, &reg).is_err());
    }
}
