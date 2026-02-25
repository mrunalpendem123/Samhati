use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};

/// Assignment of a layer range to a specific peer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardSpec {
    pub peer_id: String,
    pub layer_start: usize,
    pub layer_end: usize,
    /// Total layers in the full model — lets each shard detect whether it is
    /// the first (layer_start == 0) or last (layer_end == total_layers) shard.
    pub total_layers: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShardPlan {
    pub model: String,
    pub total_layers: usize,
    pub shards: Vec<ShardSpec>,
}

// ── Planner ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RoundRobinPlanner {
    peers: Vec<String>,
    min_layers_per_peer: usize,
}

impl RoundRobinPlanner {
    pub fn new(peers: Vec<String>, min_layers_per_peer: usize) -> Self {
        Self {
            peers,
            min_layers_per_peer: min_layers_per_peer.max(1),
        }
    }

    pub fn plan(&self, model: &str, total_layers: usize) -> Result<ShardPlan> {
        if self.peers.is_empty() {
            return Err(anyhow!("no peers provided"));
        }
        if total_layers == 0 {
            return Err(anyhow!("total_layers must be > 0"));
        }

        let peers = self.peers.len();
        let min_layers = self.min_layers_per_peer;

        if total_layers < min_layers * peers {
            return Err(anyhow!(
                "not enough layers ({total_layers}) for {peers} peers \
                 with min_layers_per_peer={min_layers}"
            ));
        }

        let remaining = total_layers - (min_layers * peers);
        let base_extra = remaining / peers;
        let rem = remaining % peers;

        let mut shards = Vec::new();
        let mut cursor = 0usize;
        for (idx, peer_id) in self.peers.iter().enumerate() {
            let extra = if idx < rem { 1 } else { 0 };
            let count = min_layers + base_extra + extra;
            let start = cursor;
            let end = cursor + count;
            cursor = end;
            shards.push(ShardSpec {
                peer_id: peer_id.clone(),
                layer_start: start,
                layer_end: end,
                total_layers: Some(total_layers),
            });
        }

        Ok(ShardPlan {
            model: model.to_string(),
            total_layers,
            shards,
        })
    }
}
