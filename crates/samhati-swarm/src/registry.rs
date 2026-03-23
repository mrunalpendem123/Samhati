use crate::types::NodeId;
use std::collections::HashMap;
use std::sync::RwLock;
use std::time::{Duration, Instant};

/// Metadata for a registered swarm node.
#[derive(Debug, Clone)]
pub struct NodeInfo {
    pub node_id: NodeId,
    /// iroh connection address / endpoint.
    pub endpoint: String,
    /// Model identifier, e.g. "samhati-hindi-3b-v2".
    pub model_name: String,
    /// Model size in billions of parameters.
    pub model_size_b: u8,
    /// Domain specializations advertised by this node.
    pub domain_tags: Vec<String>,
    /// Throughput benchmark.
    pub tokens_per_sec: f32,
    /// ELO rating (starts at 1500, clamped to >= 100).
    pub elo_score: i32,
    /// Last heartbeat / activity timestamp.
    pub last_seen: Instant,
    /// Optional Solana public key for SMTI settlement.
    pub solana_pubkey: Option<String>,
}

/// Thread-safe registry of swarm nodes.
pub struct ModelRegistry {
    nodes: RwLock<HashMap<NodeId, NodeInfo>>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        Self {
            nodes: RwLock::new(HashMap::new()),
        }
    }

    /// Register (or update) a node.
    pub fn register(&self, info: NodeInfo) {
        let mut map = self.nodes.write().unwrap();
        map.insert(info.node_id, info);
    }

    /// Remove a node by ID.
    pub fn remove(&self, node_id: &NodeId) {
        let mut map = self.nodes.write().unwrap();
        map.remove(node_id);
    }

    /// Select the top-N nodes by ELO, preferring nodes whose domain tags
    /// overlap with the requested domains (1.5x weight multiplier).
    pub fn select_nodes(&self, n: usize, domain_tags: &[String]) -> Vec<NodeInfo> {
        let map = self.nodes.read().unwrap();
        let mut scored: Vec<(f64, &NodeInfo)> = map
            .values()
            .map(|node| {
                let domain_match = node
                    .domain_tags
                    .iter()
                    .any(|t| domain_tags.contains(t));
                let multiplier: f64 = if domain_match { 1.5 } else { 1.0 };
                let score = node.elo_score as f64 * multiplier;
                (score, node)
            })
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored
            .into_iter()
            .take(n)
            .map(|(_, info)| info.clone())
            .collect()
    }

    /// Adjust a node's ELO by `delta` (can be negative). Clamps to >= 100.
    pub fn update_elo(&self, node_id: &NodeId, delta: i32) {
        let mut map = self.nodes.write().unwrap();
        if let Some(node) = map.get_mut(node_id) {
            node.elo_score = (node.elo_score + delta).max(100);
        }
    }

    /// Record a heartbeat (update last_seen to now).
    pub fn heartbeat(&self, node_id: &NodeId) {
        let mut map = self.nodes.write().unwrap();
        if let Some(node) = map.get_mut(node_id) {
            node.last_seen = Instant::now();
        }
    }

    /// Remove all nodes whose `last_seen` is older than `timeout`.
    pub fn prune_stale(&self, timeout: Duration) {
        let cutoff = Instant::now() - timeout;
        let mut map = self.nodes.write().unwrap();
        map.retain(|_, node| node.last_seen > cutoff);
    }

    /// Number of currently registered nodes.
    pub fn len(&self) -> usize {
        self.nodes.read().unwrap().len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_node(id_byte: u8, elo: i32, tags: Vec<&str>) -> NodeInfo {
        let mut node_id = [0u8; 32];
        node_id[0] = id_byte;
        NodeInfo {
            node_id,
            endpoint: format!("iroh://node-{id_byte}"),
            model_name: "test-model".into(),
            model_size_b: 3,
            domain_tags: tags.into_iter().map(String::from).collect(),
            tokens_per_sec: 50.0,
            elo_score: elo,
            last_seen: Instant::now(),
            solana_pubkey: None,
        }
    }

    #[test]
    fn register_and_len() {
        let reg = ModelRegistry::new();
        assert!(reg.is_empty());
        reg.register(make_node(1, 1500, vec![]));
        assert_eq!(reg.len(), 1);
        reg.register(make_node(2, 1500, vec![]));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn remove_node() {
        let reg = ModelRegistry::new();
        let n = make_node(1, 1500, vec![]);
        reg.register(n.clone());
        assert_eq!(reg.len(), 1);
        reg.remove(&n.node_id);
        assert!(reg.is_empty());
    }

    #[test]
    fn select_nodes_by_elo() {
        let reg = ModelRegistry::new();
        reg.register(make_node(1, 1200, vec![]));
        reg.register(make_node(2, 1800, vec![]));
        reg.register(make_node(3, 1500, vec![]));

        let selected = reg.select_nodes(2, &[]);
        assert_eq!(selected.len(), 2);
        // Highest ELO first
        assert_eq!(selected[0].node_id[0], 2);
        assert_eq!(selected[1].node_id[0], 3);
    }

    #[test]
    fn select_nodes_domain_boost() {
        let reg = ModelRegistry::new();
        // Node 1: lower elo but matches domain → should be boosted
        reg.register(make_node(1, 1300, vec!["math"]));
        // Node 2: higher elo, no domain match
        reg.register(make_node(2, 1600, vec!["code"]));
        // Node 3: mid elo, matches domain
        reg.register(make_node(3, 1100, vec!["math"]));

        let selected = reg.select_nodes(2, &["math".into()]);
        assert_eq!(selected.len(), 2);
        // Node 1 score: 1300 * 1.5 = 1950, Node 2 score: 1600, Node 3: 1100*1.5 = 1650
        assert_eq!(selected[0].node_id[0], 1); // 1950
        assert_eq!(selected[1].node_id[0], 3); // 1650
    }

    #[test]
    fn update_elo_clamp() {
        let reg = ModelRegistry::new();
        let n = make_node(1, 200, vec![]);
        let nid = n.node_id;
        reg.register(n);

        reg.update_elo(&nid, -150); // 200 - 150 = 50 → clamped to 100
        let selected = reg.select_nodes(1, &[]);
        assert_eq!(selected[0].elo_score, 100);
    }

    #[test]
    fn prune_stale_nodes() {
        let reg = ModelRegistry::new();
        let mut old_node = make_node(1, 1500, vec![]);
        old_node.last_seen = Instant::now() - Duration::from_secs(120);
        reg.register(old_node);
        reg.register(make_node(2, 1500, vec![])); // fresh

        reg.prune_stale(Duration::from_secs(60));
        assert_eq!(reg.len(), 1);
        let remaining = reg.select_nodes(10, &[]);
        assert_eq!(remaining[0].node_id[0], 2);
    }

    #[test]
    fn heartbeat_updates_last_seen() {
        let reg = ModelRegistry::new();
        let mut old_node = make_node(1, 1500, vec![]);
        old_node.last_seen = Instant::now() - Duration::from_secs(120);
        let nid = old_node.node_id;
        reg.register(old_node);

        reg.heartbeat(&nid);
        // After heartbeat, node should survive a 60s prune
        reg.prune_stale(Duration::from_secs(60));
        assert_eq!(reg.len(), 1);
    }
}
