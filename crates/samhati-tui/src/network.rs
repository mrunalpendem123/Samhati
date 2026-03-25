//! Automatic P2P node discovery using iroh gossip.
//!
//! Runs gossip in a dedicated thread (its own tokio runtime) because iroh's
//! GossipSender is !Send. Communicates with the main TUI thread via channels.
//!
//! Flow:
//!   1. Gossip thread starts iroh endpoint + joins Samhati topic
//!   2. Broadcasts node announcement every 10 seconds
//!   3. Listens for other nodes' announcements
//!   4. Sends discovered peers to main thread via channel
//!   5. Main thread auto-adds peers to SwarmOrchestrator

use anyhow::Result;
use futures_util::StreamExt;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointId};
use iroh_gossip::api::Event;
use iroh_gossip::{Gossip, TopicId};
use serde::{Deserialize, Serialize};
use std::sync::mpsc;

/// Hardcoded Samhati network topic — all nodes join this.
const SAMHATI_TOPIC: [u8; 32] = [
    0x53, 0x41, 0x4D, 0x48, 0x41, 0x54, 0x49, 0x00,
    0x53, 0x57, 0x41, 0x52, 0x4D, 0x00, 0x00, 0x01,
    0xDE, 0xCA, 0xF0, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
];

/// Gossip message — either a node announcement or demand update.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum GossipMessage {
    /// Node announcing itself (model, URL, identity)
    Announce(NodeAnnouncement),
    /// Node sharing its local demand stats
    Demand(DemandUpdate),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeAnnouncement {
    pub solana_pubkey: String,
    pub iroh_node_id: String,
    pub inference_url: String,
    pub model_name: String,
    pub port: u16,
}

/// Demand stats broadcast over gossip — each node shares what queries it's seeing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DemandUpdate {
    pub iroh_node_id: String,
    pub code: u64,
    pub math: u64,
    pub reasoning: u64,
    pub general: u64,
}

/// A discovered peer.
#[derive(Debug, Clone)]
pub struct DiscoveredPeer {
    pub solana_pubkey: String,
    pub iroh_node_id: String,
    pub inference_url: String,
    pub model_name: String,
}

/// Commands sent from main thread → gossip thread.
enum GossipCmd {
    SetAnnouncement(NodeAnnouncement),
    AddBootstrap(String),
    BroadcastDemand(DemandUpdate),
}

/// Handle to the P2P network, held by the main TUI thread.
pub struct NetworkHandle {
    /// Receive discovered peers (non-blocking try_recv)
    pub peer_rx: mpsc::Receiver<DiscoveredPeer>,
    /// Receive demand updates from other nodes
    pub demand_rx: mpsc::Receiver<DemandUpdate>,
    /// Send commands to the gossip thread
    cmd_tx: mpsc::Sender<GossipCmd>,
    /// Our NodeId
    pub node_id: String,
}

impl NetworkHandle {
    /// Start the P2P network in a background thread.
    /// Returns immediately — gossip runs in its own thread + runtime.
    pub fn start(secret_key_bytes: [u8; 32]) -> Result<Self> {
        let (peer_tx, peer_rx) = mpsc::channel::<DiscoveredPeer>();
        let (demand_tx, demand_rx) = mpsc::channel::<DemandUpdate>();
        let (cmd_tx, cmd_rx) = mpsc::channel::<GossipCmd>();
        let (node_id_tx, node_id_rx) = mpsc::channel::<String>();

        // Spawn dedicated gossip thread with its own tokio runtime
        std::thread::Builder::new()
            .name("samhati-gossip".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("gossip runtime");

                rt.block_on(gossip_loop(secret_key_bytes, peer_tx, demand_tx, cmd_rx, node_id_tx));
            })?;

        // Wait for the gossip thread to report its NodeId
        let node_id = node_id_rx
            .recv_timeout(std::time::Duration::from_secs(10))
            .unwrap_or_else(|_| "unknown".into());

        Ok(Self {
            peer_rx,
            demand_rx,
            cmd_tx,
            node_id,
        })
    }

    /// Set our announcement (call when model starts serving).
    pub fn set_announcement(&self, ann: NodeAnnouncement) {
        let _ = self.cmd_tx.send(GossipCmd::SetAnnouncement(ann));
    }

    /// Add a bootstrap peer by NodeId (hex string).
    pub fn add_bootstrap(&self, node_id: String) {
        let _ = self.cmd_tx.send(GossipCmd::AddBootstrap(node_id));
    }

    /// Broadcast our local demand stats to the network.
    pub fn broadcast_demand(&self, stats: &crate::swarm::DemandStats) {
        let _ = self.cmd_tx.send(GossipCmd::BroadcastDemand(DemandUpdate {
            iroh_node_id: self.node_id.clone(),
            code: stats.code,
            math: stats.math,
            reasoning: stats.reasoning,
            general: stats.general,
        }));
    }
}

/// The gossip event loop — runs in its own thread.
async fn gossip_loop(
    secret_key_bytes: [u8; 32],
    peer_tx: mpsc::Sender<DiscoveredPeer>,
    demand_tx: mpsc::Sender<DemandUpdate>,
    cmd_rx: mpsc::Receiver<GossipCmd>,
    node_id_tx: mpsc::Sender<String>,
) {
    // Create iroh endpoint with our unified identity key
    let secret_key = iroh::SecretKey::from_bytes(&secret_key_bytes);

    let endpoint = match Endpoint::builder()
        .secret_key(secret_key)
        .bind()
        .await
    {
        Ok(ep) => ep,
        Err(e) => {
            eprintln!("[network] Failed to bind iroh endpoint: {}", e);
            return;
        }
    };

    let our_id = endpoint.id().to_string();
    eprintln!("[network] iroh started — NodeId: {}", our_id);
    let _ = node_id_tx.send(our_id.clone());

    // Start gossip + protocol router
    let gossip = Gossip::builder().spawn(endpoint.clone());
    let _router = Router::builder(endpoint.clone())
        .accept(iroh_gossip::ALPN, gossip.clone())
        .spawn();

    let topic = TopicId::from_bytes(SAMHATI_TOPIC);

    // Subscribe to the Samhati topic (no bootstrap initially)
    let mut gossip_topic = match gossip.subscribe(topic, vec![]).await {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[network] Failed to subscribe to gossip: {}", e);
            return;
        }
    };

    eprintln!("[network] Joined Samhati gossip topic — listening for peers");

    let mut announcement: Option<NodeAnnouncement> = None;
    let mut announce_interval = tokio::time::interval(std::time::Duration::from_secs(10));

    loop {
        tokio::select! {
            // Periodically broadcast our announcement
            _ = announce_interval.tick() => {
                if let Some(ref ann) = announcement {
                    let msg = GossipMessage::Announce(ann.clone());
                    if let Ok(json) = serde_json::to_vec(&msg) {
                        let _ = gossip_topic.broadcast(json.into()).await;
                    }
                }

                // Check for commands from main thread (non-blocking)
                while let Ok(cmd) = cmd_rx.try_recv() {
                    match cmd {
                        GossipCmd::SetAnnouncement(ann) => {
                            eprintln!("[network] Announcing: {} on {}", ann.model_name, ann.inference_url);
                            announcement = Some(ann);
                        }
                        GossipCmd::AddBootstrap(peer_id) => {
                            if let Ok(id) = peer_id.parse::<EndpointId>() {
                                eprintln!("[network] Adding bootstrap peer: {}", &peer_id[..12.min(peer_id.len())]);
                                let _ = gossip.subscribe_and_join(topic, vec![id]).await;
                            }
                        }
                        GossipCmd::BroadcastDemand(update) => {
                            let msg = GossipMessage::Demand(update);
                            if let Ok(json) = serde_json::to_vec(&msg) {
                                let _ = gossip_topic.broadcast(json.into()).await;
                            }
                        }
                    }
                }
            }

            // Listen for gossip messages from other nodes
            event = gossip_topic.next() => {
                let Some(Ok(event)) = event else { break };
                if let Event::Received(msg) = event {
                    let Ok(gossip_msg) = serde_json::from_slice::<GossipMessage>(&msg.content) else {
                        continue;
                    };
                    match gossip_msg {
                        GossipMessage::Announce(ann) => {
                            if ann.iroh_node_id == our_id || ann.inference_url.is_empty() {
                                continue;
                            }
                            eprintln!(
                                "[network] Discovered: {} running {} at {}",
                                ann.iroh_node_id.get(..12).unwrap_or(&ann.iroh_node_id),
                                ann.model_name,
                                ann.inference_url,
                            );
                            let _ = peer_tx.send(DiscoveredPeer {
                                solana_pubkey: ann.solana_pubkey,
                                iroh_node_id: ann.iroh_node_id,
                                inference_url: ann.inference_url,
                                model_name: ann.model_name,
                            });
                        }
                        GossipMessage::Demand(update) => {
                            if update.iroh_node_id == our_id {
                                continue;
                            }
                            let _ = demand_tx.send(update);
                        }
                    }
                }
            }
        }
    }
}
