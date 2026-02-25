use anyhow::{anyhow, Result};
use async_trait::async_trait;
use iroh::{
    endpoint::{Connection, RecvStream, SendStream},
    Endpoint, EndpointId,
};
use proximity_router::SwarmRegistry;
use std::str::FromStr;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};

use crate::plan::ShardSpec;
use crate::rpc::{RpcRequest, RpcResponse, INFERENCE_ALPN};
use crate::tensor_frame::TensorFrame;
use crate::{InferenceRequest, ShardExecutor};

/// Implements `ShardExecutor` by opening a direct iroh QUIC stream to a peer,
/// forwarding the activation `TensorFrame`, and awaiting the response.
///
/// Optionally wraps a `SwarmRegistry` for **mid-token failover**: if the
/// primary peer fails, the executor automatically retries with the next-best
/// peer for the same layer range, updating reputation scores on every outcome.
#[derive(Clone)]
pub struct IrohDistributedExecutor {
    endpoint: Endpoint,
    /// Timeout (seconds) for the initial QUIC connection.
    connect_timeout_secs: u64,
    /// Timeout (seconds) for the full RPC round-trip.
    rpc_timeout_secs: u64,
    /// Maximum number of failover attempts per shard (0 = no failover).
    max_retries: usize,
    /// Shared swarm registry; used for peer reputation updates and failover.
    registry: Option<Arc<RwLock<SwarmRegistry>>>,
    /// Model name used when querying the registry for failover peers.
    model: String,
}

impl std::fmt::Debug for IrohDistributedExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IrohDistributedExecutor")
            .field("connect_timeout_secs", &self.connect_timeout_secs)
            .field("rpc_timeout_secs", &self.rpc_timeout_secs)
            .field("max_retries", &self.max_retries)
            .finish()
    }
}

impl IrohDistributedExecutor {
    pub fn new(endpoint: Endpoint) -> Self {
        Self {
            endpoint,
            connect_timeout_secs: 10,
            rpc_timeout_secs: 120,
            max_retries: 0,
            registry: None,
            model: String::new(),
        }
    }

    /// Attach a swarm registry for reputation tracking and mid-token failover.
    ///
    /// `model` is the model name used to query for alternative peers; it must
    /// match the `LayerRange::model` field stored in the registry.
    pub fn with_registry(
        mut self,
        registry: Arc<RwLock<SwarmRegistry>>,
        model: impl Into<String>,
        max_retries: usize,
    ) -> Self {
        self.registry = Some(registry);
        self.model = model.into();
        self.max_retries = max_retries;
        self
    }

    pub fn with_timeouts(mut self, connect_secs: u64, rpc_secs: u64) -> Self {
        self.connect_timeout_secs = connect_secs;
        self.rpc_timeout_secs = rpc_secs;
        self
    }

    // ── Low-level QUIC call ───────────────────────────────────────────────────

    /// Serialise `req`, send over a bi-directional QUIC stream to `peer_id`,
    /// and deserialise the response.
    async fn call_peer(&self, peer_id: &str, req: &RpcRequest) -> Result<RpcResponse> {
        let target_id = EndpointId::from_str(peer_id)
            .map_err(|e| anyhow!("invalid EndpointId '{}': {}", peer_id, e))?;

        let conn: Connection = timeout(
            Duration::from_secs(self.connect_timeout_secs),
            self.endpoint.connect(target_id, INFERENCE_ALPN),
        )
        .await
        .map_err(|_| anyhow!("timeout connecting to peer {}", peer_id))??;

        let (mut send, mut recv): (SendStream, RecvStream) = conn.open_bi().await?;

        let req_bytes = req.to_bytes()?;
        let size = (req_bytes.len() as u32).to_be_bytes();
        send.write_all(&size).await?;
        send.write_all(&req_bytes).await?;
        send.finish()?;

        let mut size_buf = [0u8; 4];
        recv.read_exact(&mut size_buf).await?;
        let resp_size = u32::from_be_bytes(size_buf) as usize;
        let mut resp_bytes = vec![0u8; resp_size];
        recv.read_exact(&mut resp_bytes).await?;

        let resp = RpcResponse::from_bytes(&resp_bytes)?;
        if let Some(err) = &resp.error {
            return Err(anyhow!("peer {} reported error: {}", peer_id, err));
        }
        Ok(resp)
    }

    // ── Single shard RPC with optional failover ───────────────────────────────

    async fn rpc(
        &self,
        shard: &ShardSpec,
        frame: TensorFrame,
        request: &InferenceRequest,
    ) -> Result<TensorFrame> {
        let req = RpcRequest {
            session_id: request.request_id.clone(),
            layer_start: shard.layer_start as u32,
            layer_end: shard.layer_end as u32,
            total_layers: shard.total_layers.unwrap_or(shard.layer_end) as u32,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            tensor: frame.clone(),
        };

        // ── Attempt primary peer ──────────────────────────────────────────────
        let t0 = Instant::now();
        let primary_result = timeout(
            Duration::from_secs(self.rpc_timeout_secs),
            self.call_peer(&shard.peer_id, &req),
        )
        .await
        .unwrap_or_else(|_| Err(anyhow!("RPC timeout to peer {}", shard.peer_id)));

        match primary_result {
            Ok(resp) => {
                let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
                self.record_success(&shard.peer_id, latency_ms).await;
                return Ok(resp.tensor);
            }
            Err(e) => {
                eprintln!(
                    "[failover] primary peer {} failed for layers {}..{}: {e}",
                    shard.peer_id, shard.layer_start, shard.layer_end
                );
                self.record_failure(&shard.peer_id).await;

                if self.max_retries == 0 || self.registry.is_none() {
                    return Err(e);
                }
            }
        }

        // ── Failover: try up to max_retries alternative peers ─────────────────
        let alternatives = self.get_alternatives(shard, &shard.peer_id).await;
        let mut last_err = anyhow!("all failover attempts failed for layers {}..{}", shard.layer_start, shard.layer_end);

        for (attempt, alt_peer_id) in alternatives.into_iter().enumerate() {
            if attempt >= self.max_retries {
                break;
            }

            eprintln!(
                "[failover] attempt {}/{} → peer {} layers {}..{}",
                attempt + 1,
                self.max_retries,
                alt_peer_id,
                shard.layer_start,
                shard.layer_end,
            );

            // Rebuild req with same frame (cloned before primary attempt)
            let retry_req = RpcRequest {
                session_id: request.request_id.clone(),
                layer_start: shard.layer_start as u32,
                layer_end: shard.layer_end as u32,
                total_layers: shard.total_layers.unwrap_or(shard.layer_end) as u32,
                max_tokens: request.max_tokens,
                temperature: request.temperature,
                tensor: frame.clone(),
            };

            let t0 = Instant::now();
            let result = timeout(
                Duration::from_secs(self.rpc_timeout_secs),
                self.call_peer(&alt_peer_id, &retry_req),
            )
            .await
            .unwrap_or_else(|_| Err(anyhow!("RPC timeout to failover peer {}", alt_peer_id)));

            match result {
                Ok(resp) => {
                    let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    self.record_success(&alt_peer_id, latency_ms).await;
                    eprintln!("[failover] success via {alt_peer_id}");
                    return Ok(resp.tensor);
                }
                Err(e) => {
                    eprintln!("[failover] peer {alt_peer_id} also failed: {e}");
                    self.record_failure(&alt_peer_id).await;
                    last_err = e;
                }
            }
        }

        Err(last_err)
    }

    // ── Registry helpers ──────────────────────────────────────────────────────

    /// Record a successful RPC in the registry (if attached).
    async fn record_success(&self, peer_id: &str, latency_ms: f64) {
        if let Some(reg) = &self.registry {
            reg.write().await.record_success(peer_id, latency_ms);
        }
    }

    /// Record a failed RPC in the registry (if attached).
    async fn record_failure(&self, peer_id: &str) {
        if let Some(reg) = &self.registry {
            reg.write().await.record_failure(peer_id);
        }
    }

    /// Return alternative peer IDs for the same layer range, excluding `exclude_id`.
    async fn get_alternatives(&self, shard: &ShardSpec, exclude_id: &str) -> Vec<String> {
        let Some(reg) = &self.registry else {
            return Vec::new();
        };
        let guard = reg.read().await;
        guard
            .best_peers_for_layers(
                &self.model,
                shard.layer_start,
                shard.layer_end,
                self.max_retries + 1,
            )
            .into_iter()
            .filter(|p| p.id != exclude_id)
            .map(|p| p.id.clone())
            .collect()
    }
}

// ── ShardExecutor impl ────────────────────────────────────────────────────────

#[async_trait]
impl ShardExecutor for IrohDistributedExecutor {
    /// Legacy string-based path (used by `Coordinator::run()`).
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        input: &str,
        request: &InferenceRequest,
    ) -> Result<String> {
        let token_ids: Vec<f32> = input.bytes().map(|b| b as f32).collect();
        let frame = TensorFrame::from_f32(&token_ids, vec![1, token_ids.len().max(1)], 0);
        let resp = self.rpc(shard, frame, request).await?;
        let decoded: String = resp
            .to_f32_vec()?
            .iter()
            .map(|&f| f as u8)
            .take_while(|&b| b != 0)
            .collect::<Vec<u8>>()
            .into_iter()
            .map(|b| b as char)
            .collect();
        Ok(decoded)
    }

    /// Tensor-native path — used by `Coordinator::generate()` with failover.
    async fn run_shard_tensor(
        &self,
        shard: &ShardSpec,
        frame: TensorFrame,
        request: &InferenceRequest,
    ) -> Result<TensorFrame> {
        self.rpc(shard, frame, request).await
    }
}
