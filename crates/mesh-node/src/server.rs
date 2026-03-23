use iroh::{
    endpoint::{Connection, RecvStream, SendStream},
    protocol::{AcceptError, ProtocolHandler},
};
#[cfg(feature = "burn")]
use std::sync::{Arc, Mutex};

// ── Real (burn) inference server ───────────────────────────────────────────────

#[cfg(feature = "burn")]
pub use real::InferenceServer;

// Always available for use in gossip / non-inference modes
pub use mock::MockInferenceServer;

#[cfg(feature = "burn")]
mod real {
    use super::*;
    use anyhow::Result;
    use inference_coordinator::{
        activation_cache::{CachedOutput, OutputRingBuffer},
        kv_cache::{KvCacheStore, SessionKv},
        llm_shard::{LlamaShard, LlamaShardConfig},
        qwen35_cache::{Qwen35CacheStore, Qwen35Session},
        qwen35_shard::{Qwen35Shard, Qwen35ShardConfig},
        rpc::{RpcRequest, RpcResponse},
        tensor_frame::TensorFrame,
        InferenceBackend,
    };
    use std::collections::HashMap;
    use tokio::sync::RwLock;

    type InferBackend = InferenceBackend;

    /// Which model architecture is loaded.
    enum ShardKind {
        Llama {
            shard: Arc<Mutex<LlamaShard<InferBackend>>>,
            kv: KvCacheStore<InferBackend>,
        },
        Qwen35 {
            shard: Arc<Mutex<Qwen35Shard<InferBackend>>>,
            cache: Qwen35CacheStore<InferBackend>,
        },
    }

    impl Clone for ShardKind {
        fn clone(&self) -> Self {
            match self {
                Self::Llama { shard, kv } => Self::Llama {
                    shard: shard.clone(),
                    kv: kv.clone(),
                },
                Self::Qwen35 { shard, cache } => Self::Qwen35 {
                    shard: shard.clone(),
                    cache: cache.clone(),
                },
            }
        }
    }

    /// A persistent iroh ALPN handler that supports both Llama-style and
    /// Qwen3.5-style model architectures.
    ///
    /// Includes a server-side output ring buffer (Petals dual-cache) that
    /// caches the last 64 output activations for fault recovery replay.
    #[derive(Clone)]
    pub struct InferenceServer {
        kind: ShardKind,
        /// Server-side output cache for fault recovery.
        output_cache: Arc<RwLock<OutputRingBuffer>>,
    }

    impl std::fmt::Debug for InferenceServer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("InferenceServer").finish()
        }
    }

    impl InferenceServer {
        /// Construct a server for a Llama-style shard.
        ///
        /// The KV cache is scoped to the shard's layer range so each node
        /// only stores KV for the layers it owns (layer-local ownership).
        pub fn new(shard: LlamaShard<InferBackend>, cfg: &LlamaShardConfig, kv_ttl_secs: u64) -> Self {
            let layer_range = cfg.layer_start..cfg.layer_end;
            Self {
                kind: ShardKind::Llama {
                    shard: Arc::new(Mutex::new(shard)),
                    kv: KvCacheStore::new(kv_ttl_secs, layer_range),
                },
                output_cache: Arc::new(RwLock::new(OutputRingBuffer::new(64))),
            }
        }

        /// Construct a server for a Qwen3.5-style shard.
        pub fn new_qwen35(shard: Qwen35Shard<InferBackend>, cfg: &Qwen35ShardConfig, kv_ttl_secs: u64) -> Self {
            let layer_types = cfg.local_layer_types();
            Self {
                kind: ShardKind::Qwen35 {
                    shard: Arc::new(Mutex::new(shard)),
                    cache: Qwen35CacheStore::new(kv_ttl_secs, layer_types),
                },
                output_cache: Arc::new(RwLock::new(OutputRingBuffer::new(64))),
            }
        }

        /// Load a shard from a content-addressed `ShardStore` and wrap it in
        /// an `InferenceServer` ready to serve QUIC requests.
        pub fn new_from_store(
            store: &shard_store::ShardStore,
            hash: &shard_store::Hash,
            cfg: LlamaShardConfig,
            kv_ttl_secs: u64,
        ) -> anyhow::Result<Self> {
            let bytes = store
                .get(hash)?
                .ok_or_else(|| anyhow::anyhow!("shard hash {} not in store", hash))?;
            let shard = LlamaShard::load_from_bytes_wgpu(&bytes, cfg.clone())?;
            Ok(Self::new(shard, &cfg, kv_ttl_secs))
        }

        async fn handle_stream(
            &self,
            mut send: SendStream,
            mut recv: RecvStream,
        ) -> Result<()> {
            // 1. Read size-prefixed request
            let mut size_buf = [0u8; 4];
            recv.read_exact(&mut size_buf).await?;
            let req_size = u32::from_be_bytes(size_buf) as usize;
            let mut req_bytes = vec![0u8; req_size];
            recv.read_exact(&mut req_bytes).await?;

            let req = match RpcRequest::from_bytes(&req_bytes) {
                Ok(r) => r,
                Err(e) => {
                    return send_error(&mut send, format!("bad request: {e}")).await;
                }
            };

            let session_id = req.session_id.clone();

            // 2-3. Dispatch based on shard kind
            let result = match &self.kind {
                ShardKind::Llama { shard, kv } => {
                    // Validate that the request's layer range falls within our scope.
                    if let Err(e) = kv.validate_range(req.layer_start as usize, req.layer_end as usize) {
                        return send_error(&mut send, format!("layer range error: {e}")).await;
                    }
                    kv.touch(&session_id).await;
                    let mut map: tokio::sync::RwLockWriteGuard<
                        HashMap<String, SessionKv<InferBackend>>,
                    > = kv.inner.write().await;
                    let session = map
                        .get_mut(&session_id)
                        .ok_or_else(|| anyhow::anyhow!("session vanished"))?;
                    let shard = shard
                        .lock()
                        .map_err(|_| anyhow::anyhow!("shard lock poisoned"))?;
                    shard.forward(&req.tensor, session)
                }
                ShardKind::Qwen35 { shard, cache } => {
                    cache.touch(&session_id).await;
                    let mut map: tokio::sync::RwLockWriteGuard<
                        HashMap<String, Qwen35Session<InferBackend>>,
                    > = cache.inner.write().await;
                    let session = map
                        .get_mut(&session_id)
                        .ok_or_else(|| anyhow::anyhow!("session vanished"))?;
                    let shard = shard
                        .lock()
                        .map_err(|_| anyhow::anyhow!("shard lock poisoned"))?;
                    shard.forward(&req.tensor, session)
                }
            };

            let resp = match result {
                Ok(out_frame) => {
                    // Cache the output activation for fault recovery (Petals dual-cache)
                    {
                        let mut cache = self.output_cache.write().await;
                        cache.push(CachedOutput {
                            session_id: session_id.clone(),
                            seq_offset: req.tensor.seq_offset,
                            frame: out_frame.clone(),
                        });
                    }
                    RpcResponse { tensor: out_frame, error: None }
                }
                Err(e) => {
                    let dummy = TensorFrame::from_f32(&[0.0], vec![1, 1], 0);
                    RpcResponse { tensor: dummy, error: Some(format!("{e}")) }
                }
            };

            // 4. Write size-prefixed response
            let resp_bytes = resp.to_bytes()?;
            let resp_size = (resp_bytes.len() as u32).to_be_bytes();
            send.write_all(&resp_size).await?;
            send.write_all(&resp_bytes).await?;
            send.finish()?;
            Ok(())
        }
    }

    impl ProtocolHandler for InferenceServer {
        async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
            let (send, recv) = conn.accept_bi().await.map_err(AcceptError::from_err)?;
            self.handle_stream(send, recv)
                .await
                .map_err(|e| AcceptError::from_boxed(e.into()))
        }
    }

    async fn send_error(send: &mut SendStream, msg: String) -> Result<()> {
        let dummy = TensorFrame::from_f32(&[0.0], vec![1, 1], 0);
        let resp = RpcResponse { tensor: dummy, error: Some(msg) };
        let bytes = resp.to_bytes()?;
        let size = (bytes.len() as u32).to_be_bytes();
        send.write_all(&size).await?;
        send.write_all(&bytes).await?;
        send.finish()?;
        Ok(())
    }
}

// ── Mock server (always available) ────────────────────────────────────────────
//
// Used in gossip mode and any context where no real model is loaded.
// When the `burn` feature is disabled, this is also exported as `InferenceServer`.

#[cfg(not(feature = "burn"))]
pub use mock::MockInferenceServer as InferenceServer;

mod mock {
    use super::*;
    use inference_coordinator::rpc::{LegacyRpcRequest, LegacyRpcResponse};

    /// Mock server used when no model shard is loaded.
    /// Echoes the request text back with a shard-execution annotation.
    #[derive(Debug, Clone)]
    pub struct MockInferenceServer {}

    impl MockInferenceServer {
        pub fn new() -> Self {
            Self {}
        }
    }

    impl ProtocolHandler for MockInferenceServer {
        async fn accept(&self, conn: Connection) -> Result<(), AcceptError> {
            let (mut send, mut recv): (SendStream, RecvStream) =
                conn.accept_bi().await.map_err(AcceptError::from_err)?;

            // Read size-prefixed request
            let mut size_buf = [0u8; 4];
            recv.read_exact(&mut size_buf)
                .await
                .map_err(AcceptError::from_err)?;
            let req_size = u32::from_be_bytes(size_buf) as usize;
            let mut req_bytes = vec![0u8; req_size];
            recv.read_exact(&mut req_bytes)
                .await
                .map_err(AcceptError::from_err)?;

            let rpc_req = match LegacyRpcRequest::from_bytes(&req_bytes) {
                Ok(r) => r,
                Err(e) => {
                    let err_resp = LegacyRpcResponse {
                        tensor_bytes: vec![],
                        error: Some(format!("Failed to parse request: {e}")),
                    };
                    let e_bytes = err_resp
                        .to_bytes()
                        .map_err(|e| AcceptError::from_boxed(e.into()))?;
                    let e_size = (e_bytes.len() as u32).to_be_bytes();
                    send.write_all(&e_size)
                        .await
                        .map_err(AcceptError::from_err)?;
                    send.write_all(&e_bytes)
                        .await
                        .map_err(AcceptError::from_err)?;
                    send.finish()?;
                    return Ok(());
                }
            };

            println!(
                "mock-server: session={} layers={}-{} input_bytes={}",
                rpc_req.session_id,
                rpc_req.layer_start,
                rpc_req.layer_end,
                rpc_req.tensor_bytes.len()
            );

            let input_text = String::from_utf8_lossy(&rpc_req.tensor_bytes);
            let response_text = format!(
                "{} [mock-peer layers:{}-{} max_tokens:{} temp:{:.2}]",
                input_text,
                rpc_req.layer_start,
                rpc_req.layer_end,
                rpc_req.max_tokens,
                rpc_req.temperature
            );

            let rpc_res = LegacyRpcResponse {
                tensor_bytes: response_text.into_bytes(),
                error: None,
            };
            let res_bytes = rpc_res
                .to_bytes()
                .map_err(|e| AcceptError::from_boxed(e.into()))?;
            let res_size = (res_bytes.len() as u32).to_be_bytes();
            send.write_all(&res_size)
                .await
                .map_err(AcceptError::from_err)?;
            send.write_all(&res_bytes)
                .await
                .map_err(AcceptError::from_err)?;
            send.finish()?;
            Ok(())
        }
    }
}
