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
        kv_cache::{KvCacheStore, SessionKv},
        llm_shard::{LlamaShard, LlamaShardConfig},
        rpc::{RpcRequest, RpcResponse},
        tensor_frame::TensorFrame,
        InferenceBackend,
    };
    use std::collections::HashMap;

    type InferBackend = InferenceBackend;

    /// A persistent iroh ALPN handler that:
    /// 1. Holds a loaded `LlamaShard` (transformer layers + optional embed / lm_head).
    /// 2. Maintains a `KvCacheStore` shared across all concurrent sessions.
    /// 3. For each incoming QUIC bi-stream: deserialises an `RpcRequest`,
    ///    runs the shard forward pass, and streams back an `RpcResponse`.
    // `LlamaShard<B>` is `Send` but `!Sync` because `Param<T>` uses
    // `once_cell::unsync::OnceCell`.  Wrapping in `Mutex` gives `Sync` (Mutex<T>:
    // Sync if T: Send), which satisfies the `ProtocolHandler: Sync` bound.
    #[derive(Clone)]
    pub struct InferenceServer {
        shard: Arc<Mutex<LlamaShard<InferBackend>>>,
        kv: KvCacheStore<InferBackend>,
    }

    impl std::fmt::Debug for InferenceServer {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("InferenceServer").finish()
        }
    }

    impl InferenceServer {
        pub fn new(shard: LlamaShard<InferBackend>, kv_ttl_secs: u64) -> Self {
            Self {
                shard: Arc::new(Mutex::new(shard)),
                kv: KvCacheStore::new(kv_ttl_secs),
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
            let shard = LlamaShard::load_from_bytes_wgpu(&bytes, cfg)?;
            Ok(Self::new(shard, kv_ttl_secs))
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

            let n_layers = (req.layer_end - req.layer_start) as usize;
            let session_id = req.session_id.clone();

            // 2. Ensure the session's KV cache exists and is fresh
            self.kv.touch(&session_id, n_layers).await;

            // 3. Run forward pass (sync lock on shard — no await while held)
            let result = {
                let mut map: tokio::sync::RwLockWriteGuard<
                    HashMap<String, SessionKv<InferBackend>>,
                > = self.kv.inner.write().await;
                let session = map
                    .get_mut(&session_id)
                    .ok_or_else(|| anyhow::anyhow!("session vanished"))?;
                let shard = self
                    .shard
                    .lock()
                    .map_err(|_| anyhow::anyhow!("shard lock poisoned"))?;
                shard.forward(&req.tensor, session)
            };

            let resp = match result {
                Ok(out_frame) => RpcResponse { tensor: out_frame, error: None },
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
