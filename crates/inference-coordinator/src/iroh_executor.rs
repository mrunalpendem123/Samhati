use anyhow::{anyhow, Result};
use async_trait::async_trait;
use iroh::{Endpoint, NodeId};
use std::str::FromStr;
use tokio::time::{timeout, Duration};

use crate::plan::ShardSpec;
use crate::{InferenceRequest, ShardExecutor};

use mesh_node::protocol::{InferenceRequest as RpcRequest, InferenceResponse as RpcResponse, INFERENCE_ALPN};

/// Implements ShardExecutor by opening a direct iroh QUIC stream to the target peer,
/// forwarding the prompt and tensor activations, and awaiting the processed outputs.
#[derive(Debug, Clone)]
pub struct IrohDistributedExecutor {
    endpoint: Endpoint,
}

impl IrohDistributedExecutor {
    pub fn new(endpoint: Endpoint) -> Self {
        Self { endpoint }
    }
}

#[async_trait]
impl ShardExecutor for IrohDistributedExecutor {
    async fn run_shard(
        &self,
        shard: &ShardSpec,
        input: &str,
        request: &InferenceRequest,
    ) -> Result<String> {
        let target_node_id = NodeId::from_str(&shard.peer_id)
            .map_err(|e| anyhow!("Invalid NodeId for peer {}: {}", shard.peer_id, e))?;

        // 1. Establish secure ALPN stream with the peer assigned to this shard
        let conn = timeout(
            Duration::from_secs(10), // Establish connection timeout
            self.endpoint.connect(target_node_id, INFERENCE_ALPN),
        )
        .await
        .map_err(|_| anyhow!("Timeout connecting to peer {}", target_node_id))??;

        let (mut send, mut recv) = conn.open_bi().await?;

        // 2. Build the RPC Payload
        let rpc_req = RpcRequest {
            session_id: request.request_id.clone(),
            layer_start: shard.layer_start as u32,
            layer_end: shard.layer_end as u32,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            // TODO: In Phase 2, this will be real candle tensor bytes.
            // For now, testing continuity by sending the string wrapped in bytes
            tensor_bytes: input.as_bytes().to_vec(),
        };

        let req_bytes = rpc_req.to_bytes()?;
        
        // Write size prefix (4 bytes) followed by payload
        let size = (req_bytes.len() as u32).to_be_bytes();
        send.write_all(&size).await?;
        send.write_all(&req_bytes).await?;
        send.finish()?; // tell peer we are done writing

        // 3. Await the response
        let mut size_buf = [0u8; 4];
        recv.read_exact(&mut size_buf).await?;
        let resp_size = u32::from_be_bytes(size_buf) as usize;

        let mut resp_bytes = vec![0u8; resp_size];
        recv.read_exact(&mut resp_bytes).await?;

        // 4. Decode results
        let rpc_res = RpcResponse::from_bytes(&resp_bytes)?;
        
        if let Some(err) = rpc_res.error {
            return Err(anyhow!("Peer {} reported error: {}", target_node_id, err));
        }

        // Return the parsed resulting data (later to be deserialized back to Tensors)
        let resulting_text = String::from_utf8(rpc_res.tensor_bytes)
            .unwrap_or_else(|_| "[Binary Tensor Data]".to_string());
        
        Ok(resulting_text)
    }
}
