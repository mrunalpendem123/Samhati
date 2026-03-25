//! Solana on-chain node registry — fetch all registered Samhati nodes.
//!
//! Reads NodeAccount PDAs from the Samhati program on Solana devnet.
//! Each NodeAccount's `operator` pubkey is the node's iroh NodeId
//! (unified identity — same Ed25519 key).
//!
//! Flow:
//!   1. getProgramAccounts with NodeAccount discriminator filter
//!   2. Decode each account → extract operator pubkey
//!   3. Convert to iroh NodeId (hex-encoded pubkey)
//!   4. Return list of all online nodes to bootstrap gossip

use anyhow::Result;
use serde::Deserialize;

/// Samhati program ID on devnet.
const PROGRAM_ID: &str = "AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr";

/// Solana RPC endpoint.
const RPC_URL: &str = "https://api.devnet.solana.com";

/// A registered node from the on-chain registry.
#[derive(Debug, Clone)]
pub struct RegisteredNode {
    /// Solana pubkey (base58) — also the iroh NodeId source
    pub pubkey: String,
    /// iroh NodeId (hex-encoded pubkey bytes)
    pub iroh_node_id: String,
    /// Model name from on-chain data
    pub model_name: String,
    /// ELO score
    pub elo: i32,
    /// Whether node passed TOPLOC calibration
    pub calibrated: bool,
}

/// Fetch all registered Samhati nodes from Solana.
///
/// Uses `getProgramAccounts` with a memcmp filter on the Anchor
/// account discriminator for `NodeAccount`.
pub async fn fetch_all_nodes() -> Result<Vec<RegisteredNode>> {
    let client = reqwest::Client::new();

    // Anchor discriminator for NodeAccount = sha256("account:NodeAccount")[..8]
    // We compute it once: anchor uses sha256("account:NodeAccount") first 8 bytes
    let discriminator = anchor_discriminator("NodeAccount");
    let disc_base64 = base64_encode(&discriminator);

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getProgramAccounts",
        "params": [
            PROGRAM_ID,
            {
                "encoding": "base64",
                "filters": [
                    {
                        "memcmp": {
                            "offset": 0,
                            "bytes": disc_base64,
                            "encoding": "base64"
                        }
                    }
                ]
            }
        ]
    });

    let resp: RpcResponse = client
        .post(RPC_URL)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    let accounts = resp.result.unwrap_or_default();
    let mut nodes = Vec::new();

    for account in accounts {
        if let Some(node) = decode_node_account(&account) {
            nodes.push(node);
        }
    }

    Ok(nodes)
}

/// Check if our node is already registered on-chain.
pub async fn is_registered(our_pubkey: &str) -> Result<bool> {
    let client = reqwest::Client::new();

    // Derive PDA address: seeds = [b"node", operator_pubkey]
    let pda = derive_node_pda(our_pubkey);

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getAccountInfo",
        "params": [pda, {"encoding": "base64"}]
    });

    let resp: serde_json::Value = client
        .post(RPC_URL)
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    // If account exists and has data, node is registered
    Ok(!resp["result"]["value"].is_null())
}

// ── Internal helpers ──────────────────────────────────────────────

#[derive(Deserialize)]
struct RpcResponse {
    result: Option<Vec<AccountEntry>>,
}

#[derive(Deserialize)]
struct AccountEntry {
    pubkey: String,
    account: AccountData,
}

#[derive(Deserialize)]
struct AccountData {
    data: (String, String), // (base64_data, encoding)
}

/// Decode a NodeAccount from raw base64 account data.
fn decode_node_account(entry: &AccountEntry) -> Option<RegisteredNode> {
    let data = base64_decode(&entry.account.data.0)?;

    // Layout after 8-byte discriminator:
    // operator: Pubkey (32 bytes) at offset 8
    // elo_score: i32 (4 bytes) at offset 40
    // calibrated: bool (1 byte) at offset 44
    // model_name: String (4 byte len + data) at offset 45

    if data.len() < 49 {
        return None;
    }

    let operator_bytes = &data[8..40];
    let elo_bytes = &data[40..44];
    let calibrated = data[44] != 0;

    let elo = i32::from_le_bytes([elo_bytes[0], elo_bytes[1], elo_bytes[2], elo_bytes[3]]);

    // Read model_name (Borsh string: 4-byte LE length + UTF-8 data)
    let model_name = if data.len() > 49 {
        let name_len = u32::from_le_bytes([data[45], data[46], data[47], data[48]]) as usize;
        let name_end = 49 + name_len;
        if data.len() >= name_end {
            String::from_utf8_lossy(&data[49..name_end]).to_string()
        } else {
            "unknown".into()
        }
    } else {
        "unknown".into()
    };

    let pubkey = bs58::encode(operator_bytes).into_string();
    let iroh_node_id = hex::encode(operator_bytes);

    Some(RegisteredNode {
        pubkey,
        iroh_node_id,
        model_name,
        elo,
        calibrated,
    })
}

/// Compute Anchor account discriminator: sha256("account:<Name>")[..8]
fn anchor_discriminator(name: &str) -> [u8; 8] {
    use std::io::Write;
    let input = format!("account:{}", name);
    let hash = sha256(input.as_bytes());
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash[..8]);
    disc
}

fn sha256(data: &[u8]) -> [u8; 32] {
    use sha2::{Sha256, Digest};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let result = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&result);
    out
}

/// Base64 encode bytes.
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let n = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((n >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((n >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((n >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(n & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Base64 decode string to bytes.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u32> {
        match c {
            b'A'..=b'Z' => Some((c - b'A') as u32),
            b'a'..=b'z' => Some((c - b'a' + 26) as u32),
            b'0'..=b'9' => Some((c - b'0' + 52) as u32),
            b'+' => Some(62),
            b'/' => Some(63),
            b'=' => Some(0),
            _ => None,
        }
    }
    let bytes = s.as_bytes();
    let mut result = Vec::new();
    for chunk in bytes.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let n = (a << 18) | (b << 12) | (c << 6) | d;
        result.push(((n >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' {
            result.push(((n >> 8) & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            result.push((n & 0xFF) as u8);
        }
    }
    Some(result)
}

/// Derive NodeAccount PDA address from operator pubkey.
/// seeds = [b"node", operator_pubkey_bytes]
fn derive_node_pda(operator_base58: &str) -> String {
    // For now, return the base58 pubkey itself as a placeholder.
    // Full PDA derivation requires the program ID and sha256.
    // The getProgramAccounts approach doesn't need this — we fetch all accounts.
    operator_base58.to_string()
}
