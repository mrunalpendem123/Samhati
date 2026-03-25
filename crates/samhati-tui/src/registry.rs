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

use anyhow::{bail, Result};
use ed25519_dalek::{Signer, SigningKey};
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

    let pubkey_bytes = bs58::decode(our_pubkey).into_vec()?;
    let pda = derive_node_pda_address(&pubkey_bytes);

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

    Ok(!resp["result"]["value"].is_null())
}

/// Register our node on Solana. Sends the `register_node` instruction.
/// Returns the transaction signature.
pub async fn register_node(
    secret_key: &[u8; 32],
    public_key: &[u8; 32],
    model_name: &str,
) -> Result<String> {
    let client = reqwest::Client::new();
    let signing_key = SigningKey::from_bytes(secret_key);
    let our_pubkey = bs58::encode(public_key).into_string();

    // 1. Derive PDA for our NodeAccount
    let (pda, bump) = find_pda(&[b"node", public_key], &program_id_bytes());

    // 2. Get recent blockhash
    let bh_body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "getLatestBlockhash",
        "params": [{"commitment": "finalized"}]
    });
    let bh_resp: serde_json::Value = client.post(RPC_URL).json(&bh_body).send().await?.json().await?;
    let blockhash_str = bh_resp["result"]["value"]["blockhash"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Failed to get blockhash"))?;
    let blockhash = bs58::decode(blockhash_str).into_vec()?;

    // 3. Build instruction data: Anchor discriminator + RegisterNodeArgs (Borsh)
    let ix_discriminator = anchor_ix_discriminator("register_node");
    let mut ix_data = Vec::new();
    ix_data.extend_from_slice(&ix_discriminator);
    // RegisterNodeArgs { model_name: String, model_size_b: u8, domain_tags: u64 }
    let name_bytes = model_name.as_bytes();
    ix_data.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes()); // string length
    ix_data.extend_from_slice(name_bytes);                                // string data
    ix_data.push(3);                                                       // model_size_b = 3
    ix_data.extend_from_slice(&0u64.to_le_bytes());                       // domain_tags = 0

    // 4. Build transaction (single instruction)
    let program_id = program_id_bytes();
    let pda_base58 = bs58::encode(&pda).into_string();

    // Account metas: [operator (signer, writable), node_account (writable), system_program]
    let system_program = [0u8; 32]; // 11111111111111111111111111111111

    let tx = build_transaction(
        public_key,
        &blockhash,
        &program_id,
        &[
            AccountMeta { pubkey: *public_key, is_signer: true, is_writable: true },
            AccountMeta { pubkey: pda, is_signer: false, is_writable: true },
            AccountMeta { pubkey: system_program, is_signer: false, is_writable: false },
        ],
        &ix_data,
        &signing_key,
    );

    // 5. Send transaction
    let tx_base64 = base64_encode(&tx);
    let send_body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "sendTransaction",
        "params": [tx_base64, {"encoding": "base64", "skipPreflight": true}]
    });

    let send_resp: serde_json::Value = client.post(RPC_URL).json(&send_body).send().await?.json().await?;

    if let Some(sig) = send_resp["result"].as_str() {
        Ok(sig.to_string())
    } else {
        let err = send_resp["error"]["message"].as_str().unwrap_or("unknown error");
        bail!("register_node failed: {}", err)
    }
}

// ── Transaction builder ──────────────────────────────────────────

struct AccountMeta {
    pubkey: [u8; 32],
    is_signer: bool,
    is_writable: bool,
}

fn build_transaction(
    payer: &[u8; 32],
    recent_blockhash: &[u8],
    program_id: &[u8; 32],
    accounts: &[AccountMeta],
    data: &[u8],
    signing_key: &SigningKey,
) -> Vec<u8> {
    // Solana transaction wire format (legacy, single signature)
    let mut msg = Vec::new();

    // Message header: num_required_signatures, num_readonly_signed, num_readonly_unsigned
    let num_signers = accounts.iter().filter(|a| a.is_signer).count() as u8;
    let num_readonly_unsigned = accounts.iter().filter(|a| !a.is_signer && !a.is_writable).count() as u8;
    msg.push(num_signers);  // num required signatures
    msg.push(0);            // num readonly signed accounts
    msg.push(num_readonly_unsigned); // num readonly unsigned accounts

    // Collect unique account keys (payer first, then others, then program_id)
    let mut keys: Vec<[u8; 32]> = Vec::new();
    keys.push(*payer);
    for a in accounts {
        if a.pubkey != *payer && !keys.contains(&a.pubkey) {
            keys.push(a.pubkey);
        }
    }
    if !keys.contains(program_id) {
        keys.push(*program_id);
    }

    // Compact array: num accounts
    msg.push(keys.len() as u8);
    for key in &keys {
        msg.extend_from_slice(key);
    }

    // Recent blockhash (32 bytes)
    if recent_blockhash.len() >= 32 {
        msg.extend_from_slice(&recent_blockhash[..32]);
    } else {
        msg.extend_from_slice(recent_blockhash);
        msg.resize(msg.len() + 32 - recent_blockhash.len(), 0);
    }

    // Instructions compact array (1 instruction)
    msg.push(1); // num instructions

    // Instruction: program_id_index
    let prog_idx = keys.iter().position(|k| k == program_id).unwrap_or(0) as u8;
    msg.push(prog_idx);

    // Account indices compact array
    msg.push(accounts.len() as u8);
    for a in accounts {
        let idx = keys.iter().position(|k| k == &a.pubkey).unwrap_or(0) as u8;
        msg.push(idx);
    }

    // Data compact array
    compact_u16(&mut msg, data.len() as u16);
    msg.extend_from_slice(data);

    // Sign the message
    let signature = signing_key.sign(&msg);

    // Full transaction: [num_signatures, signature(s), message]
    let mut tx = Vec::new();
    tx.push(1u8); // 1 signature
    tx.extend_from_slice(&signature.to_bytes());
    tx.extend_from_slice(&msg);
    tx
}

fn compact_u16(buf: &mut Vec<u8>, val: u16) {
    if val < 0x80 {
        buf.push(val as u8);
    } else if val < 0x4000 {
        buf.push(((val & 0x7F) | 0x80) as u8);
        buf.push((val >> 7) as u8);
    } else {
        buf.push(((val & 0x7F) | 0x80) as u8);
        buf.push((((val >> 7) & 0x7F) | 0x80) as u8);
        buf.push((val >> 14) as u8);
    }
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

/// Anchor instruction discriminator: sha256("global:<name>")[..8]
fn anchor_ix_discriminator(name: &str) -> [u8; 8] {
    let input = format!("global:{}", name);
    let hash = sha256(input.as_bytes());
    let mut disc = [0u8; 8];
    disc.copy_from_slice(&hash[..8]);
    disc
}

/// Program ID as bytes.
fn program_id_bytes() -> [u8; 32] {
    let bytes = bs58::decode(PROGRAM_ID).into_vec().unwrap();
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes[..32]);
    out
}

/// Derive PDA address: sha256(seeds + program_id + [bump])[..32]
/// Tries bump from 255 down to 0 until it finds one off the Ed25519 curve.
fn find_pda(seeds: &[&[u8]], program_id: &[u8; 32]) -> ([u8; 32], u8) {
    for bump in (0..=255u8).rev() {
        let mut hasher_input = Vec::new();
        for seed in seeds {
            hasher_input.extend_from_slice(seed);
        }
        hasher_input.push(bump);
        hasher_input.extend_from_slice(program_id);
        hasher_input.extend_from_slice(b"ProgramDerivedAddress");

        let hash = sha256(&hasher_input);

        // A valid PDA must NOT be on the Ed25519 curve.
        // Simple check: try to decompress as an Ed25519 point.
        // If it fails, it's a valid PDA.
        if curve25519_dalek_check_off_curve(&hash) {
            return (hash, bump);
        }
    }
    // Fallback (should never happen)
    ([0u8; 32], 0)
}

/// Check if a 32-byte value is NOT on the Ed25519 curve (valid PDA).
fn curve25519_dalek_check_off_curve(bytes: &[u8; 32]) -> bool {
    // Try to parse as a compressed Edwards Y point.
    // If decompression fails, it's off the curve → valid PDA.
    use ed25519_dalek::VerifyingKey;
    VerifyingKey::from_bytes(bytes).is_err()
}

/// Derive PDA address as base58 string.
fn derive_node_pda_address(operator_pubkey_bytes: &[u8]) -> String {
    let mut pubkey = [0u8; 32];
    let len = operator_pubkey_bytes.len().min(32);
    pubkey[..len].copy_from_slice(&operator_pubkey_bytes[..len]);
    let (pda, _) = find_pda(&[b"node", &pubkey], &program_id_bytes());
    bs58::encode(&pda).into_string()
}
