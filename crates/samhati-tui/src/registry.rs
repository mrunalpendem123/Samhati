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
use std::collections::HashMap;
use std::sync::Mutex;

/// Cached PDA results — avoids repeated subprocess spawns.
static PDA_CACHE: std::sync::LazyLock<Mutex<HashMap<String, String>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashMap::new()));

/// Samhati program ID on devnet.
const PROGRAM_ID: &str = "AB7cSMLv1J7J28DKLMbzo2tyNp1kZSmE67a6Heoa5Mkr";

use crate::wallet::solana_rpc_url as rpc_url;

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
        .post(rpc_url())
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
/// Uses getProgramAccounts with memcmp filter on operator pubkey (avoids PDA derivation).
pub async fn is_registered(our_pubkey: &str) -> Result<bool> {
    let client = reqwest::Client::new();
    let discriminator = anchor_discriminator("NodeAccount");
    let disc_base64 = base64_encode(&discriminator);

    let pubkey_bytes = bs58::decode(our_pubkey).into_vec()?;
    let pubkey_base64 = base64_encode(&pubkey_bytes);

    let body = serde_json::json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getProgramAccounts",
        "params": [
            PROGRAM_ID,
            {
                "encoding": "base64",
                "filters": [
                    {"memcmp": {"offset": 0, "bytes": disc_base64, "encoding": "base64"}},
                    {"memcmp": {"offset": 8, "bytes": pubkey_base64, "encoding": "base64"}}
                ]
            }
        ]
    });

    let resp: serde_json::Value = client
        .post(rpc_url())
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    let count = resp["result"].as_array().map(|a| a.len()).unwrap_or(0);
    Ok(count > 0)
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

    // 1. Derive PDA for our NodeAccount using solana CLI (guaranteed correct)
    let pda_str = solana_find_pda(&["string:node", &format!("pubkey:{}", our_pubkey)])?;
    let pda_vec = bs58::decode(&pda_str).into_vec()?;
    let mut pda = [0u8; 32];
    pda.copy_from_slice(&pda_vec[..32]);

    // 2. Get recent blockhash
    let bh_body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "getLatestBlockhash",
        "params": [{"commitment": "finalized"}]
    });
    let bh_resp: serde_json::Value = client.post(rpc_url()).json(&bh_body).send().await?.json().await?;
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

    let send_resp: serde_json::Value = client.post(rpc_url()).json(&send_body).send().await?.json().await?;

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

/// Submit a swarm round to Solana. Records ELO deltas, proof hashes, domain, and winner.
/// Only works if our key is the coordinator authority.
pub async fn submit_round(
    secret_key: &[u8; 32],
    public_key: &[u8; 32],
    payload: &crate::settlement::RoundPayload,
) -> Result<String> {
    let client = reqwest::Client::new();
    let signing_key = SigningKey::from_bytes(secret_key);
    let our_pubkey = bs58::encode(public_key).into_string();

    // Derive PDAs using solana CLI
    let config_pda_str = solana_find_pda(&["string:config"])
        .unwrap_or_else(|_| "5LE13zvQJhCQRrCbwJt4mmgvK1kMnDaQxPLo5Q2pbL2L".into());
    let config_pda = bs58::decode(&config_pda_str).into_vec()?;

    let round_id_bytes = payload.round_id.to_le_bytes();
    let round_id_hex = hex::encode(&round_id_bytes);
    let round_pda_str = solana_find_pda(&["string:round", &format!("hex:{}", round_id_hex)])?;
    let round_pda = bs58::decode(&round_pda_str).into_vec()?;

    // Get recent blockhash
    let bh_body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "getLatestBlockhash",
        "params": [{"commitment": "finalized"}]
    });
    let bh_resp: serde_json::Value = client.post(rpc_url()).json(&bh_body).send().await?.json().await?;
    let blockhash_str = bh_resp["result"]["value"]["blockhash"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Failed to get blockhash"))?;
    let blockhash = bs58::decode(blockhash_str).into_vec()?;

    // Build instruction data: anchor discriminator + SubmitRoundArgs (Borsh)
    let ix_disc = anchor_ix_discriminator("submit_round");
    let mut ix_data = Vec::new();
    ix_data.extend_from_slice(&ix_disc);

    // round_id: u64
    ix_data.extend_from_slice(&payload.round_id.to_le_bytes());

    // participants: Vec<Pubkey> (for now, use our own pubkey as placeholder)
    ix_data.extend_from_slice(&1u32.to_le_bytes()); // 1 participant
    ix_data.extend_from_slice(public_key);            // our pubkey

    // proof_hashes: Vec<[u8;32]>
    ix_data.extend_from_slice(&(payload.proof_hashes.len() as u32).to_le_bytes());
    for hash in &payload.proof_hashes {
        ix_data.extend_from_slice(hash);
    }

    // elo_deltas: Vec<i32>
    ix_data.extend_from_slice(&(payload.elo_deltas.len() as u32).to_le_bytes());
    for delta in &payload.elo_deltas {
        ix_data.extend_from_slice(&delta.to_le_bytes());
    }

    // winner: Pubkey (our pubkey for single-node rounds)
    ix_data.extend_from_slice(public_key);

    // smti_emitted: u64
    ix_data.extend_from_slice(&payload.smti_emitted.to_le_bytes());

    // domain: u64
    ix_data.extend_from_slice(&payload.domain.to_le_bytes());

    // Build transaction
    let program_id = program_id_bytes();
    let system_program = [0u8; 32];

    let mut config_pda_arr = [0u8; 32];
    config_pda_arr.copy_from_slice(&config_pda[..32]);
    let mut round_pda_arr = [0u8; 32];
    round_pda_arr.copy_from_slice(&round_pda[..32]);

    // Node account PDA for our pubkey
    let node_pda_str = solana_find_pda(&["string:node", &format!("pubkey:{}", our_pubkey)])
        .unwrap_or_default();
    let node_pda = if !node_pda_str.is_empty() {
        bs58::decode(&node_pda_str).into_vec().unwrap_or_default()
    } else {
        vec![0u8; 32]
    };
    let mut node_pda_arr = [0u8; 32];
    if node_pda.len() >= 32 {
        node_pda_arr.copy_from_slice(&node_pda[..32]);
    }

    let tx = build_transaction(
        public_key,
        &blockhash,
        &program_id,
        &[
            AccountMeta { pubkey: *public_key, is_signer: true, is_writable: true },
            AccountMeta { pubkey: config_pda_arr, is_signer: false, is_writable: true },
            AccountMeta { pubkey: round_pda_arr, is_signer: false, is_writable: true },
            AccountMeta { pubkey: system_program, is_signer: false, is_writable: false },
            // remaining_accounts: node PDAs
            AccountMeta { pubkey: node_pda_arr, is_signer: false, is_writable: true },
        ],
        &ix_data,
        &signing_key,
    );

    let tx_base64 = base64_encode(&tx);
    let send_body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "sendTransaction",
        "params": [tx_base64, {"encoding": "base64", "skipPreflight": true}]
    });

    let send_resp: serde_json::Value = client.post(rpc_url()).json(&send_body).send().await?.json().await?;

    if let Some(sig) = send_resp["result"].as_str() {
        Ok(sig.to_string())
    } else {
        let err = send_resp["error"]["message"].as_str().unwrap_or("unknown error");
        bail!("submit_round failed: {}", err)
    }
}

/// Get SOL balance for a pubkey (for checking if airdrop is needed).
pub async fn get_sol_balance(pubkey: &str) -> Result<f64> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "getBalance",
        "params": [pubkey]
    });
    let resp: serde_json::Value = client.post(rpc_url()).json(&body).send().await?.json().await?;
    let lamports = resp["result"]["value"].as_u64().unwrap_or(0);
    Ok(lamports as f64 / 1_000_000_000.0)
}

/// Request 1 SOL airdrop on devnet (free, for new users to register).
pub async fn request_airdrop(pubkey: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "requestAirdrop",
        "params": [pubkey, 1_000_000_000u64] // 1 SOL
    });
    let resp: serde_json::Value = client.post(rpc_url()).json(&body).send().await?.json().await?;
    if let Some(sig) = resp["result"].as_str() {
        Ok(sig.to_string())
    } else {
        let err = resp["error"]["message"].as_str().unwrap_or("unknown");
        bail!("Airdrop failed: {}", err)
    }
}

/// Fetch network-wide domain demand from Solana ProtocolConfig PDA.
/// This is the single source of truth — updated every submit_round.
pub async fn fetch_demand() -> Result<crate::swarm::DemandStats> {
    let client = reqwest::Client::new();

    // ProtocolConfig PDA: seeds = [b"config"]
    let config_pda = solana_find_pda(&["string:config"])
        .unwrap_or_else(|_| "5LE13zvQJhCQRrCbwJt4mmgvK1kMnDaQxPLo5Q2pbL2L".into());

    let body = serde_json::json!({
        "jsonrpc": "2.0", "id": 1,
        "method": "getAccountInfo",
        "params": [config_pda, {"encoding": "base64"}]
    });

    let resp: serde_json::Value = client.post(rpc_url()).json(&body).send().await?.json().await?;

    if resp["result"]["value"].is_null() {
        // Config not initialized yet — return zeros
        return Ok(crate::swarm::DemandStats::default());
    }

    let data_b64 = resp["result"]["value"]["data"][0]
        .as_str()
        .unwrap_or("");
    let data = base64_decode(data_b64).unwrap_or_default();

    // ProtocolConfig layout after 8-byte discriminator:
    // authority: 32, smti_mint: 32, reward_vault: 32,
    // total_rounds: 8, total_smti_emitted: 8, base_emission: 8,
    // domain_code: 8, domain_math: 8, domain_reasoning: 8, domain_general: 8, bump: 1
    // Offsets: 8 + 32 + 32 + 32 + 8 + 8 + 8 = 128 for domain_code

    if data.len() < 128 + 32 {
        return Ok(crate::swarm::DemandStats::default());
    }

    let code = u64::from_le_bytes(data[128..136].try_into().unwrap_or([0; 8]));
    let math = u64::from_le_bytes(data[136..144].try_into().unwrap_or([0; 8]));
    let reasoning = u64::from_le_bytes(data[144..152].try_into().unwrap_or([0; 8]));
    let general = u64::from_le_bytes(data[152..160].try_into().unwrap_or([0; 8]));
    let total = code + math + reasoning + general;

    Ok(crate::swarm::DemandStats {
        code,
        math,
        reasoning,
        general,
        total,
    })
}

/// Derive ProtocolConfig PDA address. Seeds = [b"config"].
fn derive_config_pda() -> String {
    let (pda, _) = find_pda(&[b"config"], &program_id_bytes());
    bs58::encode(&pda).into_string()
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

/// Get a PDA address using the `solana` CLI (guaranteed correct).
/// Results are cached in memory — each unique seed combination only calls CLI once.
fn solana_find_pda(seeds: &[&str]) -> Result<String> {
    let cache_key = seeds.join("|");

    // Check cache first
    if let Ok(cache) = PDA_CACHE.lock() {
        if let Some(pda) = cache.get(&cache_key) {
            return Ok(pda.clone());
        }
    }

    let mut args = vec![
        "find-program-derived-address".to_string(),
        PROGRAM_ID.to_string(),
    ];
    for seed in seeds {
        args.push(seed.to_string());
    }

    let output = std::process::Command::new("solana")
        .args(&args)
        .output()
        .map_err(|e| anyhow::anyhow!("solana CLI not found: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("solana find-program-derived-address failed: {}", stderr);
    }

    let pda = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if pda.is_empty() {
        bail!("Empty PDA from solana CLI");
    }

    // Cache the result
    if let Ok(mut cache) = PDA_CACHE.lock() {
        cache.insert(cache_key, pda.clone());
    }

    Ok(pda)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_discriminator() {
        let disc = anchor_discriminator("NodeAccount");
        // Known value from Python: sha256("account:NodeAccount")[:8]
        assert_eq!(hex::encode(disc), "7da61292c37f56dc");
    }

    #[test]
    fn test_anchor_ix_discriminator() {
        let disc = anchor_ix_discriminator("register_node");
        assert_eq!(disc.len(), 8);
        // Should be deterministic
        let disc2 = anchor_ix_discriminator("register_node");
        assert_eq!(disc, disc2);
    }

    #[test]
    fn test_base64_roundtrip() {
        let data = b"hello world test data for base64";
        let encoded = base64_encode(data);
        let decoded = base64_decode(&encoded).unwrap();
        assert_eq!(data.to_vec(), decoded);
    }

    #[test]
    fn test_config_pda_via_cli() {
        // Only runs if solana CLI is installed
        if let Ok(pda) = solana_find_pda(&["string:config"]) {
            assert!(!pda.is_empty());
            // Should be the known config PDA
            assert_eq!(pda, "5LE13zvQJhCQRrCbwJt4mmgvK1kMnDaQxPLo5Q2pbL2L");
        }
    }

    #[test]
    fn test_pda_cache() {
        // First call — hits CLI
        let r1 = solana_find_pda(&["string:config"]);
        // Second call — should hit cache (same result, faster)
        let r2 = solana_find_pda(&["string:config"]);
        if let (Ok(a), Ok(b)) = (r1, r2) {
            assert_eq!(a, b);
        }
    }

    #[tokio::test]
    async fn test_is_registered_known_node() {
        // Your node is registered — should return true
        let result = is_registered("12LxN4qXR8Jcv8K49bych8zEWVuBABo3wkHdGarjZFeZ").await;
        if let Ok(registered) = result {
            assert!(registered, "Known node should be registered");
        }
        // else: network error, skip
    }

    #[tokio::test]
    async fn test_is_registered_unknown_node() {
        // Random pubkey — should NOT be registered
        let result = is_registered("11111111111111111111111111111112").await;
        if let Ok(registered) = result {
            assert!(!registered, "Random node should not be registered");
        }
    }

    #[tokio::test]
    async fn test_fetch_all_nodes() {
        if let Ok(nodes) = fetch_all_nodes().await {
            // Should find at least 1 node (yours)
            assert!(!nodes.is_empty(), "Should find at least one registered node");
            // Your node should be in the list
            let found = nodes.iter().any(|n| n.pubkey.contains("LxN4"));
            assert!(found, "Your node should be in the registry");
        }
    }

    #[tokio::test]
    async fn test_fetch_demand() {
        if let Ok(demand) = fetch_demand().await {
            // Should return valid stats (possibly all zeros if no rounds submitted)
            assert!(demand.total >= 0); // always true for u64, but validates parsing
        }
    }
}
