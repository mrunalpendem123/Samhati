//! Real Solana wallet integration — generates/loads Ed25519 keypair,
//! checks SOL balance on devnet, displays real pubkey.

use anyhow::Result;
use std::fs;
use std::path::PathBuf;

/// Solana keypair stored as JSON array of 64 bytes (same format as `solana-keygen`).
pub struct SolanaWallet {
    pub pubkey: String,
    pub keypair_path: PathBuf,
    pub secret_bytes: Vec<u8>,
}

impl SolanaWallet {
    /// Load wallet from ~/.config/solana/id.json or generate a new one.
    pub fn load_or_create() -> Result<Self> {
        let default_path = dirs::home_dir()
            .unwrap_or_default()
            .join(".config/solana/id.json");

        // Also check samhati-specific wallet
        let samhati_path = dirs::data_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default())
            .join("samhati/wallet.json");

        let keypair_path = if default_path.exists() {
            default_path
        } else if samhati_path.exists() {
            samhati_path
        } else {
            // Generate new keypair
            let keypair = generate_keypair();
            if let Some(parent) = samhati_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let json = serde_json::to_string(&keypair)?;
            fs::write(&samhati_path, json)?;
            samhati_path
        };

        let data = fs::read_to_string(&keypair_path)?;
        let bytes: Vec<u8> = serde_json::from_str(&data)?;

        if bytes.len() != 64 {
            return Err(anyhow::anyhow!(
                "Invalid keypair: expected 64 bytes, got {}",
                bytes.len()
            ));
        }

        // Public key is the last 32 bytes of the 64-byte keypair
        let pubkey_bytes = &bytes[32..64];
        let pubkey = bs58::encode(pubkey_bytes).into_string();

        Ok(Self {
            pubkey,
            keypair_path,
            secret_bytes: bytes,
        })
    }

    /// Get SOL balance from Solana devnet RPC.
    pub async fn get_sol_balance(&self) -> Result<f64> {
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getBalance",
            "params": [self.pubkey]
        });

        let resp = client
            .post("https://api.devnet.solana.com")
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let lamports = resp["result"]["value"].as_u64().unwrap_or(0);
        Ok(lamports as f64 / 1_000_000_000.0) // lamports to SOL
    }

    /// Get recent transaction signatures for this wallet.
    pub async fn get_recent_transactions(&self) -> Result<Vec<TxInfo>> {
        let client = reqwest::Client::new();
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getSignaturesForAddress",
            "params": [self.pubkey, {"limit": 10}]
        });

        let resp = client
            .post("https://api.devnet.solana.com")
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        let sigs = resp["result"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .map(|s| {
                        let sig = s["signature"].as_str().unwrap_or("").to_string();
                        let slot = s["slot"].as_u64().unwrap_or(0);
                        let err = s["err"].is_null();
                        let block_time = s["blockTime"].as_i64().unwrap_or(0);
                        let memo = s["memo"].as_str().map(|s| s.to_string());

                        TxInfo {
                            signature: if sig.len() > 20 {
                                format!("{}...{}", &sig[..8], &sig[sig.len() - 8..])
                            } else {
                                sig
                            },
                            slot,
                            success: err,
                            block_time,
                            memo,
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();

        Ok(sigs)
    }

    /// Request an airdrop of SOL on devnet.
    pub async fn request_airdrop(&self, sol: f64) -> Result<String> {
        let client = reqwest::Client::new();
        let lamports = (sol * 1_000_000_000.0) as u64;
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "requestAirdrop",
            "params": [self.pubkey, lamports]
        });

        let resp = client
            .post("https://api.devnet.solana.com")
            .json(&body)
            .send()
            .await?
            .json::<serde_json::Value>()
            .await?;

        if let Some(sig) = resp["result"].as_str() {
            Ok(sig.to_string())
        } else if let Some(err) = resp["error"]["message"].as_str() {
            Err(anyhow::anyhow!("Airdrop failed: {}", err))
        } else {
            Err(anyhow::anyhow!("Airdrop failed: unknown error"))
        }
    }

    pub fn short_pubkey(&self) -> String {
        if self.pubkey.len() > 16 {
            format!("{}...{}", &self.pubkey[..8], &self.pubkey[self.pubkey.len() - 8..])
        } else {
            self.pubkey.clone()
        }
    }
}

#[derive(Debug, Clone)]
pub struct TxInfo {
    pub signature: String,
    pub slot: u64,
    pub success: bool,
    pub block_time: i64,
    pub memo: Option<String>,
}

/// Generate a random 64-byte Ed25519 keypair (Solana format).
fn generate_keypair() -> Vec<u8> {
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    let signing_key = SigningKey::generate(&mut OsRng);
    let mut bytes = Vec::with_capacity(64);
    bytes.extend_from_slice(signing_key.as_bytes());
    bytes.extend_from_slice(signing_key.verifying_key().as_bytes());
    bytes
}
