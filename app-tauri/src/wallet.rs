use std::fs;
use std::path::PathBuf;

use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};

/// Information returned to the frontend about the wallet.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletInfo {
    pub pubkey: String,
    pub balance: u64,
    pub pending_rewards: u64,
}

/// Manages a Solana-compatible Ed25519 keypair stored on disk.
pub struct WalletManager {
    keypair_path: PathBuf,
}

impl WalletManager {
    pub fn new(app_data_dir: &PathBuf) -> Self {
        let keypair_path = app_data_dir.join("wallet.key");
        Self { keypair_path }
    }

    /// Load an existing keypair or generate a new one.
    pub fn load_or_create(&self) -> Result<(String, Vec<u8>), String> {
        if self.keypair_path.exists() {
            let bytes = fs::read(&self.keypair_path)
                .map_err(|e| format!("Failed to read wallet file: {}", e))?;
            if bytes.len() != 32 {
                return Err("Corrupt wallet file — expected 32-byte secret key".into());
            }
            let secret: [u8; 32] = bytes
                .try_into()
                .map_err(|_| "Failed to convert wallet bytes".to_string())?;
            let signing_key = SigningKey::from_bytes(&secret);
            let pubkey = bs58::encode(signing_key.verifying_key().as_bytes()).into_string();
            Ok((pubkey, signing_key.to_bytes().to_vec()))
        } else {
            let signing_key = SigningKey::generate(&mut OsRng);
            if let Some(parent) = self.keypair_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|e| format!("Failed to create wallet dir: {}", e))?;
            }
            fs::write(&self.keypair_path, signing_key.to_bytes())
                .map_err(|e| format!("Failed to write wallet file: {}", e))?;
            let pubkey = bs58::encode(signing_key.verifying_key().as_bytes()).into_string();
            Ok((pubkey, signing_key.to_bytes().to_vec()))
        }
    }

    /// Check SMTI balance via Solana RPC (stub — returns stored balance for now).
    pub async fn check_balance(&self, _pubkey: &str) -> Result<u64, String> {
        // TODO: Connect to Solana devnet/mainnet RPC
        // let client = RpcClient::new("https://api.devnet.solana.com");
        // let balance = client.get_balance(&pubkey.parse().unwrap()).await;
        Ok(0)
    }

    /// Claim pending SMTI rewards (stub).
    pub async fn claim_rewards(&self, _pubkey: &str, amount: u64) -> Result<u64, String> {
        if amount == 0 {
            return Err("No rewards to claim".into());
        }
        // TODO: Build and sign Solana transaction to claim from rewards program
        Ok(amount)
    }
}
