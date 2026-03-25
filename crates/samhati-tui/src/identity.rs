//! Unified node identity — one Ed25519 keypair for everything.
//!
//! The same 32-byte secret key derives:
//!   - Solana pubkey (on-chain identity, ELO, rewards)
//!   - iroh NodeId (P2P networking, QUIC connections)
//!   - TOPLOC signer (proof of honest inference)
//!
//! Stored at ~/.samhati/identity.json (Solana-compatible 64-byte format).

use anyhow::Result;
use ed25519_dalek::{SigningKey, VerifyingKey};
use std::fs;
use std::path::PathBuf;

/// Unified identity for a Samhati node.
pub struct NodeIdentity {
    /// 32-byte Ed25519 secret key seed.
    pub secret_key: [u8; 32],
    /// 32-byte Ed25519 public key.
    pub public_key: [u8; 32],
    /// Solana-format base58 public key string.
    pub solana_pubkey: String,
    /// iroh-format NodeId (hex-encoded public key).
    pub iroh_node_id: String,
    /// Short display form: "ABC1...XYZ9"
    pub short_id: String,
    /// Where the keypair is stored.
    pub path: PathBuf,
}

impl NodeIdentity {
    /// Load or create the node identity from ~/.samhati/identity.json.
    ///
    /// If an existing Solana wallet exists at ~/.config/solana/id.json or
    /// the old samhati path, migrates it to the canonical location.
    pub fn load_or_create() -> Result<Self> {
        let samhati_dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".samhati");
        let identity_path = samhati_dir.join("identity.json");

        // Check existing locations (in priority order)
        let solana_default = dirs::home_dir()
            .unwrap_or_default()
            .join(".config/solana/id.json");
        let old_samhati = dirs::data_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default())
            .join("samhati/wallet.json");

        let source_path = if identity_path.exists() {
            identity_path.clone()
        } else if solana_default.exists() {
            solana_default
        } else if old_samhati.exists() {
            old_samhati
        } else {
            // Generate new keypair
            let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
            let mut bytes = Vec::with_capacity(64);
            bytes.extend_from_slice(signing_key.as_bytes());
            bytes.extend_from_slice(signing_key.verifying_key().as_bytes());

            fs::create_dir_all(&samhati_dir)?;
            let json = serde_json::to_string(&bytes)?;
            fs::write(&identity_path, &json)?;
            // Set file permissions to 0600 (owner read/write only) — contains private key
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&identity_path, fs::Permissions::from_mode(0o600))?;
            }
            identity_path.clone()
        };

        // Read keypair bytes
        let data = fs::read_to_string(&source_path)?;
        let bytes: Vec<u8> = serde_json::from_str(&data)?;
        if bytes.len() != 64 {
            return Err(anyhow::anyhow!(
                "Invalid keypair: expected 64 bytes, got {}",
                bytes.len()
            ));
        }

        let mut secret_key = [0u8; 32];
        let mut public_key = [0u8; 32];
        secret_key.copy_from_slice(&bytes[..32]);
        public_key.copy_from_slice(&bytes[32..64]);

        // If loaded from an old path, copy to canonical location
        if source_path != identity_path {
            fs::create_dir_all(&samhati_dir)?;
            fs::copy(&source_path, &identity_path)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(&identity_path, fs::Permissions::from_mode(0o600)).ok();
            }
        }

        let solana_pubkey = bs58::encode(&public_key).into_string();
        let iroh_node_id = hex::encode(&public_key);
        let short_id = if solana_pubkey.len() > 16 {
            format!("{}...{}", &solana_pubkey[..6], &solana_pubkey[solana_pubkey.len() - 6..])
        } else {
            solana_pubkey.clone()
        };

        Ok(Self {
            secret_key,
            public_key,
            solana_pubkey,
            iroh_node_id,
            short_id,
            path: identity_path,
        })
    }

    /// Get the Ed25519 signing key (for TOPLOC proofs and Solana transactions).
    pub fn signing_key(&self) -> SigningKey {
        SigningKey::from_bytes(&self.secret_key)
    }

    /// Get the Ed25519 verifying key.
    pub fn verifying_key(&self) -> VerifyingKey {
        self.signing_key().verifying_key()
    }

    /// Get the 32-byte secret key seed for TOPLOC prover.
    pub fn toploc_signer(&self) -> [u8; 32] {
        self.secret_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use tempfile::TempDir;

    #[test]
    fn test_identity_consistency() {
        // Create a keypair manually and verify all derivations match
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        let secret = signing_key.as_bytes();
        let public = verifying_key.as_bytes();

        // Solana pubkey = base58(public_key)
        let solana = bs58::encode(public).into_string();

        // iroh NodeId = hex(public_key)
        let iroh = hex::encode(public);

        // TOPLOC signer = secret_key bytes
        let toploc = *secret;

        // All derived from the same 32 bytes
        assert_eq!(secret.len(), 32);
        assert_eq!(public.len(), 32);
        assert_eq!(toploc, *secret);
        assert!(!solana.is_empty());
        assert_eq!(iroh.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_load_or_create_generates_new() {
        let tmp = TempDir::new().unwrap();
        // Override HOME so it creates in temp
        env::set_var("HOME", tmp.path());

        let id = NodeIdentity::load_or_create().unwrap();
        assert_eq!(id.secret_key.len(), 32);
        assert_eq!(id.public_key.len(), 32);
        assert!(!id.solana_pubkey.is_empty());
        assert!(!id.iroh_node_id.is_empty());
        assert!(id.path.exists());

        // Load again — should return the same identity
        let id2 = NodeIdentity::load_or_create().unwrap();
        assert_eq!(id.solana_pubkey, id2.solana_pubkey);
        assert_eq!(id.iroh_node_id, id2.iroh_node_id);

        env::remove_var("HOME");
    }
}
