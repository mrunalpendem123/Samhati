use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::protocol::{CapabilityPayload, ModelSummary};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedState {
    pub peers: Vec<CapabilityPayload>,
    pub models: Vec<PersistedModels>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedModels {
    pub node_id: String,
    pub models: Vec<ModelSummary>,
}

impl PersistedState {
    pub fn empty() -> Self {
        Self {
            peers: Vec::new(),
            models: Vec::new(),
        }
    }
}

pub struct StateStore {
    path: PathBuf,
}

impl StateStore {
    pub fn new(dir: &str) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let path = PathBuf::from(dir).join("state.json");
        if !path.exists() {
            let data = serde_json::to_string_pretty(&PersistedState::empty())?;
            fs::write(&path, data)
                .with_context(|| format!("initialize state file {}", path.display()))?;
        }
        Ok(Self { path })
    }

    pub fn load(&self) -> Result<PersistedState> {
        if !self.path.exists() {
            return Ok(PersistedState::empty());
        }
        let data = fs::read_to_string(&self.path)
            .with_context(|| format!("read state file {}", self.path.display()))?;
        let state = serde_json::from_str(&data)
            .with_context(|| format!("parse state file {}", self.path.display()))?;
        Ok(state)
    }

    pub fn save(&self, state: &PersistedState) -> Result<()> {
        let data = serde_json::to_string_pretty(state)?;
        fs::write(&self.path, data)
            .with_context(|| format!("write state file {}", self.path.display()))?;
        Ok(())
    }

    pub fn path(&self) -> &PathBuf {
        &self.path
    }
}
