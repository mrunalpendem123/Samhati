use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    #[serde(alias = "num_hidden_layers", alias = "n_layer")]
    pub n_layers: Option<usize>,

    #[serde(alias = "num_key_value_heads", alias = "n_kv_heads")]
    pub n_kv_heads: Option<usize>,

    #[serde(alias = "head_dim")]
    pub head_dim: Option<usize>,

    #[serde(alias = "hidden_size", alias = "n_embd")]
    pub hidden_size: Option<usize>,

    #[serde(alias = "num_attention_heads", alias = "n_head")]
    pub n_heads: Option<usize>,

    #[serde(alias = "vocab_size")]
    pub vocab_size: Option<usize>,

    #[serde(alias = "model_type")]
    pub model_type: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub params_b: f64,
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub hidden_size: usize,
    pub n_heads: usize,
    pub vocab_size: Option<usize>,
    pub model_type: Option<String>,
}

#[derive(Debug)]
pub enum ConfigError {
    Io(std::io::Error),
    Json(serde_json::Error),
    MissingField(&'static str),
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io error: {e}"),
            Self::Json(e) => write!(f, "json error: {e}"),
            Self::MissingField(name) => write!(f, "missing required field: {name}"),
        }
    }
}

impl std::error::Error for ConfigError {}

impl From<std::io::Error> for ConfigError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for ConfigError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

impl ModelConfig {
    pub fn from_json_file<P: AsRef<Path>>(path: P, params_b: f64) -> Result<Self, ConfigError> {
        let data = fs::read_to_string(path)?;
        let raw: RawConfig = serde_json::from_str(&data)?;
        Self::from_raw(raw, params_b)
    }

    pub fn from_raw(raw: RawConfig, params_b: f64) -> Result<Self, ConfigError> {
        let n_layers = raw.n_layers.ok_or(ConfigError::MissingField("n_layers"))?;
        let hidden_size = raw
            .hidden_size
            .ok_or(ConfigError::MissingField("hidden_size"))?;
        let n_heads = raw.n_heads.ok_or(ConfigError::MissingField("n_heads"))?;

        let head_dim = match raw.head_dim {
            Some(v) => v,
            None => {
                if n_heads == 0 {
                    return Err(ConfigError::MissingField("head_dim"));
                }
                hidden_size / n_heads
            }
        };

        let n_kv_heads = raw.n_kv_heads.unwrap_or(n_heads);

        Ok(Self {
            params_b,
            n_layers,
            n_kv_heads,
            head_dim,
            hidden_size,
            n_heads,
            vocab_size: raw.vocab_size,
            model_type: raw.model_type,
        })
    }
}
