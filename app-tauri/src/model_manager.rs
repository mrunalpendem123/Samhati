use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::RwLock;

/// Information about a single downloadable model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub domain: String,
    pub size_bytes: u64,
    pub size_display: String,
    pub description: String,
    pub smti_bonus: String,
    pub downloaded: bool,
    pub download_progress: Option<f64>,
}

/// Dashboard statistics payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardStats {
    pub earnings_today: u64,
    pub earnings_week: u64,
    pub earnings_total: u64,
    pub inferences_served: u64,
    pub uptime_secs: i64,
    pub elo_score: i32,
    pub elo_history: Vec<EloPoint>,
    pub domain_breakdown: Vec<DomainStat>,
    pub network_nodes_online: u64,
    pub network_inferences_24h: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EloPoint {
    pub timestamp: String,
    pub score: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainStat {
    pub domain: String,
    pub count: u64,
    pub percentage: f64,
}

/// Manages model downloads and local model registry.
pub struct ModelManager {
    models_dir: PathBuf,
    download_progress: RwLock<HashMap<String, f64>>,
}

impl ModelManager {
    pub fn new(app_data_dir: &PathBuf) -> Self {
        let models_dir = app_data_dir.join("models");
        fs::create_dir_all(&models_dir).ok();
        Self {
            models_dir,
            download_progress: RwLock::new(HashMap::new()),
        }
    }

    /// Return the built-in model registry with download status.
    pub fn list_models(&self) -> Vec<ModelInfo> {
        let registry = Self::model_registry();
        registry
            .into_iter()
            .map(|mut m| {
                let model_path = self.models_dir.join(format!("{}.gguf", m.id));
                m.downloaded = model_path.exists();
                m.download_progress = self
                    .download_progress
                    .read()
                    .ok()
                    .and_then(|p| p.get(&m.id).copied());
                m
            })
            .collect()
    }

    /// Download a model by ID (stub with simulated progress).
    pub async fn download_model(&self, model_id: &str) -> Result<(), String> {
        let model = Self::model_registry()
            .into_iter()
            .find(|m| m.id == model_id)
            .ok_or_else(|| format!("Unknown model: {}", model_id))?;

        // Set initial progress
        if let Ok(mut progress) = self.download_progress.write() {
            progress.insert(model_id.to_string(), 0.0);
        }

        // TODO: Actually download from HuggingFace
        // For now, simulate by creating a marker file.
        let model_path = self.models_dir.join(format!("{}.gguf", model.id));

        // Simulate progress updates
        for pct in (0..=100).step_by(10) {
            if let Ok(mut progress) = self.download_progress.write() {
                progress.insert(model_id.to_string(), pct as f64);
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }

        // Create marker file
        fs::write(&model_path, format!("placeholder:{}", model.name))
            .map_err(|e| format!("Failed to write model file: {}", e))?;

        // Clear progress
        if let Ok(mut progress) = self.download_progress.write() {
            progress.remove(model_id);
        }

        Ok(())
    }

    /// Built-in model catalogue.
    fn model_registry() -> Vec<ModelInfo> {
        vec![
            ModelInfo {
                id: "llama-3.2-3b".into(),
                name: "Llama 3.2 3B".into(),
                domain: "General".into(),
                size_bytes: 2_000_000_000,
                size_display: "2.0 GB".into(),
                description: "Fast general-purpose model, great for everyday tasks".into(),
                smti_bonus: "Base rate".into(),
                downloaded: false,
                download_progress: None,
            },
            ModelInfo {
                id: "llama-3.2-7b".into(),
                name: "Llama 3.2 7B".into(),
                domain: "General".into(),
                size_bytes: 4_500_000_000,
                size_display: "4.5 GB".into(),
                description: "Balanced general-purpose model with strong reasoning".into(),
                smti_bonus: "Base rate".into(),
                downloaded: false,
                download_progress: None,
            },
            ModelInfo {
                id: "hindi-specialist-7b".into(),
                name: "Hindi Specialist 7B".into(),
                domain: "Hindi".into(),
                size_bytes: 4_800_000_000,
                size_display: "4.8 GB".into(),
                description: "Optimised for Hindi language understanding and generation".into(),
                smti_bonus: "+50% SMTI on Hindi queries".into(),
                downloaded: false,
                download_progress: None,
            },
            ModelInfo {
                id: "code-specialist-7b".into(),
                name: "Code Specialist 7B".into(),
                domain: "Code".into(),
                size_bytes: 4_600_000_000,
                size_display: "4.6 GB".into(),
                description: "Tuned for code generation, debugging, and explanation".into(),
                smti_bonus: "+30% SMTI on code queries".into(),
                downloaded: false,
                download_progress: None,
            },
            ModelInfo {
                id: "medical-specialist-7b".into(),
                name: "Medical Specialist 7B".into(),
                domain: "Medical".into(),
                size_bytes: 4_700_000_000,
                size_display: "4.7 GB".into(),
                description: "Medical knowledge Q&A (not a substitute for professional advice)".into(),
                smti_bonus: "+40% SMTI on medical queries".into(),
                downloaded: false,
                download_progress: None,
            },
            ModelInfo {
                id: "math-specialist-7b".into(),
                name: "Math Specialist 7B".into(),
                domain: "Math".into(),
                size_bytes: 4_500_000_000,
                size_display: "4.5 GB".into(),
                description: "Strong mathematical reasoning and step-by-step proofs".into(),
                smti_bonus: "+35% SMTI on math queries".into(),
                downloaded: false,
                download_progress: None,
            },
        ]
    }
}
