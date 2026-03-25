use anyhow::Result;
use std::path::PathBuf;

pub struct ModelDownloader {
    cache_dir: PathBuf,
}

impl ModelDownloader {
    pub fn new() -> Self {
        let cache_dir = dirs::cache_dir()
            .unwrap_or_else(|| dirs::home_dir().unwrap_or_default())
            .join("samhati/models");
        std::fs::create_dir_all(&cache_dir).ok();
        Self { cache_dir }
    }

    /// Check if model GGUF is already downloaded
    pub fn is_downloaded(&self, model_name: &str) -> bool {
        self.model_path(model_name).exists()
    }

    /// Get the local path for a model's GGUF file
    pub fn model_path(&self, model_name: &str) -> PathBuf {
        let filename = format!("{}.gguf", model_name.to_lowercase().replace(' ', "-"));
        self.cache_dir.join(filename)
    }

    /// Download a GGUF model from HuggingFace. Returns the local path.
    /// Uses the HuggingFace Hub API to find and download Q4_K_M quantized GGUF files.
    pub async fn download(
        &self,
        model_name: &str,
        progress_callback: impl Fn(f64) + Send,
    ) -> Result<PathBuf> {
        let (repo, filename) = map_model_to_hf(model_name)?;

        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            repo, filename
        );
        let path = self.model_path(model_name);

        if path.exists() {
            progress_callback(100.0);
            return Ok(path);
        }

        let client = reqwest::Client::new();
        let resp = client.get(&url).send().await?.error_for_status()?;
        let total_size = resp.content_length().unwrap_or(0);

        let mut file = tokio::fs::File::create(&path).await?;
        let mut downloaded: u64 = 0;
        let mut stream = resp.bytes_stream();

        use futures_util::StreamExt;
        use tokio::io::AsyncWriteExt;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            file.write_all(&chunk).await?;
            downloaded += chunk.len() as u64;
            if total_size > 0 {
                progress_callback((downloaded as f64 / total_size as f64) * 100.0);
            }
        }

        file.flush().await?;
        progress_callback(100.0);
        Ok(path)
    }
}

/// Map model display names to HuggingFace repos with GGUF files.
/// These are real repos that actually exist on HuggingFace.
fn map_model_to_hf(model_name: &str) -> Result<(&'static str, &'static str)> {
    let mapping = match model_name {
        // ── Tiny ──────────────────────────────────────────────────
        "Qwen2.5-0.5B" => (
            "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
            "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Coder-0.5B" => (
            "Qwen/Qwen2.5-Coder-0.5B-Instruct-GGUF",
            "qwen2.5-coder-0.5b-instruct-q4_k_m.gguf",
        ),
        "Gemma-3-1B" => (
            "bartowski/gemma-3-1b-it-GGUF",
            "gemma-3-1b-it-Q4_K_M.gguf",
        ),
        "SmolLM2-1.7B" => (
            "bartowski/SmolLM2-1.7B-Instruct-GGUF",
            "SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
        ),
        // ── Small ─────────────────────────────────────────────────
        "Qwen2.5-1.5B" => (
            "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
            "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Coder-1.5B" => (
            "Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF",
            "qwen2.5-coder-1.5b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Math-1.5B" => (
            "bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF",
            "Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf",
        ),
        "Qwen2.5-3B" => (
            "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "qwen2.5-3b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Coder-3B" => (
            "Qwen/Qwen2.5-Coder-3B-Instruct-GGUF",
            "qwen2.5-coder-3b-instruct-q4_k_m.gguf",
        ),
        "Llama-3.2-3B" => (
            "bartowski/Llama-3.2-3B-Instruct-GGUF",
            "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        ),
        "Phi-4-mini-3.8B" => (
            "bartowski/Phi-4-mini-instruct-GGUF",
            "Phi-4-mini-instruct-Q4_K_M.gguf",
        ),
        "Gemma-3-4B" => (
            "bartowski/gemma-3-4b-it-GGUF",
            "gemma-3-4b-it-Q4_K_M.gguf",
        ),
        // ── Medium ────────────────────────────────────────────────
        "Qwen2.5-7B" => (
            "Qwen/Qwen2.5-7B-Instruct-GGUF",
            "qwen2.5-7b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Coder-7B" => (
            "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF",
            "qwen2.5-coder-7b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Math-7B" => (
            "bartowski/Qwen2.5-Math-7B-Instruct-GGUF",
            "Qwen2.5-Math-7B-Instruct-Q4_K_M.gguf",
        ),
        "DeepSeek-Coder-V2-Lite" => (
            "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
            "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        ),
        "Llama-3.1-8B" => (
            "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
            "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        ),
        "Mistral-7B-v0.3" => (
            "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
            "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
        ),
        // ── Large ─────────────────────────────────────────────────
        "Qwen2.5-14B" => (
            "Qwen/Qwen2.5-14B-Instruct-GGUF",
            "qwen2.5-14b-instruct-q4_k_m.gguf",
        ),
        "Qwen2.5-Coder-14B" => (
            "Qwen/Qwen2.5-Coder-14B-Instruct-GGUF",
            "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        ),
        "DeepSeek-R1-Distill-14B" => (
            "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
            "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
        ),
        _ => {
            return Err(anyhow::anyhow!(
                "Model not available for download: {}",
                model_name
            ))
        }
    };
    Ok(mapping)
}
