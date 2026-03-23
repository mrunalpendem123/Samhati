use anyhow::Result;
use std::path::PathBuf;
use std::process::{Child, Command};

pub struct NodeRunner {
    child: Option<Child>,
    pub port: u16,
    pub model_name: String,
    pub model_path: PathBuf,
}

impl NodeRunner {
    pub fn new() -> Self {
        Self {
            child: None,
            port: 8000,
            model_name: String::new(),
            model_path: PathBuf::new(),
        }
    }

    /// Start the mesh-node API server with the given GGUF model.
    /// Uses llama-server (from llama.cpp) if available, otherwise falls back
    /// to the mesh-node binary's local inference mode.
    pub fn start(&mut self, model_name: &str, model_path: &PathBuf) -> Result<()> {
        self.stop(); // kill any existing

        self.model_name = model_name.to_string();
        self.model_path = model_path.clone();

        // Try llama-server first (most compatible with GGUF)
        // llama-server provides an OpenAI-compatible API at localhost:8000
        let child = if which_exists("llama-server") {
            Command::new("llama-server")
                .args([
                    "-m",
                    &model_path.display().to_string(),
                    "--port",
                    &self.port.to_string(),
                    "--host",
                    "127.0.0.1",
                    "-c",
                    "4096",
                    "-ngl",
                    "99", // offload all layers to GPU if available
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        } else if which_exists("llama-cli") {
            // Older llama.cpp binary name
            Command::new("llama-cli")
                .args([
                    "--server",
                    "-m",
                    &model_path.display().to_string(),
                    "--port",
                    &self.port.to_string(),
                    "--host",
                    "127.0.0.1",
                    "-c",
                    "4096",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        } else {
            // Fall back to Python llama-cpp-python server
            Command::new("python3")
                .args([
                    "-m",
                    "llama_cpp.server",
                    "--model",
                    &model_path.display().to_string(),
                    "--port",
                    &self.port.to_string(),
                    "--host",
                    "127.0.0.1",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        };

        match child {
            Ok(c) => {
                self.child = Some(c);
                Ok(())
            }
            Err(e) => Err(anyhow::anyhow!(
                "Failed to start inference server. Install llama.cpp: brew install llama.cpp\nError: {}",
                e
            )),
        }
    }

    pub fn stop(&mut self) {
        if let Some(ref mut child) = self.child {
            child.kill().ok();
            child.wait().ok();
        }
        self.child = None;
    }

    pub fn is_running(&self) -> bool {
        self.child.is_some()
    }
}

impl Drop for NodeRunner {
    fn drop(&mut self) {
        self.stop();
    }
}

fn which_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}
