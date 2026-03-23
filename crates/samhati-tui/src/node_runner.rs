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
                // Wait for server to be ready (poll health endpoint)
                let port = self.port;
                for _ in 0..30 {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    if let Ok(resp) = reqwest::blocking::Client::new()
                        .get(format!("http://127.0.0.1:{}/health", port))
                        .timeout(std::time::Duration::from_secs(1))
                        .send()
                    {
                        if resp.status().is_success() {
                            return Ok(());
                        }
                    }
                }
                Ok(()) // server started but may still be loading
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

/// Manages multiple llama-server processes for multi-node swarm testing.
pub struct MultiNodeRunner {
    nodes: Vec<(u16, String, Child)>, // (port, model_name, process)
    next_port: u16,
}

impl MultiNodeRunner {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            next_port: 8001, // 8000 is used by the primary NodeRunner
        }
    }

    /// Start a llama-server on the next available port. Returns the port used.
    pub fn start_node(&mut self, model_name: &str, model_path: &PathBuf) -> Result<u16> {
        let port = self.next_port;
        self.next_port += 1;

        let child = if which_exists("llama-server") {
            Command::new("llama-server")
                .args([
                    "-m",
                    &model_path.display().to_string(),
                    "--port",
                    &port.to_string(),
                    "--host",
                    "127.0.0.1",
                    "-c",
                    "4096",
                    "-ngl",
                    "99",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        } else if which_exists("llama-cli") {
            Command::new("llama-cli")
                .args([
                    "--server",
                    "-m",
                    &model_path.display().to_string(),
                    "--port",
                    &port.to_string(),
                    "--host",
                    "127.0.0.1",
                    "-c",
                    "4096",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        } else {
            Command::new("python3")
                .args([
                    "-m",
                    "llama_cpp.server",
                    "--model",
                    &model_path.display().to_string(),
                    "--port",
                    &port.to_string(),
                    "--host",
                    "127.0.0.1",
                ])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .spawn()
        };

        match child {
            Ok(c) => {
                self.nodes.push((port, model_name.to_string(), c));
                // Wait for health check
                for _ in 0..30 {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    if let Ok(resp) = reqwest::blocking::Client::new()
                        .get(format!("http://127.0.0.1:{}/health", port))
                        .timeout(std::time::Duration::from_secs(1))
                        .send()
                    {
                        if resp.status().is_success() {
                            return Ok(port);
                        }
                    }
                }
                Ok(port) // started but may still be loading
            }
            Err(e) => Err(anyhow::anyhow!(
                "Failed to start node on port {}. Install llama.cpp: brew install llama.cpp\nError: {}",
                port,
                e
            )),
        }
    }

    /// Stop all managed nodes.
    pub fn stop_all(&mut self) {
        for (_, _, ref mut child) in &mut self.nodes {
            child.kill().ok();
            child.wait().ok();
        }
        self.nodes.clear();
    }

    /// Get list of (port, model_name) for running nodes.
    pub fn running_nodes(&self) -> Vec<(u16, String)> {
        self.nodes.iter().map(|(p, m, _)| (*p, m.clone())).collect()
    }

    /// Number of running swarm nodes (not counting primary).
    pub fn count(&self) -> usize {
        self.nodes.len()
    }
}

impl Drop for MultiNodeRunner {
    fn drop(&mut self) {
        self.stop_all();
    }
}
