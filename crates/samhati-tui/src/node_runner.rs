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

        // Try TOPLOC-enabled llama-server first (PrimeIntellect-strength proofs)
        // Falls back to system llama-server if TOPLOC build not available
        let toploc_server = dirs::home_dir()
            .unwrap_or_default()
            .join("Samhati/llama-toploc/bin/llama-server");
        let server_bin = if toploc_server.exists() {
            toploc_server.to_string_lossy().to_string()
        } else if which_exists("llama-server") {
            "llama-server".to_string()
        } else {
            String::new()
        };

        let child = if !server_bin.is_empty() {
            Command::new(&server_bin)
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
                Ok(()) // server starts in background, health checked by main loop
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

        let toploc_server = dirs::home_dir()
            .unwrap_or_default()
            .join("Samhati/llama-toploc/bin/llama-server");
        let server_bin = if toploc_server.exists() {
            toploc_server.to_string_lossy().to_string()
        } else if which_exists("llama-server") {
            "llama-server".to_string()
        } else {
            String::new()
        };

        let child = if !server_bin.is_empty() {
            Command::new(&server_bin)
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
                Ok(port) // server starts in background, no blocking
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
