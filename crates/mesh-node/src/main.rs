use anyhow::{anyhow, Result};
mod inference;
mod protocol;
mod scheduler;
mod server;
mod state;
mod api;
mod persist;
use cluster_manager::ClusterConstraints;
use iroh::protocol::Router;
use iroh::{Endpoint, EndpointId};
use iroh_gossip::{api::Event, Gossip, TopicId};
use protocol::INFERENCE_ALPN;
#[allow(unused_imports)]
use server::{InferenceServer, MockInferenceServer};
use inference::{infer_http, infer_local_exec, ChatMessage, LocalExecConfig};
use inference_coordinator::{ModelShardRunner, ModelShardRunnerConfig, Coordinator, EchoExecutor, InferenceRequest, IrohDistributedExecutor, RoundRobinPlanner, ShardPlan, SwarmPlanner};
use api::{serve as serve_api, ApiConfig};
use model_config::{estimate_required_vram_gb, estimate_weights_gb, EstimateInput, ModelConfig};
use n0_future::StreamExt;
use proximity_router::{PeerId, PeerInfo, PeerMetrics, ProximityRouter, RttProbe, SwarmRegistry};
use proximity_router::swarm::LayerRange;
use scheduler::{select_best, ModelCandidate, SelectionError, SelectionKey};
use protocol::{
    parse_capability, parse_models, parse_ping, parse_pong, CapabilityPayload, HostedLayerRange,
    ModelSummary, ModelsAnnouncement, NodeRole, PingMessage, PongMessage,
};
use persist::{PersistedModels, PersistedState, StateStore};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::fs;
use std::process::Command;
use std::str::FromStr;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};
use state::{PeerState, PendingPing};
use sysinfo::System;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

#[derive(Debug, Clone)]
struct ModelOption {
    name: String,
    config: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ModelRegistryEntry {
    name: String,
    config_path: String,
    params_b: f64,
    quant_bits: Option<u8>,
    kv_bits: Option<u8>,
    max_context: Option<usize>,
}

#[derive(Debug, Clone)]
struct LocalReport {
    available_gb: f64,
    advertised_free_gb: f64,
    role: NodeRole,
}

impl LocalReport {
    fn significant_change(&self, prev: &Self) -> bool {
        if self.role != prev.role {
            return true;
        }
        (self.available_gb - prev.available_gb).abs() >= 0.5
            || (self.advertised_free_gb - prev.advertised_free_gb).abs() >= 0.5
    }
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    match args[1].as_str() {
        "estimate" => cmd_estimate(&args[2..]),
        "capacity" => {
            if let Err(e) = cmd_capacity(&args[2..]) {
                eprintln!("capacity error: {e}");
                std::process::exit(2);
            }
        }
        "dist-plan" => {
            if let Err(e) = cmd_dist_plan(&args[2..]) {
                eprintln!("dist-plan error: {e}");
                std::process::exit(2);
            }
        }
        "dist-run" => {
            if let Err(e) = cmd_dist_run(&args[2..]).await {
                eprintln!("dist-run error: {e}");
                std::process::exit(2);
            }
        }
        "simulate" => cmd_simulate(&args[2..]),
        "gossip" => {
            if let Err(e) = cmd_gossip(&args[2..]).await {
                eprintln!("gossip error: {e}");
                std::process::exit(2);
            }
        }
        "serve" => {
            if let Err(e) = cmd_serve(&args[2..]).await {
                eprintln!("serve error: {e}");
                std::process::exit(2);
            }
        }
        "infer-http" => {
            if let Err(e) = cmd_infer_http(&args[2..]).await {
                eprintln!("infer-http error: {e}");
                std::process::exit(2);
            }
        }
        "infer-local" => {
            if let Err(e) = cmd_infer_local(&args[2..]).await {
                eprintln!("infer-local error: {e}");
                std::process::exit(2);
            }
        }
        "api" => {
            if let Err(e) = cmd_api(&args[2..]).await {
                eprintln!("api error: {e}");
                std::process::exit(2);
            }
        }
        _ => print_usage(),
    }
}

fn cmd_estimate(args: &[String]) {
    let config_path = get_arg(args, "--config").unwrap_or_else(|| {
        eprintln!("missing --config");
        print_usage();
        std::process::exit(2);
    });

    let params_b = get_arg(args, "--params-b")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| {
            eprintln!("--params-b is required (model size in billions, e.g. 7.0, 13.0, 70.0)");
            std::process::exit(2);
        });

    let cfg = match ModelConfig::from_json_file(&config_path, params_b) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("config error: {e}");
            std::process::exit(2);
        }
    };

    let mut input = EstimateInput::default();
    apply_estimate_args(args, &mut input);

    let weights_gb = estimate_weights_gb(&cfg, input.quant_bits, input.overhead);
    let required_gb = estimate_required_vram_gb(&cfg, &input);

    println!("model params: {:.1}B", cfg.params_b);
    println!("quant bits: {:.1}", input.quant_bits);
    println!("context: {}", input.seq_len);
    println!("batch: {}", input.batch);
    println!("nodes: {}", input.nodes);
    println!("weights only (GB): {:.2}", weights_gb);
    println!("required per node (GB): {:.2}", required_gb);
}

fn cmd_capacity(args: &[String]) -> Result<()> {
    let free_vram = get_arg(args, "--free-vram")
        .and_then(|v| v.parse::<f64>().ok())
        .ok_or_else(|| anyhow!("--free-vram is required"))?;
    let max_nodes = get_arg(args, "--max-nodes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8);
    let current_nodes = get_arg(args, "--current-nodes")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(0);
    let seq = get_arg(args, "--seq")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(4096);
    let quant = get_arg(args, "--quant")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(4.0);
    let kv_bytes = get_arg(args, "--kv-bytes")
        .and_then(|v| v.parse::<f64>().ok())
        .or_else(|| {
            get_arg(args, "--kv-bits").and_then(|v| match v.parse::<u8>().ok()? {
                4 => Some(0.5),
                8 => Some(1.0),
                16 => Some(2.0),
                _ => None,
            })
        })
        .unwrap_or(1.0);

    let (cfg, model_name) = if let Some(path) = get_arg(args, "--models") {
        let entries = load_model_registry(&path)?;
        let name = get_arg(args, "--model")
            .or_else(|| get_arg(args, "--model-name"))
            .unwrap_or_else(|| entries.get(0).map(|e| e.name.clone()).unwrap_or_default());
        let entry = entries
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("model '{name}' not found in registry"))?;
        let cfg = ModelConfig::from_json_file(&entry.config_path, entry.params_b)?;
        (cfg, entry.name)
    } else {
        let config_path = get_arg(args, "--config")
            .ok_or_else(|| anyhow!("--config or --models is required"))?;
        let params_b = get_arg(args, "--params-b")
            .and_then(|v| v.parse::<f64>().ok())
            .ok_or_else(|| anyhow!("--params-b is required when using --config (model size in billions, e.g. 7.0)"))?;
        let cfg = ModelConfig::from_json_file(&config_path, params_b)?;
        (cfg, "custom".to_string())
    };

    let mut input = EstimateInput::default();
    input.seq_len = seq;
    input.quant_bits = quant;
    input.bytes_per_kv = kv_bytes;

    println!("model: {model_name}");
    println!("free_vram_per_node_gb: {free_vram:.2}");
    println!("context: {seq}  quant_bits: {quant}  kv_bytes: {kv_bytes}");

    let mut min_nodes = None;
    for nodes in 1..=max_nodes {
        input.nodes = nodes;
        let required = estimate_required_vram_gb(&cfg, &input);
        let fits = required <= free_vram;
        println!(
            "nodes={nodes} required_per_node_gb={required:.2} fits={fits}"
        );
        if fits && min_nodes.is_none() {
            min_nodes = Some(nodes);
        }
    }

    if let Some(min) = min_nodes {
        if current_nodes > 0 {
            let needed = min.saturating_sub(current_nodes);
            println!("min_nodes_required={min} additional_peers_needed={needed}");
        } else {
            println!("min_nodes_required={min}");
        }
    } else {
        println!("min_nodes_required=none (increase max_nodes or reduce context/quant)");
    }

    Ok(())
}

fn cmd_dist_plan(args: &[String]) -> Result<()> {
    let peers_arg = get_arg(args, "--peers")
        .ok_or_else(|| anyhow!("--peers is required (comma-separated peer ids)"))?;
    let peers = peers_arg
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    if peers.is_empty() {
        return Err(anyhow!("no peers provided in --peers"));
    }

    let min_layers = get_arg(args, "--min-layers-per-peer")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);

    let (model_name, total_layers) = if let Some(path) = get_arg(args, "--models") {
        let entries = load_model_registry(&path)?;
        let name = get_arg(args, "--model")
            .or_else(|| get_arg(args, "--model-name"))
            .unwrap_or_else(|| entries.get(0).map(|e| e.name.clone()).unwrap_or_default());
        let entry = entries
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("model '{name}' not found in registry"))?;
        let cfg = ModelConfig::from_json_file(&entry.config_path, entry.params_b)?;
        (entry.name, cfg.n_layers)
    } else {
        let config_path = get_arg(args, "--config")
            .ok_or_else(|| anyhow!("--config or --models is required"))?;
        let params_b = get_arg(args, "--params-b")
            .and_then(|v| v.parse::<f64>().ok())
            .ok_or_else(|| anyhow!("--params-b is required when using --config (model size in billions, e.g. 7.0)"))?;
        let cfg = ModelConfig::from_json_file(&config_path, params_b)?;
        let name = get_arg(args, "--model")
            .or_else(|| get_arg(args, "--model-name"))
            .unwrap_or_else(|| "custom".to_string());
        (name, cfg.n_layers)
    };

    let planner = RoundRobinPlanner::new(peers, min_layers);
    let plan: ShardPlan = planner.plan(&model_name, total_layers)?;
    println!("{}", serde_json::to_string_pretty(&plan)?);
    Ok(())
}

async fn cmd_dist_run(args: &[String]) -> Result<()> {
    let peers_arg = get_arg(args, "--peers")
        .ok_or_else(|| anyhow!("--peers is required (comma-separated peer ids)"))?;
    let peers = peers_arg
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    if peers.is_empty() {
        return Err(anyhow!("no peers provided in --peers"));
    }

    let min_layers = get_arg(args, "--min-layers-per-peer")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(1);

    let (model_name, total_layers) = if let Some(path) = get_arg(args, "--models") {
        let entries = load_model_registry(&path)?;
        let name = get_arg(args, "--model")
            .or_else(|| get_arg(args, "--model-name"))
            .unwrap_or_else(|| entries.get(0).map(|e| e.name.clone()).unwrap_or_default());
        let entry = entries
            .into_iter()
            .find(|e| e.name == name)
            .ok_or_else(|| anyhow!("model '{name}' not found in registry"))?;
        let cfg = ModelConfig::from_json_file(&entry.config_path, entry.params_b)?;
        (entry.name, cfg.n_layers)
    } else {
        let config_path = get_arg(args, "--config")
            .ok_or_else(|| anyhow!("--config or --models is required"))?;
        let params_b = get_arg(args, "--params-b")
            .and_then(|v| v.parse::<f64>().ok())
            .ok_or_else(|| anyhow!("--params-b is required when using --config (model size in billions, e.g. 7.0)"))?;
        let cfg = ModelConfig::from_json_file(&config_path, params_b)?;
        let name = get_arg(args, "--model")
            .or_else(|| get_arg(args, "--model-name"))
            .unwrap_or_else(|| "custom".to_string());
        (name, cfg.n_layers)
    };

    let planner = RoundRobinPlanner::new(peers.clone(), min_layers);
    let plan = planner.plan(&model_name, total_layers)?;

    let request = InferenceRequest {
        request_id: "dist-run-1".to_string(),
        input: get_arg(args, "--input").unwrap_or_else(|| "Hello".to_string()),
        max_tokens: get_arg(args, "--max-tokens")
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(64),
        temperature: get_arg(args, "--temperature")
            .and_then(|v| v.parse::<f32>().ok())
            .unwrap_or(0.2),
    };

    let executor_name = get_arg(args, "--executor").unwrap_or_else(|| "echo".to_string());
    if executor_name == "burn" {
        let model_path = get_arg(args, "--model-path");
        let backend = get_arg(args, "--model-device").unwrap_or_else(|| "ndarray".to_string());
        let mode = get_arg(args, "--model-mode").unwrap_or_else(|| "simulate".to_string());
        let hidden = get_arg(args, "--model-hidden")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(8);
        let tensor_name = get_arg(args, "--model-tensor");
        let sample_bytes = get_arg(args, "--model-sample-bytes")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(256);
        let shard = plan
            .shards
            .get(0)
            .ok_or_else(|| anyhow!("plan has no shards"))?
            .clone();
        let runner = ModelShardRunner::new(ModelShardRunnerConfig {
            model_path,
            layer_start: shard.layer_start,
            layer_end: shard.layer_end,
            backend,
            mode,
            hidden,
            tensor_name,
            sample_bytes,
        });
        let coordinator = Coordinator::new(plan, runner);
        let output = coordinator.generate(request).await?;
        println!("output: {output}");
        return Ok(());
    }

    if executor_name == "iroh" {
        // Static peer list: open QUIC streams to each caller-supplied peer,
        // with SwarmRegistry for reputation tracking and failover.
        let endpoint = Endpoint::bind().await?;
        let registry = Arc::new(RwLock::new(SwarmRegistry::new(StdDuration::from_secs(300))));

        // Pre-populate registry with the static peers so failover can work
        for peer_id in &peers {
            registry.write().await.upsert(
                peer_id.clone(),
                proximity_router::PeerMetrics {
                    rtt_ms: 50.0,
                    bandwidth_mbps: 500.0,
                    reliability: 0.9,
                    gpu_capacity_score: 0.8,
                    free_vram_gb: None,
                    last_seen_epoch_ms: None,
                },
                vec![],  // layer info unknown for static peers
                0.0,
                0,
            );
        }

        let max_retries: usize = get_arg(args, "--max-retries")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);

        let executor = IrohDistributedExecutor::new(endpoint)
            .with_registry(registry, &model_name, max_retries);
        let coordinator = Coordinator::new(plan, executor);
        let output = coordinator.generate(request).await?;
        println!("output: {output}");
        return Ok(());
    }

    if executor_name == "swarm" {
        // Dynamic swarm mode: join gossip to discover peers, then use SwarmPlanner.
        let topic_hex = get_arg(args, "--topic")
            .ok_or_else(|| anyhow!("--topic <64-hex> is required for swarm executor"))?;
        let topic = parse_topic_id(&topic_hex)?;
        let bootstrap_ids = parse_bootstrap_ids(args)?;

        let discover_secs: u64 = get_arg(args, "--discover-secs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let max_retries: usize = get_arg(args, "--max-retries")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let layers_per_shard: usize = get_arg(args, "--layers-per-shard")
            .and_then(|v| v.parse().ok())
            .unwrap_or((total_layers / peers.len().max(1)).max(1));

        let endpoint = Endpoint::bind().await?;
        let gossip = Gossip::builder().spawn(endpoint.clone());
        let _router = Router::builder(endpoint.clone())
            .accept(iroh_gossip::ALPN, gossip.clone())
            .spawn();

        println!("joining gossip topic {topic_hex}, discovering peers for {discover_secs}s...");
        let registry = Arc::new(RwLock::new(SwarmRegistry::new(StdDuration::from_secs(120))));
        let registry_clone = registry.clone();

        let (_, mut recv) = gossip.subscribe(topic, bootstrap_ids).await?.split();
        let discovery = tokio::spawn(async move {
            while let Some(Ok(Event::Received(msg))) = recv.next().await {
                let body = String::from_utf8_lossy(&msg.content);
                if let Some(caps) = parse_capability(&body) {
                    let layers: Vec<LayerRange> = caps
                        .layers_hosted
                        .iter()
                        .cloned()
                        .map(LayerRange::from)
                        .collect();
                    let metrics = proximity_router::PeerMetrics {
                        rtt_ms: caps.rtt_ms,
                        bandwidth_mbps: caps.bandwidth_mbps,
                        reliability: caps.reliability,
                        gpu_capacity_score: caps.gpu_capacity_score,
                        free_vram_gb: Some(caps.free_vram_gb),
                        last_seen_epoch_ms: None,
                    };
                    registry_clone.write().await.upsert(
                        caps.node_id.clone(), metrics, layers, caps.load_score, caps.uptime_secs,
                    );
                    println!("  discovered peer {}", caps.node_id);
                }
            }
        });

        tokio::time::sleep(Duration::from_secs(discover_secs)).await;
        discovery.abort();

        let peer_count = registry.read().await.peer_count();
        println!("discovered {peer_count} peer(s). planning inference...");

        let planner = SwarmPlanner::new(layers_per_shard);
        let swarm_plan = {
            let reg = registry.read().await;
            planner.plan(&model_name, total_layers, &reg)?
        };
        println!("shard plan: {} shards", swarm_plan.shards.len());
        for s in &swarm_plan.shards {
            println!("  shard layers={}..{} peer={}", s.layer_start, s.layer_end, s.peer_id);
        }

        let executor = IrohDistributedExecutor::new(endpoint)
            .with_registry(registry, &model_name, max_retries);
        let coordinator = Coordinator::new(swarm_plan, executor);
        let output = coordinator.generate(request).await?;
        println!("output: {output}");
        return Ok(());
    }

    // Default: echo executor (no real inference, useful for testing plan/routing logic)
    let coordinator = Coordinator::new(plan, EchoExecutor::default());
    let result = coordinator.run(request).await?;
    println!("output: {}", result.output);
    for step in result.steps {
        println!(
            "step {} peer={} output={}",
            step.shard_index, step.peer_id, step.output
        );
    }
    Ok(())
}

fn cmd_simulate(args: &[String]) {
    let config_path = get_arg(args, "--config").unwrap_or_else(|| {
        eprintln!("missing --config");
        print_usage();
        std::process::exit(2);
    });

    let params_b = get_arg(args, "--params-b")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or_else(|| {
            eprintln!("--params-b is required (model size in billions, e.g. 7.0, 13.0, 70.0)");
            std::process::exit(2);
        });

    let cfg = match ModelConfig::from_json_file(&config_path, params_b) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("config error: {e}");
            std::process::exit(2);
        }
    };

    let mut input = EstimateInput::default();
    apply_estimate_args(args, &mut input);

    let router = ProximityRouter::default();

    let mut peers = if let Some(path) = get_arg(args, "--peers") {
        match load_peers_from_json(&path) {
            Ok(peers) => peers,
            Err(e) => {
                eprintln!("failed to load peers: {e}");
                std::process::exit(2);
            }
        }
    } else {
        default_peers()
    };

    if let Some(path) = get_arg(args, "--rtt-samples") {
        match load_rtt_samples(&path) {
            Ok(samples) => {
                let probe = StaticRttProbe::new(samples);
                apply_rtt_probe(&mut peers, &probe);
            }
            Err(e) => {
                eprintln!("failed to load rtt samples: {e}");
                std::process::exit(2);
            }
        }
    }

    let ranked = router.rank_peers(peers);
    if ranked.is_empty() {
        eprintln!("no peers passed RTT/reliability filters");
        std::process::exit(3);
    }

    let constraints = ClusterConstraints {
        min_nodes: get_arg(args, "--min-nodes")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2),
        max_nodes: get_arg(args, "--max-nodes")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6),
        min_bandwidth_mbps: get_arg(args, "--min-bw")
            .and_then(|v| v.parse::<f64>().ok()),
    };

    let candidates = vec![ModelCandidate {
        name: "primary",
        config: &cfg,
        ranked,
    }];

    match select_best(&candidates, &constraints, &input) {
        Ok(result) => {
            println!("selected cluster nodes: {:?}", result.selection.nodes);
            println!("nodes_count: {}", result.selection.nodes_count);
            println!(
                "required_vram_per_node_gb: {:.2}",
                result.selection.required_vram_gb
            );
        }
        Err(err) => {
            eprintln!("cluster selection failed: {:?}", err);
            std::process::exit(4);
        }
    }
}

async fn cmd_gossip(args: &[String]) -> Result<()> {
    let topic = match get_arg(args, "--topic") {
        Some(hex_str) => parse_topic_id(&hex_str)?,
        None => TopicId::from_bytes([23u8; 32]),
    };

    let bootstrap = parse_bootstrap_ids(args)?;

    let interval_secs = get_arg(args, "--interval")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(10);

    let mut model_options: Vec<ModelOption> = Vec::new();
    let mut local_models: Vec<ModelSummary> = Vec::new();
    if let Some(path) = get_arg(args, "--config") {
        // --params-b is required when a --config is provided — no silent default.
        let params_b = get_arg(args, "--params-b")
            .and_then(|v| v.parse::<f64>().ok())
            .ok_or_else(|| anyhow!("--params-b is required when --config is provided (model size in billions, e.g. 7.0)"))?;
        let model_cfg = ModelConfig::from_json_file(&path, params_b)?;
        let model_name = get_arg(args, "--model-name").unwrap_or_else(|| "primary".to_string());
        model_options.push(ModelOption {
            name: model_name.clone(),
            config: model_cfg,
        });
        local_models.push(ModelSummary {
            name: model_name,
            params_b,
            quant_bits: get_arg(args, "--quant").and_then(|v| v.parse::<u8>().ok()),
            kv_bits: get_arg(args, "--kv-bits").and_then(|v| v.parse::<u8>().ok()),
            max_context: get_arg(args, "--context").and_then(|v| v.parse::<usize>().ok()),
        });
    }

    let registry_path = get_arg(args, "--models");
    if let Some(path) = registry_path {
        match load_model_registry(&path) {
            Ok(entries) => {
                for entry in entries {
                    let summary = ModelSummary {
                        name: entry.name.clone(),
                        params_b: entry.params_b,
                        quant_bits: entry.quant_bits,
                        kv_bits: entry.kv_bits,
                        max_context: entry.max_context,
                    };
                    local_models.push(summary);
                    match ModelConfig::from_json_file(&entry.config_path, entry.params_b) {
                        Ok(cfg) => model_options.push(ModelOption {
                            name: entry.name,
                            config: cfg,
                        }),
                        Err(e) => {
                            eprintln!(
                                "model registry entry '{}' failed to load: {e}",
                                entry.name
                            );
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("failed to load models registry: {e}");
                std::process::exit(2);
            }
        }
    }

    for entry in get_args(args, "--fallback-model") {
        match parse_model_option(&entry) {
            Ok(opt) => {
                local_models.push(ModelSummary {
                    name: opt.name.clone(),
                    params_b: opt.config.params_b,
                    quant_bits: get_arg(args, "--quant").and_then(|v| v.parse::<u8>().ok()),
                    kv_bits: get_arg(args, "--kv-bits").and_then(|v| v.parse::<u8>().ok()),
                    max_context: get_arg(args, "--context").and_then(|v| v.parse::<usize>().ok()),
                });
                model_options.push(opt);
            }
            Err(e) => {
                eprintln!("invalid --fallback-model '{entry}': {e}");
                std::process::exit(2);
            }
        }
    }

    let mut estimate_input = EstimateInput::default();
    apply_estimate_args(args, &mut estimate_input);
    apply_kv_bits_arg(args, &mut estimate_input);

    let endpoint = Endpoint::bind().await?;
    let node_id = endpoint.id();
    let node_id_str = node_id.to_string();
    let gossip = Gossip::builder().spawn(endpoint.clone());
    let inference_server = MockInferenceServer::new();

    let net_router = Router::builder(endpoint)
        .accept(iroh_gossip::ALPN, gossip.clone())
        .accept(INFERENCE_ALPN, inference_server)
        .spawn();

    let topic_hex = hex::encode(topic.as_bytes());
    println!("node_id: {node_id_str} (len={})", node_id_str.len());
    println!("topic: {topic_hex}");
    println!(
        "bootstrap_cmd: cargo run -p mesh-node -- gossip --topic {topic_hex} --bootstrap {node_id_str}"
    );

    // ── Swarm registry (soft-state, peer_ttl = 3× announce interval) ─────────
    let swarm_ttl = std::time::Duration::from_secs(interval_secs * 3 + 30);
    let swarm_registry: Arc<RwLock<SwarmRegistry>> =
        Arc::new(RwLock::new(SwarmRegistry::new(swarm_ttl)));

    let (sender, mut receiver) = gossip.subscribe(topic, bootstrap).await?.split();
    let sender_for_recv = sender.clone();
    let node_id_for_recv = node_id_str.clone();
    if let Err(_) = tokio::time::timeout(Duration::from_secs(2), receiver.joined()).await {
        eprintln!("gossip join timeout (continuing without confirmation)");
    }

    let state_dir = get_arg(args, "--state-dir").unwrap_or_else(|| "state".to_string());
    let store = match StateStore::new(&state_dir) {
        Ok(store) => Some(store),
        Err(e) => {
            eprintln!("state init error: {e}");
            None
        }
    };
    if let Some(store) = &store {
        println!("state store: {}", store.path().display());
    }
    let (caps, _last_available) = capability_from_args(args, node_id.to_string());
    let cap_message = format!("cap:{}", serde_json::to_string(&caps)?);
    sender.broadcast(cap_message.into_bytes().into()).await?;

    let models_announcement = ModelsAnnouncement {
        node_id: caps.node_id.clone(),
        models: local_models.clone(),
    };
    let models_message = format!("models:{}", serde_json::to_string(&models_announcement)?);
    sender.broadcast(models_message.into_bytes().into()).await?;

    let state: Arc<RwLock<PeerState>> = Arc::new(RwLock::new(PeerState::new()));
    let ping_counter = Arc::new(AtomicU64::new(0));
    {
        let mut st = state.write().await;
        if let Some(store) = &store {
            match store.load() {
                Ok(persisted) => {
                    let peer_count = persisted.peers.len();
                    let model_count = persisted.models.len();
                    for peer in persisted.peers {
                        st.upsert_capability(peer);
                    }
                    for entry in persisted.models {
                        st.upsert_models(entry.node_id, entry.models);
                    }
                    if peer_count > 0 || model_count > 0 {
                        println!("loaded state: peers={} models={}", peer_count, model_count);
                    }
                }
                Err(e) => eprintln!("state load error: {e}"),
            }
        }
        st.upsert_capability(caps.clone());
        st.upsert_models(caps.node_id.clone(), local_models.clone());
        st.mark_seen(&caps.node_id, now_millis());
        if let Some(store) = &store {
            let peers_snapshot = st.peers_snapshot();
            let models_snapshot = st.models_snapshot();
            let persisted = build_persisted_state(peers_snapshot, models_snapshot);
            if let Err(e) = store.save(&persisted) {
                eprintln!("state persist error: {e}");
            }
        }
    }

    let mut ticker = interval(Duration::from_secs(interval_secs.max(1)));
    let state_clone = state.clone();
    let swarm_clone = swarm_registry.clone();
    let recv_task = tokio::spawn(async move {
        while let Some(event) = receiver.next().await {
            match event {
                Ok(Event::Received(message)) => {
                    let body = String::from_utf8_lossy(&message.content);
                    if let Some(caps) = parse_capability(&body) {
                        {
                            let mut st = state_clone.write().await;
                            st.upsert_capability(caps.clone());
                            st.mark_seen(&caps.node_id, now_millis());
                            st.clear_failure(&caps.node_id);
                        }
                        // Register peer in the swarm registry for dynamic routing
                        {
                            let layer_ranges: Vec<LayerRange> = caps
                                .layers_hosted
                                .iter()
                                .cloned()
                                .map(LayerRange::from)
                                .collect();
                            let metrics = PeerMetrics {
                                rtt_ms: caps.rtt_ms,
                                bandwidth_mbps: caps.bandwidth_mbps,
                                reliability: caps.reliability,
                                gpu_capacity_score: caps.gpu_capacity_score,
                                free_vram_gb: Some(caps.free_vram_gb),
                                last_seen_epoch_ms: Some(now_millis() as u64),
                            };
                            swarm_clone
                                .write()
                                .await
                                .upsert(
                                    caps.node_id.clone(),
                                    metrics,
                                    layer_ranges,
                                    caps.load_score,
                                    caps.uptime_secs,
                                );
                        }
                        println!(
                            "capability from {}: role={:?} vram={}GB bw={}Mbps layers={}",
                            caps.node_id,
                            caps.role,
                            caps.free_vram_gb,
                            caps.bandwidth_mbps,
                            caps.layers_hosted.len(),
                        );
                    } else if let Some(announcement) = parse_models(&body) {
                        {
                            let mut st = state_clone.write().await;
                            st.upsert_models(announcement.node_id.clone(), announcement.models.clone());
                            st.mark_seen(&announcement.node_id, now_millis());
                            st.clear_failure(&announcement.node_id);
                        }
                        let model_names = announcement
                            .models
                            .iter()
                            .map(|m| m.name.clone())
                            .collect::<Vec<_>>();
                        println!(
                            "models from {}: {:?}",
                            announcement.node_id, model_names
                        );
                    } else if let Some(ping) = parse_ping(&body) {
                        if ping.target_id == node_id_for_recv {
                            {
                                let mut st = state_clone.write().await;
                                st.mark_seen(&ping.from_id, now_millis());
                                st.clear_failure(&ping.from_id);
                            }
                            let pong = PongMessage {
                                from_id: node_id_for_recv.clone(),
                                target_id: ping.from_id,
                                nonce: ping.nonce,
                            };
                            let pong_msg = format!("pong:{}", serde_json::to_string(&pong).unwrap_or_default());
                            if let Err(e) = sender_for_recv.broadcast(pong_msg.into_bytes().into()).await {
                                eprintln!("pong send error: {e}");
                            }
                        }
                    } else if let Some(pong) = parse_pong(&body) {
                        if pong.target_id == node_id_for_recv {
                            let mut st = state_clone.write().await;
                            st.mark_seen(&pong.from_id, now_millis());
                            st.clear_failure(&pong.from_id);
                            if let Some(PendingPing { peer_id, started }) =
                                st.take_pending(&pong.nonce)
                            {
                                let rtt_ms = started.elapsed().as_secs_f64() * 1000.0;
                                st.set_rtt(peer_id, rtt_ms);
                            }
                        }
                    } else if let Some(heartbeat_id) = parse_heartbeat(&body) {
                        let mut st = state_clone.write().await;
                        st.mark_seen(&heartbeat_id, now_millis());
                        st.clear_failure(&heartbeat_id);
                    } else {
                        println!("recv: {body} (delivered_from={:?})", message.delivered_from);
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    eprintln!("gossip recv error: {e}");
                    break;
                }
            }
        }
    });

    let router = ProximityRouter::default();
    let constraints = ClusterConstraints {
        min_nodes: get_arg(args, "--min-nodes")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2),
        max_nodes: get_arg(args, "--max-nodes")
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(6),
        min_bandwidth_mbps: get_arg(args, "--min-bw")
            .and_then(|v| v.parse::<f64>().ok()),
    };
    let mut last_selection: Option<SelectionKey> = None;
    let mut last_selection_at: Option<Instant> = None;
    let mut last_local_report: Option<LocalReport> = None;
    let rtt_probe_enabled = has_flag(args, "--rtt-probe");
    let rtt_timeout_ms = get_arg(args, "--rtt-timeout-ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(3000);
    let fail_threshold = get_arg(args, "--peer-fail-threshold")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(3);
    let cooldown_ms = get_arg(args, "--peer-cooldown-ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(30_000);
    let peer_ttl_ms = get_arg(args, "--peer-ttl-ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(15_000);
    let stickiness_ms = get_arg(args, "--cluster-stickiness-ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(15_000);
    let load_penalty = get_arg(args, "--load-penalty")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.05);
    let load_decay = get_arg(args, "--load-decay")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.9);
    let mut peer_load: HashMap<String, f64> = HashMap::new();

    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                println!("shutdown requested");
                break;
            }
            _ = ticker.tick(), if interval_secs > 0 => {
                let heartbeat = format!("heartbeat from {node_id}");
                if let Err(e) = sender.broadcast(heartbeat.into_bytes().into()).await {
                    eprintln!("broadcast error: {e}");
                }
                let (caps, available_gb) = capability_from_args(args, node_id.to_string());
                let cap_message = format!("cap:{}", serde_json::to_string(&caps)?);
                if let Err(e) = sender.broadcast(cap_message.into_bytes().into()).await {
                    eprintln!("cap broadcast error: {e}");
                }
                let models_announcement = ModelsAnnouncement {
                    node_id: caps.node_id.clone(),
                    models: local_models.clone(),
                };
                let models_message = format!("models:{}", serde_json::to_string(&models_announcement)?);
                if let Err(e) = sender.broadcast(models_message.into_bytes().into()).await {
                    eprintln!("models broadcast error: {e}");
                }
                {
                    let mut st = state.write().await;
                    st.upsert_capability(caps.clone());
                    st.upsert_models(caps.node_id.clone(), local_models.clone());
                    st.mark_seen(&caps.node_id, now_millis());
                }

                let peers_snapshot = {
                    let st = state.read().await;
                    st.peers_snapshot()
                };

                if rtt_probe_enabled {
                    let peer_ids = peers_snapshot
                        .iter()
                        .filter(|p| p.node_id != node_id_str)
                        .map(|p| p.node_id.clone())
                        .collect::<Vec<_>>();

                    send_rtt_pings(
                        &sender,
                        &node_id_str,
                        &peer_ids,
                        &state,
                        &ping_counter,
                        rtt_timeout_ms,
                        fail_threshold,
                        cooldown_ms,
                    )
                    .await;
                }

                if !model_options.is_empty() {
                    let (peer_models_snapshot, rtt_snapshot, last_seen_snapshot, cooldown_snapshot) = {
                        let st = state.read().await;
                        (
                            st.models_snapshot(),
                            st.rtt_snapshot(),
                            st.last_seen_snapshot(),
                            st.cooldown_snapshot(),
                        )
                    };
                    let now_ms = now_millis();

                    let base_peers = peers_snapshot
                        .iter()
                        .filter(|c| matches!(c.role, NodeRole::Inference))
                        .filter(|c| {
                            let last_seen = last_seen_snapshot
                                .get(&c.node_id)
                                .copied()
                                .unwrap_or(0);
                            now_ms.saturating_sub(last_seen) <= peer_ttl_ms as u128
                        })
                        .filter(|c| {
                            let cooldown_until = cooldown_snapshot
                                .get(&c.node_id)
                                .copied()
                                .unwrap_or(0);
                            now_ms >= cooldown_until
                        })
                        .map(|c| PeerInfo {
                            id: c.node_id.clone(),
                            metrics: PeerMetrics {
                                rtt_ms: rtt_snapshot
                                    .get(&c.node_id)
                                    .copied()
                                    .unwrap_or(c.rtt_ms),
                                bandwidth_mbps: c.bandwidth_mbps,
                                reliability: c.reliability,
                                gpu_capacity_score: c.gpu_capacity_score,
                                free_vram_gb: Some(c.free_vram_gb),
                                last_seen_epoch_ms: None,
                            },
                        })
                        .collect::<Vec<_>>();

                    if base_peers.is_empty() {
                        eprintln!("cluster: no inference peers available");
                    } else {
                        decay_load(&mut peer_load, load_decay);
                        let mut candidates: Vec<ModelCandidate> = Vec::new();
                        for option in &model_options {
                            let model_peers = base_peers
                                .iter()
                                .filter(|p| {
                                    peer_supports_model(&peer_models_snapshot, &p.id, &option.name)
                                })
                                .cloned()
                                .collect::<Vec<_>>();

                            if model_peers.is_empty() {
                                continue;
                            }

                            let ranked = router.rank_peers(model_peers);
                            let ranked = apply_load_penalty(ranked, &peer_load, load_penalty);
                            if ranked.is_empty() {
                                continue;
                            }

                            candidates.push(ModelCandidate {
                                name: option.name.as_str(),
                                config: &option.config,
                                ranked,
                            });
                        }

                        if let Some(prev) = &last_selection {
                            if let Some(prev_at) = last_selection_at {
                                if prev_at.elapsed().as_millis() as u64 <= stickiness_ms
                                    && selection_still_valid(
                                        prev,
                                        &constraints,
                                        &estimate_input,
                                        &peer_models_snapshot,
                                        &base_peers,
                                        &model_options,
                                    )
                                {
                                    continue;
                                }
                            }
                        }

                        match select_best(&candidates, &constraints, &estimate_input) {
                            Ok(result) => {
                                if last_selection.as_ref() != Some(&result.key) {
                                    let node_details = result
                                        .selection
                                        .nodes
                                        .iter()
                                        .map(|id| {
                                            peers_snapshot
                                                .iter()
                                                .find(|p| &p.node_id == id)
                                                .map(|p| {
                                                    format!(
                                                        "{}(role={:?},vram={:.1}GB,rtt={:.0}ms)",
                                                        p.node_id, p.role, p.free_vram_gb, p.rtt_ms
                                                    )
                                                })
                                                .unwrap_or_else(|| id.clone())
                                        })
                                        .collect::<Vec<_>>();
                                    println!(
                                        "cluster selected: model={} nodes={:?} required_vram_per_node_gb={:.2}",
                                        result.key.model_name, node_details, result.selection.required_vram_gb
                                    );
                                    last_selection = Some(result.key);
                                    last_selection_at = Some(Instant::now());
                                    for node in &result.selection.nodes {
                                        *peer_load.entry(node.clone()).or_insert(0.0) += 1.0;
                                    }
                                }
                            }
                            Err(SelectionError::NoCandidates) => {
                                eprintln!("cluster selection failed: no candidate models with eligible peers");
                                last_selection = None;
                                last_selection_at = None;
                            }
                            Err(SelectionError::NoModelFits) => {
                                eprintln!("cluster selection failed: no model fits current peers");
                                last_selection = None;
                                last_selection_at = None;
                            }
                        }
                    }
                }

                if let Some(report) = build_local_report(available_gb, &caps) {
                    let should_log = match &last_local_report {
                        Some(prev) => report.significant_change(prev),
                        None => true,
                    };
                    if should_log {
                        println!(
                            "local memory: available_gb={:.2} advertised_free_gb={:.2} role={:?}",
                            report.available_gb, report.advertised_free_gb, report.role
                        );
                        last_local_report = Some(report);
                    }
                }

                if let Some(store) = &store {
                    let (models_snapshot, _rtt_snapshot) = {
                        let st = state.read().await;
                        (st.models_snapshot(), st.rtt_snapshot())
                    };
                    let persisted = build_persisted_state(peers_snapshot, models_snapshot);
                    if let Err(e) = store.save(&persisted) {
                        eprintln!("state persist error: {e}");
                    }
                }
            }
        }
    }

    recv_task.abort();
    net_router.shutdown().await?;
    Ok(())
}

async fn cmd_infer_http(args: &[String]) -> Result<()> {
    let api_base = get_arg(args, "--api-base").unwrap_or_else(|| "http://127.0.0.1:8000/v1".to_string());
    let model = get_arg(args, "--model").unwrap_or_else(|| "gpt-oss-20b".to_string());
    let prompt = get_arg(args, "--prompt").unwrap_or_else(|| "Hello".to_string());
    let system = get_arg(args, "--system");
    let max_tokens = get_arg(args, "--max-tokens")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(128);
    let temperature = get_arg(args, "--temperature")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.2);

    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys,
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt,
    });

    let api_key = get_arg(args, "--api-key");
    let content = infer_http(
        &api_base,
        api_key,
        model,
        messages,
        max_tokens,
        temperature,
    )
    .await?;
    if content.is_empty() {
        println!("no choices returned");
    } else {
        println!("{content}");
    }
    Ok(())
}

async fn cmd_infer_local(args: &[String]) -> Result<()> {
    let bin = get_arg(args, "--local-bin")
        .ok_or_else(|| anyhow!("--local-bin is required for infer-local"))?;
    let args_template = get_arg(args, "--local-args")
        .ok_or_else(|| anyhow!("--local-args is required for infer-local"))?;
    let model_path = get_arg(args, "--local-model")
        .ok_or_else(|| anyhow!("--local-model is required for infer-local"))?;
    let context = get_arg(args, "--local-context")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(4096);

    let prompt = get_arg(args, "--prompt").unwrap_or_else(|| "Hello".to_string());
    let system = get_arg(args, "--system");
    let max_tokens = get_arg(args, "--max-tokens")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(128);
    let temperature = get_arg(args, "--temperature")
        .and_then(|v| v.parse::<f32>().ok())
        .unwrap_or(0.2);

    let mut messages = Vec::new();
    if let Some(sys) = system {
        messages.push(ChatMessage {
            role: "system".to_string(),
            content: sys,
        });
    }
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: prompt,
    });

    let config = LocalExecConfig {
        bin,
        args_template,
        model_path,
        context,
    };
    let content = infer_local_exec(&config, messages, max_tokens, temperature).await?;
    if content.is_empty() {
        println!("no output returned");
    } else {
        println!("{content}");
    }
    Ok(())
}

async fn cmd_api(args: &[String]) -> Result<()> {
    let bind = get_arg(args, "--bind").unwrap_or_else(|| "127.0.0.1:8000".to_string());
    let infer_base = get_arg(args, "--infer-base");
    let infer_key = get_arg(args, "--infer-key");
    let default_model = get_arg(args, "--default-model");
    let api_auth = get_arg(args, "--api-auth");
    let allow_models = get_args(args, "--allow-model");
    let rate_limit_rps = get_arg(args, "--rate-limit-rps").and_then(|v| v.parse::<f64>().ok());
    let rate_limit_burst = get_arg(args, "--rate-limit-burst").and_then(|v| v.parse::<u32>().ok());
    let local_bin = get_arg(args, "--local-bin");
    let local_args = get_arg(args, "--local-args");
    let local_model = get_arg(args, "--local-model");
    let local_context = get_arg(args, "--local-context")
        .and_then(|v| v.parse::<u32>().ok())
        .unwrap_or(4096);

    let local_exec = match (local_bin, local_args, local_model) {
        (Some(bin), Some(args), Some(model)) => Some(LocalExecConfig {
            bin,
            args_template: args,
            model_path: model,
            context: local_context,
        }),
        _ => None,
    };

    // ── Optional swarm backend ─────────────────────────────────────────────
    let swarm = if let Some(topic_hex) = get_arg(args, "--topic") {
        let topic = parse_topic_id(&topic_hex)?;
        let bootstrap_ids = parse_bootstrap_ids(args)?;

        let discover_secs: u64 = get_arg(args, "--discover-secs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(5);
        let max_retries: usize = get_arg(args, "--max-retries")
            .and_then(|v| v.parse().ok())
            .unwrap_or(2);
        let layers_per_shard: usize = get_arg(args, "--layers-per-shard")
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);

        // Determine total_layers from --total-layers or from --models + --model
        let total_layers: usize = if let Some(n) = get_arg(args, "--total-layers").and_then(|v| v.parse().ok()) {
            n
        } else if let Some(models_path) = get_arg(args, "--models") {
            let entries = load_model_registry(&models_path)?;
            let model_name = get_arg(args, "--swarm-model")
                .or_else(|| get_arg(args, "--model"))
                .unwrap_or_else(|| entries.first().map(|e| e.name.clone()).unwrap_or_default());
            let entry = entries
                .into_iter()
                .find(|e| e.name == model_name)
                .ok_or_else(|| anyhow!("model '{model_name}' not found in registry"))?;
            let cfg = ModelConfig::from_json_file(&entry.config_path, entry.params_b)?;
            cfg.n_layers
        } else {
            return Err(anyhow!("--total-layers or --models is required when using --topic"));
        };

        let swarm_model = get_arg(args, "--swarm-model")
            .or_else(|| get_arg(args, "--model"))
            .ok_or_else(|| anyhow!("--swarm-model is required when using --topic"))?;

        let endpoint = Endpoint::bind().await?;
        let gossip = Gossip::builder().spawn(endpoint.clone());
        let _router = Router::builder(endpoint.clone())
            .accept(iroh_gossip::ALPN, gossip.clone())
            .spawn();

        eprintln!("joining gossip topic {topic_hex}, discovering peers for {discover_secs}s...");
        let registry = Arc::new(RwLock::new(SwarmRegistry::new(StdDuration::from_secs(300))));
        let registry_clone = registry.clone();

        let (_, mut recv) = gossip.subscribe(topic, bootstrap_ids).await?.split();
        let _discovery = tokio::spawn(async move {
            while let Some(Ok(Event::Received(msg))) = recv.next().await {
                let body = String::from_utf8_lossy(&msg.content);
                if let Some(caps) = parse_capability(&body) {
                    let layers: Vec<proximity_router::swarm::LayerRange> = caps
                        .layers_hosted
                        .iter()
                        .cloned()
                        .map(proximity_router::swarm::LayerRange::from)
                        .collect();
                    let metrics = proximity_router::PeerMetrics {
                        rtt_ms: caps.rtt_ms,
                        bandwidth_mbps: caps.bandwidth_mbps,
                        reliability: caps.reliability,
                        gpu_capacity_score: caps.gpu_capacity_score,
                        free_vram_gb: Some(caps.free_vram_gb),
                        last_seen_epoch_ms: None,
                    };
                    registry_clone.write().await.upsert(
                        caps.node_id.clone(), metrics, layers, caps.load_score, caps.uptime_secs,
                    );
                    eprintln!("  discovered peer {}", caps.node_id);
                }
            }
        });

        tokio::time::sleep(Duration::from_secs(discover_secs)).await;

        let peer_count = registry.read().await.peer_count();
        eprintln!("discovered {peer_count} peer(s), starting API server...");

        Some(Arc::new(api::SwarmHandle {
            endpoint,
            registry,
            model_name: swarm_model,
            total_layers,
            layers_per_shard,
            max_retries,
        }))
    } else {
        None
    };

    serve_api(ApiConfig {
        bind,
        infer_base,
        infer_key,
        default_model,
        local_exec,
        api_auth,
        allow_models,
        rate_limit_rps,
        rate_limit_burst,
        swarm,
    })
    .await
}

fn apply_estimate_args(args: &[String], input: &mut EstimateInput) {
    if let Some(v) = get_arg(args, "--quant") {
        if let Ok(parsed) = v.parse::<f64>() {
            input.quant_bits = parsed;
        }
    }
    if let Some(v) = get_arg(args, "--seq") {
        if let Ok(parsed) = v.parse::<usize>() {
            input.seq_len = parsed;
        }
    }
    if let Some(v) = get_arg(args, "--batch") {
        if let Ok(parsed) = v.parse::<usize>() {
            input.batch = parsed;
        }
    }
    if let Some(v) = get_arg(args, "--nodes") {
        if let Ok(parsed) = v.parse::<usize>() {
            input.nodes = parsed;
        }
    }
    if let Some(v) = get_arg(args, "--kv-bytes") {
        if let Ok(parsed) = v.parse::<f64>() {
            input.bytes_per_kv = parsed;
        }
    }
    if let Some(v) = get_arg(args, "--overhead") {
        if let Ok(parsed) = v.parse::<f64>() {
            input.overhead = parsed;
        }
    }
    if let Some(v) = get_arg(args, "--runtime-overhead") {
        if let Ok(parsed) = v.parse::<f64>() {
            input.runtime_overhead_ratio = parsed;
        }
    }
}

fn apply_kv_bits_arg(args: &[String], input: &mut EstimateInput) {
    if let Some(v) = get_arg(args, "--kv-bits") {
        if let Ok(bits) = v.parse::<u8>() {
            input.bytes_per_kv = match bits {
                4 => 0.5,
                8 => 1.0,
                16 => 2.0,
                _ => input.bytes_per_kv,
            };
        }
    }
}

fn load_model_registry(path: &str) -> Result<Vec<ModelRegistryEntry>> {
    let data = fs::read_to_string(path)?;
    let mut entries: Vec<ModelRegistryEntry> = serde_json::from_str(&data)?;
    // Resolve relative config_paths against the models.json directory so the
    // path works regardless of CWD.
    let base = std::path::Path::new(path)
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    for entry in &mut entries {
        if !std::path::Path::new(&entry.config_path).is_absolute() {
            entry.config_path = base
                .join(&entry.config_path)
                .to_string_lossy()
                .into_owned();
        }
    }
    Ok(entries)
}

async fn send_rtt_pings(
    sender: &iroh_gossip::api::GossipSender,
    node_id: &str,
    peer_ids: &[String],
    state: &Arc<RwLock<PeerState>>,
    counter: &Arc<AtomicU64>,
    timeout_ms: u64,
    fail_threshold: u32,
    cooldown_ms: u64,
) {
    {
        let mut st = state.write().await;
        let expired = st.take_expired_pending(StdDuration::from_millis(timeout_ms));
        let now_ms = now_millis();
        for ping in expired {
            let became_unhealthy =
                st.register_failure(&ping.peer_id, now_ms, fail_threshold, cooldown_ms);
            if became_unhealthy {
                eprintln!(
                    "peer {} marked unhealthy for {}ms",
                    ping.peer_id, cooldown_ms
                );
            }
        }
    }

    for peer_id in peer_ids {
        let nonce = format!(
            "{}-{}-{}",
            node_id,
            counter.fetch_add(1, Ordering::Relaxed),
            now_millis()
        );
        {
            let mut st = state.write().await;
            st.add_pending(nonce.clone(), peer_id.clone(), Instant::now());
        }
        let ping = PingMessage {
            from_id: node_id.to_string(),
            target_id: peer_id.clone(),
            nonce,
        };
        let msg = format!("ping:{}", serde_json::to_string(&ping).unwrap_or_default());
        let _ = sender.broadcast(msg.into_bytes().into()).await;
    }
}

fn now_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0)
}

fn peer_supports_model(
    peer_models: &HashMap<String, Vec<ModelSummary>>,
    peer_id: &str,
    model_name: &str,
) -> bool {
    peer_models
        .get(peer_id)
        .map(|models| models.iter().any(|m| m.name == model_name))
        .unwrap_or(false)
}

fn parse_heartbeat(body: &str) -> Option<String> {
    body.strip_prefix("heartbeat from ")
        .map(|id| id.trim().to_string())
}

fn build_local_report(available_gb: Option<f64>, caps: &CapabilityPayload) -> Option<LocalReport> {
    available_gb.map(|available| LocalReport {
        available_gb: available,
        advertised_free_gb: caps.free_vram_gb,
        role: caps.role.clone(),
    })
}

fn build_persisted_state(
    peers: Vec<CapabilityPayload>,
    models: HashMap<String, Vec<ModelSummary>>,
) -> PersistedState {
    let mut model_entries = Vec::new();
    for (node_id, entries) in models {
        model_entries.push(PersistedModels {
            node_id,
            models: entries,
        });
    }
    PersistedState {
        peers,
        models: model_entries,
    }
}

fn selection_still_valid(
    prev: &SelectionKey,
    constraints: &ClusterConstraints,
    estimate_input: &EstimateInput,
    peer_models: &HashMap<String, Vec<ModelSummary>>,
    base_peers: &[PeerInfo],
    model_options: &[ModelOption],
) -> bool {
    let nodes_count = prev.nodes.len();
    if nodes_count < constraints.min_nodes || nodes_count > constraints.max_nodes {
        return false;
    }

    let model_cfg = match model_options.iter().find(|m| m.name == prev.model_name) {
        Some(m) => m,
        None => return false,
    };

    let mut estimate = estimate_input.clone();
    estimate.nodes = nodes_count;
    let required_vram_gb = estimate_required_vram_gb(&model_cfg.config, &estimate);

    for node_id in &prev.nodes {
        let peer = match base_peers.iter().find(|p| &p.id == node_id) {
            Some(p) => p,
            None => return false,
        };

        if let Some(min_bw) = constraints.min_bandwidth_mbps {
            if peer.metrics.bandwidth_mbps < min_bw {
                return false;
            }
        }

        if peer.metrics.free_vram_gb.unwrap_or(0.0) < required_vram_gb {
            return false;
        }

        if !peer_supports_model(peer_models, node_id, &prev.model_name) {
            return false;
        }
    }

    true
}

fn apply_load_penalty(
    ranked: Vec<proximity_router::RankedPeer>,
    peer_load: &HashMap<String, f64>,
    load_penalty: f64,
) -> Vec<proximity_router::RankedPeer> {
    if load_penalty <= 0.0 {
        return ranked;
    }

    ranked
        .into_iter()
        .map(|mut peer| {
            let load = peer_load.get(&peer.id).copied().unwrap_or(0.0);
            peer.score += load * load_penalty;
            peer
        })
        .collect()
}

fn decay_load(peer_load: &mut HashMap<String, f64>, decay: f64) {
    if decay >= 1.0 {
        return;
    }
    peer_load.retain(|_, value| {
        *value *= decay;
        *value > 0.01
    });
}

fn parse_model_option(value: &str) -> Result<ModelOption> {
    let parts: Vec<&str> = value.splitn(3, ':').collect();
    if parts.len() != 3 {
        return Err(anyhow!(
            "expected format name:/path/to/config.json:params_b"
        ));
    }
    let name = parts[0].to_string();
    let path = parts[1];
    let params_b: f64 = parts[2]
        .parse()
        .map_err(|_| anyhow!("invalid params_b"))?;
    let config = ModelConfig::from_json_file(path, params_b)?;
    Ok(ModelOption { name, config })
}

fn detect_available_gb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    let available_bytes = sys.available_memory();
    if available_bytes > 0 {
        return available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    }

    if let Some(available_bytes) = detect_available_bytes_vm_stat() {
        return available_bytes as f64 / (1024.0 * 1024.0 * 1024.0);
    }

    0.0
}

fn detect_available_bytes_vm_stat() -> Option<u64> {
    let output = Command::new("vm_stat").output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let mut page_size: u64 = 16384;
    for line in text.lines() {
        if let Some(pos) = line.find("page size of ") {
            let rest = &line[pos + "page size of ".len()..];
            if let Some(end) = rest.find(" bytes") {
                if let Ok(parsed) = rest[..end].trim().parse::<u64>() {
                    page_size = parsed;
                }
            }
        }
    }

    let mut pages_free = 0u64;
    let mut pages_spec = 0u64;
    let mut pages_inactive = 0u64;

    for line in text.lines() {
        if let Some((key, val)) = line.split_once(':') {
            let value = val.trim().trim_end_matches('.');
            let value = value.replace('.', "");
            let parsed = value.parse::<u64>().ok()?;
            match key.trim() {
                "Pages free" => pages_free = parsed,
                "Pages speculative" => pages_spec = parsed,
                "Pages inactive" => pages_inactive = parsed,
                _ => {}
            }
        }
    }

    let available_pages = pages_free + pages_spec + pages_inactive;
    Some(available_pages * page_size)
}

fn parse_role(value: &str) -> Option<NodeRole> {
    match value.to_ascii_lowercase().as_str() {
        "inference" => Some(NodeRole::Inference),
        "routing" => Some(NodeRole::Routing),
        "cache" => Some(NodeRole::Cache),
        _ => None,
    }
}

fn get_arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|v| v == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|v| v == key)
}

fn get_args(args: &[String], key: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == key {
            if let Some(value) = args.get(i + 1) {
                values.push(value.clone());
            }
            i += 2;
        } else {
            i += 1;
        }
    }
    values
}

fn parse_topic_id(hex_str: &str) -> Result<TopicId> {
    let raw = hex::decode(hex_str)
        .map_err(|_| anyhow!("topic must be 64 hex chars (32 bytes)"))?;
    if raw.len() != 32 {
        return Err(anyhow!("topic must be 32 bytes"));
    }
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&raw);
    Ok(TopicId::from_bytes(bytes))
}

fn parse_bootstrap_ids(args: &[String]) -> Result<Vec<EndpointId>> {
    let values = get_args(args, "--bootstrap");
    let mut ids = Vec::new();
    for v in values {
        let id = EndpointId::from_str(&v)
            .map_err(|_| anyhow!("invalid bootstrap endpoint id: {v}"))?;
        ids.push(id);
    }
    Ok(ids)
}

fn peer(id: &str, rtt_ms: f64, bw: f64, rel: f64, gpu: f64, vram: f64) -> PeerInfo {
    PeerInfo {
        id: id.to_string(),
        metrics: PeerMetrics {
            rtt_ms,
            bandwidth_mbps: bw,
            reliability: rel,
            gpu_capacity_score: gpu,
            free_vram_gb: Some(vram),
            last_seen_epoch_ms: None,
        },
    }
}

#[derive(Debug, Deserialize)]
struct PeerJson {
    id: String,
    rtt_ms: f64,
    bandwidth_mbps: f64,
    reliability: f64,
    gpu_capacity_score: f64,
    free_vram_gb: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct RttSampleJson {
    id: String,
    rtt_ms: f64,
}

fn load_peers_from_json(path: &str) -> Result<Vec<PeerInfo>, String> {
    let data = fs::read_to_string(path).map_err(|e| format!("io error: {e}"))?;
    let peers: Vec<PeerJson> =
        serde_json::from_str(&data).map_err(|e| format!("json error: {e}"))?;
    Ok(peers
        .into_iter()
        .map(|p| PeerInfo {
            id: p.id,
            metrics: PeerMetrics {
                rtt_ms: p.rtt_ms,
                bandwidth_mbps: p.bandwidth_mbps,
                reliability: p.reliability,
                gpu_capacity_score: p.gpu_capacity_score,
                free_vram_gb: p.free_vram_gb,
                last_seen_epoch_ms: None,
            },
        })
        .collect())
}

fn load_rtt_samples(path: &str) -> Result<HashMap<PeerId, f64>, String> {
    let data = fs::read_to_string(path).map_err(|e| format!("io error: {e}"))?;
    let samples: Vec<RttSampleJson> =
        serde_json::from_str(&data).map_err(|e| format!("json error: {e}"))?;
    Ok(samples
        .into_iter()
        .map(|s| (s.id, s.rtt_ms))
        .collect())
}

/// Process start time — initialised on first call, used for uptime reporting.
static PROCESS_START: OnceLock<Instant> = OnceLock::new();

fn process_uptime_secs() -> u64 {
    PROCESS_START.get_or_init(Instant::now).elapsed().as_secs()
}

fn capability_from_args(args: &[String], node_id: String) -> (CapabilityPayload, Option<f64>) {
    let reserve_gb = get_arg(args, "--reserve-gb")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(4.0);
    let safety_factor = get_arg(args, "--safety-factor")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    let mut available_gb: Option<f64> = None;
    let free_vram_gb = match get_arg(args, "--free-vram")
        .and_then(|v| v.parse::<f64>().ok())
    {
        Some(v) => v,
        None => {
            let available = detect_available_gb();
            available_gb = Some(available);
            let safe = (available - reserve_gb).max(0.0) * safety_factor;
            if safe.is_nan() { 0.0 } else { safe }
        }
    };
    let bandwidth_mbps = get_arg(args, "--bandwidth")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(500.0);
    let reliability = get_arg(args, "--reliability")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.95);
    let gpu_capacity_score = get_arg(args, "--gpu-score")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.7);
    let rtt_ms = get_arg(args, "--rtt-ms")
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(20.0);
    let kv_bits = get_arg(args, "--kv-bits")
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(8);
    let context = get_arg(args, "--context")
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(8192);
    let quant_bits = get_arg(args, "--quant")
        .and_then(|v| v.parse::<u8>().ok())
        .unwrap_or(4);
    let role = get_arg(args, "--role")
        .and_then(|v| parse_role(&v))
        .unwrap_or_else(|| {
            let min_inference_gb = get_arg(args, "--min-inference-gb")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(6.0);
            if free_vram_gb >= min_inference_gb {
                NodeRole::Inference
            } else {
                NodeRole::Routing
            }
        });

    // Build the hosted-layer-range list from optional CLI args.
    // Operators pass: --layer-start N --layer-end M [--total-layers T] [--model-name NAME]
    let layers_hosted = match (
        get_arg(args, "--layer-start").and_then(|v| v.parse::<usize>().ok()),
        get_arg(args, "--layer-end").and_then(|v| v.parse::<usize>().ok()),
    ) {
        (Some(ls), Some(le)) => {
            let total = get_arg(args, "--total-layers")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(le);
            let model = get_arg(args, "--model-name")
                .or_else(|| get_arg(args, "--model"))
                .unwrap_or_else(|| "default".to_string());
            vec![HostedLayerRange { model, layer_start: ls, layer_end: le, total_layers: total }]
        }
        _ => Vec::new(),
    };

    let payload = CapabilityPayload {
        node_id,
        free_vram_gb,
        bandwidth_mbps,
        reliability,
        gpu_capacity_score,
        rtt_ms,
        kv_bits,
        context,
        quant_bits,
        role,
        layers_hosted,
        load_score: 0.0,
        uptime_secs: process_uptime_secs(),
    };
    (payload, available_gb)
}

struct StaticRttProbe {
    samples: HashMap<PeerId, f64>,
}

impl StaticRttProbe {
    fn new(samples: HashMap<PeerId, f64>) -> Self {
        Self { samples }
    }
}

impl RttProbe for StaticRttProbe {
    fn measure_rtt_ms(&self, peer: &PeerId) -> Option<f64> {
        self.samples.get(peer).copied()
    }
}

fn apply_rtt_probe(peers: &mut [PeerInfo], probe: &dyn RttProbe) {
    for peer in peers {
        if let Some(rtt) = probe.measure_rtt_ms(&peer.id) {
            peer.metrics.rtt_ms = rtt;
        }
    }
}

fn default_peers() -> Vec<PeerInfo> {
    vec![
        peer("node-a", 12.0, 1200.0, 0.98, 0.9, 80.0),
        peer("node-b", 18.0, 800.0, 0.96, 0.8, 48.0),
        peer("node-c", 25.0, 600.0, 0.92, 0.7, 24.0),
        peer("node-d", 35.0, 400.0, 0.90, 0.6, 24.0),
        peer("node-e", 55.0, 300.0, 0.88, 0.5, 16.0),
        peer("node-f", 75.0, 200.0, 0.80, 0.4, 12.0),
    ]
}

/// Start an iroh QUIC inference server that accepts tensor-pipeline requests.
///
/// Prints the node's iroh `node_id` on startup so you can pass it to
/// `dist-run --executor iroh --peers <node_id>`.
///
/// With `--model-path` (and `--features burn`), loads a real `LlamaShard` from
/// a safetensors checkpoint and serves that shard's layers.  Without a model
/// path, serves a `MockInferenceServer` (useful for end-to-end connectivity
/// tests before loading real weights).
async fn cmd_serve(args: &[String]) -> Result<()> {
    let kv_ttl_secs: u64 = get_arg(args, "--kv-ttl-secs")
        .and_then(|v| v.parse().ok())
        .unwrap_or(300);
    #[cfg(feature = "burn")]
    let kv_ttl = kv_ttl_secs;
    #[cfg(not(feature = "burn"))]
    let _ = kv_ttl_secs;

    // ── Gossip args (optional — enables swarm discoverability) ───────────────
    let topic_hex    = get_arg(args, "--topic");
    let bootstrap_ids = parse_bootstrap_ids(args)?;
    let announce_secs: u64 = get_arg(args, "--interval")
        .and_then(|v| v.parse().ok())
        .unwrap_or(10);

    let endpoint = Endpoint::bind().await?;
    let node_id   = endpoint.id();
    println!("node_id:  {node_id}");
    println!("hint:     mesh-node dist-run --executor iroh --peers {node_id} ...");

    // Build the gossip handler BEFORE the router builder consumes the endpoint.
    // If no --topic is given this stays None and the serve node is reachable only
    // via its NodeId (manually passed to --peers).
    let gossip_opt = if topic_hex.is_some() {
        Some(Gossip::builder().spawn(endpoint.clone()))
    } else {
        None
    };

    // ── Real shard (burn feature only) ───────────────────────────────────────
    #[cfg(feature = "burn")]
    {
        let model_path   = get_arg(args, "--model-path");
        let store_path   = get_arg(args, "--store-path");
        let shard_hash   = get_arg(args, "--shard-hash");

        if model_path.is_some() || (store_path.is_some() && shard_hash.is_some()) {
            use inference_coordinator::LlamaShardConfig;

            let layer_start = get_arg(args, "--layer-start")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0);
            let layer_end = get_arg(args, "--layer-end")
                .and_then(|v| v.parse::<usize>().ok())
                .ok_or_else(|| anyhow!("--layer-end is required when loading a shard"))?;
            let total_layers = get_arg(args, "--total-layers")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(layer_end);
            let num_attention_heads = get_arg(args, "--heads")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(32);
            let num_key_value_heads = get_arg(args, "--kv-heads")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(num_attention_heads);
            let hidden_size = get_arg(args, "--hidden")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4096);
            let intermediate_size = get_arg(args, "--intermediate")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(11008);
            let vocab_size = get_arg(args, "--vocab")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(32000);
            let rope_theta = get_arg(args, "--rope-theta")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(10000.0);
            let rms_norm_eps = get_arg(args, "--rms-eps")
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(1e-5);
            let max_seq_len = get_arg(args, "--max-seq-len")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4096);

            let shard_config = LlamaShardConfig {
                layer_start, layer_end, total_layers,
                num_attention_heads, num_key_value_heads,
                hidden_size, intermediate_size, vocab_size,
                rope_theta, rms_norm_eps, max_seq_len,
            };

            let server = if let (Some(sp), Some(hh)) = (store_path, shard_hash) {
                // ── Load from pre-cached ShardStore ──────────────────────────
                println!("loading shard layers {}..{} from store {} hash {} ...",
                    layer_start, layer_end, sp, hh);
                let store = shard_store::ShardStore::open(&sp)?;
                let hash  = shard_store::Hash::from_hex(&hh)?;
                InferenceServer::new_from_store(&store, &hash, shard_config, kv_ttl)?
            } else {
                // ── Load from disk; optionally populate ShardStore ───────────
                let mp = model_path.unwrap();
                let weight_files: Vec<String> = mp
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect();
                println!("loading shard layers {}..{} from {:?} ...",
                    layer_start, layer_end, weight_files);

                // Read all weight files into memory once.
                let bytes_vec: Vec<Vec<u8>> = weight_files
                    .iter()
                    .map(|p| std::fs::read(p))
                    .collect::<std::io::Result<_>>()?;

                // If --store-path is given without --shard-hash, populate the
                // ShardStore so the next startup can use --shard-hash instead of
                // reading from disk.  Prints the hash for each file.
                if let Some(sp) = get_arg(args, "--store-path") {
                    let store = shard_store::ShardStore::open(&sp)?;
                    for (i, file_bytes) in bytes_vec.iter().enumerate() {
                        let hash = store.add_shard(
                            file_bytes,
                            &format!("layers_{layer_start}_{layer_end}"),
                            layer_start as u32,
                            layer_end as u32,
                        )?;
                        println!(
                            "  [shard cache] file[{i}] → store={sp} hash={hash}"
                        );
                        println!(
                            "  hint: next time use: --store-path {sp} --shard-hash {hash}"
                        );
                    }
                }

                // Load the shard from in-memory bytes (no second disk read).
                use inference_coordinator::llm_shard::LlamaShard as LS;
                let shard = LS::load_from_bytes_multi_wgpu(bytes_vec, shard_config)?;
                InferenceServer::new(shard, kv_ttl)
            };
            println!("shard loaded — serving on {node_id}");

            let mut rb = Router::builder(endpoint).accept(INFERENCE_ALPN, server);
            if let Some(g) = &gossip_opt {
                rb = rb.accept(iroh_gossip::ALPN, g.clone());
            }
            let _router = rb.spawn();

            return run_gossip_announce_loop(
                args, node_id.to_string(), topic_hex, bootstrap_ids, gossip_opt, announce_secs,
            ).await;
        }
    }

    // ── Mock server (no weights / burn feature disabled) ─────────────────────
    let server = MockInferenceServer::new();
    println!("no --model-path provided — serving mock inference");
    let mut rb = Router::builder(endpoint).accept(INFERENCE_ALPN, server);
    if let Some(g) = &gossip_opt {
        rb = rb.accept(iroh_gossip::ALPN, g.clone());
    }
    let _router = rb.spawn();

    run_gossip_announce_loop(
        args, node_id.to_string(), topic_hex, bootstrap_ids, gossip_opt, announce_secs,
    ).await
}

/// Joins a gossip topic (if `topic_hex` is Some) and broadcasts a `CapabilityPayload`
/// every `interval_secs` seconds until Ctrl-C.
///
/// When no topic is provided the function just waits for Ctrl-C so `cmd_serve` has a
/// single unified exit point.
async fn run_gossip_announce_loop(
    args: &[String],
    node_id: String,
    topic_hex: Option<String>,
    bootstrap_ids: Vec<EndpointId>,
    gossip_opt: Option<Gossip>,
    interval_secs: u64,
) -> Result<()> {
    // No gossip — just wait for Ctrl-C.
    let (topic_hex, gossip) = match (topic_hex, gossip_opt) {
        (Some(th), Some(g)) => (th, g),
        _ => {
            println!("tip: pass --topic <64-hex> --bootstrap <id> to join gossip");
            println!("     and be discoverable by --executor swarm");
            println!("press Ctrl-C to stop");
            tokio::signal::ctrl_c().await.ok();
            return Ok(());
        }
    };

    let topic = parse_topic_id(&topic_hex)?;
    let (sender, _) = gossip.subscribe(topic, bootstrap_ids).await?.split();

    // Build capability payload once (static for this process lifetime).
    let (caps, _) = capability_from_args(args, node_id);
    let init_msg = format!("cap:{}", serde_json::to_string(&caps)?);
    if let Err(e) = sender.broadcast(init_msg.into_bytes().into()).await {
        eprintln!("initial gossip broadcast error: {e}");
    }
    println!("announcing on gossip topic {topic_hex} every {interval_secs}s");
    println!("press Ctrl-C to stop");

    // Use a pinned Ctrl-C future so a signal is never missed between ticks.
    let mut tick = tokio::time::interval(Duration::from_secs(interval_secs));
    tick.tick().await; // skip the immediate first tick (already broadcast above)
    let ctrl_c = tokio::signal::ctrl_c();
    tokio::pin!(ctrl_c);

    loop {
        tokio::select! {
            _ = &mut ctrl_c => break,
            _ = tick.tick() => {
                let m = format!("cap:{}", serde_json::to_string(&caps).unwrap_or_default());
                if let Err(e) = sender.broadcast(m.into_bytes().into()).await {
                    eprintln!("gossip broadcast error: {e}");
                }
            }
        }
    }

    Ok(())
}

fn print_usage() {
    println!("mesh-node estimate --config <path> [--params-b 117] [--quant 4] [--seq 8192] [--batch 1] [--nodes 4] [--kv-bytes 1] [--overhead 1.15] [--runtime-overhead 0.15]");
    println!("mesh-node capacity --free-vram 8 [--max-nodes 8] [--current-nodes 2] [--seq 4096] [--quant 4] [--kv-bytes 1] [--models models.json --model stablelm-3b-4e1t] OR [--config model.json --params-b 2.8]");
    println!("mesh-node dist-plan --peers id1,id2 [--min-layers-per-peer 1] [--models models.json --model stablelm-3b-4e1t] OR [--config model.json --params-b 2.8]");
    println!("mesh-node dist-run --peers id1,id2 --models models.json --model stablelm-3b-4e1t [--input \"Hello\"] [--executor echo|burn|iroh|swarm] [--max-retries 2]");
    println!("  iroh executor:  --peers id1,id2 --max-retries 2");
    println!("  swarm executor: --topic <64-hex> --bootstrap <endpoint_id> [--discover-secs 5] [--layers-per-shard 16] [--max-retries 2]");
    println!("mesh-node serve [--model-path /path.safetensors,/path2.safetensors --store-path ./cache  (loads from disk; adds to ShardStore)] [--store-path ./cache --shard-hash <hex>  (loads from cache)] [--layer-start 0 --layer-end 16 --total-layers 32 --hidden 4096 --intermediate 11008 --vocab 32000 --heads 32 --kv-heads 32 --rope-theta 10000] [--kv-ttl-secs 300]");
    println!("mesh-node simulate --config <path> [--params-b 117] [--quant 4] [--seq 8192] [--batch 1] [--min-nodes 2] [--max-nodes 6] [--min-bw 500] [--peers peers.json] [--rtt-samples rtt.json]");
    println!("mesh-node gossip --topic <64-hex> [--bootstrap <endpoint_id> ...] [--interval 10] [--rtt-probe] [--rtt-timeout-ms 3000] [--peer-ttl-ms 15000] [--peer-fail-threshold 3] [--peer-cooldown-ms 30000] [--cluster-stickiness-ms 15000] [--load-penalty 0.05] [--load-decay 0.9] [--config config.json] [--params-b 117] [--model-name primary] [--models models.json] [--fallback-model name:/path/to/config.json:params_b] [--min-nodes 2] [--max-nodes 6] [--min-bw 500] [--free-vram 48] [--reserve-gb 4] [--safety-factor 0.7] [--min-inference-gb 6] [--role inference|routing|cache] [--state-dir state] [--bandwidth 800] [--reliability 0.95] [--gpu-score 0.8] [--rtt-ms 20] [--kv-bits 8] [--context 8192] [--quant 4]");
    println!("mesh-node infer-http --api-base http://127.0.0.1:8000/v1 --model gpt-oss-20b --prompt \"Hello\" [--system \"You are ...\"] [--max-tokens 128] [--temperature 0.2] [--api-key sk-...]");
    println!("mesh-node infer-local --local-bin /path/to/llama-cli --local-args \"-m {{model}} -p {{prompt}} -n {{max_tokens}} --temp {{temperature}} --ctx-size {{context}}\" --local-model /path/to/model.gguf --prompt \"Hello\"");
    println!("mesh-node api --bind 127.0.0.1:8000 [--infer-base http://127.0.0.1:8001/v1] [--infer-key sk-...] [--default-model gpt-oss-20b] [--local-bin /path/to/llama-cli --local-args \"...\" --local-model /path/to/model.gguf] [--api-auth KEY] [--allow-model name] [--rate-limit-rps 5] [--rate-limit-burst 10]");
}
