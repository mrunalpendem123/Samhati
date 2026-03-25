mod app;
mod api;
mod events;
pub mod identity;
mod model_download;
pub mod network;
pub mod registry;
mod node_runner;
mod swarm;
mod tabs;
mod ui;
mod wallet;

use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use app::{App, ChatMessage, SwarmNodeDisplay, SwarmRoundDisplay};
use api::ApiClient;
use events::{ChatAction, handle_key, poll_event};
use identity::NodeIdentity;
use model_download::ModelDownloader;
use network::{NetworkHandle, NodeAnnouncement};
use node_runner::{NodeRunner, MultiNodeRunner};
use swarm::SwarmOrchestrator;
use wallet::SolanaWallet;

fn main() -> Result<()> {
    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let rt = tokio::runtime::Runtime::new()?;

    let mut app = App::new();

    // Load unified identity — one Ed25519 key for Solana + iroh + TOPLOC
    let mut identity_secret: Option<[u8; 32]> = None;
    match NodeIdentity::load_or_create() {
        Ok(id) => {
            app.wallet_pubkey = id.solana_pubkey.clone();
            app.wallet_short = id.short_id.clone();
            app.node_id = id.iroh_node_id.clone();
            identity_secret = Some(id.secret_key);

            // Create Solana wallet from the same identity
            let w = SolanaWallet {
                pubkey: id.solana_pubkey.clone(),
                keypair_path: id.path.clone(),
                secret_bytes: {
                    let mut bytes = Vec::with_capacity(64);
                    bytes.extend_from_slice(&id.secret_key);
                    bytes.extend_from_slice(&id.public_key);
                    bytes
                },
            };
            app.wallet = Some(w);
        }
        Err(e) => {
            app.wallet_pubkey = format!("Error: {}", e);
            app.wallet_short = "no wallet".into();
            app.node_id = String::new();
        }
    }

    let downloader = ModelDownloader::new();
    let mut node_runner = NodeRunner::new();
    let mut multi_runner = MultiNodeRunner::new();
    let swarm = Arc::new(SwarmOrchestrator::new());

    // Start P2P network — auto-discover other Samhati nodes
    let mut net_handle: Option<NetworkHandle> = None;
    if let Some(secret) = identity_secret {
        let pubkey = app.wallet_pubkey.clone();

        rt.block_on(async {
            // 1. Check if we're registered on Solana
            match registry::is_registered(&pubkey).await {
                Ok(false) => {
                    eprintln!("[registry] Not registered — registering on Solana...");
                    let mut secret_arr = [0u8; 32];
                    secret_arr.copy_from_slice(&secret[..32]);
                    let mut public_arr = [0u8; 32];
                    if let Ok(bytes) = bs58::decode(&pubkey).into_vec() {
                        let len = bytes.len().min(32);
                        public_arr[..len].copy_from_slice(&bytes[..len]);
                    }
                    match registry::register_node(&secret_arr, &public_arr, "samhati-node").await {
                        Ok(sig) => eprintln!("[registry] Registered! Tx: {}", &sig[..20.min(sig.len())]),
                        Err(e) => eprintln!("[registry] Registration failed: {} (continuing anyway)", e),
                    }
                }
                Ok(true) => eprintln!("[registry] Already registered on Solana"),
                Err(e) => eprintln!("[registry] Check failed: {} (continuing anyway)", e),
            }
        });

        // 2. Fetch all registered nodes (free RPC read)
        let bootstrap_nodes = rt.block_on(async {
            match registry::fetch_all_nodes().await {
                Ok(nodes) => {
                    eprintln!("[registry] Found {} nodes on Solana", nodes.len());
                    nodes
                }
                Err(e) => {
                    eprintln!("[registry] Fetch failed: {} (no bootstrap)", e);
                    vec![]
                }
            }
        });

        let bootstrap_count = bootstrap_nodes.len();

        // 3. Start iroh + gossip, bootstrap with on-chain nodes
        match NetworkHandle::start(secret) {
            Ok(nh) => {
                app.node_id = nh.node_id.clone();

                for node in &bootstrap_nodes {
                    if node.iroh_node_id != nh.node_id {
                        nh.add_bootstrap(node.iroh_node_id.clone());
                    }
                }

                if bootstrap_count > 1 {
                    app.download_status = format!(
                        "Connected to mesh — {} nodes from Solana",
                        bootstrap_count,
                    );
                } else {
                    app.download_status = "Registered on Solana — waiting for peers".into();
                }

                app.peers_connected = bootstrap_count.saturating_sub(1) as u32;
                net_handle = Some(nh);
            }
            Err(e) => {
                app.node_error = format!("P2P network failed: {}", e);
            }
        }
    }

    let tick_rate = Duration::from_millis(33);

    // Pending async tasks
    let mut pending_chat: Option<tokio::task::JoinHandle<Result<api::ChatResponse>>> = None;
    let mut pending_swarm: Option<tokio::task::JoinHandle<Result<swarm::SwarmRoundResult>>> = None;
    let mut pending_balance: Option<tokio::task::JoinHandle<Result<f64>>> = None;
    let mut pending_txs: Option<tokio::task::JoinHandle<Result<Vec<wallet::TxInfo>>>> = None;
    let mut pending_health: Option<tokio::task::JoinHandle<Result<bool>>> = None;
    let mut pending_airdrop: Option<tokio::task::JoinHandle<Result<String>>> = None;
    let mut pending_download: Option<tokio::task::JoinHandle<Result<PathBuf>>> = None;
    let download_progress_shared: Arc<Mutex<Option<f64>>> = Arc::new(Mutex::new(None));
    let mut download_model_idx: Option<usize> = None;
    let mut last_refresh = Instant::now().checked_sub(Duration::from_secs(30)).unwrap_or_else(Instant::now);

    while app.running {
        // Draw
        terminal.draw(|frame| ui::draw(frame, &app))?;

        // Check for newly discovered P2P peers and auto-add to swarm
        if let Some(ref nh) = net_handle {
            while let Ok(peer) = nh.peer_rx.try_recv() {
                // Auto-add discovered peer to swarm orchestrator
                let node_id = format!("p2p-{}",
                    peer.iroh_node_id.get(..8).unwrap_or(&peer.iroh_node_id));
                swarm.add_node(
                    node_id.clone(),
                    peer.inference_url.clone(),
                    peer.model_name.clone(),
                );
                app.swarm_nodes.push(SwarmNodeDisplay {
                    id: node_id.clone(),
                    url: peer.inference_url.clone(),
                    model: peer.model_name.clone(),
                    elo: 1500,
                    rounds: 0,
                    wins: 0,
                });
                app.peers_connected = swarm.node_count() as u32;
                app.download_status = format!(
                    "Discovered peer {} running {} ({} nodes)",
                    peer.iroh_node_id.get(..12).unwrap_or(&peer.iroh_node_id),
                    peer.model_name,
                    swarm.node_count(),
                );
            }
        }

        // Periodic refresh: check balance, health, txs every 15 seconds
        if last_refresh.elapsed() > Duration::from_secs(15) {
            last_refresh = Instant::now();

            // Check node health
            if pending_health.is_none() {
                let client = ApiClient::new(&app.api_endpoint);
                pending_health = Some(rt.spawn(async move { client.health().await }));
            }

            // Check SOL balance
            if pending_balance.is_none() {
                if let Some(ref w) = app.wallet {
                    let pubkey = w.pubkey.clone();
                    pending_balance = Some(rt.spawn(async move {
                        let w = SolanaWallet { pubkey, keypair_path: std::path::PathBuf::new(), secret_bytes: vec![] };
                        w.get_sol_balance().await
                    }));
                }
            }

            // Get recent transactions
            if pending_txs.is_none() {
                if let Some(ref w) = app.wallet {
                    let pubkey = w.pubkey.clone();
                    pending_txs = Some(rt.spawn(async move {
                        let w = SolanaWallet { pubkey, keypair_path: std::path::PathBuf::new(), secret_bytes: vec![] };
                        w.get_recent_transactions().await
                    }));
                }
            }
        }

        // Check completed async tasks
        check_balance(&rt, &mut pending_balance, &mut app);
        check_health(&rt, &mut pending_health, &mut app);
        check_txs(&rt, &mut pending_txs, &mut app);
        check_chat(&rt, &mut pending_chat, &mut app);
        check_swarm(&rt, &mut pending_swarm, &mut app);
        check_airdrop(&rt, &mut pending_airdrop, &mut app);

        // Update download progress from shared state
        if let Ok(guard) = download_progress_shared.lock() {
            if let Some(pct) = *guard {
                app.download_progress = Some(pct);
                if let Some(idx) = download_model_idx {
                    let name = &app.models[idx].name;
                    app.download_status = format!("Downloading {}... {:.0}%", name, pct);
                }
            }
        }

        // Check if download completed
        check_download(
            &rt,
            &mut pending_download,
            &mut download_model_idx,
            &download_progress_shared,
            &mut app,
            &mut node_runner,
            &net_handle,
        );

        // Poll events
        if let Some(event) = poll_event(tick_rate)? {
            if let Event::Key(key) = event {
                if key.kind == KeyEventKind::Press {
                    if let Some(action) = handle_key(&mut app, key) {
                        match action {
                            ChatAction::SendMessage(msg) => {
                                if swarm.node_count() > 0 {
                                    // Use swarm inference (multi-node BradleyTerry)
                                    let s = Arc::clone(&swarm);
                                    pending_swarm = Some(rt.spawn(async move {
                                        s.infer(&msg).await
                                    }));
                                } else {
                                    // Fallback to single node
                                    let client = ApiClient::new(&app.api_endpoint);
                                    let model = app.current_model.clone();
                                    pending_chat = Some(rt.spawn(async move {
                                        client.chat(&msg, &model).await
                                    }));
                                }
                            }
                            ChatAction::RequestAirdrop => {
                                if let Some(ref w) = app.wallet {
                                    let pubkey = w.pubkey.clone();
                                    let kp = w.keypair_path.clone();
                                    pending_airdrop = Some(rt.spawn(async move {
                                        let w = SolanaWallet { pubkey, keypair_path: kp, secret_bytes: vec![] };
                                        w.request_airdrop(1.0).await
                                    }));
                                    app.wallet_status = "Requesting airdrop...".into();
                                }
                            }
                            ChatAction::SelectModel(idx) => {
                                if pending_download.is_none() {
                                    let model_name = app.models[idx].name.clone();

                                    if downloader.is_downloaded(&model_name) {
                                        // Already downloaded — just start the node
                                        let path = downloader.model_path(&model_name);
                                        activate_model(&mut app, &mut node_runner, idx, &path, &net_handle);
                                    } else {
                                        // Kick off async download
                                        download_model_idx = Some(idx);
                                        let progress = Arc::clone(&download_progress_shared);
                                        if let Ok(mut g) = progress.lock() {
                                            *g = Some(0.0);
                                        }
                                        app.download_progress = Some(0.0);
                                        app.download_status = format!("Starting download of {}...", model_name);
                                        app.node_error.clear();

                                        let dl = ModelDownloader::new();
                                        let name = model_name.clone();
                                        let prog = Arc::clone(&download_progress_shared);
                                        pending_download = Some(rt.spawn(async move {
                                            dl.download(&name, move |pct| {
                                                if let Ok(mut g) = prog.lock() {
                                                    *g = Some(pct);
                                                }
                                            })
                                            .await
                                        }));
                                    }
                                }
                            }
                            ChatAction::AddSwarmNode(idx) => {
                                let model_name = app.models[idx].name.clone();
                                if downloader.is_downloaded(&model_name) {
                                    let path = downloader.model_path(&model_name);
                                    match multi_runner.start_node(&model_name, &path) {
                                        Ok(port) => {
                                            let node_id = format!("node-{}", port);
                                            let url = format!("http://127.0.0.1:{}", port);
                                            swarm.add_node(node_id.clone(), url.clone(), model_name.clone());
                                            app.swarm_nodes.push(SwarmNodeDisplay {
                                                id: node_id,
                                                url,
                                                model: model_name.clone(),
                                                elo: 1500,
                                                rounds: 0,
                                                wins: 0,
                                            });
                                            app.peers_connected = swarm.node_count() as u32;
                                            app.download_status = format!(
                                                "{} added to swarm on port {} ({} nodes total)",
                                                model_name, port, swarm.node_count()
                                            );
                                            app.node_error.clear();
                                        }
                                        Err(e) => {
                                            app.node_error = format!("Swarm node failed: {}", e);
                                        }
                                    }
                                } else {
                                    app.node_error = format!(
                                        "Download {} first (Enter) before adding to swarm (s)",
                                        model_name
                                    );
                                }
                            }
                            ChatAction::AddRemoteNode(url) => {
                                // Add a friend's remote llama-server by direct URL
                                let node_id = format!("remote-{}", swarm.node_count() + 1);
                                swarm.add_node(node_id.clone(), url.clone(), "remote".into());
                                app.swarm_nodes.push(SwarmNodeDisplay {
                                    id: node_id,
                                    url: url.clone(),
                                    model: "remote".into(),
                                    elo: 1500,
                                    rounds: 0,
                                    wins: 0,
                                });
                                app.peers_connected = swarm.node_count() as u32;
                                app.download_status = format!(
                                    "Remote node added: {} ({} nodes total)",
                                    url, swarm.node_count()
                                );
                                app.node_error.clear();
                            }
                            ChatAction::ConnectPeer(peer_node_id) => {
                                // Connect to a friend via iroh P2P (by NodeId)
                                if let Some(ref nh) = net_handle {
                                    nh.add_bootstrap(peer_node_id.clone());
                                    app.download_status = format!(
                                        "Connecting to peer {}... (will auto-add when they announce)",
                                        peer_node_id.get(..16).unwrap_or(&peer_node_id),
                                    );
                                    app.node_error.clear();
                                } else {
                                    app.node_error = "P2P network not running".into();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Stop all node runners before exiting
    node_runner.stop();
    multi_runner.stop_all();

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}

/// Activate a model: update app state, start inference server, announce to P2P network.
fn activate_model(
    app: &mut App,
    node_runner: &mut NodeRunner,
    idx: usize,
    path: &PathBuf,
    net_handle: &Option<NetworkHandle>,
) {
    let model_name = app.models[idx].name.clone();

    // Update model list state
    app.models[idx].installed = true;
    for m in app.models.iter_mut() {
        m.active = false;
    }
    app.models[idx].active = true;
    app.current_model = model_name.clone();

    // Start the inference server
    match node_runner.start(&model_name, path) {
        Ok(()) => {
            app.node_running = true;
            app.node_error.clear();
            app.download_status = format!("{} is running on port {}", model_name, node_runner.port);

            // Announce to P2P network so other nodes can discover us
            if let Some(ref nh) = net_handle {
                nh.set_announcement(NodeAnnouncement {
                    solana_pubkey: app.wallet_pubkey.clone(),
                    iroh_node_id: app.node_id.clone(),
                    inference_url: format!("http://127.0.0.1:{}", node_runner.port),
                    model_name: model_name.clone(),
                    port: node_runner.port,
                });
            }
        }
        Err(e) => {
            app.node_running = false;
            app.node_error = e.to_string();
            app.download_status = format!("Failed to start {}", model_name);
        }
    }
}

fn check_download(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<PathBuf>>>,
    download_model_idx: &mut Option<usize>,
    progress_shared: &Arc<Mutex<Option<f64>>>,
    app: &mut App,
    node_runner: &mut NodeRunner,
    net_handle: &Option<NetworkHandle>,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            let idx = download_model_idx.take();

            // Clear shared progress
            if let Ok(mut g) = progress_shared.lock() {
                *g = None;
            }
            app.download_progress = None;

            match rt.block_on(handle) {
                Ok(Ok(path)) => {
                    if let Some(idx) = idx {
                        activate_model(app, node_runner, idx, &path, net_handle);
                    }
                }
                Ok(Err(e)) => {
                    app.node_error = format!("Download failed: {}", e);
                    app.download_status = "Download failed".into();
                }
                Err(e) => {
                    app.node_error = format!("Download task error: {}", e);
                    app.download_status = "Download failed".into();
                }
            }
        }
    }
}

fn check_balance(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<f64>>>,
    app: &mut App,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            if let Ok(Ok(balance)) = rt.block_on(handle) {
                app.sol_balance = balance;
            }
        }
    }
}

fn check_health(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<bool>>>,
    app: &mut App,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            if let Ok(Ok(healthy)) = rt.block_on(handle) {
                app.node_running = healthy;
            } else {
                app.node_running = false;
            }
        }
    }
}

fn check_txs(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<Vec<wallet::TxInfo>>>>,
    app: &mut App,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            if let Ok(Ok(txs)) = rt.block_on(handle) {
                app.tx_history = txs
                    .into_iter()
                    .map(|tx| {
                        let timestamp = if tx.block_time > 0 {
                            chrono::DateTime::from_timestamp(tx.block_time, 0)
                                .map(|dt| dt.format("%Y-%m-%d %H:%M").to_string())
                                .unwrap_or_else(|| "unknown".into())
                        } else {
                            "pending".into()
                        };
                        app::TxEntry {
                            timestamp,
                            tx_type: if tx.success { "confirmed".into() } else { "failed".into() },
                            amount: 0.0, // can't determine from signature alone
                            status: tx.signature,
                        }
                    })
                    .collect();
            }
        }
    }
}

fn check_swarm(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<swarm::SwarmRoundResult>>>,
    app: &mut App,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            match rt.block_on(handle) {
                Ok(Ok(result)) => {
                    // Build assistant message with swarm metadata
                    let meta = format!(
                        "[Swarm: {} won | {:.0}% confidence | {} nodes | {}ms]",
                        result.winner_id,
                        result.confidence * 100.0,
                        result.n_nodes,
                        result.total_time_ms,
                    );
                    let content = format!("{}\n\n{}", result.winning_answer, meta);

                    app.chat_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content,
                        timestamp: chrono::Local::now().format("%H:%M").to_string(),
                        confidence: Some(result.confidence),
                        n_nodes: Some(result.n_nodes),
                    });

                    // Update app-level ELO to winner's ELO
                    if let Some((_, new_elo)) = result.elo_updates.iter().find(|(id, _)| *id == result.winner_id) {
                        app.elo_score = *new_elo;
                        app.elo_history.push(*new_elo as u64);
                        if app.elo_history.len() > 20 {
                            app.elo_history.remove(0);
                        }
                    }

                    app.inferences_served += 1;

                    // Update swarm node display
                    for (id, new_elo) in &result.elo_updates {
                        if let Some(sn) = app.swarm_nodes.iter_mut().find(|n| n.id == *id) {
                            sn.elo = *new_elo;
                            sn.rounds += 1;
                            if *id == result.winner_id {
                                sn.wins += 1;
                            }
                        }
                    }

                    // Store last round for dashboard
                    app.last_round_result = Some(SwarmRoundDisplay {
                        winner: result.winner_id,
                        confidence: result.confidence,
                        n_nodes: result.n_nodes,
                        total_time_ms: result.total_time_ms,
                    });

                    app.chat_loading = false;
                }
                Ok(Err(e)) => {
                    app.chat_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: format!("Swarm error: {}", e),
                        timestamp: chrono::Local::now().format("%H:%M").to_string(),
                        confidence: None,
                        n_nodes: None,
                    });
                    app.chat_loading = false;
                }
                Err(e) => {
                    app.chat_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: format!("Swarm task error: {}", e),
                        timestamp: chrono::Local::now().format("%H:%M").to_string(),
                        confidence: None,
                        n_nodes: None,
                    });
                    app.chat_loading = false;
                }
            }
        }
    }
}

fn check_chat(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<api::ChatResponse>>>,
    app: &mut App,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            match rt.block_on(handle) {
                Ok(Ok(resp)) => {
                    if let Some(choice) = resp.choices.first() {
                        app.chat_messages.push(ChatMessage {
                            role: "assistant".into(),
                            content: choice.message.content.clone(),
                            timestamp: chrono::Local::now().format("%H:%M").to_string(),
                            confidence: resp.confidence.or(Some(0.95)),
                            n_nodes: resp.n_nodes.or(Some(3)),
                        });
                    }
                    app.chat_loading = false;
                }
                Ok(Err(e)) => {
                    app.chat_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: format!("Error: {} — is the node running at {}?", e, app.api_endpoint),
                        timestamp: chrono::Local::now().format("%H:%M").to_string(),
                        confidence: None,
                        n_nodes: None,
                    });
                    app.chat_loading = false;
                }
                Err(e) => {
                    app.chat_messages.push(ChatMessage {
                        role: "assistant".into(),
                        content: format!("Internal error: {}", e),
                        timestamp: chrono::Local::now().format("%H:%M").to_string(),
                        confidence: None,
                        n_nodes: None,
                    });
                    app.chat_loading = false;
                }
            }
        }
    }
}

fn check_airdrop(
    rt: &tokio::runtime::Runtime,
    pending: &mut Option<tokio::task::JoinHandle<Result<String>>>,
    app: &mut App,
) {
    if let Some(handle) = pending.as_ref() {
        if handle.is_finished() {
            let handle = pending.take().unwrap();
            match rt.block_on(handle) {
                Ok(Ok(sig)) => {
                    app.wallet_status = format!("Airdrop sent! Sig: {}...{}", &sig[..8], &sig[sig.len().saturating_sub(8)..]);
                }
                Ok(Err(e)) => {
                    app.wallet_status = format!("Airdrop failed: {}", e);
                }
                Err(e) => {
                    app.wallet_status = format!("Airdrop error: {}", e);
                }
            }
        }
    }
}
