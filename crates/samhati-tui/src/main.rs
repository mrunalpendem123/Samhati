mod app;
mod api;
mod events;
mod tabs;
mod ui;
mod wallet;

use std::io;
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{Event, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::prelude::*;

use app::{App, ChatMessage};
use api::ApiClient;
use events::{ChatAction, handle_key, poll_event};
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

    // Load real Solana wallet
    match SolanaWallet::load_or_create() {
        Ok(w) => {
            app.wallet_pubkey = w.pubkey.clone();
            app.wallet_short = w.short_pubkey();
            app.wallet = Some(w);
        }
        Err(e) => {
            app.wallet_pubkey = format!("Error: {}", e);
            app.wallet_short = "no wallet".into();
        }
    }

    let tick_rate = Duration::from_millis(33);

    // Pending async tasks
    let mut pending_chat: Option<tokio::task::JoinHandle<Result<api::ChatResponse>>> = None;
    let mut pending_balance: Option<tokio::task::JoinHandle<Result<f64>>> = None;
    let mut pending_txs: Option<tokio::task::JoinHandle<Result<Vec<wallet::TxInfo>>>> = None;
    let mut pending_health: Option<tokio::task::JoinHandle<Result<bool>>> = None;
    let mut pending_airdrop: Option<tokio::task::JoinHandle<Result<String>>> = None;
    let mut last_refresh = Instant::now().checked_sub(Duration::from_secs(30)).unwrap_or_else(Instant::now);

    while app.running {
        // Draw
        terminal.draw(|frame| ui::draw(frame, &app))?;

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
        check_airdrop(&rt, &mut pending_airdrop, &mut app);

        // Poll events
        if let Some(event) = poll_event(tick_rate)? {
            if let Event::Key(key) = event {
                if key.kind == KeyEventKind::Press {
                    if let Some(action) = handle_key(&mut app, key) {
                        match action {
                            ChatAction::SendMessage(msg) => {
                                let client = ApiClient::new(&app.api_endpoint);
                                let model = app.current_model.clone();
                                pending_chat = Some(rt.spawn(async move {
                                    client.chat(&msg, &model).await
                                }));
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
                        }
                    }
                }
            }
        }
    }

    // Restore terminal
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
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
