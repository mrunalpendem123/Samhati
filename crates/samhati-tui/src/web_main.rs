//! WASM entry point — runs the Samhati TUI in the browser via ratzilla.
//!
//! Build:  cd crates/samhati-tui && trunk serve
//! Deploy: cd crates/samhati-tui && trunk build --release
//!
//! Shared rendering code (ui.rs, tabs/*) works unchanged in both targets.
//! Native-only features (filesystem, P2P, local inference) are stubbed —
//! the browser build connects to a remote Samhati node via HTTP API.

mod app;
mod api;
mod events;
mod tabs;
mod ui;

// These modules have native-only deps (tokio, iroh, sysinfo, fs).
// Include them as dead code so shared types resolve, but they won't
// be called at runtime in the browser.
#[cfg(not(target_arch = "wasm32"))]
pub mod identity;
#[cfg(not(target_arch = "wasm32"))]
mod model_download;
#[cfg(not(target_arch = "wasm32"))]
pub mod network;
#[cfg(not(target_arch = "wasm32"))]
pub mod registry;
#[cfg(not(target_arch = "wasm32"))]
pub mod settlement;
#[cfg(not(target_arch = "wasm32"))]
mod node_runner;
#[cfg(not(target_arch = "wasm32"))]
mod swarm;
#[cfg(not(target_arch = "wasm32"))]
mod wallet;

use std::cell::RefCell;
use std::io;
use std::rc::Rc;

use app::{App, ChatMessage};
use api::ApiClient;
use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use events::{ChatAction, handle_key};

use ratzilla::event::KeyCode as WebKeyCode;
use ratzilla::ratatui::Terminal;
use ratzilla::{DomBackend, WebRenderer};

fn main() -> io::Result<()> {
    let app = Rc::new(RefCell::new(App::new()));

    // Browser-specific defaults
    {
        let mut a = app.borrow_mut();
        a.api_endpoint = "http://localhost:8000".to_string();
        a.wallet_pubkey = "Browser mode — connect to a Samhati node".to_string();
        a.wallet_short = "browser".to_string();
        a.download_status =
            "Set your node's API endpoint in Settings (tab 5)".to_string();
    }

    let backend = DomBackend::new()?;
    let terminal = Terminal::new(backend)?;

    // Register key event handler (before draw_web which consumes terminal)
    terminal.on_key_event({
        let app = app.clone();
        move |key_event| {
            if let Some(key) = map_key_event(&key_event) {
                let mut app_ref = app.borrow_mut();
                let action = handle_key(&mut app_ref, key);

                if let Some(action) = action {
                    match action {
                        ChatAction::SendMessage(msg) => {
                            let endpoint = app_ref.api_endpoint.clone();
                            let model = app_ref.current_model.clone();
                            app_ref.chat_loading = true;
                            drop(app_ref); // release borrow before spawning async

                            let app_async = app.clone();
                            wasm_bindgen_futures::spawn_local(async move {
                                let client = ApiClient::new(&endpoint);
                                let model_name = if model.is_empty() {
                                    "default".to_string()
                                } else {
                                    model
                                };

                                match client.chat(&msg, &model_name).await {
                                    Ok(resp) => {
                                        let mut a = app_async.borrow_mut();
                                        if let Some(choice) = resp.choices.first() {
                                            a.chat_messages.push(ChatMessage {
                                                role: "assistant".into(),
                                                content: choice.message.content.clone(),
                                                timestamp: "now".to_string(),
                                                confidence: resp.confidence.or(Some(0.95)),
                                                n_nodes: resp.n_nodes.or(Some(1)),
                                            });
                                        }
                                        a.chat_loading = false;
                                    }
                                    Err(e) => {
                                        let mut a = app_async.borrow_mut();
                                        a.chat_messages.push(ChatMessage {
                                            role: "assistant".into(),
                                            content: format!(
                                                "Error: {} — check Settings tab for API endpoint",
                                                e
                                            ),
                                            timestamp: "now".to_string(),
                                            confidence: None,
                                            n_nodes: None,
                                        });
                                        a.chat_loading = false;
                                    }
                                }
                            });
                        }
                        ChatAction::RequestAirdrop => {
                            app_ref.wallet_status =
                                "Airdrop not available in browser — use the native TUI".into();
                        }
                        _ => {
                            // SelectModel, AddSwarmNode, ConnectPeer etc. require native features
                        }
                    }
                }
            }
        }
    });

    // Start the render loop (consumes terminal, runs forever)
    terminal.draw_web({
        let app = app.clone();
        move |frame| {
            let a = app.borrow();
            ui::draw(frame, &a);
        }
    });

    Ok(())
}

/// Map a ratzilla web KeyEvent to a crossterm KeyEvent so the shared
/// event handling code in events.rs works unchanged.
fn map_key_event(event: &ratzilla::event::KeyEvent) -> Option<KeyEvent> {
    let code = match event.code {
        WebKeyCode::Char(c) => KeyCode::Char(c),
        WebKeyCode::Enter => KeyCode::Enter,
        WebKeyCode::Backspace => KeyCode::Backspace,
        WebKeyCode::Esc => KeyCode::Esc,
        WebKeyCode::Tab if event.shift => KeyCode::BackTab,
        WebKeyCode::Tab => KeyCode::Tab,
        WebKeyCode::Up => KeyCode::Up,
        WebKeyCode::Down => KeyCode::Down,
        WebKeyCode::Left => KeyCode::Left,
        WebKeyCode::Right => KeyCode::Right,
        WebKeyCode::Home => KeyCode::Home,
        WebKeyCode::End => KeyCode::End,
        WebKeyCode::PageUp => KeyCode::PageUp,
        WebKeyCode::PageDown => KeyCode::PageDown,
        WebKeyCode::Delete => KeyCode::Delete,
        WebKeyCode::F(n) => KeyCode::F(n),
        _ => return None,
    };

    let mut modifiers = KeyModifiers::empty();
    if event.ctrl {
        modifiers |= KeyModifiers::CONTROL;
    }
    if event.shift {
        modifiers |= KeyModifiers::SHIFT;
    }
    if event.alt {
        modifiers |= KeyModifiers::ALT;
    }

    Some(KeyEvent {
        code,
        modifiers,
        kind: KeyEventKind::Press,
        state: KeyEventState::empty(),
    })
}
