mod app;
mod api;
mod events;
mod tabs;
mod ui;

use std::io;
use std::time::Duration;

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

fn main() -> Result<()> {
    // Set up terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Build the tokio runtime manually so we can use it for async API calls
    // while keeping the main loop synchronous for smooth TUI rendering.
    let rt = tokio::runtime::Runtime::new()?;

    let mut app = App::new();
    let tick_rate = Duration::from_millis(33); // ~30 fps

    // Pending async chat response
    let mut pending_chat: Option<tokio::task::JoinHandle<Result<api::ChatResponse>>> = None;

    while app.running {
        // Draw
        terminal.draw(|frame| ui::draw(frame, &app))?;

        // Check if a pending chat response has completed
        if let Some(handle) = pending_chat.as_ref() {
            if handle.is_finished() {
                let handle = pending_chat.take().unwrap();
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
                            content: format!("Error: {}", e),
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

        // Poll events
        if let Some(event) = poll_event(tick_rate)? {
            if let Event::Key(key) = event {
                // crossterm 0.28 fires Press and Release; we only want Press
                if key.kind == KeyEventKind::Press {
                    if let Some(action) = handle_key(&mut app, key) {
                        match action {
                            ChatAction::SendMessage(msg) => {
                                let client =
                                    ApiClient::new(&app.api_endpoint);
                                let model = app.current_model.clone();
                                pending_chat =
                                    Some(rt.spawn(async move {
                                        client.chat(&msg, &model).await
                                    }));
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
