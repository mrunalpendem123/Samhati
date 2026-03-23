use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyModifiers};

use crate::app::{App, ChatMessage, Tab};
use chrono::Local;

pub fn handle_key(app: &mut App, key: KeyEvent) -> Option<ChatAction> {
    // Global keybindings
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        app.running = false;
        return None;
    }

    // If editing a setting, route all input there
    if app.tab == Tab::Settings && app.editing_setting {
        return handle_settings_editing(app, key);
    }

    match key.code {
        KeyCode::Char('q') if app.tab != Tab::Chat => {
            app.running = false;
        }
        KeyCode::Char('1') => app.tab = Tab::Chat,
        KeyCode::Char('2') => app.tab = Tab::Dashboard,
        KeyCode::Char('3') => app.tab = Tab::Models,
        KeyCode::Char('4') => app.tab = Tab::Wallet,
        KeyCode::Char('5') => app.tab = Tab::Settings,
        KeyCode::Tab => app.tab = app.tab.next(),
        KeyCode::BackTab => app.tab = app.tab.prev(),
        _ => {
            return handle_tab_key(app, key);
        }
    }
    None
}

fn handle_tab_key(app: &mut App, key: KeyEvent) -> Option<ChatAction> {
    match app.tab {
        Tab::Chat => handle_chat_key(app, key),
        Tab::Dashboard => {
            handle_dashboard_key(app, key);
            None
        }
        Tab::Models => {
            handle_models_key(app, key);
            None
        }
        Tab::Wallet => handle_wallet_key(app, key),
        Tab::Settings => {
            handle_settings_key(app, key);
            None
        }
    }
}

/// Returned when the chat tab wants to fire off an async API call.
pub enum ChatAction {
    SendMessage(String),
    RequestAirdrop,
}

fn handle_chat_key(app: &mut App, key: KeyEvent) -> Option<ChatAction> {
    if app.chat_loading {
        return None;
    }

    match key.code {
        KeyCode::Enter => {
            let msg = app.chat_input.trim().to_string();
            if msg.is_empty() {
                return None;
            }
            app.chat_messages.push(ChatMessage {
                role: "user".into(),
                content: msg.clone(),
                timestamp: Local::now().format("%H:%M").to_string(),
                confidence: None,
                n_nodes: None,
            });
            app.chat_input.clear();
            app.chat_loading = true;
            app.chat_scroll = 0;
            return Some(ChatAction::SendMessage(msg));
        }
        KeyCode::Backspace => {
            app.chat_input.pop();
        }
        KeyCode::Char(c) => {
            app.chat_input.push(c);
        }
        KeyCode::Up => {
            app.chat_scroll = app.chat_scroll.saturating_add(1);
        }
        KeyCode::Down => {
            app.chat_scroll = app.chat_scroll.saturating_sub(1);
        }
        KeyCode::Esc => {
            app.chat_input.clear();
        }
        _ => {}
    }
    None
}

fn handle_dashboard_key(app: &mut App, key: KeyEvent) {
    if key.code == KeyCode::Char('s') {
        app.node_running = !app.node_running;
    }
}

fn handle_models_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up => {
            if app.selected_model_idx > 0 {
                app.selected_model_idx -= 1;
            }
        }
        KeyCode::Down => {
            if app.selected_model_idx + 1 < app.models.len() {
                app.selected_model_idx += 1;
            }
        }
        KeyCode::Enter => {
            let idx = app.selected_model_idx;
            // Install if not installed, then activate
            app.models[idx].installed = true;
            for m in app.models.iter_mut() {
                m.active = false;
            }
            app.models[idx].active = true;
            app.current_model = app.models[idx].name.clone();
        }
        _ => {}
    }
}

fn handle_wallet_key(app: &mut App, key: KeyEvent) -> Option<ChatAction> {
    match key.code {
        KeyCode::Char('a') => {
            // Request airdrop
            return Some(ChatAction::RequestAirdrop);
        }
        _ => {}
    }
    None
}

fn handle_settings_key(app: &mut App, key: KeyEvent) {
    match key.code {
        KeyCode::Up => {
            if app.selected_setting > 0 {
                app.selected_setting -= 1;
            }
        }
        KeyCode::Down => {
            if app.selected_setting < 1 {
                app.selected_setting += 1;
            }
        }
        KeyCode::Enter => {
            app.editing_setting = true;
            app.setting_input = match app.selected_setting {
                0 => app.api_endpoint.clone(),
                1 => format!("{:.1}", app.max_vram_gb),
                _ => String::new(),
            };
        }
        _ => {}
    }
}

fn handle_settings_editing(app: &mut App, key: KeyEvent) -> Option<ChatAction> {
    match key.code {
        KeyCode::Esc => {
            app.editing_setting = false;
            app.setting_input.clear();
        }
        KeyCode::Enter => {
            match app.selected_setting {
                0 => app.api_endpoint = app.setting_input.clone(),
                1 => {
                    if let Ok(v) = app.setting_input.parse::<f64>() {
                        app.max_vram_gb = v;
                    }
                }
                _ => {}
            }
            app.editing_setting = false;
            app.setting_input.clear();
        }
        KeyCode::Backspace => {
            app.setting_input.pop();
        }
        KeyCode::Char(c) => {
            app.setting_input.push(c);
        }
        _ => {}
    }
    None
}

pub fn poll_event(tick_rate: std::time::Duration) -> anyhow::Result<Option<Event>> {
    if event::poll(tick_rate)? {
        Ok(Some(event::read()?))
    } else {
        Ok(None)
    }
}
