use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::{App, Tab};
use crate::tabs;

/// Purple accent color matching Samhati branding (#8B5CF6).
pub const PURPLE: Color = Color::Rgb(139, 92, 246);
pub const DIM_PURPLE: Color = Color::Rgb(100, 66, 176);
pub const BG: Color = Color::Rgb(15, 15, 25);
pub const SURFACE: Color = Color::Rgb(24, 24, 40);

pub fn draw(frame: &mut Frame, app: &App) {
    let area = frame.area();

    // Fill background
    frame.render_widget(Block::default().style(Style::default().bg(BG)), area);

    let layout = Layout::vertical([
        Constraint::Length(3), // tab bar
        Constraint::Min(1),   // content
        Constraint::Length(1), // status bar (single line, no border)
    ])
    .split(area);

    draw_tab_bar(frame, app, layout[0]);
    draw_content(frame, app, layout[1]);
    draw_status_bar(frame, app, layout[2]);
}

fn draw_tab_bar(frame: &mut Frame, app: &App, area: Rect) {
    let titles: Vec<Line> = Tab::ALL
        .iter()
        .enumerate()
        .map(|(i, t)| {
            let num = format!("{}", i + 1);
            Line::from(vec![
                Span::styled(
                    format!(" {} ", num),
                    Style::default().fg(Color::DarkGray),
                ),
                Span::styled(
                    format!("{} ", t.title()),
                    Style::default().fg(Color::White).bold(),
                ),
            ])
        })
        .collect();

    let tabs = Tabs::new(titles)
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Samhati ",
                    Style::default().fg(PURPLE).bold(),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(SURFACE)),
        )
        .highlight_style(Style::default().fg(PURPLE).bold().underlined())
        .select(app.tab.index());

    frame.render_widget(tabs, area);
}

fn draw_content(frame: &mut Frame, app: &App, area: Rect) {
    // Clear content area first to prevent tab overlap artifacts
    frame.render_widget(Block::default().style(Style::default().bg(BG)), area);

    match app.tab {
        Tab::Chat => tabs::chat::draw(frame, app, area),
        Tab::Dashboard => tabs::dashboard::draw(frame, app, area),
        Tab::Models => tabs::models::draw(frame, app, area),
        Tab::Wallet => tabs::wallet::draw(frame, app, area),
        Tab::Settings => tabs::settings::draw(frame, app, area),
    }
}

fn draw_status_bar(frame: &mut Frame, app: &App, area: Rect) {
    let node_dot = if app.node_running { "●" } else { "○" };
    let node_color = if app.node_running { Color::Green } else { Color::Red };

    let status = Line::from(vec![
        Span::styled(format!(" {} ", node_dot), Style::default().fg(node_color)),
        Span::styled(
            format!("Rep:{:.0}", app.elo_score as f64 / 20.0),
            Style::default().fg(PURPLE).bold(),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("Peers:{}", app.peers_connected),
            Style::default().fg(Color::Cyan),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            format!("{:.1}SOL", app.sol_balance),
            Style::default().fg(Color::Yellow),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.wallet_short.clone(),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(" │ ", Style::default().fg(Color::DarkGray)),
        Span::styled(
            app.current_model.clone(),
            Style::default().fg(Color::White),
        ),
    ]);

    let paragraph = Paragraph::new(status).style(Style::default().bg(SURFACE));
    frame.render_widget(paragraph, area);
}
