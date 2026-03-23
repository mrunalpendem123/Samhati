use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Length(7),  // ELO hero + sparkline
        Constraint::Length(9),  // stats grid
        Constraint::Min(1),     // node info
    ])
    .split(area);

    draw_elo_section(frame, app, layout[0]);
    draw_stats(frame, app, layout[1]);
    draw_node_info(frame, app, layout[2]);
}

fn draw_elo_section(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Percentage(50),
        Constraint::Percentage(50),
    ])
    .split(area);

    // ELO hero
    let elo_text = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("{}", app.elo_score),
            Style::default()
                .fg(PURPLE)
                .bold()
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "ELO Rating",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let elo_block = Block::bordered()
        .title(Span::styled(
            " Rating ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let elo_para = Paragraph::new(elo_text)
        .block(elo_block)
        .alignment(Alignment::Center);

    frame.render_widget(elo_para, cols[0]);

    // Sparkline for ELO history
    let spark_block = Block::bordered()
        .title(Span::styled(
            " ELO History ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let sparkline = Sparkline::default()
        .block(spark_block)
        .data(&app.elo_history)
        .style(Style::default().fg(PURPLE));

    frame.render_widget(sparkline, cols[1]);
}

fn draw_stats(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered()
        .title(Span::styled(
            " Statistics ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
    ])
    .split(inner);

    let top_cols = Layout::horizontal([
        Constraint::Percentage(33),
        Constraint::Percentage(34),
        Constraint::Percentage(33),
    ])
    .split(rows[0]);

    let bot_cols = Layout::horizontal([
        Constraint::Percentage(33),
        Constraint::Percentage(34),
        Constraint::Percentage(33),
    ])
    .split(rows[1]);

    render_stat(frame, "SMTI Today", &format!("+{:.2}", app.smti_earned_today), Color::Green, top_cols[0]);
    render_stat(frame, "Total Balance", &format!("{:.3}", app.smti_balance), Color::Yellow, top_cols[1]);
    render_stat(frame, "Inferences", &format!("{}", app.inferences_served), Color::Cyan, top_cols[2]);

    render_stat(frame, "Uptime", &app.format_uptime(), Color::White, bot_cols[0]);
    render_stat(frame, "Peers", &format!("{}", app.peers_connected), Color::Cyan, bot_cols[1]);
    render_stat(frame, "Model", &app.current_model, Color::White, bot_cols[2]);
}

fn render_stat(frame: &mut Frame, label: &str, value: &str, color: Color, area: Rect) {
    let text = vec![
        Line::from(Span::styled(value, Style::default().fg(color).bold())),
        Line::from(Span::styled(label, Style::default().fg(Color::DarkGray))),
    ];
    let p = Paragraph::new(text).alignment(Alignment::Center);
    frame.render_widget(p, area);
}

fn draw_node_info(frame: &mut Frame, app: &App, area: Rect) {
    let status_indicator = if app.node_running {
        Span::styled("RUNNING", Style::default().fg(Color::Green).bold())
    } else {
        Span::styled("STOPPED", Style::default().fg(Color::Red).bold())
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Node Status: ", Style::default().fg(Color::DarkGray)),
            status_indicator,
            Span::styled(
                "    (press 's' to toggle)",
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Endpoint:    ", Style::default().fg(Color::DarkGray)),
            Span::styled(&app.api_endpoint, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  VRAM Alloc:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.1} GB", app.max_vram_gb),
                Style::default().fg(Color::White),
            ),
        ]),
    ];

    let block = Block::bordered()
        .title(Span::styled(
            " Node ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(BG));

    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, area);
}
