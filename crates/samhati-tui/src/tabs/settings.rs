use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Length(12), // settings form
        Constraint::Min(1),     // info
    ])
    .split(area);

    draw_form(frame, app, layout[0]);
    draw_info(frame, app, layout[1]);
}

fn draw_form(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered()
        .title(Span::styled(
            " Settings ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let rows = Layout::vertical([
        Constraint::Length(3),
        Constraint::Length(3),
        Constraint::Length(2),
    ])
    .split(inner);

    // API Endpoint
    let ep_style = if app.selected_setting == 0 {
        Style::default().fg(PURPLE).bold()
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let ep_value = if app.editing_setting && app.selected_setting == 0 {
        format!("{}|", app.setting_input)
    } else {
        app.api_endpoint.clone()
    };

    let ep = Paragraph::new(Line::from(vec![
        Span::styled("  API Endpoint:  ", ep_style),
        Span::styled(ep_value, Style::default().fg(Color::White)),
    ]));
    frame.render_widget(ep, rows[0]);

    // VRAM
    let vram_style = if app.selected_setting == 1 {
        Style::default().fg(PURPLE).bold()
    } else {
        Style::default().fg(Color::DarkGray)
    };

    let vram_value = if app.editing_setting && app.selected_setting == 1 {
        format!("{}|", app.setting_input)
    } else {
        format!("{:.1} GB", app.max_vram_gb)
    };

    let vram = Paragraph::new(Line::from(vec![
        Span::styled("  Max VRAM:      ", vram_style),
        Span::styled(vram_value, Style::default().fg(Color::White)),
    ]));
    frame.render_widget(vram, rows[1]);

    // Help
    let help = Paragraph::new(Line::from(vec![
        Span::styled(
            "  Up/Down to select, Enter to edit, Esc to cancel",
            Style::default().fg(Color::DarkGray),
        ),
    ]));
    frame.render_widget(help, rows[2]);
}

fn draw_info(frame: &mut Frame, app: &App, area: Rect) {
    let node_status = if app.node_running {
        Span::styled("Online", Style::default().fg(Color::Green).bold())
    } else {
        Span::styled("Offline", Style::default().fg(Color::Red).bold())
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Node Status:     ", Style::default().fg(Color::DarkGray)),
            node_status,
        ]),
        Line::from(vec![
            Span::styled("  Current Model:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(&app.current_model, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Peers Connected: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.peers_connected),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Uptime:          ", Style::default().fg(Color::DarkGray)),
            Span::styled(app.format_uptime(), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  ELO Score:       ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{}", app.elo_score),
                Style::default().fg(PURPLE).bold(),
            ),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "  Samhati v0.1.0 — Decentralized P2P LLM Inference",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::bordered()
        .title(Span::styled(
            " Node Info ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(BG));

    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, area);
}
