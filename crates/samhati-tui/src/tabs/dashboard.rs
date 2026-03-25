use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let has_swarm = !app.swarm_nodes.is_empty();
    let layout = Layout::vertical([
        Constraint::Length(7),   // ELO hero + sparkline
        Constraint::Length(9),   // stats grid
        Constraint::Length(6),   // domain demand
        Constraint::Min(1),      // swarm nodes or node info
    ])
    .split(area);

    draw_elo_section(frame, app, layout[0]);
    draw_stats(frame, app, layout[1]);
    draw_demand(frame, app, layout[2]);

    if has_swarm {
        draw_swarm_nodes(frame, app, layout[3]);
    } else {
        draw_node_info(frame, app, layout[3]);
    }
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

fn draw_demand(frame: &mut Frame, app: &App, area: Rect) {
    let d = &app.demand;
    let bar_width = 20usize;

    let make_bar = |pct: f64| -> String {
        let filled = ((pct / 100.0) * bar_width as f64).round() as usize;
        let empty = bar_width.saturating_sub(filled);
        format!("{}{}", "█".repeat(filled), "░".repeat(empty))
    };

    let tip = if d.total == 0 {
        "  No queries yet — start chatting to generate demand data".into()
    } else {
        let best = if d.code_pct() >= d.math_pct() && d.code_pct() >= d.reasoning_pct() {
            "Code"
        } else if d.math_pct() >= d.reasoning_pct() {
            "Math"
        } else {
            "Reasoning"
        };
        format!("  Run a {} model for highest demand + 1.5x SMTI bonus", best)
    };

    let lines = vec![
        Line::from(vec![
            Span::styled("  Code      ", Style::default().fg(Color::Cyan)),
            Span::styled(make_bar(d.code_pct()), Style::default().fg(Color::Cyan)),
            Span::styled(format!(" {:.0}%", d.code_pct()), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Math      ", Style::default().fg(Color::Yellow)),
            Span::styled(make_bar(d.math_pct()), Style::default().fg(Color::Yellow)),
            Span::styled(format!(" {:.0}%", d.math_pct()), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  Reasoning ", Style::default().fg(Color::Magenta)),
            Span::styled(make_bar(d.reasoning_pct()), Style::default().fg(Color::Magenta)),
            Span::styled(format!(" {:.0}%", d.reasoning_pct()), Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("  General   ", Style::default().fg(Color::DarkGray)),
            Span::styled(make_bar(d.general_pct()), Style::default().fg(Color::DarkGray)),
            Span::styled(format!(" {:.0}%", d.general_pct()), Style::default().fg(Color::White)),
        ]),
    ];

    let block = Block::bordered()
        .title(Span::styled(
            format!(" Network Demand ({} queries) ", d.total),
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(BG));

    let para = Paragraph::new(lines).block(block);
    frame.render_widget(para, area);
}

fn draw_swarm_nodes(frame: &mut Frame, app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Node ID").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Model").style(Style::default().fg(PURPLE).bold()),
        Cell::from("ELO").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Rounds").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Wins").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Win %").style(Style::default().fg(PURPLE).bold()),
        Cell::from("URL").style(Style::default().fg(PURPLE).bold()),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row> = app
        .swarm_nodes
        .iter()
        .map(|n| {
            let win_pct = if n.rounds > 0 {
                format!("{:.0}%", (n.wins as f64 / n.rounds as f64) * 100.0)
            } else {
                "-".into()
            };
            let elo_color = if n.elo >= 1550 {
                Color::Green
            } else if n.elo >= 1450 {
                Color::Yellow
            } else {
                Color::Red
            };
            Row::new(vec![
                Cell::from(n.id.clone()).style(Style::default().fg(Color::White)),
                Cell::from(n.model.clone()).style(Style::default().fg(Color::Cyan)),
                Cell::from(format!("{}", n.elo)).style(Style::default().fg(elo_color).bold()),
                Cell::from(format!("{}", n.rounds)).style(Style::default().fg(Color::White)),
                Cell::from(format!("{}", n.wins)).style(Style::default().fg(Color::Green)),
                Cell::from(win_pct).style(Style::default().fg(Color::White)),
                Cell::from(n.url.clone()).style(Style::default().fg(Color::DarkGray)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Percentage(16),
        Constraint::Percentage(20),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(24),
    ];

    let mut title_text = format!(" Swarm Nodes ({}) ", app.swarm_nodes.len());
    if let Some(ref round) = app.last_round_result {
        title_text = format!(
            " Swarm Nodes ({}) | Last: {} won {:.0}% conf {}ms ",
            app.swarm_nodes.len(),
            round.winner,
            round.confidence * 100.0,
            round.total_time_ms,
        );
    }

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(Span::styled(
                    title_text,
                    Style::default().fg(PURPLE).bold(),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(BG)),
        );

    frame.render_widget(table, area);
}

fn draw_node_info(frame: &mut Frame, app: &App, area: Rect) {
    let status_indicator = if app.node_running {
        Span::styled("RUNNING", Style::default().fg(Color::Green).bold())
    } else {
        Span::styled("STOPPED", Style::default().fg(Color::Red).bold())
    };

    let node_id_display = if app.node_id.len() > 16 {
        format!("{}...{}", &app.node_id[..8], &app.node_id[app.node_id.len() - 8..])
    } else if app.node_id.is_empty() {
        "loading...".into()
    } else {
        app.node_id.clone()
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Status:   ", Style::default().fg(Color::DarkGray)),
            status_indicator,
        ]),
        Line::from(vec![
            Span::styled("  Node ID:  ", Style::default().fg(Color::DarkGray)),
            Span::styled(node_id_display, Style::default().fg(Color::Cyan)),
            Span::styled("  (Solana + iroh + TOPLOC — same key)", Style::default().fg(Color::DarkGray)),
        ]),
        Line::from(vec![
            Span::styled("  Wallet:   ", Style::default().fg(Color::DarkGray)),
            Span::styled(&app.wallet_short, Style::default().fg(Color::Yellow)),
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
