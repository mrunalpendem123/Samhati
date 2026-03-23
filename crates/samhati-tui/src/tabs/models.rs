use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    // If downloading, add a progress bar row; also show errors if any
    let has_progress = app.download_progress.is_some();
    let has_error = !app.node_error.is_empty();
    let has_status = !app.download_status.is_empty();
    let extra_rows = has_progress as u16 + has_error as u16 + has_status as u16;

    let layout = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3 + extra_rows * 2),
    ])
    .split(area);

    draw_table(frame, app, layout[0]);
    draw_footer(frame, app, layout[1], has_progress, has_error, has_status);
}

fn draw_table(frame: &mut Frame, app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Model").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Params").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Domain").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Size").style(Style::default().fg(PURPLE).bold()),
        Cell::from("RAM").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Bonus").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Status").style(Style::default().fg(PURPLE).bold()),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row> = app
        .models
        .iter()
        .enumerate()
        .map(|(i, m)| {
            let status = if m.active {
                Span::styled("● Active", Style::default().fg(Color::Green).bold())
            } else if m.installed {
                Span::styled("✓ Installed", Style::default().fg(Color::Yellow))
            } else if m.recommended {
                Span::styled("★ Recommended", Style::default().fg(Color::Cyan))
            } else {
                Span::styled("  Available", Style::default().fg(Color::DarkGray))
            };

            let domain_color = match m.domain.as_str() {
                "Rust" | "Python" | "Math" | "DeFi" | "Science" | "Legal" => Color::Magenta,
                "Code" => Color::Cyan,
                "Reasoning" => Color::Yellow,
                "Medical" => Color::Red,
                _ => Color::White,
            };

            let name_style = if m.domain.starts_with("Samhati-") || m.name.starts_with("Samhati-") {
                Style::default().fg(Color::Magenta)
            } else {
                Style::default().fg(Color::White)
            };

            let style = if i == app.selected_model_idx {
                Style::default().bg(SURFACE).fg(Color::White)
            } else {
                Style::default()
            };

            Row::new(vec![
                Cell::from(Span::styled(m.name.clone(), name_style)),
                Cell::from(Span::styled(m.params.clone(), Style::default().fg(Color::DarkGray))),
                Cell::from(Span::styled(m.domain.clone(), Style::default().fg(domain_color))),
                Cell::from(format!("{:.1} GB", m.size_gb)),
                Cell::from(format!("{}+ GB", m.min_ram_gb as u32)),
                Cell::from(Span::styled(
                    m.smti_bonus.clone(),
                    if m.smti_bonus != "1.0x" {
                        Style::default().fg(Color::Green).bold()
                    } else {
                        Style::default().fg(Color::DarkGray)
                    },
                )),
                Cell::from(status),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Percentage(24),
        Constraint::Percentage(8),
        Constraint::Percentage(12),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(10),
        Constraint::Percentage(16),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Available Models (auto-detected for your device) ",
                    Style::default().fg(PURPLE).bold(),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(BG)),
        )
        .row_highlight_style(Style::default().bg(SURFACE).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    let mut state = TableState::default().with_selected(Some(app.selected_model_idx));
    frame.render_stateful_widget(table, area, &mut state);
}

fn draw_footer(
    frame: &mut Frame,
    app: &App,
    area: Rect,
    has_progress: bool,
    has_error: bool,
    has_status: bool,
) {
    let mut constraints = vec![Constraint::Length(3)]; // help bar always
    if has_progress {
        constraints.push(Constraint::Length(2));
    }
    if has_status && !has_progress {
        // Show status line when not downloading (e.g. "Qwen2.5-3B is running on port 8000")
        constraints.push(Constraint::Length(1));
    }
    if has_error {
        constraints.push(Constraint::Length(1));
    }

    let chunks = Layout::vertical(constraints).split(area);
    let mut chunk_idx = 0;

    // Help bar
    draw_help(frame, chunks[chunk_idx]);
    chunk_idx += 1;

    // Download progress gauge
    if has_progress {
        if chunk_idx < chunks.len() {
            let pct = app.download_progress.unwrap_or(0.0);
            let label = format!("{} ({:.0}%)", app.download_status, pct);
            let gauge = Gauge::default()
                .block(Block::default())
                .gauge_style(
                    Style::default()
                        .fg(PURPLE)
                        .bg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                )
                .ratio((pct / 100.0).clamp(0.0, 1.0))
                .label(Span::styled(label, Style::default().fg(Color::White)));
            frame.render_widget(gauge, chunks[chunk_idx]);
            chunk_idx += 1;
        }
    } else if has_status {
        // Status line (post-download)
        if chunk_idx < chunks.len() {
            let status = Paragraph::new(Span::styled(
                app.download_status.clone(),
                Style::default().fg(Color::Green),
            ));
            frame.render_widget(status, chunks[chunk_idx]);
            chunk_idx += 1;
        }
    }

    // Error line
    if has_error && chunk_idx < chunks.len() {
        let err = Paragraph::new(Span::styled(
            app.node_error.clone(),
            Style::default().fg(Color::Red),
        ));
        frame.render_widget(err, chunks[chunk_idx]);
    }
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let help = Paragraph::new(Line::from(vec![
        Span::styled(" ↑/↓ ", Style::default().fg(PURPLE).bold()),
        Span::styled("select  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Enter ", Style::default().fg(PURPLE).bold()),
        Span::styled("download & activate  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" ★ ", Style::default().fg(Color::Cyan).bold()),
        Span::styled("= fits your RAM  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" 1.5x+ ", Style::default().fg(Color::Green).bold()),
        Span::styled("= domain specialist bonus", Style::default().fg(Color::DarkGray)),
    ]))
    .block(
        Block::bordered()
            .border_style(Style::default().fg(DIM_PURPLE))
            .style(Style::default().bg(SURFACE)),
    );

    frame.render_widget(help, area);
}
