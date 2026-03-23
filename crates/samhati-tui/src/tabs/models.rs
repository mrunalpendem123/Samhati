use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3),
    ])
    .split(area);

    draw_table(frame, app, layout[0]);
    draw_help(frame, layout[1]);
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
                "Hindi" | "Rust" | "Python" | "Math" | "DeFi" | "Science" | "Legal" => Color::Magenta,
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

fn draw_help(frame: &mut Frame, area: Rect) {
    let help = Paragraph::new(Line::from(vec![
        Span::styled(" ↑/↓ ", Style::default().fg(PURPLE).bold()),
        Span::styled("select  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Enter ", Style::default().fg(PURPLE).bold()),
        Span::styled("activate  ", Style::default().fg(Color::DarkGray)),
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
