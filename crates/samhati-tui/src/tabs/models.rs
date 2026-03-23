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
        Cell::from("Domain").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Size").style(Style::default().fg(PURPLE).bold()),
        Cell::from("SMTI Bonus").style(Style::default().fg(PURPLE).bold()),
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
                Span::styled("Active", Style::default().fg(Color::Green).bold())
            } else if m.installed {
                Span::styled("Installed", Style::default().fg(Color::Yellow))
            } else {
                Span::styled("Not installed", Style::default().fg(Color::DarkGray))
            };

            let style = if i == app.selected_model_idx {
                Style::default().bg(SURFACE).fg(Color::White)
            } else {
                Style::default().fg(Color::White)
            };

            Row::new(vec![
                Cell::from(m.name.clone()),
                Cell::from(m.domain.clone()),
                Cell::from(format!("{:.1} GB", m.size_gb)),
                Cell::from(m.smti_bonus.clone()),
                Cell::from(status),
            ])
            .style(style)
        })
        .collect();

    let widths = [
        Constraint::Percentage(30),
        Constraint::Percentage(15),
        Constraint::Percentage(15),
        Constraint::Percentage(15),
        Constraint::Percentage(25),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Available Models ",
                    Style::default().fg(PURPLE).bold(),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(BG)),
        )
        .row_highlight_style(Style::default().bg(SURFACE).add_modifier(Modifier::BOLD))
        .highlight_symbol(">> ");

    // Use StatefulWidget for highlight
    let mut state = TableState::default().with_selected(Some(app.selected_model_idx));
    frame.render_stateful_widget(table, area, &mut state);
}

fn draw_help(frame: &mut Frame, area: Rect) {
    let help = Paragraph::new(Line::from(vec![
        Span::styled(" Up/Down ", Style::default().fg(PURPLE).bold()),
        Span::styled("select  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" Enter ", Style::default().fg(PURPLE).bold()),
        Span::styled("activate model  ", Style::default().fg(Color::DarkGray)),
        Span::styled(" (only installed models can be activated) ", Style::default().fg(Color::DarkGray)),
    ]))
    .block(
        Block::bordered()
            .border_style(Style::default().fg(DIM_PURPLE))
            .style(Style::default().bg(SURFACE)),
    );

    frame.render_widget(help, area);
}
