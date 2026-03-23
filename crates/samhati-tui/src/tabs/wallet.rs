use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Length(8),  // balance overview
        Constraint::Min(1),     // tx history
    ])
    .split(area);

    draw_balance(frame, app, layout[0]);
    draw_tx_history(frame, app, layout[1]);
}

fn draw_balance(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Percentage(40),
        Constraint::Percentage(30),
        Constraint::Percentage(30),
    ])
    .split(area);

    // Balance
    let balance_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("{:.3} SMTI", app.smti_balance),
            Style::default().fg(Color::Yellow).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("Pubkey: {}", app.wallet_pubkey),
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let balance_block = Block::bordered()
        .title(Span::styled(
            " Balance ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let p = Paragraph::new(balance_lines)
        .block(balance_block)
        .alignment(Alignment::Center);
    frame.render_widget(p, cols[0]);

    // Pending rewards
    let reward_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("+{:.2} SMTI", app.pending_rewards),
            Style::default().fg(Color::Green).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Pending Rewards",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let reward_block = Block::bordered()
        .title(Span::styled(
            " Rewards ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let r = Paragraph::new(reward_lines)
        .block(reward_block)
        .alignment(Alignment::Center);
    frame.render_widget(r, cols[1]);

    // Earned today
    let today_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("+{:.2} SMTI", app.smti_earned_today),
            Style::default().fg(Color::Cyan).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Earned Today",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let today_block = Block::bordered()
        .title(Span::styled(
            " Today ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    let t = Paragraph::new(today_lines)
        .block(today_block)
        .alignment(Alignment::Center);
    frame.render_widget(t, cols[2]);
}

fn draw_tx_history(frame: &mut Frame, app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Timestamp").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Type").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Amount").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Status").style(Style::default().fg(PURPLE).bold()),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row> = app
        .tx_history
        .iter()
        .map(|tx| {
            let type_style = match tx.tx_type.as_str() {
                "reward" => Style::default().fg(Color::Green),
                "claim" => Style::default().fg(Color::Yellow),
                "stake" => Style::default().fg(Color::Magenta),
                _ => Style::default().fg(Color::White),
            };

            let amount_str = if tx.amount >= 0.0 {
                format!("+{:.2}", tx.amount)
            } else {
                format!("{:.2}", tx.amount)
            };
            let amount_color = if tx.amount >= 0.0 {
                Color::Green
            } else {
                Color::Red
            };

            let status_style = match tx.status.as_str() {
                "confirmed" => Style::default().fg(Color::Green),
                "pending" => Style::default().fg(Color::Yellow),
                _ => Style::default().fg(Color::DarkGray),
            };

            Row::new(vec![
                Cell::from(tx.timestamp.clone()),
                Cell::from(Span::styled(tx.tx_type.clone(), type_style)),
                Cell::from(Span::styled(amount_str, Style::default().fg(amount_color))),
                Cell::from(Span::styled(tx.status.clone(), status_style)),
            ])
        })
        .collect();

    let widths = [
        Constraint::Percentage(30),
        Constraint::Percentage(20),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Transaction History ",
                    Style::default().fg(PURPLE).bold(),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(BG)),
        );

    frame.render_widget(table, area);
}
