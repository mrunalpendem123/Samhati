use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Length(5),  // pubkey
        Constraint::Length(8),  // balance overview
        Constraint::Length(3),  // status / help
        Constraint::Min(1),    // tx history
    ])
    .split(area);

    draw_pubkey(frame, app, layout[0]);
    draw_balance(frame, app, layout[1]);
    draw_status(frame, app, layout[2]);
    draw_tx_history(frame, app, layout[3]);
}

fn draw_pubkey(frame: &mut Frame, app: &App, area: Rect) {
    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("  Solana Pubkey: ", Style::default().fg(Color::DarkGray)),
            Span::styled(&app.wallet_pubkey, Style::default().fg(Color::Yellow).bold()),
        ]),
        Line::from(vec![
            Span::styled("  Network: ", Style::default().fg(Color::DarkGray)),
            Span::styled("Devnet", Style::default().fg(Color::Cyan)),
            Span::styled("  |  Keypair: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                app.wallet.as_ref()
                    .map(|w| w.keypair_path.display().to_string())
                    .unwrap_or_else(|| "none".into()),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    let block = Block::bordered()
        .title(Span::styled(" Wallet Identity ", Style::default().fg(PURPLE).bold()))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));

    frame.render_widget(Paragraph::new(lines).block(block), area);
}

fn draw_balance(frame: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::horizontal([
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(34),
    ])
    .split(area);

    // SOL Balance (real from devnet)
    let sol_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("{:.4} SOL", app.sol_balance),
            Style::default().fg(Color::Yellow).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Devnet Balance",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let sol_block = Block::bordered()
        .title(Span::styled(" SOL ", Style::default().fg(PURPLE).bold()))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));
    frame.render_widget(Paragraph::new(sol_lines).block(sol_block).alignment(Alignment::Center), cols[0]);

    // SMTI Balance (protocol token — will be real after token mint)
    let smti_lines = vec![
        Line::from(""),
        Line::from(Span::styled(
            format!("{:.3} SMTI", app.smti_balance),
            Style::default().fg(Color::Magenta).bold(),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Protocol Token",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let smti_block = Block::bordered()
        .title(Span::styled(" SMTI ", Style::default().fg(PURPLE).bold()))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));
    frame.render_widget(Paragraph::new(smti_lines).block(smti_block).alignment(Alignment::Center), cols[1]);

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
        .title(Span::styled(" Rewards ", Style::default().fg(PURPLE).bold()))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(SURFACE));
    frame.render_widget(Paragraph::new(reward_lines).block(reward_block).alignment(Alignment::Center), cols[2]);
}

fn draw_status(frame: &mut Frame, app: &App, area: Rect) {
    let status_text = if !app.wallet_status.is_empty() {
        vec![Span::styled(&app.wallet_status, Style::default().fg(Color::Cyan))]
    } else {
        vec![
            Span::styled(" a ", Style::default().fg(PURPLE).bold()),
            Span::styled("request devnet airdrop (1 SOL)  ", Style::default().fg(Color::DarkGray)),
            Span::styled(" r ", Style::default().fg(PURPLE).bold()),
            Span::styled("refresh balance", Style::default().fg(Color::DarkGray)),
        ]
    };

    let help = Paragraph::new(Line::from(status_text))
        .block(
            Block::bordered()
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(SURFACE)),
        );
    frame.render_widget(help, area);
}

fn draw_tx_history(frame: &mut Frame, app: &App, area: Rect) {
    let header = Row::new(vec![
        Cell::from("Timestamp").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Status").style(Style::default().fg(PURPLE).bold()),
        Cell::from("Signature").style(Style::default().fg(PURPLE).bold()),
    ])
    .height(1)
    .bottom_margin(1);

    let rows: Vec<Row> = if app.tx_history.is_empty() {
        vec![Row::new(vec![
            Cell::from(""),
            Cell::from(Span::styled(
                "No transactions yet — request an airdrop with 'a'",
                Style::default().fg(Color::DarkGray),
            )),
            Cell::from(""),
        ])]
    } else {
        app.tx_history
            .iter()
            .map(|tx| {
                let status_style = match tx.tx_type.as_str() {
                    "confirmed" => Style::default().fg(Color::Green),
                    "failed" => Style::default().fg(Color::Red),
                    _ => Style::default().fg(Color::Yellow),
                };

                Row::new(vec![
                    Cell::from(tx.timestamp.clone()),
                    Cell::from(Span::styled(tx.tx_type.clone(), status_style)),
                    Cell::from(Span::styled(tx.status.clone(), Style::default().fg(Color::DarkGray))),
                ])
            })
            .collect()
    };

    let widths = [
        Constraint::Percentage(25),
        Constraint::Percentage(15),
        Constraint::Percentage(60),
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Recent Transactions (Solana Devnet) ",
                    Style::default().fg(PURPLE).bold(),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(BG)),
        );

    frame.render_widget(table, area);
}
