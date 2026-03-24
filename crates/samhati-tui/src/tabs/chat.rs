use ratatui::prelude::*;
use ratatui::widgets::*;

use crate::app::App;
use crate::ui::{BG, DIM_PURPLE, PURPLE, SURFACE};

pub fn draw(frame: &mut Frame, app: &App, area: Rect) {
    let layout = Layout::vertical([
        Constraint::Min(1),   // messages
        Constraint::Length(3), // input
    ])
    .split(area);

    draw_messages(frame, app, layout[0]);
    draw_input(frame, app, layout[1]);
}

fn draw_messages(frame: &mut Frame, app: &App, area: Rect) {
    let block = Block::bordered()
        .title(Span::styled(
            " Messages ",
            Style::default().fg(PURPLE).bold(),
        ))
        .border_style(Style::default().fg(DIM_PURPLE))
        .style(Style::default().bg(BG));

    let inner = block.inner(area);
    frame.render_widget(block, area);

    let mut lines: Vec<Line> = Vec::new();

    for msg in &app.chat_messages {
        let is_user = msg.role == "user";

        // Blank line between messages
        if !lines.is_empty() {
            lines.push(Line::from(""));
        }

        // Header
        let header_style = if is_user {
            Style::default().fg(Color::Cyan).bold()
        } else {
            Style::default().fg(Color::Green).bold()
        };
        let role_label = if is_user { "You" } else { "Samhati" };

        let mut header_spans = vec![
            Span::styled(format!(" {} ", role_label), header_style),
            Span::styled(
                format!(" {}", msg.timestamp),
                Style::default().fg(Color::DarkGray),
            ),
        ];

        if let (Some(conf), Some(nodes)) = (msg.confidence, msg.n_nodes) {
            header_spans.push(Span::styled(
                format!("  conf: {:.0}%  nodes: {}", conf * 100.0, nodes),
                Style::default().fg(Color::DarkGray),
            ));
        }

        lines.push(Line::from(header_spans));

        // Content lines
        let prefix = if is_user { "  > " } else { "  " };
        let content_style = if is_user {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default().fg(Color::White)
        };

        let wrap_width = inner.width.saturating_sub(prefix.len() as u16 + 1) as usize;
        for text_line in msg.content.lines() {
            if wrap_width == 0 || text_line.len() <= wrap_width {
                lines.push(Line::from(Span::styled(
                    format!("{}{}", prefix, text_line),
                    content_style,
                )));
            } else {
                // Word-wrap long lines
                let continuation = " ".repeat(prefix.len());
                let mut remaining = text_line;
                let mut first = true;
                while !remaining.is_empty() {
                    let p = if first { prefix } else { &continuation };
                    first = false;
                    if remaining.len() <= wrap_width {
                        lines.push(Line::from(Span::styled(
                            format!("{}{}", p, remaining),
                            content_style,
                        )));
                        break;
                    }
                    // Find last space within wrap_width for word boundary
                    let split = remaining[..wrap_width]
                        .rfind(' ')
                        .unwrap_or(wrap_width);
                    let (chunk, rest) = remaining.split_at(split);
                    lines.push(Line::from(Span::styled(
                        format!("{}{}", p, chunk),
                        content_style,
                    )));
                    remaining = rest.trim_start();
                }
            }
        }
    }

    // Loading indicator
    if app.chat_loading {
        lines.push(Line::from(""));
        lines.push(Line::from(Span::styled(
            "  Thinking...",
            Style::default().fg(PURPLE).italic(),
        )));
    }

    // Scroll to bottom
    let visible_height = inner.height as usize;
    let total = lines.len();
    let scroll = if app.chat_scroll > 0 {
        let max_scroll = total.saturating_sub(visible_height);
        max_scroll.saturating_sub(app.chat_scroll as usize)
    } else {
        total.saturating_sub(visible_height)
    };

    let paragraph = Paragraph::new(lines)
        .scroll((scroll as u16, 0))
        .style(Style::default().bg(BG));
    frame.render_widget(paragraph, inner);
}

fn draw_input(frame: &mut Frame, app: &App, area: Rect) {
    let style = if app.chat_loading {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input = Paragraph::new(app.chat_input.as_str())
        .style(style)
        .block(
            Block::bordered()
                .title(Span::styled(
                    " Message (Enter to send) ",
                    Style::default().fg(PURPLE),
                ))
                .border_style(Style::default().fg(DIM_PURPLE))
                .style(Style::default().bg(SURFACE)),
        );

    frame.render_widget(input, area);

    // Place cursor
    if !app.chat_loading {
        frame.set_cursor_position(Position::new(
            area.x + app.chat_input.len() as u16 + 1,
            area.y + 1,
        ));
    }
}
