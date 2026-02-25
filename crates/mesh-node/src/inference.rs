use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use tokio::process::Command;

#[derive(Debug, Clone)]
pub struct LocalExecConfig {
    pub bin: String,
    pub args_template: String,
    pub model_path: String,
    pub context: u32,
}

#[derive(Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: u32,
    pub temperature: f32,
    pub stream: bool,
}

#[derive(Debug, Deserialize)]
struct ChatResponseChoiceMessage {
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatResponseChoice {
    message: ChatResponseChoiceMessage,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatResponseChoice>,
}

pub async fn infer_http(
    api_base: &str,
    api_key: Option<String>,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
) -> Result<String> {
    let request = ChatRequest {
        model,
        messages,
        max_tokens,
        temperature,
        stream: false,
    };

    let url = format!("{api_base}/chat/completions");
    let client = reqwest::Client::new();
    let mut req = client.post(url).json(&request);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }

    let resp = req.send().await?;
    let status = resp.status();
    if !status.is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("http {status}: {body}"));
    }
    let parsed: ChatResponse = resp.json().await?;
    if let Some(choice) = parsed.choices.get(0) {
        Ok(choice.message.content.clone())
    } else {
        Ok("".to_string())
    }
}

pub async fn infer_local_exec(
    config: &LocalExecConfig,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
) -> Result<String> {
    let prompt = build_prompt(&messages);
    let args_line = apply_template(
        &config.args_template,
        &config.model_path,
        &prompt,
        max_tokens,
        temperature,
        config.context,
    );
    let args = split_shell_words(&args_line)?;

    let output = Command::new(&config.bin).args(args).output().await?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("local exec failed: {stderr}"));
    }
    let text = String::from_utf8_lossy(&output.stdout).to_string();
    Ok(text.trim().to_string())
}

fn build_prompt(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for msg in messages {
        out.push_str(&msg.role);
        out.push_str(": ");
        out.push_str(&msg.content);
        out.push('\n');
    }
    out.push_str("assistant: ");
    out
}

fn apply_template(
    template: &str,
    model: &str,
    prompt: &str,
    max_tokens: u32,
    temperature: f32,
    context: u32,
) -> String {
    let model_escaped = escape_for_split(model);
    let prompt_escaped = escape_for_split(prompt);
    template
        .replace("{model}", &model_escaped)
        .replace("{prompt}", &prompt_escaped)
        .replace("{max_tokens}", &max_tokens.to_string())
        .replace("{temperature}", &format!("{temperature:.2}"))
        .replace("{context}", &context.to_string())
}

fn escape_for_split(value: &str) -> String {
    let mut out = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        let ch = match ch {
            '\n' | '\r' => ' ',
            other => other,
        };
        if ch.is_whitespace() || ch == '\\' || ch == '"' {
            out.push('\\');
        }
        out.push(ch);
    }
    out
}

fn split_shell_words(input: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;

    for ch in input.chars() {
        if escaped {
            cur.push(ch);
            escaped = false;
            continue;
        }

        if ch == '\\' && !in_single {
            escaped = true;
            continue;
        }

        if ch == '\'' && !in_double {
            in_single = !in_single;
            continue;
        }

        if ch == '"' && !in_single {
            in_double = !in_double;
            continue;
        }

        if ch.is_whitespace() && !in_single && !in_double {
            if !cur.is_empty() {
                out.push(cur.clone());
                cur.clear();
            }
            continue;
        }

        cur.push(ch);
    }

    if escaped {
        return Err(anyhow!("unterminated escape in local args"));
    }
    if in_single || in_double {
        return Err(anyhow!("unterminated quote in local args"));
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Ok(out)
}
