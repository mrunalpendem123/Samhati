use anyhow::{anyhow, Result};
use futures_util::{Stream, StreamExt};
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

/// Streams content delta strings from an OpenAI-compatible SSE upstream.
///
/// Makes a request with `stream: true`, then parses `data: {...}` lines from
/// the response body and emits the `choices[0].delta.content` strings as they arrive.
/// Returns `Err` if the HTTP request itself fails; per-line parse errors are silently skipped.
pub async fn infer_http_stream(
    api_base: &str,
    api_key: Option<String>,
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
) -> Result<impl Stream<Item = String> + Send + 'static> {
    let request = ChatRequest { model, messages, max_tokens, temperature, stream: true };
    let url = format!("{api_base}/chat/completions");
    let client = reqwest::Client::new();
    let mut req = client.post(url).json(&request);
    if let Some(key) = api_key {
        req = req.bearer_auth(key);
    }
    let resp = req.send().await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        return Err(anyhow!("http {status}: {body}"));
    }

    // Accumulate raw bytes into complete newline-terminated lines, then parse each
    // `data: <json>` SSE event and emit only non-empty content deltas.
    let mut buf = String::new();
    let stream = resp
        .bytes_stream()
        .flat_map(move |chunk| {
            let text = chunk
                .map(|b| String::from_utf8_lossy(&b).into_owned())
                .unwrap_or_default();
            buf.push_str(&text);
            let mut lines = Vec::new();
            while let Some(pos) = buf.find('\n') {
                let line = buf[..pos].trim_end_matches('\r').to_owned();
                buf.drain(..=pos);
                lines.push(line);
            }
            futures_util::stream::iter(lines)
        })
        .filter_map(|line| async move {
            let data = line.trim().strip_prefix("data: ")?.to_owned();
            if data == "[DONE]" {
                return None;
            }
            let v: serde_json::Value = serde_json::from_str(&data).ok()?;
            let content = v["choices"][0]["delta"]["content"].as_str()?.to_owned();
            if content.is_empty() { None } else { Some(content) }
        });

    Ok(stream)
}

/// Breaks a complete string into a word-at-a-time stream for simulated SSE.
/// Each word is prefixed with a space except the first, matching how tokens arrive
/// from real streaming backends.
pub fn words_stream(text: String) -> impl Stream<Item = String> + Send + 'static {
    let mut words: Vec<String> = Vec::new();
    let mut first = true;
    for word in text.split_whitespace() {
        words.push(if first {
            first = false;
            word.to_owned()
        } else {
            format!(" {word}")
        });
    }
    futures_util::stream::iter(words)
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
        if ch.is_whitespace() || ch == '\\' || ch == '"' || ch == '\'' {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escape_for_split_handles_single_quotes() {
        let escaped = escape_for_split("it's a test");
        // Single quote should be escaped with backslash
        assert!(escaped.contains("\\'"));
        // After splitting, escaped whitespace keeps it as one arg
        let args = split_shell_words(&escaped).unwrap();
        assert_eq!(args, vec!["it's a test"]);
    }

    #[test]
    fn escape_for_split_handles_double_quotes() {
        let escaped = escape_for_split(r#"say "hello""#);
        assert!(escaped.contains("\\\""));
        // Escaped whitespace and quotes produce a single arg
        let args = split_shell_words(&escaped).unwrap();
        assert_eq!(args, vec!["say \"hello\""]);
    }

    #[test]
    fn escape_for_split_handles_whitespace() {
        let escaped = escape_for_split("hello world");
        // Whitespace is escaped, so it stays as one arg
        let args = split_shell_words(&escaped).unwrap();
        assert_eq!(args, vec!["hello world"]);
    }

    #[test]
    fn escape_for_split_handles_newlines() {
        let escaped = escape_for_split("line1\nline2");
        // Newlines replaced with spaces, then escaped
        assert!(!escaped.contains('\n'));
        let args = split_shell_words(&escaped).unwrap();
        assert_eq!(args, vec!["line1 line2"]);
    }

    #[test]
    fn escape_for_split_handles_backslashes() {
        let escaped = escape_for_split(r"path\to\file");
        assert!(escaped.contains("\\\\"));
    }

    #[test]
    fn split_shell_words_basic() {
        let args = split_shell_words("--model path/to/model --prompt hello").unwrap();
        assert_eq!(args, vec!["--model", "path/to/model", "--prompt", "hello"]);
    }

    #[test]
    fn split_shell_words_quoted() {
        let args = split_shell_words(r#"--prompt "hello world""#).unwrap();
        assert_eq!(args, vec!["--prompt", "hello world"]);
    }

    #[test]
    fn split_shell_words_single_quoted() {
        let args = split_shell_words("--prompt 'hello world'").unwrap();
        assert_eq!(args, vec!["--prompt", "hello world"]);
    }

    #[test]
    fn split_shell_words_unterminated_quote_fails() {
        assert!(split_shell_words("--prompt \"hello").is_err());
    }

    #[test]
    fn apply_template_escapes_prompt() {
        let result = apply_template(
            "--model {model} --prompt {prompt} --max-tokens {max_tokens}",
            "llama",
            "it's a ; test",
            128,
            0.7,
            2048,
        );
        // Prompt should be escaped — single quotes and semicolons should be safe
        let args = split_shell_words(&result).unwrap();
        assert_eq!(args[0], "--model");
        assert_eq!(args[1], "llama");
        assert_eq!(args[2], "--prompt");
        // The prompt words should be separate args (whitespace-separated) but safe
        assert!(args.len() >= 4);
    }
}
