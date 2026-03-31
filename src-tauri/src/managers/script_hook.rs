use crate::settings::AppSettings;
use log::{debug, warn};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::io::{Read, Write};
use std::path::Path;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub enum ScriptHookStage {
    AsrPost,
    LlmPost,
}

impl ScriptHookStage {
    fn as_str(self) -> &'static str {
        match self {
            Self::AsrPost => "asr_post",
            Self::LlmPost => "llm_post",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ScriptHookContext<'a> {
    pub lang: Option<&'a str>,
    pub model_id: Option<&'a str>,
    pub provider_id: Option<&'a str>,
    pub prompt_id: Option<&'a str>,
    pub system_prompt: Option<&'a str>,
    pub user_prompt_template: Option<&'a str>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
struct ScriptHookRequest<'a> {
    stage: &'a str,
    text: &'a str,
    lang: Option<&'a str>,
    model_id: Option<&'a str>,
    provider_id: Option<&'a str>,
    prompt_id: Option<&'a str>,
    system_prompt: Option<&'a str>,
    user_prompt_template: Option<&'a str>,
    metadata: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ScriptHookResponse {
    text: String,
    #[serde(default)]
    warnings: Vec<String>,
}

fn build_script_command(path: &str) -> Command {
    let script_path = Path::new(path);
    let extension = script_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.to_ascii_lowercase());

    match extension.as_deref() {
        Some("py") => {
            let mut cmd = Command::new("python3");
            cmd.arg(path);
            cmd
        }
        Some("js") | Some("mjs") | Some("cjs") => {
            let mut cmd = Command::new("node");
            cmd.arg(path);
            cmd
        }
        Some("sh") => {
            let mut cmd = Command::new("bash");
            cmd.arg(path);
            cmd
        }
        _ => Command::new(path),
    }
}

fn read_all_from_pipe<R: Read>(pipe: Option<R>) -> Vec<u8> {
    let Some(mut stream) = pipe else {
        return Vec::new();
    };
    let mut buffer = Vec::new();
    let _ = stream.read_to_end(&mut buffer);
    buffer
}

pub fn run_script_hook(
    stage: ScriptHookStage,
    script_path: &str,
    input_text: &str,
    timeout_ms: u64,
    context: ScriptHookContext<'_>,
) -> Result<String, String> {
    let normalized_path = script_path.trim();
    if normalized_path.is_empty() {
        return Err("script path is empty".to_string());
    }
    if !Path::new(normalized_path).exists() {
        return Err(format!("script path does not exist: {}", normalized_path));
    }

    let timeout = Duration::from_millis(timeout_ms.clamp(100, 10_000));
    let request = ScriptHookRequest {
        stage: stage.as_str(),
        text: input_text,
        lang: context.lang,
        model_id: context.model_id,
        provider_id: context.provider_id,
        prompt_id: context.prompt_id,
        system_prompt: context.system_prompt,
        user_prompt_template: context.user_prompt_template,
        metadata: context.metadata.unwrap_or_else(|| json!({})),
    };

    let payload = serde_json::to_vec(&request)
        .map_err(|err| format!("failed to serialize script request: {}", err))?;

    let mut command = build_script_command(normalized_path);
    command
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    debug!(
        "script hook ({}) starting: path='{}', input_len={}, timeout_ms={}",
        stage.as_str(),
        normalized_path,
        input_text.len(),
        timeout.as_millis()
    );

    let mut child = command
        .spawn()
        .map_err(|err| format!("failed to spawn script '{}': {}", normalized_path, err))?;

    if let Some(mut stdin) = child.stdin.take() {
        stdin
            .write_all(&payload)
            .map_err(|err| format!("failed to write request to script stdin: {}", err))?;
        stdin
            .write_all(b"\n")
            .map_err(|err| format!("failed to write newline to script stdin: {}", err))?;
    }

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    let stdout_reader = thread::spawn(move || read_all_from_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_all_from_pipe(stderr));

    let wait_start = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {
                if wait_start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    let stdout_bytes = stdout_reader.join().unwrap_or_default();
                    let stderr_bytes = stderr_reader.join().unwrap_or_default();
                    let stdout_text = String::from_utf8_lossy(&stdout_bytes);
                    let stderr_text = String::from_utf8_lossy(&stderr_bytes);
                    return Err(format!(
                        "script '{}' timed out after {}ms (stdout='{}', stderr='{}')",
                        normalized_path,
                        timeout.as_millis(),
                        stdout_text.trim(),
                        stderr_text.trim()
                    ));
                }
                thread::sleep(Duration::from_millis(10));
            }
            Err(err) => return Err(format!("failed while waiting for script: {}", err)),
        }
    };

    let stdout_bytes = stdout_reader.join().unwrap_or_default();
    let stderr_bytes = stderr_reader.join().unwrap_or_default();
    let stdout_text = String::from_utf8_lossy(&stdout_bytes).trim().to_string();
    let stderr_text = String::from_utf8_lossy(&stderr_bytes).trim().to_string();

    if !status.success() {
        return Err(format!(
            "script '{}' exited with code {:?} (stderr='{}', stdout='{}')",
            normalized_path,
            status.code(),
            stderr_text,
            stdout_text
        ));
    }

    if stdout_text.is_empty() {
        return Err(format!(
            "script '{}' returned empty stdout",
            normalized_path
        ));
    }

    if let Ok(parsed) = serde_json::from_str::<ScriptHookResponse>(&stdout_text) {
        if parsed.text.trim().is_empty() {
            return Err(format!(
                "script '{}' returned empty text field",
                normalized_path
            ));
        }
        debug!(
            "script hook ({}) finished with JSON output: path='{}', output_len={}",
            stage.as_str(),
            normalized_path,
            parsed.text.trim().len()
        );
        if !parsed.warnings.is_empty() {
            warn!(
                "script hook '{}' warnings: {}",
                normalized_path,
                parsed.warnings.join(" | ")
            );
        }
        return Ok(parsed.text.trim().to_string());
    }

    debug!(
        "script hook ({}) finished with plain text output: path='{}', output_len={}",
        stage.as_str(),
        normalized_path,
        stdout_text.len()
    );
    Ok(stdout_text)
}

pub fn maybe_apply_script_hook(
    settings: &AppSettings,
    stage: ScriptHookStage,
    script_path: Option<&str>,
    input_text: &str,
    context: ScriptHookContext<'_>,
) -> String {
    if !settings.script_hooks_enabled {
        return input_text.to_string();
    }
    let Some(path) = script_path.map(str::trim).filter(|value| !value.is_empty()) else {
        return input_text.to_string();
    };
    match run_script_hook(
        stage,
        path,
        input_text,
        settings.script_hook_timeout_ms,
        context,
    ) {
        Ok(output) => output,
        Err(err) => {
            warn!(
                "Script hook ({}) failed, fallback to original text: {}",
                stage.as_str(),
                err
            );
            input_text.to_string()
        }
    }
}
