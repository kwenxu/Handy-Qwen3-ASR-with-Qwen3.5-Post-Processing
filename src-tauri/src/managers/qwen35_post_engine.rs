use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use tauri::AppHandle;

pub(crate) const QWEN35_DEFAULT_ENDPOINT: &str = "https://hf-mirror.com";

static PYTHON_COMMAND_CACHE: OnceLock<String> = OnceLock::new();
static QWEN35_PYTHON_PATH: OnceLock<String> = OnceLock::new();
static QWEN35_SCRIPT_PATH: OnceLock<String> = OnceLock::new();

pub(crate) fn init_qwen35_python_path(
    app_handle: &AppHandle,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let app_data_dir = crate::portable::app_data_dir(app_handle)?;
    let runtime_dir = app_data_dir.join("qwen35_post_mlx");
    let python_path = runtime_dir.join(".venv/bin/python3");
    let _ = QWEN35_PYTHON_PATH.set(python_path.to_string_lossy().to_string());

    let script_path = runtime_dir.join("qwen35_post_server.py");
    if script_path.exists() {
        let _ = QWEN35_SCRIPT_PATH.set(script_path.to_string_lossy().to_string());
    }

    Ok(())
}

fn resolve_server_script_path() -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(cached) = QWEN35_SCRIPT_PATH.get() {
        let path = PathBuf::from(cached);
        if path.exists() {
            return Ok(path);
        }
    }

    let python_path = QWEN35_PYTHON_PATH.get().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Qwen3.5 Python path is not initialized",
        )
    })?;

    let runtime_dir = runtime_dir_from_python_path(python_path)?;
    let script_path = runtime_dir.join("qwen35_post_server.py");
    if !script_path.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "Qwen3.5 post server script not found: {}",
                script_path.display()
            ),
        )));
    }

    let _ = QWEN35_SCRIPT_PATH.set(script_path.to_string_lossy().to_string());
    Ok(script_path)
}

fn runtime_dir_from_python_path(
    python_path: &str,
) -> std::result::Result<PathBuf, Box<dyn std::error::Error>> {
    let python = PathBuf::from(python_path);
    let bin_dir = python.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Invalid Qwen3.5 python path: no parent",
        )
    })?;
    let venv_dir = bin_dir.parent().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Invalid Qwen3.5 python path: no .venv",
        )
    })?;
    let runtime_dir = if venv_dir.file_name().and_then(|s| s.to_str()) == Some(".venv") {
        venv_dir.parent().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::Other,
                "Invalid Qwen3.5 python path: no runtime parent",
            )
        })?
    } else {
        venv_dir
    };
    Ok(runtime_dir.to_path_buf())
}

fn resolve_python_command() -> std::result::Result<String, Box<dyn std::error::Error>> {
    let fixed_python = QWEN35_PYTHON_PATH.get().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::Other,
            "Qwen3.5 Python path is not initialized",
        )
    })?;

    let python_path = PathBuf::from(fixed_python);
    if !python_path.exists() {
        return Err(Box::new(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Qwen3.5 Python not found: {}", python_path.display()),
        )));
    }

    Ok(python_path.to_string_lossy().to_string())
}

fn get_python_command() -> std::result::Result<(String, Vec<String>), Box<dyn std::error::Error>> {
    if let Some(cached) = PYTHON_COMMAND_CACHE.get() {
        return Ok((cached.clone(), vec![]));
    }

    let python = resolve_python_command()?;
    let _ = PYTHON_COMMAND_CACHE.set(python.clone());
    Ok((python, vec![]))
}

pub(crate) fn get_qwen35_python_command(
) -> std::result::Result<(String, Vec<String>), Box<dyn std::error::Error>> {
    get_python_command()
}

#[derive(Debug, Serialize)]
struct Qwen35PostRequest<'a> {
    text: &'a str,
    system_prompt: &'a str,
    template_id: Option<&'a str>,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    repetition_context_size: usize,
}

#[derive(Debug, Deserialize)]
struct Qwen35PostResponse {
    text: Option<String>,
    error: Option<String>,
}

pub struct Qwen35PostEngine {
    model_id: Option<String>,
    child_process: Option<Arc<Mutex<Child>>>,
    stdin: Option<Arc<Mutex<ChildStdin>>>,
    stdout: Option<Arc<Mutex<BufReader<ChildStdout>>>>,
    startup_timeout_secs: u64,
    inference_timeout_secs: u64,
    max_threads: usize,
}

impl Qwen35PostEngine {
    pub fn new() -> Self {
        Self {
            model_id: None,
            child_process: None,
            stdin: None,
            stdout: None,
            startup_timeout_secs: 90,
            inference_timeout_secs: 45,
            max_threads: 0,
        }
    }

    pub fn configure_runtime(
        &mut self,
        startup_timeout_secs: u64,
        inference_timeout_secs: u64,
        max_threads: usize,
    ) {
        self.startup_timeout_secs = startup_timeout_secs.clamp(20, 300);
        self.inference_timeout_secs = inference_timeout_secs.clamp(5, 180);
        self.max_threads = max_threads.clamp(0, 16);
    }

    pub fn load_model(
        &mut self,
        model_id: &str,
        model_path: &Path,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        info!(
            "Loading local Qwen3.5 post-process model {} ({})",
            model_id,
            model_path.display()
        );

        self.start_server(model_path)?;
        self.model_id = Some(model_id.to_string());
        Ok(())
    }

    pub fn unload_model(&mut self) {
        self.model_id = None;
        if let Some(child) = self.child_process.take() {
            match child.lock() {
                Ok(mut shared_child) => {
                    let _ = shared_child.kill();
                }
                Err(poisoned) => {
                    let mut shared_child = poisoned.into_inner();
                    let _ = shared_child.kill();
                }
            }
        }
        self.stdin = None;
        self.stdout = None;
    }

    fn start_server(
        &mut self,
        model_path: &Path,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let script_path = resolve_server_script_path()?;
        let (python_cmd, python_args) = get_python_command()?;

        let mut cmd = Command::new(&python_cmd);
        for arg in &python_args {
            cmd.arg(arg);
        }

        let mut child = cmd
            .env("PYTHONDONTWRITEBYTECODE", "1")
            .env("HF_ENDPOINT", QWEN35_DEFAULT_ENDPOINT)
            .env("HANDY_QWEN35_MODEL_PATH", model_path)
            .envs({
                let mut vars = Vec::new();
                if self.max_threads > 0 {
                    let thread_count = self.max_threads.to_string();
                    vars.push(("OMP_NUM_THREADS", thread_count.clone()));
                    vars.push(("VECLIB_MAXIMUM_THREADS", thread_count.clone()));
                    vars.push(("MKL_NUM_THREADS", thread_count.clone()));
                    vars.push(("NUMEXPR_NUM_THREADS", thread_count));
                }
                vars
            })
            .arg("-u")
            .arg(script_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3.5 post server stdin unavailable",
            )) as Box<dyn std::error::Error>
        })?;

        let stdout = child.stdout.take().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3.5 post server stdout unavailable",
            )) as Box<dyn std::error::Error>
        })?;

        let (ready_tx, ready_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => {
                        let _ = ready_tx.send(Err(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Qwen3.5 post server stdout closed before READY",
                        )));
                        return;
                    }
                    Ok(_) => {
                        let trimmed = line.trim().to_string();
                        if trimmed == "READY" {
                            let _ = ready_tx.send(Ok(reader));
                            return;
                        }
                        if trimmed.starts_with("FAILED:") {
                            let _ = ready_tx
                                .send(Err(std::io::Error::new(std::io::ErrorKind::Other, trimmed)));
                            return;
                        }
                        debug!("Qwen3.5 post server startup: {}", trimmed);
                    }
                    Err(err) => {
                        let _ = ready_tx.send(Err(err));
                        return;
                    }
                }
            }
        });

        let reader = match ready_rx
            .recv_timeout(std::time::Duration::from_secs(self.startup_timeout_secs))
        {
            Ok(Ok(reader)) => reader,
            Ok(Err(err)) => {
                let _ = child.kill();
                let mut stderr = String::new();
                if let Some(mut stderr_pipe) = child.stderr.take() {
                    let _ = stderr_pipe.read_to_string(&mut stderr);
                }
                if !stderr.trim().is_empty() {
                    warn!("Qwen3.5 post server startup stderr: {}", stderr.trim());
                }
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Qwen3.5 post server startup failed: {}", err),
                )));
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                let _ = child.kill();
                let mut stderr = String::new();
                if let Some(mut stderr_pipe) = child.stderr.take() {
                    let _ = stderr_pipe.read_to_string(&mut stderr);
                }
                if !stderr.trim().is_empty() {
                    warn!(
                        "Qwen3.5 post server startup timeout stderr: {}",
                        stderr.trim()
                    );
                }
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::TimedOut,
                    format!(
                        "Timed out waiting for Qwen3.5 post server READY ({}s)",
                        self.startup_timeout_secs
                    ),
                )));
            }
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                let _ = child.kill();
                return Err(Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    "Qwen3.5 post server readiness channel disconnected",
                )));
            }
        };

        self.child_process = Some(Arc::new(Mutex::new(child)));
        self.stdin = Some(Arc::new(Mutex::new(stdin)));
        self.stdout = Some(Arc::new(Mutex::new(reader)));
        Ok(())
    }

    pub fn process_text(
        &mut self,
        text: &str,
        system_prompt: &str,
        template_id: Option<&str>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        repetition_context_size: usize,
    ) -> std::result::Result<String, Box<dyn std::error::Error>> {
        if self.model_id.is_none() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3.5 post model is not loaded",
            )));
        }

        let stdin = self.stdin.as_ref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3.5 post server stdin is missing",
            )) as Box<dyn std::error::Error>
        })?;

        let stdout = self.stdout.as_ref().ok_or_else(|| {
            Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "Qwen3.5 post server stdout is missing",
            )) as Box<dyn std::error::Error>
        })?;

        let req = Qwen35PostRequest {
            text,
            system_prompt,
            template_id,
            max_tokens,
            temperature,
            top_p,
            repetition_penalty,
            repetition_context_size,
        };
        let payload = serde_json::to_string(&req)?;

        {
            let mut stdin_guard = stdin.lock().map_err(|e| {
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("Failed to lock Qwen3.5 stdin: {}", e),
                )) as Box<dyn std::error::Error>
            })?;
            stdin_guard.write_all(payload.as_bytes())?;
            stdin_guard.write_all(b"\n")?;
            stdin_guard.flush()?;
        }

        let stdout_reader = Arc::clone(stdout);
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            let mut output_line = String::new();
            let result: std::result::Result<String, std::io::Error> = (|| {
                let mut stdout_guard = stdout_reader.lock().map_err(|e| {
                    std::io::Error::new(
                        std::io::ErrorKind::Other,
                        format!("Failed to lock Qwen3.5 stdout: {}", e),
                    )
                })?;
                stdout_guard.read_line(&mut output_line)?;
                Ok(output_line)
            })();
            let _ = tx.send(result);
        });

        let output_line =
            match rx.recv_timeout(std::time::Duration::from_secs(self.inference_timeout_secs)) {
                Ok(Ok(line)) => line,
                Ok(Err(err)) => return Err(Box::new(err)),
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    self.unload_model();
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        format!(
                            "Timed out waiting for Qwen3.5 post server response ({}s)",
                            self.inference_timeout_secs
                        ),
                    )));
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    self.unload_model();
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "Qwen3.5 post server response channel disconnected",
                    )));
                }
            };

        if output_line.trim().is_empty() {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "Empty response from Qwen3.5 post server",
            )));
        }

        let response: Qwen35PostResponse =
            serde_json::from_str(output_line.trim()).map_err(|e| {
                warn!("Raw Qwen3.5 post response: {}", output_line.trim());
                Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to parse Qwen3.5 post response: {}", e),
                )) as Box<dyn std::error::Error>
            })?;

        if let Some(error) = response.error {
            return Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                format!("Qwen3.5 post server error: {}", error),
            )));
        }

        let result = response.text.unwrap_or_default().trim().to_string();
        Ok(result)
    }
}
