use crate::managers::qwen35_post_engine::{
    get_qwen35_python_command, init_qwen35_python_path, QWEN35_DEFAULT_ENDPOINT,
};
use crate::settings::{get_settings, write_settings};
use anyhow::Result;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use specta::Type;
use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter, Manager};

const LOCAL_PROVIDER_ID: &str = "local-qwen35";
const DEFAULT_LOCAL_MODEL_ID: &str = "qwen35-optiq-2b";
const QWEN35_FALLBACK_ENDPOINT: &str = "https://huggingface.co";
const DEFAULT_PYPI_INDEX_URL: &str = "https://pypi.tuna.tsinghua.edu.cn/simple";
const FALLBACK_PYPI_INDEX_URL: &str = "https://pypi.org/simple";

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| default.to_string())
}

fn uv_index_candidates() -> Vec<String> {
    let mut candidates = Vec::new();
    let mut push_unique = |value: Option<String>| {
        if let Some(value) = value {
            let trimmed = value.trim().to_string();
            if trimmed.is_empty() {
                return;
            }
            if !candidates.iter().any(|item| item == &trimmed) {
                candidates.push(trimmed);
            }
        }
    };

    push_unique(std::env::var("PIP_INDEX_URL").ok());
    push_unique(std::env::var("UV_INDEX_URL").ok());
    push_unique(std::env::var("UV_DEFAULT_INDEX").ok());
    push_unique(Some(DEFAULT_PYPI_INDEX_URL.to_string()));
    push_unique(Some(FALLBACK_PYPI_INDEX_URL.to_string()));
    candidates
}

fn apply_uv_runtime_env_with_index(cmd: &mut std::process::Command, index_url: &str) {
    cmd.env(
        "HF_ENDPOINT",
        env_or_default("HF_ENDPOINT", QWEN35_DEFAULT_ENDPOINT),
    );
    cmd.env("PIP_INDEX_URL", index_url);
    cmd.env("UV_INDEX_URL", index_url);
    cmd.env("UV_DEFAULT_INDEX", index_url);
    cmd.env("PIP_DISABLE_PIP_VERSION_CHECK", "1");
}

fn apply_uv_runtime_env(cmd: &mut std::process::Command) {
    let index = uv_index_candidates()
        .into_iter()
        .next()
        .unwrap_or_else(|| DEFAULT_PYPI_INDEX_URL.to_string());
    apply_uv_runtime_env_with_index(cmd, &index);
}

fn detect_compatible_system_python() -> Option<String> {
    let output = std::process::Command::new("python3")
        .arg("-c")
        .arg("import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}|{sys.executable}\")")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.trim();
    let mut parts = line.split('|');
    let version = parts.next()?.trim();
    let executable = parts.next()?.trim();
    if executable.is_empty() {
        return None;
    }

    let mut version_parts = version.split('.');
    let major = version_parts.next()?.parse::<u32>().ok()?;
    let minor = version_parts.next()?.parse::<u32>().ok()?;
    if major > 3 || (major == 3 && minor >= 11) {
        Some(executable.to_string())
    } else {
        None
    }
}

fn resolve_python_spec_for_uv() -> String {
    if let Some(python_path) = std::env::var("HANDY_QWEN_PYTHON")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
    {
        info!("Using HANDY_QWEN_PYTHON: {}", python_path);
        return python_path;
    }

    if let Some(system_python) = detect_compatible_system_python() {
        info!(
            "Using compatible system Python for Qwen3.5 runtime: {}",
            system_python
        );
        return system_python;
    }

    info!("No compatible system Python detected, falling back to uv-managed Python 3.11.");
    "3.11".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PostProcessModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub repo_id: String,
    pub size_mb: u64,
    pub tier: String,
    pub is_recommended: bool,
    pub is_experimental: bool,
    pub is_downloaded: bool,
    pub is_downloading: bool,
    pub partial_size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct PostProcessDownloadProgress {
    pub model_id: String,
    pub downloaded: u64,
    pub total: u64,
    pub percentage: f64,
}

#[derive(Debug, Deserialize)]
struct ModelInfoFile {
    filename: String,
    size: u64,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    revision: String,
    files: Vec<ModelInfoFile>,
    total: u64,
}

struct ResolvedModelInfo {
    endpoint: String,
    info: ModelInfo,
}

#[derive(Debug, Deserialize, Serialize)]
struct DownloadManifest {
    files: Vec<DownloadManifestFile>,
}

#[derive(Debug, Deserialize, Serialize)]
struct DownloadManifestFile {
    filename: String,
    size: u64,
}

struct FileDownloadResult {
    downloaded: u64,
    total: u64,
    cancelled: bool,
}

struct DownloadCleanup<'a> {
    available_models: &'a Mutex<HashMap<String, PostProcessModelInfo>>,
    cancel_flags: &'a Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
    model_id: String,
    disarmed: bool,
}

impl<'a> Drop for DownloadCleanup<'a> {
    fn drop(&mut self) {
        if self.disarmed {
            return;
        }

        {
            let mut models = self.available_models.lock().unwrap();
            if let Some(model) = models.get_mut(self.model_id.as_str()) {
                model.is_downloading = false;
                model.partial_size = 0;
            }
        }
        self.cancel_flags.lock().unwrap().remove(&self.model_id);
    }
}

pub struct PostProcessModelManager {
    app_handle: AppHandle,
    models_dir: PathBuf,
    runtime_dir: PathBuf,
    available_models: Mutex<HashMap<String, PostProcessModelInfo>>,
    cancel_flags: Arc<Mutex<HashMap<String, Arc<AtomicBool>>>>,
}

impl PostProcessModelManager {
    pub fn new(app_handle: &AppHandle) -> Result<Self> {
        let app_data_dir = crate::portable::app_data_dir(app_handle)
            .map_err(|e| anyhow::anyhow!("Failed to get app data dir: {}", e))?;

        let models_dir = app_data_dir.join("post_process_models");
        if !models_dir.exists() {
            fs::create_dir_all(&models_dir)?;
        }

        let runtime_dir = app_data_dir.join("qwen35_post_mlx");
        if !runtime_dir.exists() {
            fs::create_dir_all(&runtime_dir)?;
        }

        let mut available_models = HashMap::new();
        available_models.insert(
            "qwen35-optiq-0.8b".to_string(),
            PostProcessModelInfo {
                id: "qwen35-optiq-0.8b".to_string(),
                name: "Qwen3.5-0.8B-OptiQ-4bit".to_string(),
                description: "Fast tier for local post-processing".to_string(),
                repo_id: "mlx-community/Qwen3.5-0.8B-OptiQ-4bit".to_string(),
                size_mb: 620,
                tier: "Fast".to_string(),
                is_recommended: false,
                is_experimental: false,
                is_downloaded: false,
                is_downloading: false,
                partial_size: 0,
            },
        );
        available_models.insert(
            "qwen35-optiq-2b".to_string(),
            PostProcessModelInfo {
                id: "qwen35-optiq-2b".to_string(),
                name: "Qwen3.5-2B-OptiQ-4bit".to_string(),
                description: "Balanced tier for local post-processing".to_string(),
                repo_id: "mlx-community/Qwen3.5-2B-OptiQ-4bit".to_string(),
                size_mb: 1452,
                tier: "Balance".to_string(),
                is_recommended: true,
                is_experimental: false,
                is_downloaded: false,
                is_downloading: false,
                partial_size: 0,
            },
        );
        available_models.insert(
            "qwen35-optiq-4b".to_string(),
            PostProcessModelInfo {
                id: "qwen35-optiq-4b".to_string(),
                name: "Qwen3.5-4B-OptiQ-4bit".to_string(),
                description: "Quality tier for local post-processing".to_string(),
                repo_id: "mlx-community/Qwen3.5-4B-OptiQ-4bit".to_string(),
                size_mb: 2968,
                tier: "Quality".to_string(),
                is_recommended: false,
                is_experimental: false,
                is_downloaded: false,
                is_downloading: false,
                partial_size: 0,
            },
        );
        available_models.insert(
            "qwen35-optiq-9b".to_string(),
            PostProcessModelInfo {
                id: "qwen35-optiq-9b".to_string(),
                name: "Qwen3.5-9B-OptiQ-4bit".to_string(),
                description: "Experimental high-memory tier".to_string(),
                repo_id: "mlx-community/Qwen3.5-9B-OptiQ-4bit".to_string(),
                size_mb: 6200,
                tier: "Experimental".to_string(),
                is_recommended: false,
                is_experimental: true,
                is_downloaded: false,
                is_downloading: false,
                partial_size: 0,
            },
        );

        let manager = Self {
            app_handle: app_handle.clone(),
            models_dir,
            runtime_dir,
            available_models: Mutex::new(available_models),
            cancel_flags: Arc::new(Mutex::new(HashMap::new())),
        };

        manager.update_download_status()?;
        Ok(manager)
    }

    pub fn get_available_models(&self) -> Vec<PostProcessModelInfo> {
        let models = self.available_models.lock().unwrap();
        models.values().cloned().collect()
    }

    pub fn get_model_info(&self, model_id: &str) -> Option<PostProcessModelInfo> {
        let models = self.available_models.lock().unwrap();
        models.get(model_id).cloned()
    }

    pub fn get_model_ids(&self) -> Vec<String> {
        let models = self.available_models.lock().unwrap();
        let mut ids = models.keys().cloned().collect::<Vec<_>>();
        ids.sort();
        ids
    }

    pub fn ensure_qwen35_python_runtime_ready(&self) -> Result<()> {
        init_qwen35_python_path(&self.app_handle)
            .map_err(|e| anyhow::anyhow!("Failed to initialize Qwen3.5 python path: {}", e))?;

        let bundled_runtime_dir = self
            .app_handle
            .path()
            .resolve(
                "resources/qwen35_post_mlx",
                tauri::path::BaseDirectory::Resource,
            )
            .map_err(|e| anyhow::anyhow!("Failed to resolve Qwen3.5 runtime resources: {}", e))?;

        if !bundled_runtime_dir.exists() {
            return Err(anyhow::anyhow!(
                "Bundled Qwen3.5 runtime resources not found: {}",
                bundled_runtime_dir.display()
            ));
        }

        Self::sync_dir_if_newer(&bundled_runtime_dir, &self.runtime_dir)?;

        let uv_bin = self.runtime_dir.join("uv");
        if !uv_bin.exists() {
            let bundled_uv_archive = self
                .app_handle
                .path()
                .resolve(
                    "resources/qwen3_asr_mlx/uv.tar.gz",
                    tauri::path::BaseDirectory::Resource,
                )
                .map_err(|e| anyhow::anyhow!("Failed to resolve bundled uv archive: {}", e))?;
            if !bundled_uv_archive.exists() {
                return Err(anyhow::anyhow!(
                    "Bundled uv archive missing: {}",
                    bundled_uv_archive.display()
                ));
            }
            let target_uv_archive = self.runtime_dir.join("uv.tar.gz");
            fs::copy(&bundled_uv_archive, &target_uv_archive)?;

            let archive_file = File::open(&target_uv_archive)?;
            let decoder = flate2::read::GzDecoder::new(archive_file);
            let mut archive = tar::Archive::new(decoder);
            archive.unpack(&self.runtime_dir)?;
        }

        if !uv_bin.exists() {
            return Err(anyhow::anyhow!(
                "uv binary not found after extraction: {}",
                uv_bin.display()
            ));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = fs::metadata(&uv_bin)?.permissions();
            perms.set_mode(0o755);
            fs::set_permissions(&uv_bin, perms)?;
        }

        let venv_dir = self.runtime_dir.join(".venv");
        let venv_python = venv_dir.join("bin/python3");
        let python_spec = resolve_python_spec_for_uv();
        if !venv_python.exists() {
            let mut venv_cmd = std::process::Command::new(&uv_bin);
            apply_uv_runtime_env(&mut venv_cmd);
            let status = venv_cmd
                .arg("venv")
                .arg("--clear")
                .arg("--python")
                .arg(&python_spec)
                .arg(&venv_dir)
                .status()
                .map_err(|e| anyhow::anyhow!("Failed to execute uv venv: {}", e))?;
            if !status.success() {
                return Err(anyhow::anyhow!("uv venv failed with status: {}", status));
            }
        }

        let requirements = self.runtime_dir.join("requirements.txt");
        if !requirements.exists() {
            return Err(anyhow::anyhow!(
                "requirements.txt missing in runtime dir: {}",
                requirements.display()
            ));
        }

        let requirements_hash = Self::compute_sha256(&requirements)?;
        let marker = self.runtime_dir.join(".deps.hash");
        let marker_hash = fs::read_to_string(&marker).unwrap_or_default();

        if marker_hash.trim() != requirements_hash {
            info!(
                "Syncing Qwen3.5 runtime python deps from {}",
                requirements.display()
            );

            let mut install_ok = false;
            let mut install_errors = Vec::new();
            for index_url in uv_index_candidates() {
                info!(
                    "Running uv pip install for Qwen3.5 runtime using index {}",
                    index_url
                );
                let mut install_cmd = std::process::Command::new(&uv_bin);
                apply_uv_runtime_env_with_index(&mut install_cmd, &index_url);
                let status = install_cmd
                    .arg("pip")
                    .arg("install")
                    .arg("--python")
                    .arg(&venv_python)
                    .arg("-r")
                    .arg(&requirements)
                    .status()
                    .map_err(|e| anyhow::anyhow!("Failed to execute uv pip install: {}", e))?;

                if status.success() {
                    install_ok = true;
                    break;
                }
                install_errors.push(format!("{} (status: {})", index_url, status));
            }
            if !install_ok {
                return Err(anyhow::anyhow!(
                    "uv pip install failed across all index candidates: {}",
                    install_errors.join(" | ")
                ));
            }

            fs::write(&marker, requirements_hash)?;
        }

        init_qwen35_python_path(&self.app_handle)
            .map_err(|e| anyhow::anyhow!("Failed to refresh Qwen3.5 python path: {}", e))?;
        Ok(())
    }

    fn normalize_endpoint(raw: &str) -> Option<String> {
        let trimmed = raw.trim().trim_end_matches('/');
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    }

    fn candidate_endpoints(&self, preferred: Option<&str>) -> Vec<String> {
        let mut endpoints = Vec::new();
        let mut push_unique = |endpoint: Option<String>| {
            if let Some(endpoint) = endpoint {
                if !endpoints.iter().any(|item| item == &endpoint) {
                    endpoints.push(endpoint);
                }
            }
        };

        push_unique(preferred.and_then(Self::normalize_endpoint));
        push_unique(
            std::env::var("HF_ENDPOINT")
                .ok()
                .and_then(|v| Self::normalize_endpoint(&v)),
        );
        push_unique(Self::normalize_endpoint(QWEN35_DEFAULT_ENDPOINT));
        push_unique(Self::normalize_endpoint(QWEN35_FALLBACK_ENDPOINT));

        endpoints
    }

    fn resolve_model_info_for_endpoint(&self, repo_id: &str, endpoint: &str) -> Result<ModelInfo> {
        let (python_cmd, python_args) = get_qwen35_python_command().map_err(|e| {
            anyhow::anyhow!(
                "Failed to resolve Python command for Qwen3.5 runtime: {}",
                e
            )
        })?;

        let script_path = self.runtime_dir.join("qwen35_model_info.py");
        if !script_path.exists() {
            return Err(anyhow::anyhow!(
                "Qwen3.5 model info script not found: {}",
                script_path.display()
            ));
        }

        let mut cmd = std::process::Command::new(&python_cmd);
        cmd.env("PYTHONDONTWRITEBYTECODE", "1");
        cmd.env("HF_ENDPOINT", endpoint);
        for arg in &python_args {
            cmd.arg(arg);
        }

        let output = cmd
            .arg("-B")
            .arg(&script_path)
            .arg(repo_id)
            .arg(endpoint)
            .output()
            .map_err(|e| anyhow::anyhow!("Failed to execute model info script: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
            let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
            return Err(anyhow::anyhow!(
                "qwen35_model_info.py failed with status {}. stderr: {}. stdout: {}",
                output.status,
                if stderr.is_empty() {
                    "<empty>"
                } else {
                    stderr.as_str()
                },
                if stdout.is_empty() {
                    "<empty>"
                } else {
                    stdout.as_str()
                }
            ));
        }

        let stdout = String::from_utf8(output.stdout)
            .map_err(|e| anyhow::anyhow!("Invalid UTF-8 from model info stdout: {}", e))?;
        let trimmed = stdout.trim();
        if trimmed.is_empty() {
            return Err(anyhow::anyhow!(
                "qwen35_model_info.py returned empty stdout"
            ));
        }

        if let Ok(info) = serde_json::from_str::<ModelInfo>(trimmed) {
            if info.files.is_empty() {
                return Err(anyhow::anyhow!(
                    "Model info for {} contains no files",
                    repo_id
                ));
            }
            return Ok(info);
        }

        if let Some(last_line) = trimmed.lines().rev().find(|line| !line.trim().is_empty()) {
            let info = serde_json::from_str::<ModelInfo>(last_line.trim())?;
            if info.files.is_empty() {
                return Err(anyhow::anyhow!(
                    "Model info for {} contains no files",
                    repo_id
                ));
            }
            return Ok(info);
        }

        Err(anyhow::anyhow!("Unable to parse model info output"))
    }

    fn resolve_model_info_with_fallback(&self, repo_id: &str) -> Result<ResolvedModelInfo> {
        let mut errors = Vec::new();

        for endpoint in self.candidate_endpoints(None) {
            match self.resolve_model_info_for_endpoint(repo_id, &endpoint) {
                Ok(info) => {
                    info!(
                        "Resolved model info for {} via endpoint {}",
                        repo_id, endpoint
                    );
                    return Ok(ResolvedModelInfo { endpoint, info });
                }
                Err(err) => {
                    warn!(
                        "Failed to resolve model info for {} via endpoint {}: {}",
                        repo_id, endpoint, err
                    );
                    errors.push(format!("{} => {}", endpoint, err));
                }
            }
        }

        Err(anyhow::anyhow!(
            "Unable to resolve model info for {} from all endpoints: {}",
            repo_id,
            errors.join(" | ")
        ))
    }

    pub async fn download_model(&self, model_id: &str) -> Result<()> {
        let model = self
            .get_model_info(model_id)
            .ok_or_else(|| anyhow::anyhow!("Post-process model not found: {}", model_id))?;

        {
            let mut models = self.available_models.lock().unwrap();
            if let Some(item) = models.get_mut(model_id) {
                item.is_downloading = true;
                item.partial_size = 0;
            }
        }

        self.ensure_qwen35_python_runtime_ready()?;

        if self.check_model_cached(model_id) {
            let mut models = self.available_models.lock().unwrap();
            if let Some(item) = models.get_mut(model_id) {
                item.is_downloading = false;
                item.is_downloaded = true;
                item.partial_size = 0;
            }
            let _ = self
                .app_handle
                .emit("post-process-model-download-complete", model_id.to_string());
            return Ok(());
        }

        let cancel_flag = Arc::new(AtomicBool::new(false));
        {
            let mut flags = self.cancel_flags.lock().unwrap();
            flags.insert(model_id.to_string(), cancel_flag.clone());
        }

        let mut cleanup = DownloadCleanup {
            available_models: &self.available_models,
            cancel_flags: &self.cancel_flags,
            model_id: model_id.to_string(),
            disarmed: false,
        };

        let resolved_info = self.resolve_model_info_with_fallback(&model.repo_id)?;
        let model_info = resolved_info.info;
        let download_endpoints = self.candidate_endpoints(Some(&resolved_info.endpoint));
        let model_dir = self.local_model_dir(model_id)?;
        fs::create_dir_all(&model_dir)?;

        let client = reqwest::Client::new();
        let mut total_bytes = if model_info.total > 0 {
            model_info.total
        } else {
            model_info.files.iter().map(|f| f.size).sum()
        };

        let mut downloaded_bytes = 0u64;
        for file in &model_info.files {
            let path = model_dir.join(&file.filename);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            let existing = path.metadata().map(|m| m.len()).unwrap_or(0);
            let expected = if file.size > 0 { file.size } else { existing };
            downloaded_bytes += existing.min(expected);
            if total_bytes < downloaded_bytes {
                total_bytes = downloaded_bytes;
            }
        }

        let emit_progress = |downloaded: u64, total: u64| {
            let percentage = if total > 0 {
                (downloaded as f64 / total as f64) * 100.0
            } else {
                0.0
            };
            {
                let mut models = self.available_models.lock().unwrap();
                if let Some(item) = models.get_mut(model_id) {
                    item.partial_size = downloaded;
                }
            }
            let _ = self.app_handle.emit(
                "post-process-model-download-progress",
                PostProcessDownloadProgress {
                    model_id: model_id.to_string(),
                    downloaded,
                    total,
                    percentage,
                },
            );
        };

        emit_progress(downloaded_bytes, total_bytes);

        let revision = model_info.revision.clone();
        for file in &model_info.files {
            let path = model_dir.join(&file.filename);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }

            let existing = path.metadata().map(|m| m.len()).unwrap_or(0);
            let previous_expected = if file.size > 0 { file.size } else { existing };
            let previous_counted = existing.min(previous_expected);

            if file.size > 0 && existing >= file.size {
                continue;
            }

            let result = self
                .download_file_from_endpoints(
                    &client,
                    &download_endpoints,
                    &model.repo_id,
                    &revision,
                    &file.filename,
                    &path,
                    &cancel_flag,
                    &mut |file_downloaded, file_total| {
                        let effective_total = if total_bytes == 0 {
                            file_total
                        } else {
                            total_bytes
                        };
                        let aggregate_downloaded =
                            downloaded_bytes.saturating_sub(previous_counted) + file_downloaded;
                        let aggregate_total = effective_total.max(aggregate_downloaded);
                        emit_progress(aggregate_downloaded.min(aggregate_total), aggregate_total);
                    },
                )
                .await?;

            if result.cancelled {
                info!("Post-process model download cancelled: {}", model_id);
                return Ok(());
            }

            if file.size > 0 {
                let final_size = path.metadata()?.len();
                if final_size < file.size {
                    return Err(anyhow::anyhow!(
                        "Incomplete file {}: expected {}, got {}",
                        file.filename,
                        file.size,
                        final_size
                    ));
                }
            }

            let discovered_total = if file.size > 0 {
                file.size
            } else {
                result.total
            };
            if discovered_total > previous_expected {
                total_bytes += discovered_total - previous_expected;
            }

            downloaded_bytes =
                downloaded_bytes.saturating_sub(previous_counted) + result.downloaded;
            if total_bytes < downloaded_bytes {
                total_bytes = downloaded_bytes;
            }
            emit_progress(downloaded_bytes, total_bytes);
        }

        let manifest = DownloadManifest {
            files: model_info
                .files
                .iter()
                .map(|f| DownloadManifestFile {
                    filename: f.filename.clone(),
                    size: f.size,
                })
                .collect(),
        };
        fs::write(
            model_dir.join(".download-manifest.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;

        if !self.check_model_cached(model_id) {
            return Err(anyhow::anyhow!("Post-process model verification failed"));
        }

        emit_progress(total_bytes, total_bytes);

        cleanup.disarmed = true;
        {
            let mut models = self.available_models.lock().unwrap();
            if let Some(item) = models.get_mut(model_id) {
                item.is_downloading = false;
                item.is_downloaded = true;
                item.partial_size = 0;
            }
        }
        self.cancel_flags.lock().unwrap().remove(model_id);

        let _ = self
            .app_handle
            .emit("post-process-model-download-complete", model_id.to_string());
        Ok(())
    }

    pub fn delete_model(&self, model_id: &str) -> Result<()> {
        let model_dir = self.local_model_dir(model_id)?;
        if model_dir.exists() {
            fs::remove_dir_all(&model_dir)?;
        }

        // If the deleted local model is currently selected for local provider,
        // fallback to default so post-processing remains usable.
        let mut settings = get_settings(&self.app_handle);
        if settings.post_process_provider_id == LOCAL_PROVIDER_ID {
            let selected = settings
                .post_process_models
                .get(LOCAL_PROVIDER_ID)
                .cloned()
                .unwrap_or_default();
            if selected == model_id {
                settings.post_process_models.insert(
                    LOCAL_PROVIDER_ID.to_string(),
                    DEFAULT_LOCAL_MODEL_ID.to_string(),
                );
                write_settings(&self.app_handle, settings);
            }
        }

        self.update_download_status()?;
        let _ = self
            .app_handle
            .emit("post-process-model-deleted", model_id.to_string());
        Ok(())
    }

    pub fn cancel_download(&self, model_id: &str) -> Result<()> {
        let flags = self.cancel_flags.lock().unwrap();
        if let Some(flag) = flags.get(model_id) {
            flag.store(true, Ordering::Relaxed);
            Ok(())
        } else {
            Err(anyhow::anyhow!("No active download for {}", model_id))
        }
    }

    pub fn resolve_local_model_dir(&self, model_id: &str) -> Result<PathBuf> {
        self.local_model_dir(model_id)
    }

    pub fn check_model_cached(&self, model_id: &str) -> bool {
        let model_dir = match self.local_model_dir(model_id) {
            Ok(path) => path,
            Err(_) => return false,
        };
        Self::is_model_dir_complete(&model_dir)
    }

    fn update_download_status(&self) -> Result<()> {
        let model_ids: Vec<String> = {
            let models = self.available_models.lock().unwrap();
            models.keys().cloned().collect()
        };

        let statuses: HashMap<String, bool> = {
            model_ids
                .iter()
                .map(|model_id| (model_id.clone(), self.check_model_cached(model_id)))
                .collect()
        };

        let mut models = self.available_models.lock().unwrap();
        for model in models.values_mut() {
            model.is_downloaded = *statuses.get(&model.id).unwrap_or(&false);
            if !model.is_downloading {
                model.partial_size = 0;
            }
        }

        Ok(())
    }

    fn repo_id_for_model(&self, model_id: &str) -> Result<String> {
        let models = self.available_models.lock().unwrap();
        let model = models
            .get(model_id)
            .ok_or_else(|| anyhow::anyhow!("Unknown post-process model: {}", model_id))?;
        Ok(model.repo_id.clone())
    }

    fn local_model_dir(&self, model_id: &str) -> Result<PathBuf> {
        let repo_id = self.repo_id_for_model(model_id)?;
        Ok(self.models_dir.join(repo_id.replace('/', "--")))
    }

    fn is_model_dir_complete(model_dir: &Path) -> bool {
        if !model_dir.exists() || !model_dir.is_dir() {
            return false;
        }

        let manifest_path = model_dir.join(".download-manifest.json");
        if manifest_path.exists() {
            let manifest_ok = (|| -> Result<bool> {
                let manifest_content = fs::read_to_string(&manifest_path)?;
                let manifest: DownloadManifest = serde_json::from_str(&manifest_content)?;
                if manifest.files.is_empty() {
                    return Ok(false);
                }

                for file in manifest.files {
                    let file_path = model_dir.join(&file.filename);
                    if !file_path.exists() || !file_path.is_file() {
                        return Ok(false);
                    }
                    if file.size > 0 {
                        let actual = file_path.metadata()?.len();
                        if actual < file.size {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            })();

            if let Ok(true) = manifest_ok {
                return true;
            }
        }

        // Fallback for older downloads without manifest.
        let has_config = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("config.json")
        });
        let has_tokenizer = Self::find_file_recursive(model_dir, &|p| {
            p.file_name().and_then(|n| n.to_str()) == Some("tokenizer.json")
        });
        let has_weights = Self::find_file_recursive(model_dir, &|p| {
            p.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
        });

        has_config && has_tokenizer && has_weights
    }

    fn find_file_recursive(dir: &Path, predicate: &impl Fn(&Path) -> bool) -> bool {
        let entries = match fs::read_dir(dir) {
            Ok(entries) => entries,
            Err(_) => return false,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            let file_type = match entry.file_type() {
                Ok(ft) => ft,
                Err(_) => continue,
            };

            if file_type.is_file() {
                if predicate(&path) {
                    return true;
                }
            } else if file_type.is_dir() && Self::find_file_recursive(&path, predicate) {
                return true;
            }
        }

        false
    }

    fn compute_sha256(path: &Path) -> Result<String> {
        let mut file = File::open(path)?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 65536];
        loop {
            let n = file.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn sync_dir_if_newer(src: &Path, dst: &Path) -> Result<()> {
        if !src.exists() || !src.is_dir() {
            return Ok(());
        }
        fs::create_dir_all(dst)?;

        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let entry_name = entry.file_name();
            let entry_name_str = entry_name.to_string_lossy();
            if entry_name_str == ".venv" || entry_name_str == "__pycache__" {
                continue;
            }
            let src_path = entry.path();
            let dst_path = dst.join(entry_name);
            let metadata = entry.metadata()?;

            if metadata.is_dir() {
                Self::sync_dir_if_newer(&src_path, &dst_path)?;
            } else if metadata.is_file() {
                let should_copy = if !dst_path.exists() {
                    true
                } else {
                    let src_mtime = src_path.metadata()?.modified()?;
                    let dst_mtime = dst_path.metadata()?.modified()?;
                    src_mtime > dst_mtime
                };

                if should_copy {
                    if let Some(parent) = dst_path.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::copy(&src_path, &dst_path)?;
                }
            }
        }

        Ok(())
    }

    fn build_file_url(
        endpoint: &str,
        repo_id: &str,
        revision: &str,
        filename: &str,
    ) -> Result<reqwest::Url> {
        let mut url = reqwest::Url::parse(endpoint)?;
        {
            let mut segments = url
                .path_segments_mut()
                .map_err(|_| anyhow::anyhow!("Endpoint cannot be a base URL"))?;
            for s in repo_id.split('/') {
                segments.push(s);
            }
            segments.push("resolve");
            segments.push(revision);
            for s in filename.split('/') {
                segments.push(s);
            }
        }
        Ok(url)
    }

    async fn download_file_from_endpoints<F>(
        &self,
        client: &reqwest::Client,
        endpoints: &[String],
        repo_id: &str,
        revision: &str,
        filename: &str,
        target_path: &Path,
        cancel_flag: &Arc<AtomicBool>,
        on_progress: &mut F,
    ) -> Result<FileDownloadResult>
    where
        F: FnMut(u64, u64),
    {
        let mut errors = Vec::new();

        for endpoint in endpoints {
            if cancel_flag.load(Ordering::Relaxed) {
                return Ok(FileDownloadResult {
                    downloaded: target_path.metadata().map(|m| m.len()).unwrap_or(0),
                    total: 0,
                    cancelled: true,
                });
            }

            let url = Self::build_file_url(endpoint, repo_id, revision, filename)?.to_string();
            match self
                .download_file_with_resume(client, &url, target_path, cancel_flag, |d, t| {
                    on_progress(d, t)
                })
                .await
            {
                Ok(result) => return Ok(result),
                Err(err) => {
                    warn!(
                        "Failed to download '{}' from endpoint {}: {}",
                        filename, endpoint, err
                    );
                    errors.push(format!("{} => {}", endpoint, err));
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to download '{}' from all endpoints: {}",
            filename,
            errors.join(" | ")
        ))
    }

    async fn download_file_with_resume<F>(
        &self,
        client: &reqwest::Client,
        url: &str,
        target_path: &Path,
        cancel_flag: &Arc<AtomicBool>,
        mut on_progress: F,
    ) -> Result<FileDownloadResult>
    where
        F: FnMut(u64, u64),
    {
        let mut resume_from = if target_path.exists() {
            target_path.metadata()?.len()
        } else {
            0
        };

        let mut request = client.get(url);
        if resume_from > 0 {
            request = request.header("Range", format!("bytes={}-", resume_from));
        }

        let mut response = request.send().await?;
        if resume_from > 0 && response.status() == reqwest::StatusCode::OK {
            warn!(
                "Server doesn't support range requests for {}, restarting download",
                url
            );
            drop(response);
            fs::remove_file(target_path)?;
            resume_from = 0;
            response = client.get(url).send().await?;
        }

        response.error_for_status_ref()?;

        let content_len = response.content_length().unwrap_or(0);
        let total_size = if response.status() == reqwest::StatusCode::PARTIAL_CONTENT {
            resume_from + content_len
        } else {
            content_len
        };

        let mut file = if resume_from > 0 {
            fs::OpenOptions::new().append(true).open(target_path)?
        } else {
            fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(target_path)?
        };

        let mut downloaded = resume_from;
        on_progress(downloaded, total_size);

        while let Some(chunk) = response.chunk().await? {
            if cancel_flag.load(Ordering::Relaxed) {
                return Ok(FileDownloadResult {
                    downloaded,
                    total: total_size,
                    cancelled: true,
                });
            }

            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            on_progress(downloaded, total_size);
        }

        file.flush()?;

        if total_size > 0 && downloaded < total_size {
            return Err(anyhow::anyhow!(
                "Incomplete download: expected {} bytes, got {}",
                total_size,
                downloaded
            ));
        }

        Ok(FileDownloadResult {
            downloaded,
            total: total_size,
            cancelled: false,
        })
    }
}
