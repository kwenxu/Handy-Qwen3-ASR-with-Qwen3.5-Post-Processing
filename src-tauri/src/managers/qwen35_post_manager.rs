use crate::managers::post_process_model::PostProcessModelManager;
use crate::managers::qwen35_post_engine::Qwen35PostEngine;
use crate::settings::get_settings;
use anyhow::Result;
use log::{info, warn};
use std::sync::{Arc, Mutex};
use tauri::AppHandle;

pub struct Qwen35PostManager {
    _app_handle: AppHandle,
    model_manager: Arc<PostProcessModelManager>,
    engine: Mutex<Option<Qwen35PostEngine>>,
    current_model_id: Mutex<Option<String>>,
}

impl Qwen35PostManager {
    pub fn new(
        app_handle: &AppHandle,
        model_manager: Arc<PostProcessModelManager>,
    ) -> Result<Self> {
        Ok(Self {
            _app_handle: app_handle.clone(),
            model_manager,
            engine: Mutex::new(None),
            current_model_id: Mutex::new(None),
        })
    }

    pub fn unload_model(&self) {
        let mut engine_guard = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(engine) = engine_guard.as_mut() {
            engine.unload_model();
        }
        *engine_guard = None;

        let mut current_guard = self
            .current_model_id
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        *current_guard = None;
    }

    pub fn preload_model(&self, model_id: &str) -> Result<()> {
        let runtime_settings = get_settings(&self._app_handle);

        if !self.model_manager.check_model_cached(model_id) {
            return Err(anyhow::anyhow!(
                "Local post-process model '{}' is not downloaded",
                model_id
            ));
        }

        self.model_manager.ensure_qwen35_python_runtime_ready()?;
        let model_dir = self.model_manager.resolve_local_model_dir(model_id)?;

        let mut engine_guard = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        let mut current_guard = self
            .current_model_id
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if engine_guard.is_some() && current_guard.as_deref() == Some(model_id) {
            return Ok(());
        }

        if let Some(engine) = engine_guard.as_mut() {
            engine.unload_model();
        }

        let mut engine = Qwen35PostEngine::new();
        engine.configure_runtime(
            runtime_settings.qwen35_server_ready_timeout_sec,
            runtime_settings.qwen35_inference_timeout_sec,
            runtime_settings.qwen35_max_threads,
        );
        engine
            .load_model(model_id, &model_dir)
            .map_err(|e| anyhow::anyhow!("Failed to preload local Qwen3.5 model: {}", e))?;

        // One short warmup pass to reduce first real-call latency.
        if runtime_settings.qwen35_warmup_enabled {
            if let Err(err) = engine.process_text(
                "Translate this to English: 你好",
                "You are a strict transcription post-processor. Output final text only.",
                Some("warmup"),
                12,
                0.0,
                1.0,
                1.1,
                64,
            ) {
                warn!("Local Qwen3.5 warmup failed (non-fatal): {}", err);
            }
        }

        *engine_guard = Some(engine);
        *current_guard = Some(model_id.to_string());
        info!("Preloaded local Qwen3.5 post-process model: {}", model_id);
        Ok(())
    }

    pub fn process_text(
        &self,
        model_id: &str,
        text: &str,
        system_prompt: &str,
        template_id: Option<&str>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        repetition_penalty: f32,
        repetition_context_size: usize,
    ) -> Result<String> {
        let runtime_settings = get_settings(&self._app_handle);

        if !self.model_manager.check_model_cached(model_id) {
            return Err(anyhow::anyhow!(
                "Local post-process model '{}' is not downloaded",
                model_id
            ));
        }

        self.model_manager.ensure_qwen35_python_runtime_ready()?;
        let model_dir = self.model_manager.resolve_local_model_dir(model_id)?;

        let mut engine_guard = self.engine.lock().unwrap_or_else(|e| e.into_inner());
        let mut current_guard = self
            .current_model_id
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if let Some(engine) = engine_guard.as_mut() {
            engine.configure_runtime(
                runtime_settings.qwen35_server_ready_timeout_sec,
                runtime_settings.qwen35_inference_timeout_sec,
                runtime_settings.qwen35_max_threads,
            );
        }

        let needs_reload = engine_guard.is_none() || current_guard.as_deref() != Some(model_id);
        if needs_reload {
            if let Some(engine) = engine_guard.as_mut() {
                engine.unload_model();
            }

            let mut engine = Qwen35PostEngine::new();
            engine.configure_runtime(
                runtime_settings.qwen35_server_ready_timeout_sec,
                runtime_settings.qwen35_inference_timeout_sec,
                runtime_settings.qwen35_max_threads,
            );
            engine
                .load_model(model_id, &model_dir)
                .map_err(|e| anyhow::anyhow!("Failed to load local Qwen3.5 model: {}", e))?;

            *engine_guard = Some(engine);
            *current_guard = Some(model_id.to_string());
            info!("Loaded local Qwen3.5 post-process model: {}", model_id);
        }

        let engine = engine_guard.as_mut().ok_or_else(|| {
            anyhow::anyhow!("Qwen3.5 post-process engine is unavailable after initialization")
        })?;

        let result = engine
            .process_text(
                text,
                system_prompt,
                template_id,
                max_tokens,
                temperature,
                top_p,
                repetition_penalty,
                repetition_context_size,
            )
            .map_err(|e| anyhow::anyhow!("Local Qwen3.5 post-processing failed: {}", e))?;

        if result.trim().is_empty() {
            return Err(anyhow::anyhow!(
                "Local Qwen3.5 post-processing returned empty text"
            ));
        }

        Ok(result)
    }
}
