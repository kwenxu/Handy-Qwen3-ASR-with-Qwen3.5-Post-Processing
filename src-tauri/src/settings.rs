use log::{debug, warn};
use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use specta::Type;
use std::collections::HashMap;
use tauri::AppHandle;
use tauri_plugin_store::StoreExt;

pub const APPLE_INTELLIGENCE_PROVIDER_ID: &str = "apple_intelligence";
pub const APPLE_INTELLIGENCE_DEFAULT_MODEL_ID: &str = "Apple Intelligence";
pub const LOCAL_QWEN35_PROVIDER_ID: &str = "local-qwen35";
pub const LOCAL_QWEN35_DEFAULT_MODEL_ID: &str = "qwen35-optiq-0.8b";

#[derive(Serialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

// Custom deserializer to handle both old numeric format (1-5) and new string format ("trace", "debug", etc.)
impl<'de> Deserialize<'de> for LogLevel {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct LogLevelVisitor;

        impl<'de> Visitor<'de> for LogLevelVisitor {
            type Value = LogLevel;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a string or integer representing log level")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<LogLevel, E> {
                match value.to_lowercase().as_str() {
                    "trace" => Ok(LogLevel::Trace),
                    "debug" => Ok(LogLevel::Debug),
                    "info" => Ok(LogLevel::Info),
                    "warn" => Ok(LogLevel::Warn),
                    "error" => Ok(LogLevel::Error),
                    _ => Err(E::unknown_variant(
                        value,
                        &["trace", "debug", "info", "warn", "error"],
                    )),
                }
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> Result<LogLevel, E> {
                match value {
                    1 => Ok(LogLevel::Trace),
                    2 => Ok(LogLevel::Debug),
                    3 => Ok(LogLevel::Info),
                    4 => Ok(LogLevel::Warn),
                    5 => Ok(LogLevel::Error),
                    _ => Err(E::invalid_value(de::Unexpected::Unsigned(value), &"1-5")),
                }
            }
        }

        deserializer.deserialize_any(LogLevelVisitor)
    }
}

impl From<LogLevel> for tauri_plugin_log::LogLevel {
    fn from(level: LogLevel) -> Self {
        match level {
            LogLevel::Trace => tauri_plugin_log::LogLevel::Trace,
            LogLevel::Debug => tauri_plugin_log::LogLevel::Debug,
            LogLevel::Info => tauri_plugin_log::LogLevel::Info,
            LogLevel::Warn => tauri_plugin_log::LogLevel::Warn,
            LogLevel::Error => tauri_plugin_log::LogLevel::Error,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct ShortcutBinding {
    pub id: String,
    pub name: String,
    pub description: String,
    pub default_binding: String,
    pub current_binding: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct LLMPrompt {
    pub id: String,
    pub name: String,
    pub prompt: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct PostProcessProvider {
    pub id: String,
    pub label: String,
    pub base_url: String,
    #[serde(default)]
    pub allow_base_url_edit: bool,
    #[serde(default)]
    pub models_endpoint: Option<String>,
    #[serde(default)]
    pub supports_structured_output: bool,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "lowercase")]
pub enum OverlayPosition {
    None,
    Top,
    Bottom,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum ModelUnloadTimeout {
    Never,
    Immediately,
    Min2,
    Min5,
    Min10,
    Min15,
    Hour1,
    Sec15, // Debug mode only
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum PasteMethod {
    CtrlV,
    Direct,
    None,
    ShiftInsert,
    CtrlShiftV,
    ExternalScript,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum ClipboardHandling {
    DontModify,
    CopyToClipboard,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum AutoSubmitKey {
    Enter,
    CtrlEnter,
    CmdEnter,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum RecordingRetentionPeriod {
    Never,
    PreserveLimit,
    Days3,
    Weeks2,
    Months3,
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum KeyboardImplementation {
    Tauri,
    HandyKeys,
}

impl Default for KeyboardImplementation {
    fn default() -> Self {
        #[cfg(target_os = "linux")]
        return KeyboardImplementation::Tauri;
        #[cfg(not(target_os = "linux"))]
        return KeyboardImplementation::HandyKeys;
    }
}

impl Default for ModelUnloadTimeout {
    fn default() -> Self {
        ModelUnloadTimeout::Min5
    }
}

impl Default for PasteMethod {
    fn default() -> Self {
        // Default to CtrlV for macOS and Windows, Direct for Linux
        #[cfg(target_os = "linux")]
        return PasteMethod::Direct;
        #[cfg(not(target_os = "linux"))]
        return PasteMethod::CtrlV;
    }
}

impl Default for ClipboardHandling {
    fn default() -> Self {
        ClipboardHandling::DontModify
    }
}

impl Default for AutoSubmitKey {
    fn default() -> Self {
        AutoSubmitKey::Enter
    }
}

#[allow(dead_code)]
impl ModelUnloadTimeout {
    pub fn to_minutes(self) -> Option<u64> {
        match self {
            ModelUnloadTimeout::Never => None,
            ModelUnloadTimeout::Immediately => Some(0), // Special case for immediate unloading
            ModelUnloadTimeout::Min2 => Some(2),
            ModelUnloadTimeout::Min5 => Some(5),
            ModelUnloadTimeout::Min10 => Some(10),
            ModelUnloadTimeout::Min15 => Some(15),
            ModelUnloadTimeout::Hour1 => Some(60),
            ModelUnloadTimeout::Sec15 => Some(0), // Special case for debug - handled separately
        }
    }

    pub fn to_seconds(self) -> Option<u64> {
        match self {
            ModelUnloadTimeout::Never => None,
            ModelUnloadTimeout::Immediately => Some(0), // Special case for immediate unloading
            ModelUnloadTimeout::Sec15 => Some(15),
            _ => self.to_minutes().map(|m| m * 60),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum SoundTheme {
    Marimba,
    Pop,
    Custom,
}

impl SoundTheme {
    fn as_str(&self) -> &'static str {
        match self {
            SoundTheme::Marimba => "marimba",
            SoundTheme::Pop => "pop",
            SoundTheme::Custom => "custom",
        }
    }

    pub fn to_start_path(&self) -> String {
        format!("resources/{}_start.wav", self.as_str())
    }

    pub fn to_stop_path(&self) -> String {
        format!("resources/{}_stop.wav", self.as_str())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum TypingTool {
    Auto,
    Wtype,
    Kwtype,
    Dotool,
    Ydotool,
    Xdotool,
}

impl Default for TypingTool {
    fn default() -> Self {
        TypingTool::Auto
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum WhisperAcceleratorSetting {
    Auto,
    Cpu,
    Gpu,
}

impl Default for WhisperAcceleratorSetting {
    fn default() -> Self {
        WhisperAcceleratorSetting::Auto
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq, Type)]
#[serde(rename_all = "snake_case")]
pub enum OrtAcceleratorSetting {
    Auto,
    Cpu,
    Cuda,
    #[serde(rename = "directml")]
    DirectMl,
    Rocm,
}

impl Default for OrtAcceleratorSetting {
    fn default() -> Self {
        OrtAcceleratorSetting::Auto
    }
}

/* still handy for composing the initial JSON in the store ------------- */
#[derive(Serialize, Deserialize, Debug, Clone, Type)]
pub struct AppSettings {
    pub bindings: HashMap<String, ShortcutBinding>,
    pub push_to_talk: bool,
    pub audio_feedback: bool,
    #[serde(default = "default_audio_feedback_volume")]
    pub audio_feedback_volume: f32,
    #[serde(default = "default_sound_theme")]
    pub sound_theme: SoundTheme,
    #[serde(default = "default_start_hidden")]
    pub start_hidden: bool,
    #[serde(default = "default_autostart_enabled")]
    pub autostart_enabled: bool,
    #[serde(default = "default_update_checks_enabled")]
    pub update_checks_enabled: bool,
    #[serde(default = "default_model")]
    pub selected_model: String,
    #[serde(default = "default_always_on_microphone")]
    pub always_on_microphone: bool,
    #[serde(default)]
    pub selected_microphone: Option<String>,
    #[serde(default)]
    pub clamshell_microphone: Option<String>,
    #[serde(default)]
    pub selected_output_device: Option<String>,
    #[serde(default = "default_translate_to_english")]
    pub translate_to_english: bool,
    #[serde(default = "default_selected_language")]
    pub selected_language: String,
    #[serde(default = "default_overlay_position")]
    pub overlay_position: OverlayPosition,
    #[serde(default = "default_debug_mode")]
    pub debug_mode: bool,
    #[serde(default = "default_log_level")]
    pub log_level: LogLevel,
    #[serde(default)]
    pub custom_words: Vec<String>,
    #[serde(default)]
    pub model_unload_timeout: ModelUnloadTimeout,
    #[serde(default = "default_qwen_adaptive_unload_enabled")]
    pub qwen_adaptive_unload_enabled: bool,
    #[serde(default = "default_qwen_periodic_keep_warm_enabled")]
    pub qwen_periodic_keep_warm_enabled: bool,
    #[serde(default = "default_qwen_periodic_keep_warm_interval_sec")]
    pub qwen_periodic_keep_warm_interval_sec: u64,
    #[serde(default = "default_qwen_periodic_keep_warm_active_window_min")]
    pub qwen_periodic_keep_warm_active_window_min: u64,
    #[serde(default = "default_word_correction_threshold")]
    pub word_correction_threshold: f64,
    #[serde(default = "default_history_limit")]
    pub history_limit: usize,
    #[serde(default = "default_recording_retention_period")]
    pub recording_retention_period: RecordingRetentionPeriod,
    #[serde(default)]
    pub paste_method: PasteMethod,
    #[serde(default)]
    pub clipboard_handling: ClipboardHandling,
    #[serde(default = "default_auto_submit")]
    pub auto_submit: bool,
    #[serde(default)]
    pub auto_submit_key: AutoSubmitKey,
    #[serde(default = "default_post_process_enabled")]
    pub post_process_enabled: bool,
    #[serde(default = "default_post_process_provider_id")]
    pub post_process_provider_id: String,
    #[serde(default = "default_post_process_providers")]
    pub post_process_providers: Vec<PostProcessProvider>,
    #[serde(default = "default_post_process_api_keys")]
    pub post_process_api_keys: HashMap<String, String>,
    #[serde(default = "default_post_process_models")]
    pub post_process_models: HashMap<String, String>,
    #[serde(default = "default_post_process_prompts")]
    pub post_process_prompts: Vec<LLMPrompt>,
    #[serde(default)]
    pub post_process_selected_prompt_id: Option<String>,
    #[serde(default = "default_post_process_system_prompt")]
    pub post_process_system_prompt: String,
    #[serde(default = "default_post_process_quality")]
    pub post_process_quality: String,
    #[serde(default = "default_post_process_local_max_tokens")]
    pub post_process_local_max_tokens: usize,
    #[serde(default = "default_post_process_local_temperature")]
    pub post_process_local_temperature: f64,
    #[serde(default = "default_post_process_local_top_p")]
    pub post_process_local_top_p: f64,
    #[serde(default = "default_post_process_local_repetition_penalty")]
    pub post_process_local_repetition_penalty: f64,
    #[serde(default = "default_post_process_local_repetition_context_size")]
    pub post_process_local_repetition_context_size: usize,
    #[serde(default = "default_qwen_startup_preload_strategy")]
    pub qwen_startup_preload_strategy: String,
    #[serde(default = "default_qwen3_startup_preload_enabled")]
    pub qwen3_startup_preload_enabled: bool,
    #[serde(default = "default_qwen3_startup_preload_delay_ms")]
    pub qwen3_startup_preload_delay_ms: u64,
    #[serde(default = "default_qwen3_max_threads")]
    pub qwen3_max_threads: usize,
    #[serde(default = "default_qwen3_server_ready_timeout_sec")]
    pub qwen3_server_ready_timeout_sec: u64,
    #[serde(default = "default_qwen3_warmup_enabled")]
    pub qwen3_warmup_enabled: bool,
    #[serde(default = "default_qwen35_startup_preload_enabled")]
    pub qwen35_startup_preload_enabled: bool,
    #[serde(default = "default_qwen35_startup_preload_delay_ms")]
    pub qwen35_startup_preload_delay_ms: u64,
    #[serde(default = "default_qwen35_warmup_enabled")]
    pub qwen35_warmup_enabled: bool,
    #[serde(default = "default_qwen35_max_threads")]
    pub qwen35_max_threads: usize,
    #[serde(default = "default_qwen35_server_ready_timeout_sec")]
    pub qwen35_server_ready_timeout_sec: u64,
    #[serde(default = "default_qwen35_inference_timeout_sec")]
    pub qwen35_inference_timeout_sec: u64,
    #[serde(default = "default_script_hooks_enabled")]
    pub script_hooks_enabled: bool,
    #[serde(default)]
    pub post_asr_script_path: Option<String>,
    #[serde(default)]
    pub post_llm_script_path: Option<String>,
    #[serde(default = "default_script_hook_timeout_ms")]
    pub script_hook_timeout_ms: u64,
    #[serde(default)]
    pub mute_while_recording: bool,
    #[serde(default)]
    pub append_trailing_space: bool,
    #[serde(default = "default_app_language")]
    pub app_language: String,
    #[serde(default)]
    pub experimental_enabled: bool,
    #[serde(default)]
    pub lazy_stream_close: bool,
    #[serde(default)]
    pub keyboard_implementation: KeyboardImplementation,
    #[serde(default = "default_show_tray_icon")]
    pub show_tray_icon: bool,
    #[serde(default = "default_paste_delay_ms")]
    pub paste_delay_ms: u64,
    #[serde(default = "default_typing_tool")]
    pub typing_tool: TypingTool,
    pub external_script_path: Option<String>,
    #[serde(default)]
    pub custom_filler_words: Option<Vec<String>>,
    #[serde(default)]
    pub whisper_accelerator: WhisperAcceleratorSetting,
    #[serde(default)]
    pub ort_accelerator: OrtAcceleratorSetting,
    #[serde(default = "default_whisper_gpu_device")]
    pub whisper_gpu_device: i32,
    #[serde(default)]
    pub extra_recording_buffer_ms: u64,
}

fn default_model() -> String {
    "".to_string()
}

fn default_always_on_microphone() -> bool {
    false
}

fn default_translate_to_english() -> bool {
    false
}

fn default_start_hidden() -> bool {
    false
}

fn default_autostart_enabled() -> bool {
    false
}

fn default_update_checks_enabled() -> bool {
    true
}

fn default_selected_language() -> String {
    "auto".to_string()
}

fn default_overlay_position() -> OverlayPosition {
    #[cfg(target_os = "linux")]
    return OverlayPosition::None;
    #[cfg(not(target_os = "linux"))]
    return OverlayPosition::Bottom;
}

fn default_debug_mode() -> bool {
    false
}

fn default_log_level() -> LogLevel {
    LogLevel::Debug
}

fn default_word_correction_threshold() -> f64 {
    0.18
}

fn default_paste_delay_ms() -> u64 {
    60
}

fn default_auto_submit() -> bool {
    false
}

fn default_history_limit() -> usize {
    5
}

fn default_recording_retention_period() -> RecordingRetentionPeriod {
    RecordingRetentionPeriod::PreserveLimit
}

fn default_audio_feedback_volume() -> f32 {
    1.0
}

fn default_sound_theme() -> SoundTheme {
    SoundTheme::Marimba
}

fn default_post_process_enabled() -> bool {
    true
}

fn default_post_process_system_prompt() -> String {
    "你是严格的中文转录后处理器。\n输出契约：\n1. 仅输出最终结果，不要解释。\n2. 严格遵循所选用户提示词模板；不要复述模板条款、要求、规则或输入标题。\n3. 禁止输出思考过程、分析、<think> 标签、包装语。\n4. 在不改变事实与结论的前提下，优先提升可读性与结构化表达。\n5. 若用户模板要求列表化：当出现并列/序列信号（如“并且、而且、同时、以及、另外、然后、第一/第二/第三、1、2、3、;、；”）时，必须使用 Markdown 列表。\n6. 专有名词、产品名、模型名、缩写、URL、代码、数字保持准确。\n7. 输入为空时返回空字符串。".to_string()
}

fn default_post_process_quality() -> String {
    "balanced".to_string()
}

fn normalize_post_process_quality(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "fast" => "fast".to_string(),
        "quality" => "quality".to_string(),
        "custom" => "custom".to_string(),
        _ => "balanced".to_string(),
    }
}

fn default_post_process_local_max_tokens() -> usize {
    192
}

fn default_post_process_local_temperature() -> f64 {
    0.07
}

fn default_post_process_local_top_p() -> f64 {
    0.82
}

fn default_post_process_local_repetition_penalty() -> f64 {
    1.15
}

fn default_post_process_local_repetition_context_size() -> usize {
    160
}

fn default_qwen_startup_preload_strategy() -> String {
    "parallel".to_string()
}

fn default_qwen_adaptive_unload_enabled() -> bool {
    true
}

fn default_qwen_periodic_keep_warm_enabled() -> bool {
    false
}

fn default_qwen_periodic_keep_warm_interval_sec() -> u64 {
    300
}

fn default_qwen_periodic_keep_warm_active_window_min() -> u64 {
    30
}

fn default_qwen3_startup_preload_enabled() -> bool {
    true
}

fn default_qwen3_startup_preload_delay_ms() -> u64 {
    0
}

fn default_qwen3_max_threads() -> usize {
    0
}

fn default_qwen3_server_ready_timeout_sec() -> u64 {
    30
}

fn default_qwen3_warmup_enabled() -> bool {
    true
}

fn default_qwen35_startup_preload_enabled() -> bool {
    true
}

fn default_qwen35_startup_preload_delay_ms() -> u64 {
    0
}

fn default_qwen35_warmup_enabled() -> bool {
    true
}

fn default_qwen35_max_threads() -> usize {
    0
}

fn default_qwen35_server_ready_timeout_sec() -> u64 {
    90
}

fn default_qwen35_inference_timeout_sec() -> u64 {
    45
}

fn default_script_hooks_enabled() -> bool {
    false
}

fn default_script_hook_timeout_ms() -> u64 {
    1200
}

fn normalize_post_process_local_max_tokens(value: usize) -> usize {
    value.clamp(64, 512)
}

fn normalize_post_process_local_temperature(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

fn normalize_post_process_local_top_p(value: f64) -> f64 {
    value.clamp(0.1, 1.0)
}

fn normalize_post_process_local_repetition_penalty(value: f64) -> f64 {
    value.clamp(1.0, 1.5)
}

fn normalize_post_process_local_repetition_context_size(value: usize) -> usize {
    value.clamp(32, 256)
}

pub fn normalize_qwen_startup_preload_strategy(value: &str) -> String {
    match value.trim().to_ascii_lowercase().as_str() {
        "serial" => "serial".to_string(),
        _ => "parallel".to_string(),
    }
}

pub fn normalize_qwen_startup_preload_delay_ms(value: u64) -> u64 {
    value.clamp(0, 15_000)
}

pub fn normalize_qwen_periodic_keep_warm_interval_sec(value: u64) -> u64 {
    value.clamp(60, 3_600)
}

pub fn normalize_qwen_periodic_keep_warm_active_window_min(value: u64) -> u64 {
    value.clamp(5, 240)
}

pub fn normalize_qwen_max_threads(value: usize) -> usize {
    value.clamp(0, 16)
}

pub fn normalize_qwen3_server_ready_timeout_sec(value: u64) -> u64 {
    value.clamp(10, 120)
}

pub fn normalize_qwen35_server_ready_timeout_sec(value: u64) -> u64 {
    value.clamp(20, 300)
}

pub fn normalize_qwen35_inference_timeout_sec(value: u64) -> u64 {
    value.clamp(5, 180)
}

pub fn normalize_script_hook_timeout_ms(value: u64) -> u64 {
    value.clamp(100, 10_000)
}

fn default_app_language() -> String {
    tauri_plugin_os::locale()
        .map(|l| l.replace('_', "-"))
        .unwrap_or_else(|| "en".to_string())
}

fn default_show_tray_icon() -> bool {
    true
}

fn default_post_process_provider_id() -> String {
    LOCAL_QWEN35_PROVIDER_ID.to_string()
}

fn default_post_process_providers() -> Vec<PostProcessProvider> {
    let mut providers = vec![
        PostProcessProvider {
            id: "openai".to_string(),
            label: "OpenAI".to_string(),
            base_url: "https://api.openai.com/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "zai".to_string(),
            label: "Z.AI".to_string(),
            base_url: "https://api.z.ai/api/paas/v4".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "openrouter".to_string(),
            label: "OpenRouter".to_string(),
            base_url: "https://openrouter.ai/api/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: "anthropic".to_string(),
            label: "Anthropic".to_string(),
            base_url: "https://api.anthropic.com/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        },
        PostProcessProvider {
            id: "groq".to_string(),
            label: "Groq".to_string(),
            base_url: "https://api.groq.com/openai/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: false,
        },
        PostProcessProvider {
            id: "cerebras".to_string(),
            label: "Cerebras".to_string(),
            base_url: "https://api.cerebras.ai/v1".to_string(),
            allow_base_url_edit: false,
            models_endpoint: Some("/models".to_string()),
            supports_structured_output: true,
        },
        PostProcessProvider {
            id: LOCAL_QWEN35_PROVIDER_ID.to_string(),
            label: "Local".to_string(),
            base_url: "local://qwen35".to_string(),
            allow_base_url_edit: false,
            models_endpoint: None,
            supports_structured_output: false,
        },
    ];

    // Note: We always include Apple Intelligence on macOS ARM64 without checking availability
    // at startup. The availability check is deferred to when the user actually tries to use it
    // (in actions.rs). This prevents crashes on macOS 26.x beta where accessing
    // SystemLanguageModel.default during early app initialization causes SIGABRT.
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        providers.push(PostProcessProvider {
            id: APPLE_INTELLIGENCE_PROVIDER_ID.to_string(),
            label: "Apple Intelligence".to_string(),
            base_url: "apple-intelligence://local".to_string(),
            allow_base_url_edit: false,
            models_endpoint: None,
            supports_structured_output: true,
        });
    }

    // Custom provider always comes last
    providers.push(PostProcessProvider {
        id: "custom".to_string(),
        label: "Custom".to_string(),
        base_url: "http://localhost:11434/v1".to_string(),
        allow_base_url_edit: true,
        models_endpoint: Some("/models".to_string()),
        supports_structured_output: false,
    });

    providers
}

fn default_post_process_api_keys() -> HashMap<String, String> {
    let mut map = HashMap::new();
    for provider in default_post_process_providers() {
        map.insert(provider.id, String::new());
    }
    map
}

fn default_model_for_provider(provider_id: &str) -> String {
    if provider_id == APPLE_INTELLIGENCE_PROVIDER_ID {
        return APPLE_INTELLIGENCE_DEFAULT_MODEL_ID.to_string();
    }
    if provider_id == LOCAL_QWEN35_PROVIDER_ID {
        return LOCAL_QWEN35_DEFAULT_MODEL_ID.to_string();
    }
    String::new()
}

fn default_post_process_models() -> HashMap<String, String> {
    let mut map = HashMap::new();
    for provider in default_post_process_providers() {
        map.insert(
            provider.id.clone(),
            default_model_for_provider(&provider.id),
        );
    }
    map
}

fn default_post_process_prompts() -> Vec<LLMPrompt> {
    vec![
        LLMPrompt {
            id: "template_translate_english_default".to_string(),
            name: "Translate to English (Default)".to_string(),
            prompt: "Translate the transcript into natural English.\n\nInput:\n${output}\n\nRules:\n1. Translate all Chinese content, including short utterances.\n2. For short Chinese interjections, use concise natural English (e.g. 好 -> okay, 行 -> okay, 棒 -> great).\n3. If Chinese numerals appear as standalone number/list items, convert them to Arabic digits (e.g. 一二三四五六七 -> 1234567; 一、二、三 -> 1、2、3).\n4. Normalize common model-size wording to Arabic numeric form when appropriate (e.g. 零点8B -> 0.8B, 两B -> 2B).\n5. Do not convert inside words/compounds (e.g. 一些 must stay semantic, not 1些).\n6. Preserve existing English words, names, acronyms, numbers, and mixed-language tokens when already correct.\n7. Output only the final translation text.".to_string(),
        },
        LLMPrompt {
            id: "template_chinese_markdown_polish".to_string(),
            name: "中文口语整理（Markdown）".to_string(),
            prompt: "请将下面的转录文本做中文后处理与排版，不要翻译。\n\n输入：\n${output}\n\n目标：\n把口语化表达整理成清晰、简洁、可读的文本，不改变原意。\n\n规则：\n1. 只输出最终结果，不解释，不复述模板文字。\n2. 列表优先：当出现并列或序列关系时，必须转成 Markdown 列表。\n   - 若出现“第一/第二/第三/首先/其次/最后/1、2、3”等明确序号，输出有序列表（1. 2. 3.）。\n   - 若出现“并且/而且/同时/以及/另外/还有/然后/并”等并列连接词且包含 2 个及以上分句，输出无序列表（-）。\n3. 非列表场景输出一行简洁文本。\n4. 去口头重复与语气词；专有名词、产品名、模型名、缩写、URL、代码、数字保持准确；中文数字按语义可转阿拉伯数字（如“零点8B/零点八B -> 0.8B”，“两B -> 2B”），但禁止词内替换（如“一些”不能变“1些”）。\n\n示例：\n输入：我们第一点呢，要控糖；第二点呢，要早睡；第三点要运动。\n输出：\n1. 要控糖。\n2. 要早睡。\n3. 要运动。\n\n输入：我们还要做复盘，并且整理资料，而且同步进度。\n输出：\n- 做复盘。\n- 整理资料。\n- 同步进度。".to_string(),
        },
    ]
}

fn is_legacy_default_translate_prompt(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed == "${output}"
        || trimmed
            == "Translate the transcript into natural English.\n\nInput:\n${output}\n\nRules:\n1. Translate all Chinese content, including short utterances.\n2. For short Chinese interjections, use concise natural English (e.g. 好 -> okay, 行 -> okay, 棒 -> great).\n3. For short standalone Chinese numerals, convert to Arabic digits (e.g. 一二三 -> 123). Do not apply digit substitution inside words or compounds.\n4. Preserve existing English words, names, acronyms, numbers, and mixed-language tokens when already correct.\n5. Output only the final translation text."
        || trimmed
            == "Translate the transcript into natural English.\n\nInput:\n${output}\n\nRules:\n1. Translate all Chinese content, including short utterances.\n2. For short Chinese interjections, use concise natural English (e.g. 好 -> okay, 行 -> okay, 棒 -> great).\n3. If Chinese numerals appear as a standalone number/list item, convert them to Arabic digits (e.g. 一二三四五六七 -> 1234567; 一、二、三 -> 1、2、3).\n4. Do not convert inside words/compounds (e.g. 一些 must stay semantic, not 1些).\n5. Preserve existing English words, names, acronyms, numbers, and mixed-language tokens when already correct.\n6. Output only the final translation text."
}

fn is_legacy_default_chinese_markdown_prompt(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed
        == "请将下面的转录文本做中文后处理与排版，不要翻译。\n\n输入：\n${output}\n\n要求：\n1. 保持原意与事实，不新增信息，不改变结论。\n2. 去除口头重复、语气词和明显噪音，让表达更简洁。\n3. 专有名词、产品名、模型名、缩写、数字、URL、代码符号保持原样。\n4. 内容是多点信息时用 Markdown 列表整理；短句则输出一行简洁文本。\n5. 仅输出最终结果，不要解释。"
        || trimmed
            == "请将下面的转录文本做中文后处理与排版，不要翻译。\n\n输入：\n${output}\n\n要求：\n1. 保持原意与事实，不新增信息，不改变结论。\n2. 去除口头重复、语气词和明显噪音，让表达更简洁。\n3. 专有名词、产品名、模型名、缩写、数字、URL、代码符号保持原样。\n4. 中文数字尽量转阿拉伯数字（示例：一二三四五六七 -> 1234567；零点8B/零点八B -> 0.8B；两B -> 2B）。不要把词内字符误替换（例如“一些”不能变成“1些”）。\n5. 如内容包含“第一点/第二点/第X点/1、2、3”等并列结构，输出为 Markdown 列表；否则输出一行简洁文本。\n6. 禁止输出本模板条款本身（例如“1. 保持原意与事实...”这类说明文字）。\n7. 仅输出最终结果，不要解释。"
        || trimmed
            == "请将下面的转录文本做中文后处理与排版，不要翻译。\n\n输入：\n${output}\n\n要求：\n1. 只输出最终结果，不要解释，不要复述“要求/规则/输入”等模板内容。\n2. 保持原意与事实，去除口头重复和明显噪音；专有名词、产品名、模型名、缩写、数字、URL、代码符号保持原样。\n3. 列表优先：若出现两个及以上并列观点，或含“第一点/第二点/另外/最后/1、2、3”等序列信号，必须输出 Markdown 有序列表（每点一行简短句）。\n4. 非列表场景输出一行简洁文本。\n5. 中文数字尽量转阿拉伯数字（示例：一二三四五六七 -> 1234567；零点8B/零点八B -> 0.8B；两B -> 2B）；不要词内替换（例如“一些”不能变成“1些”）。"
        || trimmed
            == "请将下面的转录文本做中文后处理与排版，不要翻译。\n\n输入：\n${output}\n\n要求：\n1. 只输出最终结果，不要解释，不要输出“要求/规则/输入”等模板文字。\n2. 保持原意，去口头重复和语气词；专有名词、产品名、模型名、缩写、URL、代码保持原样；中文数字按语义转阿拉伯数字（如“零点8B/零点八B -> 0.8B”，“两B -> 2B”），但不要词内替换（如“一些”不能变“1些”）。\n3. 列表优先：只要出现并列观点或序号信号（如“第一/第二/另外/最后/1、2、3/请列出/分点”），必须输出 Markdown 有序列表；否则输出一行简洁文本。\n\n示例：\n- 输入：第一点要控糖，第二点要早睡，第三点要运动。\n  输出：\n  1. 要控糖。\n  2. 要早睡。\n  3. 要运动。\n- 输入：嗯这个模型还可以吧。\n  输出：这个模型还可以。"
}

fn is_prunable_legacy_preset_prompt(prompt: &LLMPrompt) -> bool {
    if matches!(
        prompt.id.as_str(),
        "default_improve_transcriptions"
            | "template_translate_english_strict"
            | "template_translate_english_concise"
            | "template_standard_normalize"
            | "template_strict_literal"
            | "template_readable_polish"
            | "template_domain_tech"
    ) {
        return true;
    }

    let normalized_name = prompt.name.trim().to_ascii_lowercase();
    let normalized_prompt = prompt.prompt.trim().to_ascii_lowercase();

    if normalized_name == "english" && normalized_prompt == "translate it into english" {
        return true;
    }

    prompt.name.trim() == "英语"
        && prompt.prompt.trim() == "把${output}翻译成英语翻译成对应的英语。"
}

fn is_legacy_default_post_process_system_prompt(value: &str) -> bool {
    let trimmed = value.trim();
    trimmed == "You are a strict transcription post-processor.\nOutput rules:\n1. Output only the final processed text.\n2. Do not include reasoning or analysis.\n3. Do not use <think> tags.\n4. Do not include bullet points, examples, or explanations.\n5. Keep original meaning and language unless explicitly requested otherwise."
        || trimmed
            == "You are a strict transcript translator.\nTask:\nTranslate incoming transcript text into natural English.\nOutput rules:\n1. Output English only.\n2. Always translate, including very short inputs (single-word or 1-3 character phrases).\n3. For short Chinese interjections, produce concise natural English (e.g. 好 -> okay, 棒 -> great, 行 -> okay).\n4. Convert Chinese numerals to English words when short and standalone (e.g. 一二三 -> one two three).\n5. Preserve existing English words, product names, acronyms, and numbers accurately (e.g. HANDY, Qwen3.5, 1.7B).\n6. Do not include reasoning, <think>, bullet points, or explanations.\n7. Return only the final translated text."
        || trimmed
            == "You are a strict transcript post-processor.\nOutput contract:\n1. Produce only the final processed text.\n2. Follow the selected user prompt template exactly.\n3. Never output reasoning, analysis, chain-of-thought, or <think> tags.\n4. Never output explanations, bullet examples, wrappers, or meta commentary.\n5. Preserve meaning and key facts unless the selected user prompt explicitly requests transformation.\n6. Preserve proper nouns, product names, acronyms, numbers, and code-like tokens accurately.\n7. If input content is empty, return an empty string."
        || trimmed
            == "You are a strict transcript post-processor.\nOutput contract:\n1. Produce only the final processed text.\n2. Follow the selected user prompt template exactly.\n3. If the user prompt requests Arabic-digit conversion, apply it strictly while avoiding in-word substitution.\n4. Never output reasoning, analysis, chain-of-thought, or <think> tags.\n5. Never output explanations, bullet examples, wrappers, or meta commentary.\n6. Preserve meaning and key facts unless the selected user prompt explicitly requests transformation.\n7. Preserve proper nouns, product names, acronyms, numbers, and code-like tokens accurately.\n8. If input content is empty, return an empty string."
        || trimmed
            == "You are a strict transcript post-processor.\nOutput contract:\n1. Produce only the final processed text.\n2. Treat the selected user prompt template as instruction metadata; do not echo, paraphrase, or restate template rule lines.\n3. If the user prompt requests Arabic-digit conversion, apply it strictly while avoiding in-word substitution.\n4. Never output reasoning, analysis, chain-of-thought, or <think> tags.\n5. Never output explanations, wrappers, or meta commentary.\n6. Preserve meaning and key facts unless the selected user prompt explicitly requests transformation.\n7. Preserve proper nouns, product names, acronyms, numbers, and code-like tokens accurately.\n8. If input content is empty, return an empty string."
}

fn default_whisper_gpu_device() -> i32 {
    -1 // auto
}

fn default_typing_tool() -> TypingTool {
    TypingTool::Auto
}

fn ensure_post_process_defaults(settings: &mut AppSettings) -> bool {
    let mut changed = false;
    for provider in default_post_process_providers() {
        // Use match to do a single lookup - either sync existing or add new
        match settings
            .post_process_providers
            .iter_mut()
            .find(|p| p.id == provider.id)
        {
            Some(existing) => {
                if existing.label != provider.label {
                    existing.label = provider.label.clone();
                    changed = true;
                }
                if existing.allow_base_url_edit != provider.allow_base_url_edit {
                    existing.allow_base_url_edit = provider.allow_base_url_edit;
                    changed = true;
                }
                if provider.id != "custom" && existing.base_url != provider.base_url {
                    existing.base_url = provider.base_url.clone();
                    changed = true;
                }
                if provider.id != "custom" && existing.models_endpoint != provider.models_endpoint {
                    existing.models_endpoint = provider.models_endpoint.clone();
                    changed = true;
                }
                // Sync supports_structured_output field for existing providers (migration)
                if existing.supports_structured_output != provider.supports_structured_output {
                    debug!(
                        "Updating supports_structured_output for provider '{}' from {} to {}",
                        provider.id,
                        existing.supports_structured_output,
                        provider.supports_structured_output
                    );
                    existing.supports_structured_output = provider.supports_structured_output;
                    changed = true;
                }
            }
            None => {
                // Provider doesn't exist, add it
                settings.post_process_providers.push(provider.clone());
                changed = true;
            }
        }

        if !settings.post_process_api_keys.contains_key(&provider.id) {
            settings
                .post_process_api_keys
                .insert(provider.id.clone(), String::new());
            changed = true;
        }

        let default_model = default_model_for_provider(&provider.id);
        match settings.post_process_models.get_mut(&provider.id) {
            Some(existing) => {
                if existing.is_empty() && !default_model.is_empty() {
                    *existing = default_model.clone();
                    changed = true;
                }
            }
            None => {
                settings
                    .post_process_models
                    .insert(provider.id.clone(), default_model);
                changed = true;
            }
        }
    }

    for default_prompt in default_post_process_prompts() {
        match settings
            .post_process_prompts
            .iter_mut()
            .find(|prompt| prompt.id == default_prompt.id)
        {
            Some(existing) => {
                if existing.name != default_prompt.name {
                    existing.name = default_prompt.name.clone();
                    changed = true;
                }

                let should_sync_prompt = existing.prompt.trim().is_empty()
                    || (existing.id == "template_translate_english_default"
                        && is_legacy_default_translate_prompt(&existing.prompt))
                    || (existing.id == "template_chinese_markdown_polish"
                        && is_legacy_default_chinese_markdown_prompt(&existing.prompt));
                if should_sync_prompt && existing.prompt != default_prompt.prompt {
                    existing.prompt = default_prompt.prompt.clone();
                    changed = true;
                }
            }
            None => {
                settings.post_process_prompts.push(default_prompt);
                changed = true;
            }
        }
    }

    let before_prune_len = settings.post_process_prompts.len();
    settings
        .post_process_prompts
        .retain(|prompt| !is_prunable_legacy_preset_prompt(prompt));
    if settings.post_process_prompts.len() != before_prune_len {
        changed = true;
    }

    let provider_is_valid = settings
        .post_process_providers
        .iter()
        .any(|provider| provider.id == settings.post_process_provider_id);
    if !provider_is_valid {
        settings.post_process_provider_id = default_post_process_provider_id();
        changed = true;
    }

    let selected_prompt_is_valid = settings
        .post_process_selected_prompt_id
        .as_ref()
        .is_some_and(|selected_id| {
            settings
                .post_process_prompts
                .iter()
                .any(|prompt| prompt.id == *selected_id)
        });

    if !selected_prompt_is_valid {
        let fallback_id = if settings
            .post_process_prompts
            .iter()
            .any(|prompt| prompt.id == "template_translate_english_default")
        {
            Some("template_translate_english_default".to_string())
        } else {
            settings.post_process_prompts.first().map(|p| p.id.clone())
        };

        if settings.post_process_selected_prompt_id != fallback_id {
            settings.post_process_selected_prompt_id = fallback_id;
            changed = true;
        }
    }

    if settings.post_process_system_prompt.trim().is_empty()
        || is_legacy_default_post_process_system_prompt(&settings.post_process_system_prompt)
    {
        settings.post_process_system_prompt = default_post_process_system_prompt();
        changed = true;
    }

    let normalized_quality = normalize_post_process_quality(&settings.post_process_quality);
    if settings.post_process_quality != normalized_quality {
        settings.post_process_quality = normalized_quality;
        changed = true;
    }

    let normalized_max_tokens =
        normalize_post_process_local_max_tokens(settings.post_process_local_max_tokens);
    if settings.post_process_local_max_tokens != normalized_max_tokens {
        settings.post_process_local_max_tokens = normalized_max_tokens;
        changed = true;
    }

    let normalized_temperature =
        normalize_post_process_local_temperature(settings.post_process_local_temperature);
    if (settings.post_process_local_temperature - normalized_temperature).abs() > f64::EPSILON {
        settings.post_process_local_temperature = normalized_temperature;
        changed = true;
    }

    let normalized_top_p = normalize_post_process_local_top_p(settings.post_process_local_top_p);
    if (settings.post_process_local_top_p - normalized_top_p).abs() > f64::EPSILON {
        settings.post_process_local_top_p = normalized_top_p;
        changed = true;
    }

    let normalized_repetition_penalty = normalize_post_process_local_repetition_penalty(
        settings.post_process_local_repetition_penalty,
    );
    if (settings.post_process_local_repetition_penalty - normalized_repetition_penalty).abs()
        > f64::EPSILON
    {
        settings.post_process_local_repetition_penalty = normalized_repetition_penalty;
        changed = true;
    }

    let normalized_repetition_context_size = normalize_post_process_local_repetition_context_size(
        settings.post_process_local_repetition_context_size,
    );
    if settings.post_process_local_repetition_context_size != normalized_repetition_context_size {
        settings.post_process_local_repetition_context_size = normalized_repetition_context_size;
        changed = true;
    }

    let normalized_preload_strategy =
        normalize_qwen_startup_preload_strategy(&settings.qwen_startup_preload_strategy);
    if settings.qwen_startup_preload_strategy != normalized_preload_strategy {
        settings.qwen_startup_preload_strategy = normalized_preload_strategy;
        changed = true;
    }

    let normalized_keep_warm_interval = normalize_qwen_periodic_keep_warm_interval_sec(
        settings.qwen_periodic_keep_warm_interval_sec,
    );
    if settings.qwen_periodic_keep_warm_interval_sec != normalized_keep_warm_interval {
        settings.qwen_periodic_keep_warm_interval_sec = normalized_keep_warm_interval;
        changed = true;
    }

    let normalized_keep_warm_window = normalize_qwen_periodic_keep_warm_active_window_min(
        settings.qwen_periodic_keep_warm_active_window_min,
    );
    if settings.qwen_periodic_keep_warm_active_window_min != normalized_keep_warm_window {
        settings.qwen_periodic_keep_warm_active_window_min = normalized_keep_warm_window;
        changed = true;
    }

    let normalized_qwen3_delay =
        normalize_qwen_startup_preload_delay_ms(settings.qwen3_startup_preload_delay_ms);
    if settings.qwen3_startup_preload_delay_ms != normalized_qwen3_delay {
        settings.qwen3_startup_preload_delay_ms = normalized_qwen3_delay;
        changed = true;
    } else if settings.qwen3_startup_preload_delay_ms == 900 {
        // Migrate previous default to immediate preload for better first-use latency.
        settings.qwen3_startup_preload_delay_ms = default_qwen3_startup_preload_delay_ms();
        changed = true;
    }

    let normalized_qwen3_threads = normalize_qwen_max_threads(settings.qwen3_max_threads);
    if settings.qwen3_max_threads != normalized_qwen3_threads {
        settings.qwen3_max_threads = normalized_qwen3_threads;
        changed = true;
    }

    let normalized_qwen3_timeout =
        normalize_qwen3_server_ready_timeout_sec(settings.qwen3_server_ready_timeout_sec);
    if settings.qwen3_server_ready_timeout_sec != normalized_qwen3_timeout {
        settings.qwen3_server_ready_timeout_sec = normalized_qwen3_timeout;
        changed = true;
    }

    let normalized_qwen35_delay =
        normalize_qwen_startup_preload_delay_ms(settings.qwen35_startup_preload_delay_ms);
    if settings.qwen35_startup_preload_delay_ms != normalized_qwen35_delay {
        settings.qwen35_startup_preload_delay_ms = normalized_qwen35_delay;
        changed = true;
    } else if settings.qwen35_startup_preload_delay_ms == 1400 {
        // Migrate previous default to immediate preload for better first-use latency.
        settings.qwen35_startup_preload_delay_ms = default_qwen35_startup_preload_delay_ms();
        changed = true;
    }

    let normalized_qwen35_threads = normalize_qwen_max_threads(settings.qwen35_max_threads);
    if settings.qwen35_max_threads != normalized_qwen35_threads {
        settings.qwen35_max_threads = normalized_qwen35_threads;
        changed = true;
    }

    let normalized_qwen35_server_timeout =
        normalize_qwen35_server_ready_timeout_sec(settings.qwen35_server_ready_timeout_sec);
    if settings.qwen35_server_ready_timeout_sec != normalized_qwen35_server_timeout {
        settings.qwen35_server_ready_timeout_sec = normalized_qwen35_server_timeout;
        changed = true;
    }

    let normalized_qwen35_inference_timeout =
        normalize_qwen35_inference_timeout_sec(settings.qwen35_inference_timeout_sec);
    if settings.qwen35_inference_timeout_sec != normalized_qwen35_inference_timeout {
        settings.qwen35_inference_timeout_sec = normalized_qwen35_inference_timeout;
        changed = true;
    }

    let normalized_script_timeout =
        normalize_script_hook_timeout_ms(settings.script_hook_timeout_ms);
    if settings.script_hook_timeout_ms != normalized_script_timeout {
        settings.script_hook_timeout_ms = normalized_script_timeout;
        changed = true;
    }

    let normalized_post_asr_path = settings
        .post_asr_script_path
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if settings.post_asr_script_path != normalized_post_asr_path {
        settings.post_asr_script_path = normalized_post_asr_path;
        changed = true;
    }

    let normalized_post_llm_path = settings
        .post_llm_script_path
        .as_ref()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    if settings.post_llm_script_path != normalized_post_llm_path {
        settings.post_llm_script_path = normalized_post_llm_path;
        changed = true;
    }

    changed
}

pub const SETTINGS_STORE_PATH: &str = "settings_store.json";

pub fn get_default_settings() -> AppSettings {
    #[cfg(target_os = "windows")]
    let default_shortcut = "ctrl+space";
    #[cfg(target_os = "macos")]
    let default_shortcut = "option+space";
    #[cfg(target_os = "linux")]
    let default_shortcut = "ctrl+space";
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let default_shortcut = "alt+space";

    let mut bindings = HashMap::new();
    bindings.insert(
        "transcribe".to_string(),
        ShortcutBinding {
            id: "transcribe".to_string(),
            name: "Transcribe".to_string(),
            description: "Converts your speech into text.".to_string(),
            default_binding: default_shortcut.to_string(),
            current_binding: default_shortcut.to_string(),
        },
    );
    #[cfg(target_os = "windows")]
    let default_post_process_shortcut = "ctrl+shift+space";
    #[cfg(target_os = "macos")]
    let default_post_process_shortcut = "option+shift+space";
    #[cfg(target_os = "linux")]
    let default_post_process_shortcut = "ctrl+shift+space";
    #[cfg(not(any(target_os = "windows", target_os = "macos", target_os = "linux")))]
    let default_post_process_shortcut = "alt+shift+space";

    bindings.insert(
        "transcribe_with_post_process".to_string(),
        ShortcutBinding {
            id: "transcribe_with_post_process".to_string(),
            name: "Transcribe with Post-Processing".to_string(),
            description: "Converts your speech into text and applies AI post-processing."
                .to_string(),
            default_binding: default_post_process_shortcut.to_string(),
            current_binding: default_post_process_shortcut.to_string(),
        },
    );
    bindings.insert(
        "cancel".to_string(),
        ShortcutBinding {
            id: "cancel".to_string(),
            name: "Cancel".to_string(),
            description: "Cancels the current recording.".to_string(),
            default_binding: "escape".to_string(),
            current_binding: "escape".to_string(),
        },
    );

    AppSettings {
        bindings,
        push_to_talk: true,
        audio_feedback: false,
        audio_feedback_volume: default_audio_feedback_volume(),
        sound_theme: default_sound_theme(),
        start_hidden: default_start_hidden(),
        autostart_enabled: default_autostart_enabled(),
        update_checks_enabled: default_update_checks_enabled(),
        selected_model: "".to_string(),
        always_on_microphone: false,
        selected_microphone: None,
        clamshell_microphone: None,
        selected_output_device: None,
        translate_to_english: false,
        selected_language: "auto".to_string(),
        overlay_position: default_overlay_position(),
        debug_mode: false,
        log_level: default_log_level(),
        custom_words: Vec::new(),
        model_unload_timeout: ModelUnloadTimeout::default(),
        qwen_adaptive_unload_enabled: default_qwen_adaptive_unload_enabled(),
        qwen_periodic_keep_warm_enabled: default_qwen_periodic_keep_warm_enabled(),
        qwen_periodic_keep_warm_interval_sec: default_qwen_periodic_keep_warm_interval_sec(),
        qwen_periodic_keep_warm_active_window_min:
            default_qwen_periodic_keep_warm_active_window_min(),
        word_correction_threshold: default_word_correction_threshold(),
        history_limit: default_history_limit(),
        recording_retention_period: default_recording_retention_period(),
        paste_method: PasteMethod::default(),
        clipboard_handling: ClipboardHandling::default(),
        auto_submit: default_auto_submit(),
        auto_submit_key: AutoSubmitKey::default(),
        post_process_enabled: default_post_process_enabled(),
        post_process_provider_id: default_post_process_provider_id(),
        post_process_providers: default_post_process_providers(),
        post_process_api_keys: default_post_process_api_keys(),
        post_process_models: default_post_process_models(),
        post_process_prompts: default_post_process_prompts(),
        post_process_selected_prompt_id: Some("template_translate_english_default".to_string()),
        post_process_system_prompt: default_post_process_system_prompt(),
        post_process_quality: default_post_process_quality(),
        post_process_local_max_tokens: default_post_process_local_max_tokens(),
        post_process_local_temperature: default_post_process_local_temperature(),
        post_process_local_top_p: default_post_process_local_top_p(),
        post_process_local_repetition_penalty: default_post_process_local_repetition_penalty(),
        post_process_local_repetition_context_size:
            default_post_process_local_repetition_context_size(),
        qwen_startup_preload_strategy: default_qwen_startup_preload_strategy(),
        qwen3_startup_preload_enabled: default_qwen3_startup_preload_enabled(),
        qwen3_startup_preload_delay_ms: default_qwen3_startup_preload_delay_ms(),
        qwen3_max_threads: default_qwen3_max_threads(),
        qwen3_server_ready_timeout_sec: default_qwen3_server_ready_timeout_sec(),
        qwen3_warmup_enabled: default_qwen3_warmup_enabled(),
        qwen35_startup_preload_enabled: default_qwen35_startup_preload_enabled(),
        qwen35_startup_preload_delay_ms: default_qwen35_startup_preload_delay_ms(),
        qwen35_warmup_enabled: default_qwen35_warmup_enabled(),
        qwen35_max_threads: default_qwen35_max_threads(),
        qwen35_server_ready_timeout_sec: default_qwen35_server_ready_timeout_sec(),
        qwen35_inference_timeout_sec: default_qwen35_inference_timeout_sec(),
        script_hooks_enabled: default_script_hooks_enabled(),
        post_asr_script_path: None,
        post_llm_script_path: None,
        script_hook_timeout_ms: default_script_hook_timeout_ms(),
        mute_while_recording: false,
        append_trailing_space: false,
        app_language: default_app_language(),
        experimental_enabled: false,
        lazy_stream_close: false,
        keyboard_implementation: KeyboardImplementation::default(),
        show_tray_icon: default_show_tray_icon(),
        paste_delay_ms: default_paste_delay_ms(),
        typing_tool: default_typing_tool(),
        external_script_path: None,
        custom_filler_words: None,
        whisper_accelerator: WhisperAcceleratorSetting::default(),
        ort_accelerator: OrtAcceleratorSetting::default(),
        whisper_gpu_device: default_whisper_gpu_device(),
        extra_recording_buffer_ms: 0,
    }
}

impl AppSettings {
    pub fn active_post_process_provider(&self) -> Option<&PostProcessProvider> {
        self.post_process_providers
            .iter()
            .find(|provider| provider.id == self.post_process_provider_id)
    }

    pub fn post_process_provider(&self, provider_id: &str) -> Option<&PostProcessProvider> {
        self.post_process_providers
            .iter()
            .find(|provider| provider.id == provider_id)
    }

    pub fn post_process_provider_mut(
        &mut self,
        provider_id: &str,
    ) -> Option<&mut PostProcessProvider> {
        self.post_process_providers
            .iter_mut()
            .find(|provider| provider.id == provider_id)
    }
}

pub fn load_or_create_app_settings(app: &AppHandle) -> AppSettings {
    // Initialize store
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    let mut settings = if let Some(settings_value) = store.get("settings") {
        // Parse the entire settings object
        match serde_json::from_value::<AppSettings>(settings_value) {
            Ok(mut settings) => {
                debug!("Found existing settings: {:?}", settings);
                let default_settings = get_default_settings();
                let mut updated = false;

                // Merge default bindings into existing settings
                for (key, value) in default_settings.bindings {
                    if !settings.bindings.contains_key(&key) {
                        debug!("Adding missing binding: {}", key);
                        settings.bindings.insert(key, value);
                        updated = true;
                    }
                }

                if updated {
                    debug!("Settings updated with new bindings");
                    store.set("settings", serde_json::to_value(&settings).unwrap());
                }

                settings
            }
            Err(e) => {
                warn!("Failed to parse settings: {}", e);
                // Fall back to default settings if parsing fails
                let default_settings = get_default_settings();
                store.set("settings", serde_json::to_value(&default_settings).unwrap());
                default_settings
            }
        }
    } else {
        let default_settings = get_default_settings();
        store.set("settings", serde_json::to_value(&default_settings).unwrap());
        default_settings
    };

    if ensure_post_process_defaults(&mut settings) {
        store.set("settings", serde_json::to_value(&settings).unwrap());
    }

    settings
}

pub fn get_settings(app: &AppHandle) -> AppSettings {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    let mut settings = if let Some(settings_value) = store.get("settings") {
        serde_json::from_value::<AppSettings>(settings_value).unwrap_or_else(|_| {
            let default_settings = get_default_settings();
            store.set("settings", serde_json::to_value(&default_settings).unwrap());
            default_settings
        })
    } else {
        let default_settings = get_default_settings();
        store.set("settings", serde_json::to_value(&default_settings).unwrap());
        default_settings
    };

    if ensure_post_process_defaults(&mut settings) {
        store.set("settings", serde_json::to_value(&settings).unwrap());
    }

    settings
}

pub fn write_settings(app: &AppHandle, settings: AppSettings) {
    let store = app
        .store(crate::portable::store_path(SETTINGS_STORE_PATH))
        .expect("Failed to initialize store");

    store.set("settings", serde_json::to_value(&settings).unwrap());
}

pub fn get_bindings(app: &AppHandle) -> HashMap<String, ShortcutBinding> {
    let settings = get_settings(app);

    settings.bindings
}

pub fn get_stored_binding(app: &AppHandle, id: &str) -> ShortcutBinding {
    let bindings = get_bindings(app);

    let binding = bindings.get(id).unwrap().clone();

    binding
}

pub fn get_history_limit(app: &AppHandle) -> usize {
    let settings = get_settings(app);
    settings.history_limit
}

pub fn get_recording_retention_period(app: &AppHandle) -> RecordingRetentionPeriod {
    let settings = get_settings(app);
    settings.recording_retention_period
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_settings_disable_auto_submit() {
        let settings = get_default_settings();
        assert!(!settings.auto_submit);
        assert_eq!(settings.auto_submit_key, AutoSubmitKey::Enter);
    }

    #[test]
    fn ensure_post_process_defaults_keeps_openai_selection_without_key_or_model() {
        let mut settings = get_default_settings();
        settings.post_process_provider_id = "openai".to_string();
        settings
            .post_process_api_keys
            .insert("openai".to_string(), String::new());
        settings
            .post_process_models
            .insert("openai".to_string(), String::new());

        let _ = ensure_post_process_defaults(&mut settings);

        assert_eq!(settings.post_process_provider_id, "openai");
    }

    #[test]
    fn ensure_post_process_defaults_migrates_local_provider_label() {
        let mut settings = get_default_settings();
        if let Some(local) = settings.post_process_provider_mut(LOCAL_QWEN35_PROVIDER_ID) {
            local.label = "Local Qwen3.5".to_string();
        }

        let changed = ensure_post_process_defaults(&mut settings);
        assert!(changed);
        let local = settings
            .post_process_provider(LOCAL_QWEN35_PROVIDER_ID)
            .expect("local provider must exist");
        assert_eq!(local.label, "Local");
    }

    #[test]
    fn ensure_post_process_defaults_migrates_legacy_translation_template_text() {
        let mut settings = get_default_settings();
        let translate = settings
            .post_process_prompts
            .iter_mut()
            .find(|prompt| prompt.id == "template_translate_english_default")
            .expect("translation template should exist");
        translate.prompt = "${output}".to_string();

        let changed = ensure_post_process_defaults(&mut settings);
        assert!(changed);

        let translate = settings
            .post_process_prompts
            .iter()
            .find(|prompt| prompt.id == "template_translate_english_default")
            .expect("translation template should exist");
        assert!(translate
            .prompt
            .contains("Translate the transcript into natural English."));
        assert!(translate.prompt.contains("一二三四五六七 -> 1234567"));
    }

    #[test]
    fn ensure_post_process_defaults_migrates_legacy_translation_system_prompt() {
        let mut settings = get_default_settings();
        settings.post_process_system_prompt = "You are a strict transcript translator.\nTask:\nTranslate incoming transcript text into natural English.\nOutput rules:\n1. Output English only.\n2. Always translate, including very short inputs (single-word or 1-3 character phrases).\n3. For short Chinese interjections, produce concise natural English (e.g. 好 -> okay, 棒 -> great, 行 -> okay).\n4. Convert Chinese numerals to English words when short and standalone (e.g. 一二三 -> one two three).\n5. Preserve existing English words, product names, acronyms, and numbers accurately (e.g. HANDY, Qwen3.5, 1.7B).\n6. Do not include reasoning, <think>, bullet points, or explanations.\n7. Return only the final translated text.".to_string();

        let changed = ensure_post_process_defaults(&mut settings);
        assert!(changed);
        assert_eq!(
            settings.post_process_system_prompt,
            default_post_process_system_prompt()
        );
    }
}
