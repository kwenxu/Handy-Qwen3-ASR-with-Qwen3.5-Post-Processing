//! Keyboard shortcut management module
//!
//! This module provides a unified interface for keyboard shortcuts with
//! multiple backend implementations:
//!
//! - `tauri`: Uses Tauri's built-in global-shortcut plugin
//! - `handy_keys`: Uses the handy-keys library for more control
//!
//! The active implementation is determined by the `keyboard_implementation`
//! setting and can be changed at runtime.

mod handler;
pub mod handy_keys;
mod tauri_impl;

use log::{error, info, warn};
use serde::Serialize;
use specta::Type;
use std::fs;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager};
use tauri_plugin_autostart::ManagerExt;

use crate::managers::post_process_model::PostProcessModelManager;
use crate::managers::qwen35_post_manager::Qwen35PostManager;
use crate::managers::script_hook::{run_script_hook, ScriptHookContext, ScriptHookStage};
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::settings::APPLE_INTELLIGENCE_DEFAULT_MODEL_ID;
use crate::settings::{
    self, get_settings, AutoSubmitKey, ClipboardHandling, KeyboardImplementation, LLMPrompt,
    OverlayPosition, PasteMethod, ShortcutBinding, SoundTheme, TypingTool,
    APPLE_INTELLIGENCE_PROVIDER_ID, LOCAL_QWEN35_PROVIDER_ID,
};
use crate::tray;

const DEFAULT_SCRIPT_HOOKS_DIR: &str = "script-hooks";
const DEFAULT_ASR_SCRIPT_FILE: &str = "asr_post_hook.py";
const DEFAULT_LLM_SCRIPT_FILE: &str = "llm_post_hook.py";
const DESKTOP_EXPORT_ASR_SCRIPT_FILE: &str = "Handy-asr-post-script.py";
const DESKTOP_EXPORT_LLM_SCRIPT_FILE: &str = "Handy-llm-post-script.py";
const DEFAULT_ASR_SCRIPT_TEMPLATE: &str = r#"#!/usr/bin/env python3
# Handy external script template (stage: asr_post)
# Layering:
# - System/User prompts are model-side constraints.
# - This external script is for replaceable business rules.
# - Built-in logic remains the final safety fallback.
# How APP runs scripts:
# - .py           -> python3 <script_path>
# - .js/.mjs/.cjs -> node <script_path>
# - .sh           -> bash <script_path>
# - other suffix  -> run directly as executable (e.g. Rust compiled binary)
#
# stdin:  one JSON line
#   {"stage":"asr_post|llm_post","text":"...","lang":"...","model_id":"...","provider_id":"...","prompt_id":"...","system_prompt":"...","user_prompt_template":"...","metadata":{...}}
# stdout: plain text OR JSON {"text":"..."} (recommended)
# fallback: timeout / error / invalid output -> APP falls back to original text
import json
import re
import sys
from typing import List, Tuple


# ---- Tunable defaults (Toolkit-style anti-hallucination cleanup) ----
# Repeated-char suppression: "aaaaaaaaaaaa" -> "a"
CHAR_REPEAT_THRESHOLD = 14
# Repeated-pattern suppression: "abcabcabcabcabcabc" -> "abc"
PATTERN_REPEAT_THRESHOLD = 6
PATTERN_MAX_LEN = 20
# Consecutive repeated-line suppression threshold
LINE_REPEAT_THRESHOLD = 3

NOISE_LATIN_A_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])a{2,}(?:[-—~!！?？]+)?(?![A-Za-z0-9])"
)
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def collapse_char_repeats(text: str, threshold: int) -> str:
    if threshold <= 1 or len(text) < threshold:
        return text
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        count = 1
        while i + count < n and text[i + count] == text[i]:
            count += 1
        if count > threshold:
            out.append(text[i])
        else:
            out.append(text[i : i + count])
        i += count
    return "".join(out)


def collapse_pattern_repeats(text: str, threshold: int, max_len: int) -> str:
    n = len(text)
    if threshold < 2 or n < threshold * 2:
        return text

    i = 0
    out: List[str] = []
    while i <= n - threshold * 2:
        found = False
        for k in range(1, max_len + 1):
            if i + k * threshold > n:
                break
            pattern = text[i : i + k]

            valid = True
            for rep in range(1, threshold):
                start = i + rep * k
                if text[start : start + k] != pattern:
                    valid = False
                    break
            if not valid:
                continue

            end = i + k * threshold
            while end + k <= n and text[end : end + k] == pattern:
                end += k
            out.append(pattern)
            i = end
            found = True
            break

        if not found:
            out.append(text[i])
            i += 1

    if i < n:
        out.append(text[i:])
    return "".join(out)


def collapse_repeated_lines(text: str, threshold: int) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()

    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        repeats = j - i
        if repeats >= threshold:
            out.append(lines[i])
        else:
            out.extend(lines[i:j])
        i = j
    return "\n".join(out)


def normalize_spacing(text: str) -> str:
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def process_asr_text(text: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    out = text

    next_out = collapse_char_repeats(out, CHAR_REPEAT_THRESHOLD)
    if next_out != out:
        warnings.append("collapse_char_repeats")
        out = next_out

    next_out = collapse_pattern_repeats(out, PATTERN_REPEAT_THRESHOLD, PATTERN_MAX_LEN)
    if next_out != out:
        warnings.append("collapse_pattern_repeats")
        out = next_out

    next_out = NOISE_LATIN_A_RE.sub(" ", out)
    if next_out != out:
        warnings.append("remove_noise_a_tokens")
        out = next_out

    next_out = collapse_repeated_lines(out, LINE_REPEAT_THRESHOLD)
    if next_out != out:
        warnings.append("collapse_repeated_lines")
        out = next_out

    out = normalize_spacing(out)
    return out, warnings


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"text": ""}, ensure_ascii=False))
        return

    data = json.loads(raw)
    stage = str(data.get("stage", ""))
    text = str(data.get("text", ""))
    # lang = data.get("lang")
    # model_id = data.get("model_id")
    # metadata = data.get("metadata", {})

    # Keep stage guard so one file can be safely reused in both stages.
    if stage != "asr_post":
        print(json.dumps({"text": text}, ensure_ascii=False))
        return

    cleaned, warnings = process_asr_text(text)
    print(json.dumps({"text": cleaned, "warnings": warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
"#;
const DEFAULT_LLM_SCRIPT_TEMPLATE: &str = r#"#!/usr/bin/env python3
# Handy external script template (stage: llm_post)
# Layering:
# - System prompt: global contract (highest priority).
# - User prompt template: task/style instructions.
# - This external script: replaceable post-clean rules.
# - Built-in fallback: safety and stability guardrail.
# See the ASR template header for runner mapping and protocol details.
import json
import re
import sys
from typing import List, Tuple


# ---- Tunable defaults ----
CHAR_REPEAT_THRESHOLD = 14
PATTERN_REPEAT_THRESHOLD = 6
PATTERN_MAX_LEN = 20
LINE_REPEAT_THRESHOLD = 3

RULE_HEADER_RE = re.compile(
    r"^(要求|规则|输入|输出|目标|task|rules?|input|output|objective)\s*[:：]?$",
    re.IGNORECASE,
)
ENUM_LINE_RE = re.compile(r"^\s*\d+\s*[\.、\)\]]\s*(.*)$")
MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
MULTI_NEWLINE_RE = re.compile(r"\n{3,}")
NOISE_LATIN_A_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])a{2,}(?:[-—~!！?？]+)?(?![A-Za-z0-9])"
)
LEAK_KEYWORDS = (
    "只输出最终",
    "仅输出最终",
    "不要解释",
    "不要复述",
    "不执行摘要",
    "output contract",
    "return only the final",
    "do not include reasoning",
    "do not summarize",
)


def collapse_char_repeats(text: str, threshold: int) -> str:
    if threshold <= 1 or len(text) < threshold:
        return text
    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        count = 1
        while i + count < n and text[i + count] == text[i]:
            count += 1
        if count > threshold:
            out.append(text[i])
        else:
            out.append(text[i : i + count])
        i += count
    return "".join(out)


def collapse_pattern_repeats(text: str, threshold: int, max_len: int) -> str:
    n = len(text)
    if threshold < 2 or n < threshold * 2:
        return text

    i = 0
    out: List[str] = []
    while i <= n - threshold * 2:
        found = False
        for k in range(1, max_len + 1):
            if i + k * threshold > n:
                break
            pattern = text[i : i + k]

            valid = True
            for rep in range(1, threshold):
                start = i + rep * k
                if text[start : start + k] != pattern:
                    valid = False
                    break
            if not valid:
                continue

            end = i + k * threshold
            while end + k <= n and text[end : end + k] == pattern:
                end += k
            out.append(pattern)
            i = end
            found = True
            break

        if not found:
            out.append(text[i])
            i += 1

    if i < n:
        out.append(text[i:])
    return "".join(out)


def collapse_repeated_lines(text: str, threshold: int) -> str:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip()

    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        j = i + 1
        while j < n and lines[j] == lines[i]:
            j += 1
        repeats = j - i
        if repeats >= threshold:
            out.append(lines[i])
        else:
            out.extend(lines[i:j])
        i = j
    return "\n".join(out)


def is_probable_template_rule(line: str) -> bool:
    lower = line.lower()
    matched = ENUM_LINE_RE.match(line)
    body = matched.group(1).strip().lower() if matched else lower
    if matched and any(key in body for key in LEAK_KEYWORDS):
        return True
    if matched and (
        body.startswith("只处理")
        or body.startswith("当前转录")
        or body.startswith("保留全部")
        or body.startswith("短文本保护")
        or body.startswith("排版策略")
        or body.startswith("数字规范")
        or body.startswith("only process")
        or body.startswith("preserve")
        or body.startswith("return only")
    ):
        return True
    return False


def strip_template_leakage(text: str) -> Tuple[str, bool]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return text.strip(), False

    kept: List[str] = []
    removed = 0
    for line in lines:
        lower = line.lower()
        if RULE_HEADER_RE.match(line):
            removed += 1
            continue
        if lower in ("input data", "input data:", "canonical transcript data"):
            removed += 1
            continue
        if any(marker in lower for marker in ("begin_transcript", "end_transcript")):
            removed += 1
            continue
        if any(
            marker in lower
            for marker in (
                "optional_context_hints",
                "end_optional_context_hints",
                "optional_history_context",
                "user_history_context",
            )
        ):
            removed += 1
            continue
        if is_probable_template_rule(line):
            removed += 1
            continue
        kept.append(line)

    if kept:
        return "\n".join(kept), removed > 0
    return text.strip(), False


def normalize_spacing(text: str) -> str:
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = MULTI_SPACE_RE.sub(" ", text)
    text = MULTI_NEWLINE_RE.sub("\n\n", text)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def is_degenerate_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    informative = [ch.lower() for ch in stripped if ch.isalnum() or "\u4e00" <= ch <= "\u9fff"]
    if not informative:
        return True
    unique = set(informative)
    if len(informative) <= 4 and len(unique) <= 2:
        return True
    if len(informative) >= 2 and len(unique) == 1:
        return True
    return False


def process_llm_text(text: str) -> Tuple[str, List[str]]:
    warnings: List[str] = []
    original = normalize_spacing(text)
    out = text

    next_out = collapse_char_repeats(out, CHAR_REPEAT_THRESHOLD)
    if next_out != out:
        warnings.append("collapse_char_repeats")
        out = next_out

    next_out = collapse_pattern_repeats(out, PATTERN_REPEAT_THRESHOLD, PATTERN_MAX_LEN)
    if next_out != out:
        warnings.append("collapse_pattern_repeats")
        out = next_out

    next_out = NOISE_LATIN_A_RE.sub(" ", out)
    if next_out != out:
        warnings.append("remove_noise_a_tokens")
        out = next_out

    next_out, removed_template = strip_template_leakage(out)
    if removed_template:
        warnings.append("strip_template_leakage")
        out = next_out
    else:
        out = next_out

    next_out = collapse_repeated_lines(out, LINE_REPEAT_THRESHOLD)
    if next_out != out:
        warnings.append("collapse_repeated_lines")
        out = next_out

    out = normalize_spacing(out)
    if not out and original:
        warnings.append("fallback_original_after_empty_cleanup")
        return original, warnings
    if is_degenerate_text(out) and not is_degenerate_text(original):
        warnings.append("fallback_original_after_degenerate_cleanup")
        return original, warnings
    return out, warnings


def main() -> None:
    raw = sys.stdin.read().strip()
    if not raw:
        print(json.dumps({"text": ""}, ensure_ascii=False))
        return

    data = json.loads(raw)
    stage = str(data.get("stage", ""))
    text = str(data.get("text", ""))
    # prompt_id = data.get("prompt_id")
    # system_prompt = data.get("system_prompt")
    # user_prompt_template = data.get("user_prompt_template")
    # metadata = data.get("metadata", {})

    if stage != "llm_post":
        print(json.dumps({"text": text}, ensure_ascii=False))
        return

    cleaned, warnings = process_llm_text(text)
    print(json.dumps({"text": cleaned, "warnings": warnings}, ensure_ascii=False))


if __name__ == "__main__":
    main()
"#;

// Note: Commands are accessed via shortcut::handy_keys:: in lib.rs

/// Initialize shortcuts using the configured implementation
pub fn init_shortcuts(app: &AppHandle) {
    let user_settings = settings::load_or_create_app_settings(app);

    // Check which implementation to use
    match user_settings.keyboard_implementation {
        KeyboardImplementation::Tauri => {
            tauri_impl::init_shortcuts(app);
        }
        KeyboardImplementation::HandyKeys => {
            if let Err(e) = handy_keys::init_shortcuts(app) {
                error!("Failed to initialize handy-keys shortcuts: {}", e);
                // Fall back to Tauri implementation and persist this fallback
                warn!("Falling back to Tauri global shortcut implementation and saving fallback to settings");

                // Update settings to persist the fallback so we don't retry HandyKeys on next launch
                let mut settings = settings::get_settings(app);
                settings.keyboard_implementation = KeyboardImplementation::Tauri;
                settings::write_settings(app, settings);

                tauri_impl::init_shortcuts(app);
            }
        }
    }
}

/// Register the cancel shortcut (called when recording starts)
pub fn register_cancel_shortcut(app: &AppHandle) {
    let settings = get_settings(app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => tauri_impl::register_cancel_shortcut(app),
        KeyboardImplementation::HandyKeys => handy_keys::register_cancel_shortcut(app),
    }
}

/// Unregister the cancel shortcut (called when recording stops)
pub fn unregister_cancel_shortcut(app: &AppHandle) {
    let settings = get_settings(app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => tauri_impl::unregister_cancel_shortcut(app),
        KeyboardImplementation::HandyKeys => handy_keys::unregister_cancel_shortcut(app),
    }
}

/// Register a shortcut using the appropriate implementation
pub fn register_shortcut(app: &AppHandle, binding: ShortcutBinding) -> Result<(), String> {
    let settings = get_settings(app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => tauri_impl::register_shortcut(app, binding),
        KeyboardImplementation::HandyKeys => handy_keys::register_shortcut(app, binding),
    }
}

/// Unregister a shortcut using the appropriate implementation
pub fn unregister_shortcut(app: &AppHandle, binding: ShortcutBinding) -> Result<(), String> {
    let settings = get_settings(app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => tauri_impl::unregister_shortcut(app, binding),
        KeyboardImplementation::HandyKeys => handy_keys::unregister_shortcut(app, binding),
    }
}

// ============================================================================
// Binding Management Commands
// ============================================================================

#[derive(Serialize, Type)]
pub struct BindingResponse {
    success: bool,
    binding: Option<ShortcutBinding>,
    error: Option<String>,
}

#[tauri::command]
#[specta::specta]
pub fn change_binding(
    app: AppHandle,
    id: String,
    binding: String,
) -> Result<BindingResponse, String> {
    // Reject empty bindings — every shortcut should have a value
    if binding.trim().is_empty() {
        return Err("Binding cannot be empty".to_string());
    }

    let mut settings = settings::get_settings(&app);

    // Get the binding to modify, or create it from defaults if it doesn't exist
    let binding_to_modify = match settings.bindings.get(&id) {
        Some(binding) => binding.clone(),
        None => {
            // Try to get the default binding for this id
            let default_settings = settings::get_default_settings();
            match default_settings.bindings.get(&id) {
                Some(default_binding) => {
                    warn!(
                        "Binding '{}' not found in settings, creating from defaults",
                        id
                    );
                    default_binding.clone()
                }
                None => {
                    let error_msg = format!("Binding with id '{}' not found in defaults", id);
                    warn!("change_binding error: {}", error_msg);
                    return Ok(BindingResponse {
                        success: false,
                        binding: None,
                        error: Some(error_msg),
                    });
                }
            }
        }
    };

    // If this is the cancel binding, just update the settings and return
    // It's managed dynamically, so we don't register/unregister here
    if id == "cancel" {
        if let Some(mut b) = settings.bindings.get(&id).cloned() {
            b.current_binding = binding;
            settings.bindings.insert(id.clone(), b.clone());
            settings::write_settings(&app, settings);
            return Ok(BindingResponse {
                success: true,
                binding: Some(b.clone()),
                error: None,
            });
        }
    }

    // Unregister the existing binding
    if let Err(e) = unregister_shortcut(&app, binding_to_modify.clone()) {
        let error_msg = format!("Failed to unregister shortcut: {}", e);
        error!("change_binding error: {}", error_msg);
    }

    // Validate the new shortcut for the current keyboard implementation
    if let Err(e) = validate_shortcut_for_implementation(&binding, settings.keyboard_implementation)
    {
        warn!("change_binding validation error: {}", e);
        return Err(e);
    }

    // Create an updated binding
    let mut updated_binding = binding_to_modify;
    updated_binding.current_binding = binding;

    // Register the new binding
    if let Err(e) = register_shortcut(&app, updated_binding.clone()) {
        let error_msg = format!("Failed to register shortcut: {}", e);
        error!("change_binding error: {}", error_msg);
        return Ok(BindingResponse {
            success: false,
            binding: None,
            error: Some(error_msg),
        });
    }

    // Update the binding in the settings
    settings.bindings.insert(id, updated_binding.clone());

    // Save the settings
    settings::write_settings(&app, settings);

    // Return the updated binding
    Ok(BindingResponse {
        success: true,
        binding: Some(updated_binding),
        error: None,
    })
}

#[tauri::command]
#[specta::specta]
pub fn reset_binding(app: AppHandle, id: String) -> Result<BindingResponse, String> {
    let binding = settings::get_stored_binding(&app, &id);
    change_binding(app, id, binding.default_binding)
}

/// Temporarily unregister a binding while the user is editing it in the UI.
/// This avoids firing the action while keys are being recorded.
#[tauri::command]
#[specta::specta]
pub fn suspend_binding(app: AppHandle, id: String) -> Result<(), String> {
    if let Some(b) = settings::get_bindings(&app).get(&id).cloned() {
        if let Err(e) = unregister_shortcut(&app, b) {
            error!("suspend_binding error for id '{}': {}", id, e);
            return Err(e);
        }
    }
    Ok(())
}

/// Re-register the binding after the user has finished editing.
#[tauri::command]
#[specta::specta]
pub fn resume_binding(app: AppHandle, id: String) -> Result<(), String> {
    if let Some(b) = settings::get_bindings(&app).get(&id).cloned() {
        if let Err(e) = register_shortcut(&app, b) {
            error!("resume_binding error for id '{}': {}", id, e);
            return Err(e);
        }
    }
    Ok(())
}

// ============================================================================
// Keyboard Implementation Switching
// ============================================================================

/// Result of changing keyboard implementation
#[derive(Serialize, Type)]
pub struct ImplementationChangeResult {
    pub success: bool,
    /// List of binding IDs that were reset to defaults due to incompatibility
    pub reset_bindings: Vec<String>,
}

/// Change the keyboard implementation with runtime switching.
/// This will unregister all shortcuts from the old implementation,
/// validate shortcuts for the new implementation (resetting invalid ones to defaults),
/// and register them with the new implementation.
#[tauri::command]
#[specta::specta]
pub fn change_keyboard_implementation_setting(
    app: AppHandle,
    implementation: String,
) -> Result<ImplementationChangeResult, String> {
    let current_settings = settings::get_settings(&app);
    let current_impl = current_settings.keyboard_implementation;
    let new_impl = parse_keyboard_implementation(&implementation);

    // If same implementation, nothing to do
    if current_impl == new_impl {
        return Ok(ImplementationChangeResult {
            success: true,
            reset_bindings: vec![],
        });
    }

    info!(
        "Switching keyboard implementation from {:?} to {:?}",
        current_impl, new_impl
    );

    // Unregister all shortcuts from the current implementation
    unregister_all_shortcuts(&app, current_impl);

    // Update the setting
    let mut settings = settings::get_settings(&app);
    settings.keyboard_implementation = new_impl;
    settings::write_settings(&app, settings);

    // Initialize new implementation if needed (HandyKeys needs state)
    if new_impl == KeyboardImplementation::HandyKeys {
        if initialize_handy_keys_with_rollback(&app)? {
            // Shortcuts already registered during init
            return Ok(ImplementationChangeResult {
                success: true,
                reset_bindings: vec![],
            });
        }
    }

    // Register all shortcuts with new implementation, resetting invalid ones
    let reset_bindings = register_all_shortcuts_for_implementation(&app, new_impl);

    // Emit event to notify frontend of the change
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "keyboard_implementation",
            "value": implementation,
            "reset_bindings": reset_bindings
        }),
    );

    info!("Keyboard implementation switched to {:?}", new_impl);

    Ok(ImplementationChangeResult {
        success: true,
        reset_bindings,
    })
}

/// Get the current keyboard implementation
#[tauri::command]
#[specta::specta]
pub fn get_keyboard_implementation(app: AppHandle) -> String {
    let settings = settings::get_settings(&app);
    match settings.keyboard_implementation {
        KeyboardImplementation::Tauri => "tauri".to_string(),
        KeyboardImplementation::HandyKeys => "handy_keys".to_string(),
    }
}

// ============================================================================
// Validation Helpers
// ============================================================================

pub(crate) fn schedule_active_local_post_process_preload(app: &AppHandle, reason: &str) {
    let runtime_settings = settings::get_settings(app);
    if !runtime_settings.post_process_enabled
        || runtime_settings.post_process_provider_id != LOCAL_QWEN35_PROVIDER_ID
    {
        return;
    }

    let selected_model = runtime_settings
        .post_process_models
        .get(LOCAL_QWEN35_PROVIDER_ID)
        .cloned()
        .unwrap_or_default();
    if selected_model.trim().is_empty() {
        return;
    }

    let post_model_manager = app.state::<Arc<PostProcessModelManager>>();
    if !post_model_manager.check_model_cached(&selected_model) {
        return;
    }

    let app_handle = app.clone();
    let model_id = selected_model;
    let reason_text = reason.to_string();
    tauri::async_runtime::spawn_blocking(move || {
        if let Some(manager_state) = app_handle.try_state::<Arc<Qwen35PostManager>>() {
            let manager = manager_state.inner().clone();
            info!(
                "Scheduling active local Qwen3.5 preload (reason={}, model={})",
                reason_text, model_id
            );
            if let Err(err) = manager.preload_model(&model_id) {
                warn!(
                    "Active local Qwen3.5 preload failed (reason={}, model={}): {}",
                    reason_text, model_id, err
                );
            }
        } else {
            warn!(
                "Qwen35PostManager unavailable; cannot preload active local model {}",
                model_id
            );
        }
    });
}

/// Validate a shortcut for a specific implementation
fn validate_shortcut_for_implementation(
    raw: &str,
    implementation: KeyboardImplementation,
) -> Result<(), String> {
    match implementation {
        KeyboardImplementation::Tauri => tauri_impl::validate_shortcut(raw),
        KeyboardImplementation::HandyKeys => handy_keys::validate_shortcut(raw),
    }
}

/// Parse a keyboard implementation string into the enum
fn parse_keyboard_implementation(s: &str) -> KeyboardImplementation {
    match s {
        "tauri" => KeyboardImplementation::Tauri,
        "handy_keys" => KeyboardImplementation::HandyKeys,
        other => {
            warn!(
                "Invalid keyboard implementation '{}', defaulting to tauri",
                other
            );
            KeyboardImplementation::Tauri
        }
    }
}

/// Unregister all shortcuts for the current implementation
fn unregister_all_shortcuts(app: &AppHandle, implementation: KeyboardImplementation) {
    let bindings = settings::get_bindings(app);

    for (id, binding) in bindings {
        // Skip cancel shortcut as it's dynamically registered
        if id == "cancel" {
            continue;
        }

        let result = match implementation {
            KeyboardImplementation::Tauri => tauri_impl::unregister_shortcut(app, binding),
            KeyboardImplementation::HandyKeys => handy_keys::unregister_shortcut(app, binding),
        };

        if let Err(e) = result {
            warn!(
                "Failed to unregister shortcut '{}' during switch: {}",
                id, e
            );
        }
    }
}

/// Register all shortcuts for a specific implementation, validating and resetting invalid ones
fn register_all_shortcuts_for_implementation(
    app: &AppHandle,
    implementation: KeyboardImplementation,
) -> Vec<String> {
    let mut reset_bindings = Vec::new();
    let default_bindings = settings::get_default_settings().bindings;
    let mut current_settings = settings::get_settings(app);

    for (id, default_binding) in &default_bindings {
        // Skip cancel shortcut as it's dynamically registered
        if id == "cancel" {
            continue;
        }

        // Skip post-processing shortcut when the feature is disabled
        if id == "transcribe_with_post_process" && !current_settings.post_process_enabled {
            continue;
        }

        let mut binding = current_settings
            .bindings
            .get(id)
            .cloned()
            .unwrap_or_else(|| default_binding.clone());

        // Validate the shortcut for the target implementation
        if let Err(e) =
            validate_shortcut_for_implementation(&binding.current_binding, implementation)
        {
            info!(
                "Shortcut '{}' ({}) is invalid for {:?}: {}. Resetting to default.",
                id, binding.current_binding, implementation, e
            );

            // Reset to default
            binding.current_binding = default_binding.current_binding.clone();
            current_settings
                .bindings
                .insert(id.clone(), binding.clone());
            reset_bindings.push(id.clone());
        }

        // Register with the appropriate implementation
        let result = match implementation {
            KeyboardImplementation::Tauri => tauri_impl::register_shortcut(app, binding),
            KeyboardImplementation::HandyKeys => handy_keys::register_shortcut(app, binding),
        };

        if let Err(e) = result {
            error!(
                "Failed to register shortcut '{}' for {:?}: {}",
                id, implementation, e
            );
        }
    }

    // Save settings if any bindings were reset
    if !reset_bindings.is_empty() {
        settings::write_settings(app, current_settings);
    }

    reset_bindings
}

/// Initialize HandyKeys if not already initialized, with rollback on failure
fn initialize_handy_keys_with_rollback(app: &AppHandle) -> Result<bool, String> {
    if app.try_state::<handy_keys::HandyKeysState>().is_some() {
        return Ok(false); // Already initialized, caller should continue
    }

    if let Err(e) = handy_keys::init_shortcuts(app) {
        error!("Failed to initialize HandyKeys: {}", e);
        // Rollback to Tauri
        let mut settings = settings::get_settings(app);
        settings.keyboard_implementation = KeyboardImplementation::Tauri;
        settings::write_settings(app, settings);
        tauri_impl::init_shortcuts(app);
        return Err(format!(
            "Failed to initialize HandyKeys: {}. Reverted to Tauri.",
            e
        ));
    }

    // init_shortcuts already registered shortcuts
    Ok(true)
}

// ============================================================================
// General Settings Commands
// ============================================================================

#[tauri::command]
#[specta::specta]
pub fn change_ptt_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.push_to_talk = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_audio_feedback_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.audio_feedback = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_audio_feedback_volume_setting(app: AppHandle, volume: f32) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.audio_feedback_volume = volume;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_sound_theme_setting(app: AppHandle, theme: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match theme.as_str() {
        "marimba" => SoundTheme::Marimba,
        "pop" => SoundTheme::Pop,
        "custom" => SoundTheme::Custom,
        other => {
            warn!("Invalid sound theme '{}', defaulting to marimba", other);
            SoundTheme::Marimba
        }
    };
    settings.sound_theme = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_translate_to_english_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.translate_to_english = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_selected_language_setting(app: AppHandle, language: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.selected_language = language;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_overlay_position_setting(app: AppHandle, position: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match position.as_str() {
        "none" => OverlayPosition::None,
        "top" => OverlayPosition::Top,
        "bottom" => OverlayPosition::Bottom,
        other => {
            warn!("Invalid overlay position '{}', defaulting to bottom", other);
            OverlayPosition::Bottom
        }
    };
    settings.overlay_position = parsed;
    settings::write_settings(&app, settings);

    // Update overlay position without recreating window
    crate::utils::update_overlay_position(&app);

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_debug_mode_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.debug_mode = enabled;
    settings::write_settings(&app, settings);

    // Emit event to notify frontend of debug mode change
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "debug_mode",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_start_hidden_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.start_hidden = enabled;
    settings::write_settings(&app, settings);

    // Notify frontend
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "start_hidden",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_autostart_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.autostart_enabled = enabled;
    settings::write_settings(&app, settings);

    // Apply the autostart setting immediately
    let autostart_manager = app.autolaunch();
    if enabled {
        let _ = autostart_manager.enable();
    } else {
        let _ = autostart_manager.disable();
    }

    // Notify frontend
    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "autostart_enabled",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_update_checks_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.update_checks_enabled = enabled;
    settings::write_settings(&app, settings);

    let _ = app.emit(
        "settings-changed",
        serde_json::json!({
            "setting": "update_checks_enabled",
            "value": enabled
        }),
    );

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn update_custom_words(app: AppHandle, words: Vec<String>) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.custom_words = words;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_word_correction_threshold_setting(
    app: AppHandle,
    threshold: f64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.word_correction_threshold = threshold;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_extra_recording_buffer_setting(app: AppHandle, ms: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.extra_recording_buffer_ms = ms;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_paste_delay_ms_setting(app: AppHandle, ms: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.paste_delay_ms = ms;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_paste_method_setting(app: AppHandle, method: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match method.as_str() {
        "ctrl_v" => PasteMethod::CtrlV,
        "direct" => PasteMethod::Direct,
        "none" => PasteMethod::None,
        "shift_insert" => PasteMethod::ShiftInsert,
        "ctrl_shift_v" => PasteMethod::CtrlShiftV,
        "external_script" => PasteMethod::ExternalScript,
        other => {
            warn!("Invalid paste method '{}', defaulting to ctrl_v", other);
            PasteMethod::CtrlV
        }
    };
    settings.paste_method = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn get_available_typing_tools() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        crate::clipboard::get_available_typing_tools()
    }
    #[cfg(not(target_os = "linux"))]
    {
        vec!["auto".to_string()]
    }
}

#[tauri::command]
#[specta::specta]
pub fn change_typing_tool_setting(app: AppHandle, tool: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match tool.as_str() {
        "auto" => TypingTool::Auto,
        "wtype" => TypingTool::Wtype,
        "kwtype" => TypingTool::Kwtype,
        "dotool" => TypingTool::Dotool,
        "ydotool" => TypingTool::Ydotool,
        "xdotool" => TypingTool::Xdotool,
        other => {
            warn!("Invalid typing tool '{}', defaulting to auto", other);
            TypingTool::Auto
        }
    };
    settings.typing_tool = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_external_script_path_setting(
    app: AppHandle,
    path: Option<String>,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.external_script_path = path;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_script_hooks_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.script_hooks_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_asr_script_path_setting(
    app: AppHandle,
    path: Option<String>,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_asr_script_path = path
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_llm_script_path_setting(
    app: AppHandle,
    path: Option<String>,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_llm_script_path = path
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_script_hook_timeout_ms_setting(app: AppHandle, value: u64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.script_hook_timeout_ms = settings::normalize_script_hook_timeout_ms(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[derive(Serialize, Type)]
pub struct ScriptHookTemplatePaths {
    pub directory: String,
    pub post_asr_script_path: String,
    pub post_llm_script_path: String,
}

#[derive(Serialize, Type)]
pub struct ScriptHookTestResult {
    pub stage: String,
    pub used_script_path: String,
    pub input_text: String,
    pub output_text: String,
    pub duration_ms: u64,
}

fn parse_script_hook_stage(stage: &str) -> Result<ScriptHookStage, String> {
    match stage.trim().to_ascii_lowercase().as_str() {
        "asr_post" | "asr" | "post_asr" => Ok(ScriptHookStage::AsrPost),
        "llm_post" | "llm" | "post_llm" => Ok(ScriptHookStage::LlmPost),
        other => Err(format!(
            "Unsupported stage '{}'. Use 'asr_post' or 'llm_post'.",
            other
        )),
    }
}

fn ensure_script_hooks_dir(app: &AppHandle) -> Result<std::path::PathBuf, String> {
    let dir = crate::portable::resolve_app_data(app, DEFAULT_SCRIPT_HOOKS_DIR)
        .map_err(|err| format!("Failed to resolve script-hooks directory: {}", err))?;
    fs::create_dir_all(&dir).map_err(|err| {
        format!(
            "Failed to create script-hooks directory {}: {}",
            dir.display(),
            err
        )
    })?;
    Ok(dir)
}

fn write_template_file(
    path: &std::path::Path,
    content: &str,
    overwrite: bool,
) -> Result<(), String> {
    if path.exists() && !overwrite {
        return Ok(());
    }

    fs::write(path, content)
        .map_err(|err| format!("Failed to write template {}: {}", path.display(), err))?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let _ = fs::set_permissions(path, fs::Permissions::from_mode(0o755));
    }

    Ok(())
}

fn export_default_script_hook_templates_internal(
    app: &AppHandle,
    overwrite: bool,
) -> Result<ScriptHookTemplatePaths, String> {
    let dir = ensure_script_hooks_dir(app)?;
    let asr_path = dir.join(DEFAULT_ASR_SCRIPT_FILE);
    let llm_path = dir.join(DEFAULT_LLM_SCRIPT_FILE);

    write_template_file(&asr_path, DEFAULT_ASR_SCRIPT_TEMPLATE, overwrite)?;
    write_template_file(&llm_path, DEFAULT_LLM_SCRIPT_TEMPLATE, overwrite)?;

    Ok(ScriptHookTemplatePaths {
        directory: dir.to_string_lossy().to_string(),
        post_asr_script_path: asr_path.to_string_lossy().to_string(),
        post_llm_script_path: llm_path.to_string_lossy().to_string(),
    })
}

fn ensure_script_source_for_stage(
    app: &AppHandle,
    settings: &settings::AppSettings,
    stage: ScriptHookStage,
) -> Result<std::path::PathBuf, String> {
    let configured_path = match stage {
        ScriptHookStage::AsrPost => settings.post_asr_script_path.as_deref(),
        ScriptHookStage::LlmPost => settings.post_llm_script_path.as_deref(),
    }
    .map(str::trim)
    .filter(|value| !value.is_empty())
    .map(std::path::PathBuf::from);

    if let Some(path) = configured_path {
        if path.exists() {
            return Ok(path);
        }
    }

    let template_paths = export_default_script_hook_templates_internal(app, false)?;
    let path = match stage {
        ScriptHookStage::AsrPost => template_paths.post_asr_script_path,
        ScriptHookStage::LlmPost => template_paths.post_llm_script_path,
    };
    Ok(std::path::PathBuf::from(path))
}

#[tauri::command]
#[specta::specta]
pub fn export_default_script_hook_templates(
    app: AppHandle,
) -> Result<ScriptHookTemplatePaths, String> {
    export_default_script_hook_templates_internal(&app, false)
}

#[tauri::command]
#[specta::specta]
pub fn restore_default_script_hook_templates(
    app: AppHandle,
) -> Result<ScriptHookTemplatePaths, String> {
    export_default_script_hook_templates_internal(&app, true)
}

#[tauri::command]
#[specta::specta]
pub fn export_script_stage_to_desktop(app: AppHandle, stage: String) -> Result<String, String> {
    let parsed_stage = parse_script_hook_stage(&stage)?;
    let runtime_settings = get_settings(&app);
    let source_path = ensure_script_source_for_stage(&app, &runtime_settings, parsed_stage)?;

    let desktop_dir = app
        .path()
        .desktop_dir()
        .map_err(|err| format!("Failed to resolve desktop directory: {}", err))?;
    fs::create_dir_all(&desktop_dir).map_err(|err| {
        format!(
            "Failed to prepare desktop directory {}: {}",
            desktop_dir.display(),
            err
        )
    })?;

    let target_name = match parsed_stage {
        ScriptHookStage::AsrPost => DESKTOP_EXPORT_ASR_SCRIPT_FILE,
        ScriptHookStage::LlmPost => DESKTOP_EXPORT_LLM_SCRIPT_FILE,
    };
    let target_path = desktop_dir.join(target_name);

    fs::copy(&source_path, &target_path).map_err(|err| {
        format!(
            "Failed to export script from {} to {}: {}",
            source_path.display(),
            target_path.display(),
            err
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let _ = fs::set_permissions(&target_path, fs::Permissions::from_mode(0o755));
    }

    Ok(target_path.to_string_lossy().to_string())
}

#[tauri::command]
#[specta::specta]
pub fn import_script_stage_from_path(
    app: AppHandle,
    stage: String,
    source_path: String,
) -> Result<String, String> {
    let parsed_stage = parse_script_hook_stage(&stage)?;
    let source = std::path::PathBuf::from(source_path.trim());

    if source_path.trim().is_empty() {
        return Err("Source script path is empty.".to_string());
    }
    if !source.exists() {
        return Err(format!(
            "Source script does not exist: {}",
            source.display()
        ));
    }
    if !source.is_file() {
        return Err(format!("Source path is not a file: {}", source.display()));
    }

    let scripts_dir = ensure_script_hooks_dir(&app)?;
    let file_name = match parsed_stage {
        ScriptHookStage::AsrPost => DEFAULT_ASR_SCRIPT_FILE,
        ScriptHookStage::LlmPost => DEFAULT_LLM_SCRIPT_FILE,
    };
    let target_path = scripts_dir.join(file_name);

    fs::copy(&source, &target_path).map_err(|err| {
        format!(
            "Failed to import script from {} to {}: {}",
            source.display(),
            target_path.display(),
            err
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        let _ = fs::set_permissions(&target_path, fs::Permissions::from_mode(0o755));
    }

    let mut runtime_settings = get_settings(&app);
    let target_str = target_path.to_string_lossy().to_string();
    match parsed_stage {
        ScriptHookStage::AsrPost => {
            runtime_settings.post_asr_script_path = Some(target_str.clone());
        }
        ScriptHookStage::LlmPost => {
            runtime_settings.post_llm_script_path = Some(target_str.clone());
        }
    }
    settings::write_settings(&app, runtime_settings);

    Ok(target_str)
}

#[tauri::command]
#[specta::specta]
pub fn test_script_hook(
    app: AppHandle,
    stage: String,
    text: String,
    script_path: Option<String>,
) -> Result<ScriptHookTestResult, String> {
    let parsed_stage = parse_script_hook_stage(&stage)?;
    let settings = get_settings(&app);

    let selected_prompt_template = settings
        .post_process_selected_prompt_id
        .as_ref()
        .and_then(|prompt_id| {
            settings
                .post_process_prompts
                .iter()
                .find(|prompt| &prompt.id == prompt_id)
        })
        .map(|prompt| prompt.prompt.as_str());

    let fallback_path = match parsed_stage {
        ScriptHookStage::AsrPost => settings.post_asr_script_path.as_deref(),
        ScriptHookStage::LlmPost => settings.post_llm_script_path.as_deref(),
    };

    let effective_path = script_path
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .or(fallback_path)
        .ok_or_else(|| "Script path is empty for selected stage.".to_string())?
        .to_string();

    let stage_name = match parsed_stage {
        ScriptHookStage::AsrPost => "asr_post",
        ScriptHookStage::LlmPost => "llm_post",
    };

    let timer = std::time::Instant::now();
    let output_text = run_script_hook(
        parsed_stage,
        &effective_path,
        &text,
        settings.script_hook_timeout_ms,
        ScriptHookContext {
            lang: Some(settings.selected_language.as_str()),
            model_id: Some(settings.selected_model.as_str()),
            provider_id: Some(settings.post_process_provider_id.as_str()),
            prompt_id: settings.post_process_selected_prompt_id.as_deref(),
            system_prompt: Some(settings.post_process_system_prompt.as_str()),
            user_prompt_template: selected_prompt_template,
            metadata: Some(serde_json::json!({
                "phase": "manual_test",
            })),
        },
    )?;

    Ok(ScriptHookTestResult {
        stage: stage_name.to_string(),
        used_script_path: effective_path,
        input_text: text,
        output_text,
        duration_ms: timer.elapsed().as_millis() as u64,
    })
}

#[tauri::command]
#[specta::specta]
pub fn change_clipboard_handling_setting(app: AppHandle, handling: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match handling.as_str() {
        "dont_modify" => ClipboardHandling::DontModify,
        "copy_to_clipboard" => ClipboardHandling::CopyToClipboard,
        other => {
            warn!(
                "Invalid clipboard handling '{}', defaulting to dont_modify",
                other
            );
            ClipboardHandling::DontModify
        }
    };
    settings.clipboard_handling = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_auto_submit_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.auto_submit = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_auto_submit_key_setting(app: AppHandle, key: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let parsed = match key.as_str() {
        "enter" => AutoSubmitKey::Enter,
        "ctrl_enter" => AutoSubmitKey::CtrlEnter,
        "cmd_enter" => AutoSubmitKey::CmdEnter,
        other => {
            warn!("Invalid auto submit key '{}', defaulting to enter", other);
            AutoSubmitKey::Enter
        }
    };
    settings.auto_submit_key = parsed;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_enabled = enabled;
    settings::write_settings(&app, settings.clone());

    // Register or unregister the post-processing shortcut
    if let Some(binding) = settings
        .bindings
        .get("transcribe_with_post_process")
        .cloned()
    {
        if enabled {
            let _ = register_shortcut(&app, binding);
        } else {
            let _ = unregister_shortcut(&app, binding);
        }
    }

    if enabled {
        schedule_active_local_post_process_preload(&app, "post_process_enabled");
    }

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_system_prompt_setting(
    app: AppHandle,
    system_prompt: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let trimmed = system_prompt.trim();
    if trimmed.is_empty() {
        return Err("System prompt cannot be empty".to_string());
    }
    settings.post_process_system_prompt = trimmed.to_string();
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_quality_setting(app: AppHandle, quality: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let normalized = quality.trim().to_ascii_lowercase();
    settings.post_process_quality = match normalized.as_str() {
        "fast" => "fast".to_string(),
        "quality" => "quality".to_string(),
        "custom" => "custom".to_string(),
        "" | "balanced" => "balanced".to_string(),
        _ => {
            return Err("Quality must be one of: fast, balanced, quality, custom".to_string());
        }
    };
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_local_max_tokens_setting(
    app: AppHandle,
    value: usize,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_local_max_tokens = value.clamp(64, 2048);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_local_temperature_setting(
    app: AppHandle,
    value: f64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_local_temperature = value.clamp(0.0, 1.0);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_local_top_p_setting(app: AppHandle, value: f64) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_local_top_p = value.clamp(0.1, 1.0);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_local_repetition_penalty_setting(
    app: AppHandle,
    value: f64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_local_repetition_penalty = value.clamp(1.0, 1.5);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_local_repetition_context_size_setting(
    app: AppHandle,
    value: usize,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.post_process_local_repetition_context_size = value.clamp(32, 256);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen_startup_preload_strategy_setting(
    app: AppHandle,
    strategy: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen_startup_preload_strategy =
        settings::normalize_qwen_startup_preload_strategy(strategy.trim());
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen3_startup_preload_enabled_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen3_startup_preload_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen3_startup_preload_delay_ms_setting(
    app: AppHandle,
    value: u64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen3_startup_preload_delay_ms =
        settings::normalize_qwen_startup_preload_delay_ms(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen3_max_threads_setting(app: AppHandle, value: usize) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen3_max_threads = settings::normalize_qwen_max_threads(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen3_server_ready_timeout_sec_setting(
    app: AppHandle,
    value: u64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen3_server_ready_timeout_sec =
        settings::normalize_qwen3_server_ready_timeout_sec(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen3_warmup_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen3_warmup_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen35_startup_preload_enabled_setting(
    app: AppHandle,
    enabled: bool,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen35_startup_preload_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen35_startup_preload_delay_ms_setting(
    app: AppHandle,
    value: u64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen35_startup_preload_delay_ms =
        settings::normalize_qwen_startup_preload_delay_ms(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen35_warmup_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen35_warmup_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen35_max_threads_setting(app: AppHandle, value: usize) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen35_max_threads = settings::normalize_qwen_max_threads(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen35_server_ready_timeout_sec_setting(
    app: AppHandle,
    value: u64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen35_server_ready_timeout_sec =
        settings::normalize_qwen35_server_ready_timeout_sec(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_qwen35_inference_timeout_sec_setting(
    app: AppHandle,
    value: u64,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.qwen35_inference_timeout_sec = settings::normalize_qwen35_inference_timeout_sec(value);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_experimental_enabled_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.experimental_enabled = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_base_url_setting(
    app: AppHandle,
    provider_id: String,
    base_url: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    let label = settings
        .post_process_provider(&provider_id)
        .map(|provider| provider.label.clone())
        .ok_or_else(|| format!("Provider '{}' not found", provider_id))?;

    let provider = settings
        .post_process_provider_mut(&provider_id)
        .expect("Provider looked up above must exist");

    if provider.id != "custom" {
        return Err(format!(
            "Provider '{}' does not allow editing the base URL",
            label
        ));
    }

    provider.base_url = base_url;
    settings::write_settings(&app, settings);
    Ok(())
}

/// Generic helper to validate provider exists
fn validate_provider_exists(
    settings: &settings::AppSettings,
    provider_id: &str,
) -> Result<(), String> {
    if !settings
        .post_process_providers
        .iter()
        .any(|provider| provider.id == provider_id)
    {
        return Err(format!("Provider '{}' not found", provider_id));
    }
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_api_key_setting(
    app: AppHandle,
    provider_id: String,
    api_key: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    validate_provider_exists(&settings, &provider_id)?;
    settings.post_process_api_keys.insert(provider_id, api_key);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_post_process_model_setting(
    app: AppHandle,
    provider_id: String,
    model: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    validate_provider_exists(&settings, &provider_id)?;
    settings.post_process_models.insert(provider_id, model);
    settings::write_settings(&app, settings);
    schedule_active_local_post_process_preload(&app, "model_changed");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn set_post_process_provider(app: AppHandle, provider_id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    validate_provider_exists(&settings, &provider_id)?;
    settings.post_process_provider_id = provider_id;
    settings::write_settings(&app, settings);
    schedule_active_local_post_process_preload(&app, "provider_changed");
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn add_post_process_prompt(
    app: AppHandle,
    name: String,
    prompt: String,
) -> Result<LLMPrompt, String> {
    let mut settings = settings::get_settings(&app);

    // Generate unique ID using timestamp and random component
    let id = format!("prompt_{}", chrono::Utc::now().timestamp_millis());

    let new_prompt = LLMPrompt {
        id: id.clone(),
        name,
        prompt,
    };

    settings.post_process_prompts.push(new_prompt.clone());
    settings::write_settings(&app, settings);

    Ok(new_prompt)
}

#[tauri::command]
#[specta::specta]
pub fn update_post_process_prompt(
    app: AppHandle,
    id: String,
    name: String,
    prompt: String,
) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);

    if let Some(existing_prompt) = settings
        .post_process_prompts
        .iter_mut()
        .find(|p| p.id == id)
    {
        existing_prompt.name = name;
        existing_prompt.prompt = prompt;
        settings::write_settings(&app, settings);
        Ok(())
    } else {
        Err(format!("Prompt with id '{}' not found", id))
    }
}

#[tauri::command]
#[specta::specta]
pub fn delete_post_process_prompt(app: AppHandle, id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);

    // Don't allow deleting the last prompt
    if settings.post_process_prompts.len() <= 1 {
        return Err("Cannot delete the last prompt".to_string());
    }

    // Find and remove the prompt
    let original_len = settings.post_process_prompts.len();
    settings.post_process_prompts.retain(|p| p.id != id);

    if settings.post_process_prompts.len() == original_len {
        return Err(format!("Prompt with id '{}' not found", id));
    }

    // If the deleted prompt was selected, select the first one or None
    if settings.post_process_selected_prompt_id.as_ref() == Some(&id) {
        settings.post_process_selected_prompt_id =
            settings.post_process_prompts.first().map(|p| p.id.clone());
    }

    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub async fn fetch_post_process_models(
    app: AppHandle,
    provider_id: String,
) -> Result<Vec<String>, String> {
    let settings = settings::get_settings(&app);

    // Find the provider
    let provider = settings
        .post_process_providers
        .iter()
        .find(|p| p.id == provider_id)
        .ok_or_else(|| format!("Provider '{}' not found", provider_id))?;

    if provider.id == LOCAL_QWEN35_PROVIDER_ID {
        let manager = app
            .state::<std::sync::Arc<crate::managers::post_process_model::PostProcessModelManager>>(
            );
        let mut ids = manager.get_model_ids();
        // Keep a stable UX order by moving default local model to front.
        if let Some(pos) = ids.iter().position(|id| id == "qwen35-optiq-0.8b") {
            let default_id = ids.remove(pos);
            ids.insert(0, default_id);
        }
        return Ok(ids);
    }

    if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            return Ok(vec![APPLE_INTELLIGENCE_DEFAULT_MODEL_ID.to_string()]);
        }

        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            return Err("Apple Intelligence is only available on Apple silicon Macs running macOS 15 or later.".to_string());
        }
    }

    // Get API key
    let api_key = settings
        .post_process_api_keys
        .get(&provider_id)
        .cloned()
        .unwrap_or_default();

    // Skip fetching if no API key for providers that typically need one
    if api_key.trim().is_empty() && provider.id != "custom" {
        return Err(format!(
            "API key is required for {}. Please add an API key to list available models.",
            provider.label
        ));
    }

    crate::llm_client::fetch_models(provider, api_key).await
}

#[tauri::command]
#[specta::specta]
pub fn set_post_process_selected_prompt(app: AppHandle, id: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);

    // Verify the prompt exists
    if !settings.post_process_prompts.iter().any(|p| p.id == id) {
        return Err(format!("Prompt with id '{}' not found", id));
    }

    settings.post_process_selected_prompt_id = Some(id);
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_mute_while_recording_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.mute_while_recording = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_append_trailing_space_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.append_trailing_space = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_lazy_stream_close_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.lazy_stream_close = enabled;
    settings::write_settings(&app, settings);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_app_language_setting(app: AppHandle, language: String) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.app_language = language.clone();
    settings::write_settings(&app, settings);

    // Refresh the tray menu with the new language
    tray::update_tray_menu(&app, &tray::TrayIconState::Idle, Some(&language));

    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_show_tray_icon_setting(app: AppHandle, enabled: bool) -> Result<(), String> {
    let mut settings = settings::get_settings(&app);
    settings.show_tray_icon = enabled;
    settings::write_settings(&app, settings);

    // Apply change immediately
    tray::set_tray_visibility(&app, enabled);

    Ok(())
}

/// Save accelerator settings, re-apply globals, and unload the model so it
/// reloads with the new backend on next transcription.
fn apply_and_reload_accelerator(app: &AppHandle, s: settings::AppSettings) {
    settings::write_settings(app, s);
    crate::managers::transcription::apply_accelerator_settings(app);

    let tm = app.state::<std::sync::Arc<crate::managers::transcription::TranscriptionManager>>();
    if tm.is_model_loaded() {
        if let Err(e) = tm.unload_model() {
            log::warn!("Failed to unload model after accelerator change: {e}");
        }
    }
}

#[tauri::command]
#[specta::specta]
pub fn change_whisper_accelerator_setting(
    app: AppHandle,
    accelerator: settings::WhisperAcceleratorSetting,
) -> Result<(), String> {
    let mut s = settings::get_settings(&app);
    s.whisper_accelerator = accelerator;
    apply_and_reload_accelerator(&app, s);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_ort_accelerator_setting(
    app: AppHandle,
    accelerator: settings::OrtAcceleratorSetting,
) -> Result<(), String> {
    let mut s = settings::get_settings(&app);
    s.ort_accelerator = accelerator;
    apply_and_reload_accelerator(&app, s);
    Ok(())
}

#[tauri::command]
#[specta::specta]
pub fn change_whisper_gpu_device(app: AppHandle, device: i32) -> Result<(), String> {
    let mut s = settings::get_settings(&app);
    s.whisper_gpu_device = device;
    apply_and_reload_accelerator(&app, s);
    Ok(())
}

/// Return which accelerators and GPU devices are available for this build.
#[tauri::command]
#[specta::specta]
pub fn get_available_accelerators() -> crate::managers::transcription::AvailableAccelerators {
    crate::managers::transcription::get_available_accelerators()
}
