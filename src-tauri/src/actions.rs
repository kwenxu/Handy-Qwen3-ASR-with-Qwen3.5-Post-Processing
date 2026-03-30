#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
use crate::apple_intelligence;
use crate::audio_feedback::{play_feedback_sound, play_feedback_sound_blocking, SoundType};
use crate::audio_toolkit::{is_microphone_access_denied, is_no_input_device_error};
use crate::managers::audio::AudioRecordingManager;
use crate::managers::history::HistoryManager;
use crate::managers::qwen35_post_manager::Qwen35PostManager;
use crate::managers::transcription::TranscriptionManager;
use crate::settings::{
    get_settings, AppSettings, APPLE_INTELLIGENCE_PROVIDER_ID, LOCAL_QWEN35_PROVIDER_ID,
};
use crate::shortcut;
use crate::tray::{change_tray_icon, TrayIconState};
use crate::utils::{
    self, show_processing_overlay, show_recording_overlay, show_transcribing_overlay,
};
use crate::TranscriptionCoordinator;
use ferrous_opencc::{config::BuiltinConfig, OpenCC};
use log::{debug, error, warn};
use once_cell::sync::Lazy;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use tauri::Manager;
use tauri::{AppHandle, Emitter};

#[derive(Clone, serde::Serialize)]
struct RecordingErrorEvent {
    error_type: String,
    detail: Option<String>,
}

/// Drop guard that notifies the [`TranscriptionCoordinator`] when the
/// transcription pipeline finishes — whether it completes normally or panics.
struct FinishGuard(AppHandle);
impl Drop for FinishGuard {
    fn drop(&mut self) {
        if let Some(c) = self.0.try_state::<TranscriptionCoordinator>() {
            c.notify_processing_finished();
        }
    }
}

// Shortcut Action Trait
pub trait ShortcutAction: Send + Sync {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str);
}

// Transcribe Action
struct TranscribeAction {
    post_process: bool,
}

/// Field name for structured output JSON schema
const TRANSCRIPTION_FIELD: &str = "transcription";

/// Strip invisible Unicode characters that some LLMs may insert
fn strip_invisible_chars(s: &str) -> String {
    s.replace(['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}'], "")
}

fn normalize_for_compare(s: &str) -> String {
    s.chars()
        .filter(|ch| ch.is_alphanumeric())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn collapse_repeated_lines(s: &str) -> String {
    let mut result: Vec<String> = Vec::new();
    let mut previous_key = String::new();

    for line in s.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let key = normalize_for_compare(trimmed);
        if key.is_empty() {
            continue;
        }
        if key == previous_key {
            continue;
        }
        result.push(trimmed.to_string());
        previous_key = key;
    }

    // If the model got stuck and repeated near-identical lines, keep only the
    // first occurrence of each normalized line.
    if result.len() >= 6 {
        let mut seen = HashSet::new();
        let mut deduped = Vec::new();
        for line in &result {
            let key = normalize_for_compare(line);
            if seen.insert(key) {
                deduped.push(line.clone());
            }
        }
        if deduped.len() * 2 <= result.len() {
            return deduped.join("\n");
        }
    }

    result.join("\n")
}

fn clean_post_process_output(s: &str) -> String {
    let mut out = strip_invisible_chars(s);
    out = out.replace("<think>", "").replace("</think>", "");
    out = collapse_repeated_lines(&out);
    out.trim().to_string()
}

fn template_requests_arabic_digits(prompt_template: &str) -> bool {
    let lowered = prompt_template.to_ascii_lowercase();
    lowered.contains("arabic digit")
        || lowered.contains("arabic numeral")
        || lowered.contains("arabic number")
        || lowered.contains("1234567")
        || prompt_template.contains("阿拉伯数字")
}

fn is_chinese_digit_char(ch: char) -> bool {
    matches!(
        ch,
        '零' | '〇'
            | '○'
            | '幺'
            | '壹'
            | '一'
            | '贰'
            | '二'
            | '两'
            | '叁'
            | '三'
            | '肆'
            | '四'
            | '伍'
            | '五'
            | '陆'
            | '六'
            | '柒'
            | '七'
            | '捌'
            | '八'
            | '玖'
            | '九'
    )
}

fn chinese_digit_to_arabic(ch: char) -> Option<char> {
    match ch {
        '零' | '〇' | '○' => Some('0'),
        '幺' | '壹' | '一' => Some('1'),
        '贰' | '二' => Some('2'),
        '两' => Some('2'),
        '叁' | '三' => Some('3'),
        '肆' | '四' => Some('4'),
        '伍' | '五' => Some('5'),
        '陆' | '六' => Some('6'),
        '柒' | '七' => Some('7'),
        '捌' | '八' => Some('8'),
        '玖' | '九' => Some('9'),
        _ => None,
    }
}

fn is_cjk_non_digit_char(ch: char) -> bool {
    ('\u{4E00}'..='\u{9FFF}').contains(&ch) && !is_chinese_digit_char(ch)
}

fn is_list_separator_char(ch: char) -> bool {
    matches!(ch, '、' | ',' | '，' | ';' | '；' | ':' | '：' | '/' | '／')
}

fn normalize_model_size_tokens_with_b_unit(input: &str) -> String {
    fn parse_digit_char(ch: char) -> Option<char> {
        if ch.is_ascii_digit() {
            Some(ch)
        } else {
            chinese_digit_to_arabic(ch)
        }
    }

    fn is_b_unit(ch: char) -> bool {
        matches!(ch, 'B' | 'b')
    }

    fn skip_spaces(chars: &[char], mut idx: usize) -> usize {
        while idx < chars.len() && chars[idx].is_whitespace() {
            idx += 1;
        }
        idx
    }

    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;

    while i < chars.len() {
        if let Some(int_digit) = parse_digit_char(chars[i]) {
            let mut j = skip_spaces(&chars, i + 1);
            let mut handled = false;

            // decimal form, e.g. 零点8B / 零点八 B / 0.8B / 一点七B
            if j < chars.len() && matches!(chars[j], '点' | '.' | '．') {
                j = skip_spaces(&chars, j + 1);
                if j < chars.len() {
                    if let Some(frac_digit) = parse_digit_char(chars[j]) {
                        let k = skip_spaces(&chars, j + 1);
                        if k < chars.len() && is_b_unit(chars[k]) {
                            out.push(int_digit);
                            out.push('.');
                            out.push(frac_digit);
                            out.push('B');
                            i = k + 1;
                            handled = true;
                        }
                    }
                }
            }

            if handled {
                continue;
            }

            // integer form, e.g. 两B / 2 B / 二 B / 9b
            let k = skip_spaces(&chars, i + 1);
            if k < chars.len() && is_b_unit(chars[k]) {
                out.push(int_digit);
                out.push('B');
                i = k + 1;
                continue;
            }
        }

        out.push(chars[i]);
        i += 1;
    }

    out
}

fn normalize_standalone_chinese_digits_to_arabic(input: &str) -> String {
    let chars: Vec<char> = input.chars().collect();
    let mut out = String::with_capacity(input.len());
    let mut i = 0usize;

    while i < chars.len() {
        if !is_chinese_digit_char(chars[i]) {
            out.push(chars[i]);
            i += 1;
            continue;
        }

        let start = i;
        while i < chars.len() && is_chinese_digit_char(chars[i]) {
            i += 1;
        }
        let end = i;
        let seq_len = end - start;

        let prev = if start > 0 {
            chars.get(start - 1).copied()
        } else {
            None
        };
        let mut prev_non_ws = None;
        let mut k = start;
        while k > 0 {
            let ch = chars[k - 1];
            if !ch.is_whitespace() {
                prev_non_ws = Some(ch);
                break;
            }
            k -= 1;
        }

        let next = chars.get(end).copied();
        let mut next_non_ws = None;
        let mut j = end;
        while j < chars.len() {
            let ch = chars[j];
            if !ch.is_whitespace() {
                next_non_ws = Some(ch);
                break;
            }
            j += 1;
        }

        // Boundary guard:
        // - Convert multi-digit sequences eagerly (e.g. 一二三 -> 123).
        // - For single digits, avoid in-word conversion (e.g. 一些 must stay semantic).
        // - Convert single-digit list items (e.g. 一、二、三、四).
        let prev_is_cjk = prev.is_some_and(is_cjk_non_digit_char);
        let next_is_cjk = next.is_some_and(is_cjk_non_digit_char);
        let should_convert = if seq_len >= 2 {
            true
        } else if prev_is_cjk && next_is_cjk {
            false
        } else {
            prev.is_some_and(is_list_separator_char)
                || next.is_some_and(is_list_separator_char)
                || prev_non_ws.is_some_and(is_list_separator_char)
                || next_non_ws.is_some_and(|ch| ch.is_ascii_alphanumeric())
        };

        if should_convert {
            for ch in &chars[start..end] {
                out.push(chinese_digit_to_arabic(*ch).unwrap_or(*ch));
            }
        } else {
            for ch in &chars[start..end] {
                out.push(*ch);
            }
        }
    }

    normalize_model_size_tokens_with_b_unit(&out)
}

fn build_user_prompt_content(prompt_template: &str, transcription: &str) -> String {
    if prompt_template.contains("${output}") {
        prompt_template.replace("${output}", transcription)
    } else {
        format!(
            "{}\n\nTranscript:\n{}",
            prompt_template.trim(),
            transcription
        )
    }
}

fn has_meaningful_text(input: &str) -> bool {
    let stripped = strip_invisible_chars(input);
    let compact = stripped.trim();
    if compact.is_empty() {
        return false;
    }
    let informative_chars = compact
        .chars()
        .filter(|ch| ch.is_alphanumeric() || ('\u{4E00}'..='\u{9FFF}').contains(ch))
        .count();
    informative_chars >= 1
}

fn trim_rule_prefix(line: &str) -> &str {
    let trimmed = line.trim_start();
    let mut consumed = 0usize;
    let mut saw_digit = false;
    for (idx, ch) in trimmed.char_indices() {
        if ch.is_ascii_digit() {
            saw_digit = true;
            consumed = idx + ch.len_utf8();
            continue;
        }
        if saw_digit && matches!(ch, '.' | '、' | ')' | '）' | ':') {
            consumed = idx + ch.len_utf8();
            continue;
        }
        if consumed > 0 && ch.is_whitespace() {
            consumed = idx + ch.len_utf8();
            continue;
        }
        break;
    }
    if consumed > 0 {
        trimmed[consumed..].trim_start()
    } else {
        trimmed
    }
}

fn extract_template_rule_keys(prompt_template: &str) -> Vec<String> {
    prompt_template
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .filter(|line| !line.contains("${output}"))
        .filter(|line| {
            !matches!(
                *line,
                "Input:"
                    | "输入："
                    | "输入:"
                    | "Transcript:"
                    | "Rules:"
                    | "规则："
                    | "规则:"
                    | "要求："
                    | "要求:"
                    | "Task:"
                    | "任务："
            )
        })
        .map(trim_rule_prefix)
        .map(normalize_for_compare)
        .filter(|key| key.len() >= 6)
        .collect()
}

fn looks_like_instruction_template_line(line: &str, template_keys: &[String]) -> bool {
    let normalized = normalize_for_compare(trim_rule_prefix(line));
    if normalized.is_empty() {
        return false;
    }

    let line_lower = line.trim().to_ascii_lowercase();
    let hard_markers = [
        "output contract",
        "rules:",
        "task:",
        "do not include reasoning",
        "return only the final",
    ];
    if hard_markers
        .iter()
        .any(|marker| line_lower.contains(marker))
    {
        return true;
    }

    let hard_markers_zh = [
        "保持原意与事实",
        "去除口头重复",
        "专有名词",
        "仅输出最终结果",
        "不要解释",
        "中文输出阿拉伯数字",
    ];
    if hard_markers_zh.iter().any(|marker| line.contains(marker)) {
        return true;
    }

    template_keys.iter().any(|key| {
        normalized.contains(key)
            || key.contains(&normalized)
            || prefix_chars_equal(&normalized, key, 8)
    })
}

fn looks_like_rule_enumeration_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    let mut chars = trimmed.chars().peekable();
    let mut saw_digit = false;
    while let Some(ch) = chars.peek().copied() {
        if ch.is_ascii_digit() {
            saw_digit = true;
            chars.next();
            continue;
        }
        break;
    }
    if !saw_digit {
        return false;
    }
    matches!(chars.peek().copied(), Some('.' | '、' | ')' | '）' | ':'))
}

fn prefix_chars_equal(a: &str, b: &str, n: usize) -> bool {
    a.chars().take(n).eq(b.chars().take(n))
}

fn strip_prompt_template_leakage(output: &str, prompt_template: &str) -> String {
    let template_keys = extract_template_rule_keys(prompt_template);
    let mut kept_lines = Vec::new();

    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if looks_like_instruction_template_line(trimmed, &template_keys) {
            continue;
        }
        kept_lines.push(trimmed.to_string());
    }

    if kept_lines.is_empty() {
        String::new()
    } else {
        kept_lines.join("\n")
    }
}

fn normalize_post_process_candidate(
    raw_output: &str,
    prompt_template: &str,
    force_arabic_digits: bool,
) -> String {
    let mut cleaned = clean_post_process_output(raw_output);
    cleaned = strip_prompt_template_leakage(&cleaned, prompt_template);
    cleaned = collapse_repeated_lines(&cleaned);
    if force_arabic_digits {
        cleaned = normalize_standalone_chinese_digits_to_arabic(&cleaned);
    }
    cleaned.trim().to_string()
}

fn reject_post_output_reason(output: &str, prompt_template: &str) -> Option<&'static str> {
    if output.trim().is_empty() {
        return Some("empty");
    }
    if output.contains("${output}") {
        return Some("prompt_placeholder_leakage");
    }
    let output_lower = output.to_ascii_lowercase();
    if output.contains("要求：")
        || output.contains("规则：")
        || output.contains("Output contract:")
        || output_lower.contains("rules:")
        || output_lower.contains("input:")
    {
        return Some("template_header_leakage");
    }
    let template_keys = extract_template_rule_keys(prompt_template);
    let mut suspicious_lines = 0usize;
    let mut total_non_empty = 0usize;
    let mut enum_rule_lines = 0usize;
    for line in output.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        total_non_empty += 1;
        if looks_like_instruction_template_line(trimmed, &template_keys) {
            suspicious_lines += 1;
        }
        if looks_like_rule_enumeration_line(trimmed) {
            enum_rule_lines += 1;
        }
    }
    if enum_rule_lines >= 2 && suspicious_lines >= 1 {
        return Some("enumerated_rule_leakage");
    }
    if suspicious_lines >= 2 && suspicious_lines * 2 >= total_non_empty {
        return Some("prompt_template_leakage");
    }
    None
}

fn should_reject_post_output(output: &str, prompt_template: &str) -> bool {
    reject_post_output_reason(output, prompt_template).is_some()
}

#[derive(Clone, Copy)]
struct LocalGenerationParams {
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    repetition_penalty: f32,
    repetition_context_size: usize,
}

fn local_generation_params_from_settings(settings: &AppSettings) -> LocalGenerationParams {
    LocalGenerationParams {
        max_tokens: settings.post_process_local_max_tokens.clamp(64, 512),
        temperature: settings.post_process_local_temperature.clamp(0.0, 1.0) as f32,
        top_p: settings.post_process_local_top_p.clamp(0.1, 1.0) as f32,
        repetition_penalty: settings
            .post_process_local_repetition_penalty
            .clamp(1.0, 1.5) as f32,
        repetition_context_size: settings
            .post_process_local_repetition_context_size
            .clamp(32, 256),
    }
}

async fn post_process_transcription(
    app: &AppHandle,
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    if !has_meaningful_text(transcription) {
        debug!("Post-processing skipped because transcription appears empty or non-informative");
        return None;
    }

    let transcription_text = strip_invisible_chars(transcription).trim().to_string();

    let provider = match settings.active_post_process_provider().cloned() {
        Some(provider) => provider,
        None => {
            debug!("Post-processing enabled but no provider is selected");
            return None;
        }
    };

    let model = settings
        .post_process_models
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if model.trim().is_empty() {
        debug!(
            "Post-processing skipped because provider '{}' has no model configured",
            provider.id
        );
        return None;
    }

    let selected_prompt_id = match &settings.post_process_selected_prompt_id {
        Some(id) => id.clone(),
        None => {
            debug!("Post-processing skipped because no prompt is selected");
            return None;
        }
    };

    let prompt = match settings
        .post_process_prompts
        .iter()
        .find(|prompt| prompt.id == selected_prompt_id)
    {
        Some(prompt) => prompt.prompt.clone(),
        None => {
            debug!(
                "Post-processing skipped because prompt '{}' was not found",
                selected_prompt_id
            );
            return None;
        }
    };

    if prompt.trim().is_empty() {
        debug!("Post-processing skipped because the selected prompt is empty");
        return None;
    }

    let system_prompt = settings.post_process_system_prompt.trim().to_string();
    let quality_params = local_generation_params_from_settings(settings);
    let force_arabic_digits = template_requests_arabic_digits(&prompt);
    let prompt_template_for_post = prompt.clone();

    if provider.id == LOCAL_QWEN35_PROVIDER_ID {
        let manager = app.state::<Arc<Qwen35PostManager>>().inner().clone();
        let local_model = model.clone();
        let local_text = transcription_text.clone();
        let local_user_content = build_user_prompt_content(&prompt, &local_text);
        let local_user_content_for_infer = local_user_content.clone();
        let local_template_id = selected_prompt_id.clone();
        let local_quality = quality_params;
        let local_system_prompt = system_prompt.clone();
        let local_force_arabic_digits = force_arabic_digits;
        let local_prompt_template = prompt_template_for_post.clone();

        return match tauri::async_runtime::spawn_blocking(move || {
            debug!(
                "Local Qwen3.5 params => max_tokens={}, temperature={}, top_p={}, repetition_penalty={}, repetition_context_size={}",
                local_quality.max_tokens,
                local_quality.temperature,
                local_quality.top_p,
                local_quality.repetition_penalty,
                local_quality.repetition_context_size
            );

            let first = manager.process_text(
                &local_model,
                &local_user_content_for_infer,
                &local_system_prompt,
                Some(local_template_id.as_str()),
                local_quality.max_tokens,
                local_quality.temperature,
                local_quality.top_p,
                local_quality.repetition_penalty,
                local_quality.repetition_context_size,
            )?;

            let first_clean = normalize_post_process_candidate(
                &first,
                &local_prompt_template,
                local_force_arabic_digits,
            );

            if !should_reject_post_output(&first_clean, &local_prompt_template) {
                return Ok(first_clean);
            }
            if let Some(reason) = reject_post_output_reason(&first_clean, &local_prompt_template) {
                warn!(
                    "Local Qwen3.5 first pass rejected (reason={}, template_id={})",
                    reason, local_template_id
                );
            }

            // Retry once when first pass leaks template/instructions.
            // This especially helps immediately after model switch/cold load.
            let second = manager.process_text(
                &local_model,
                &local_user_content_for_infer,
                &local_system_prompt,
                Some(local_template_id.as_str()),
                local_quality.max_tokens,
                local_quality.temperature,
                local_quality.top_p,
                local_quality.repetition_penalty,
                local_quality.repetition_context_size,
            )?;

            let second_clean = normalize_post_process_candidate(
                &second,
                &local_prompt_template,
                local_force_arabic_digits,
            );
            if !should_reject_post_output(&second_clean, &local_prompt_template) {
                return Ok(second_clean);
            }
            if let Some(reason) = reject_post_output_reason(&second_clean, &local_prompt_template) {
                warn!(
                    "Local Qwen3.5 second pass rejected (reason={}, template_id={})",
                    reason, local_template_id
                );
            }

            Err(anyhow::anyhow!(
                "Local post-processing output rejected by validators"
            ))
        })
        .await
        {
            Ok(Ok(result)) => {
                if result.trim().is_empty() {
                    debug!("Local Qwen3.5 post-processing returned empty output");
                    None
                } else {
                    debug!(
                        "Local Qwen3.5 post-processing succeeded. Output length: {} chars",
                        result.len()
                    );
                    Some(result)
                }
            }
            Ok(Err(err)) => {
                error!(
                    "Local Qwen3.5 post-processing failed: {}. Falling back to original transcription.",
                    err
                );
                None
            }
            Err(err) => {
                error!(
                    "Local Qwen3.5 post-processing task panicked: {}. Falling back to original transcription.",
                    err
                );
                None
            }
        };
    }

    debug!(
        "Starting LLM post-processing with provider '{}' (model: {})",
        provider.id, model
    );

    let api_key = settings
        .post_process_api_keys
        .get(&provider.id)
        .cloned()
        .unwrap_or_default();

    if provider.supports_structured_output {
        debug!("Using structured outputs for provider '{}'", provider.id);

        let user_content = build_user_prompt_content(&prompt, &transcription_text);

        // Handle Apple Intelligence separately since it uses native Swift APIs
        if provider.id == APPLE_INTELLIGENCE_PROVIDER_ID {
            #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
            {
                if !apple_intelligence::check_apple_intelligence_availability() {
                    debug!(
                        "Apple Intelligence selected but not currently available on this device"
                    );
                    return None;
                }

                let token_limit = model.trim().parse::<i32>().unwrap_or(0);
                return match apple_intelligence::process_text_with_system_prompt(
                    &system_prompt,
                    &user_content,
                    token_limit,
                ) {
                    Ok(result) => {
                        if result.trim().is_empty() {
                            debug!("Apple Intelligence returned an empty response");
                            None
                        } else {
                            let result = normalize_post_process_candidate(
                                &result,
                                &prompt_template_for_post,
                                force_arabic_digits,
                            );
                            debug!(
                                "Apple Intelligence post-processing succeeded. Output length: {} chars",
                                result.len()
                            );
                            Some(result)
                        }
                    }
                    Err(err) => {
                        error!("Apple Intelligence post-processing failed: {}", err);
                        None
                    }
                };
            }

            #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
            {
                debug!("Apple Intelligence provider selected on unsupported platform");
                return None;
            }
        }

        // Define JSON schema for transcription output
        let json_schema = serde_json::json!({
            "type": "object",
            "properties": {
                (TRANSCRIPTION_FIELD): {
                    "type": "string",
                    "description": "The cleaned and processed transcription text"
                }
            },
            "required": [TRANSCRIPTION_FIELD],
            "additionalProperties": false
        });

        match crate::llm_client::send_chat_completion_with_schema(
            &provider,
            api_key.clone(),
            &model,
            user_content.clone(),
            Some(system_prompt.clone()),
            Some(json_schema),
        )
        .await
        {
            Ok(Some(content)) => {
                // Parse the JSON response to extract the transcription field
                match serde_json::from_str::<serde_json::Value>(&content) {
                    Ok(json) => {
                        if let Some(transcription_value) =
                            json.get(TRANSCRIPTION_FIELD).and_then(|t| t.as_str())
                        {
                            let result = normalize_post_process_candidate(
                                transcription_value,
                                &prompt_template_for_post,
                                force_arabic_digits,
                            );
                            if should_reject_post_output(&result, &prompt_template_for_post) {
                                warn!("Structured post-processing output rejected by validators");
                                return None;
                            }
                            debug!(
                                "Structured output post-processing succeeded for provider '{}'. Output length: {} chars",
                                provider.id,
                                result.len()
                            );
                            return Some(result);
                        } else {
                            error!("Structured output response missing 'transcription' field");
                            let cleaned = normalize_post_process_candidate(
                                &content,
                                &prompt_template_for_post,
                                force_arabic_digits,
                            );
                            if should_reject_post_output(&cleaned, &prompt_template_for_post) {
                                warn!("Structured raw content rejected by validators");
                                return None;
                            }
                            return Some(cleaned);
                        }
                    }
                    Err(e) => {
                        error!(
                            "Failed to parse structured output JSON: {}. Returning raw content.",
                            e
                        );
                        let cleaned = normalize_post_process_candidate(
                            &content,
                            &prompt_template_for_post,
                            force_arabic_digits,
                        );
                        if should_reject_post_output(&cleaned, &prompt_template_for_post) {
                            warn!("Structured fallback content rejected by validators");
                            return None;
                        }
                        return Some(cleaned);
                    }
                }
            }
            Ok(None) => {
                error!("LLM API response has no content");
                return None;
            }
            Err(e) => {
                warn!(
                    "Structured output failed for provider '{}': {}. Falling back to legacy mode.",
                    provider.id, e
                );
                // Fall through to legacy mode below
            }
        }
    }

    // Legacy mode: Build user content from prompt template.
    let processed_prompt = build_user_prompt_content(&prompt, &transcription_text);
    debug!("Processed prompt length: {} chars", processed_prompt.len());

    match crate::llm_client::send_chat_completion(
        &provider,
        api_key,
        &model,
        processed_prompt.clone(),
    )
    .await
    {
        Ok(Some(content)) => {
            let content = normalize_post_process_candidate(
                &content,
                &prompt_template_for_post,
                force_arabic_digits,
            );
            if should_reject_post_output(&content, &prompt_template_for_post) {
                warn!("Legacy post-processing output rejected by validators");
                return None;
            }
            debug!(
                "LLM post-processing succeeded for provider '{}'. Output length: {} chars",
                provider.id,
                content.len()
            );
            Some(content)
        }
        Ok(None) => {
            error!("LLM API response has no content");
            None
        }
        Err(e) => {
            error!(
                "LLM post-processing failed for provider '{}': {}. Falling back to original transcription.",
                provider.id,
                e
            );
            None
        }
    }
}

async fn maybe_convert_chinese_variant(
    settings: &AppSettings,
    transcription: &str,
) -> Option<String> {
    // Check if language is set to Simplified or Traditional Chinese
    let is_simplified = settings.selected_language == "zh-Hans";
    let is_traditional = settings.selected_language == "zh-Hant";

    if !is_simplified && !is_traditional {
        debug!("selected_language is not Simplified or Traditional Chinese; skipping translation");
        return None;
    }

    debug!(
        "Starting Chinese translation using OpenCC for language: {}",
        settings.selected_language
    );

    // Use OpenCC to convert based on selected language
    let config = if is_simplified {
        // Convert Traditional Chinese to Simplified Chinese
        BuiltinConfig::Tw2sp
    } else {
        // Convert Simplified Chinese to Traditional Chinese
        BuiltinConfig::S2tw
    };

    match OpenCC::from_config(config) {
        Ok(converter) => {
            let converted = converter.convert(transcription);
            debug!(
                "OpenCC translation completed. Input length: {}, Output length: {}",
                transcription.len(),
                converted.len()
            );
            Some(converted)
        }
        Err(e) => {
            error!("Failed to initialize OpenCC converter: {}. Falling back to original transcription.", e);
            None
        }
    }
}

pub(crate) struct ProcessedTranscription {
    pub final_text: String,
    pub post_processed_text: Option<String>,
    pub post_process_prompt: Option<String>,
}

pub(crate) async fn process_transcription_output(
    app: &AppHandle,
    transcription: &str,
    post_process: bool,
) -> ProcessedTranscription {
    let settings = get_settings(app);
    let mut final_text = transcription.to_string();
    let mut post_processed_text: Option<String> = None;
    let mut post_process_prompt: Option<String> = None;

    if let Some(converted_text) = maybe_convert_chinese_variant(&settings, transcription).await {
        final_text = converted_text;
    }

    if post_process {
        if let Some(processed_text) = post_process_transcription(app, &settings, &final_text).await
        {
            post_processed_text = Some(processed_text.clone());
            final_text = processed_text;

            if let Some(prompt_id) = &settings.post_process_selected_prompt_id {
                if let Some(prompt) = settings
                    .post_process_prompts
                    .iter()
                    .find(|prompt| &prompt.id == prompt_id)
                {
                    post_process_prompt = Some(prompt.prompt.clone());
                }
            }
        }
    } else if final_text != transcription {
        post_processed_text = Some(final_text.clone());
    }

    ProcessedTranscription {
        final_text,
        post_processed_text,
        post_process_prompt,
    }
}

impl ShortcutAction for TranscribeAction {
    fn start(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        let start_time = Instant::now();
        debug!("TranscribeAction::start called for binding: {}", binding_id);

        // Load model in the background
        let tm = app.state::<Arc<TranscriptionManager>>();
        tm.initiate_model_load();
        if self.post_process {
            // Fallback trigger preload: if local post model is not hot, start warming now.
            shortcut::schedule_active_local_post_process_preload(app, "shortcut_start");
        }

        let binding_id = binding_id.to_string();
        change_tray_icon(app, TrayIconState::Recording);
        show_recording_overlay(app);

        let rm = app.state::<Arc<AudioRecordingManager>>();

        // Get the microphone mode to determine audio feedback timing
        let settings = get_settings(app);
        let is_always_on = settings.always_on_microphone;
        debug!("Microphone mode - always_on: {}", is_always_on);

        let mut recording_error: Option<String> = None;
        if is_always_on {
            // Always-on mode: Play audio feedback immediately, then apply mute after sound finishes
            debug!("Always-on mode: Playing audio feedback immediately");
            let rm_clone = Arc::clone(&rm);
            let app_clone = app.clone();
            // The blocking helper exits immediately if audio feedback is disabled,
            // so we can always reuse this thread to ensure mute happens right after playback.
            std::thread::spawn(move || {
                play_feedback_sound_blocking(&app_clone, SoundType::Start);
                rm_clone.apply_mute();
            });

            if let Err(e) = rm.try_start_recording(&binding_id) {
                debug!("Recording failed: {}", e);
                recording_error = Some(e);
            }
        } else {
            // On-demand mode: Start recording first, then play audio feedback, then apply mute
            // This allows the microphone to be activated before playing the sound
            debug!("On-demand mode: Starting recording first, then audio feedback");
            let recording_start_time = Instant::now();
            match rm.try_start_recording(&binding_id) {
                Ok(()) => {
                    debug!("Recording started in {:?}", recording_start_time.elapsed());
                    // Small delay to ensure microphone stream is active
                    let app_clone = app.clone();
                    let rm_clone = Arc::clone(&rm);
                    std::thread::spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(100));
                        debug!("Handling delayed audio feedback/mute sequence");
                        // Helper handles disabled audio feedback by returning early, so we reuse it
                        // to keep mute sequencing consistent in every mode.
                        play_feedback_sound_blocking(&app_clone, SoundType::Start);
                        rm_clone.apply_mute();
                    });
                }
                Err(e) => {
                    debug!("Failed to start recording: {}", e);
                    recording_error = Some(e);
                }
            }
        }

        if recording_error.is_none() {
            // Dynamically register the cancel shortcut in a separate task to avoid deadlock
            shortcut::register_cancel_shortcut(app);
        } else {
            // Starting failed (for example due to blocked microphone permissions).
            // Revert UI state so we don't stay stuck in the recording overlay.
            utils::hide_recording_overlay(app);
            change_tray_icon(app, TrayIconState::Idle);
            if let Some(err) = recording_error {
                let error_type = if is_microphone_access_denied(&err) {
                    "microphone_permission_denied"
                } else if is_no_input_device_error(&err) {
                    "no_input_device"
                } else {
                    "unknown"
                };
                let _ = app.emit(
                    "recording-error",
                    RecordingErrorEvent {
                        error_type: error_type.to_string(),
                        detail: Some(err),
                    },
                );
            }
        }

        debug!(
            "TranscribeAction::start completed in {:?}",
            start_time.elapsed()
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, _shortcut_str: &str) {
        // Unregister the cancel shortcut when transcription stops
        shortcut::unregister_cancel_shortcut(app);

        let stop_time = Instant::now();
        debug!("TranscribeAction::stop called for binding: {}", binding_id);

        let ah = app.clone();
        let rm = Arc::clone(&app.state::<Arc<AudioRecordingManager>>());
        let tm = Arc::clone(&app.state::<Arc<TranscriptionManager>>());
        let hm = Arc::clone(&app.state::<Arc<HistoryManager>>());

        change_tray_icon(app, TrayIconState::Transcribing);
        show_transcribing_overlay(app);

        // Unmute before playing audio feedback so the stop sound is audible
        rm.remove_mute();

        // Play audio feedback for recording stop
        play_feedback_sound(app, SoundType::Stop);

        let binding_id = binding_id.to_string(); // Clone binding_id for the async task
        let post_process = self.post_process;

        tauri::async_runtime::spawn(async move {
            let _guard = FinishGuard(ah.clone());
            debug!(
                "Starting async transcription task for binding: {}",
                binding_id
            );

            let stop_recording_time = Instant::now();
            if let Some(samples) = rm.stop_recording(&binding_id) {
                debug!(
                    "Recording stopped and samples retrieved in {:?}, sample count: {}",
                    stop_recording_time.elapsed(),
                    samples.len()
                );

                if samples.is_empty() {
                    debug!("Recording produced no audio samples; skipping persistence");
                    utils::hide_recording_overlay(&ah);
                    change_tray_icon(&ah, TrayIconState::Idle);
                } else {
                    // Save WAV concurrently with transcription
                    let sample_count = samples.len();
                    let file_name = format!("handy-{}.wav", chrono::Utc::now().timestamp());
                    let wav_path = hm.recordings_dir().join(&file_name);
                    let wav_path_for_verify = wav_path.clone();
                    let samples_for_wav = samples.clone();
                    let wav_handle = tauri::async_runtime::spawn_blocking(move || {
                        crate::audio_toolkit::save_wav_file(&wav_path, &samples_for_wav)
                    });

                    // Transcribe concurrently with WAV save
                    let transcription_time = Instant::now();
                    let transcription_result = tm.transcribe(samples);

                    // Await WAV save and verify
                    let wav_saved = match wav_handle.await {
                        Ok(Ok(())) => {
                            match crate::audio_toolkit::verify_wav_file(
                                &wav_path_for_verify,
                                sample_count,
                            ) {
                                Ok(()) => true,
                                Err(e) => {
                                    error!("WAV verification failed: {}", e);
                                    false
                                }
                            }
                        }
                        Ok(Err(e)) => {
                            error!("Failed to save WAV file: {}", e);
                            false
                        }
                        Err(e) => {
                            error!("WAV save task panicked: {}", e);
                            false
                        }
                    };

                    match transcription_result {
                        Ok(transcription) => {
                            debug!(
                                "Transcription completed in {:?}: '{}'",
                                transcription_time.elapsed(),
                                transcription
                            );

                            if post_process {
                                show_processing_overlay(&ah);
                            }
                            let processed =
                                process_transcription_output(&ah, &transcription, post_process)
                                    .await;

                            // Save to history if WAV was saved
                            if wav_saved {
                                if let Err(err) = hm.save_entry(
                                    file_name,
                                    transcription,
                                    post_process,
                                    processed.post_processed_text.clone(),
                                    processed.post_process_prompt.clone(),
                                ) {
                                    error!("Failed to save history entry: {}", err);
                                }
                            }

                            if processed.final_text.is_empty() {
                                utils::hide_recording_overlay(&ah);
                                change_tray_icon(&ah, TrayIconState::Idle);
                            } else {
                                let ah_clone = ah.clone();
                                let paste_time = Instant::now();
                                let final_text = processed.final_text;
                                ah.run_on_main_thread(move || {
                                    match utils::paste(final_text, ah_clone.clone()) {
                                        Ok(()) => debug!(
                                            "Text pasted successfully in {:?}",
                                            paste_time.elapsed()
                                        ),
                                        Err(e) => error!("Failed to paste transcription: {}", e),
                                    }
                                    utils::hide_recording_overlay(&ah_clone);
                                    change_tray_icon(&ah_clone, TrayIconState::Idle);
                                })
                                .unwrap_or_else(|e| {
                                    error!("Failed to run paste on main thread: {:?}", e);
                                    utils::hide_recording_overlay(&ah);
                                    change_tray_icon(&ah, TrayIconState::Idle);
                                });
                            }
                        }
                        Err(err) => {
                            debug!("Global Shortcut Transcription error: {}", err);
                            // Save entry with empty text so user can retry
                            if wav_saved {
                                if let Err(save_err) = hm.save_entry(
                                    file_name,
                                    String::new(),
                                    post_process,
                                    None,
                                    None,
                                ) {
                                    error!("Failed to save failed history entry: {}", save_err);
                                }
                            }
                            utils::hide_recording_overlay(&ah);
                            change_tray_icon(&ah, TrayIconState::Idle);
                        }
                    }
                }
            } else {
                debug!("No samples retrieved from recording stop");
                utils::hide_recording_overlay(&ah);
                change_tray_icon(&ah, TrayIconState::Idle);
            }
        });

        debug!(
            "TranscribeAction::stop completed in {:?}",
            stop_time.elapsed()
        );
    }
}

// Cancel Action
struct CancelAction;

impl ShortcutAction for CancelAction {
    fn start(&self, app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        utils::cancel_current_operation(app);
    }

    fn stop(&self, _app: &AppHandle, _binding_id: &str, _shortcut_str: &str) {
        // Nothing to do on stop for cancel
    }
}

// Test Action
struct TestAction;

impl ShortcutAction for TestAction {
    fn start(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Started - {} (App: {})", // Changed "Pressed" to "Started" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }

    fn stop(&self, app: &AppHandle, binding_id: &str, shortcut_str: &str) {
        log::info!(
            "Shortcut ID '{}': Stopped - {} (App: {})", // Changed "Released" to "Stopped" for consistency
            binding_id,
            shortcut_str,
            app.package_info().name
        );
    }
}

// Static Action Map
pub static ACTION_MAP: Lazy<HashMap<String, Arc<dyn ShortcutAction>>> = Lazy::new(|| {
    let mut map = HashMap::new();
    map.insert(
        "transcribe".to_string(),
        Arc::new(TranscribeAction {
            post_process: false,
        }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "transcribe_with_post_process".to_string(),
        Arc::new(TranscribeAction { post_process: true }) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "cancel".to_string(),
        Arc::new(CancelAction) as Arc<dyn ShortcutAction>,
    );
    map.insert(
        "test".to_string(),
        Arc::new(TestAction) as Arc<dyn ShortcutAction>,
    );
    map
});

#[cfg(test)]
mod tests {
    use super::{
        normalize_post_process_candidate, normalize_standalone_chinese_digits_to_arabic,
        template_requests_arabic_digits,
    };

    #[test]
    fn template_detects_arabic_digit_intent() {
        assert!(template_requests_arabic_digits(
            "Convert standalone Chinese numerals to Arabic digits.",
        ));
        assert!(template_requests_arabic_digits("请转换为阿拉伯数字。"));
        assert!(!template_requests_arabic_digits(
            "Translate transcript into concise English.",
        ));
    }

    #[test]
    fn converts_standalone_chinese_digit_sequences() {
        let input = "一二三四五六七。阿拉伯数字的一、二、三、四。";
        let output = normalize_standalone_chinese_digits_to_arabic(input);
        assert_eq!(output, "1234567。阿拉伯数字的1、2、3、4。");
    }

    #[test]
    fn keeps_in_word_boundaries_safe() {
        let input = "一些人说一二三很好。";
        let output = normalize_standalone_chinese_digits_to_arabic(input);
        assert_eq!(output, "一些人说123很好。");
    }

    #[test]
    fn converts_model_size_tokens_with_b_unit() {
        let input = "零点8 B、零点八B、两 B、幺234、4 B、9 B。";
        let output = normalize_standalone_chinese_digits_to_arabic(input);
        assert_eq!(output, "0.8B、0.8B、2B、1234、4B、9B。");
    }

    #[test]
    fn strips_template_instruction_leakage_lines() {
        let prompt = "要求：\n1. 保持原意与事实，不新增信息，不改变结论。\n2. 去除口头重复、语气词和明显噪音。\n3. 专有名词保持原样。\n4. 内容是多点信息时用 Markdown 列表整理。\n5. 仅输出最终结果，不要解释。";
        let raw = "1. 保持原意与事实：请提供关于“两个东西”的具体信息。\n2. 去除口头重复、语气词和明显噪音：简化表达。\n请帮我看看这两个东西是什么。";
        let output = normalize_post_process_candidate(raw, prompt, false);
        assert_eq!(output, "请帮我看看这两个东西是什么。");
    }
}
