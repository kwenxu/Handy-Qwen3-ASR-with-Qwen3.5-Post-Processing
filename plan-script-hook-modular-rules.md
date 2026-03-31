# Handy 规则脚本化改造计划（转录后 + 后处理后）

## 1. 目标
- 将现有“内置清洗规则”从纯硬编码形态升级为“可见 + 可配置 + 可插拔脚本”的形态。
- 提供两个可插拔阶段：
  1. `ASR 转录输出后`（ASR Post Hook）
  2. `后处理模型输出后`（LLM Post Hook）
- 保留最小安全兜底，避免脚本错误导致主流程不可用。

## 2. 设计原则
- 不执行不受控远程代码；仅执行用户本地显式配置的脚本路径。
- 业务规则可替换，安全兜底最小化且透明化。
- 脚本失败不影响主流程：超时/异常/空输出时自动回退原文。
- 提供“内置默认规则说明”和“默认脚本模板导出”。

## 3. 现状梳理（当前代码）
- 转录后清洗（Rust）：
  - `src-tauri/src/audio_toolkit/text.rs`
  - 主要逻辑：`apply_custom_words`、`filter_transcription_output`
- 后处理后清洗（Rust/Python）：
  - `src-tauri/src/actions.rs`
  - `src-tauri/resources/qwen35_post_mlx/qwen35_post_server.py`
- 用户可见层：
  - 系统提示词、用户提示词模板、本地高级参数（温度等）

## 4. 目标能力（交付后）
- 新增两段用户脚本 Hook：
  - `post_asr_script_path`
  - `post_llm_script_path`
- 新增脚本执行策略：
  - 开关：启用/禁用
  - 超时（毫秒）
  - 失败策略：`fallback_original`
  - 执行顺序：`builtin_then_script` / `script_only`
- 新增“默认模板导出”：
  - `asr_post_hook.py`
  - `llm_post_hook.py`
- 新增“内置规则只读说明面板”：
  - 展示每条默认规则、作用、风险说明

## 5. 脚本接口协议（固定）
### 5.1 输入（stdin JSON）
```json
{
  "stage": "asr_post | llm_post",
  "text": "待处理文本",
  "lang": "zh | en | ...",
  "model_id": "当前模型ID",
  "provider_id": "当前后处理提供商",
  "prompt_id": "当前提示词模板ID",
  "system_prompt": "系统提示词",
  "user_prompt_template": "用户提示词模板",
  "metadata": {}
}
```

### 5.2 输出（stdout JSON）
```json
{
  "text": "处理后的文本",
  "warnings": ["可选告警"],
  "meta": {}
}
```

### 5.3 失败判定
- 非 0 退出码
- stdout 非法 JSON
- `text` 缺失或非字符串
- 超时

以上任一情况触发回退策略：使用脚本前原文继续流程。

## 6. 实施拆分
## Phase A：后端基础能力
- 新增设置项（Rust `settings.rs` + 前端 store/bindings）：
  - `script_hooks_enabled`
  - `post_asr_script_path`
  - `post_llm_script_path`
  - `script_hook_timeout_ms`
  - `script_hook_order`（`builtin_then_script` / `script_only`）
- 新增脚本执行器模块（建议：`src-tauri/src/managers/script_hook.rs`）：
  - 统一 spawn、stdin/stdout、超时、错误封装、日志
- 在两处接入：
  - `TranscriptionManager::transcribe` 结果后（ASR 后）
  - `process_transcription_output` 的后处理结果后（LLM 后）

## Phase B：UI 与可见性
- 后处理页面增加“脚本 Hook（极客）”分组：
  - 总开关
  - 两个脚本路径
  - 执行顺序
  - 超时
  - 测试按钮（输入示例文本，回显脚本输出）
- 增加“内置规则说明”只读区域：
  - 转录后默认规则
  - 后处理后默认规则
  - 明确哪些是不可关闭的最小安全兜底

## Phase C：模板导出与回滚
- 在 App Data 目录导出默认脚本模板（按钮触发）：
  - `script-hooks/asr_post_hook.py`
  - `script-hooks/llm_post_hook.py`
- 提供“恢复默认模板”按钮（覆盖导出模板）。
- 记录脚本执行日志到 `~/Library/Logs/com.pais.handy/handy.log`。

## 7. 最小安全兜底（保留内置）
- 空文本快速返回
- 脚本超时保护
- 异常回退原文
- 非法输出回退原文
- 基础不可见字符清理（仅安全级）

## 8. 验收标准（DoD）
- 可在 UI 选择/启用两段脚本并实时生效。
- 任一脚本崩溃或超时，转录流程仍可完成且不丢文本。
- 能导出默认脚本模板并成功运行。
- 日志中可看到每次 Hook 的耗时与结果状态（success/fallback/error）。
- 关闭脚本 Hook 后行为与当前版本一致。

## 9. 风险与对策
- 性能波动：默认超时 1200ms，可调；超时即回退。
- 用户脚本质量差：提供模板、说明和测试按钮。
- 跨平台路径问题：先保证 macOS；其他平台后续补齐。
- 安全风险：仅执行本地明确路径，不支持远程下载自动执行。

## 10. 里程碑建议
- M1（后端接线）：1 天
- M2（UI/设置接线）：1 天
- M3（模板导出/日志/测试按钮）：0.5~1 天
- M4（联调与回归）：0.5 天

## 11. 本次改造边界（不做）
- 不做在线脚本市场/自动下载执行
- 不做任意动态插件沙箱
- 不做远程脚本签名验证

## 12. 后续可扩展
- 支持 Node.js Hook（与 Python 并行）
- 支持“脚本包”导入导出（zip）
- 支持每个模型独立脚本策略
