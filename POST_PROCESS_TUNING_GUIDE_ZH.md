# Handy 本地后处理调参手册（Qwen3.5）

> 适用范围：`local-qwen35` 本地后处理链路  
> 更新时间：2026-03-29

## 1. 现在可以直接调什么？

### 1.1 UI 可调（推荐优先用 UI）
- `后处理模型`：在 `Settings -> Models` 里切换本地后处理模型（0.8B / 2B / 4B / 9B）。
- `系统提示词`：在 `Settings -> Post Process` 里编辑。
- `用户提示词模板`：在 `Settings -> Post Process` 的 Prompt 模板里编辑。
- `本地质量档位`：当前版本固定为 `balanced`（稳定优先）。
- `本地高级参数（极客）`：可直接在 UI 手动修改：
  - `max_tokens`
  - `temperature`
  - `top_p`
  - `repetition_penalty`
  - `repetition_context_size`

### 1.2 代码可调（进阶）
- 本地采样参数：`max_tokens / temperature / top_p / repetition_penalty / repetition_context_size`
- 后处理超时：当前 45 秒（超时自动重启后处理引擎）
- 启动预热逻辑：应用启动时可预热本地后处理模型

---

## 2. 当前均衡参数（代码默认值）

文件：`src-tauri/src/settings.rs`（默认值） + `src-tauri/src/actions.rs`（实际使用）

| 参数 | 默认值 | UI可调范围 |
|---|---:|---:|
| max_tokens | 192 | 64 ~ 512 |
| temperature | 0.00 | 0.00 ~ 1.00 |
| top_p | 1.00 | 0.10 ~ 1.00 |
| repetition_penalty | 1.17 | 1.00 ~ 1.50 |
| repetition_context_size | 128 | 32 ~ 256 |

---

## 3. 每个参数怎么理解？

- `max_tokens`
  - 控制输出最大长度。
  - 太小会截断翻译；太大可能变慢。
  - 建议范围：`128 ~ 320`。

- `temperature`
  - 越低越稳定、越少跑偏。
  - 翻译/规范化建议固定 `0.0`。
  - 建议范围：`0.0 ~ 0.2`。

- `top_p`
  - 采样截断阈值。`1.0` 接近不截断，通常比过低更不容易怪异复读。
  - 建议范围：`0.95 ~ 1.0`。

- `repetition_penalty`
  - 防复读强度。过低会复读，过高会伤语义。
  - 建议范围：`1.12 ~ 1.22`。

- `repetition_context_size`
  - 防复读窗口大小。越大越稳，但略增开销。
  - 建议范围：`96 ~ 192`。

---

## 4. “禁思考”现在怎么做的？

- 主要靠系统提示词硬约束：只输出最终文本，不输出推理。
- 后端还有兜底清洗：
  - 清理 `<think>...</think>`
  - 清理 `no_think/No_think`
  - 清理重复行
  - 检测“提示词回显”，命中后回退原文（不写垃圾结果）

说明：禁思考不等于更快。速度主要由模型大小、首轮图编译/缓存、文本长度决定。

---

## 5. 推荐提示词模板（可直接复制）

## 5.1 系统提示词（翻译场景）

```text
You are a strict transcription post-processor.
Output rules:
1. Output only the final processed text.
2. Never output reasoning, analysis, or chain-of-thought.
3. Never output <think> tags.
4. Never output explanations, bullet points, examples, or wrappers.
5. If user asks for translation, return translation only in the target language.
```

## 5.2 用户提示词（中译英，强约束版）

```text
Translate the following text to natural English.
Rules:
1. Output English only.
2. Keep original meaning.
3. Do not add any explanation.
4. Do not repeat the instruction.

Text:
${output}
```

## 5.3 用户提示词（轻量润色，不翻译）

```text
Polish punctuation and readability only.
Keep original language and meaning.
Output only the final text.

${output}
```

---

## 6. 进阶：在哪里改代码参数？

- 本地默认参数与范围：
  - `src-tauri/src/settings.rs`
- 本地实际推理参数读取：
  - `src-tauri/src/actions.rs` -> `local_generation_params_from_settings()`
- 本地后处理服务器清洗策略：
  - `src-tauri/resources/qwen35_post_mlx/qwen35_post_server.py`
- 后处理响应超时（防卡死）：
  - `src-tauri/src/managers/qwen35_post_engine.rs`（当前 45 秒）
- 启动预热逻辑：
  - `src-tauri/src/managers/qwen35_post_manager.rs`
  - `src-tauri/src/lib.rs`

改完后重新构建：

```bash
bun run build
cargo check --manifest-path src-tauri/Cargo.toml
bun run tauri build --bundles app
```

---

## 7. 翻译不稳定时的排查顺序

1. 先确认 Provider 是 `local-qwen35`，且后处理模型已下载并选中。  
2. 系统提示词里必须保留“输出最终文本 only”的硬约束。  
3. 用户提示词必须包含 `${output}`，并明确 `Output English only`。  
4. 使用固定 `balanced` 预设，再通过高级参数微调。  
5. 如果仍有原文残留，提升 `max_tokens` 或换更大模型（2B -> 4B）。  
6. 如果出现卡住，重启 app（新版本已加超时自恢复）。  

---

## 8. 建议的默认组合（当前版本）

- 默认：`qwen35-optiq-2b + balanced + 上述默认参数`
- 想更稳翻译：优先升级模型（2B -> 4B），其次再调 `max_tokens` 与 `repetition_penalty`
