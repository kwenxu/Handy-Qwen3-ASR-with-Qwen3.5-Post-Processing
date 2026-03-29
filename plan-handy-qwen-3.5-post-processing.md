# Handy-Qwen3-ASR-with-Qwen3.5-Post-Processing 计划书

## 0. 当前基线确认
- 本地仓库当前分支：`work/v0.8.1-from-smilingpoplar`
- 当前提交：`3229617`（对应你说的 `v0.8.1` 基线）
- 现状：
  - ASR 已有本地 Python 常驻通道（`qwen3_asr_server.py`），不是每次冷启动。
  - ASR 下载链路已经走 `HF_ENDPOINT = https://hf-mirror.com`。
  - 后处理当前仍是「远程 OpenAI 兼容 API」模式（`llm_client.rs`），不是本地 MLX。

## 1. 目标定义

### 1.1 仓库命名与描述
- 仓库名目标：`Handy-Qwen3-ASR-with-Qwen3.5-Post-Processing`
- 仓库描述目标：`Integrated local Qwen3-ASR (0.6B/1.7B) and Qwen3.5 small LLMs for ASR post-processing.`

### 1.2 功能目标
- 在 `v0.8.1` 基线上新增「本地 Qwen3.5 后处理引擎」。
- 运行时同时持有两个模型：
  - 模型 A：ASR（现有 Qwen3-ASR-0.6B/1.7B）
  - 模型 B：后处理（新增 Qwen3.5 小模型）
- 后处理改为本地 Python 通道，避免每次请求走外部 API。
- 运行目标是「纯本地离线后处理」：模型下载完成后，推理不依赖联网。

### 1.3 后处理范围归属（按 Prompt 模板管理）
- 后处理“做什么”不作为独立功能节点，而是归到 `Prompt 模板` 配置层。
- 具体规则（数字归一、字母归一、标点分句、轻量纠错）统一由模板定义并可切换。
- 详见 `3.5 后处理 Prompt 模板（预置）`。

## 2. 模型清单与可用性（已移除 Claude/Huihui）

### 2.1 首发模型（Qwen3.5 OptiQ，非 VLM）
- `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`（默认档，速度优先）
- `mlx-community/Qwen3.5-2B-OptiQ-4bit`（均衡档）
- `mlx-community/Qwen3.5-4B-OptiQ-4bit`（质量档）
- `mlx-community/Qwen3.5-9B-OptiQ-4bit`（可选极客档，首发可不默认）

### 2.2 “是否严格 Instruct”结论
- `Instruct`：代表该模型是“指令对齐版本”，对“按你给的规则执行”通常更稳定。
- `OptiQ`：不是任务类型，而是量化方式。根据模型卡，`OptiQ` 是按层敏感度做混合精度量化（不是每层统一 4bit），目标是在体积/速度和质量之间做更优平衡。
- `Qwen3.5-*-OptiQ-4bit` 命名不带 `Instruct`，不属于“严格按 Instruct 命名”的系列。
- 对你当前场景（单 prompt 后处理，不调用工具）先用 `Qwen3.5 OptiQ` 三档即可。
- Instruct 先不作为首发模型池，后续只有在“规则遵循明显不稳”时再增补。

### 2.3 可用性核验（2026-03-28）
- 以下模型在 `huggingface.co/api/models/...` 与 `hf-mirror.com/api/models/...` 均返回 `HTTP 200`：
  - `Qwen3.5-0.8B-OptiQ-4bit`
  - `Qwen3.5-2B-OptiQ-4bit`
  - `Qwen3.5-4B-OptiQ-4bit`
  - `Qwen3.5-9B-OptiQ-4bit`
- 备注：镜像网页端偶发 `429`（页面限流）不影响 API 下载端点。

### 2.4 镜像站实测结果（看得到 + 拉得下）
- `hf-mirror` API 可读到仓库元信息：
  - `0.8B`: `sha=9ce89f2a58...`，`files=9`
  - `2B`: `sha=9560d17a56...`，`files=9`
  - `4B`: `sha=c631f42328...`，`files=9`
- `README.md` 实际拉取 smoke test 通过（`/resolve/main/README.md` 可返回有效内容）。
- 使用项目当前链路 `model_info.py`（`HfApi(endpoint="https://hf-mirror.com")`）实测通过：
  - `0.8B`: `revision=9ce89f2a58cb...`，`total=618,259,103 bytes`
  - `2B`: `revision=9560d17a5636...`，`total=1,451,311,389 bytes`
  - `4B`: `revision=c631f42328b4...`，`total=2,968,025,326 bytes`
- 使用 `hf_hub_download(..., filename="README.md", endpoint="https://hf-mirror.com")` 实测可下载到本地缓存。

### 2.5 首发建议
- 首发只采用 `Qwen3.5 OptiQ` 三档模型池。
- 默认模型建议：`Qwen3.5-2B-OptiQ-4bit`（速度与质量均衡）。
- `9B` 作为实验开关，不放默认。
- Claude/Huihui 全部从计划中移除，不纳入后处理主线。

### 2.6 后处理模型可选池（镜像实查后）
- A 组：Qwen3.5 OptiQ（主线，文本后处理）
  - `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`（快）
  - `mlx-community/Qwen3.5-2B-OptiQ-4bit`（均衡）
  - `mlx-community/Qwen3.5-4B-OptiQ-4bit`（质量）
  - 说明：`pipeline_tag=text-generation`，适合后处理。
- B 组：未来可选（非首发）
  - Instruct 线保留为未来扩展项，仅在实测显示 OptiQ 规则遵循不足时再接入。
- 不建议纳入后处理默认白名单：
  - `mlx-community/Qwen3.5-0.8B-4bit`
  - `mlx-community/Qwen3.5-2B-4bit`
  - 原因：模型卡 `pipeline_tag=image-text-to-text`（`mlx-vlm`链路），不是我们当前文本后处理主线。
  - `mlx-community/Qwen3.5-4B-4bit`
  - 原因：元数据不完整（实查无 `README.md`），不作为首发稳定选项。

## 3. 技术方案（双模型常驻）

## 3.1 架构改造
- 保留现有 ASR 通道：
  - `src-tauri/resources/qwen3_asr_mlx/qwen3_asr_server.py`
- 新增后处理通道：
  - `src-tauri/resources/qwen35_post_mlx/qwen35_post_server.py`
  - 职责：接收文本 + prompt，返回后处理文本；模型常驻内存。
- 该通道仅本地调用，不引入联网推理依赖。

## 3.2 Rust 侧新增模块
- 新增：`src-tauri/src/managers/qwen35_post_engine.rs`
  - 与 `qwen3_engine.rs` 同类 IPC 机制（stdin/stdout JSON）。
  - 提供 `load_model / process_text / unload_model`。
- 在 `actions.rs::post_process_transcription()` 增加本地分支：
  - 当 provider 为 `local-qwen35` 时，走本地引擎，不走 `llm_client.rs`。
- 增加后处理模板策略：
  - `normalize_numbers = true`
  - `normalize_spelled_letters = true`
  - `preserve_meaning = true`
  - `no_fabrication = true`

## 3.3 设置与 UI
- `settings.rs` 增加本地 provider：
  - `id = local-qwen35`
  - `base_url` 可保留占位（本地引擎实际不使用 HTTP）
  - `supports_structured_output = false`（初期）
- 新增本地后处理模型选择项（独立于 ASR 模型选择）：
  - 例如：`post_process_local_model_id`
- 前端设置页新增：
  - 后处理模式：`Remote API / Local Qwen3.5`
  - 模型区改为双栏并排：
    - 左栏：`转录模型`（保持现有）
    - 右栏：`后处理模型`（新增，放在转录模型右边）
  - 后处理模型列表提供与转录模型一致的交互：下载、删除、加载、切换当前模型。
  - 后处理模型列表按分组展示：`Qwen3.5 OptiQ / Experimental`。
  - 默认只显示“推荐模型”，用户可展开“实验模型”。
  - 预设档位：`Fast(0.8B) / Balance(2B) / Quality(4B)`。
  - prompt 模板下拉：可直接切换后处理策略（见 3.5）。

## 3.4 下载与镜像策略
- 复用 `model_info.py` + `HF_ENDPOINT` 机制。
- 后处理模型下载 endpoint 优先级：
  1. 用户配置镜像（可选）
  2. `https://hf-mirror.com`
  3. `https://huggingface.co`（回退）
- 下载完整性继续用 manifest 校验（与现有 Qwen3 下载一致）。
- 模型入池硬规则（避免 VLM 混入后处理）：
  - 仅允许 `pipeline_tag = text-generation`。
  - 显式排除 `pipeline_tag = image-text-to-text`。
  - 名称包含 `-VL-` / `VLM` 的模型不进入后处理推荐列表。
  - 默认优先 `mlx-community`；第三方作者模型仅放“实验模型”分组。

## 3.5 后处理 Prompt 模板（预置）
- 模板目标：减少你每次手改 prompt 的成本，按场景一键切换。
- 模板 A：`standard-normalize`（默认）
  - 中文数字转阿拉伯数字。
  - 拼读字母合并成标准词形（如 `H A N D Y -> HANDY`）。
  - 标点、空格、分句清理。
  - 严格保留语义，不补事实。
- 模板 B：`strict-literal`
  - 最小改写，尽量逐字保留，仅修正明显 ASR 错字和标点。
- 模板 C：`readable-polish`
  - 在不改原意前提下增强可读性，适合长段口语转书面。
- 模板 D：`domain-tech`
  - 对英文缩写、术语、版本号（如 `0.8B`、`Qwen3.5`）做更保守规范化。
- 交互规则：
  - UI 可选模板 + “自定义模板”文本框（可覆盖预置）。
  - 每次调用回传 `template_id` 到后端，后端拼接系统提示词执行。

## 4. 开发分阶段（执行顺序）

## 阶段 A：命名与文档
- 更新仓库名、描述（GitHub 侧手动）。
- 更新 `README` 顶部标题与一句话介绍。
- 新增后处理模型说明区（模型档位建议 + 内存建议）。

## 阶段 B：后处理本地引擎 MVP
- 落地 `qwen35_post_server.py`（单模型常驻）。
- Rust 加 `qwen35_post_engine.rs`，打通请求/响应。
- `actions.rs` 接入 `local-qwen35` provider 分支。
- 同时接入 4 套预置 prompt 模板（含默认模板）。

## 阶段 C：模型管理与下载（Qwen3.5 OptiQ 三档）
- 在 `ModelManager` 扩展后处理模型清单（首发）：
  - `Qwen3.5-0.8B-OptiQ-4bit`
  - `Qwen3.5-2B-OptiQ-4bit`
  - `Qwen3.5-4B-OptiQ-4bit`
- 增加下载、删除、完整性校验、进度事件。
- 前端提供：
  - 后处理模型逐个下载按钮（不做一键全下）。
  - 后处理模型逐个加载/切换（与转录模型同逻辑）。
- 模型选择策略：
  - 默认建议：`Qwen3.5-2B-OptiQ-4bit`（均衡）。
  - 速度优先：`Qwen3.5-0.8B-OptiQ-4bit`。
  - 质量优先：`Qwen3.5-4B-OptiQ-4bit`。

## 阶段 D：双模型资源管理
- 明确并发策略：
  - ASR 与后处理串行执行（同一次录音），避免峰值内存叠加过猛。
- 增加闲置卸载策略：
  - ASR 超时卸载（已有）
  - 后处理超时卸载（新增，默认更短）

## 阶段 E：质量与回归
- 功能回归：ASR-only、后处理-only、ASR+后处理链路。
- 压测：长文本、连续触发、模型切换、下载中断恢复。
- 兼容回归：现有远程 provider（OpenAI/OpenRouter/Z.AI 等）不退化。
- 模型选型评测（决定默认后处理模型）：
  - 评测集：数字归一、英文拼写归一、口语词清理、列表排版、中文转英文等场景。
  - 指标：规则遵循率、误改率、平均耗时（ms）、tokens/s。
  - 输出：按 `Fast / Balance / Quality` 给出默认模型建议，并记录可复现结果。

## 5. 风险与控制
- 风险 1：内存峰值过高（双模型常驻）
  - 控制：默认 2B OptiQ；4B 标注“高内存需求”；启用空闲卸载。
- 风险 2：镜像端限流（429）
  - 控制：优先 API 端点；失败自动回退到 huggingface 官方端点。
- 风险 3：后处理改写过度
  - 控制：约束 prompt + 最大改写幅度限制 + 一键回退原文。

## 6. 验收标准（Definition of Done）
- 用户可在设置里切换到本地后处理模式。
- 本地后处理模型可下载、可删除、可切换。
- 三档模型（`0.8B/2B/4B`）均可从镜像下载并通过完整性校验。
- 录音一次后，能得到：
  - 原始转录文本（ASR）
  - 本地后处理文本（Qwen 后处理模型）
- 示例规则可通过：
  - `一二三四五六` -> `123456`
  - `H A N D Y` -> `HANDY`
- 断网条件下（模型已下载）后处理仍可运行。
- 远程 API 后处理路径功能保持可用。
- 预置 prompt 模板可切换，模板切换后输出风格可观察到一致差异。

## 7. 你需要手动做的事
- 在 GitHub 仓库页面手动改：
  - Repository name -> `Handy-Qwen3-ASR-with-Qwen3.5-Post-Processing`
  - Description -> `Integrated local Qwen3-ASR (0.6B/1.7B) and Qwen3.5 small LLMs for ASR post-processing.`
- 确认首批模型范围（建议先定 3 个）：
  - OptiQ：`0.8B / 2B / 4B`

## 8. 下一步实施建议
- 本轮直接落地 `Qwen3.5 OptiQ` 三档首发模型，不拆第二轮。
- 先保证 `standard-normalize` 默认模板质量，再完善其余模板。
- `9B` 作为可选实验档。
- 若后续评测显示规则遵循不足，再考虑增补 Instruct 线。
