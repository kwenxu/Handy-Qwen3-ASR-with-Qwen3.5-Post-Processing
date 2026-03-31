import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { ChevronDown } from "lucide-react";
import type { AppSettings as Settings } from "@/bindings";
import { Dropdown, SettingContainer, ToggleSwitch } from "@/components/ui";
import { Input } from "../../ui/Input";
import { useSettings } from "../../../hooks/useSettings";
import { AccelerationSelector } from "../AccelerationSelector";
import { PostProcessingSettingsScriptHooks } from "../post-processing/PostProcessingSettings";

type NumericSettingKey =
  | "qwen3_startup_preload_delay_ms"
  | "qwen35_startup_preload_delay_ms"
  | "qwen3_server_ready_timeout_sec"
  | "qwen35_server_ready_timeout_sec"
  | "qwen35_inference_timeout_sec";

type PerfBlockKey =
  | "pipeline"
  | "transcription"
  | "asrScript"
  | "postProcessing"
  | "llmScript"
  | "output";

interface NumericSettingFieldProps {
  settingKey: NumericSettingKey;
  label: string;
  hint: string;
  min: number;
  max: number;
  step?: number;
  defaultValue: number;
  disabled?: boolean;
}

interface CollapsibleBlockProps {
  blockKey: PerfBlockKey;
  title: string;
  description?: string;
  open: boolean;
  onToggle: () => void;
  children: React.ReactNode;
}

type PipelineStepAction =
  | "focus_pipeline"
  | "goto_models"
  | "focus_asr_script"
  | "goto_postprocessing"
  | "focus_llm_script"
  | "focus_output";

interface PipelineOverviewProps {
  onStepAction: (action: PipelineStepAction) => void;
}

const NAVIGATE_SECTION_EVENT = "handy:navigate-section";

const clamp = (value: number, min: number, max: number): number =>
  Math.min(max, Math.max(min, value));

const NumericSettingField: React.FC<NumericSettingFieldProps> = ({
  settingKey,
  label,
  hint,
  min,
  max,
  step = 1,
  defaultValue,
  disabled = false,
}) => {
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const currentValue = Number(getSetting(settingKey) ?? defaultValue);
  const [draft, setDraft] = useState(String(currentValue));

  useEffect(() => {
    setDraft(String(currentValue));
  }, [currentValue]);

  const commitDraft = () => {
    const parsed = Number.parseInt(draft, 10);
    if (!Number.isFinite(parsed)) {
      setDraft(String(currentValue));
      return;
    }

    const normalized = clamp(parsed, min, max);
    setDraft(String(normalized));
    if (normalized !== currentValue) {
      void updateSetting(settingKey, normalized as Settings[NumericSettingKey]);
    }
  };

  return (
    <div className="space-y-1">
      <label className="text-xs text-mid-gray">{label}</label>
      <Input
        type="number"
        min={min}
        max={max}
        step={step}
        value={draft}
        onChange={(e) => {
          const next = e.target.value;
          setDraft(next);

          const parsed = Number.parseInt(next, 10);
          if (!Number.isFinite(parsed)) return;
          const normalized = clamp(parsed, min, max);
          if (normalized !== currentValue) {
            void updateSetting(
              settingKey,
              normalized as Settings[NumericSettingKey],
            );
          }
        }}
        onBlur={commitDraft}
        disabled={disabled || isUpdating(settingKey)}
        variant="compact"
      />
      <p className="text-xs text-mid-gray/70">{hint}</p>
    </div>
  );
};

const CollapsibleBlock: React.FC<CollapsibleBlockProps> = ({
  blockKey,
  title,
  description,
  open,
  onToggle,
  children,
}) => {
  return (
    <section id={`perf-block-${blockKey}`} className="space-y-2 scroll-mt-4">
      <button
        type="button"
        onClick={onToggle}
        className="w-full rounded-lg border border-mid-gray/20 bg-background px-4 py-3 text-left hover:border-logo-primary/50 transition-colors"
      >
        <div className="flex items-center justify-between gap-2">
          <div>
            <h2 className="text-xs font-medium text-mid-gray uppercase tracking-wide">
              {title}
            </h2>
            {description ? (
              <p className="mt-1 text-xs text-mid-gray/80">{description}</p>
            ) : null}
          </div>
          <ChevronDown
            className={`h-4 w-4 text-logo-primary transition-transform ${open ? "rotate-180" : "rotate-0"}`}
          />
        </div>
      </button>

      {open ? (
        <div className="bg-background border border-mid-gray/20 rounded-lg overflow-visible">
          <div className="divide-y divide-mid-gray/20">{children}</div>
        </div>
      ) : null}
    </section>
  );
};

const PipelineOverview: React.FC<PipelineOverviewProps> = ({ onStepAction }) => {
  const { t } = useTranslation();

  const steps: Array<{
    title: string;
    description: string;
    kind: "model" | "app" | "io";
    action: PipelineStepAction;
  }> = [
    {
      title: t("settings.performance.pipeline.steps.input.title"),
      description: t("settings.performance.pipeline.steps.input.description"),
      kind: "io",
      action: "focus_pipeline",
    },
    {
      title: t("settings.performance.pipeline.steps.transcriptionModel.title"),
      description: t(
        "settings.performance.pipeline.steps.transcriptionModel.description",
      ),
      kind: "model",
      action: "goto_models",
    },
    {
      title: t("settings.performance.pipeline.steps.asrAppScript.title"),
      description: t(
        "settings.performance.pipeline.steps.asrAppScript.description",
      ),
      kind: "app",
      action: "focus_asr_script",
    },
    {
      title: t("settings.performance.pipeline.steps.postProcessModel.title"),
      description: t(
        "settings.performance.pipeline.steps.postProcessModel.description",
      ),
      kind: "model",
      action: "goto_postprocessing",
    },
    {
      title: t("settings.performance.pipeline.steps.llmAppScript.title"),
      description: t(
        "settings.performance.pipeline.steps.llmAppScript.description",
      ),
      kind: "app",
      action: "focus_llm_script",
    },
    {
      title: t("settings.performance.pipeline.steps.output.title"),
      description: t("settings.performance.pipeline.steps.output.description"),
      kind: "io",
      action: "focus_output",
    },
  ];

  const badgeClasses = (kind: "model" | "app" | "io"): string => {
    if (kind === "model") {
      return "bg-logo-primary/20 border-logo-primary/40 text-text";
    }
    if (kind === "app") {
      return "bg-mid-gray/20 border-mid-gray/50 text-text";
    }
    return "bg-background-ui/20 border-background-ui/40 text-text";
  };

  const badgeLabel = (kind: "model" | "app" | "io"): string => {
    if (kind === "model") {
      return t("settings.performance.pipeline.legend.model");
    }
    if (kind === "app") {
      return t("settings.performance.pipeline.legend.app");
    }
    return t("settings.performance.pipeline.legend.io");
  };

  return (
    <SettingContainer
      title={t("settings.performance.pipeline.title")}
      description={t("settings.performance.pipeline.description")}
      descriptionMode="tooltip"
      grouped={true}
      layout="stacked"
    >
      <div className="rounded-md border border-mid-gray/20 bg-mid-gray/5 p-3">
        <div className="mb-3 flex flex-wrap gap-2">
          <span className="rounded border border-logo-primary/40 bg-logo-primary/20 px-2 py-0.5 text-xs text-text">
            {t("settings.performance.pipeline.legend.model")}
          </span>
          <span className="rounded border border-mid-gray/50 bg-mid-gray/20 px-2 py-0.5 text-xs text-text">
            {t("settings.performance.pipeline.legend.app")}
          </span>
          <span className="rounded border border-background-ui/40 bg-background-ui/20 px-2 py-0.5 text-xs text-text">
            {t("settings.performance.pipeline.legend.io")}
          </span>
        </div>

        <div className="space-y-2">
          {steps.map((step, index) => (
            <React.Fragment key={`${step.kind}-${index}`}>
              <button
                type="button"
                onClick={() => onStepAction(step.action)}
                className="w-full flex items-start justify-between gap-3 rounded-md border border-mid-gray/20 bg-mid-gray/10 p-2 text-left hover:border-logo-primary/50 hover:bg-logo-primary/5 transition-colors"
              >
                <div className="min-w-0">
                  <p className="text-sm font-semibold text-text">
                    {index + 1}. {step.title}
                  </p>
                  <p className="text-xs text-mid-gray/80">{step.description}</p>
                </div>
                <span
                  className={`shrink-0 rounded border px-2 py-0.5 text-xs ${badgeClasses(step.kind)}`}
                >
                  {badgeLabel(step.kind)}
                </span>
              </button>
              {index < steps.length - 1 ? (
                <p className="text-center text-sm font-semibold text-logo-primary">
                  ↓
                </p>
              ) : null}
            </React.Fragment>
          ))}
        </div>
      </div>
    </SettingContainer>
  );
};

export const PerformanceSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const [openBlocks, setOpenBlocks] = useState<Record<PerfBlockKey, boolean>>({
    pipeline: true,
    transcription: false,
    asrScript: false,
    postProcessing: false,
    llmScript: false,
    output: false,
  });

  const preloadStrategy =
    (getSetting("qwen_startup_preload_strategy") || "parallel").toString() ||
    "parallel";
  const qwen3StartupPreloadEnabled =
    getSetting("qwen3_startup_preload_enabled") ?? true;
  const qwen3WarmupEnabled = getSetting("qwen3_warmup_enabled") ?? true;
  const qwen35StartupPreloadEnabled =
    getSetting("qwen35_startup_preload_enabled") ?? true;
  const qwen35WarmupEnabled = getSetting("qwen35_warmup_enabled") ?? true;

  const toggleBlock = (key: PerfBlockKey) => {
    setOpenBlocks((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const focusBlock = (key: PerfBlockKey) => {
    setOpenBlocks((prev) => ({ ...prev, [key]: true }));
    window.requestAnimationFrame(() => {
      document
        .getElementById(`perf-block-${key}`)
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  const navigateToSection = (section: "models" | "postprocessing") => {
    window.dispatchEvent(
      new CustomEvent(NAVIGATE_SECTION_EVENT, {
        detail: { section },
      }),
    );
  };

  const handlePipelineStepAction = (action: PipelineStepAction) => {
    if (action === "goto_models") {
      navigateToSection("models");
      return;
    }
    if (action === "goto_postprocessing") {
      navigateToSection("postprocessing");
      return;
    }
    if (action === "focus_asr_script") {
      focusBlock("asrScript");
      return;
    }
    if (action === "focus_llm_script") {
      focusBlock("llmScript");
      return;
    }
    if (action === "focus_output") {
      focusBlock("output");
      return;
    }
    focusBlock("pipeline");
  };

  return (
    <div className="max-w-6xl w-full mx-auto space-y-6">
      <CollapsibleBlock
        blockKey="pipeline"
        title={t("settings.performance.groups.pipeline")}
        open={openBlocks.pipeline}
        onToggle={() => toggleBlock("pipeline")}
      >
        <PipelineOverview onStepAction={handlePipelineStepAction} />
      </CollapsibleBlock>

      <CollapsibleBlock
        blockKey="transcription"
        title={t("settings.performance.groups.transcription")}
        open={openBlocks.transcription}
        onToggle={() => toggleBlock("transcription")}
      >
        <SettingContainer
          title={t("settings.performance.qwenTranscription.title")}
          description={t("settings.performance.qwenTranscription.description")}
          descriptionMode="tooltip"
          grouped={true}
          layout="stacked"
        >
          <p className="text-xs text-mid-gray">
            {t("settings.performance.qwenTranscription.hint")}
          </p>
        </SettingContainer>

        <ToggleSwitch
          checked={qwen3StartupPreloadEnabled}
          onChange={(enabled) => {
            void updateSetting("qwen3_startup_preload_enabled", enabled);
          }}
          isUpdating={isUpdating("qwen3_startup_preload_enabled")}
          label={t("settings.performance.qwenPanel.qwen3Preload.label")}
          description={t(
            "settings.performance.qwenPanel.qwen3Preload.description",
          )}
          descriptionMode="tooltip"
          grouped={true}
        />

        <ToggleSwitch
          checked={qwen3WarmupEnabled}
          onChange={(enabled) => {
            void updateSetting("qwen3_warmup_enabled", enabled);
          }}
          isUpdating={isUpdating("qwen3_warmup_enabled")}
          label={t("settings.performance.qwenPanel.qwen3Warmup.label")}
          description={t("settings.performance.qwenPanel.qwen3Warmup.description")}
          descriptionMode="tooltip"
          grouped={true}
        />

        <SettingContainer
          title={t("settings.performance.qwenPanel.startupTiming.title")}
          description={t("settings.performance.qwenPanel.startupTiming.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <NumericSettingField
              settingKey="qwen3_startup_preload_delay_ms"
              label={t(
                "settings.performance.qwenPanel.startupTiming.qwen3Delay.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.startupTiming.qwen3Delay.hint",
              )}
              min={0}
              max={15000}
              step={100}
              defaultValue={0}
              disabled={!qwen3StartupPreloadEnabled}
            />
            <NumericSettingField
              settingKey="qwen3_server_ready_timeout_sec"
              label={t("settings.performance.qwenPanel.timeouts.qwen3Ready.label")}
              hint={t("settings.performance.qwenPanel.timeouts.qwen3Ready.hint")}
              min={10}
              max={120}
              defaultValue={30}
            />
          </div>
        </SettingContainer>

        <AccelerationSelector
          descriptionMode="tooltip"
          grouped={true}
          showOrt={false}
          whisperTitleOverride={t("settings.performance.whisperExperimental")}
        />
      </CollapsibleBlock>

      <CollapsibleBlock
        blockKey="asrScript"
        title={t("settings.performance.scriptBlocks.asr.title")}
        description={t("settings.performance.scriptBlocks.asr.description")}
        open={openBlocks.asrScript}
        onToggle={() => toggleBlock("asrScript")}
      >
        <PostProcessingSettingsScriptHooks
          scope="asr"
          sectionId="perf-script-asr"
        />
      </CollapsibleBlock>

      <CollapsibleBlock
        blockKey="postProcessing"
        title={t("settings.performance.groups.postProcessing")}
        open={openBlocks.postProcessing}
        onToggle={() => toggleBlock("postProcessing")}
      >
        <SettingContainer
          title={t("settings.performance.qwenPanel.title")}
          description={t("settings.postProcessing.api.localModel.description")}
          descriptionMode="tooltip"
          grouped={true}
          layout="stacked"
        >
          <p className="text-xs text-mid-gray">
            {t("settings.postProcessing.api.localModel.title")}
          </p>
        </SettingContainer>

        <SettingContainer
          title={t("settings.performance.qwenPanel.strategy.title")}
          description={t("settings.performance.qwenPanel.strategy.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <Dropdown
            selectedValue={preloadStrategy}
            options={[
              {
                value: "parallel",
                label: t(
                  "settings.performance.qwenPanel.strategy.options.parallel",
                ),
              },
              {
                value: "serial",
                label: t("settings.performance.qwenPanel.strategy.options.serial"),
              },
            ]}
            onSelect={(value) => {
              if (!value) return;
              void updateSetting(
                "qwen_startup_preload_strategy",
                value as Settings["qwen_startup_preload_strategy"],
              );
            }}
            disabled={isUpdating("qwen_startup_preload_strategy")}
            className="w-[220px]"
          />
        </SettingContainer>

        <ToggleSwitch
          checked={qwen35StartupPreloadEnabled}
          onChange={(enabled) => {
            void updateSetting("qwen35_startup_preload_enabled", enabled);
          }}
          isUpdating={isUpdating("qwen35_startup_preload_enabled")}
          label={t("settings.performance.qwenPanel.qwen35Preload.label")}
          description={t(
            "settings.performance.qwenPanel.qwen35Preload.description",
          )}
          descriptionMode="tooltip"
          grouped={true}
        />

        <ToggleSwitch
          checked={qwen35WarmupEnabled}
          onChange={(enabled) => {
            void updateSetting("qwen35_warmup_enabled", enabled);
          }}
          isUpdating={isUpdating("qwen35_warmup_enabled")}
          label={t("settings.performance.qwenPanel.qwen35Warmup.label")}
          description={t("settings.performance.qwenPanel.qwen35Warmup.description")}
          descriptionMode="tooltip"
          grouped={true}
        />

        <SettingContainer
          title={t("settings.performance.qwenPanel.startupTiming.title")}
          description={t("settings.performance.qwenPanel.startupTiming.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <NumericSettingField
              settingKey="qwen35_startup_preload_delay_ms"
              label={t(
                "settings.performance.qwenPanel.startupTiming.qwen35Delay.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.startupTiming.qwen35Delay.hint",
              )}
              min={0}
              max={15000}
              step={100}
              defaultValue={0}
              disabled={!qwen35StartupPreloadEnabled}
            />
            <NumericSettingField
              settingKey="qwen35_server_ready_timeout_sec"
              label={t(
                "settings.performance.qwenPanel.timeouts.qwen35Startup.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.timeouts.qwen35Startup.hint",
              )}
              min={20}
              max={300}
              defaultValue={90}
            />
          </div>
        </SettingContainer>

        <SettingContainer
          title={t("settings.performance.qwenPanel.timeouts.title")}
          description={t("settings.performance.qwenPanel.timeouts.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <NumericSettingField
              settingKey="qwen35_inference_timeout_sec"
              label={t(
                "settings.performance.qwenPanel.timeouts.qwen35Inference.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.timeouts.qwen35Inference.hint",
              )}
              min={5}
              max={180}
              defaultValue={45}
            />
          </div>
        </SettingContainer>
      </CollapsibleBlock>

      <CollapsibleBlock
        blockKey="llmScript"
        title={t("settings.performance.scriptBlocks.llm.title")}
        description={t("settings.performance.scriptBlocks.llm.description")}
        open={openBlocks.llmScript}
        onToggle={() => toggleBlock("llmScript")}
      >
        <PostProcessingSettingsScriptHooks
          scope="llm"
          sectionId="perf-script-llm"
        />
      </CollapsibleBlock>

      <CollapsibleBlock
        blockKey="output"
        title={t("settings.performance.outputStage.title")}
        description={t("settings.performance.outputStage.description")}
        open={openBlocks.output}
        onToggle={() => toggleBlock("output")}
      >
        <SettingContainer
          title={t("settings.performance.outputStage.noteTitle")}
          description={t("settings.performance.outputStage.noteDescription")}
          descriptionMode="tooltip"
          grouped={true}
          layout="stacked"
        >
          <p className="text-xs text-mid-gray/80">
            {t("settings.performance.outputStage.note")}
          </p>
        </SettingContainer>
      </CollapsibleBlock>
    </div>
  );
};
