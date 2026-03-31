import React, { useEffect, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { RefreshCcw } from "lucide-react";
import { open } from "@tauri-apps/plugin-dialog";
import { commands } from "@/bindings";

import { Alert } from "../../ui/Alert";
import {
  Dropdown,
  SettingContainer,
  SettingsGroup,
  Textarea,
  ToggleSwitch,
} from "@/components/ui";
import { Button } from "../../ui/Button";
import { ResetButton } from "../../ui/ResetButton";
import { Input } from "../../ui/Input";

import { ProviderSelect } from "../PostProcessingSettingsApi/ProviderSelect";
import { BaseUrlField } from "../PostProcessingSettingsApi/BaseUrlField";
import { ApiKeyField } from "../PostProcessingSettingsApi/ApiKeyField";
import { ModelSelect } from "../PostProcessingSettingsApi/ModelSelect";
import { usePostProcessProviderState } from "../PostProcessingSettingsApi/usePostProcessProviderState";
import { useSettings } from "../../../hooks/useSettings";
import { LocalQwen35Models } from "./LocalQwen35Models";
import { LocalPostProcessAdvancedSettings } from "./LocalPostProcessAdvancedSettings";

type PromptTemplate = {
  id: string;
  name: string;
  prompt: string;
};

const LOCAL_POST_PROCESS_PROVIDER_ID = "local-qwen35";
const SCRIPT_HOOK_TIMEOUT_MIN = 100;
const SCRIPT_HOOK_TIMEOUT_MAX = 10_000;

const isPresetPrompt = (prompt: PromptTemplate): boolean =>
  prompt.id.startsWith("template_") || prompt.id.startsWith("default_");

const isTranslatePresetPrompt = (prompt: PromptTemplate): boolean =>
  prompt.id.startsWith("template_translate_english");

const promptSortRank = (prompt: PromptTemplate): number => {
  if (isTranslatePresetPrompt(prompt)) return 0;
  if (isPresetPrompt(prompt)) return 1;
  return 2;
};

const PostProcessingSettingsApiComponent: React.FC = () => {
  const { t } = useTranslation();
  const state = usePostProcessProviderState();
  const { getSetting, updatePostProcessModel, setPostProcessProvider } =
    useSettings();
  const selectedLocalPostProcessModel =
    getSetting("post_process_models")?.[LOCAL_POST_PROCESS_PROVIDER_ID] || "";

  const handleLocalModelSelect = async (modelId: string) => {
    await updatePostProcessModel(LOCAL_POST_PROCESS_PROVIDER_ID, modelId);
    await setPostProcessProvider(LOCAL_POST_PROCESS_PROVIDER_ID);
  };

  return (
    <>
      <SettingContainer
        title={t("settings.postProcessing.api.provider.title")}
        description={t("settings.postProcessing.api.provider.description")}
        descriptionMode="tooltip"
        layout="horizontal"
        grouped={true}
      >
        <div className="flex items-center gap-2">
          <ProviderSelect
            options={state.providerOptions}
            value={state.selectedProviderId}
            onChange={state.handleProviderSelect}
          />
        </div>
      </SettingContainer>

      {state.isLocalProvider && (
        <>
          <LocalQwen35Models
            selectedModelId={selectedLocalPostProcessModel}
            onSelectModel={(modelId) => {
              void handleLocalModelSelect(modelId);
            }}
            title={t("settings.postProcessing.api.localModel.title")}
            description={t(
              "settings.postProcessing.api.localModel.description",
            )}
            grouped={true}
          />
          <LocalPostProcessAdvancedSettings grouped={true} />
        </>
      )}

      {state.isAppleProvider ? (
        state.appleIntelligenceUnavailable ? (
          <Alert variant="error" contained>
            {t("settings.postProcessing.api.appleIntelligence.unavailable")}
          </Alert>
        ) : null
      ) : !state.isLocalProvider ? (
        <>
          {state.selectedProvider?.id === "custom" && (
            <SettingContainer
              title={t("settings.postProcessing.api.baseUrl.title")}
              description={t("settings.postProcessing.api.baseUrl.description")}
              descriptionMode="tooltip"
              layout="horizontal"
              grouped={true}
            >
              <div className="flex items-center gap-2">
                <BaseUrlField
                  value={state.baseUrl}
                  onBlur={state.handleBaseUrlChange}
                  placeholder={t(
                    "settings.postProcessing.api.baseUrl.placeholder",
                  )}
                  disabled={state.isBaseUrlUpdating}
                  className="min-w-[380px]"
                />
              </div>
            </SettingContainer>
          )}

          <SettingContainer
            title={t("settings.postProcessing.api.apiKey.title")}
            description={t("settings.postProcessing.api.apiKey.description")}
            descriptionMode="tooltip"
            layout="horizontal"
            grouped={true}
          >
            <div className="flex items-center gap-2">
              <ApiKeyField
                value={state.apiKey}
                onBlur={state.handleApiKeyChange}
                placeholder={t(
                  "settings.postProcessing.api.apiKey.placeholder",
                )}
                disabled={state.isApiKeyUpdating}
                className="min-w-[320px]"
              />
            </div>
          </SettingContainer>
        </>
      ) : null}

      {!state.isAppleProvider && !state.isLocalProvider && (
        <SettingContainer
          title={t("settings.postProcessing.api.model.title")}
          description={
            state.isCustomProvider
              ? t("settings.postProcessing.api.model.descriptionCustom")
              : t("settings.postProcessing.api.model.descriptionDefault")
          }
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="flex items-center gap-2">
            <ModelSelect
              value={state.model}
              options={state.modelOptions}
              disabled={state.isModelUpdating}
              isLoading={state.isFetchingModels}
              placeholder={
                state.modelOptions.length > 0
                  ? t(
                      "settings.postProcessing.api.model.placeholderWithOptions",
                    )
                  : t("settings.postProcessing.api.model.placeholderNoOptions")
              }
              onSelect={state.handleModelSelect}
              onCreate={state.handleModelCreate}
              onBlur={() => {}}
              className="flex-1 min-w-[380px]"
            />
            <ResetButton
              onClick={state.handleRefreshModels}
              disabled={state.isFetchingModels}
              ariaLabel={t("settings.postProcessing.api.model.refreshModels")}
              className="flex h-10 w-10 items-center justify-center"
            >
              <RefreshCcw
                className={`h-4 w-4 ${state.isFetchingModels ? "animate-spin" : ""}`}
              />
            </ResetButton>
          </div>
        </SettingContainer>
      )}
    </>
  );
};

const PostProcessingSettingsPromptsComponent: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating, refreshSettings } =
    useSettings();
  const [isCreating, setIsCreating] = useState(false);
  const currentSystemPrompt = (
    getSetting("post_process_system_prompt") || ""
  ).toString();
  const [systemPromptDraft, setSystemPromptDraft] =
    useState(currentSystemPrompt);
  const [draftName, setDraftName] = useState("");
  const [draftText, setDraftText] = useState("");

  const prompts = (getSetting("post_process_prompts") ||
    []) as PromptTemplate[];
  const sortedPrompts = [...prompts].sort((a, b) => {
    const rankDiff = promptSortRank(a) - promptSortRank(b);
    if (rankDiff !== 0) return rankDiff;
    return a.name.localeCompare(b.name);
  });
  const selectedPromptId = getSetting("post_process_selected_prompt_id") || "";
  const selectedPrompt =
    prompts.find((prompt) => prompt.id === selectedPromptId) || null;

  useEffect(() => {
    if (isCreating) return;

    if (selectedPrompt) {
      setDraftName(selectedPrompt.name);
      setDraftText(selectedPrompt.prompt);
    } else {
      setDraftName("");
      setDraftText("");
    }
  }, [
    isCreating,
    selectedPromptId,
    selectedPrompt?.name,
    selectedPrompt?.prompt,
  ]);

  useEffect(() => {
    setSystemPromptDraft(currentSystemPrompt);
  }, [currentSystemPrompt]);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      const trimmed = systemPromptDraft.trim();
      if (!trimmed || trimmed === currentSystemPrompt.trim()) return;
      void updateSetting("post_process_system_prompt", trimmed);
    }, 350);

    return () => window.clearTimeout(timer);
  }, [systemPromptDraft, currentSystemPrompt, updateSetting]);

  const saveSystemPrompt = () => {
    const trimmed = systemPromptDraft.trim();
    if (!trimmed || trimmed === currentSystemPrompt.trim()) return;
    void updateSetting("post_process_system_prompt", trimmed);
  };

  const handlePromptSelect = (promptId: string | null) => {
    if (!promptId) return;
    updateSetting("post_process_selected_prompt_id", promptId);
    setIsCreating(false);
  };

  const handleCreatePrompt = async () => {
    if (!draftName.trim() || !draftText.trim()) return;

    try {
      const result = await commands.addPostProcessPrompt(
        draftName.trim(),
        draftText.trim(),
      );
      if (result.status === "ok") {
        await refreshSettings();
        updateSetting("post_process_selected_prompt_id", result.data.id);
        setIsCreating(false);
      }
    } catch (error) {
      console.error("Failed to create prompt:", error);
    }
  };

  const handleUpdatePrompt = async () => {
    if (!selectedPromptId || !draftName.trim() || !draftText.trim()) return;

    try {
      await commands.updatePostProcessPrompt(
        selectedPromptId,
        draftName.trim(),
        draftText.trim(),
      );
      await refreshSettings();
    } catch (error) {
      console.error("Failed to update prompt:", error);
    }
  };

  const handleDeletePrompt = async (promptId: string) => {
    if (!promptId) return;

    try {
      await commands.deletePostProcessPrompt(promptId);
      await refreshSettings();
      setIsCreating(false);
    } catch (error) {
      console.error("Failed to delete prompt:", error);
    }
  };

  const handleCancelCreate = () => {
    setIsCreating(false);
    if (selectedPrompt) {
      setDraftName(selectedPrompt.name);
      setDraftText(selectedPrompt.prompt);
    } else {
      setDraftName("");
      setDraftText("");
    }
  };

  const handleStartCreate = () => {
    setIsCreating(true);
    setDraftName("");
    setDraftText("");
  };

  const hasPrompts = prompts.length > 0;
  const presetBadge = t("settings.postProcessing.prompts.presetBadge", {
    defaultValue: "预置",
  });
  const translatePresetBadge = t(
    "settings.postProcessing.prompts.translatePresetBadge",
    {
      defaultValue: "翻译预置",
    },
  );
  const isDirty =
    !!selectedPrompt &&
    (draftName.trim() !== selectedPrompt.name ||
      draftText.trim() !== selectedPrompt.prompt.trim());

  return (
    <SettingContainer
      title={t("settings.postProcessing.prompts.selectedPrompt.title")}
      description={t(
        "settings.postProcessing.prompts.selectedPrompt.description",
      )}
      descriptionMode="tooltip"
      layout="stacked"
      grouped={true}
    >
      <div className="space-y-3">
        <div className="flex gap-2">
          <Dropdown
            selectedValue={selectedPromptId || null}
            options={sortedPrompts.map((p) => {
              const badge = isTranslatePresetPrompt(p)
                ? translatePresetBadge
                : isPresetPrompt(p)
                  ? presetBadge
                  : null;
              return {
                value: p.id,
                label: badge ? `${p.name} · ${badge}` : p.name,
              };
            })}
            onSelect={(value) => handlePromptSelect(value)}
            placeholder={
              prompts.length === 0
                ? t("settings.postProcessing.prompts.noPrompts")
                : t("settings.postProcessing.prompts.selectPrompt")
            }
            disabled={
              isUpdating("post_process_selected_prompt_id") || isCreating
            }
            className="flex-1"
          />
          <Button
            onClick={handleStartCreate}
            variant="primary"
            size="md"
            disabled={isCreating}
          >
            {t("settings.postProcessing.prompts.createNew")}
          </Button>
        </div>

        {!isCreating && hasPrompts && selectedPrompt && (
          <div className="space-y-3">
            <div className="space-y-2 flex flex-col">
              <label className="text-sm font-semibold">
                {t("settings.postProcessing.prompts.promptLabel")}
              </label>
              <Input
                type="text"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptLabelPlaceholder",
                )}
                variant="compact"
              />
            </div>

            <div className="space-y-2 flex flex-col">
              <label className="text-sm font-semibold">
                {t("settings.postProcessing.prompts.promptInstructions")}
              </label>
              <Textarea
                value={draftText}
                onChange={(e) => setDraftText(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptInstructionsPlaceholder",
                )}
              />
              <p className="text-xs text-mid-gray/70">
                <Trans
                  i18nKey="settings.postProcessing.prompts.promptTip"
                  components={{ code: <code /> }}
                />
              </p>
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                onClick={handleUpdatePrompt}
                variant="primary"
                size="md"
                disabled={!draftName.trim() || !draftText.trim() || !isDirty}
              >
                {t("settings.postProcessing.prompts.updatePrompt")}
              </Button>
              <Button
                onClick={() => handleDeletePrompt(selectedPromptId)}
                variant="secondary"
                size="md"
                disabled={!selectedPromptId || prompts.length <= 1}
              >
                {t("settings.postProcessing.prompts.deletePrompt")}
              </Button>
            </div>
          </div>
        )}

        {!isCreating && !selectedPrompt && (
          <div className="p-3 bg-mid-gray/5 rounded-md border border-mid-gray/20">
            <p className="text-sm text-mid-gray">
              {hasPrompts
                ? t("settings.postProcessing.prompts.selectToEdit")
                : t("settings.postProcessing.prompts.createFirst")}
            </p>
          </div>
        )}

        {isCreating && (
          <div className="space-y-3">
            <div className="space-y-2 block flex flex-col">
              <label className="text-sm font-semibold text-text">
                {t("settings.postProcessing.prompts.promptLabel")}
              </label>
              <Input
                type="text"
                value={draftName}
                onChange={(e) => setDraftName(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptLabelPlaceholder",
                )}
                variant="compact"
              />
            </div>

            <div className="space-y-2 flex flex-col">
              <label className="text-sm font-semibold">
                {t("settings.postProcessing.prompts.promptInstructions")}
              </label>
              <Textarea
                value={draftText}
                onChange={(e) => setDraftText(e.target.value)}
                placeholder={t(
                  "settings.postProcessing.prompts.promptInstructionsPlaceholder",
                )}
              />
              <p className="text-xs text-mid-gray/70">
                <Trans
                  i18nKey="settings.postProcessing.prompts.promptTip"
                  components={{ code: <code /> }}
                />
              </p>
            </div>

            <div className="flex gap-2 pt-2">
              <Button
                onClick={handleCreatePrompt}
                variant="primary"
                size="md"
                disabled={!draftName.trim() || !draftText.trim()}
              >
                {t("settings.postProcessing.prompts.createPrompt")}
              </Button>
              <Button
                onClick={handleCancelCreate}
                variant="secondary"
                size="md"
              >
                {t("settings.postProcessing.prompts.cancel")}
              </Button>
            </div>
          </div>
        )}
        <div className="border-t border-mid-gray/20 pt-3 space-y-2">
          <label className="text-sm font-semibold">
            {t("settings.postProcessing.api.systemPrompt.title")}
          </label>
          <p className="text-xs text-mid-gray">
            {t("settings.postProcessing.api.systemPrompt.description")}
          </p>
          <Textarea
            value={systemPromptDraft}
            onChange={(e) => setSystemPromptDraft(e.target.value)}
            onBlur={saveSystemPrompt}
            placeholder={t(
              "settings.postProcessing.api.systemPrompt.placeholder",
            )}
            className="w-full min-h-[160px]"
            disabled={isUpdating("post_process_system_prompt")}
          />
        </div>
      </div>
    </SettingContainer>
  );
};

type ScriptPathSettingKey = "post_asr_script_path" | "post_llm_script_path";
type ScriptStage = "asr_post" | "llm_post";

interface PostProcessingSettingsScriptHooksProps {
  scope?: "all" | "asr" | "llm";
  sectionId?: string;
}

const PostProcessingSettingsScriptHooksComponent: React.FC<
  PostProcessingSettingsScriptHooksProps
> = ({ scope = "all", sectionId }) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();
  const showAsrStage = scope === "all" || scope === "asr";
  const showLlmStage = scope === "all" || scope === "llm";
  const showGlobalSection = scope !== "llm";
  const showTemplateSection = scope !== "asr";
  const defaultSampleInput =
    scope === "asr"
      ? t("settings.postProcessing.api.scriptHooks.test.defaultInputAsr")
      : scope === "llm"
        ? t("settings.postProcessing.api.scriptHooks.test.defaultInputLlm")
        : t("settings.postProcessing.api.scriptHooks.test.defaultInput");

  const scriptHooksEnabled = getSetting("script_hooks_enabled") ?? false;
  const currentPostAsrScriptPath = (
    getSetting("post_asr_script_path") || ""
  ).toString();
  const currentPostLlmScriptPath = (
    getSetting("post_llm_script_path") || ""
  ).toString();
  const currentScriptHookTimeoutMs = Number(
    getSetting("script_hook_timeout_ms") || 1200,
  );

  const [postAsrScriptPathDraft, setPostAsrScriptPathDraft] = useState(
    currentPostAsrScriptPath,
  );
  const [postLlmScriptPathDraft, setPostLlmScriptPathDraft] = useState(
    currentPostLlmScriptPath,
  );
  const [scriptHookTimeoutDraft, setScriptHookTimeoutDraft] = useState(
    String(currentScriptHookTimeoutMs),
  );
  const [sampleInputDraft, setSampleInputDraft] = useState(defaultSampleInput);
  const [sampleOutput, setSampleOutput] = useState("");
  const [statusMessage, setStatusMessage] = useState<{
    variant: "error" | "success" | "info";
    text: string;
  } | null>(null);
  const [isRestoringTemplates, setIsRestoringTemplates] = useState(false);
  const [testingStage, setTestingStage] = useState<"asr_post" | "llm_post" | null>(
    null,
  );
  const [isImportingAsr, setIsImportingAsr] = useState(false);
  const [isExportingAsr, setIsExportingAsr] = useState(false);
  const [isImportingLlm, setIsImportingLlm] = useState(false);
  const [isExportingLlm, setIsExportingLlm] = useState(false);

  useEffect(() => {
    setPostAsrScriptPathDraft(currentPostAsrScriptPath);
  }, [currentPostAsrScriptPath]);

  useEffect(() => {
    setPostLlmScriptPathDraft(currentPostLlmScriptPath);
  }, [currentPostLlmScriptPath]);

  useEffect(() => {
    setScriptHookTimeoutDraft(String(currentScriptHookTimeoutMs));
  }, [currentScriptHookTimeoutMs]);

  const normalizeTimeout = (value: number): number =>
    Math.min(
      SCRIPT_HOOK_TIMEOUT_MAX,
      Math.max(SCRIPT_HOOK_TIMEOUT_MIN, value),
    );

  const commitScriptPath = (
    key: ScriptPathSettingKey,
    draftValue: string,
    currentValue: string,
  ) => {
    const normalizedDraft = draftValue.trim();
    const nextValue = normalizedDraft.length > 0 ? normalizedDraft : null;
    const normalizedCurrent = currentValue.trim();
    const current = normalizedCurrent.length > 0 ? normalizedCurrent : null;
    if (nextValue === current) return;
    void updateSetting(key, nextValue);
  };

  const commitScriptHookTimeout = () => {
    const parsed = Number.parseInt(scriptHookTimeoutDraft, 10);
    if (!Number.isFinite(parsed)) {
      setScriptHookTimeoutDraft(String(currentScriptHookTimeoutMs));
      return;
    }
    const normalized = normalizeTimeout(parsed);
    setScriptHookTimeoutDraft(String(normalized));
    if (normalized !== currentScriptHookTimeoutMs) {
      void updateSetting("script_hook_timeout_ms", normalized);
    }
  };

  const applyTemplatePaths = async (
    postAsrPath: string,
    postLlmPath: string,
  ) => {
    await updateSetting("post_asr_script_path", postAsrPath);
    await updateSetting("post_llm_script_path", postLlmPath);
    setPostAsrScriptPathDraft(postAsrPath);
    setPostLlmScriptPathDraft(postLlmPath);
  };

  const handleRestoreTemplates = async () => {
    setIsRestoringTemplates(true);
    setStatusMessage(null);
    try {
      const result = await commands.restoreDefaultScriptHookTemplates();
      if (result.status === "ok") {
        await applyTemplatePaths(
          result.data.post_asr_script_path,
          result.data.post_llm_script_path,
        );
        setStatusMessage({
          variant: "success",
          text: t("settings.postProcessing.api.scriptHooks.templates.restoreSuccess", {
            dir: result.data.directory,
          }),
        });
      } else {
        setStatusMessage({
          variant: "error",
          text: String(result.error),
        });
      }
    } catch (error) {
      setStatusMessage({
        variant: "error",
        text: String(error),
      });
    } finally {
      setIsRestoringTemplates(false);
    }
  };

  const applyImportedStagePath = async (stage: ScriptStage, path: string) => {
    if (stage === "asr_post") {
      await updateSetting("post_asr_script_path", path);
      setPostAsrScriptPathDraft(path);
      return;
    }
    await updateSetting("post_llm_script_path", path);
    setPostLlmScriptPathDraft(path);
  };

  const handleImportStageScript = async (stage: ScriptStage) => {
    if (stage === "asr_post") setIsImportingAsr(true);
    else setIsImportingLlm(true);
    setStatusMessage(null);

    try {
      const selected = await open({
        directory: false,
        multiple: false,
        filters: [
          {
            name: "Script",
            extensions: ["py", "js", "mjs", "cjs", "sh"],
          },
        ],
      });
      if (!selected || Array.isArray(selected)) return;

      const result = await commands.importScriptStageFromPath(stage, selected);
      if (result.status === "ok") {
        await applyImportedStagePath(stage, result.data);
        setStatusMessage({
          variant: "success",
          text: t("settings.postProcessing.api.scriptHooks.stageImportSuccess", {
            path: result.data,
          }),
        });
      } else {
        setStatusMessage({
          variant: "error",
          text: String(result.error),
        });
      }
    } catch (error) {
      setStatusMessage({
        variant: "error",
        text: String(error),
      });
    } finally {
      if (stage === "asr_post") setIsImportingAsr(false);
      else setIsImportingLlm(false);
    }
  };

  const handleExportStageScript = async (stage: ScriptStage) => {
    if (stage === "asr_post") setIsExportingAsr(true);
    else setIsExportingLlm(true);
    setStatusMessage(null);
    try {
      const result = await commands.exportScriptStageToDesktop(stage);
      if (result.status === "ok") {
        setStatusMessage({
          variant: "success",
          text: t("settings.postProcessing.api.scriptHooks.stageExportSuccess", {
            path: result.data,
          }),
        });
      } else {
        setStatusMessage({
          variant: "error",
          text: String(result.error),
        });
      }
    } catch (error) {
      setStatusMessage({
        variant: "error",
        text: String(error),
      });
    } finally {
      if (stage === "asr_post") setIsExportingAsr(false);
      else setIsExportingLlm(false);
    }
  };

  const handleTestStage = async (stage: "asr_post" | "llm_post") => {
    setTestingStage(stage);
    setStatusMessage(null);

    const stagePath =
      stage === "asr_post" ? postAsrScriptPathDraft.trim() : postLlmScriptPathDraft.trim();
    const stageLabelKey =
      stage === "asr_post"
        ? "settings.postProcessing.api.scriptHooks.test.asrButton"
        : "settings.postProcessing.api.scriptHooks.test.llmButton";

    try {
      const result = await commands.testScriptHook(
        stage,
        sampleInputDraft,
        stagePath.length > 0 ? stagePath : null,
      );
      if (result.status === "ok") {
        setSampleOutput(result.data.output_text);
        setStatusMessage({
          variant: "success",
          text: t("settings.postProcessing.api.scriptHooks.test.success", {
            stage: t(stageLabelKey),
            ms: result.data.duration_ms,
            path: result.data.used_script_path,
          }),
        });
      } else {
        setStatusMessage({
          variant: "error",
          text: String(result.error),
        });
      }
    } catch (error) {
      setStatusMessage({
        variant: "error",
        text: String(error),
      });
    } finally {
      setTestingStage(null);
    }
  };

  return (
    <div id={sectionId}>
      {showGlobalSection && (
        <ToggleSwitch
          checked={scriptHooksEnabled}
          onChange={(enabled) => {
            void updateSetting("script_hooks_enabled", enabled);
          }}
          isUpdating={isUpdating("script_hooks_enabled")}
          label={t("settings.postProcessing.api.scriptHooks.enabled.title")}
          description={t(
            "settings.postProcessing.api.scriptHooks.enabled.description",
          )}
          descriptionMode="tooltip"
          grouped={true}
        />
      )}

      {showAsrStage && (
        <SettingContainer
          title={t("settings.postProcessing.api.scriptHooks.postAsrPath.title")}
          description={t(
            "settings.postProcessing.api.scriptHooks.postAsrPath.description",
          )}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <div className="flex items-center gap-2">
            <Input
              type="text"
              value={postAsrScriptPathDraft}
              onChange={(event) => setPostAsrScriptPathDraft(event.target.value)}
              onBlur={() =>
                commitScriptPath(
                  "post_asr_script_path",
                  postAsrScriptPathDraft,
                  currentPostAsrScriptPath,
                )
              }
              placeholder={t(
                "settings.postProcessing.api.scriptHooks.postAsrPath.placeholder",
              )}
              disabled={isUpdating("post_asr_script_path")}
              className="min-w-[320px]"
            />
            <Button
              variant="secondary"
              size="sm"
              disabled={isImportingAsr || isExportingAsr}
              onClick={() => {
                void handleImportStageScript("asr_post");
              }}
            >
              {t("settings.postProcessing.api.scriptHooks.postAsrPath.importButton")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={isImportingAsr || isExportingAsr}
              onClick={() => {
                void handleExportStageScript("asr_post");
              }}
            >
              {t("settings.postProcessing.api.scriptHooks.postAsrPath.exportButton")}
            </Button>
          </div>
        </SettingContainer>
      )}

      {showLlmStage && (
        <SettingContainer
          title={t("settings.postProcessing.api.scriptHooks.postLlmPath.title")}
          description={t(
            "settings.postProcessing.api.scriptHooks.postLlmPath.description",
          )}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <div className="flex items-center gap-2">
            <Input
              type="text"
              value={postLlmScriptPathDraft}
              onChange={(event) => setPostLlmScriptPathDraft(event.target.value)}
              onBlur={() =>
                commitScriptPath(
                  "post_llm_script_path",
                  postLlmScriptPathDraft,
                  currentPostLlmScriptPath,
                )
              }
              placeholder={t(
                "settings.postProcessing.api.scriptHooks.postLlmPath.placeholder",
              )}
              disabled={isUpdating("post_llm_script_path")}
              className="min-w-[320px]"
            />
            <Button
              variant="secondary"
              size="sm"
              disabled={isImportingLlm || isExportingLlm}
              onClick={() => {
                void handleImportStageScript("llm_post");
              }}
            >
              {t("settings.postProcessing.api.scriptHooks.postLlmPath.importButton")}
            </Button>
            <Button
              variant="secondary"
              size="sm"
              disabled={isImportingLlm || isExportingLlm}
              onClick={() => {
                void handleExportStageScript("llm_post");
              }}
            >
              {t("settings.postProcessing.api.scriptHooks.postLlmPath.exportButton")}
            </Button>
          </div>
        </SettingContainer>
      )}

      {showGlobalSection && (
        <SettingContainer
          title={t("settings.postProcessing.api.scriptHooks.timeout.title")}
          description={t("settings.postProcessing.api.scriptHooks.timeout.description")}
          descriptionMode="tooltip"
          layout="horizontal"
          grouped={true}
        >
          <Input
            type="number"
            min={SCRIPT_HOOK_TIMEOUT_MIN}
            max={SCRIPT_HOOK_TIMEOUT_MAX}
            step={50}
            value={scriptHookTimeoutDraft}
            onChange={(event) => {
              const next = event.target.value;
              setScriptHookTimeoutDraft(next);
              const parsed = Number.parseInt(next, 10);
              if (!Number.isFinite(parsed)) return;
              const normalized = normalizeTimeout(parsed);
              if (normalized !== currentScriptHookTimeoutMs) {
                void updateSetting("script_hook_timeout_ms", normalized);
              }
            }}
            onBlur={commitScriptHookTimeout}
            disabled={isUpdating("script_hook_timeout_ms")}
            className="w-[140px]"
            variant="compact"
          />
        </SettingContainer>
      )}

      {showGlobalSection && (
        <SettingContainer
          title={t("settings.postProcessing.api.scriptHooks.protocol.title")}
          description={t("settings.postProcessing.api.scriptHooks.protocol.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="rounded-md border border-mid-gray/20 bg-mid-gray/5 p-3 text-xs leading-relaxed text-mid-gray whitespace-pre-wrap">
            {t("settings.postProcessing.api.scriptHooks.protocol.contract")}
          </div>
          <p className="mt-2 text-xs text-mid-gray/70">
            <Trans
              i18nKey="settings.postProcessing.api.scriptHooks.protocol.tip"
              components={{ code: <code /> }}
            />
          </p>
        </SettingContainer>
      )}

      {showTemplateSection && (
        <SettingContainer
          title={t("settings.postProcessing.api.scriptHooks.templates.title")}
          description={t("settings.postProcessing.api.scriptHooks.templates.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="flex flex-wrap gap-2">
            <Button
              onClick={handleRestoreTemplates}
              variant="secondary"
              size="md"
              disabled={isRestoringTemplates}
            >
              {t("settings.postProcessing.api.scriptHooks.templates.restoreButton")}
            </Button>
          </div>
          <p className="mt-2 text-xs text-mid-gray/70">
            {t("settings.postProcessing.api.scriptHooks.templates.hint")}
          </p>
        </SettingContainer>
      )}

      <SettingContainer
        title={t("settings.postProcessing.api.scriptHooks.test.title")}
        description={t("settings.postProcessing.api.scriptHooks.test.description")}
        descriptionMode="tooltip"
        layout="stacked"
        grouped={true}
      >
        <div className="space-y-3">
          <div className="space-y-1">
            <label className="text-sm font-semibold text-text">
              {t("settings.postProcessing.api.scriptHooks.test.inputLabel")}
            </label>
            <Textarea
              value={sampleInputDraft}
              onChange={(event) => setSampleInputDraft(event.target.value)}
              placeholder={t(
                "settings.postProcessing.api.scriptHooks.test.inputPlaceholder",
              )}
              className="w-full min-h-[110px]"
            />
          </div>

          <div className="flex flex-wrap gap-2">
            {showAsrStage && (
              <Button
                onClick={() => {
                  void handleTestStage("asr_post");
                }}
                variant="primary-soft"
                size="md"
                disabled={testingStage !== null}
              >
                {t("settings.postProcessing.api.scriptHooks.test.asrButton")}
              </Button>
            )}
            {showLlmStage && (
              <Button
                onClick={() => {
                  void handleTestStage("llm_post");
                }}
                variant="primary-soft"
                size="md"
                disabled={testingStage !== null}
              >
                {t("settings.postProcessing.api.scriptHooks.test.llmButton")}
              </Button>
            )}
          </div>

          {statusMessage && (
            <Alert variant={statusMessage.variant} contained>
              {statusMessage.text}
            </Alert>
          )}

          <div className="space-y-1">
            <label className="text-sm font-semibold text-text">
              {t("settings.postProcessing.api.scriptHooks.test.outputLabel")}
            </label>
            <Textarea
              value={sampleOutput}
              readOnly
              className="w-full min-h-[100px]"
            />
          </div>
        </div>
      </SettingContainer>
    </div>
  );
};

export const PostProcessingSettingsApi = React.memo(
  PostProcessingSettingsApiComponent,
);
PostProcessingSettingsApi.displayName = "PostProcessingSettingsApi";

export const PostProcessingSettingsPrompts = React.memo(
  PostProcessingSettingsPromptsComponent,
);
PostProcessingSettingsPrompts.displayName = "PostProcessingSettingsPrompts";

export const PostProcessingSettingsScriptHooks = React.memo(
  PostProcessingSettingsScriptHooksComponent,
);
PostProcessingSettingsScriptHooks.displayName =
  "PostProcessingSettingsScriptHooks";

export const PostProcessingSettings: React.FC = () => {
  const { t } = useTranslation();

  return (
    <div className="max-w-6xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.postProcessing.api.title")}>
        <PostProcessingSettingsApi />
      </SettingsGroup>

      <SettingsGroup title={t("settings.postProcessing.prompts.title")}>
        <PostProcessingSettingsPrompts />
      </SettingsGroup>
    </div>
  );
};
