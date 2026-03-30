import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Check, Download, Loader2, RefreshCcw, Trash2 } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import { useTranslation } from "react-i18next";
import { Button } from "../../ui/Button";
import { ResetButton } from "../../ui/ResetButton";
import { SettingContainer } from "../../ui";

type LocalPostProcessModelInfo = {
  id: string;
  name: string;
  description: string;
  repo_id: string;
  size_mb: number;
  tier: string;
  is_recommended: boolean;
  is_experimental: boolean;
  is_downloaded: boolean;
  is_downloading: boolean;
  partial_size: number;
};

interface LocalQwen35ModelsProps {
  selectedModelId: string;
  onSelectModel: (modelId: string) => void;
  title?: string;
  description?: string;
  grouped?: boolean;
}

export const LocalQwen35Models: React.FC<LocalQwen35ModelsProps> = ({
  selectedModelId,
  onSelectModel,
  title = "Local Post-Process Models",
  description = "Download and switch local Qwen3.5 models for offline post-processing.",
  grouped = true,
}) => {
  const { t } = useTranslation();
  const [models, setModels] = useState<LocalPostProcessModelInfo[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [busyModelIds, setBusyModelIds] = useState<Record<string, boolean>>({});

  const fetchModels = useCallback(async () => {
    try {
      const result = await invoke<LocalPostProcessModelInfo[]>(
        "get_local_post_process_models",
      );
      setModels(result);
    } catch (error) {
      console.error("Failed to fetch local post-process models:", error);
    }
  }, []);

  const refreshModels = useCallback(async () => {
    setIsLoadingModels(true);
    await fetchModels();
    setIsLoadingModels(false);
  }, [fetchModels]);

  useEffect(() => {
    void refreshModels();
  }, [refreshModels]);

  const hasDownloadingModels = useMemo(
    () => models.some((model) => model.is_downloading),
    [models],
  );

  useEffect(() => {
    if (!hasDownloadingModels) return;

    const timer = window.setInterval(() => {
      void fetchModels();
    }, 1500);

    return () => window.clearInterval(timer);
  }, [fetchModels, hasDownloadingModels]);

  const setBusy = (modelId: string, busy: boolean) => {
    setBusyModelIds((state) => ({ ...state, [modelId]: busy }));
  };

  const handleDownload = async (modelId: string) => {
    if (busyModelIds[modelId]) return;

    setBusy(modelId, true);
    try {
      await invoke("download_local_post_process_model", { modelId });
    } catch (error) {
      console.error(`Failed to download local model ${modelId}:`, error);
    } finally {
      setBusy(modelId, false);
      await fetchModels();
    }
  };

  const handleDelete = async (modelId: string) => {
    setBusy(modelId, true);
    try {
      await invoke("delete_local_post_process_model", { modelId });
    } catch (error) {
      console.error(`Failed to delete local model ${modelId}:`, error);
    } finally {
      setBusy(modelId, false);
      await fetchModels();
    }
  };

  const handleCancel = async (modelId: string) => {
    if (busyModelIds[modelId]) return;

    setBusy(modelId, true);
    try {
      await invoke("cancel_local_post_process_model_download", { modelId });
    } catch (error) {
      console.error(`Failed to cancel local model download ${modelId}:`, error);
    } finally {
      setBusy(modelId, false);
      await fetchModels();
    }
  };

  const sortedModels = useMemo(() => {
    const preferredOrder: Record<string, number> = {
      "qwen35-optiq-0.8b": 0,
      "qwen35-optiq-2b": 1,
      "qwen35-optiq-4b": 2,
      "qwen35-optiq-9b": 3,
    };
    return [...models].sort((a, b) => {
      const aOrder = preferredOrder[a.id] ?? Number.MAX_SAFE_INTEGER;
      const bOrder = preferredOrder[b.id] ?? Number.MAX_SAFE_INTEGER;
      if (aOrder !== bOrder) {
        return aOrder - bOrder;
      }
      if (a.is_recommended !== b.is_recommended) {
        return a.is_recommended ? -1 : 1;
      }
      if (a.is_experimental !== b.is_experimental) {
        return a.is_experimental ? 1 : -1;
      }
      return a.name.localeCompare(b.name);
    });
  }, [models]);

  return (
    <SettingContainer
      title={title}
      description={description}
      descriptionMode="tooltip"
      layout="stacked"
      grouped={grouped}
      headerAction={
        <ResetButton
          onClick={refreshModels}
          disabled={isLoadingModels}
          ariaLabel={t("settings.postProcessing.api.model.refreshModels")}
          className="flex h-10 w-10 items-center justify-center"
        >
          <RefreshCcw
            className={`h-4 w-4 ${isLoadingModels ? "animate-spin" : ""}`}
          />
        </ResetButton>
      }
    >
      <div className="space-y-3">
        {sortedModels.map((model) => {
          const isBusy = !!busyModelIds[model.id];
          const isSelected = selectedModelId === model.id;
          const sizeLabel = `${model.size_mb} MB`;
          const totalBytes = model.size_mb * 1024 * 1024;
          const downloadedBytes = Math.max(0, model.partial_size || 0);
          const progressRatio =
            totalBytes > 0 ? Math.min(downloadedBytes / totalBytes, 1) : 0;
          const progressPercentage = Math.floor(progressRatio * 100);
          const downloadedLabel =
            totalBytes > 0
              ? `${Math.floor(downloadedBytes / (1024 * 1024))}/${model.size_mb} MB`
              : `${Math.floor(downloadedBytes / (1024 * 1024))} MB`;

          return (
            <div
              key={model.id}
              className="p-3 rounded-md border border-mid-gray/20 bg-mid-gray/5"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-semibold">{model.name}</p>
                    {model.is_recommended && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-primary/15 text-primary">
                        {t("onboarding.recommended")}
                      </span>
                    )}
                    {model.is_experimental && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-yellow-500/15 text-yellow-700">
                        {t("settings.advanced.groups.experimental")}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-mid-gray">
                    {model.description} · {model.tier} · {sizeLabel}
                  </p>
                  {model.is_downloading && (
                    <div className="space-y-1">
                      <p className="text-[11px] text-mid-gray">
                        {downloadedLabel} · {progressPercentage}%
                      </p>
                      <div className="h-1.5 w-56 rounded bg-mid-gray/20 overflow-hidden">
                        <div
                          className="h-full rounded bg-primary transition-all duration-200"
                          style={{ width: `${progressPercentage}%` }}
                        />
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center gap-2">
                  {model.is_downloading ? (
                    <>
                      <Button
                        onClick={() => handleCancel(model.id)}
                        variant="secondary"
                        size="md"
                        disabled={isBusy}
                      >
                        {t("modelSelector.cancel")}
                      </Button>
                      <span className="text-xs text-mid-gray">
                        {t("onboarding.downloading")}
                      </span>
                    </>
                  ) : model.is_downloaded ? (
                    <>
                      <Button
                        onClick={() => onSelectModel(model.id)}
                        variant={isSelected ? "primary" : "secondary"}
                        size="md"
                        disabled={isBusy}
                      >
                        {isSelected ? (
                          <span className="inline-flex items-center gap-1">
                            <Check className="h-4 w-4" />
                            {t("modelSelector.active")}
                          </span>
                        ) : (
                          t("common.open")
                        )}
                      </Button>
                      <Button
                        onClick={() => handleDelete(model.id)}
                        variant="secondary"
                        size="md"
                        disabled={isBusy}
                      >
                        <span className="inline-flex items-center gap-1">
                          <Trash2 className="h-4 w-4" />
                          {t("common.delete")}
                        </span>
                      </Button>
                    </>
                  ) : (
                    <Button
                      onClick={() => handleDownload(model.id)}
                      variant="primary"
                      size="md"
                      disabled={isBusy}
                    >
                      <span className="inline-flex items-center gap-1">
                        {isBusy ? (
                          <Loader2 className="h-4 w-4 animate-spin" />
                        ) : (
                          <Download className="h-4 w-4" />
                        )}
                        {t("onboarding.download")}
                      </span>
                    </Button>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </SettingContainer>
  );
};
