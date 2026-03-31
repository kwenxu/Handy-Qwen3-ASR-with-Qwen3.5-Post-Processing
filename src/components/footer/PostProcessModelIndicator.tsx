import React from "react";
import { useTranslation } from "react-i18next";
import { useSettings } from "../../hooks/useSettings";

const LOCAL_PROVIDER_ID = "local-qwen35";

const compactModelName = (modelId: string): string => {
  if (!modelId) return modelId;
  const cleaned = modelId.trim();
  if (!cleaned) return cleaned;
  const lastPart = cleaned.split("/").pop();
  return lastPart && lastPart.length > 0 ? lastPart : cleaned;
};

const PostProcessModelIndicator: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting } = useSettings();

  const enabled = getSetting("post_process_enabled") ?? false;
  if (!enabled) return null;

  const providerId = (getSetting("post_process_provider_id") || "").toString();
  const providerList = (getSetting("post_process_providers") ||
    []) as Array<{ id: string; label: string }>;
  const modelsByProvider = (getSetting("post_process_models") ||
    {}) as Record<string, string>;

  const providerLabel =
    providerList.find((p) => p.id === providerId)?.label ||
    t("footer.postProcess.providerUnknown");

  const selectedModel = (modelsByProvider[providerId] || "").toString();
  const modelLabel = selectedModel
    ? compactModelName(selectedModel)
    : t("footer.postProcess.notConfigured");

  const summary =
    providerId === LOCAL_PROVIDER_ID
      ? t("footer.postProcess.localSummary", { model: modelLabel })
      : t("footer.postProcess.remoteSummary", {
          provider: providerLabel,
          model: modelLabel,
        });

  const statusClass = selectedModel
    ? "bg-green-400"
    : "bg-yellow-400 animate-pulse";

  return (
    <div
      className="flex items-center gap-2 text-text/70"
      title={t("footer.postProcess.tooltip", {
        provider: providerLabel,
        model: modelLabel,
      })}
    >
      <div className={`w-2 h-2 rounded-full ${statusClass}`} />
      <span className="max-w-44 truncate">{summary}</span>
    </div>
  );
};

export default PostProcessModelIndicator;
