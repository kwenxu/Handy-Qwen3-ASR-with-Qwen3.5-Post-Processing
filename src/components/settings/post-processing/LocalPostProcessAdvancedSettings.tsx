import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { SettingContainer } from "@/components/ui";
import { Input } from "../../ui/Input";
import { useSettings } from "../../../hooks/useSettings";

interface LocalPostProcessAdvancedSettingsProps {
  grouped?: boolean;
}

export const LocalPostProcessAdvancedSettings: React.FC<
  LocalPostProcessAdvancedSettingsProps
> = ({ grouped = true }) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const currentLocalMaxTokens = Number(
    getSetting("post_process_local_max_tokens") || 192,
  );
  const currentLocalTemperature = Number(
    getSetting("post_process_local_temperature") || 0,
  );
  const currentLocalTopP = Number(getSetting("post_process_local_top_p") || 1);
  const currentLocalRepetitionPenalty = Number(
    getSetting("post_process_local_repetition_penalty") || 1.17,
  );
  const currentLocalRepetitionContextSize = Number(
    getSetting("post_process_local_repetition_context_size") || 128,
  );

  const [maxTokensDraft, setMaxTokensDraft] = useState(
    String(currentLocalMaxTokens),
  );
  const [temperatureDraft, setTemperatureDraft] = useState(
    String(currentLocalTemperature),
  );
  const [topPDraft, setTopPDraft] = useState(String(currentLocalTopP));
  const [repetitionPenaltyDraft, setRepetitionPenaltyDraft] = useState(
    String(currentLocalRepetitionPenalty),
  );
  const [repetitionContextSizeDraft, setRepetitionContextSizeDraft] = useState(
    String(currentLocalRepetitionContextSize),
  );

  useEffect(() => {
    setMaxTokensDraft(String(currentLocalMaxTokens));
  }, [currentLocalMaxTokens]);

  useEffect(() => {
    setTemperatureDraft(String(currentLocalTemperature));
  }, [currentLocalTemperature]);

  useEffect(() => {
    setTopPDraft(String(currentLocalTopP));
  }, [currentLocalTopP]);

  useEffect(() => {
    setRepetitionPenaltyDraft(String(currentLocalRepetitionPenalty));
  }, [currentLocalRepetitionPenalty]);

  useEffect(() => {
    setRepetitionContextSizeDraft(String(currentLocalRepetitionContextSize));
  }, [currentLocalRepetitionContextSize]);

  const clamp = (value: number, min: number, max: number): number =>
    Math.min(max, Math.max(min, value));

  const saveIntegerSetting = (
    draft: string,
    fallback: number,
    min: number,
    max: number,
    onDraftReset: (v: string) => void,
    onUpdate: (v: number) => void,
  ) => {
    const parsed = Number.parseInt(draft, 10);
    if (!Number.isFinite(parsed)) {
      onDraftReset(String(fallback));
      return;
    }
    const normalized = clamp(parsed, min, max);
    onDraftReset(String(normalized));
    if (normalized !== fallback) onUpdate(normalized);
  };

  const saveFloatSetting = (
    draft: string,
    fallback: number,
    min: number,
    max: number,
    digits: number,
    onDraftReset: (v: string) => void,
    onUpdate: (v: number) => void,
  ) => {
    const parsed = Number.parseFloat(draft);
    if (!Number.isFinite(parsed)) {
      onDraftReset(String(fallback));
      return;
    }
    const normalized = clamp(parsed, min, max);
    const rounded = Number(normalized.toFixed(digits));
    onDraftReset(String(rounded));
    if (rounded !== fallback) onUpdate(rounded);
  };

  const updateIntegerRealtime = (
    raw: string,
    current: number,
    min: number,
    max: number,
    updater: (value: number) => void,
  ) => {
    const parsed = Number.parseInt(raw, 10);
    if (!Number.isFinite(parsed)) return;
    const normalized = clamp(parsed, min, max);
    if (normalized !== current) updater(normalized);
  };

  const updateFloatRealtime = (
    raw: string,
    current: number,
    min: number,
    max: number,
    digits: number,
    updater: (value: number) => void,
  ) => {
    const parsed = Number.parseFloat(raw);
    if (!Number.isFinite(parsed)) return;
    const normalized = clamp(parsed, min, max);
    const rounded = Number(normalized.toFixed(digits));
    if (rounded !== current) updater(rounded);
  };

  return (
    <SettingContainer
      title={t("settings.postProcessing.api.advancedLocal.title")}
      description={t("settings.postProcessing.api.advancedLocal.description")}
      descriptionMode="tooltip"
      layout="stacked"
      grouped={grouped}
    >
      <p className="text-xs text-mid-gray">
        {t("settings.postProcessing.api.advancedLocal.singlePresetNotice")}
      </p>
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        <div className="space-y-1">
          <label className="text-sm font-semibold text-text">
            {t("settings.postProcessing.api.advancedLocal.maxTokens")}
          </label>
          <Input
            type="number"
            min={64}
            max={512}
            step={1}
            value={maxTokensDraft}
            onChange={(e) => {
              const next = e.target.value;
              setMaxTokensDraft(next);
              updateIntegerRealtime(
                next,
                currentLocalMaxTokens,
                64,
                512,
                (v) => void updateSetting("post_process_local_max_tokens", v),
              );
            }}
            onBlur={() =>
              saveIntegerSetting(
                maxTokensDraft,
                currentLocalMaxTokens,
                64,
                512,
                setMaxTokensDraft,
                (v) => void updateSetting("post_process_local_max_tokens", v),
              )
            }
            disabled={isUpdating("post_process_local_max_tokens")}
            variant="compact"
          />
          <p className="text-xs text-mid-gray/70">
            {t("settings.postProcessing.api.advancedLocal.maxTokensHint")}
          </p>
        </div>

        <div className="space-y-1">
          <label className="text-sm font-semibold text-text">
            {t("settings.postProcessing.api.advancedLocal.temperature")}
          </label>
          <Input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={temperatureDraft}
            onChange={(e) => {
              const next = e.target.value;
              setTemperatureDraft(next);
              updateFloatRealtime(
                next,
                currentLocalTemperature,
                0,
                1,
                2,
                (v) => void updateSetting("post_process_local_temperature", v),
              );
            }}
            onBlur={() =>
              saveFloatSetting(
                temperatureDraft,
                currentLocalTemperature,
                0,
                1,
                2,
                setTemperatureDraft,
                (v) => void updateSetting("post_process_local_temperature", v),
              )
            }
            disabled={isUpdating("post_process_local_temperature")}
            variant="compact"
          />
          <p className="text-xs text-mid-gray/70">
            {t("settings.postProcessing.api.advancedLocal.temperatureHint")}
          </p>
        </div>

        <div className="space-y-1">
          <label className="text-sm font-semibold text-text">
            {t("settings.postProcessing.api.advancedLocal.topP")}
          </label>
          <Input
            type="number"
            min={0.1}
            max={1}
            step={0.01}
            value={topPDraft}
            onChange={(e) => {
              const next = e.target.value;
              setTopPDraft(next);
              updateFloatRealtime(
                next,
                currentLocalTopP,
                0.1,
                1,
                2,
                (v) => void updateSetting("post_process_local_top_p", v),
              );
            }}
            onBlur={() =>
              saveFloatSetting(
                topPDraft,
                currentLocalTopP,
                0.1,
                1,
                2,
                setTopPDraft,
                (v) => void updateSetting("post_process_local_top_p", v),
              )
            }
            disabled={isUpdating("post_process_local_top_p")}
            variant="compact"
          />
          <p className="text-xs text-mid-gray/70">
            {t("settings.postProcessing.api.advancedLocal.topPHint")}
          </p>
        </div>

        <div className="space-y-1">
          <label className="text-sm font-semibold text-text">
            {t("settings.postProcessing.api.advancedLocal.repetitionPenalty")}
          </label>
          <Input
            type="number"
            min={1}
            max={1.5}
            step={0.01}
            value={repetitionPenaltyDraft}
            onChange={(e) => {
              const next = e.target.value;
              setRepetitionPenaltyDraft(next);
              updateFloatRealtime(
                next,
                currentLocalRepetitionPenalty,
                1,
                1.5,
                2,
                (v) =>
                  void updateSetting(
                    "post_process_local_repetition_penalty",
                    v,
                  ),
              );
            }}
            onBlur={() =>
              saveFloatSetting(
                repetitionPenaltyDraft,
                currentLocalRepetitionPenalty,
                1,
                1.5,
                2,
                setRepetitionPenaltyDraft,
                (v) =>
                  void updateSetting(
                    "post_process_local_repetition_penalty",
                    v,
                  ),
              )
            }
            disabled={isUpdating("post_process_local_repetition_penalty")}
            variant="compact"
          />
          <p className="text-xs text-mid-gray/70">
            {t(
              "settings.postProcessing.api.advancedLocal.repetitionPenaltyHint",
            )}
          </p>
        </div>

        <div className="space-y-1 md:col-span-2">
          <label className="text-sm font-semibold text-text">
            {t(
              "settings.postProcessing.api.advancedLocal.repetitionContextSize",
            )}
          </label>
          <Input
            type="number"
            min={32}
            max={256}
            step={1}
            value={repetitionContextSizeDraft}
            onChange={(e) => {
              const next = e.target.value;
              setRepetitionContextSizeDraft(next);
              updateIntegerRealtime(
                next,
                currentLocalRepetitionContextSize,
                32,
                256,
                (v) =>
                  void updateSetting(
                    "post_process_local_repetition_context_size",
                    v,
                  ),
              );
            }}
            onBlur={() =>
              saveIntegerSetting(
                repetitionContextSizeDraft,
                currentLocalRepetitionContextSize,
                32,
                256,
                setRepetitionContextSizeDraft,
                (v) =>
                  void updateSetting(
                    "post_process_local_repetition_context_size",
                    v,
                  ),
              )
            }
            disabled={isUpdating("post_process_local_repetition_context_size")}
            variant="compact"
          />
          <p className="text-xs text-mid-gray/70">
            {t(
              "settings.postProcessing.api.advancedLocal.repetitionContextSizeHint",
            )}
          </p>
        </div>
      </div>
    </SettingContainer>
  );
};
