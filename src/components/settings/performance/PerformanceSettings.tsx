import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import type { AppSettings as Settings } from "@/bindings";
import {
  Dropdown,
  SettingContainer,
  SettingsGroup,
  ToggleSwitch,
} from "@/components/ui";
import { Input } from "../../ui/Input";
import { useSettings } from "../../../hooks/useSettings";
import { AccelerationSelector } from "../AccelerationSelector";

type NumericSettingKey =
  | "qwen3_startup_preload_delay_ms"
  | "qwen35_startup_preload_delay_ms"
  | "qwen3_max_threads"
  | "qwen35_max_threads"
  | "qwen3_server_ready_timeout_sec"
  | "qwen35_server_ready_timeout_sec"
  | "qwen35_inference_timeout_sec";

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

export const PerformanceSettings: React.FC = () => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const preloadStrategy =
    (getSetting("qwen_startup_preload_strategy") || "parallel").toString() ||
    "parallel";
  const qwen3StartupPreloadEnabled =
    getSetting("qwen3_startup_preload_enabled") ?? true;
  const qwen35StartupPreloadEnabled =
    getSetting("qwen35_startup_preload_enabled") ?? true;
  const qwen35WarmupEnabled = getSetting("qwen35_warmup_enabled") ?? true;

  return (
    <div className="max-w-5xl w-full mx-auto space-y-6">
      <SettingsGroup title={t("settings.performance.groups.transcription")}>
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

        <SettingContainer
          title={t("settings.performance.qwenPanel.startupTiming.title")}
          description={t(
            "settings.performance.qwenPanel.startupTiming.description",
          )}
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
              defaultValue={900}
              disabled={!qwen3StartupPreloadEnabled}
            />
            <NumericSettingField
              settingKey="qwen3_server_ready_timeout_sec"
              label={t(
                "settings.performance.qwenPanel.timeouts.qwen3Ready.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.timeouts.qwen3Ready.hint",
              )}
              min={10}
              max={120}
              defaultValue={30}
            />
          </div>
        </SettingContainer>

        <SettingContainer
          title={t("settings.performance.qwenPanel.runtime.title")}
          description={t("settings.performance.qwenPanel.runtime.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <NumericSettingField
              settingKey="qwen3_max_threads"
              label={t(
                "settings.performance.qwenPanel.runtime.qwen3Threads.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.runtime.qwen3Threads.hint",
              )}
              min={0}
              max={16}
              defaultValue={0}
            />
          </div>
        </SettingContainer>

        <AccelerationSelector
          descriptionMode="tooltip"
          grouped={true}
          showOrt={false}
          whisperTitleOverride={t("settings.performance.whisperExperimental")}
        />
      </SettingsGroup>

      <SettingsGroup title={t("settings.performance.groups.postProcessing")}>
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
                label: t(
                  "settings.performance.qwenPanel.strategy.options.serial",
                ),
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
          description={t(
            "settings.performance.qwenPanel.qwen35Warmup.description",
          )}
          descriptionMode="tooltip"
          grouped={true}
        />

        <SettingContainer
          title={t("settings.performance.qwenPanel.startupTiming.title")}
          description={t(
            "settings.performance.qwenPanel.startupTiming.description",
          )}
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
              defaultValue={1400}
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
          title={t("settings.performance.qwenPanel.runtime.title")}
          description={t("settings.performance.qwenPanel.runtime.description")}
          descriptionMode="tooltip"
          layout="stacked"
          grouped={true}
        >
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <NumericSettingField
              settingKey="qwen35_max_threads"
              label={t(
                "settings.performance.qwenPanel.runtime.qwen35Threads.label",
              )}
              hint={t(
                "settings.performance.qwenPanel.runtime.qwen35Threads.hint",
              )}
              min={0}
              max={16}
              defaultValue={0}
            />
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
      </SettingsGroup>
    </div>
  );
};
