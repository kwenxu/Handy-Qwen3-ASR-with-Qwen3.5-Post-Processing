import { type FC, useState } from "react";
import { useTranslation } from "react-i18next";
import { SettingContainer } from "../ui/SettingContainer";
import { Dropdown, type DropdownOption } from "../ui/Dropdown";
import { useSettings } from "../../hooks/useSettings";
import { commands } from "@/bindings";
import type {
  WhisperAcceleratorSetting,
  OrtAcceleratorSetting,
} from "@/bindings";

const ORT_LABELS: Record<OrtAcceleratorSetting, string> = {
  auto: "Auto",
  cpu: "CPU",
  cuda: "CUDA",
  directml: "DirectML",
  rocm: "ROCm",
};

interface AccelerationSelectorProps {
  descriptionMode?: "tooltip" | "inline";
  grouped?: boolean;
  showOrt?: boolean;
  whisperTitleOverride?: string;
}

/**
 * Whisper dropdown encodes accelerator + device in a single value:
 *   "auto"   → accelerator=auto,  gpu_device=-1
 *   "cpu"    → accelerator=cpu,   gpu_device=-1
 *   "gpu:0"  → accelerator=gpu,   gpu_device=0
 *   "gpu:1"  → accelerator=gpu,   gpu_device=1
 */
function encodeWhisperValue(
  accelerator: WhisperAcceleratorSetting,
  gpuDevice: number,
): string {
  if (accelerator === "cpu") return "cpu";
  if (accelerator === "gpu" && gpuDevice >= 0) return `gpu:${gpuDevice}`;
  return "auto";
}

function decodeWhisperValue(value: string): {
  accelerator: WhisperAcceleratorSetting;
  gpuDevice: number;
} {
  if (value === "cpu") return { accelerator: "cpu", gpuDevice: -1 };
  if (value.startsWith("gpu:")) {
    const id = parseInt(value.slice(4), 10);
    return { accelerator: "gpu", gpuDevice: id };
  }
  return { accelerator: "auto", gpuDevice: -1 };
}

export const AccelerationSelector: FC<AccelerationSelectorProps> = ({
  descriptionMode = "tooltip",
  grouped = false,
  showOrt = true,
  whisperTitleOverride,
}) => {
  const { t } = useTranslation();
  const { getSetting, updateSetting, isUpdating } = useSettings();

  const currentAccelerator = getSetting("whisper_accelerator") ?? "auto";
  const currentGpuDevice = getSetting("whisper_gpu_device") ?? -1;
  const currentWhisper = encodeWhisperValue(
    currentAccelerator as WhisperAcceleratorSetting,
    currentGpuDevice as number,
  );
  const currentOrt = getSetting("ort_accelerator") ?? "auto";

  const [whisperOptions, setWhisperOptions] = useState<DropdownOption[]>([]);
  const [ortOptions, setOrtOptions] = useState<DropdownOption[]>([]);
  const [isLoadingAccelerators, setIsLoadingAccelerators] = useState(false);
  const [hasLoadedAccelerators, setHasLoadedAccelerators] = useState(false);

  const loadAcceleratorOptions = async () => {
    if (isLoadingAccelerators) return;

    setIsLoadingAccelerators(true);
    try {
      const available = await commands.getAvailableAccelerators();

      const nextWhisperOptions: DropdownOption[] = [
        {
          value: "auto",
          label: t("settings.advanced.acceleration.gpuDevice.auto"),
        },
      ];

      for (const dev of available.gpu_devices) {
        const vramLabel =
          dev.total_vram_mb >= 1024
            ? `${(dev.total_vram_mb / 1024).toFixed(1)} GB`
            : `${dev.total_vram_mb} MB`;
        nextWhisperOptions.push({
          value: `gpu:${dev.id}`,
          label: `${dev.name} (${vramLabel})`,
        });
      }

      nextWhisperOptions.push({ value: "cpu", label: "CPU" });
      setWhisperOptions(nextWhisperOptions);

      const ortVals = available.ort.includes("auto")
        ? available.ort
        : ["auto", ...available.ort];
      setOrtOptions(
        ortVals.map((v) => ({
          value: v,
          label: ORT_LABELS[v as OrtAcceleratorSetting] ?? v,
        })),
      );
      setHasLoadedAccelerators(true);
    } catch (error) {
      console.error("Failed to load accelerators:", error);
    } finally {
      setIsLoadingAccelerators(false);
    }
  };

  const fallbackWhisperOptions: DropdownOption[] = [
    {
      value: "auto",
      label: t("settings.advanced.acceleration.gpuDevice.auto"),
    },
    ...(currentAccelerator === "gpu" && Number(currentGpuDevice) >= 0
      ? [
          {
            value: `gpu:${currentGpuDevice}`,
            label: `GPU ${currentGpuDevice}`,
          },
        ]
      : []),
    { value: "cpu", label: "CPU" },
  ];

  const fallbackOrtOptions: DropdownOption[] = Array.from(
    new Set(["auto", String(currentOrt)]),
  ).map((value) => ({
    value,
    label: ORT_LABELS[value as OrtAcceleratorSetting] ?? value,
  }));

  const effectiveWhisperOptions =
    whisperOptions.length > 0 ? whisperOptions : fallbackWhisperOptions;
  const effectiveOrtOptions =
    ortOptions.length > 0 ? ortOptions : fallbackOrtOptions;

  const handleWhisperChange = async (value: string) => {
    const { accelerator, gpuDevice } = decodeWhisperValue(value);
    await updateSetting("whisper_accelerator", accelerator);
    await updateSetting("whisper_gpu_device", gpuDevice);
  };

  return (
    <>
      <SettingContainer
        title={
          whisperTitleOverride ||
          t("settings.advanced.acceleration.whisper.title")
        }
        description={t("settings.advanced.acceleration.whisper.description")}
        descriptionMode={descriptionMode}
        grouped={grouped}
        layout="horizontal"
      >
        <Dropdown
          options={effectiveWhisperOptions}
          selectedValue={currentWhisper}
          onSelect={handleWhisperChange}
          onRefresh={loadAcceleratorOptions}
          disabled={
            isLoadingAccelerators ||
            isUpdating("whisper_accelerator") ||
            isUpdating("whisper_gpu_device")
          }
        />
      </SettingContainer>
      {showOrt &&
        (hasLoadedAccelerators
          ? ortOptions.length > 2
          : currentOrt !== "auto") && (
          <SettingContainer
            title={t("settings.advanced.acceleration.ort.title")}
            description={t("settings.advanced.acceleration.ort.description")}
            descriptionMode={descriptionMode}
            grouped={grouped}
            layout="horizontal"
          >
            <Dropdown
              options={effectiveOrtOptions}
              selectedValue={currentOrt}
              onSelect={(value) =>
                updateSetting("ort_accelerator", value as OrtAcceleratorSetting)
              }
              onRefresh={loadAcceleratorOptions}
              disabled={isLoadingAccelerators || isUpdating("ort_accelerator")}
            />
          </SettingContainer>
        )}
    </>
  );
};
