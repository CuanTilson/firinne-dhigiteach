import React from "react";

interface Props {
  settings?: Record<string, unknown> | null;
  compact?: boolean;
}

const getThresholdValue = (
  settings: Record<string, unknown> | null | undefined,
  group: string,
  key?: string
) => {
  const thresholds =
    settings && typeof settings.thresholds === "object"
      ? (settings.thresholds as Record<string, unknown>)
      : null;
  const baseValue = thresholds?.[group];
  if (!key) return baseValue;
  const target =
    baseValue && typeof baseValue === "object"
      ? (baseValue as Record<string, unknown>)
      : null;
  return target?.[key];
};

const getPathValue = (settings: Record<string, unknown> | null | undefined, key: string) => {
  const paths =
    settings && typeof settings.paths === "object"
      ? (settings.paths as Record<string, unknown>)
      : null;
  return paths?.[key];
};

const formatValue = (value: unknown) => {
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" && value.trim()) return value;
  return "Unavailable";
};

export const AppliedSettingsPanel: React.FC<Props> = ({ settings, compact = false }) => {
  if (!settings) {
    return (
      <div className={compact ? "border border-slate-200 rounded-lg p-4" : "fd-card p-4"}>
        <div className={compact ? "text-xs uppercase tracking-wider text-slate-500 mb-2" : "fd-section-title mb-2"}>
          Applied Settings
        </div>
        <div className="text-sm text-slate-500">No settings snapshot was stored for this record.</div>
      </div>
    );
  }

  const pipeline =
    typeof settings.pipeline === "object"
      ? (settings.pipeline as Record<string, unknown>)
      : null;
  const toolchain =
    typeof settings.toolchain === "object"
      ? (settings.toolchain as Record<string, unknown>)
      : null;

  const containerClass = compact
    ? "border border-slate-200 rounded-lg p-4 space-y-4"
    : "fd-card p-4 space-y-4";
  const titleClass = compact
    ? "text-xs uppercase tracking-wider text-slate-500"
    : "fd-section-title";
  const valueClass = compact ? "text-sm text-slate-800" : "text-sm text-slate-200";
  const labelClass = compact
    ? "text-xs uppercase tracking-wider text-slate-500"
    : "text-xs uppercase tracking-wider text-slate-500";

  return (
    <div className={containerClass}>
      <div className={titleClass}>Applied Settings</div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        <Field
          label="Pipeline Version"
          value={formatValue(pipeline?.pipeline_version)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Image Detector"
          value={formatValue(pipeline?.image_detector)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Model Version"
          value={formatValue(pipeline?.model_version)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Dataset Version"
          value={formatValue(pipeline?.dataset_version)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Image AI Threshold"
          value={formatValue(
            getThresholdValue(settings, "classification_bands", "ai_likely_min")
          )}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Image Real Threshold"
          value={formatValue(
            getThresholdValue(settings, "classification_bands", "real_likely_max")
          )}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Audio AI Threshold"
          value={formatValue(
            getThresholdValue(settings, "audio_classification_bands", "ai_likely_min")
          )}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Audio Real Threshold"
          value={formatValue(
            getThresholdValue(settings, "audio_classification_bands", "real_likely_max")
          )}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Video Sample Frames"
          value={formatValue(getThresholdValue(settings, "video_sample_frames"))}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="FFmpeg Path"
          value={formatValue(getPathValue(settings, "ffmpeg_path"))}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="FFmpeg Available"
          value={formatValue(toolchain?.ffmpeg_available)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
      </div>
    </div>
  );
};

const Field = ({
  label,
  value,
  labelClass,
  valueClass,
}: {
  label: string;
  value: string;
  labelClass: string;
  valueClass: string;
}) => (
  <div>
    <div className={labelClass}>{label}</div>
    <div className={`${valueClass} mt-1 break-all`}>{value}</div>
  </div>
);
