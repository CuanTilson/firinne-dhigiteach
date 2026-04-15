import React from "react";

interface Props {
  settings?: Record<string, unknown> | null;
  compact?: boolean;
}

const getThresholdValue = (
  settings: Record<string, unknown> | null | undefined,
  group: string,
  key?: string,
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

const getPathValue = (
  settings: Record<string, unknown> | null | undefined,
  key: string,
) => {
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

const Field = ({
  label,
  value,
  mono = false,
  labelClass,
  valueClass,
}: {
  label: string;
  value: string;
  mono?: boolean;
  labelClass: string;
  valueClass: string;
}) => (
  <div>
    <div className={labelClass}>{label}</div>
    <div
      className={`${valueClass} mt-1 break-all ${
        mono ? "font-mono text-[13px]" : ""
      }`}
    >
      {value}
    </div>
  </div>
);

export const AppliedSettingsPanel: React.FC<Props> = ({
  settings,
  compact = false,
}) => {
  const containerClass = compact
    ? "rounded-2xl border border-slate-200 p-4"
    : "rounded-3xl border border-slate-800/80 bg-slate-950/55 p-5 shadow-[0_10px_30px_rgba(2,6,23,0.28)]";

  const titleClass = compact
    ? "mb-3 text-xs uppercase tracking-[0.18em] text-slate-500"
    : "mb-4 text-[11px] uppercase tracking-[0.2em] text-slate-500";

  const valueClass = compact
    ? "text-sm text-slate-800"
    : "text-sm text-slate-200";
  const labelClass = "text-xs uppercase tracking-[0.16em] text-slate-500";

  if (!settings) {
    return (
      <div className={containerClass}>
        <div className={titleClass}>Applied Settings</div>
        <div className="text-sm text-slate-500">
          No settings snapshot was stored for this record.
        </div>
      </div>
    );
  }

  const pipeline =
    typeof settings.pipeline === "object"
      ? (settings.pipeline as Record<string, unknown>)
      : null;

  return (
    <div className={containerClass}>
      <div className={titleClass}>Applied Settings</div>

      <div className="space-y-5">
        <div>
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Pipeline
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              label="Pipeline Version"
              value={formatValue(pipeline?.pipeline_version)}
              mono
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
              mono
              labelClass={labelClass}
              valueClass={valueClass}
            />
            <Field
              label="Dataset Version"
              value={formatValue(pipeline?.dataset_version)}
              mono
              labelClass={labelClass}
              valueClass={valueClass}
            />
          </div>
        </div>

        <div>
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Thresholds
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              label="Image AI Threshold"
              value={formatValue(
                getThresholdValue(
                  settings,
                  "classification_bands",
                  "ai_likely_min",
                ),
              )}
              labelClass={labelClass}
              valueClass={valueClass}
            />
            <Field
              label="Image Real Threshold"
              value={formatValue(
                getThresholdValue(
                  settings,
                  "classification_bands",
                  "real_likely_max",
                ),
              )}
              labelClass={labelClass}
              valueClass={valueClass}
            />
            <Field
              label="Audio AI Threshold"
              value={formatValue(
                getThresholdValue(
                  settings,
                  "audio_classification_bands",
                  "ai_likely_min",
                ),
              )}
              labelClass={labelClass}
              valueClass={valueClass}
            />
            <Field
              label="Audio Real Threshold"
              value={formatValue(
                getThresholdValue(
                  settings,
                  "audio_classification_bands",
                  "real_likely_max",
                ),
              )}
              labelClass={labelClass}
              valueClass={valueClass}
            />
            <Field
              label="Video Sample Frames"
              value={formatValue(
                getThresholdValue(settings, "video_sample_frames"),
              )}
              labelClass={labelClass}
              valueClass={valueClass}
            />
          </div>
        </div>

        <div>
          <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
            Runtime
          </div>
          <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <Field
              label="FFmpeg Path"
              value={formatValue(getPathValue(settings, "ffmpeg_path"))}
              mono
              labelClass={labelClass}
              valueClass={valueClass}
            />
          </div>
        </div>
      </div>
    </div>
  );
};
