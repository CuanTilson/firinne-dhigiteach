import React, { useEffect, useState } from "react";
import { getSettings, updateSettings } from "../services/api";
import type { SettingsSnapshot, SettingsUpdatePayload } from "../types";
import { DEFAULT_ADMIN_KEY } from "../constants";
import {
  AlertCircle,
  CheckCircle2,
  RotateCcw,
  Save,
  Settings2,
} from "lucide-react";

const buildFormFromSettings = (
  data: SettingsSnapshot,
): SettingsUpdatePayload => ({
  pipeline: {
    image_detector: data.pipeline.image_detector,
  },
  thresholds: {
    classification_bands: { ...data.thresholds.classification_bands },
    audio_classification_bands: {
      ...data.thresholds.audio_classification_bands,
    },
    video_max_duration_seconds: data.thresholds.video_max_duration_seconds,
    video_sample_frames: data.thresholds.video_sample_frames,
  },
  paths: {
    ffmpeg_path: data.paths.ffmpeg_path || "",
  },
});

const normaliseDisplayValue = (value: unknown) =>
  String(value === "" || value == null ? "Auto-discovery" : value);

const isDifferent = (currentValue: unknown, baselineValue: unknown) =>
  String(currentValue ?? "") !== String(baselineValue ?? "");

const KV = ({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) => (
  <div>
    <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <div
      className={`mt-1 text-sm text-slate-200 ${
        mono ? "break-all font-mono text-[13px]" : "break-words"
      }`}
    >
      {value}
    </div>
  </div>
);

const SectionCard = ({
  title,
  children,
  right,
}: {
  title: string;
  children: React.ReactNode;
  right?: React.ReactNode;
}) => (
  <section className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-5 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
    <div className="mb-4 flex items-center justify-between gap-3">
      <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
        {title}
      </div>
      {right}
    </div>
    {children}
  </section>
);

const FieldGroup = ({
  title,
  description,
  children,
}: {
  title: string;
  description?: string;
  children: React.ReactNode;
}) => (
  <div className="rounded-2xl border border-slate-800/80 bg-slate-950/70 p-4">
    <div className="mb-4">
      <div className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
        {title}
      </div>
      {description ? (
        <p className="mt-1 text-sm text-slate-500">{description}</p>
      ) : null}
    </div>
    {children}
  </div>
);

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SettingsSnapshot | null>(null);
  const [form, setForm] = useState<SettingsUpdatePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await getSettings();
        setSettings(data);
        setForm(buildFormFromSettings(data));
      } catch {
        setPageError("Could not load settings.");
      } finally {
        setLoading(false);
      }
    };

    fetchSettings();
  }, []);

  const updateBandField = (
    section: "classification_bands" | "audio_classification_bands",
    field: "ai_likely_min" | "real_likely_max",
    value: number,
  ) => {
    setForm((current) =>
      current
        ? {
            ...current,
            thresholds: {
              ...current.thresholds,
              [section]: {
                ...current.thresholds[section],
                [field]: value,
              },
            },
          }
        : current,
    );
  };

  const resetForm = () => {
    if (!settings) return;
    setForm(buildFormFromSettings(settings));
    setSaveError(null);
    setSaveMessage(null);
  };

  const save = async () => {
    if (!form || !DEFAULT_ADMIN_KEY) return;

    setSaving(true);
    setSaveError(null);
    setSaveMessage(null);

    try {
      const next = await updateSettings(form, DEFAULT_ADMIN_KEY);
      setSettings(next);
      setForm(buildFormFromSettings(next));
      setSaveMessage(
        "Settings updated. New analyses will store this snapshot.",
      );
    } catch {
      setSaveError("Could not update settings.");
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-slate-500">
        Loading settings...
      </div>
    );
  }

  if (pageError || !settings || !form) {
    return (
      <div className="flex min-h-screen items-center justify-center text-red-400">
        {pageError || "Settings unavailable."}
      </div>
    );
  }

  const baselineForm = buildFormFromSettings(settings);
  const isDirty = JSON.stringify(form) !== JSON.stringify(baselineForm);

  const currentImageDetector = form.pipeline.image_detector;
  const currentImageAiLikelyMin =
    form.thresholds.classification_bands.ai_likely_min;
  const currentImageRealLikelyMax =
    form.thresholds.classification_bands.real_likely_max;
  const currentAudioAiLikelyMin =
    form.thresholds.audio_classification_bands.ai_likely_min;
  const currentAudioRealLikelyMax =
    form.thresholds.audio_classification_bands.real_likely_max;
  const currentVideoMaxDuration = form.thresholds.video_max_duration_seconds;
  const currentVideoSampleFrames = form.thresholds.video_sample_frames;
  const currentFfmpegPath = form.paths.ffmpeg_path;

  const modifiedCount = [
    isDifferent(currentImageDetector, settings.pipeline.image_detector),
    isDifferent(
      currentImageAiLikelyMin,
      settings.thresholds.classification_bands.ai_likely_min,
    ),
    isDifferent(
      currentImageRealLikelyMax,
      settings.thresholds.classification_bands.real_likely_max,
    ),
    isDifferent(
      currentAudioAiLikelyMin,
      settings.thresholds.audio_classification_bands.ai_likely_min,
    ),
    isDifferent(
      currentAudioRealLikelyMax,
      settings.thresholds.audio_classification_bands.real_likely_max,
    ),
    isDifferent(
      currentVideoMaxDuration,
      settings.thresholds.video_max_duration_seconds,
    ),
    isDifferent(
      currentVideoSampleFrames,
      settings.thresholds.video_sample_frames,
    ),
    isDifferent(currentFfmpegPath, settings.paths.ffmpeg_path),
  ].filter(Boolean).length;

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-slate-800/80 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.10),_transparent_35%),linear-gradient(180deg,rgba(2,6,23,0.96),rgba(2,6,23,0.82))] p-6 shadow-[0_0_0_1px_rgba(15,23,42,0.5)]">
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:32px_32px] opacity-30" />

          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-2 text-[11px] uppercase tracking-[0.28em] text-cyan-300/80">
                Governance
              </div>
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-300">
                  <Settings2 size={20} />
                </div>
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-50">
                    Settings &amp; Model Registry
                  </h1>
                  <p className="mt-1 text-sm text-slate-400">
                    Snapshot of configuration, thresholds, limits, and model
                    metadata.
                  </p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
              <SummaryStat label="Pending changes" value={modifiedCount} />
              <SummaryStat
                label="Detector"
                value={currentImageDetector}
                mono={false}
              />
              <SummaryStat
                label="FFmpeg path"
                value={currentFfmpegPath || "Auto"}
                mono
              />
            </div>
          </div>
        </section>

        <div className="rounded-2xl border border-slate-800/80 bg-slate-950/55 px-4 py-4 text-sm text-slate-300">
          Each completed analysis stores the exact applied settings snapshot.
          Changes here affect future analyses only; existing records retain
          their original configuration.
        </div>

        <SectionCard title="Current Effective Configuration">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
            <OverrideBadge
              label="Image Detector"
              changed={isDifferent(
                currentImageDetector,
                settings.pipeline.image_detector,
              )}
              value={currentImageDetector}
              baselineValue={settings.pipeline.image_detector}
            />
            <OverrideBadge
              label="Image AI Likely Min"
              changed={isDifferent(
                currentImageAiLikelyMin,
                settings.thresholds.classification_bands.ai_likely_min,
              )}
              value={currentImageAiLikelyMin}
              baselineValue={
                settings.thresholds.classification_bands.ai_likely_min
              }
            />
            <OverrideBadge
              label="Image Real Likely Max"
              changed={isDifferent(
                currentImageRealLikelyMax,
                settings.thresholds.classification_bands.real_likely_max,
              )}
              value={currentImageRealLikelyMax}
              baselineValue={
                settings.thresholds.classification_bands.real_likely_max
              }
            />
            <OverrideBadge
              label="Audio AI Likely Min"
              changed={isDifferent(
                currentAudioAiLikelyMin,
                settings.thresholds.audio_classification_bands.ai_likely_min,
              )}
              value={currentAudioAiLikelyMin}
              baselineValue={
                settings.thresholds.audio_classification_bands.ai_likely_min
              }
            />
            <OverrideBadge
              label="Audio Real Likely Max"
              changed={isDifferent(
                currentAudioRealLikelyMax,
                settings.thresholds.audio_classification_bands.real_likely_max,
              )}
              value={currentAudioRealLikelyMax}
              baselineValue={
                settings.thresholds.audio_classification_bands.real_likely_max
              }
            />
            <OverrideBadge
              label="Video Max Duration"
              changed={isDifferent(
                currentVideoMaxDuration,
                settings.thresholds.video_max_duration_seconds,
              )}
              value={currentVideoMaxDuration}
              baselineValue={settings.thresholds.video_max_duration_seconds}
            />
            <OverrideBadge
              label="Video Sample Frames"
              changed={isDifferent(
                currentVideoSampleFrames,
                settings.thresholds.video_sample_frames,
              )}
              value={currentVideoSampleFrames}
              baselineValue={settings.thresholds.video_sample_frames}
            />
            <OverrideBadge
              label="FFmpeg Path"
              changed={isDifferent(
                currentFfmpegPath,
                settings.paths.ffmpeg_path,
              )}
              value={currentFfmpegPath}
              baselineValue={settings.paths.ffmpeg_path}
            />
          </div>
        </SectionCard>

        <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <SectionCard
            title="Editable Analysis Settings"
            right={
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={resetForm}
                  disabled={!isDirty || saving}
                  className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/60 px-3 py-2 text-sm text-slate-200 transition hover:border-slate-600 hover:bg-slate-900 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <RotateCcw size={15} />
                  Reset
                </button>
                <button
                  type="button"
                  onClick={save}
                  disabled={saving || !DEFAULT_ADMIN_KEY || !isDirty}
                  className="inline-flex items-center gap-2 rounded-xl border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-sm font-medium text-cyan-100 transition hover:border-cyan-400/40 hover:bg-cyan-500/15 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <Save size={15} />
                  {saving ? "Saving..." : "Save Settings"}
                </button>
              </div>
            }
          >
            <div className="space-y-4">
              <div className="rounded-2xl border border-slate-800/80 bg-slate-900/40 px-4 py-3">
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  {isDirty ? (
                    <span className="inline-flex items-center gap-2 text-amber-300">
                      <AlertCircle size={16} />
                      You have unsaved changes.
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-2 text-emerald-300">
                      <CheckCircle2 size={16} />
                      No pending changes.
                    </span>
                  )}

                  {saveMessage ? (
                    <span className="text-emerald-300" aria-live="polite">
                      {saveMessage}
                    </span>
                  ) : null}

                  {saveError ? (
                    <span className="text-red-400" aria-live="assertive">
                      {saveError}
                    </span>
                  ) : null}
                </div>

                {!DEFAULT_ADMIN_KEY ? (
                  <div className="mt-2 text-sm text-amber-300">
                    `VITE_ADMIN_KEY` is not configured, so settings updates are
                    disabled in this UI.
                  </div>
                ) : null}
              </div>

              <FieldGroup
                title="Detector"
                description="Select the image detector used for future analyses."
              >
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <SelectField
                    label="Image Detector"
                    value={currentImageDetector}
                    onChange={(value) =>
                      setForm((current) =>
                        current
                          ? {
                              ...current,
                              pipeline: {
                                ...current.pipeline,
                                image_detector: value,
                              },
                            }
                          : current,
                      )
                    }
                    options={Object.entries(
                      settings.pipeline.available_image_detectors || {},
                    ).map(([value, item]) => ({
                      value,
                      label: `${item.display_name}${
                        item.available ? "" : " (Unavailable)"
                      }`,
                      disabled: !item.available,
                    }))}
                  />
                </div>
              </FieldGroup>

              <FieldGroup
                title="Image Thresholds"
                description="Classification thresholds for image analysis."
              >
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <NumberField
                    label="AI Likely Min"
                    value={currentImageAiLikelyMin}
                    onChange={(value) =>
                      updateBandField(
                        "classification_bands",
                        "ai_likely_min",
                        value,
                      )
                    }
                    step="0.01"
                    min={0}
                    max={1}
                  />
                  <NumberField
                    label="Real Likely Max"
                    value={currentImageRealLikelyMax}
                    onChange={(value) =>
                      updateBandField(
                        "classification_bands",
                        "real_likely_max",
                        value,
                      )
                    }
                    step="0.01"
                    min={0}
                    max={1}
                  />
                </div>
              </FieldGroup>

              <FieldGroup
                title="Audio Thresholds"
                description="Classification thresholds for audio analysis."
              >
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <NumberField
                    label="AI Likely Min"
                    value={currentAudioAiLikelyMin}
                    onChange={(value) =>
                      updateBandField(
                        "audio_classification_bands",
                        "ai_likely_min",
                        value,
                      )
                    }
                    step="0.01"
                    min={0}
                    max={1}
                  />
                  <NumberField
                    label="Real Likely Max"
                    value={currentAudioRealLikelyMax}
                    onChange={(value) =>
                      updateBandField(
                        "audio_classification_bands",
                        "real_likely_max",
                        value,
                      )
                    }
                    step="0.01"
                    min={0}
                    max={1}
                  />
                </div>
              </FieldGroup>

              <FieldGroup
                title="Video Limits"
                description="Controls for duration and frame sampling."
              >
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <NumberField
                    label="Max Duration (seconds)"
                    value={currentVideoMaxDuration}
                    onChange={(value) =>
                      setForm((current) =>
                        current
                          ? {
                              ...current,
                              thresholds: {
                                ...current.thresholds,
                                video_max_duration_seconds: value,
                              },
                            }
                          : current,
                      )
                    }
                    min={1}
                  />
                  <NumberField
                    label="Sample Frames"
                    value={currentVideoSampleFrames}
                    onChange={(value) =>
                      setForm((current) =>
                        current
                          ? {
                              ...current,
                              thresholds: {
                                ...current.thresholds,
                                video_sample_frames: value,
                              },
                            }
                          : current,
                      )
                    }
                    min={1}
                  />
                </div>
              </FieldGroup>

              <FieldGroup
                title="Paths"
                description="Optional runtime path override."
              >
                <TextField
                  label="FFmpeg Path Override"
                  value={currentFfmpegPath}
                  onChange={(value) =>
                    setForm((current) =>
                      current
                        ? {
                            ...current,
                            paths: {
                              ...current.paths,
                              ffmpeg_path: value,
                            },
                          }
                        : current,
                    )
                  }
                  placeholder="Leave blank to use PATH or auto-discovery"
                  mono
                />
              </FieldGroup>

              <FieldGroup
                title="Fusion Weights"
                description="Current weighting values used by the fusion stage."
              >
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {Object.entries(settings.thresholds.fusion_weights).map(
                    ([key, value]) => (
                      <div
                        key={key}
                        className="flex items-center justify-between rounded-xl border border-slate-800 bg-slate-900/70 px-3 py-2 text-sm"
                      >
                        <span className="text-slate-400">{key}</span>
                        <span className="font-semibold text-slate-100">
                          {value}
                        </span>
                      </div>
                    ),
                  )}
                </div>
              </FieldGroup>
            </div>
          </SectionCard>

          <div className="space-y-6">
            <SectionCard title="Pipeline Registry">
              <div className="space-y-4">
                <KV
                  label="Pipeline Version"
                  value={settings.pipeline.pipeline_version}
                  mono
                />
                <KV
                  label="Model Version"
                  value={settings.pipeline.model_version}
                  mono
                />
                <KV
                  label="Dataset Version"
                  value={settings.pipeline.dataset_version}
                  mono
                />
                <KV
                  label="Weights SHA256"
                  value={settings.pipeline.weights.sha256 || "n/a"}
                  mono
                />
                <KV
                  label="Weights MD5"
                  value={settings.pipeline.weights.md5 || "n/a"}
                  mono
                />
              </div>
            </SectionCard>

            <SectionCard title="Limits">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <KV
                  label="Max Image Size"
                  value={`${settings.limits.max_image_mb} MB`}
                />
                <KV
                  label="Max Video Size"
                  value={`${settings.limits.max_video_mb} MB`}
                />
                <KV
                  label="Max Upload Size"
                  value={`${settings.limits.max_upload_mb} MB`}
                />
                <KV
                  label="Rate Limit"
                  value={`${settings.limits.rate_limit_per_minute} req/min`}
                />
                <KV
                  label="Retention"
                  value={`${settings.limits.retention_days} days`}
                />
                <KV
                  label="Retention Interval"
                  value={`${settings.limits.retention_interval_hours} hours`}
                />
              </div>
            </SectionCard>

            <SectionCard title="Toolchain">
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {Object.entries(settings.toolchain).map(([key, value]) => (
                  <KV key={key} label={key} value={String(value)} mono />
                ))}
              </div>
            </SectionCard>
          </div>
        </div>

        <div className="rounded-2xl border border-slate-800/80 bg-slate-950/45 px-4 py-4 text-sm text-slate-400">
          Settings changes are audit logged. Every completed analysis stores the
          applied settings snapshot for traceability.
        </div>
      </div>
    </div>
  );
};

const SummaryStat = ({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: React.ReactNode;
  mono?: boolean;
}) => (
  <div className="rounded-2xl border border-slate-800/80 bg-slate-950/55 px-4 py-3">
    <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <div
      className={`mt-1 text-sm font-semibold text-slate-100 ${
        mono ? "break-all font-mono text-[13px]" : ""
      }`}
    >
      {value}
    </div>
  </div>
);

const NumberField = ({
  label,
  value,
  onChange,
  step = "1",
  min,
  max,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: string;
  min?: number;
  max?: number;
}) => (
  <label className="block">
    <div className="mb-2 text-xs uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <input
      type="number"
      step={step}
      min={min}
      max={max}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
    />
  </label>
);

const TextField = ({
  label,
  value,
  onChange,
  placeholder,
  mono = false,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  mono?: boolean;
}) => (
  <label className="block">
    <div className="mb-2 text-xs uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <input
      type="text"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      className={`w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20 ${
        mono ? "font-mono text-[13px]" : ""
      }`}
    />
  </label>
);

const SelectField = ({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: Array<{ value: string; label: string; disabled?: boolean }>;
}) => (
  <label className="block">
    <div className="mb-2 text-xs uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
    >
      {options.map((option) => (
        <option
          key={option.value}
          value={option.value}
          disabled={option.disabled}
        >
          {option.label}
        </option>
      ))}
    </select>
  </label>
);

const OverrideBadge = ({
  label,
  changed,
  value,
  baselineValue,
}: {
  label: string;
  changed: boolean;
  value: unknown;
  baselineValue: unknown;
}) => {
  const displayValue = normaliseDisplayValue(value);
  const displayBaseline = normaliseDisplayValue(baselineValue);

  return (
    <div
      className={`rounded-2xl border px-4 py-3 ${
        changed
          ? "border-amber-500/25 bg-amber-500/10"
          : "border-emerald-500/25 bg-emerald-500/10"
      }`}
    >
      <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
        {label}
      </div>

      <div className="mt-1 break-words text-sm font-medium text-slate-100">
        {displayValue}
      </div>

      <div className="mt-2 flex items-center gap-2 text-[11px]">
        <span
          className={`inline-block h-2 w-2 rounded-full ${
            changed ? "bg-amber-300" : "bg-emerald-300"
          }`}
        />
        <span className={changed ? "text-amber-200" : "text-emerald-200"}>
          {changed ? "Modified" : "Default"}
        </span>
      </div>

      {changed ? (
        <div className="mt-1 text-[11px] text-slate-400">
          Baseline: {displayBaseline}
        </div>
      ) : null}
    </div>
  );
};
