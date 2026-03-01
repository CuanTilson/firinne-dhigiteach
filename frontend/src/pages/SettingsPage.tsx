import React, { useEffect, useState } from "react";
import { getSettings, updateSettings } from "../services/api";
import type { SettingsSnapshot, SettingsUpdatePayload } from "../types";
import { DEFAULT_ADMIN_KEY } from "../constants";

const KV = ({ label, value }: { label: string; value: React.ReactNode }) => (
  <div>
    <div className="text-xs uppercase tracking-widest text-slate-500">
      {label}
    </div>
    <div className="text-sm text-slate-200 mt-1 break-all">{value}</div>
  </div>
);

export const SettingsPage: React.FC = () => {
  const [settings, setSettings] = useState<SettingsSnapshot | null>(null);
  const [form, setForm] = useState<SettingsUpdatePayload | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<string | null>(null);

  const isOverridden = (currentValue: unknown, defaultValue: unknown) =>
    String(currentValue ?? "") !== String(defaultValue ?? "");

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await getSettings();
        setSettings(data);
        setForm({
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
      } catch {
        setError("Could not load settings.");
      } finally {
        setLoading(false);
      }
    };
    fetchSettings();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        Loading settings...
      </div>
    );
  }

  if (error || !settings) {
    return (
      <div className="min-h-screen flex items-center justify-center text-red-400">
        {error || "Settings unavailable."}
      </div>
    );
  }

  const updateBandField = (
    section: "classification_bands" | "audio_classification_bands",
    field: "ai_likely_min" | "real_likely_max",
    value: number
  ) => {
    setForm((current) =>
      current
        ? {
            ...current,
            thresholds: {
              ...current.thresholds,
              [section]: {
                ...current.thresholds?.[section],
                [field]: value,
              },
            },
          }
        : current
    );
  };

  const currentImageAiLikelyMin =
    form?.thresholds?.classification_bands?.ai_likely_min ??
    settings.thresholds.classification_bands.ai_likely_min;
  const currentImageRealLikelyMax =
    form?.thresholds?.classification_bands?.real_likely_max ??
    settings.thresholds.classification_bands.real_likely_max;
  const currentAudioAiLikelyMin =
    form?.thresholds?.audio_classification_bands?.ai_likely_min ??
    settings.thresholds.audio_classification_bands.ai_likely_min;
  const currentAudioRealLikelyMax =
    form?.thresholds?.audio_classification_bands?.real_likely_max ??
    settings.thresholds.audio_classification_bands.real_likely_max;
  const currentVideoMaxDuration =
    form?.thresholds?.video_max_duration_seconds ??
    settings.thresholds.video_max_duration_seconds;
  const currentVideoSampleFrames =
    form?.thresholds?.video_sample_frames ??
    settings.thresholds.video_sample_frames;
  const currentFfmpegPath = form?.paths?.ffmpeg_path ?? settings.paths.ffmpeg_path;
  const currentImageDetector =
    form?.pipeline?.image_detector ?? settings.pipeline.image_detector;

  const save = async () => {
    if (!form) return;
    setSaving(true);
    setSaveMessage(null);
    setError(null);
    try {
      const next = await updateSettings(form, DEFAULT_ADMIN_KEY);
      setSettings(next);
      setSaveMessage("Settings updated. New analyses will store this snapshot.");
    } catch {
      setError("Could not update settings.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div>
        <div className="fd-kicker mb-2">Governance</div>
        <h1 className="text-3xl font-semibold text-slate-100 mb-1 fd-title">
          Settings & Model Registry
        </h1>
        <p className="text-slate-400">
          Snapshot of configuration, thresholds, and model metadata.
        </p>
      </div>

      <div className="fd-panel p-4 text-sm text-slate-300">
        Each completed analysis now stores the exact applied settings snapshot.
        Changing settings affects future analyses only; existing records retain their original configuration.
      </div>

      <div className="fd-card p-5 space-y-3">
        <div className="fd-section-title">Runtime Override Status</div>
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3">
          <OverrideBadge
            label="Image Detector"
            overridden={isOverridden(
              currentImageDetector,
              settings.pipeline.image_detector
            )}
            value={currentImageDetector}
            defaultValue={settings.pipeline.image_detector}
          />
          <OverrideBadge
            label="Image AI Likely Min"
            overridden={isOverridden(
              currentImageAiLikelyMin,
              settings.thresholds.classification_bands.ai_likely_min
            )}
            value={currentImageAiLikelyMin}
            defaultValue={settings.thresholds.classification_bands.ai_likely_min}
          />
          <OverrideBadge
            label="Image Real Likely Max"
            overridden={isOverridden(
              currentImageRealLikelyMax,
              settings.thresholds.classification_bands.real_likely_max
            )}
            value={currentImageRealLikelyMax}
            defaultValue={settings.thresholds.classification_bands.real_likely_max}
          />
          <OverrideBadge
            label="Audio AI Likely Min"
            overridden={isOverridden(
              currentAudioAiLikelyMin,
              settings.thresholds.audio_classification_bands.ai_likely_min
            )}
            value={currentAudioAiLikelyMin}
            defaultValue={settings.thresholds.audio_classification_bands.ai_likely_min}
          />
          <OverrideBadge
            label="Audio Real Likely Max"
            overridden={isOverridden(
              currentAudioRealLikelyMax,
              settings.thresholds.audio_classification_bands.real_likely_max
            )}
            value={currentAudioRealLikelyMax}
            defaultValue={settings.thresholds.audio_classification_bands.real_likely_max}
          />
          <OverrideBadge
            label="Video Max Duration"
            overridden={isOverridden(
              currentVideoMaxDuration,
              settings.thresholds.video_max_duration_seconds
            )}
            value={currentVideoMaxDuration}
            defaultValue={settings.thresholds.video_max_duration_seconds}
          />
          <OverrideBadge
            label="Video Sample Frames"
            overridden={isOverridden(
              currentVideoSampleFrames,
              settings.thresholds.video_sample_frames
            )}
            value={currentVideoSampleFrames}
            defaultValue={settings.thresholds.video_sample_frames}
          />
          <OverrideBadge
            label="FFmpeg Path"
            overridden={isOverridden(currentFfmpegPath, settings.paths.ffmpeg_path)}
            value={currentFfmpegPath || "Auto-discovery"}
            defaultValue={settings.paths.ffmpeg_path || "Auto-discovery"}
          />
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="fd-card p-5 space-y-4">
          <div className="fd-section-title">Pipeline</div>
          <KV label="Pipeline Version" value={settings.pipeline.pipeline_version} />
          <KV label="Model Version" value={settings.pipeline.model_version} />
          <KV label="Dataset Version" value={settings.pipeline.dataset_version} />
          <KV label="Weights SHA256" value={settings.pipeline.weights.sha256} />
          <KV label="Weights MD5" value={settings.pipeline.weights.md5} />
        </div>

        <div className="fd-card p-5 space-y-4">
          <div className="fd-section-title">Limits</div>
          <KV label="Max Image Size" value={`${settings.limits.max_image_mb} MB`} />
          <KV label="Max Video Size" value={`${settings.limits.max_video_mb} MB`} />
          <KV label="Max Upload Size" value={`${settings.limits.max_upload_mb} MB`} />
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
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="fd-card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="fd-section-title">Analysis Settings</div>
            <button
              onClick={save}
              disabled={saving || !form || !DEFAULT_ADMIN_KEY}
              className="px-3 py-2 rounded border border-slate-700 text-sm text-slate-200 disabled:opacity-50"
            >
              {saving ? "Saving..." : "Save Settings"}
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
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
                    : current
                )
              }
              options={Object.entries(
                settings.pipeline.available_image_detectors || {}
              ).map(([value, item]) => ({
                value,
                label: `${item.display_name}${item.available ? "" : " (Unavailable)"}`,
                disabled: !item.available,
              }))}
            />
            <NumberField
              label="Image AI Likely Min"
              value={currentImageAiLikelyMin}
              onChange={(value) => updateBandField("classification_bands", "ai_likely_min", value)}
              step="0.01"
            />
            <NumberField
              label="Image Real Likely Max"
              value={currentImageRealLikelyMax}
              onChange={(value) => updateBandField("classification_bands", "real_likely_max", value)}
              step="0.01"
            />
            <NumberField
              label="Audio AI Likely Min"
              value={currentAudioAiLikelyMin}
              onChange={(value) => updateBandField("audio_classification_bands", "ai_likely_min", value)}
              step="0.01"
            />
            <NumberField
              label="Audio Real Likely Max"
              value={currentAudioRealLikelyMax}
              onChange={(value) => updateBandField("audio_classification_bands", "real_likely_max", value)}
              step="0.01"
            />
            <NumberField
              label="Video Max Duration (s)"
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
                    : current
                )
              }
            />
            <NumberField
              label="Video Sample Frames"
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
                    : current
                )
              }
            />
          </div>
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
                  : current
              )
            }
            placeholder="Leave blank to use PATH or auto-discovery"
          />
          {saveMessage ? (
            <div className="text-sm text-emerald-300">{saveMessage}</div>
          ) : null}
          {!DEFAULT_ADMIN_KEY ? (
            <div className="text-sm text-amber-300">
              `VITE_ADMIN_KEY` is not configured, so settings updates are disabled in this UI.
            </div>
          ) : null}
          {error ? <div className="text-sm text-red-400">{error}</div> : null}
          <div>
            <div className="text-xs uppercase tracking-widest text-slate-500">
              Fusion Weights
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-sm text-slate-300">
              {Object.entries(settings.thresholds.fusion_weights).map(
                ([key, value]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between bg-slate-900/70 border border-slate-800 rounded px-2 py-1"
                  >
                    <span className="text-slate-400">{key}</span>
                    <span className="font-semibold">{value}</span>
                  </div>
                )
              )}
            </div>
          </div>
        </div>

        <div className="fd-card p-5 space-y-4">
          <div className="fd-section-title">Toolchain</div>
          {Object.entries(settings.toolchain).map(([key, value]) => (
            <KV key={key} label={key} value={value} />
          ))}
        </div>
      </div>

      <div className="fd-panel p-4 text-sm text-slate-400">
        Settings changes are audit logged. Every completed analysis stores the applied settings snapshot for traceability.
      </div>
    </div>
  );
};

const NumberField = ({
  label,
  value,
  onChange,
  step = "1",
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: string;
}) => (
  <label className="block">
    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">{label}</div>
    <input
      type="number"
      step={step}
      value={value}
      onChange={(event) => onChange(Number(event.target.value))}
      className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
    />
  </label>
);

const TextField = ({
  label,
  value,
  onChange,
  placeholder,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
}) => (
  <label className="block">
    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">{label}</div>
    <input
      type="text"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      placeholder={placeholder}
      className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
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
    <div className="text-xs uppercase tracking-widest text-slate-500 mb-2">{label}</div>
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="w-full rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100"
    >
      {options.map((option) => (
        <option key={option.value} value={option.value} disabled={option.disabled}>
          {option.label}
        </option>
      ))}
    </select>
  </label>
);

const OverrideBadge = ({
  label,
  overridden,
  value,
  defaultValue,
}: {
  label: string;
  overridden: boolean;
  value: unknown;
  defaultValue: unknown;
}) => (
  <div
    className={`rounded-lg border px-3 py-2 ${
      overridden
        ? "border-amber-500/30 bg-amber-500/10"
        : "border-emerald-500/30 bg-emerald-500/10"
    }`}
  >
    <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
    <div className="mt-1 text-sm text-slate-100">
      {String(value === "" ? "Auto-discovery" : value)}
    </div>
    <div className="mt-1 text-[11px] text-slate-400">
      {overridden ? "Runtime override active" : "Using stored default"}
      {" · "}
      baseline {String(defaultValue === "" ? "Auto-discovery" : defaultValue)}
    </div>
  </div>
);
