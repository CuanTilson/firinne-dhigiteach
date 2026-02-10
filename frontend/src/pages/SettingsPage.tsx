import React, { useEffect, useState } from "react";
import { getSettings } from "../services/api";
import type { SettingsSnapshot } from "../types";

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
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchSettings = async () => {
      try {
        const data = await getSettings();
        setSettings(data);
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
          <div className="fd-section-title">Thresholds</div>
          <KV
            label="AI Likely Min"
            value={settings.thresholds.classification_bands.ai_likely_min}
          />
          <KV
            label="Real Likely Max"
            value={settings.thresholds.classification_bands.real_likely_max}
          />
          <KV
            label="Video Max Duration"
            value={`${settings.thresholds.video_max_duration_seconds}s`}
          />
          <KV
            label="Video Sample Frames"
            value={settings.thresholds.video_sample_frames}
          />
          <KV
            label="Scene Cut Threshold"
            value={settings.thresholds.scene_cut_threshold}
          />
          <KV
            label="Scene Cut Stride"
            value={settings.thresholds.scene_cut_stride}
          />
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
        Settings are read-only to preserve chain of custody. Any change should
        be logged as a new audit entry and recorded with the analysis results.
      </div>
    </div>
  );
};
