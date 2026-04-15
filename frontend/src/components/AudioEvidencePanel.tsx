import React from "react";
import type { AudioAnalysisDetail } from "../types";
import { fixPath } from "../constants";

const formatClassification = (value?: string | null) => {
  if (!value) return "Unavailable";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const formatUnknown = (value: unknown, suffix?: string) => {
  if (typeof value === "number") {
    const formatted = Number.isInteger(value) ? String(value) : value.toFixed(3);
    return suffix ? `${formatted} ${suffix}` : formatted;
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" && value.trim()) {
    return suffix ? `${value} ${suffix}` : value;
  }
  return "Unavailable";
};

const CompactField = ({
  label,
  value,
  detail,
  mono = false,
}: {
  label: string;
  value: string;
  detail?: string;
  mono?: boolean;
}) => (
  <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
    <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
      {label}
    </div>
    <div
      className={`mt-1 text-slate-200 ${
        mono ? "break-all font-mono text-[13px]" : ""
      }`}
    >
      {value}
    </div>
    {detail ? <div className="mt-1 text-xs text-slate-500">{detail}</div> : null}
  </div>
);

const SectionCard = ({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) => (
  <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-4 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
    <div className="mb-3 text-[11px] uppercase tracking-[0.2em] text-slate-500">
      {title}
    </div>
    {children}
  </div>
);

interface Props {
  result: AudioAnalysisDetail;
}

export const AudioEvidencePanel: React.FC<Props> = ({ result }) => {
  return (
    <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.1fr_0.9fr]">
      <SectionCard title="Audio Evidence">
        <div className="space-y-4">
          <div>
            <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              Summary
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <CompactField
                label="Classification"
                value={formatClassification(result.classification)}
              />
              <CompactField
                label="Forensic Score"
                value={result.forensic_score.toFixed(3)}
              />
              <CompactField
                label="Analysis Mode"
                value={formatUnknown(result.audio_features?.analysis_mode)}
              />
              <CompactField
                label="Duration"
                value={formatUnknown(result.audio_metadata?.duration_seconds, "seconds")}
              />
              <CompactField
                label="Sample Rate"
                value={formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")}
              />
              <CompactField
                label="Channels"
                value={formatUnknown(result.audio_metadata?.channels)}
              />
              <CompactField
                label="Hashes Match"
                value={formatUnknown(result.file_integrity?.hashes_match)}
              />
            </div>
          </div>

          <div>
            <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
              Signal Features
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <CompactField
                label="Peak Level"
                value={formatUnknown(result.audio_features?.peak_level)}
              />
              <CompactField
                label="Dynamic Range"
                value={formatUnknown(result.audio_features?.dynamic_range_db, "dB")}
              />
              <CompactField
                label="Zero Crossing Rate"
                value={formatUnknown(result.audio_features?.zero_crossing_rate)}
              />
              <CompactField
                label="Crest Factor"
                value={formatUnknown(result.audio_features?.crest_factor)}
              />
              <CompactField
                label="Repetition Score"
                value={formatUnknown(result.audio_features?.repetition_score)}
              />
              <CompactField
                label="Spectral Flatness"
                value={formatUnknown(result.audio_features?.spectral_flatness)}
              />
              <CompactField
                label="Dominant Frequency"
                value={formatUnknown(result.audio_features?.dominant_frequency_hz, "Hz")}
              />
              <CompactField
                label="Spectral Centroid"
                value={formatUnknown(result.audio_features?.spectral_centroid_hz, "Hz")}
              />
            </div>
          </div>
        </div>
      </SectionCard>

      <SectionCard title="Signal Exhibits">
        <div className="space-y-4">
          {result.waveform_path ? (
            <div>
              <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                Waveform Preview
              </div>
              <img
                src={fixPath(result.waveform_path)}
                alt="Waveform preview"
                className="w-full rounded border border-slate-800 bg-slate-950"
              />
            </div>
          ) : (
            <div className="text-sm text-slate-400">
              No waveform preview available for this file.
            </div>
          )}

          {typeof result.audio_features?.spectrogram_path === "string" ? (
            <div>
              <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                Spectrogram Preview
              </div>
              <img
                src={fixPath(result.audio_features.spectrogram_path)}
                alt="Spectrogram preview"
                className="w-full rounded border border-slate-800 bg-slate-950"
              />
            </div>
          ) : null}
        </div>
      </SectionCard>
    </div>
  );
};