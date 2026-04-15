import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { getAudioById } from "../services/api";
import type { AudioAnalysisDetail } from "../types";
import { fixPath } from "../constants";

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === "object"
    ? (value as Record<string, unknown>)
    : null;

const formatClassification = (value?: string | null) => {
  if (!value) return "Unavailable";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const formatUnknown = (value: unknown, suffix?: string) => {
  if (typeof value === "number") {
    const formatted = Number.isInteger(value)
      ? String(value)
      : value.toFixed(3);
    return suffix ? `${formatted} ${suffix}` : formatted;
  }
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" && value.trim()) {
    return suffix ? `${value} ${suffix}` : value;
  }
  return "Unavailable";
};

const formatNestedStat = (
  source: Record<string, unknown>,
  key: string,
  stat: string,
) => {
  const nested = asRecord(source[key]);
  return formatUnknown(nested?.[stat]);
};

const Section = ({
  title,
  children,
  avoidBreak = true,
}: {
  title: string;
  children: React.ReactNode;
  avoidBreak?: boolean;
}) => (
  <section
    className="rounded-lg border border-slate-200 p-4"
    style={avoidBreak ? { breakInside: "avoid" } : undefined}
  >
    <div className="mb-3 text-xs uppercase tracking-[0.18em] text-slate-500">
      {title}
    </div>
    {children}
  </section>
);

const KV = ({
  label,
  value,
  mono = false,
}: {
  label: string;
  value?: string | null;
  mono?: boolean;
}) => (
  <div>
    <div className="text-xs uppercase text-slate-500">{label}</div>
    <div className={`${mono ? "font-mono text-xs" : "font-medium"} break-all`}>
      {value || "Unavailable"}
    </div>
  </div>
);

const PrintDecisionSummary = ({
  verdict,
  scoreValue,
  rationale,
  note,
}: {
  verdict: string;
  scoreValue: string;
  rationale: string[];
  note?: string;
}) => (
  <Section title="Decision Summary">
    <div className="grid grid-cols-1 gap-4 md:grid-cols-[1fr_180px]">
      <div>
        <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
          Final Verdict
        </div>
        <div className="mt-1 text-2xl font-semibold text-slate-900">
          {verdict}
        </div>
      </div>
      <div className="rounded-lg border border-slate-200 p-3">
        <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
          Audio Forensic Score
        </div>
        <div className="mt-1 text-2xl font-semibold text-slate-900">
          {scoreValue}
        </div>
      </div>
    </div>

    <div className="mt-4">
      <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
        Basis for Current Decision
      </div>
      <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800">
        {rationale.map((item, index) => (
          <li key={`${item}-${index}`}>{item}</li>
        ))}
      </ul>
    </div>

    {note ? <div className="mt-4 text-sm text-slate-600">{note}</div> : null}
  </Section>
);

const PrintAppliedSettings = ({
  settings,
}: {
  settings?: Record<string, unknown> | null;
}) => {
  const pipeline = asRecord(settings?.pipeline);
  const thresholds = asRecord(settings?.thresholds);
  const imageBands = asRecord(thresholds?.classification_bands);
  const audioBands = asRecord(thresholds?.audio_classification_bands);
  const paths = asRecord(settings?.paths);

  return (
    <Section title="Applied Settings">
      {!settings ? (
        <div className="text-sm text-slate-600">
          No settings snapshot was stored for this record.
        </div>
      ) : (
        <div className="space-y-4">
          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              Pipeline
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <KV
                label="Pipeline Version"
                value={formatUnknown(pipeline?.pipeline_version)}
              />
              <KV
                label="Image Detector"
                value={formatUnknown(pipeline?.image_detector)}
              />
              <KV
                label="Model Version"
                value={formatUnknown(pipeline?.model_version)}
              />
              <KV
                label="Dataset Version"
                value={formatUnknown(pipeline?.dataset_version)}
              />
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              Thresholds
            </div>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <KV
                label="Image AI Threshold"
                value={formatUnknown(imageBands?.ai_likely_min)}
              />
              <KV
                label="Image Real Threshold"
                value={formatUnknown(imageBands?.real_likely_max)}
              />
              <KV
                label="Audio AI Threshold"
                value={formatUnknown(audioBands?.ai_likely_min)}
              />
              <KV
                label="Audio Real Threshold"
                value={formatUnknown(audioBands?.real_likely_max)}
              />
              <KV
                label="Video Sample Frames"
                value={formatUnknown(thresholds?.video_sample_frames)}
              />
            </div>
          </div>

          <div>
            <div className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
              Runtime
            </div>
            <KV
              label="FFmpeg Path"
              value={formatUnknown(paths?.ffmpeg_path)}
              mono
            />
          </div>
        </div>
      )}
    </Section>
  );
};

export const PrintAudioPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AudioAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) {
        setError("No record id was provided.");
        setLoading(false);
        return;
      }

      try {
        setResult(await getAudioById(Number(id)));
      } catch {
        setError("Could not load audio analysis details.");
      } finally {
        setLoading(false);
      }
    };

    fetchDetail();
  }, [id]);

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center text-slate-500">
        Loading report...
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center gap-4 text-red-500">
        <p>{error || "Record not found"}</p>
        <Link to="/history" className="text-cyan-600 hover:underline">
          Back to Case History
        </Link>
      </div>
    );
  }

  const findings = Array.isArray(result.audio_features?.findings)
    ? result.audio_features.findings.map(String)
    : [];

  const segmentSummary = asRecord(result.audio_features?.segment_summary);
  const toolchain = asRecord(asRecord(result.applied_settings)?.toolchain);

  return (
    <div className="min-h-screen bg-slate-100 px-4 py-8 text-slate-900 print:bg-white print:p-0">
      <div className="mx-auto max-w-5xl space-y-6 rounded-xl border border-slate-200 bg-white p-6 print:border-0 print:p-0">
        <div
          className="flex items-start justify-between border-b border-slate-200 pb-4"
          style={{ breakInside: "avoid" }}
        >
          <div>
            <Link
              to={`/audio/${id}`}
              className="mb-2 inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-800 print:hidden"
            >
              <ArrowLeft size={14} />
              Back to Case
            </Link>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Forensic Report
            </div>
            <h1 className="mt-1 text-2xl font-semibold">
              Firinne Dhigiteach - Audio Evidence Assessment
            </h1>
            <p className="mt-1 text-slate-500">
              Case #{id} | Generated {new Date(result.created_at).toUTCString()}
            </p>
          </div>

          <button
            onClick={() => window.print()}
            className="rounded border border-slate-300 px-4 py-2 text-sm print:hidden"
          >
            Print / Save PDF
          </button>
        </div>

        <Section title="Case Metadata">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV label="Filename" value={result.filename} />
            <KV
              label="Classification"
              value={formatClassification(result.classification)}
            />
            <KV
              label="Forensic Score"
              value={result.forensic_score.toFixed(3)}
            />
            <KV
              label="Analysis Mode"
              value={formatUnknown(result.audio_features?.analysis_mode)}
            />
          </div>
        </Section>

        <Section title="Integrity Snapshot">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV
              label="SHA-256"
              value={
                result.file_integrity?.hashes_after?.sha256 ||
                result.file_integrity?.hashes_before?.sha256 ||
                "Unavailable"
              }
              mono
            />
            <KV
              label="MD5"
              value={
                result.file_integrity?.hashes_after?.md5 ||
                result.file_integrity?.hashes_before?.md5 ||
                "Unavailable"
              }
              mono
            />
            <KV
              label="Hashes Match"
              value={formatUnknown(result.file_integrity?.hashes_match)}
            />
          </div>
        </Section>

        <Section title="Audio Findings">
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800">
            {(findings.length
              ? findings
              : ["No explicit findings recorded."]
            ).map((item, index) => (
              <li key={`${item}-${index}`}>{item}</li>
            ))}
          </ul>
        </Section>

        <Section title="Signal Metadata">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV
              label="Duration"
              value={formatUnknown(
                result.audio_metadata?.duration_seconds,
                "seconds",
              )}
            />
            <KV
              label="Sample Rate"
              value={formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")}
            />
            <KV
              label="Channels"
              value={formatUnknown(result.audio_metadata?.channels)}
            />
            <KV
              label="Peak Level"
              value={formatUnknown(result.audio_features?.peak_level)}
            />
            <KV
              label="Dynamic Range"
              value={formatUnknown(
                result.audio_features?.dynamic_range_db,
                "dB",
              )}
            />
            <KV
              label="Zero Crossing Rate"
              value={formatUnknown(result.audio_features?.zero_crossing_rate)}
            />
            <KV
              label="Crest Factor"
              value={formatUnknown(result.audio_features?.crest_factor)}
            />
            <KV
              label="Repetition Score"
              value={formatUnknown(result.audio_features?.repetition_score)}
            />
            <KV
              label="Spectral Flatness"
              value={formatUnknown(result.audio_features?.spectral_flatness)}
            />
            {segmentSummary ? (
              <>
                <KV
                  label="Segment Size"
                  value={formatUnknown(
                    segmentSummary.segment_duration_seconds,
                    "seconds",
                  )}
                />
                <KV
                  label="Segment Count"
                  value={formatUnknown(segmentSummary.segment_count)}
                />
                <KV
                  label="Segment RMS Std"
                  value={formatNestedStat(segmentSummary, "rms_level", "std")}
                />
                <KV
                  label="Segment ZCR Std"
                  value={formatNestedStat(
                    segmentSummary,
                    "zero_crossing_rate",
                    "std",
                  )}
                />
              </>
            ) : null}
          </div>
        </Section>

        <Section title="Analysis Diagnostics">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <KV
              label="FFmpeg Available"
              value={formatUnknown(toolchain?.ffmpeg_available)}
            />
            <KV
              label="Analysis Mode"
              value={formatUnknown(result.audio_features?.analysis_mode)}
            />
            <KV
              label="Transcoded"
              value={formatUnknown(
                result.audio_features?.transcoded_for_analysis,
              )}
            />
            <KV
              label="FFmpeg Error"
              value={formatUnknown(
                result.audio_features?.ffmpeg_transcode_error,
              )}
            />
          </div>
        </Section>

        <PrintDecisionSummary
          verdict={formatClassification(result.classification)}
          scoreValue={result.forensic_score.toFixed(3)}
          rationale={[
            `Analysis mode: ${formatUnknown(result.audio_features?.analysis_mode)}`,
            `Duration: ${formatUnknown(result.audio_metadata?.duration_seconds, "seconds")}`,
            `Sample rate: ${formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")}`,
            ...(findings.length
              ? findings.slice(0, 2)
              : ["No explicit findings recorded."]),
          ]}
          note="This audio result is a triage output and should be interpreted with surrounding case context."
        />

        {result.waveform_path ? (
          <Section title="Waveform Exhibit">
            <img
              src={fixPath(result.waveform_path)}
              alt="Waveform preview"
              className="w-full object-contain"
            />
          </Section>
        ) : null}

        {typeof result.audio_features?.spectrogram_path === "string" ? (
          <Section title="Spectrogram Exhibit">
            <img
              src={fixPath(result.audio_features.spectrogram_path)}
              alt="Spectrogram preview"
              className="w-full object-contain"
            />
          </Section>
        ) : null}

        <PrintAppliedSettings settings={result.applied_settings} />

        <section className="border-t border-slate-200 pt-3 text-[11px] text-slate-500">
          This report is decision-support evidence and should be interpreted
          with contextual forensic review.
        </section>
      </div>
    </div>
  );
};
