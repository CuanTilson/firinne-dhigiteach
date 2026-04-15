import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { getAudioById } from "../services/api";
import type { AudioAnalysisDetail } from "../types";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";
import { EvidenceStatusStrip } from "../components/EvidenceStatusStrip";
import { CasePageScaffold } from "../components/CasePageScaffold";
import { AudioEvidencePanel } from "../components/AudioEvidencePanel";

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

const getClassificationTone = (
  classification?: string | null,
): "good" | "warn" | "neutral" => {
  if (classification === "likely_real") return "good";
  if (classification === "likely_ai_generated") return "warn";
  return "neutral";
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
    {detail ? (
      <div className="mt-1 text-xs text-slate-500">{detail}</div>
    ) : null}
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

export const AudioDetailPage: React.FC = () => {
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
        const data = await getAudioById(Number(id));
        setResult(data);
      } catch {
        setError("Could not load audio analysis details.");
      } finally {
        setLoading(false);
      }
    };

    fetchDetail();
  }, [id]);

  const findings = useMemo(() => {
    const value = result?.audio_features?.findings;
    return Array.isArray(value) ? value.map(String) : [];
  }, [result]);

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
        <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 px-6 py-12 text-center text-slate-400">
          Loading audio analysis #{id}...
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
        <div className="rounded-3xl border border-red-500/20 bg-red-500/10 px-6 py-12 text-center">
          <p className="text-red-300">{error || "Record not found."}</p>
          <Link
            to="/history"
            className="mt-4 inline-flex text-cyan-300 underline-offset-4 hover:text-cyan-200 hover:underline"
          >
            Back to Case History
          </Link>
        </div>
      </div>
    );
  }

  const segmentSummary = asRecord(result.audio_features?.segment_summary);
  const toolchain = asRecord(asRecord(result.applied_settings)?.toolchain);
  const hashes = result.file_integrity?.hashes;
  const channels =
    typeof result.audio_metadata?.channels === "number"
      ? `${result.audio_metadata.channels} channel${
          result.audio_metadata.channels === 1 ? "" : "s"
        }`
      : "Channel count unavailable";

  return (
    <CasePageScaffold
      backTo="/history"
      backLabel="Back to Case History"
      header={
        <CaseHeader
          title="Audio Analysis"
          caseId={id || ""}
          filename={result.filename}
          createdAt={result.created_at}
          printUrl={`#/print/audio/${id}`}
          hashes={{
            sha256: hashes?.sha256,
            md5: hashes?.md5,
            sha256_before: result.file_integrity?.hashes_before?.sha256,
            sha256_after: result.file_integrity?.hashes_after?.sha256,
            md5_before: result.file_integrity?.hashes_before?.md5,
            md5_after: result.file_integrity?.hashes_after?.md5,
          }}
        />
      }
      statusStrip={
        <EvidenceStatusStrip
          items={[
            {
              label: "Audio Verdict",
              status: formatClassification(result.classification),
              tone: getClassificationTone(result.classification),
              detail: `Score ${result.forensic_score.toFixed(3)}`,
            },
            {
              label: "Analysis Mode",
              status: formatUnknown(result.audio_features?.analysis_mode),
              detail: formatUnknown(
                result.audio_metadata?.duration_seconds,
                "seconds",
              ),
            },
            {
              label: "Signal Metadata",
              status: formatUnknown(
                result.audio_metadata?.sample_rate_hz,
                "Hz",
              ),
              tone: result.audio_metadata?.sample_rate_hz ? "good" : "warn",
              detail: channels,
            },
            {
              label: "Integrity",
              status:
                typeof result.file_integrity?.hashes_match === "boolean"
                  ? result.file_integrity.hashes_match
                    ? "Hashes match"
                    : "Hash mismatch"
                  : "Hashes recorded",
              tone:
                result.file_integrity?.hashes_match === false ? "bad" : "good",
              detail: result.file_integrity?.hashes_after?.sha256
                ? "Post-analysis hash stored"
                : undefined,
            },
          ]}
        />
      }
      sidebar={
        <>
          <DecisionSummaryPanel
            verdict={formatClassification(result.classification)}
            scoreLabel="Audio Forensic Score"
            scoreValue={result.forensic_score.toFixed(3)}
            rationale={[
              `Analysis mode: ${formatUnknown(result.audio_features?.analysis_mode)}`,
              `Duration: ${formatUnknown(result.audio_metadata?.duration_seconds, "seconds")}`,
              `Sample rate: ${formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")}`,
              ...(findings.length
                ? findings.slice(0, 3)
                : ["No dominant audio findings recorded."]),
            ]}
            note="This is an audio triage result and should be read with contextual forensic review."
          />

          <ChainOfCustody
            steps={[
              {
                label: "Upload received",
                timestamp: result.created_at,
                status: "complete",
              },
              {
                label: "Audio triage completed",
                timestamp: result.created_at,
                status: "complete",
              },
            ]}
          />

          <AppliedSettingsPanel settings={result.applied_settings} />
        </>
      }
    >
      <AudioEvidencePanel result={result} />

      {segmentSummary ? (
        <SectionCard title="Segment Summary">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            <CompactField
              label="Segment Size"
              value={formatUnknown(
                segmentSummary.segment_duration_seconds,
                "seconds",
              )}
              detail={`${formatUnknown(segmentSummary.segment_count)} segments`}
            />
            <CompactField
              label="RMS Variation"
              value={formatNestedStat(segmentSummary, "rms_level", "std")}
              detail={`Mean ${formatNestedStat(segmentSummary, "rms_level", "mean")}`}
            />
            <CompactField
              label="ZCR Variation"
              value={formatNestedStat(
                segmentSummary,
                "zero_crossing_rate",
                "std",
              )}
              detail={`Mean ${formatNestedStat(
                segmentSummary,
                "zero_crossing_rate",
                "mean",
              )}`}
            />
            <CompactField
              label="Flatness Variation"
              value={formatNestedStat(
                segmentSummary,
                "spectral_flatness",
                "std",
              )}
              detail={`Mean ${formatNestedStat(
                segmentSummary,
                "spectral_flatness",
                "mean",
              )}`}
            />
          </div>
        </SectionCard>
      ) : null}

      <SectionCard title="Analysis Diagnostics">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <CompactField
            label="FFmpeg Available"
            value={formatUnknown(toolchain?.ffmpeg_available)}
            detail={formatUnknown(toolchain?.ffmpeg_resolved_path)}
            mono
          />
          <CompactField
            label="Analysis Mode"
            value={formatUnknown(result.audio_features?.analysis_mode)}
            detail={
              result.audio_features?.transcoded_for_analysis === true
                ? "Transcoded before waveform analysis"
                : "Direct analysis path"
            }
          />
          <CompactField
            label="Transcoded"
            value={formatUnknown(
              result.audio_features?.transcoded_for_analysis,
            )}
          />
          <CompactField
            label="FFmpeg Error"
            value={formatUnknown(result.audio_features?.ffmpeg_transcode_error)}
            mono
          />
        </div>
      </SectionCard>

      <SectionCard title="Analyst Findings">
        <ul className="space-y-2 text-sm text-slate-300">
          {(findings.length
            ? findings
            : ["No explicit findings recorded."]
          ).map((item, index) => (
            <li
              key={`${item}-${index}`}
              className="rounded-xl border border-slate-800 bg-slate-950/60 px-3 py-2"
            >
              {item}
            </li>
          ))}
        </ul>
      </SectionCard>
    </CasePageScaffold>
  );
};
