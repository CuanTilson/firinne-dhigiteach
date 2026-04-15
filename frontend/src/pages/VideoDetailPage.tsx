import React, { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getVideoById } from "../services/api";
import type {
  AudioAnalysisSummary,
  VideoAnalysisDetail,
  VideoFrameResult,
} from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { fixPath } from "../constants";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";
import { EvidenceStatusStrip } from "../components/EvidenceStatusStrip";
import { AnalysisProvenancePanel } from "../components/AnalysisProvenancePanel";
import { CasePageScaffold } from "../components/CasePageScaffold";

type AudioValue = AudioAnalysisSummary["forensic_score"] | string | unknown;

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

const formatUnknown = (value: AudioValue, suffix?: string) => {
  if (typeof value === "number") {
    const formatted = Number.isInteger(value)
      ? String(value)
      : value.toFixed(3);
    return suffix ? `${formatted} ${suffix}` : formatted;
  }
  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }
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

const getClassificationTone = (
  classification?: string | null,
): "good" | "warn" | "neutral" => {
  if (classification === "likely_real") return "good";
  if (classification === "likely_ai_generated") return "warn";
  return "neutral";
};

const getAudioTone = (
  audio: VideoAnalysisDetail["audio_analysis"],
): "good" | "warn" | "bad" => {
  if (audio?.error) return "bad";
  if (audio?.available) {
    if (audio.classification === "likely_ai_generated") return "warn";
    return "good";
  }
  return "warn";
};

const getAudioExtractionStep = (
  audio: VideoAnalysisDetail["audio_analysis"],
) => {
  if (audio?.error) {
    return {
      label: "Audio extraction failed",
      status: "complete" as const,
    };
  }

  if (audio?.available) {
    return {
      label: "Audio extraction completed",
      status: "complete" as const,
    };
  }

  return {
    label: "Audio extraction unavailable",
    status: "complete" as const,
  };
};

const AudioField = ({
  label,
  value,
  mono = false,
}: {
  label: string;
  value: string;
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

export const VideoDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<VideoAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) {
        setError("No record id was provided.");
        setLoading(false);
        return;
      }

      try {
        const data = await getVideoById(Number(id));
        setResult(data);
      } catch {
        setError("Could not load video analysis details.");
      } finally {
        setLoading(false);
      }
    };

    fetchDetail();
  }, [id]);

  const selectedFrame: VideoFrameResult | null = useMemo(() => {
    if (!result?.frames?.length) return null;
    return result.frames[Math.min(selectedIndex, result.frames.length - 1)];
  }, [result, selectedIndex]);

  if (loading) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
        <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 px-6 py-12 text-center text-slate-400">
          Loading video analysis #{id}...
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
            Back to History
          </Link>
        </div>
      </div>
    );
  }

  const videoMeta = asRecord(result.video_metadata);
  const hashesBefore = asRecord(videoMeta?.hashes_before);
  const hashesAfter = asRecord(videoMeta?.hashes_after);
  const hashesCurrent = asRecord(videoMeta?.hashes);
  const imageDetector = asRecord(videoMeta?.image_detector);

  const audio = result.audio_analysis ?? null;
  const audioIntegrity = asRecord(audio?.file_integrity);
  const audioMeta = asRecord(audio?.audio_metadata);
  const audioFeatures = asRecord(audio?.audio_features);
  const audioSegments = asRecord(audioFeatures?.segment_summary);
  const audioHashesBefore = asRecord(audioIntegrity?.hashes_before);
  const audioHashesAfter = asRecord(audioIntegrity?.hashes_after);
  const extractedAudioSha256 =
    typeof audioHashesAfter?.sha256 === "string"
      ? audioHashesAfter.sha256
      : typeof audioHashesBefore?.sha256 === "string"
        ? audioHashesBefore.sha256
        : null;
  const extractedAudioMd5 =
    typeof audioHashesAfter?.md5 === "string"
      ? audioHashesAfter.md5
      : typeof audioHashesBefore?.md5 === "string"
        ? audioHashesBefore.md5
        : null;

  const audioStep = getAudioExtractionStep(audio);

  return (
    <CasePageScaffold
      backTo="/history"
      backLabel="Back to History"
      header={
        <CaseHeader
          title="Video Analysis"
          caseId={id || ""}
          filename={result.filename}
          createdAt={result.created_at}
          printUrl={`#/print/videos/${id}`}
          hashes={{
            sha256:
              typeof hashesCurrent?.sha256 === "string"
                ? hashesCurrent.sha256
                : undefined,
            md5:
              typeof hashesCurrent?.md5 === "string"
                ? hashesCurrent.md5
                : undefined,
            sha256_before:
              typeof hashesBefore?.sha256 === "string"
                ? hashesBefore.sha256
                : undefined,
            sha256_after:
              typeof hashesAfter?.sha256 === "string"
                ? hashesAfter.sha256
                : undefined,
            md5_before:
              typeof hashesBefore?.md5 === "string"
                ? hashesBefore.md5
                : undefined,
            md5_after:
              typeof hashesAfter?.md5 === "string"
                ? hashesAfter.md5
                : undefined,
          }}
        />
      }
      statusStrip={
        <EvidenceStatusStrip
          items={[
            {
              label: "Visual Verdict",
              status: formatClassification(result.classification),
              tone: getClassificationTone(result.classification),
              detail: `Score ${result.forensic_score.toFixed(3)}`,
            },
            {
              label: "Frame Sampling",
              status: `${result.frame_count} frames`,
              tone: result.frame_count > 0 ? "good" : "bad",
              detail: selectedFrame
                ? `Current frame ${selectedFrame.frame_index}`
                : "No frame selected",
            },
            {
              label: "Extracted Audio",
              status: audio?.available
                ? formatClassification(audio.classification)
                : "Unavailable",
              tone: getAudioTone(audio),
              detail:
                audio?.error ||
                (typeof audio?.forensic_score === "number"
                  ? `Score ${audio.forensic_score.toFixed(3)}`
                  : undefined),
            },
            {
              label: "Video Integrity",
              status: hashesCurrent?.sha256
                ? "Hashes recorded"
                : "Hashes unavailable",
              tone: hashesCurrent?.sha256 ? "good" : "warn",
              detail:
                hashesBefore?.sha256 && hashesAfter?.sha256
                  ? "Before/after hashes stored"
                  : undefined,
            },
          ]}
        />
      }
      sidebar={
        <>
          <DecisionSummaryPanel
            verdict={formatClassification(result.classification)}
            scoreLabel="Visual Aggregate Score"
            scoreValue={result.forensic_score.toFixed(3)}
            rationale={[
              `Frame count analysed: ${result.frame_count}`,
              `Selected frame score: ${
                selectedFrame
                  ? selectedFrame.forensic_score.toFixed(3)
                  : "Unavailable"
              }`,
              `Extracted audio classification: ${formatClassification(
                audio?.classification,
              )}`,
              `Extracted audio score: ${
                typeof audio?.forensic_score === "number"
                  ? audio.forensic_score.toFixed(3)
                  : "Unavailable"
              }`,
            ]}
            note="Current video decisions are based on sampled frame analysis, with extracted audio triage shown as a parallel evidence stream."
          />

          <AnalysisProvenancePanel
            detector={imageDetector}
            fusionMode="rule_based_forensic_fusion"
          />

          <ChainOfCustody
            steps={[
              {
                label: "Upload received",
                timestamp: result.created_at,
                status: "complete",
              },
              {
                label: "Frame sampling completed",
                timestamp: result.created_at,
                status: "complete",
              },
              {
                label: audioStep.label,
                timestamp: result.created_at,
                status: audioStep.status,
              },
              {
                label: "Analysis completed",
                timestamp: result.created_at,
                status: "complete",
              },
            ]}
          />

          <AppliedSettingsPanel settings={result.applied_settings} />
        </>
      }
    >
      <SectionCard title="Frame Strip">
        {result.frames?.length ? (
          <div className="flex gap-3 overflow-x-auto pb-2">
            {result.frames.map((frame, idx) => (
              <button
                key={`${frame.frame_index}-${idx}`}
                type="button"
                onClick={() => setSelectedIndex(idx)}
                className={`min-w-[160px] rounded-2xl border p-2 text-left transition ${
                  selectedIndex === idx
                    ? "border-cyan-400/50 bg-cyan-900/20"
                    : "border-slate-800 hover:bg-slate-900/60"
                }`}
              >
                <div className="h-24 w-full overflow-hidden rounded-xl border border-slate-800 bg-slate-950">
                  <img
                    src={fixPath(frame.saved_path)}
                    alt={`Frame ${frame.frame_index}`}
                    className="h-full w-full object-cover"
                  />
                </div>

                <div className="mt-2 text-xs text-slate-300">
                  <div>Frame {frame.frame_index}</div>
                  <div className="text-slate-500">
                    {frame.timestamp_sec.toFixed(2)}s
                  </div>
                  <div className="text-slate-400">
                    Score {frame.forensic_score.toFixed(3)}
                  </div>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="text-slate-400">
            No sampled frames were stored for this case.
          </div>
        )}
      </SectionCard>

      <SectionCard title="Selected Frame Analysis">
        {selectedFrame ? (
          <AnalysisDashboard result={selectedFrame} />
        ) : (
          <div className="text-slate-400">
            No frames available for analysis.
          </div>
        )}
      </SectionCard>

      <SectionCard title="Extracted Audio Evidence">
        {!audio ? (
          <div className="text-sm text-slate-400">
            No extracted-audio analysis was stored for this case.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-6 xl:grid-cols-[1.05fr_0.95fr]">
            <div className="space-y-4">
              <div>
                <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  Summary
                </div>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <AudioField
                    label="Available"
                    value={audio.available ? "Yes" : "No"}
                  />
                  <AudioField
                    label="Classification"
                    value={formatClassification(audio.classification)}
                  />
                  <AudioField
                    label="Forensic Score"
                    value={
                      typeof audio.forensic_score === "number"
                        ? audio.forensic_score.toFixed(3)
                        : "Unavailable"
                    }
                  />
                  <AudioField
                    label="Hashes Match"
                    value={
                      typeof audioIntegrity?.hashes_match === "boolean"
                        ? audioIntegrity.hashes_match
                          ? "Yes"
                          : "No"
                        : "Unavailable"
                    }
                  />
                </div>
              </div>

              <div>
                <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                  Signal Features
                </div>
                <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                  <AudioField
                    label="Analysis Mode"
                    value={formatUnknown(audioFeatures?.analysis_mode)}
                  />
                  <AudioField
                    label="Duration"
                    value={formatUnknown(
                      audioMeta?.duration_seconds,
                      "seconds",
                    )}
                  />
                  <AudioField
                    label="Sample Rate"
                    value={formatUnknown(audioMeta?.sample_rate_hz, "Hz")}
                  />
                  <AudioField
                    label="Channels"
                    value={formatUnknown(audioMeta?.channels)}
                  />
                  <AudioField
                    label="Bit Depth"
                    value={formatUnknown(audioMeta?.sample_width_bits, "bits")}
                  />
                  <AudioField
                    label="Peak Level"
                    value={formatUnknown(audioFeatures?.peak_level)}
                  />
                  <AudioField
                    label="Dynamic Range"
                    value={formatUnknown(audioFeatures?.dynamic_range_db, "dB")}
                  />
                  <AudioField
                    label="Clipping Ratio"
                    value={formatUnknown(audioFeatures?.clipping_ratio)}
                  />
                  <AudioField
                    label="Zero Crossing Rate"
                    value={formatUnknown(audioFeatures?.zero_crossing_rate)}
                  />
                  <AudioField
                    label="Repetition Score"
                    value={formatUnknown(audioFeatures?.repetition_score)}
                  />
                  <AudioField
                    label="Spectral Flatness"
                    value={formatUnknown(audioFeatures?.spectral_flatness)}
                  />
                  <AudioField
                    label="Transcoded"
                    value={formatUnknown(
                      audioFeatures?.transcoded_for_analysis,
                    )}
                  />
                  <AudioField
                    label="FFmpeg Error"
                    value={formatUnknown(audioFeatures?.ffmpeg_transcode_error)}
                    mono
                  />
                </div>
              </div>

              {audio.error ? (
                <div className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
                  {audio.error}
                </div>
              ) : null}

              {extractedAudioSha256 && (
                <div>
                  <div className="mb-3 text-xs font-semibold uppercase tracking-[0.18em] text-slate-400">
                    Integrity
                  </div>
                  <div className="grid grid-cols-1 gap-4">
                    <AudioField
                      label="Extracted Audio SHA-256"
                      value={extractedAudioSha256}
                      mono
                    />
                    <AudioField
                      label="Extracted Audio MD5"
                      value={formatUnknown(extractedAudioMd5)}
                      mono
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-4">
              {audio.waveform_path ? (
                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                    Waveform Preview
                  </div>
                  <img
                    src={fixPath(audio.waveform_path)}
                    alt="Extracted audio waveform"
                    className="w-full rounded border border-slate-800 bg-slate-900"
                  />
                </div>
              ) : null}

              {typeof audioFeatures?.spectrogram_path === "string" ? (
                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                    Spectrogram Preview
                  </div>
                  <img
                    src={fixPath(audioFeatures.spectrogram_path)}
                    alt="Extracted audio spectrogram"
                    className="w-full rounded border border-slate-800 bg-slate-900"
                  />
                </div>
              ) : null}

              {audioSegments ? (
                <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                  <div className="mb-3 text-xs uppercase tracking-[0.16em] text-slate-500">
                    Segmented Signal Summary
                  </div>
                  <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <AudioField
                      label="Segment Size"
                      value={formatUnknown(
                        audioSegments.segment_duration_seconds,
                        "seconds",
                      )}
                    />
                    <AudioField
                      label="Segment Count"
                      value={formatUnknown(audioSegments.segment_count)}
                    />
                    <AudioField
                      label="RMS Std"
                      value={formatNestedStat(
                        audioSegments,
                        "rms_level",
                        "std",
                      )}
                    />
                    <AudioField
                      label="ZCR Std"
                      value={formatNestedStat(
                        audioSegments,
                        "zero_crossing_rate",
                        "std",
                      )}
                    />
                  </div>
                </div>
              ) : null}

              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
                  Audio Metadata
                </div>
                <pre className="max-h-[26rem] overflow-auto whitespace-pre-wrap break-words rounded-xl border border-slate-800 bg-slate-950/70 p-3 text-xs text-slate-300">
                  {JSON.stringify(audioMeta ?? {}, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </SectionCard>
    </CasePageScaffold>
  );
};
