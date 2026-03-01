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
import { ChevronLeft } from "lucide-react";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";
import { EvidenceStatusStrip } from "../components/EvidenceStatusStrip";
import { AnalysisProvenancePanel } from "../components/AnalysisProvenancePanel";

export const VideoDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<VideoAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
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
      <div className="min-h-screen flex items-center justify-center text-slate-400">
        Loading video analysis #{id}...
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-red-400 gap-4">
        <p>{error || "Record not found"}</p>
        <Link to="/history" className="text-cyan-400 hover:underline">
          Back to History
        </Link>
      </div>
    );
  }

  const hashes =
    result.video_metadata && typeof result.video_metadata === "object"
      ? (result.video_metadata as Record<string, unknown>)
      : null;
  const hashesBefore =
    hashes && typeof hashes.hashes_before === "object"
      ? (hashes.hashes_before as Record<string, string>)
      : null;
  const hashesAfter =
    hashes && typeof hashes.hashes_after === "object"
      ? (hashes.hashes_after as Record<string, string>)
      : null;
  const hashesCurrent =
    hashes && typeof hashes.hashes === "object"
      ? (hashes.hashes as Record<string, string>)
      : null;
  const audio = result.audio_analysis ?? null;
  const audioHashes =
    audio?.file_integrity && typeof audio.file_integrity === "object"
      ? audio.file_integrity
      : null;
  const audioMeta =
    audio?.audio_metadata && typeof audio.audio_metadata === "object"
      ? (audio.audio_metadata as Record<string, unknown>)
      : null;
  const audioFeatures =
    audio?.audio_features && typeof audio.audio_features === "object"
      ? (audio.audio_features as Record<string, unknown>)
      : null;
  const imageDetector =
    hashes && typeof hashes.image_detector === "object"
      ? (hashes.image_detector as Record<string, unknown>)
      : null;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to="/history"
          className="p-2 bg-slate-900 rounded-full hover:bg-slate-800 text-slate-300 transition-colors border border-slate-800"
        >
          <ChevronLeft size={20} />
        </Link>
        <span className="text-sm text-slate-400">Back to History</span>
      </div>

      <CaseHeader
        title="Video Analysis"
        caseId={id || ""}
        filename={result.filename}
        createdAt={result.created_at}
        printUrl={`#/print/videos/${id}`}
        hashes={{
          sha256: hashesCurrent?.sha256,
          md5: hashesCurrent?.md5,
          sha256_before: hashesBefore?.sha256,
          sha256_after: hashesAfter?.sha256,
          md5_before: hashesBefore?.md5,
          md5_after: hashesAfter?.md5,
        }}
      />

      <ChainOfCustody
        steps={[
          { label: "Upload received", timestamp: result.created_at, status: "complete" },
          { label: "Frame sampling completed", timestamp: result.created_at, status: "complete" },
          {
            label: audio?.available ? "Audio extraction completed" : "Audio extraction attempted",
            timestamp: result.created_at,
            status: audio?.available ? "complete" : "pending",
          },
          { label: "Analysis completed", timestamp: result.created_at, status: "complete" },
        ]}
      />

      <EvidenceStatusStrip
        items={[
          {
            label: "Visual Verdict",
            status: result.classification,
            tone: result.forensic_score >= 0.7 ? "warn" : "neutral",
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
              ? audio.classification ?? "Available"
              : "Unavailable",
            tone: audio?.error ? "bad" : audio?.available ? "good" : "warn",
            detail:
              audio?.error ||
              (typeof audio?.forensic_score === "number"
                ? `Score ${audio.forensic_score.toFixed(3)}`
                : undefined),
          },
          {
            label: "Video Integrity",
            status: hashesCurrent?.sha256 ? "Hashes recorded" : "Hashes unavailable",
            tone: hashesCurrent?.sha256 ? "good" : "warn",
            detail:
              hashesBefore?.sha256 && hashesAfter?.sha256
                ? "Before/after hashes stored"
                : undefined,
          },
        ]}
      />

      <div className="fd-card p-4">
        <div className="flex flex-wrap gap-4 text-sm text-slate-300">
          <div>
            <span className="text-slate-500">Filename:</span>{" "}
            <span
              className="truncate inline-block max-w-[260px] align-bottom"
              title={result.filename}
            >
              {result.filename}
            </span>
          </div>
          <div>
            <span className="text-slate-500">Frames:</span> {result.frame_count}
          </div>
          <div>
            <span className="text-slate-500">Score:</span>{" "}
            {result.forensic_score.toFixed(3)}
          </div>
          <div>
            <span className="text-slate-500">Classification:</span>{" "}
            {result.classification}
          </div>
        </div>
      </div>

      <div className="fd-card p-4">
        <div className="fd-section-title mb-3">Frames</div>
        <div className="flex gap-3 overflow-x-auto pb-2">
          {result.frames.map((frame, idx) => (
            <button
              key={`${frame.frame_index}-${idx}`}
              onClick={() => setSelectedIndex(idx)}
              className={`min-w-[160px] text-left flex flex-col gap-2 p-2 rounded-lg border transition-colors ${
                selectedIndex === idx
                  ? "border-cyan-400/50 bg-cyan-900/20"
                  : "border-slate-800 hover:bg-slate-900/60"
              }`}
            >
              <div className="h-24 w-full bg-slate-950 rounded overflow-hidden border border-slate-800">
                <img
                  src={fixPath(frame.saved_path)}
                  alt={`Frame ${frame.frame_index}`}
                  className="h-full w-full object-cover"
                />
              </div>
              <div className="text-xs text-slate-300">
                <div>Frame {frame.frame_index}</div>
                <div className="text-slate-500">
                  {frame.timestamp_sec.toFixed(2)}s
                </div>
                <div className="text-slate-400">
                  Score {frame.forensic_score.toFixed(2)}
                </div>
              </div>
            </button>
          ))}
        </div>
      </div>

      <div className="fd-card p-4">
        <div className="fd-section-title mb-3">Extracted Audio Evidence</div>
        {!audio ? (
          <div className="text-sm text-slate-400">
            No extracted-audio analysis was stored for this case.
          </div>
        ) : (
          <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-6">
            <div className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <AudioField label="Available" value={audio.available ? "Yes" : "No"} />
                <AudioField
                  label="Classification"
                  value={audio.classification ?? "Unavailable"}
                />
                <AudioField
                  label="Forensic score"
                  value={
                    typeof audio.forensic_score === "number"
                      ? audio.forensic_score.toFixed(3)
                      : "Unavailable"
                  }
                />
                <AudioField
                  label="Hashes match"
                  value={
                    typeof audioHashes?.hashes_match === "boolean"
                      ? audioHashes.hashes_match
                        ? "Yes"
                        : "No"
                      : "Unavailable"
                  }
                />
              </div>

              {audio.error ? (
                <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-200">
                  {audio.error}
                </div>
              ) : null}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <AudioField
                  label="Duration"
                  value={formatUnknown(audioMeta?.duration_seconds, "seconds")}
                />
                <AudioField
                  label="Sample rate"
                  value={formatUnknown(audioMeta?.sample_rate_hz, "Hz")}
                />
                <AudioField
                  label="Channels"
                  value={formatUnknown(audioMeta?.channels)}
                />
                <AudioField
                  label="Bit depth"
                  value={formatUnknown(audioMeta?.sample_width_bits, "bits")}
                />
                <AudioField
                  label="Peak amplitude"
                  value={formatUnknown(audioFeatures?.peak_amplitude)}
                />
                <AudioField
                  label="Clipping ratio"
                  value={formatUnknown(audioFeatures?.clipping_ratio)}
                />
              </div>

              {(audioHashes?.hashes_after?.sha256 || audioHashes?.hashes_before?.sha256) && (
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3 text-xs">
                  <div className="text-slate-500 uppercase tracking-wider mb-2">
                    Extracted Audio Hashes
                  </div>
                  <div className="space-y-2 text-slate-300">
                    <div>
                      <div className="text-slate-500">SHA-256</div>
                      <div className="font-mono break-all">
                        {audioHashes?.hashes_after?.sha256 ||
                          audioHashes?.hashes_before?.sha256 ||
                          "Unavailable"}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-500">MD5</div>
                      <div className="font-mono break-all">
                        {audioHashes?.hashes_after?.md5 ||
                          audioHashes?.hashes_before?.md5 ||
                          "Unavailable"}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="space-y-4">
              {audio.waveform_path ? (
                <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
                    Waveform Preview
                  </div>
                  <img
                    src={fixPath(audio.waveform_path)}
                    alt="Extracted audio waveform"
                    className="w-full rounded border border-slate-800 bg-slate-900"
                  />
                </div>
              ) : null}

              <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
                <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
                  Audio Metadata
                </div>
                <pre className="text-xs text-slate-300 whitespace-pre-wrap break-words">
                  {JSON.stringify(audioMeta ?? {}, null, 2)}
                </pre>
              </div>
            </div>
          </div>
        )}
      </div>

      <AppliedSettingsPanel settings={result.applied_settings} />

      <AnalysisProvenancePanel
        detector={imageDetector}
        fusionMode="rule_based_forensic_fusion"
      />

      <DecisionSummaryPanel
        verdict={result.classification}
        scoreLabel="Visual Aggregate Score"
        scoreValue={result.forensic_score.toFixed(3)}
        rationale={[
          `Frame count analysed: ${result.frame_count}`,
          `Selected frame score: ${selectedFrame ? selectedFrame.forensic_score.toFixed(3) : "Unavailable"}`,
          `Extracted audio classification: ${audio?.classification ?? "Unavailable"}`,
          `Extracted audio score: ${
            typeof audio?.forensic_score === "number"
              ? audio.forensic_score.toFixed(3)
              : "Unavailable"
          }`,
        ]}
        note="Current video decisions are based on sampled frame analysis, with extracted audio triage shown as a parallel evidence stream."
      />

      <div>
        {selectedFrame ? (
          <AnalysisDashboard result={selectedFrame} />
        ) : (
          <div className="text-slate-400">
            No frames available for analysis.
          </div>
        )}
      </div>
    </div>
  );
};

const AudioField = ({ label, value }: { label: string; value: string }) => (
  <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
    <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
    <div className="mt-1 text-slate-200">{value}</div>
  </div>
);

const formatUnknown = (value: AudioValue, suffix?: string) => {
  if (typeof value === "number") {
    return suffix ? `${value} ${suffix}` : String(value);
  }
  if (typeof value === "string" && value.trim()) {
    return suffix ? `${value} ${suffix}` : value;
  }
  return "Unavailable";
};

type AudioValue = AudioAnalysisSummary["forensic_score"] | string | unknown;
