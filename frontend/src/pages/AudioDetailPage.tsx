import React, { useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ChevronLeft } from "lucide-react";
import { getAudioById } from "../services/api";
import type { AudioAnalysisDetail } from "../types";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";
import { fixPath } from "../constants";
import { EvidenceStatusStrip } from "../components/EvidenceStatusStrip";

export const AudioDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AudioAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
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
    return <div className="min-h-screen flex items-center justify-center text-slate-400">Loading audio analysis #{id}...</div>;
  }

  if (error || !result) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-red-400 gap-4">
        <p>{error || "Record not found"}</p>
        <Link to="/audio-history" className="text-cyan-400 hover:underline">
          Back to Audio History
        </Link>
      </div>
    );
  }

  const hashes = result.file_integrity?.hashes;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to="/audio-history"
          className="p-2 bg-slate-900 rounded-full hover:bg-slate-800 text-slate-300 transition-colors border border-slate-800"
        >
          <ChevronLeft size={20} />
        </Link>
        <span className="text-sm text-slate-400">Back to Audio History</span>
      </div>

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

      <ChainOfCustody
        steps={[
          { label: "Upload received", timestamp: result.created_at, status: "complete" },
          { label: "Audio triage completed", timestamp: result.created_at, status: "complete" },
        ]}
      />

      <EvidenceStatusStrip
        items={[
          {
            label: "Audio Verdict",
            status: result.classification,
            tone: result.forensic_score >= 0.7 ? "warn" : "neutral",
            detail: `Score ${result.forensic_score.toFixed(3)}`,
          },
          {
            label: "Analysis Mode",
            status: String(result.audio_features?.analysis_mode || "unknown"),
            detail: formatUnknown(result.audio_metadata?.duration_seconds, "seconds"),
          },
          {
            label: "Signal Metadata",
            status: formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz"),
            tone: result.audio_metadata?.sample_rate_hz ? "good" : "warn",
            detail: `${formatUnknown(result.audio_metadata?.channels)} channels`,
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

      <DecisionSummaryPanel
        verdict={result.classification}
        scoreLabel="Audio Forensic Score"
        scoreValue={result.forensic_score.toFixed(3)}
        rationale={[
          `Analysis mode: ${String(result.audio_features?.analysis_mode || "unknown")}`,
          `Duration: ${String(result.audio_metadata?.duration_seconds ?? "Unavailable")} seconds`,
          `Sample rate: ${String(result.audio_metadata?.sample_rate_hz ?? "Unavailable")} Hz`,
          ...(findings.length ? findings.slice(0, 3) : ["No dominant audio findings recorded."]),
        ]}
        note="This is an audio triage result and should be read with contextual forensic review."
      />

      <div className="grid grid-cols-1 xl:grid-cols-[1.1fr_0.9fr] gap-6">
        <div className="fd-card p-4 space-y-4">
          <div className="fd-section-title">Audio Evidence</div>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <AudioField label="Classification" value={result.classification} />
            <AudioField label="Forensic Score" value={result.forensic_score.toFixed(3)} />
            <AudioField label="Duration" value={formatUnknown(result.audio_metadata?.duration_seconds, "seconds")} />
            <AudioField label="Sample Rate" value={formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")} />
            <AudioField label="Channels" value={formatUnknown(result.audio_metadata?.channels)} />
            <AudioField label="Hashes Match" value={formatUnknown(result.file_integrity?.hashes_match)} />
          </div>
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">Findings</div>
            <ul className="space-y-2 text-sm text-slate-300">
              {(findings.length ? findings : ["No explicit findings recorded."]).map((item, index) => (
                <li key={`${item}-${index}`}>- {item}</li>
              ))}
            </ul>
          </div>
        </div>

        <div className="fd-card p-4 space-y-4">
          <div className="fd-section-title">Waveform</div>
          {result.waveform_path ? (
            <img
              src={fixPath(result.waveform_path)}
              alt="Waveform preview"
              className="w-full rounded border border-slate-800 bg-slate-950"
            />
          ) : (
            <div className="text-sm text-slate-400">No waveform preview available for this file.</div>
          )}
        </div>
      </div>

      <AppliedSettingsPanel settings={result.applied_settings} />
    </div>
  );
};

const AudioField = ({ label, value }: { label: string; value: string }) => (
  <div className="rounded-lg border border-slate-800 bg-slate-950/60 p-3">
    <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
    <div className="mt-1 text-slate-200">{value}</div>
  </div>
);

const formatUnknown = (value: unknown, suffix?: string) => {
  if (typeof value === "number") return suffix ? `${value} ${suffix}` : String(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" && value.trim()) return suffix ? `${value} ${suffix}` : value;
  return "Unavailable";
};
