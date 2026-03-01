import React, { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import { getAudioById } from "../services/api";
import type { AudioAnalysisDetail } from "../types";
import { fixPath } from "../constants";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";

export const PrintAudioPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AudioAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
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
    return <div className="min-h-screen flex items-center justify-center text-slate-500">Loading report...</div>;
  }

  if (error || !result) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-red-500 gap-4">
        <p>{error || "Record not found"}</p>
        <Link to="/audio-history" className="text-cyan-600 hover:underline">Back to Audio History</Link>
      </div>
    );
  }

  const findings = Array.isArray(result.audio_features?.findings)
    ? result.audio_features?.findings.map(String)
    : [];

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 px-4 py-8 print:bg-white print:p-0">
      <div className="max-w-5xl mx-auto space-y-6 bg-white border border-slate-200 rounded-xl p-6 print:border-0 print:rounded-none print:p-0">
        <div className="flex items-start justify-between border-b border-slate-200 pb-4">
          <div>
            <Link to={`/audio/${id}`} className="inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-800 mb-2">
              <ArrowLeft size={14} />
              Back to Case
            </Link>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">Forensic Report</div>
            <h1 className="text-2xl font-semibold mt-1">Firinne Dhigiteach - Audio Evidence Assessment</h1>
            <p className="text-slate-500 mt-1">Case #{id} | Generated {new Date(result.created_at).toUTCString()}</p>
          </div>
          <button onClick={() => window.print()} className="px-4 py-2 rounded border border-slate-300 text-sm print:hidden">
            Print / Save PDF
          </button>
        </div>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="text-xs uppercase tracking-wider text-slate-500">Case Metadata</div>
            <KV label="Filename" value={result.filename} />
            <KV label="Classification" value={result.classification} />
            <KV label="Forensic Score" value={result.forensic_score.toFixed(3)} />
            <KV label="Analysis Mode" value={String(result.audio_features?.analysis_mode || "unknown")} />
          </div>

          <div className="border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="text-xs uppercase tracking-wider text-slate-500">Integrity Snapshot</div>
            <KV label="SHA-256" value={result.file_integrity?.hashes_after?.sha256 || result.file_integrity?.hashes_before?.sha256 || "Unavailable"} mono />
            <KV label="MD5" value={result.file_integrity?.hashes_after?.md5 || result.file_integrity?.hashes_before?.md5 || "Unavailable"} mono />
            <KV label="Hashes Match" value={formatUnknown(result.file_integrity?.hashes_match)} />
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="border border-slate-200 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">Audio Findings</div>
            <ul className="space-y-1 text-sm text-slate-800 list-disc pl-5">
              {(findings.length ? findings : ["No explicit findings recorded."]).map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="text-xs uppercase tracking-wider text-slate-500">Signal Metadata</div>
            <KV label="Duration" value={formatUnknown(result.audio_metadata?.duration_seconds, "seconds")} />
            <KV label="Sample Rate" value={formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")} />
            <KV label="Channels" value={formatUnknown(result.audio_metadata?.channels)} />
            <KV label="Peak Amplitude" value={formatUnknown(result.audio_features?.peak_amplitude)} />
          </div>
        </section>

        <DecisionSummaryPanel
          title="Decision Summary"
          verdict={result.classification}
          scoreLabel="Audio Forensic Score"
          scoreValue={result.forensic_score.toFixed(3)}
          rationale={[
            `Analysis mode: ${String(result.audio_features?.analysis_mode || "unknown")}`,
            `Duration: ${formatUnknown(result.audio_metadata?.duration_seconds, "seconds")}`,
            `Sample rate: ${formatUnknown(result.audio_metadata?.sample_rate_hz, "Hz")}`,
            ...(findings.length ? findings.slice(0, 2) : ["No explicit findings recorded."]),
          ]}
          note="This audio result is a triage output and should be interpreted with surrounding case context."
        />

        {result.waveform_path ? (
          <section className="border border-slate-200 rounded-lg p-4">
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">Waveform Exhibit</div>
            <img
              src={fixPath(result.waveform_path)}
              alt="Waveform preview"
              className="w-full object-contain"
              crossOrigin="anonymous"
            />
          </section>
        ) : null}

        <AppliedSettingsPanel settings={result.applied_settings} compact />

        <section className="text-[11px] text-slate-500 border-t border-slate-200 pt-3">
          This report is decision-support evidence and should be interpreted with contextual forensic review.
        </section>
      </div>
    </div>
  );
};

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
    <div className="text-slate-500 text-xs uppercase">{label}</div>
    <div className={`${mono ? "font-mono text-xs" : "font-medium"} break-all`}>{value || "Unavailable"}</div>
  </div>
);

const formatUnknown = (value: unknown, suffix?: string) => {
  if (typeof value === "number") return suffix ? `${value} ${suffix}` : String(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string" && value.trim()) return suffix ? `${value} ${suffix}` : value;
  return "Unavailable";
};
