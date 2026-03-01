import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getVideoById } from "../services/api";
import type { VideoAnalysisDetail } from "../types";
import { fixPath } from "../constants";
import { ArrowLeft } from "lucide-react";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";

export const PrintVideoPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<VideoAnalysisDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-500">
        Loading report...
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-red-500 gap-4">
        <p>{error || "Record not found"}</p>
        <Link to="/history" className="text-cyan-600 hover:underline">
          Back to History
        </Link>
      </div>
    );
  }

  const frames = result.frames?.slice(0, 8) || [];
  const videoMeta =
    result.video_metadata && typeof result.video_metadata === "object"
      ? (result.video_metadata as Record<string, unknown>)
      : null;

  const hashes =
    videoMeta && typeof videoMeta.hashes === "object"
      ? (videoMeta.hashes as Record<string, string>)
      : null;
  const audio = result.audio_analysis ?? null;
  const audioMeta =
    audio?.audio_metadata && typeof audio.audio_metadata === "object"
      ? (audio.audio_metadata as Record<string, unknown>)
      : null;
  const audioFeatures =
    audio?.audio_features && typeof audio.audio_features === "object"
      ? (audio.audio_features as Record<string, unknown>)
      : null;
  const audioHashes = audio?.file_integrity ?? null;
  const handleGeneratePdf = () => {
    window.print();
  };

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900 px-4 py-8 print:bg-white print:p-0">
      <div
        className="max-w-5xl mx-auto space-y-6 bg-white border border-slate-200 rounded-xl p-6 print:border-0 print:rounded-none print:p-0"
      >
        <div className="flex items-start justify-between border-b border-slate-200 pb-4">
          <div>
            <Link
              to={`/videos/${id}`}
              className="inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-800 mb-2"
            >
              <ArrowLeft size={14} />
              Back to Case
            </Link>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Forensic Report
            </div>
            <h1 className="text-2xl font-semibold mt-1">
              Firinne Dhigiteach - Video Evidence Assessment
            </h1>
            <p className="text-slate-500 mt-1">
              Case #{id} | Generated {new Date(result.created_at).toUTCString()}
            </p>
          </div>
          <button
            onClick={handleGeneratePdf}
            className="px-4 py-2 rounded border border-slate-300 text-sm print:hidden"
          >
            Print / Save PDF
          </button>
        </div>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="text-xs uppercase tracking-wider text-slate-500">
              Case Metadata
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">Filename</div>
              <div className="font-medium break-all">{result.filename}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">Classification</div>
              <div className="font-medium">{result.classification}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">Forensic Score</div>
              <div className="font-medium">{result.forensic_score.toFixed(3)}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">Frame Count</div>
              <div className="font-medium">{result.frame_count}</div>
            </div>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="text-xs uppercase tracking-wider text-slate-500">
              Integrity Snapshot
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">SHA-256</div>
              <div className="font-mono text-xs break-all">
                {hashes?.sha256 || "Not available"}
              </div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">MD5</div>
              <div className="font-mono text-xs break-all">
                {hashes?.md5 || "Not available"}
              </div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">Model Note</div>
              <div className="text-slate-700">
                Classification is based on sampled frame analysis.
              </div>
            </div>
          </div>
        </section>

        <section className="border border-slate-200 rounded-lg p-4 space-y-4">
          <div className="text-xs uppercase tracking-wider text-slate-500">
            Extracted Audio Evidence
          </div>

          {!audio ? (
            <p className="text-sm text-slate-700">
              No extracted-audio analysis was stored for this case.
            </p>
          ) : (
            <>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-slate-500 text-xs uppercase">Available</div>
                  <div className="font-medium">{audio.available ? "Yes" : "No"}</div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase">Classification</div>
                  <div className="font-medium">
                    {audio.classification || "Unavailable"}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase">Forensic Score</div>
                  <div className="font-medium">
                    {typeof audio.forensic_score === "number"
                      ? audio.forensic_score.toFixed(3)
                      : "Unavailable"}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase">Hashes Match</div>
                  <div className="font-medium">
                    {typeof audioHashes?.hashes_match === "boolean"
                      ? audioHashes.hashes_match
                        ? "Yes"
                        : "No"
                      : "Unavailable"}
                  </div>
                </div>
              </div>

              {audio.error ? (
                <div className="border border-amber-200 bg-amber-50 text-amber-900 rounded-lg p-3 text-sm">
                  {audio.error}
                </div>
              ) : null}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                <div>
                  <div className="text-slate-500 text-xs uppercase">Duration</div>
                  <div className="font-medium">
                    {formatUnknown(audioMeta?.duration_seconds, "seconds")}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase">Sample Rate</div>
                  <div className="font-medium">
                    {formatUnknown(audioMeta?.sample_rate_hz, "Hz")}
                  </div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase">Channels</div>
                  <div className="font-medium">{formatUnknown(audioMeta?.channels)}</div>
                </div>
                <div>
                  <div className="text-slate-500 text-xs uppercase">Peak Amplitude</div>
                  <div className="font-medium">
                    {formatUnknown(audioFeatures?.peak_amplitude)}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 items-start">
                {audio.waveform_path ? (
                  <figure className="border border-slate-200 rounded-lg p-3">
                    <figcaption className="text-xs uppercase text-slate-500 mb-2">
                      Exhibit - Extracted Audio Waveform
                    </figcaption>
                    <img
                      src={fixPath(audio.waveform_path)}
                      alt="Extracted audio waveform"
                      className="w-full object-contain"
                      crossOrigin="anonymous"
                    />
                  </figure>
                ) : null}

                <div className="border border-slate-200 rounded-lg p-3">
                  <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
                    Audio Hashes
                  </div>
                  <div className="space-y-3 text-sm">
                    <div>
                      <div className="text-slate-500 text-xs uppercase">SHA-256</div>
                      <div className="font-mono text-xs break-all">
                        {audioHashes?.hashes_after?.sha256 ||
                          audioHashes?.hashes_before?.sha256 ||
                          "Unavailable"}
                      </div>
                    </div>
                    <div>
                      <div className="text-slate-500 text-xs uppercase">MD5</div>
                      <div className="font-mono text-xs break-all">
                        {audioHashes?.hashes_after?.md5 ||
                          audioHashes?.hashes_before?.md5 ||
                          "Unavailable"}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}
        </section>

        <DecisionSummaryPanel
          title="Decision Summary"
          verdict={result.classification}
          scoreLabel="Forensic Score"
          scoreValue={result.forensic_score.toFixed(3)}
          rationale={[
            `Frame sample count: ${result.frame_count}`,
            "Primary decision is based on sampled frame analysis.",
            audio?.available
              ? `Extracted audio classification: ${audio.classification || "Unavailable"}`
              : "No extracted-audio evidence was available.",
            hashes?.sha256 ? "Integrity hashes were recorded for the source video." : "Integrity hashes were not fully available.",
          ]}
          note="Visual and extracted-audio evidence streams should be interpreted together rather than as independent final determinations."
        />

        <section className="border border-slate-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-3">
            Sample Frame Exhibits
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {frames.map((frame) => (
              <figure
                key={frame.frame_index}
                className="border border-slate-200 rounded-lg overflow-hidden"
              >
                <img
                  src={fixPath(frame.saved_path)}
                  alt={`Frame ${frame.frame_index}`}
                  className="w-full h-28 object-cover"
                  crossOrigin="anonymous"
                />
                <figcaption className="p-2 text-xs text-slate-700">
                  Frame {frame.frame_index} | {frame.timestamp_sec.toFixed(2)}s
                </figcaption>
              </figure>
            ))}
          </div>
        </section>

        <AppliedSettingsPanel settings={result.applied_settings} compact />

        <section className="text-[11px] text-slate-500 border-t border-slate-200 pt-3">
          This report is decision-support evidence and should be interpreted with
          contextual forensic review.
        </section>
      </div>
    </div>
  );
};

const formatUnknown = (value: unknown, suffix?: string) => {
  if (typeof value === "number") {
    return suffix ? `${value} ${suffix}` : String(value);
  }
  if (typeof value === "string" && value.trim()) {
    return suffix ? `${value} ${suffix}` : value;
  }
  return "Unavailable";
};
