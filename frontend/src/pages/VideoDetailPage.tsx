import React, { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getVideoById } from "../services/api";
import type { VideoAnalysisDetail, VideoFrameResult } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { fixPath } from "../constants";
import { ChevronLeft } from "lucide-react";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";

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
          { label: "Analysis completed", timestamp: result.created_at, status: "complete" },
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
