import React, { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getVideoById } from "../services/api";
import type { VideoAnalysisDetail, VideoFrameResult } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { fixPath } from "../constants";
import { ChevronLeft, Film } from "lucide-react";

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

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6 flex items-center gap-4">
        <Link
          to="/history"
          className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 text-slate-300 transition-colors"
        >
          <ChevronLeft size={20} />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-2">
            <Film size={20} className="text-cyan-400" /> Video Case #{id}
          </h1>
          <p className="text-slate-500 text-sm">
            Analysed on {new Date(result.created_at).toLocaleString()}
          </p>
        </div>
      </div>

      <div className="bg-slate-800 rounded-lg border border-slate-700 p-4 mb-6">
        <div className="flex flex-wrap gap-4 text-sm text-slate-300">
          <div>
            <span className="text-slate-500">Filename:</span> {result.filename}
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

      <div className="grid grid-cols-1 lg:grid-cols-[320px_1fr] gap-6">
        <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
          <h3 className="text-slate-200 font-semibold mb-3">Frames</h3>
          <div className="space-y-2 max-h-[600px] overflow-y-auto pr-2 custom-scrollbar">
            {result.frames.map((frame, idx) => (
              <button
                key={`${frame.frame_index}-${idx}`}
                onClick={() => setSelectedIndex(idx)}
                className={`w-full text-left flex items-center gap-3 p-2 rounded-lg border transition-colors ${
                  selectedIndex === idx
                    ? "border-cyan-500/50 bg-cyan-900/20"
                    : "border-slate-700 hover:bg-slate-700/40"
                }`}
              >
                <div className="h-14 w-20 bg-slate-900 rounded overflow-hidden border border-slate-700 shrink-0">
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
    </div>
  );
};
