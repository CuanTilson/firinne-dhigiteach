import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getVideoById } from "../services/api";
import type { VideoAnalysisDetail } from "../types";
import { fixPath } from "../constants";

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

  return (
    <div className="min-h-screen bg-white text-slate-900 px-8 py-10">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs uppercase tracking-widest text-slate-500">
              Forensic Report
            </div>
            <h1 className="text-3xl font-semibold mt-1">Video Analysis</h1>
            <p className="text-slate-500 mt-1">Case #{id}</p>
          </div>
          <button
            onClick={() => window.print()}
            className="px-4 py-2 rounded border border-slate-300 text-sm"
          >
            Print
          </button>
        </div>

        <section className="border border-slate-200 rounded-lg p-4 grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <div className="text-slate-500 text-xs uppercase">Filename</div>
            <div className="font-medium">{result.filename}</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs uppercase">Created</div>
            <div className="font-medium">
              {new Date(result.created_at).toISOString()}
            </div>
          </div>
          <div>
            <div className="text-slate-500 text-xs uppercase">Classification</div>
            <div className="font-medium">{result.classification}</div>
          </div>
          <div>
            <div className="text-slate-500 text-xs uppercase">Forensic Score</div>
            <div className="font-medium">{result.forensic_score.toFixed(3)}</div>
          </div>
        </section>

        <section className="border border-slate-200 rounded-lg p-4">
          <div className="text-xs uppercase text-slate-500 mb-3">
            Sample Frames
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {frames.map((frame) => (
              <div key={frame.frame_index} className="border border-slate-200 rounded">
                <img
                  src={fixPath(frame.saved_path)}
                  alt={`Frame ${frame.frame_index}`}
                  className="w-full h-32 object-cover"
                />
                <div className="p-2 text-xs text-slate-600">
                  Frame {frame.frame_index} - {frame.timestamp_sec.toFixed(2)}s
                </div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};
