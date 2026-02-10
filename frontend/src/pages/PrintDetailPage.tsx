import React, { useEffect, useMemo, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { fixPath } from "../constants";
import { sanitizeMetadata } from "../utils/metadata";

export const PrintDetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) return;
      try {
        const data = await getRecordById(Number(id));
        setResult(data);
      } catch {
        setError("Could not load analysis details.");
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

  const safeMetadata = useMemo(
    () => sanitizeMetadata(result.raw_metadata ?? {}),
    [result.raw_metadata]
  );

  return (
    <div className="min-h-screen bg-white text-slate-900 px-8 py-10">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-start justify-between">
          <div>
            <div className="text-xs uppercase tracking-widest text-slate-500">
              Forensic Report
            </div>
            <h1 className="text-3xl font-semibold mt-1">Image Analysis</h1>
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

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.saved_path && (
            <div className="border border-slate-200 rounded-lg p-3">
              <div className="text-xs uppercase text-slate-500 mb-2">
                Original
              </div>
              <img
                src={fixPath(result.saved_path)}
                alt="Original"
                className="w-full object-contain max-h-[320px]"
              />
            </div>
          )}
          {result.gradcam_heatmap && (
            <div className="border border-slate-200 rounded-lg p-3">
              <div className="text-xs uppercase text-slate-500 mb-2">
                GradCAM
              </div>
              <img
                src={fixPath(result.gradcam_heatmap)}
                alt="GradCAM"
                className="w-full object-contain max-h-[320px]"
              />
            </div>
          )}
          {result.ela_heatmap && (
            <div className="border border-slate-200 rounded-lg p-3">
              <div className="text-xs uppercase text-slate-500 mb-2">ELA</div>
              <img
                src={fixPath(result.ela_heatmap)}
                alt="ELA"
                className="w-full object-contain max-h-[320px]"
              />
            </div>
          )}
          {result.noise_residual?.noise_heatmap_path && (
            <div className="border border-slate-200 rounded-lg p-3">
              <div className="text-xs uppercase text-slate-500 mb-2">Noise</div>
              <img
                src={fixPath(result.noise_residual.noise_heatmap_path)}
                alt="Noise"
                className="w-full object-contain max-h-[320px]"
              />
            </div>
          )}
        </section>

        <section className="border border-slate-200 rounded-lg p-4">
          <div className="text-xs uppercase text-slate-500 mb-2">
            Raw Metadata
          </div>
          <pre className="text-xs whitespace-pre-wrap">
            {JSON.stringify(safeMetadata, null, 2)}
          </pre>
          <div className="text-[11px] text-slate-500 mt-2">
            Large binary fields are omitted for readability.
          </div>
        </section>
      </div>
    </div>
  );
};
