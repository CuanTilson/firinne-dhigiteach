import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { fixPath } from "../constants";
import { sanitizeMetadata } from "../utils/metadata";
import { ArrowLeft } from "lucide-react";

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

  const safeMetadata = sanitizeMetadata(result.raw_metadata ?? {});
  const sha256 =
    result.file_integrity?.hashes_after?.sha256 ||
    result.file_integrity?.hashes_before?.sha256 ||
    result.file_integrity?.hashes?.sha256 ||
    "Not available";
  const md5 =
    result.file_integrity?.hashes_after?.md5 ||
    result.file_integrity?.hashes_before?.md5 ||
    result.file_integrity?.hashes?.md5 ||
    "Not available";
  const findings = result.metadata_anomalies?.findings ?? [];

  const qualityBadges: string[] = [];
  if (result.c2pa?.has_c2pa) qualityBadges.push("C2PA Present");
  if (result.ai_watermark?.stable_diffusion_detected) {
    qualityBadges.push("Watermark Signal Detected");
  }
  if ((result.jpeg_qtables?.inconsistency_score ?? 0) > 0.15) {
    qualityBadges.push("JPEG Recompression Signal");
  }
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
              to={`/records/${id}`}
              className="inline-flex items-center gap-2 text-xs text-slate-500 hover:text-slate-800 mb-2"
            >
              <ArrowLeft size={14} />
              Back to Case
            </Link>
            <div className="text-xs uppercase tracking-[0.2em] text-slate-500">
              Forensic Report
            </div>
            <h1 className="text-2xl font-semibold mt-1">
              Firinne Dhigiteach - Image Evidence Assessment
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
            <div className="flex flex-wrap gap-2 pt-1">
              {qualityBadges.length === 0 ? (
                <span className="px-2 py-1 rounded border border-slate-300 text-xs text-slate-600">
                  No dominant quality flags
                </span>
              ) : (
                qualityBadges.map((badge) => (
                  <span
                    key={badge}
                    className="px-2 py-1 rounded border border-slate-300 text-xs text-slate-700"
                  >
                    {badge}
                  </span>
                ))
              )}
            </div>
          </div>

          <div className="border border-slate-200 rounded-lg p-4 space-y-3">
            <div className="text-xs uppercase tracking-wider text-slate-500">
              Integrity Snapshot
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">SHA-256</div>
              <div className="font-mono text-xs break-all">{sha256}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">MD5</div>
              <div className="font-mono text-xs break-all">{md5}</div>
            </div>
            <div>
              <div className="text-slate-500 text-xs uppercase">JPEG Structure</div>
              <div className="font-medium">
                {result.file_integrity?.jpeg_structure?.valid_jpeg
                  ? "Valid"
                  : "Issues Detected"}
              </div>
            </div>
          </div>
        </section>

        <section className="border border-slate-200 rounded-lg p-4 text-sm">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
            Analyst Findings
          </div>
          {findings.length === 0 ? (
            <p className="text-slate-700">No suspicious metadata patterns flagged.</p>
          ) : (
            <ul className="space-y-1 text-slate-800 list-disc pl-5">
              {findings.slice(0, 8).map((item, index) => (
                <li key={`${item}-${index}`}>{item}</li>
              ))}
            </ul>
          )}
        </section>

        <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
              Visual Exhibits
            </div>
            <div className="grid grid-cols-1 gap-4">
              {result.saved_path && (
                <figure className="border border-slate-200 rounded-lg p-3">
                  <figcaption className="text-xs uppercase text-slate-500 mb-2">
                    Exhibit A - Original
                  </figcaption>
                  <img
                    src={fixPath(result.saved_path)}
                    alt="Original"
                    className="w-full object-contain max-h-[320px]"
                    crossOrigin="anonymous"
                  />
                </figure>
              )}
              {result.gradcam_heatmap && (
                <figure className="border border-slate-200 rounded-lg p-3">
                  <figcaption className="text-xs uppercase text-slate-500 mb-2">
                    Exhibit B - GradCAM
                  </figcaption>
                  <img
                    src={fixPath(result.gradcam_heatmap)}
                    alt="GradCAM"
                    className="w-full object-contain max-h-[320px]"
                    crossOrigin="anonymous"
                  />
                </figure>
              )}
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4">
            {result.ela_heatmap && (
              <figure className="border border-slate-200 rounded-lg p-3">
                <figcaption className="text-xs uppercase text-slate-500 mb-2">
                  Exhibit C - ELA
                </figcaption>
                <img
                  src={fixPath(result.ela_heatmap)}
                  alt="ELA"
                  className="w-full object-contain max-h-[320px]"
                  crossOrigin="anonymous"
                />
              </figure>
            )}
            {result.noise_residual?.noise_heatmap_path && (
              <figure className="border border-slate-200 rounded-lg p-3">
                <figcaption className="text-xs uppercase text-slate-500 mb-2">
                  Exhibit D - Noise Residual
                </figcaption>
                <img
                  src={fixPath(result.noise_residual.noise_heatmap_path)}
                  alt="Noise"
                  className="w-full object-contain max-h-[320px]"
                  crossOrigin="anonymous"
                />
              </figure>
            )}
          </div>
        </section>

        <section className="border border-slate-200 rounded-lg p-4">
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
            Metadata Appendix
          </div>
          <pre
            data-pdf-append-text
            className="text-xs whitespace-pre-wrap max-h-[420px] overflow-auto bg-slate-50 border border-slate-200 rounded p-3"
          >
            {JSON.stringify(safeMetadata, null, 2)}
          </pre>
          <div className="text-[11px] text-slate-500 mt-2">
            Large binary fields are omitted for readability.
          </div>
        </section>

        <section className="text-[11px] text-slate-500 border-t border-slate-200 pt-3">
          This report is decision-support evidence and should be interpreted with
          contextual forensic review.
        </section>
      </div>
    </div>
  );
};
