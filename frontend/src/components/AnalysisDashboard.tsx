import React, { useMemo, useState } from "react";
import type { AnalysisResult } from "../types";
import { HeatmapViewer } from "./HeatmapViewer";
import { ForensicScoreCard } from "./ForensicScoreCard";
import { fixPath } from "../constants";
import {
  AlertTriangle,
  CheckCircle,
  FileDigit,
  Fingerprint,
  XCircle,
} from "lucide-react";
import { sanitizeMetadata } from "../utils/metadata";
import { C2PAProvenanceSummary } from "./C2PAProvenanceSummary";

interface Props {
  result: AnalysisResult;
}

const STD_LUMA = [
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99],
];

const getTabClass = (active: boolean) =>
  `flex-1 py-3 text-sm font-medium border-b-2 transition ${
    active
      ? "border-cyan-500 bg-slate-700/50 text-cyan-400"
      : "border-transparent text-slate-400"
  } focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-inset`;

export const AnalysisDashboard: React.FC<Props> = ({ result }) => {
  const [activeTab, setActiveTab] = useState<"metadata" | "c2pa" | "jpeg">(
    "metadata",
  );

  const qtables = result.jpeg_qtables?.qtables;
  const firstTable = qtables
    ? (Object.values(qtables)[0] as number[] | undefined)
    : undefined;

  const tableGrid: number[][] | null =
    firstTable && firstTable.length >= 64
      ? Array.from({ length: 8 }, (_, r) => firstTable.slice(r * 8, r * 8 + 8))
      : null;

  const safeMetadata = useMemo(
    () => sanitizeMetadata(result.raw_metadata ?? {}),
    [result.raw_metadata],
  );

  return (
    <div className="animate-fade-in space-y-6">
      <div>
        <div className="mb-3 text-[11px] uppercase tracking-[0.2em] text-slate-500">
          Media Evidence
        </div>
        <HeatmapViewer
          elaUrl={result.ela_heatmap ? fixPath(result.ela_heatmap) : undefined}
          gradCamUrl={
            result.gradcam_heatmap ? fixPath(result.gradcam_heatmap) : undefined
          }
          noiseUrl={
            result.noise_residual?.noise_heatmap_path
              ? fixPath(result.noise_residual.noise_heatmap_path)
              : undefined
          }
          jpegQualityUrl={
            result.jpeg_qtables?.double_compression?.jpeg_quality_heatmap_path
              ? fixPath(
                  result.jpeg_qtables.double_compression
                    .jpeg_quality_heatmap_path,
                )
              : undefined
          }
          originalUrl={
            result.saved_path ? fixPath(result.saved_path) : undefined
          }
        />
      </div>

      <ForensicScoreCard data={result} />

      <div>
        <div className="mb-3 text-[11px] uppercase tracking-[0.2em] text-slate-500">
          Evidence Panels
        </div>

        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <Fingerprint className="text-cyan-400" size={20} />
                  <h4 className="font-semibold text-slate-200">
                    Invisible Watermark (optional)
                  </h4>
                </div>

                {result.ai_watermark?.stable_diffusion_detected ? (
                  <div className="flex items-center gap-2 text-sm font-medium text-red-400">
                    <XCircle size={16} /> Detected
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-sm font-medium text-green-400">
                    <CheckCircle size={16} /> Not Detected
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
                <div className="mb-2 flex items-center gap-2">
                  <FileDigit className="text-cyan-400" size={20} />
                  <h4 className="font-semibold text-slate-200">
                    File Integrity
                  </h4>
                </div>

                {result.file_integrity?.jpeg_structure?.valid_jpeg ? (
                  <div className="flex items-center gap-2 text-sm font-medium text-green-400">
                    <CheckCircle size={16} /> Valid
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-sm font-medium text-red-400">
                    <XCircle size={16} /> Issues Found
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="rounded-2xl border border-slate-800 bg-slate-950/60 p-4">
            <div className="mb-3 flex items-center justify-between gap-3">
              <h4 className="flex items-center gap-2 font-semibold text-slate-200">
                <AlertTriangle className="text-yellow-500" size={18} />
                Metadata Anomalies
              </h4>
              <span className="rounded-full border border-slate-700 bg-slate-900/70 px-2.5 py-1 text-xs text-slate-400">
                Score{" "}
                {(result.metadata_anomalies?.anomaly_score ?? 0).toFixed(3)}
              </span>
            </div>

            {result.metadata_anomalies?.findings?.length === 0 ? (
              <p className="text-sm italic text-slate-500">
                No suspicious metadata patterns found.
              </p>
            ) : (
              <ul className="space-y-2">
                {result.metadata_anomalies?.findings?.map(
                  (anom: string, idx: number) => (
                    <li
                      key={idx}
                      className="rounded border border-red-500/20 bg-red-500/10 px-3 py-2 text-sm text-red-200"
                    >
                      {anom}
                    </li>
                  ),
                )}
              </ul>
            )}

            {result.metadata_anomalies?.camera_consistency ? (
              <div className="mt-4 text-xs text-slate-400">
                <div>
                  Camera Make:{" "}
                  {result.metadata_anomalies.camera_consistency.make ||
                    "Unknown"}
                </div>
                <div>
                  Camera Model:{" "}
                  {result.metadata_anomalies.camera_consistency.model ||
                    "Unknown"}
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      <div>
        <div className="mb-3 text-[11px] uppercase tracking-[0.2em] text-slate-500">
          Forensic Details
        </div>

        <div className="overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/55 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
          <div
            className="flex border-b border-slate-700"
            role="tablist"
            aria-label="Forensic detail panels"
          >
            <button
              onClick={() => setActiveTab("metadata")}
              id="tab-metadata"
              role="tab"
              aria-selected={activeTab === "metadata"}
              aria-controls="panel-metadata"
              className={getTabClass(activeTab === "metadata")}
            >
              Raw Metadata
            </button>

            <button
              onClick={() => setActiveTab("c2pa")}
              id="tab-c2pa"
              role="tab"
              aria-selected={activeTab === "c2pa"}
              aria-controls="panel-c2pa"
              className={getTabClass(activeTab === "c2pa")}
            >
              C2PA Info
            </button>

            <button
              onClick={() => setActiveTab("jpeg")}
              id="tab-jpeg"
              role="tab"
              aria-selected={activeTab === "jpeg"}
              aria-controls="panel-jpeg"
              className={getTabClass(activeTab === "jpeg")}
            >
              JPEG Tables
            </button>
          </div>

          <div className="p-4">
            {activeTab === "metadata" && (
              <div
                id="panel-metadata"
                role="tabpanel"
                aria-labelledby="tab-metadata"
                className="space-y-3"
              >
                <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap rounded-2xl border border-slate-800 bg-slate-950/70 p-4 font-mono text-xs text-slate-300">
                  {JSON.stringify(safeMetadata, null, 2)}
                </pre>
                <div className="text-xs text-slate-500">
                  Large binary fields are omitted for readability.
                </div>
              </div>
            )}

            {activeTab === "c2pa" && (
              <div
                id="panel-c2pa"
                role="tabpanel"
                aria-labelledby="tab-c2pa"
                className="space-y-4"
              >
                <C2PAProvenanceSummary c2pa={result.c2pa} />
                <details className="text-xs text-slate-400">
                  <summary className="cursor-pointer">Raw C2PA JSON</summary>
                  <pre className="mt-2 max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-2xl border border-slate-800 bg-slate-950/70 p-4 font-mono text-cyan-300">
                    {JSON.stringify(result.c2pa, null, 2)}
                  </pre>
                </details>
              </div>
            )}

            {activeTab === "jpeg" && (
              <div
                id="panel-jpeg"
                role="tabpanel"
                aria-labelledby="tab-jpeg"
                className="space-y-4"
              >
                <div className="flex flex-wrap gap-2">
                  {typeof result.jpeg_qtables?.quality_estimate === "number" ? (
                    <span className="rounded-full border border-emerald-500/20 bg-emerald-500/10 px-2 py-1 text-xs font-semibold text-emerald-300">
                      Quality ~ {result.jpeg_qtables.quality_estimate}
                    </span>
                  ) : null}

                  {typeof result.jpeg_qtables?.inconsistency_score ===
                  "number" ? (
                    <span
                      className="rounded-full border border-amber-500/20 bg-amber-500/10 px-2 py-1 text-xs font-semibold text-amber-300"
                      title="High inconsistency often suggests recompression or edits."
                    >
                      Double Compression:{" "}
                      {(result.jpeg_qtables.inconsistency_score * 100).toFixed(
                        1,
                      )}
                      %
                    </span>
                  ) : null}
                </div>

                {tableGrid ? (
                  <div className="grid grid-cols-1 gap-4 lg:grid-cols-[1fr_220px]">
                    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3">
                      <div className="mb-2 text-xs text-slate-400">
                        Quantization Table (8x8)
                      </div>

                      <div className="grid grid-cols-8 gap-1">
                        {tableGrid.flatMap((row, rIdx) =>
                          row.map((value, cIdx) => {
                            const diff =
                              Math.abs(value - STD_LUMA[rIdx][cIdx]) / 255;
                            const intensity = Math.min(1, diff * 2.5);
                            const bg = `rgba(239, 68, 68, ${0.12 + intensity * 0.45})`;

                            return (
                              <div
                                key={`${rIdx}-${cIdx}`}
                                className="flex h-7 items-center justify-center rounded text-[10px] text-slate-100"
                                style={{ backgroundColor: bg }}
                                title={`Value ${value} (diff ${Math.round(diff * 255)})`}
                              >
                                {value}
                              </div>
                            );
                          }),
                        )}
                      </div>
                    </div>

                    <div className="rounded-2xl border border-slate-700 bg-slate-900 p-3">
                      <div className="mb-2 text-xs text-slate-400">
                        JPEG Quality Heatmap
                      </div>

                      {result.jpeg_qtables?.double_compression
                        ?.jpeg_quality_heatmap_path ? (
                        <img
                          src={fixPath(
                            result.jpeg_qtables.double_compression
                              .jpeg_quality_heatmap_path,
                          )}
                          alt="JPEG quality heatmap"
                          className="w-full rounded border border-slate-700"
                        />
                      ) : (
                        <div className="text-xs text-slate-500">
                          Not available for this file.
                        </div>
                      )}
                    </div>
                  </div>
                ) : (
                  <div className="text-sm text-slate-500">
                    No JPEG tables found for this file.
                  </div>
                )}

                <details className="text-xs text-slate-400">
                  <summary className="cursor-pointer">Raw JPEG data</summary>
                  <pre className="mt-2 max-h-[28rem] overflow-auto whitespace-pre-wrap rounded-2xl border border-slate-800 bg-slate-950/70 p-4 font-mono text-emerald-300">
                    {JSON.stringify(result.jpeg_qtables, null, 2)}
                  </pre>
                </details>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
