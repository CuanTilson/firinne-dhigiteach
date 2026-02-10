import React, { useState } from "react";
import type { AnalysisResult } from "../types";
import { HeatmapViewer } from "./HeatmapViewer";
import { ForensicScoreCard } from "./ForensicScoreCard";
import { fixPath } from "../constants";
import { CheckCircle, XCircle, FileDigit, Fingerprint } from "lucide-react";

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

export const AnalysisDashboard: React.FC<Props> = ({ result }) => {
  const [activeTab, setActiveTab] = useState<"metadata" | "c2pa" | "jpeg">(
    "metadata"
  );

  const qtables = result.jpeg_qtables?.qtables;
  const firstTable = qtables
    ? (Object.values(qtables)[0] as number[] | undefined)
    : undefined;
  const tableGrid: number[][] | null =
    firstTable && firstTable.length >= 64
      ? Array.from({ length: 8 }, (_, r) =>
          firstTable.slice(r * 8, r * 8 + 8)
        )
      : null;

  return (
    <div className="space-y-6 animate-fade-in">
      <ForensicScoreCard data={result} />

      <div>
        <div className="fd-section-title mb-3">Evidence Panels</div>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <HeatmapViewer
              elaUrl={
                result.ela_heatmap ? fixPath(result.ela_heatmap) : undefined
              }
              gradCamUrl={
                result.gradcam_heatmap
                  ? fixPath(result.gradcam_heatmap)
                  : undefined
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
                        .jpeg_quality_heatmap_path
                    )
                  : undefined
              }
              originalUrl={
                result.saved_path ? fixPath(result.saved_path) : undefined
              }
            />
          </div>

          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="fd-panel p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Fingerprint className="text-cyan-400" size={20} />
                  <h4 className="font-semibold text-slate-200">
                    Invisible Watermark (optional)
                  </h4>
                </div>

                {result.ai_watermark?.stable_diffusion_detected ? (
                  <div className="flex items-center gap-2 text-red-400 text-sm font-medium">
                    <XCircle size={16} /> Detected
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-green-400 text-sm font-medium">
                    <CheckCircle size={16} /> Not Detected
                  </div>
                )}
              </div>

              <div className="fd-panel p-4">
                <div className="flex items-center gap-2 mb-2">
                  <FileDigit className="text-cyan-400" size={20} />
                  <h4 className="font-semibold text-slate-200">File Integrity</h4>
                </div>

                {result.file_integrity?.jpeg_structure?.valid_jpeg ? (
                  <div className="flex items-center gap-2 text-green-400 text-sm font-medium">
                    <CheckCircle size={16} /> Valid
                  </div>
                ) : (
                  <div className="flex items-center gap-2 text-red-400 text-sm font-medium">
                    <XCircle size={16} /> Issues Found
                  </div>
                )}
              </div>
            </div>

            <div className="fd-panel p-4">
              <h4 className="text-slate-200 font-semibold mb-3 flex items-center gap-2">
                <AlertIcon className="text-yellow-500" /> Metadata Anomalies
              </h4>

              {result.metadata_anomalies?.findings?.length === 0 ? (
                <p className="text-slate-500 italic text-sm">
                  No suspicious metadata patterns found.
                </p>
              ) : (
                <ul className="space-y-2">
                  {result.metadata_anomalies?.findings?.map(
                    (anom: string, idx: number) => (
                      <li
                        key={idx}
                        className="bg-red-500/10 border border-red-500/20 text-red-200 px-3 py-2 rounded text-sm"
                      >
                        {anom}
                      </li>
                    )
                  )}
                </ul>
              )}
              {result.metadata_anomalies?.camera_consistency && (
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
              )}
            </div>
          </div>
        </div>
      </div>

      <div>
        <div className="fd-section-title mb-3">Forensic Details</div>
        <div className="fd-card overflow-hidden">
          <div className="flex border-b border-slate-700">
            <button
              onClick={() => setActiveTab("metadata")}
              className={`flex-1 py-3 text-sm font-medium border-b-2 ${
                activeTab === "metadata"
                  ? "border-cyan-500 text-cyan-400 bg-slate-700/50"
                  : "border-transparent text-slate-400"
              }`}
            >
              Raw Metadata
            </button>

            <button
              onClick={() => setActiveTab("c2pa")}
              className={`flex-1 py-3 text-sm font-medium border-b-2 ${
                activeTab === "c2pa"
                  ? "border-cyan-500 text-cyan-400 bg-slate-700/50"
                  : "border-transparent text-slate-400"
              }`}
            >
              C2PA Info
            </button>

            <button
              onClick={() => setActiveTab("jpeg")}
              className={`flex-1 py-3 text-sm font-medium border-b-2 ${
                activeTab === "jpeg"
                  ? "border-cyan-500 text-cyan-400 bg-slate-700/50"
                  : "border-transparent text-slate-400"
              }`}
            >
              JPEG Tables
            </button>
          </div>

          <div className="p-4">
            {activeTab === "metadata" && (
              <pre className="text-xs text-slate-300 font-mono whitespace-pre-wrap">
                {JSON.stringify(result.raw_metadata, null, 2)}
              </pre>
            )}

            {activeTab === "c2pa" && (
              <pre className="text-xs text-cyan-300 font-mono whitespace-pre-wrap">
                {JSON.stringify(result.c2pa, null, 2)}
              </pre>
            )}

            {activeTab === "jpeg" && (
              <div className="space-y-4">
                <div className="flex flex-wrap gap-2">
                  {typeof result.jpeg_qtables?.quality_estimate === "number" && (
                    <span className="px-2 py-1 rounded-full text-xs font-semibold bg-emerald-500/10 text-emerald-300 border border-emerald-500/20">
                      Quality ~ {result.jpeg_qtables.quality_estimate}
                    </span>
                  )}
                  {typeof result.jpeg_qtables?.inconsistency_score ===
                    "number" && (
                    <span
                      className="px-2 py-1 rounded-full text-xs font-semibold bg-amber-500/10 text-amber-300 border border-amber-500/20"
                      title="High inconsistency often suggests recompression or edits."
                    >
                      Double Compression:{" "}
                      {(result.jpeg_qtables.inconsistency_score * 100).toFixed(
                        1
                      )}
                      %
                    </span>
                  )}
                </div>

                {tableGrid ? (
                  <div className="grid grid-cols-1 lg:grid-cols-[1fr_220px] gap-4">
                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-3">
                      <div className="text-xs text-slate-400 mb-2">
                        Quantization Table (8x8)
                      </div>
                      <div className="grid grid-cols-8 gap-1">
                        {tableGrid.flatMap((row, rIdx) =>
                          row.map((value, cIdx) => {
                            const diff =
                              Math.abs(value - STD_LUMA[rIdx][cIdx]) / 255;
                            const intensity = Math.min(1, diff * 2.5);
                            const bg = `rgba(239, 68, 68, ${
                              0.12 + intensity * 0.45
                            })`;
                            return (
                              <div
                                key={`${rIdx}-${cIdx}`}
                                className="text-[10px] text-slate-100 rounded flex items-center justify-center h-7"
                                style={{ backgroundColor: bg }}
                                title={`Value ${value} (diff ${Math.round(
                                  diff * 255
                                )})`}
                              >
                                {value}
                              </div>
                            );
                          })
                        )}
                      </div>
                    </div>

                    <div className="rounded-lg border border-slate-700 bg-slate-900 p-3">
                      <div className="text-xs text-slate-400 mb-2">
                        JPEG Quality Heatmap
                      </div>
                      {result.jpeg_qtables?.double_compression
                        ?.jpeg_quality_heatmap_path ? (
                        <img
                          src={fixPath(
                            result.jpeg_qtables.double_compression
                              .jpeg_quality_heatmap_path
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
                  <pre className="mt-2 text-emerald-300 font-mono whitespace-pre-wrap">
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

const AlertIcon = ({ className }: { className?: string }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    width="20"
    height="20"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
    className={className}
  >
    <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
    <path d="M12 9v4" />
    <path d="M12 17h.01" />
  </svg>
);
