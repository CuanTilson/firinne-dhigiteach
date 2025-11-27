import React, { useState } from "react";
import type { AnalysisResult } from "../types";
import { HeatmapViewer } from "./HeatmapViewer";
import { ForensicScoreCard } from "./ForensicScoreCard";
import { fixPath } from "../constants";
import { CheckCircle, XCircle, FileDigit, Fingerprint } from "lucide-react";

interface Props {
  result: AnalysisResult;
}

export const AnalysisDashboard: React.FC<Props> = ({ result }) => {
  const [activeTab, setActiveTab] = useState<"metadata" | "c2pa" | "jpeg">(
    "metadata"
  );

  return (
    <div className="space-y-6 animate-fade-in">
      <ForensicScoreCard data={result} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-auto lg:h-[600px]">
        <div className="h-full">
          <HeatmapViewer
            elaUrl={
              result.ela_analysis?.ela_heatmap
                ? fixPath(result.ela_analysis.ela_heatmap)
                : undefined
            }
            gradCamUrl={
              result.gradcam_heatmap
                ? fixPath(result.gradcam_heatmap)
                : undefined
            }
            originalUrl={
              result.saved_path ? fixPath(result.saved_path) : undefined
            }
          />
        </div>

        {/* Right side */}
        <div className="flex flex-col gap-4 overflow-y-auto pr-2 custom-scrollbar">
          {/* Anomaly cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            {/* Stable Diffusion */}
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
              <div className="flex items-center gap-2 mb-2">
                <Fingerprint className="text-cyan-400" size={20} />
                <h4 className="font-semibold text-slate-200">SD Watermark</h4>
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

            {/* File Integrity */}
            <div className="bg-slate-800 p-4 rounded-lg border border-slate-700">
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

          {/* Metadata anomalies */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 p-4">
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
          </div>

          {/* Raw tabs */}
          <div className="bg-slate-800 rounded-lg border border-slate-700 grow flex flex-col overflow-hidden min-h-[300px]">
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

            <div className="grow p-0 overflow-hidden relative">
              <div className="absolute inset-0 overflow-auto p-4">
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
                  <pre className="text-xs text-emerald-300 font-mono whitespace-pre-wrap">
                    {JSON.stringify(result.jpeg_qtables, null, 2)}
                  </pre>
                )}
              </div>
            </div>
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
