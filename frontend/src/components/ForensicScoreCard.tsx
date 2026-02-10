import React from "react";
import type { AnalysisResult } from "../types";
import { Badge } from "./ui/Badge";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";

interface Props {
  data: AnalysisResult;
}

export const ForensicScoreCard: React.FC<Props> = ({ data }) => {
  // Correct fields
  const classification = data.classification ?? "uncertain";
  const scorePercent = (data.forensic_score_json?.final_score ?? 0) * 100;
  const mlConfidence = data.ml_prediction?.probability ?? 0;
  const metadataAnomalyCount = data.metadata_anomalies?.findings?.length ?? 0;
  const noiseScore = data.noise_residual?.combined_anomaly_score ?? 0;
  const jpegInconsistency = data.jpeg_qtables?.inconsistency_score ?? 0;
  const c2paAssertions = data.c2pa?.ai_assertions_found?.length ?? 0;

  const getIcon = () => {
    switch (classification) {
      case "likely_real":
        return <ShieldCheck size={48} className="text-green-500" />;
      case "likely_ai_generated":
        return <ShieldAlert size={48} className="text-red-500" />;
      default:
        return <AlertTriangle size={48} className="text-yellow-500" />;
    }
  };

  const getColor = () => {
    if (classification === "likely_real") return "bg-emerald-500";
    if (classification === "likely_ai_generated") return "bg-rose-500";
    return "bg-amber-400";
  };

  const findings = [
    c2paAssertions > 0
      ? `C2PA AI assertions found (${c2paAssertions})`
      : null,
    jpegInconsistency > 0.15
      ? `JPEG double compression signal (${(jpegInconsistency * 100).toFixed(
          1
        )}%)`
      : null,
    noiseScore > 0.4
      ? `Noise inconsistency score (${noiseScore.toFixed(2)})`
      : null,
    metadataAnomalyCount > 0
      ? `Metadata anomalies (${metadataAnomalyCount})`
      : null,
    mlConfidence > 0.7
      ? `Model confidence (${(mlConfidence * 100).toFixed(1)}%)`
      : null,
  ]
    .filter(Boolean)
    .slice(0, 3) as string[];

  return (
    <div className="fd-card p-6">
      <div className="flex flex-col lg:flex-row gap-6">
        <div className="flex items-center gap-4">
          {getIcon()}
          <div>
            <p className="fd-section-title">Final Classification</p>
            <div className="mt-1">
              <Badge type={classification} />
            </div>
          </div>
        </div>

        <div className="flex-1">
          <div className="flex items-center justify-between text-xs text-slate-500 uppercase tracking-wider">
            <span>Forensic Score</span>
            <span>{scorePercent.toFixed(1)}%</span>
          </div>
          <div className="mt-2 h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className={`h-full ${getColor()}`}
              style={{ width: `${Math.min(100, scorePercent)}%` }}
            ></div>
          </div>
          <div className="mt-2 text-xs text-slate-500">
            Threshold guidance: AI likely &gt;= 70%, Real likely &lt;= 30%.
          </div>
        </div>

        <div className="min-w-[220px]">
          <p className="text-xs uppercase tracking-widest text-slate-500 mb-2">
            Key Findings
          </p>
          {findings.length === 0 ? (
            <p className="text-sm text-slate-400">
              No dominant indicators detected.
            </p>
          ) : (
            <ul className="space-y-1 text-sm text-slate-300">
              {findings.map((finding, idx) => (
                <li key={`${finding}-${idx}`}>- {finding}</li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};
