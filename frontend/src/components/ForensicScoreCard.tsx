import React from "react";
import type { AnalysisResult, ClassificationType } from "../types";
import { Badge } from "./ui/Badge";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";

interface Props {
  data: AnalysisResult;
}

const getClassificationMeta = (
  classification: ClassificationType | "uncertain",
) => {
  switch (classification) {
    case "likely_real":
      return {
        icon: <ShieldCheck size={40} className="text-emerald-400" />,
        barClass: "bg-emerald-500",
      };
    case "likely_ai_generated":
      return {
        icon: <ShieldAlert size={40} className="text-rose-400" />,
        barClass: "bg-rose-500",
      };
    default:
      return {
        icon: <AlertTriangle size={40} className="text-amber-400" />,
        barClass: "bg-amber-400",
      };
  }
};

export const ForensicScoreCard: React.FC<Props> = ({ data }) => {
  const classification = (data.classification ?? "uncertain") as
    | ClassificationType
    | "uncertain";

  const scorePercent = Math.max(
    0,
    Math.min(
      100,
      (data.forensic_score_json?.final_score ?? data.forensic_score ?? 0) * 100,
    ),
  );

  const mlConfidence = data.ml_prediction?.probability ?? 0;
  const metadataAnomalyCount = data.metadata_anomalies?.findings?.length ?? 0;
  const noiseScore = data.noise_residual?.combined_anomaly_score ?? 0;
  const jpegInconsistency = data.jpeg_qtables?.inconsistency_score ?? 0;
  const c2paAssertions = data.c2pa?.ai_assertions_found?.length ?? 0;

  const findings = [
    c2paAssertions > 0 ? `C2PA AI assertions found (${c2paAssertions})` : null,
    jpegInconsistency > 0.15
      ? `JPEG double compression signal (${(jpegInconsistency * 100).toFixed(1)}%)`
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
  ].filter(Boolean) as string[];

  const meta = getClassificationMeta(classification);

  return (
    <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-5 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[auto_minmax(0,1fr)_260px]">
        <div className="flex items-center gap-4">
          <div className="flex h-16 w-16 items-center justify-center rounded-2xl border border-slate-800 bg-slate-950/70">
            {meta.icon}
          </div>
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
              Final Classification
            </div>
            <div className="mt-2">
              <Badge type={classification as ClassificationType} />
            </div>
          </div>
        </div>

        <div>
          <div className="flex items-center justify-between text-xs uppercase tracking-[0.16em] text-slate-500">
            <span>Forensic Score</span>
            <span>{scorePercent.toFixed(1)}%</span>
          </div>

          <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800">
            <div
              className={`h-full ${meta.barClass}`}
              style={{ width: `${scorePercent}%` }}
            />
          </div>

          <div className="mt-2 text-xs text-slate-500">
            Threshold guidance: AI likely &gt;= 70%, Real likely &lt;= 30%.
          </div>
        </div>

        <div>
          <div className="mb-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
            Key Findings
          </div>

          {findings.length === 0 ? (
            <p className="text-sm text-slate-400">
              No dominant indicators detected.
            </p>
          ) : (
            <ul className="space-y-2 text-sm text-slate-300">
              {findings.slice(0, 3).map((finding, idx) => (
                <li key={`${finding}-${idx}`} className="leading-6">
                  - {finding}
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};
