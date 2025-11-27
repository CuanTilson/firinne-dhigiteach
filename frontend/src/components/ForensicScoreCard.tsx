import React from "react";
import type { AnalysisResult } from "../types";
import { Badge } from "./ui/Badge";
import { ShieldCheck, ShieldAlert, AlertTriangle } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer } from "recharts";

interface Props {
  data: AnalysisResult;
}

export const ForensicScoreCard: React.FC<Props> = ({ data }) => {
  // Correct fields
  const classification = data.classification ?? "uncertain";
  const scorePercent = (data.forensic_score_json?.final_score ?? 0) * 100;
  const mlConfidence = data.ml_prediction?.probability ?? 0;
  const metadataAnomalyCount = data.metadata_anomalies?.findings?.length ?? 0;
  const noiseLevel = data.noise_residual?.variance ?? 0;

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

  const chartData = [
    { name: "Confidence", value: scorePercent },
    { name: "Remaining", value: 100 - scorePercent },
  ];

  const getColor = () => {
    if (classification === "likely_real") return "#22c55e";
    if (classification === "likely_ai_generated") return "#ef4444";
    return "#eab308";
  };

  return (
    <div className="bg-slate-800 rounded-xl p-6 border border-slate-700 shadow-lg flex flex-col md:flex-row gap-6 items-center">
      {/* Visual Gauge */}
      <div className="relative shrink-0">
        <div style={{ width: 128, height: 128 }}>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={chartData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={60}
                startAngle={90}
                endAngle={-270}
                dataKey="value"
                stroke="none"
              >
                <Cell fill={getColor()} />
                <Cell fill="#334155" />
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Overlay text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-2xl font-bold text-slate-100">
            {scorePercent.toFixed(0)}%
          </span>
          <span className="text-[10px] text-slate-400 uppercase tracking-wide">
            Score
          </span>
        </div>
      </div>

      {/* Text info */}
      <div className="grow text-center md:text-left">
        <div className="flex items-center justify-center md:justify-start gap-4 mb-2">
          {getIcon()}
          <div>
            <h2 className="text-sm text-slate-400 font-medium uppercase tracking-wider">
              Final Classification
            </h2>
            <Badge type={classification} />
          </div>
        </div>

        <p className="text-slate-400 text-sm mt-3 max-w-lg">
          The model predicts this image is
          <span className="font-semibold text-slate-200">
            {" "}
            {classification.replace(/_/g, " ")}{" "}
          </span>
          with a model confidence of{" "}
          <span className="font-mono text-cyan-400">
            {(mlConfidence * 100).toFixed(1)}%
          </span>
          .
          {data.file_integrity.jpeg_structure.valid_jpeg
            ? " File structure integrity is intact."
            : " File structure anomalies detected."}
        </p>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 gap-4 w-full md:w-auto">
        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
          <div className="text-xs text-slate-500 uppercase">Noise Level</div>
          <div className="text-lg font-mono text-slate-200">
            {noiseLevel.toFixed(4)}
          </div>
        </div>

        <div className="bg-slate-900/50 p-3 rounded-lg border border-slate-700/50">
          <div className="text-xs text-slate-500 uppercase">Anomalies</div>
          <div
            className={`text-lg font-mono ${
              metadataAnomalyCount > 0 ? "text-red-400" : "text-green-400"
            }`}
          >
            {metadataAnomalyCount}
          </div>
        </div>
      </div>
    </div>
  );
};
