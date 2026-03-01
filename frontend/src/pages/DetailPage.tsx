import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { ChevronLeft } from "lucide-react";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";
import { EvidenceStatusStrip } from "../components/EvidenceStatusStrip";
import { AnalysisProvenancePanel } from "../components/AnalysisProvenancePanel";

export const DetailPage: React.FC = () => {
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

  if (loading)
    return (
      <div className="min-h-screen flex items-center justify-center text-slate-400">
        Loading case file #{id}...
      </div>
    );

  if (error || !result)
    return (
      <div className="min-h-screen flex flex-col items-center justify-center text-red-400 gap-4">
        <p>{error || "Record not found"}</p>
        <Link to="/history" className="text-cyan-400 hover:underline">
          Back to History
        </Link>
      </div>
    );

  const detector =
    result.ml_prediction?.detector && typeof result.ml_prediction.detector === "object"
      ? result.ml_prediction.detector
      : null;
  const fusionMode =
    result.forensic_score_json &&
    typeof result.forensic_score_json === "object" &&
    typeof result.forensic_score_json.provenance === "object"
      ? String(
          (result.forensic_score_json.provenance as Record<string, unknown>)
            .fusion_mode || ""
        )
      : null;

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to="/history"
          className="p-2 bg-slate-900 rounded-full hover:bg-slate-800 text-slate-300 transition-colors border border-slate-800"
        >
          <ChevronLeft size={20} />
        </Link>
        <span className="text-sm text-slate-400">Back to History</span>
      </div>

      <CaseHeader
        title="Image Analysis"
        caseId={id || ""}
        filename={result.filename}
        createdAt={result.created_at}
        printUrl={`#/print/records/${id}`}
        hashes={{
          sha256: result.file_integrity?.hashes?.sha256,
          md5: result.file_integrity?.hashes?.md5,
          sha256_before: result.file_integrity?.hashes_before?.sha256,
          sha256_after: result.file_integrity?.hashes_after?.sha256,
          md5_before: result.file_integrity?.hashes_before?.md5,
          md5_after: result.file_integrity?.hashes_after?.md5,
        }}
      />

      <ChainOfCustody
        steps={[
          { label: "Upload received", timestamp: result.created_at, status: "complete" },
          { label: "Analysis completed", timestamp: result.created_at, status: "complete" },
          { label: "Report generated", timestamp: result.created_at, status: "complete" },
        ]}
      />

      <EvidenceStatusStrip
        items={[
          {
            label: "ML Signal",
            status: `${((result.ml_prediction?.probability ?? 0) * 100).toFixed(1)}% AI`,
            tone: (result.ml_prediction?.probability ?? 0) >= 0.7 ? "warn" : "neutral",
            detail: result.ml_prediction?.label ?? "Unavailable",
          },
          {
            label: "Metadata",
            status: `${result.metadata_anomalies?.findings?.length ?? 0} findings`,
            tone: (result.metadata_anomalies?.anomaly_score ?? 0) >= 0.4 ? "warn" : "neutral",
            detail: `Score ${(result.metadata_anomalies?.anomaly_score ?? 0).toFixed(3)}`,
          },
          {
            label: "Provenance",
            status: result.c2pa?.has_c2pa ? "C2PA present" : "No C2PA",
            tone: result.c2pa?.has_c2pa ? "good" : "neutral",
            detail: result.c2pa?.signature_valid ? "Signature valid" : undefined,
          },
          {
            label: "Integrity",
            status:
              result.file_integrity?.hashes_match === false
                ? "Hash mismatch"
                : "Hashes recorded",
            tone:
              result.file_integrity?.hashes_match === false ? "bad" : "good",
            detail: result.file_integrity?.jpeg_structure?.valid_jpeg
              ? "JPEG structure valid"
              : "Structure warnings present",
          },
        ]}
      />

      <AppliedSettingsPanel settings={result.applied_settings} />

      <AnalysisProvenancePanel detector={detector} fusionMode={fusionMode} />

      <DecisionSummaryPanel
        verdict={result.classification}
        scoreLabel="Final Forensic Score"
        scoreValue={result.forensic_score.toFixed(3)}
        rationale={[
          `Model probability: ${(result.ml_prediction?.probability ?? 0).toFixed(3)}`,
          `Metadata anomaly score: ${(result.metadata_anomalies?.anomaly_score ?? 0).toFixed(3)}`,
          `C2PA AI assertions: ${(result.c2pa?.ai_assertions_found?.length ?? 0).toString()}`,
          `JPEG inconsistency score: ${String(result.jpeg_qtables?.inconsistency_score ?? "Unavailable")}`,
        ]}
        note="This decision combines ML output with forensic signals and provenance indicators."
      />

      <AnalysisDashboard result={result} />
    </div>
  );
};
