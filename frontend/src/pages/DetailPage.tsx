import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";
import { AppliedSettingsPanel } from "../components/AppliedSettingsPanel";
import { DecisionSummaryPanel } from "../components/DecisionSummaryPanel";
import { EvidenceStatusStrip } from "../components/EvidenceStatusStrip";
import { AnalysisProvenancePanel } from "../components/AnalysisProvenancePanel";
import { CasePageScaffold } from "../components/CasePageScaffold";

const formatClassification = (value?: string | null) => {
  if (!value) return "Unavailable";
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const getDetectorInfo = (result: AnalysisResult) => {
  return result.ml_prediction?.detector &&
    typeof result.ml_prediction.detector === "object"
    ? result.ml_prediction.detector
    : null;
};

const getFusionMode = (result: AnalysisResult) => {
  const provenance =
    result.forensic_score_json &&
    typeof result.forensic_score_json === "object" &&
    typeof result.forensic_score_json.provenance === "object"
      ? (result.forensic_score_json.provenance as Record<string, unknown>)
      : null;

  return provenance?.fusion_mode ? String(provenance.fusion_mode) : null;
};

const getStatusItems = (result: AnalysisResult) => [
  {
    label: "ML Signal",
    status: `${((result.ml_prediction?.probability ?? 0) * 100).toFixed(1)}% AI`,
    tone:
      (result.ml_prediction?.probability ?? 0) >= 0.7
        ? ("warn" as const)
        : ("neutral" as const),
    detail: result.ml_prediction?.label ?? "Unavailable",
  },
  {
    label: "Metadata",
    status: `${result.metadata_anomalies?.findings?.length ?? 0} findings`,
    tone:
      (result.metadata_anomalies?.anomaly_score ?? 0) >= 0.4
        ? ("warn" as const)
        : ("neutral" as const),
    detail: `Score ${(result.metadata_anomalies?.anomaly_score ?? 0).toFixed(3)}`,
  },
  {
    label: "Provenance",
    status: result.c2pa?.has_c2pa ? "C2PA present" : "No C2PA",
    tone: result.c2pa?.has_c2pa ? ("good" as const) : ("neutral" as const),
    detail: result.c2pa?.signature_valid ? "Signature valid" : undefined,
  },
  {
    label: "Integrity",
    status:
      result.file_integrity?.hashes_match === false
        ? "Hash mismatch"
        : "Hashes recorded",
    tone:
      result.file_integrity?.hashes_match === false
        ? ("bad" as const)
        : ("good" as const),
    detail: result.file_integrity?.jpeg_structure?.valid_jpeg
      ? "JPEG structure valid"
      : "Structure warnings present",
  },
];

const getDecisionRationale = (result: AnalysisResult) => [
  `Model probability: ${(result.ml_prediction?.probability ?? 0).toFixed(3)}`,
  `Metadata anomaly score: ${(result.metadata_anomalies?.anomaly_score ?? 0).toFixed(3)}`,
  `C2PA AI assertions: ${(result.c2pa?.ai_assertions_found?.length ?? 0).toString()}`,
  `JPEG inconsistency score: ${String(
    result.jpeg_qtables?.inconsistency_score ?? "Unavailable",
  )}`,
];

const getChainSteps = (result: AnalysisResult) => [
  {
    label: "Upload received",
    timestamp: result.created_at,
    status: "complete" as const,
  },
  {
    label: "Analysis completed",
    timestamp: result.created_at,
    status: "complete" as const,
  },
  {
    label: "Report generated",
    timestamp: result.created_at,
    status: "complete" as const,
  },
];

export const DetailPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchDetail = async () => {
      if (!id) {
        setError("No record id was provided.");
        setLoading(false);
        return;
      }

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
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
        <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 px-6 py-12 text-center text-slate-400">
          Loading case file #{id}...
        </div>
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
        <div className="rounded-3xl border border-red-500/20 bg-red-500/10 px-6 py-12 text-center">
          <p className="text-red-300">{error || "Record not found."}</p>
          <Link
            to="/history"
            className="mt-4 inline-flex text-cyan-300 underline-offset-4 hover:text-cyan-200 hover:underline"
          >
            Back to History
          </Link>
        </div>
      </div>
    );
  }

  const detector = getDetectorInfo(result);
  const fusionMode = getFusionMode(result);

  return (
    <CasePageScaffold
      backTo="/history"
      backLabel="Back to History"
      header={
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
      }
      statusStrip={<EvidenceStatusStrip items={getStatusItems(result)} />}
      sidebar={
        <>
          <DecisionSummaryPanel
            verdict={formatClassification(result.classification)}
            scoreLabel="Final Forensic Score"
            scoreValue={result.forensic_score.toFixed(3)}
            rationale={getDecisionRationale(result)}
            note="This decision combines ML output with forensic signals and provenance indicators."
          />
          <AnalysisProvenancePanel
            detector={detector}
            fusionMode={fusionMode}
          />
          <ChainOfCustody steps={getChainSteps(result)} />
          <AppliedSettingsPanel settings={result.applied_settings} />
        </>
      }
    >
      <AnalysisDashboard result={result} />
    </CasePageScaffold>
  );
};
