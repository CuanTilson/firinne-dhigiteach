import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { ChevronLeft } from "lucide-react";
import { CaseHeader } from "../components/CaseHeader";
import { ChainOfCustody } from "../components/ChainOfCustody";

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

      <AnalysisDashboard result={result} />
    </div>
  );
};
