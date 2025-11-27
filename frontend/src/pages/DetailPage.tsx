import React, { useEffect, useState } from "react";
import { useParams, Link } from "react-router-dom";
import { getRecordById } from "../services/api";
import type { AnalysisResult } from "../types";
import { AnalysisDashboard } from "../components/AnalysisDashboard";
import { ChevronLeft } from "lucide-react";

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
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-6 flex items-center gap-4">
        <Link
          to="/history"
          className="p-2 bg-slate-800 rounded-full hover:bg-slate-700 text-slate-300 transition-colors"
        >
          <ChevronLeft size={20} />
        </Link>
        <div>
          <h1 className="text-2xl font-bold text-white">Case File #{id}</h1>
          <p className="text-slate-500 text-sm">
            Analysed on {new Date(result.created_at!).toLocaleString()}
          </p>
        </div>
      </div>

      <AnalysisDashboard result={result} />
    </div>
  );
};
