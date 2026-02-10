import React from "react";
import type { AnalysisRecordSummary } from "../types";
import { Badge } from "./ui/Badge";
import { Eye, Trash2, Calendar, Film, FileDown, Printer } from "lucide-react";
import { API_BASE_URL } from "../constants";
import { Link } from "react-router-dom";

interface Props {
  records: AnalysisRecordSummary[];
  loading: boolean;
  onDelete: (id: number, mediaType: AnalysisRecordSummary["media_type"]) => void;
}

export const CaseList: React.FC<Props> = ({ records, loading, onDelete }) => {
  if (loading) {
    return (
      <div className="w-full h-64 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-400"></div>
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="fd-card border-dashed text-center py-12">
        <p className="text-slate-400">
          No analysis records found matching your filters.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {records.map((record) => {
        const detailPath =
          record.media_type === "video"
            ? `/videos/${record.id}`
            : `/records/${record.id}`;
        const reportPath =
          record.media_type === "video"
            ? `${API_BASE_URL}/analysis/video/${record.id}/report.pdf`
            : `${API_BASE_URL}/analysis/${record.id}/report.pdf`;
        const printPath =
          record.media_type === "video"
            ? `#/print/videos/${record.id}`
            : `#/print/records/${record.id}`;

        return (
          <div
            key={record.id}
            className="fd-card p-4 flex flex-col md:flex-row md:items-center gap-4"
          >
            <div className="flex items-center gap-4 min-w-0">
              <div className="h-16 w-16 rounded-xl bg-slate-950 overflow-hidden border border-slate-800 shrink-0">
                <img
                  src={`${API_BASE_URL}${record.thumbnail_url}`}
                  alt="thumb"
                  className="h-full w-full object-cover"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src =
                      "https://picsum.photos/64/64?blur=5";
                  }}
                />
              </div>
              <div className="min-w-0">
                <div
                  className="flex items-center gap-2 text-slate-200 font-semibold truncate max-w-[240px] md:max-w-[360px]"
                  title={record.filename}
                >
                  {record.media_type === "video" && (
                    <Film size={14} className="text-cyan-300" />
                  )}
                  {record.filename}
                </div>
                <div className="text-xs text-slate-500 mt-1">
                  Case #{record.id} · {record.media_type}
                </div>
                <div className="mt-2 flex flex-wrap gap-2 items-center text-xs">
                  <Badge type={record.classification} />
                  <span className="fd-pill">
                    Score {record.forensic_score.toFixed(3)}
                  </span>
                  <span className="fd-pill">
                    <Calendar size={12} />
                    {new Date(record.created_at).toLocaleDateString()}
                  </span>
                </div>
              </div>
            </div>

            <div className="flex-1" />

            <div className="flex flex-wrap gap-2">
              <Link
                to={detailPath}
                className="px-3 py-2 rounded-full text-xs font-medium border border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800 transition-colors inline-flex items-center gap-2"
              >
                <Eye size={14} />
                View
              </Link>
              <a
                href={reportPath}
                className="px-3 py-2 rounded-full text-xs font-medium border border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800 transition-colors inline-flex items-center gap-2"
              >
                <FileDown size={14} />
                Report
              </a>
              <a
                href={printPath}
                className="px-3 py-2 rounded-full text-xs font-medium border border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800 transition-colors inline-flex items-center gap-2"
              >
                <Printer size={14} />
                Print
              </a>
              <button
                onClick={() => onDelete(record.id, record.media_type)}
                className="px-3 py-2 rounded-full text-xs font-medium border border-rose-400/30 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20 transition-colors inline-flex items-center gap-2"
              >
                <Trash2 size={14} />
                Delete
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
};
