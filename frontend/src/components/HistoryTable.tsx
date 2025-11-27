import React from "react";
import type { AnalysisRecordSummary } from "../types";
import { Badge } from "./ui/Badge";
import { Eye, Trash2, Calendar } from "lucide-react";
import { API_BASE_URL } from "../constants";
import { Link } from "react-router-dom";

interface Props {
  records: AnalysisRecordSummary[];
  loading: boolean;
  onDelete: (id: number) => void;
}

export const HistoryTable: React.FC<Props> = ({
  records,
  loading,
  onDelete,
}) => {
  if (loading) {
    return (
      <div className="w-full h-64 flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-cyan-500"></div>
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="text-center py-12 bg-slate-800 rounded-lg border border-slate-700 border-dashed">
        <p className="text-slate-400">
          No analysis records found matching your filters.
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-slate-700 bg-slate-800">
      <table className="w-full text-left text-sm text-slate-400">
        <thead className="bg-slate-900 text-xs uppercase text-slate-200">
          <tr>
            <th className="px-6 py-4 font-medium">Preview</th>
            <th className="px-6 py-4 font-medium">Filename</th>
            <th className="px-6 py-4 font-medium">Classification</th>
            <th className="px-6 py-4 font-medium">Date</th>
            <th className="px-6 py-4 font-medium text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-700">
          {records.map((record) => (
            <tr
              key={record.id}
              className="hover:bg-slate-700/50 transition-colors"
            >
              <td className="px-6 py-3">
                <div className="h-12 w-12 rounded bg-slate-900 overflow-hidden border border-slate-600">
                  <img
                    src={`${API_BASE_URL}${record.thumbnail_url}`}
                    alt="thumb"
                    className="h-full w-full object-cover"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src =
                        "https://picsum.photos/50/50?blur=5"; // Fallback
                    }}
                  />
                </div>
              </td>
              <td className="px-6 py-3 font-medium text-slate-200">
                {record.filename}
                <div className="text-xs text-slate-500 mt-0.5">
                  ID: #{record.id}
                </div>
              </td>
              <td className="px-6 py-3">
                <Badge type={record.classification} />
              </td>
              <td className="px-6 py-3">
                <div className="flex items-center gap-2">
                  <Calendar size={14} />
                  {new Date(record.created_at).toLocaleDateString()}
                </div>
              </td>
              <td className="px-6 py-3 text-right">
                <div className="flex items-center justify-end gap-2">
                  <Link
                    to={`/records/${record.id}`}
                    className="p-2 text-cyan-400 hover:bg-cyan-900/30 rounded-full transition-colors"
                    title="View Details"
                  >
                    <Eye size={18} />
                  </Link>
                  <button
                    onClick={() => onDelete(record.id)}
                    className="p-2 text-red-400 hover:bg-red-900/30 rounded-full transition-colors"
                    title="Delete Record"
                  >
                    <Trash2 size={18} />
                  </button>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};
