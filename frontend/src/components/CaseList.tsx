import React from "react";
import type { AnalysisRecordSummary } from "../types";
import { Badge } from "./ui/Badge";
import {
  Eye,
  Trash2,
  Calendar,
  Film,
  Printer,
  AudioWaveform,
  Image as ImageIcon,
} from "lucide-react";
import { API_BASE_URL } from "../constants";
import { Link } from "react-router-dom";

interface Props {
  records: AnalysisRecordSummary[];
  loading: boolean;
  onDelete: (
    id: number,
    mediaType: AnalysisRecordSummary["media_type"],
  ) => void;
}

const formatDate = (value: string) =>
  new Date(value).toLocaleDateString("en-IE", {
    day: "2-digit",
    month: "2-digit",
    year: "numeric",
  });

const getDetailPath = (record: AnalysisRecordSummary) => {
  if (record.media_type === "video") return `/videos/${record.id}`;
  if (record.media_type === "audio") return `/audio/${record.id}`;
  return `/records/${record.id}`;
};

const getPrintPath = (record: AnalysisRecordSummary) => {
  if (record.media_type === "video") return `#/print/videos/${record.id}`;
  if (record.media_type === "audio") return `#/print/audio/${record.id}`;
  return `#/print/records/${record.id}`;
};

const MediaIcon = ({
  mediaType,
}: {
  mediaType: AnalysisRecordSummary["media_type"];
}) => {
  if (mediaType === "video") {
    return <Film size={14} className="text-cyan-300" />;
  }
  if (mediaType === "audio") {
    return <AudioWaveform size={14} className="text-cyan-300" />;
  }
  return <ImageIcon size={14} className="text-cyan-300" />;
};

export const CaseList: React.FC<Props> = ({ records, loading, onDelete }) => {
  if (loading) {
    return (
      <div className="flex h-64 w-full items-center justify-center">
        <div className="h-12 w-12 animate-spin rounded-full border-b-2 border-cyan-400" />
      </div>
    );
  }

  if (records.length === 0) {
    return (
      <div className="rounded-3xl border border-dashed border-slate-700 bg-slate-950/45 py-12 text-center">
        <p className="text-slate-400">
          No analysis records found matching your filters.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {records.map((record) => {
        const detailPath = getDetailPath(record);
        const printPath = getPrintPath(record);

        return (
          <article
            key={record.id}
            className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-4 shadow-[0_8px_24px_rgba(2,6,23,0.20)] transition hover:border-slate-700"
          >
            <div className="flex flex-col gap-4 md:flex-row md:items-center">
              <div className="flex min-w-0 items-center gap-4">
                <div className="h-16 w-16 shrink-0 overflow-hidden rounded-2xl border border-slate-800 bg-slate-950">
                  {record.thumbnail_url ? (
                    <img
                      src={`${API_BASE_URL}${record.thumbnail_url}`}
                      alt={`${record.media_type} thumbnail for ${record.filename}`}
                      className="h-full w-full object-cover"
                      onError={(event) => {
                        event.currentTarget.style.display = "none";
                      }}
                    />
                  ) : (
                    <div className="flex h-full w-full items-center justify-center text-slate-500">
                      <MediaIcon mediaType={record.media_type} />
                    </div>
                  )}
                </div>

                <div className="min-w-0">
                  <div
                    className="flex max-w-[240px] items-center gap-2 truncate font-semibold text-slate-200 md:max-w-[360px]"
                    title={record.filename}
                  >
                    <MediaIcon mediaType={record.media_type} />
                    {record.filename}
                  </div>

                  <div className="mt-1 text-xs text-slate-500">
                    Case #{record.id} · {record.media_type}
                  </div>

                  <div className="mt-2 flex flex-wrap items-center gap-2 text-xs">
                    <Badge type={record.classification} />
                    <span className="rounded-full border border-slate-800 bg-slate-900/70 px-2.5 py-1 text-slate-400">
                      Score {record.forensic_score.toFixed(3)}
                    </span>
                    <span className="inline-flex items-center gap-1 rounded-full border border-slate-800 bg-slate-900/70 px-2.5 py-1 text-slate-400">
                      <Calendar size={12} />
                      {formatDate(record.created_at)}
                    </span>
                  </div>
                </div>
              </div>

              <div className="flex-1" />

              <div className="flex flex-wrap gap-2">
                <Link
                  to={detailPath}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs font-medium text-slate-200 transition hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                >
                  <Eye size={14} />
                  View
                </Link>

                <a
                  href={printPath}
                  className="inline-flex items-center gap-2 rounded-full border border-slate-800 bg-slate-900/70 px-3 py-2 text-xs font-medium text-slate-200 transition hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                >
                  <Printer size={14} />
                  Report
                </a>

                <button
                  type="button"
                  onClick={() => onDelete(record.id, record.media_type)}
                  className="inline-flex items-center gap-2 rounded-full border border-rose-400/30 bg-rose-500/10 px-3 py-2 text-xs font-medium text-rose-200 transition hover:bg-rose-500/20 focus:outline-none focus:ring-2 focus:ring-rose-500 focus:ring-offset-2 focus:ring-offset-slate-900"
                >
                  <Trash2 size={14} />
                  Delete
                </button>
              </div>
            </div>
          </article>
        );
      })}
    </div>
  );
};
