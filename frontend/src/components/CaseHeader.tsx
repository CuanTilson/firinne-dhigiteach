import React from "react";
import { FileDown } from "lucide-react";

interface Props {
  title: string;
  caseId: string | number;
  filename?: string | null;
  createdAt?: string | null;
  reportUrl?: string;
  printUrl?: string;
  hashes?: {
    sha256?: string;
    md5?: string;
    sha256_before?: string;
    sha256_after?: string;
    md5_before?: string;
    md5_after?: string;
  };
}

const formatUtc = (value?: string | null) => {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  return date.toLocaleString("en-GB", {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
};

export const CaseHeader: React.FC<Props> = ({
  title,
  caseId,
  filename,
  createdAt,
  reportUrl,
  printUrl,
  hashes,
}) => {
  return (
    <div className="fd-card p-6 flex flex-col gap-4">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="fd-kicker">Case File</p>
          <h1 className="text-2xl font-semibold text-slate-100 mt-1 fd-title">
            {title} #{caseId}
          </h1>
          <p
            className="text-sm text-slate-400 mt-1 truncate max-w-full md:max-w-[520px]"
            title={filename || "Unnamed file"}
          >
            {filename || "Unnamed file"}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {printUrl && (
            <a
              href={printUrl}
              className="px-4 py-2 rounded-full font-medium transition-all duration-200 flex items-center justify-center gap-2 border border-slate-800 bg-slate-900/70 text-slate-200 hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-slate-500"
            >
              Print View
            </a>
          )}
          {reportUrl && (
            <a
              href={reportUrl}
              className="px-4 py-2 rounded-full font-medium transition-all duration-200 flex items-center justify-center gap-2 bg-cyan-500/15 text-cyan-200 border border-cyan-400/30 hover:bg-cyan-400/20 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-slate-900 focus:ring-cyan-400"
            >
              <FileDown size={16} />
              Download Report
            </a>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
        <div className="fd-panel px-3 py-2">
          <div className="text-xs text-slate-500 uppercase tracking-wide">
            Analysis Time (UTC)
          </div>
          <div className="text-slate-200 font-mono">
            {formatUtc(createdAt)}
          </div>
        </div>
        <div className="fd-panel px-3 py-2">
          <div className="text-xs text-slate-500 uppercase tracking-wide">
            SHA-256
          </div>
          <div className="text-slate-200 font-mono text-xs break-all">
            <div>
              Before: {hashes?.sha256_before || hashes?.sha256 || "Not available"}
            </div>
            <div>
              After: {hashes?.sha256_after || hashes?.sha256 || "Not available"}
            </div>
          </div>
        </div>
        <div className="fd-panel px-3 py-2">
          <div className="text-xs text-slate-500 uppercase tracking-wide">
            MD5
          </div>
          <div className="text-slate-200 font-mono text-xs break-all">
            <div>
              Before: {hashes?.md5_before || hashes?.md5 || "Not available"}
            </div>
            <div>
              After: {hashes?.md5_after || hashes?.md5 || "Not available"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
