import React from "react";
import { CheckCircle2, AlertTriangle, FileText } from "lucide-react";

interface Props {
  title: string;
  caseId: string | number;
  filename?: string | null;
  createdAt?: string | null;
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

const getHashState = (before?: string, after?: string, fallback?: string) => {
  const a = before || fallback;
  const b = after || fallback;
  if (!a || !b) return "Unavailable";
  return a === b ? "Match" : "Changed";
};

const HashBlock = ({
  label,
  before,
  after,
  fallback,
}: {
  label: string;
  before?: string;
  after?: string;
  fallback?: string;
}) => {
  const state = getHashState(before, after, fallback);

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-3 py-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
          {label}
        </div>
        <div
          className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] ${
            state === "Match"
              ? "bg-emerald-500/10 text-emerald-300"
              : state === "Changed"
                ? "bg-amber-500/10 text-amber-300"
                : "bg-slate-800 text-slate-400"
          }`}
        >
          {state === "Match" ? (
            <CheckCircle2 size={12} />
          ) : state === "Changed" ? (
            <AlertTriangle size={12} />
          ) : (
            <FileText size={12} />
          )}
          {state}
        </div>
      </div>
      <div className="mt-2 space-y-1 font-mono text-xs text-slate-300">
        <div className="break-all">
          Before: {before || fallback || "Not available"}
        </div>
        <div className="break-all">
          After: {after || fallback || "Not available"}
        </div>
      </div>
    </div>
  );
};

export const CaseHeader: React.FC<Props> = ({
  title,
  caseId,
  filename,
  createdAt,
  printUrl,
  hashes,
}) => {
  return (
    <div className="rounded-3xl border border-slate-800/80 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.08),_transparent_35%),linear-gradient(180deg,rgba(2,6,23,0.96),rgba(2,6,23,0.82))] p-6 shadow-[0_0_0_1px_rgba(15,23,42,0.5)]">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-[11px] uppercase tracking-[0.28em] text-cyan-300/80">
            Case File
          </p>
          <h1 className="mt-1 text-2xl font-semibold text-slate-100">
            {title} #{caseId}
          </h1>
          <p
            className="mt-1 max-w-full truncate text-sm text-slate-400 md:max-w-[520px]"
            title={filename || "Unnamed file"}
          >
            {filename || "Unnamed file"}
          </p>
        </div>

        {printUrl ? (
          <a
            href={printUrl}
            className="inline-flex items-center justify-center rounded-full border border-slate-800 bg-slate-900/70 px-4 py-2 text-sm font-medium text-slate-200 transition hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-slate-500"
          >
            Print Layout
          </a>
        ) : null}
      </div>

      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        <div className="rounded-2xl border border-slate-800 bg-slate-950/60 px-3 py-3">
          <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
            Analysis Time (UTC)
          </div>
          <div className="mt-1 font-mono text-sm text-slate-200">
            {formatUtc(createdAt)}
          </div>
        </div>

        <HashBlock
          label="SHA-256"
          before={hashes?.sha256_before}
          after={hashes?.sha256_after}
          fallback={hashes?.sha256}
        />

        <HashBlock
          label="MD5"
          before={hashes?.md5_before}
          after={hashes?.md5_after}
          fallback={hashes?.md5}
        />
      </div>
    </div>
  );
};
