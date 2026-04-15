import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { getAuditLogs } from "../services/api";
import type { AuditLogEntry } from "../types";
import { Button } from "../components/ui/Button";
import {
  ChevronDown,
  FileText,
  Image as ImageIcon,
  Music,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Video,
} from "lucide-react";

const truncate = (value: string, max: number) =>
  value.length > max ? `${value.slice(0, max)}…` : value;

const formatRecord = (entry: AuditLogEntry) => {
  if (!entry.record_type) return { label: "system" };
  if (!entry.record_id) return { label: entry.record_type };

  const path =
    entry.record_type === "video"
      ? `/videos/${entry.record_id}`
      : `/records/${entry.record_id}`;

  return {
    label: `${entry.record_type} #${entry.record_id}`,
    path,
  };
};

const getActionLabel = (action: string) =>
  action
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");

const getActionTone = (action: string) => {
  if (action.includes("settings")) {
    return "border-amber-500/20 bg-amber-500/10 text-amber-200";
  }
  if (action.includes("report")) {
    return "border-sky-500/20 bg-sky-500/10 text-sky-200";
  }
  if (action.includes("completed")) {
    return "border-emerald-500/20 bg-emerald-500/10 text-emerald-200";
  }
  if (action.includes("queued")) {
    return "border-violet-500/20 bg-violet-500/10 text-violet-200";
  }
  return "border-cyan-500/20 bg-cyan-500/10 text-cyan-200";
};

const getRecordIcon = (recordType?: string | null) => {
  if (recordType === "image") return <ImageIcon size={15} />;
  if (recordType === "video") return <Video size={15} />;
  if (recordType === "audio") return <Music size={15} />;
  if (recordType === "settings") return <Settings2 size={15} />;
  return <FileText size={15} />;
};

const formatDate = (value: string) =>
  new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "medium",
  }).format(new Date(value));

const formatDetailsPreview = (
  details?: Record<string, unknown> | null | unknown,
) => {
  if (!details) return "No additional detail";
  if (typeof details !== "object") return truncate(String(details), 120);

  const data = details as Record<string, unknown>;
  const keys = Object.keys(data);
  if (keys.length === 0) return "No additional detail";

  const preferredKeys = [
    "classification",
    "forensic_score",
    "pipeline_version",
    "model_version",
    "job_id",
    "updated_fields",
    "error",
  ];

  const previewParts = preferredKeys
    .filter((key) => key in data)
    .slice(0, 3)
    .map((key) => `${key}: ${truncate(JSON.stringify(data[key]), 34)}`);

  return previewParts.length > 0
    ? previewParts.join(" · ")
    : `${keys.length} detail field${keys.length === 1 ? "" : "s"}`;
};

const formatDetailsJson = (
  details?: Record<string, unknown> | null | unknown,
) => {
  if (!details) return "";
  if (typeof details !== "object") return String(details);
  if (Object.keys(details as Record<string, unknown>).length === 0) return "";
  return JSON.stringify(details, null, 2);
};

const StatCard = ({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) => (
  <div className="rounded-2xl border border-slate-800/80 bg-slate-950/50 px-4 py-3">
    <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <div className="mt-1 text-lg font-semibold text-slate-100">{value}</div>
  </div>
);

export const AuditLogPage: React.FC = () => {
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const fetchLogs = useCallback(async (isManualRefresh = false) => {
    if (isManualRefresh) setRefreshing(true);
    else setLoading(true);

    try {
      const response = await getAuditLogs(page, 50);
      setLogs(response.data);
      setTotalPages(response.total_pages);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }, [page]);

  useEffect(() => {
    fetchLogs();
  }, [fetchLogs]);

  const stats = useMemo(() => {
    const completed = logs.filter((log) =>
      log.action.includes("completed"),
    ).length;
    const queued = logs.filter((log) => log.action.includes("queued")).length;
    const settings = logs.filter((log) =>
      log.action.includes("settings"),
    ).length;

    return { completed, queued, settings };
  }, [logs]);

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-slate-800/80 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.10),_transparent_35%),linear-gradient(180deg,rgba(2,6,23,0.96),rgba(2,6,23,0.82))] p-6 shadow-[0_0_0_1px_rgba(15,23,42,0.5)]">
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:32px_32px] opacity-30" />

          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-2 text-[11px] uppercase tracking-[0.28em] text-cyan-300/80">
                Governance
              </div>
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-300">
                  <ShieldCheck size={20} />
                </div>
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-50">
                    Audit Log
                  </h1>
                  <p className="mt-1 text-sm text-slate-400">
                    Immutable record of actions across the evidence lifecycle.
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <Button
                variant="secondary"
                onClick={() => fetchLogs(true)}
                disabled={refreshing}
              >
                <RefreshCw
                  size={16}
                  className={refreshing ? "animate-spin" : ""}
                />
                Refresh
              </Button>
            </div>
          </div>

          <div className="relative mt-6 grid grid-cols-2 gap-3 lg:grid-cols-4">
            <StatCard label="Events on page" value={logs.length} />
            <StatCard label="Completed" value={stats.completed} />
            <StatCard label="Queued" value={stats.queued} />
            <StatCard label="Settings changes" value={stats.settings} />
          </div>
        </section>

        <section className="overflow-hidden rounded-3xl border border-slate-800/80 bg-slate-950/55 shadow-[0_10px_30px_rgba(2,6,23,0.35)]">
          <div className="border-b border-slate-800/80 px-4 py-3 md:px-6">
            <div className="grid grid-cols-12 gap-4 text-[11px] uppercase tracking-[0.2em] text-slate-500">
              <div className="col-span-12 md:col-span-3">Timestamp</div>
              <div className="col-span-12 md:col-span-2">Action</div>
              <div className="col-span-12 md:col-span-5">Record</div>
              <div className="col-span-12 md:col-span-2">Actor</div>
            </div>
          </div>

          {loading ? (
            <div className="px-6 py-10 text-sm text-slate-500">
              Loading audit log…
            </div>
          ) : logs.length === 0 ? (
            <div className="px-6 py-10 text-sm text-slate-500">
              No audit events found.
            </div>
          ) : (
            <div className="divide-y divide-slate-800/80">
              {logs.map((entry) => {
                const record = formatRecord(entry);
                const detailsJson = formatDetailsJson(entry.details);

                return (
                  <article
                    key={entry.id}
                    className="group px-4 py-4 transition-colors hover:bg-slate-900/35 md:px-6"
                  >
                    <div className="grid grid-cols-12 gap-4">
                      <div className="col-span-12 md:col-span-3">
                        <div className="text-sm text-slate-200">
                          {formatDate(entry.created_at)}
                        </div>
                        <div className="mt-1 text-xs text-slate-500">
                          Event #{entry.id}
                        </div>
                      </div>

                      <div className="col-span-12 md:col-span-2">
                        <span
                          className={`inline-flex rounded-full border px-2.5 py-1 text-xs font-medium ${getActionTone(
                            entry.action,
                          )}`}
                        >
                          {getActionLabel(entry.action)}
                        </span>
                      </div>

                      <div className="col-span-12 md:col-span-5 min-w-0">
                        <div className="flex items-start gap-3">
                          <div className="mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-slate-800 bg-slate-900 text-cyan-300">
                            {getRecordIcon(entry.record_type)}
                          </div>

                          <div className="min-w-0 flex-1">
                            <div className="flex flex-wrap items-center gap-2">
                              <div className="truncate text-sm font-semibold text-slate-100">
                                {record.path ? (
                                  <Link
                                    to={record.path}
                                    className="underline-offset-4 hover:text-cyan-200 hover:underline"
                                  >
                                    {record.label}
                                  </Link>
                                ) : (
                                  record.label
                                )}
                              </div>

                              {entry.record_type && (
                                <span className="rounded-md border border-slate-800 bg-slate-900/70 px-2 py-0.5 text-[10px] uppercase tracking-[0.18em] text-slate-400">
                                  {entry.record_type}
                                </span>
                              )}
                            </div>

                            {entry.filename && (
                              <div className="mt-1 truncate text-sm text-slate-400">
                                {entry.filename}
                              </div>
                            )}

                            <p className="mt-2 break-words text-sm leading-6 text-slate-500">
                              {formatDetailsPreview(entry.details)}
                            </p>

                            {detailsJson && (
                              <details className="mt-3 rounded-2xl border border-slate-800/70 bg-slate-950/70 open:bg-slate-950">
                                <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-400 transition hover:text-cyan-300">
                                  <span>Raw details</span>
                                  <ChevronDown
                                    size={14}
                                    className="transition group-open:rotate-180"
                                  />
                                </summary>
                                <div className="border-t border-slate-800/70 px-3 py-3">
                                  <pre className="max-h-64 overflow-auto text-xs leading-6 text-slate-400">
                                    {detailsJson}
                                  </pre>
                                </div>
                              </details>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="col-span-12 md:col-span-2">
                        <div className="rounded-2xl border border-slate-800/70 bg-slate-900/50 px-3 py-2 text-sm text-slate-300">
                          {entry.actor || "unknown"}
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
          )}
        </section>

        <div className="flex flex-col gap-3 rounded-2xl border border-slate-800/80 bg-slate-950/45 px-4 py-4 md:flex-row md:items-center md:justify-between">
          <p className="text-sm text-slate-500">
            Page <span className="text-slate-300">{page}</span> of{" "}
            <span className="text-slate-300">{totalPages}</span>
          </p>

          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              disabled={page <= 1}
              onClick={() => setPage((p) => p - 1)}
            >
              Previous
            </Button>
            <Button
              variant="secondary"
              disabled={page >= totalPages}
              onClick={() => setPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
};
