import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getAuditLogs } from "../services/api";
import type { AuditLogEntry } from "../types";
import { Button } from "../components/ui/Button";
import { RefreshCw } from "lucide-react";

const truncate = (value: string, max: number) =>
  value.length > max ? `${value.slice(0, max)}...` : value;

const formatRecord = (entry: AuditLogEntry) => {
  if (!entry.record_type) return { label: "system" };
  if (!entry.record_id) return { label: entry.record_type };
  const path =
    entry.record_type === "video"
      ? `/videos/${entry.record_id}`
      : `/records/${entry.record_id}`;
  return { label: `${entry.record_type} #${entry.record_id}`, path };
};

const formatDetails = (details?: Record<string, unknown> | null | unknown) => {
  if (!details) return "-";
  if (typeof details !== "object") {
    return truncate(String(details), 140);
  }
  if (Object.keys(details as Record<string, unknown>).length === 0) return "-";
  return truncate(JSON.stringify(details), 140);
};

export const AuditLogPage: React.FC = () => {
  const [logs, setLogs] = useState<AuditLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);

  const fetchLogs = async () => {
    setLoading(true);
    try {
      const response = await getAuditLogs(page, 50);
      setLogs(response.data);
      setTotalPages(response.total_pages);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchLogs();
  }, [page]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="fd-kicker mb-2">Governance</div>
          <h1 className="text-3xl font-semibold text-slate-100 mb-1 fd-title">
            Audit Log
          </h1>
          <p className="text-slate-400">
            Immutable record of actions across the evidence lifecycle.
          </p>
        </div>
        <Button variant="secondary" onClick={fetchLogs}>
          <RefreshCw size={16} /> Refresh
        </Button>
      </div>

      <div className="fd-card p-4">
        <div className="grid grid-cols-1 md:grid-cols-[160px_140px_1fr_120px] gap-3 text-xs uppercase tracking-widest text-slate-500 border-b border-slate-800 pb-2">
          <div>Timestamp</div>
          <div>Action</div>
          <div>Record</div>
          <div>Actor</div>
        </div>

        {loading ? (
          <div className="py-6 text-slate-500">Loading audit log...</div>
        ) : logs.length === 0 ? (
          <div className="py-6 text-slate-500">No audit events found.</div>
        ) : (
          <div className="divide-y divide-slate-800">
            {logs.map((entry) => {
              const record = formatRecord(entry);
              return (
                <div
                  key={entry.id}
                  className="grid grid-cols-1 md:grid-cols-[160px_140px_1fr_120px] gap-3 py-3 text-sm text-slate-300"
                >
                  <div className="text-xs text-slate-400">
                    {new Date(entry.created_at).toLocaleString()}
                  </div>
                  <div className="font-semibold text-cyan-300">
                    {entry.action}
                  </div>
                  <div>
                    <div className="font-medium text-slate-200">
                      {record.path ? (
                        <Link
                          to={record.path}
                          className="text-cyan-300 hover:underline"
                        >
                          {record.label}
                        </Link>
                      ) : (
                        record.label
                      )}
                    </div>
                    {entry.filename && (
                      <div className="text-xs text-slate-500">
                        {truncate(entry.filename, 48)}
                      </div>
                    )}
                    <div className="text-xs text-slate-500">
                    {formatDetails(entry.details)}
                  </div>
                  </div>
                  <div className="text-xs text-slate-500">
                    {entry.actor || "unknown"}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="mt-6 flex justify-between items-center">
        <p className="text-sm text-slate-500">
          Page {page} of {totalPages}
        </p>
        <div className="flex gap-2">
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
  );
};
