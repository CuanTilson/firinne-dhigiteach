import React, { useEffect, useMemo, useState, useCallback } from "react";
import { useSearchParams } from "react-router-dom";
import { getRecords, deleteRecord } from "../services/api";
import type {
  AnalysisRecordSummary,
  ClassificationType,
  MediaType,
  RecordFilters,
} from "../types";
import { CaseList } from "../components/CaseList";
import { Button } from "../components/ui/Button";
import {
  AlertCircle,
  Filter,
  RefreshCw,
  Search,
  ShieldCheck,
  X,
} from "lucide-react";
import { DEFAULT_ADMIN_KEY } from "../constants";

interface HistoryPageProps {
  initialMediaType?: MediaType | "";
  title?: string;
  description?: string;
}

const getScopedMediaType = (
  queryValue: string | null,
  fallback: MediaType | ""
): MediaType | "" =>
  queryValue === "image" || queryValue === "video" || queryValue === "audio"
    ? queryValue
    : fallback;

const SummaryStat = ({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) => (
  <div className="rounded-2xl border border-slate-800/80 bg-slate-950/55 px-4 py-3">
    <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
      {label}
    </div>
    <div className="mt-1 text-sm font-semibold text-slate-100">{value}</div>
  </div>
);

export const HistoryPage: React.FC<HistoryPageProps> = ({
  initialMediaType = "",
  title = "Analysis History",
  description = "Archive of all forensic investigations.",
}) => {
  const [searchParams] = useSearchParams();

  const queryMediaType = searchParams.get("media_type");
  const scopedMediaType = getScopedMediaType(queryMediaType, initialMediaType);

  const [records, setRecords] = useState<AnalysisRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [adminKey, setAdminKey] = useState<string>(DEFAULT_ADMIN_KEY);

  const [searchInput, setSearchInput] = useState("");
  const [filename, setFilename] = useState("");
  const [classification, setClassification] = useState<ClassificationType | "">(
    ""
  );
  const [mediaType, setMediaType] = useState<MediaType | "">(scopedMediaType);
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      setFilename(searchInput.trim());
    }, 300);

    return () => window.clearTimeout(timeout);
  }, [searchInput]);

  useEffect(() => {
    setMediaType(scopedMediaType);
    setPage(1);
  }, [scopedMediaType]);

  useEffect(() => {
    setPage(1);
  }, [filename, classification, mediaType, dateFrom, dateTo]);

  const hasInvalidDateRange = Boolean(dateFrom && dateTo && dateFrom > dateTo);

  const activeFilterCount = useMemo(() => {
    return [
      Boolean(filename),
      Boolean(classification),
      Boolean(mediaType),
      Boolean(dateFrom),
      Boolean(dateTo),
    ].filter(Boolean).length;
  }, [filename, classification, mediaType, dateFrom, dateTo]);

  const fetchData = useCallback(
    async (isManualRefresh = false) => {
      if (hasInvalidDateRange) {
        setError("The date range is invalid. ‘From’ must be before ‘To’."); 
        setRecords([]);
        setTotalPages(1);
        setLoading(false);
        setRefreshing(false);
        return;
      }

      if (isManualRefresh) setRefreshing(true);
      else setLoading(true);

      setError(null);

      try {
        const filters: RecordFilters = {
          filename,
          classification,
          media_type: mediaType,
          date_from: dateFrom,
          date_to: dateTo,
        };

        const response = await getRecords(page, 20, filters);
        setRecords(response.data);
        setTotalPages(response.total_pages);
      } catch (fetchError) {
        console.error(fetchError);
        setError("Could not load analysis history.");
      } finally {
        setLoading(false);
        setRefreshing(false);
      }
    },
    [page, filename, classification, mediaType, dateFrom, dateTo, hasInvalidDateRange]
  );

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const clearFilters = () => {
    setSearchInput("");
    setFilename("");
    setClassification("");
    setMediaType(scopedMediaType);
    setDateFrom("");
    setDateTo("");
    setPage(1);
  };

  const handleDelete = async (
    id: number,
    recordMediaType: AnalysisRecordSummary["media_type"]
  ) => {
    if (!window.confirm("Are you sure you want to delete this record?")) return;

    let key = adminKey;
    if (!key) {
      const entered = window.prompt("Enter admin key to delete this record:");
      if (!entered) return;
      key = entered;
      setAdminKey(entered);
    }

    try {
      await deleteRecord(id, key, recordMediaType);
      fetchData(true);
    } catch (deleteError) {
      console.error(deleteError);
      window.alert("Failed to delete record.");
    }
  };

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
      <div className="space-y-6">
        <section className="relative overflow-hidden rounded-3xl border border-slate-800/80 bg-[radial-gradient(circle_at_top,_rgba(56,189,248,0.10),_transparent_35%),linear-gradient(180deg,rgba(2,6,23,0.96),rgba(2,6,23,0.82))] p-6 shadow-[0_0_0_1px_rgba(15,23,42,0.5)]">
          <div className="pointer-events-none absolute inset-0 bg-[linear-gradient(to_right,rgba(148,163,184,0.04)_1px,transparent_1px),linear-gradient(to_bottom,rgba(148,163,184,0.04)_1px,transparent_1px)] bg-[size:32px_32px] opacity-30" />

          <div className="relative flex flex-col gap-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <div className="mb-2 text-[11px] uppercase tracking-[0.28em] text-cyan-300/80">
                Archive
              </div>
              <div className="flex items-center gap-3">
                <div className="flex h-11 w-11 items-center justify-center rounded-2xl border border-cyan-500/20 bg-cyan-500/10 text-cyan-300">
                  <ShieldCheck size={20} />
                </div>
                <div>
                  <h1 className="text-3xl font-semibold tracking-tight text-slate-50">
                    {title}
                  </h1>
                  <p className="mt-1 text-sm text-slate-400">{description}</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
              <SummaryStat label="Results on page" value={records.length} />
              <SummaryStat label="Active filters" value={activeFilterCount} />
              <SummaryStat
                label="Scope"
                value={scopedMediaType ? scopedMediaType : "All media"}
              />
            </div>
          </div>
        </section>

        <section className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-4 shadow-[0_10px_30px_rgba(2,6,23,0.28)] md:p-5">
          <div className="flex flex-col gap-4">
            <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.4fr)_180px_180px_160px_160px]">
              <div>
                <label className="mb-2 block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  Filename
                </label>
                <div className="relative">
                  <Search
                    className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                    size={16}
                  />
                  <input
                    type="text"
                    value={searchInput}
                    onChange={(e) => setSearchInput(e.target.value)}
                    placeholder="Search files..."
                    className="w-full rounded-xl border border-slate-700 bg-slate-950 px-10 py-2.5 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
                  />
                </div>
              </div>

              <div>
                <label className="mb-2 block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  Media Type
                </label>
                <div className="relative">
                  <Filter
                    className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                    size={16}
                  />
                  <select
                    value={mediaType}
                    onChange={(e) =>
                      setMediaType(e.target.value as MediaType | "")
                    }
                    disabled={scopedMediaType !== ""}
                    className="w-full appearance-none rounded-xl border border-slate-700 bg-slate-950 px-10 py-2.5 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20 disabled:cursor-not-allowed disabled:opacity-70"
                  >
                    <option value="">All Media</option>
                    <option value="image">Images</option>
                    <option value="video">Videos</option>
                    <option value="audio">Audio</option>
                  </select>
                </div>
                {scopedMediaType ? (
                  <div className="mt-1 text-[11px] text-slate-500">
                    This view is scoped to {scopedMediaType} records.
                  </div>
                ) : null}
              </div>

              <div>
                <label className="mb-2 block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  Classification
                </label>
                <div className="relative">
                  <Filter
                    className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                    size={16}
                  />
                  <select
                    value={classification}
                    onChange={(e) =>
                      setClassification(e.target.value as ClassificationType | "")
                    }
                    className="w-full appearance-none rounded-xl border border-slate-700 bg-slate-950 px-10 py-2.5 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
                  >
                    <option value="">All Classifications</option>
                    <option value="likely_real">Likely Real</option>
                    <option value="likely_ai_generated">Likely AI</option>
                    <option value="uncertain">Uncertain</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="mb-2 block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  From
                </label>
                <input
                  type="date"
                  value={dateFrom}
                  onChange={(e) => setDateFrom(e.target.value)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
                />
              </div>

              <div>
                <label className="mb-2 block text-[11px] uppercase tracking-[0.18em] text-slate-500">
                  To
                </label>
                <input
                  type="date"
                  value={dateTo}
                  onChange={(e) => setDateTo(e.target.value)}
                  className="w-full rounded-xl border border-slate-700 bg-slate-950 px-3 py-2.5 text-sm text-slate-100 outline-none transition focus:border-cyan-400/60 focus:ring-2 focus:ring-cyan-400/20"
                />
              </div>
            </div>

            <div className="flex flex-col gap-3 border-t border-slate-800/80 pt-4 md:flex-row md:items-center md:justify-between">
              <div className="text-sm text-slate-500">
                {activeFilterCount > 0 ? (
                  <span>
                    {activeFilterCount} filter{activeFilterCount === 1 ? "" : "s"} active
                  </span>
                ) : (
                  <span>No filters applied</span>
                )}
              </div>

              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="secondary"
                  onClick={clearFilters}
                  disabled={
                    !searchInput &&
                    !filename &&
                    !classification &&
                    !dateFrom &&
                    !dateTo &&
                    mediaType === scopedMediaType
                  }
                >
                  <X size={16} />
                  Clear filters
                </Button>

                <Button
                  variant="secondary"
                  onClick={() => fetchData(true)}
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

            {hasInvalidDateRange ? (
              <div className="rounded-2xl border border-amber-500/25 bg-amber-500/10 px-4 py-3 text-sm text-amber-200">
                The selected date range is invalid. Please make sure the “From”
                date is before the “To” date.
              </div>
            ) : null}

            {error ? (
              <div className="rounded-2xl border border-red-500/20 bg-red-500/10 px-4 py-3 text-sm text-red-300">
                <div className="flex items-center gap-2">
                  <AlertCircle size={16} />
                  {error}
                </div>
              </div>
            ) : null}
          </div>
        </section>

        <CaseList records={records} loading={loading} onDelete={handleDelete} />

        <div className="flex flex-col gap-3 rounded-2xl border border-slate-800/80 bg-slate-950/45 px-4 py-4 md:flex-row md:items-center md:justify-between">
          <p className="text-sm text-slate-500">
            Page <span className="text-slate-300">{page}</span> of{" "}
            <span className="text-slate-300">{totalPages}</span>
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
    </div>
  );
};