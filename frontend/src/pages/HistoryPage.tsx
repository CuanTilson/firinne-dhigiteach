import React, { useEffect, useState, useCallback } from "react";
import { getRecords, deleteRecord } from "../services/api";
import type {
  AnalysisRecordSummary,
  ClassificationType,
  RecordFilters,
} from "../types";
import { CaseList } from "../components/CaseList";
import { Button } from "../components/ui/Button";
import { Search, Filter, RefreshCw } from "lucide-react";
import { DEFAULT_ADMIN_KEY } from "../constants";

export const HistoryPage: React.FC = () => {
  const [records, setRecords] = useState<AnalysisRecordSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [adminKey, setAdminKey] = useState<string>(DEFAULT_ADMIN_KEY);

  // Filter state
  const [filename, setFilename] = useState("");
  const [classification, setClassification] = useState<ClassificationType | "">(
    ""
  );
  const [dateFrom, setDateFrom] = useState("");
  const [dateTo, setDateTo] = useState("");

  const fetchData = useCallback(async () => {
    setLoading(true);
    try {
      const filters: RecordFilters = {
        filename,
        classification,
        date_from: dateFrom,
        date_to: dateTo,
      };
      const response = await getRecords(page, 20, filters);
      setRecords(response.data);
      setTotalPages(response.total_pages);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  }, [page, filename, classification, dateFrom, dateTo]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const handleDelete = async (
    id: number,
    mediaType: AnalysisRecordSummary["media_type"]
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
      await deleteRecord(id, key, mediaType);
      fetchData(); // Refresh
    } catch {
      alert("Failed to delete");
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="fd-kicker mb-2">Archive</div>
          <h1 className="text-3xl font-semibold text-slate-100 mb-1 fd-title">
            Analysis History
          </h1>
          <p className="text-slate-400">
            Archive of all forensic investigations.
          </p>
        </div>
        <Button variant="secondary" onClick={fetchData}>
          <RefreshCw size={16} /> Refresh
        </Button>
      </div>

      {/* Filters Toolbar */}
      <div className="fd-card p-4">
        <div className="flex flex-col md:flex-row gap-4 items-end">
          <div className="grow w-full md:w-auto">
            <label className="text-xs font-semibold text-slate-400 mb-1 block">
              Filename
            </label>
            <div className="relative">
              <Search
                className="absolute left-3 top-2.5 text-slate-500"
                size={16}
              />
              <input
                type="text"
                value={filename}
                onChange={(e) => setFilename(e.target.value)}
                placeholder="Search files..."
                className="w-full bg-slate-900 border border-slate-700 rounded-lg py-2 pl-10 pr-4 text-sm text-white focus:ring-2 focus:ring-cyan-500 focus:outline-none"
              />
            </div>
          </div>

          <div className="w-full md:w-48">
            <label className="text-xs font-semibold text-slate-400 mb-1 block">
              Classification
            </label>
            <div className="relative">
              <Filter
                className="absolute left-3 top-2.5 text-slate-500"
                size={16}
              />
              <select
                value={classification}
                onChange={(e) =>
                  setClassification(e.target.value as ClassificationType | "")
                }
                className="w-full bg-slate-900 border border-slate-700 rounded-lg py-2 pl-10 pr-4 text-sm text-white focus:ring-2 focus:ring-cyan-500 focus:outline-none appearance-none"
              >
                <option value="">All Types</option>
                <option value="likely_real">Likely Real</option>
                <option value="likely_ai_generated">Likely AI</option>
                <option value="uncertain">Uncertain</option>
              </select>
            </div>
          </div>

          <div className="flex gap-2 w-full md:w-auto">
            <div>
              <label className="text-xs font-semibold text-slate-400 mb-1 block">
                From
              </label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg py-2 px-3 text-sm text-white focus:ring-2 focus:ring-cyan-500 focus:outline-none"
              />
            </div>
            <div>
              <label className="text-xs font-semibold text-slate-400 mb-1 block">
                To
              </label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg py-2 px-3 text-sm text-white focus:ring-2 focus:ring-cyan-500 focus:outline-none"
              />
            </div>
          </div>
        </div>
      </div>

      <CaseList
        records={records}
        loading={loading}
        onDelete={handleDelete}
      />

      {/* Simple Pagination */}
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
