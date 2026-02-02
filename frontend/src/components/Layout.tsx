import React, { useEffect, useState } from "react";
import { NavLink, Outlet, useNavigate, useLocation } from "react-router-dom";
import { ScanFace, LayoutDashboard, History, X, Film } from "lucide-react";
import { checkBackend, getVideoJob } from "../services/api";

export const Layout: React.FC = () => {
  const [online, setOnline] = useState<boolean | null>(null);
  const [jobToast, setJobToast] = useState<{
    jobId: string;
    status: string;
    filename?: string;
    resultId?: number;
  } | null>(null);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    const runCheck = async () => {
      const ok = await checkBackend();
      setOnline(ok);
    };

    runCheck();

    // optional: repeat check every 10 seconds
    const interval = setInterval(runCheck, 10000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const readJob = () => {
      try {
        const raw = localStorage.getItem("fd_video_job");
        if (!raw) return null;
        return JSON.parse(raw) as {
          jobId: string;
          status: string;
          filename?: string;
          resultId?: number;
          mediaType?: "video" | "image";
          previewUrl?: string | null;
        };
      } catch {
        return null;
      }
    };

    const writeJob = (value: Record<string, unknown> | null) => {
      try {
        if (!value) {
          localStorage.removeItem("fd_video_job");
        } else {
          localStorage.setItem("fd_video_job", JSON.stringify(value));
        }
      } catch {
        // ignore storage errors
      }
    };

    const tick = async () => {
      const job = readJob();
      if (!job?.jobId) {
        setJobToast(null);
        return;
      }

      setJobToast(job);
      if (job.status === "completed" || job.status === "failed") {
        return;
      }

      try {
        const status = await getVideoJob(job.jobId);
        const updated = {
          jobId: job.jobId,
          status: status.status,
          filename: status.filename || job.filename,
          resultId: status.result?.id ?? job.resultId,
          mediaType: job.mediaType,
          previewUrl: job.previewUrl,
        };
        writeJob(updated);
        setJobToast(updated);
      } catch {
        // keep last known state
      }
    };

    tick();
    const interval = window.setInterval(tick, 2000);
    return () => window.clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-200 flex flex-col md:flex-row">
      <aside className="w-full md:w-64 bg-slate-900 border-b md:border-b-0 md:border-r border-slate-800 shrink-0">
        <div className="p-6 border-b border-slate-800 flex items-center gap-3">
          <div className="bg-cyan-500/20 p-2 rounded-lg">
            <ScanFace className="text-cyan-400" size={24} />
          </div>
          <span className="font-bold text-lg tracking-wide text-white">
            Fírinne Dhigiteach
          </span>
        </div>

        <nav className="p-4 space-y-2">
          <NavLink
            to="/"
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors font-medium ${
                isActive
                  ? "bg-cyan-900/30 text-cyan-400 border border-cyan-500/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`
            }
          >
            <LayoutDashboard size={20} />
            <span>Detector</span>
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) =>
              `flex items-center gap-3 px-4 py-3 rounded-lg transition-colors font-medium ${
                isActive
                  ? "bg-cyan-900/30 text-cyan-400 border border-cyan-500/30"
                  : "text-slate-400 hover:text-white hover:bg-slate-800"
              }`
            }
          >
            <History size={20} />
            <span>Case History</span>
          </NavLink>
        </nav>

        <div className="mt-auto p-6 hidden md:block">
          <div className="bg-slate-800/50 rounded-lg p-4 text-xs text-slate-500 border border-slate-800">
            <p className="font-semibold text-slate-400 mb-1">System Status</p>

            {online === null ? (
              <div className="flex items-center gap-2 opacity-60">
                <span className="w-2 h-2 bg-yellow-500 rounded-full animate-pulse"></span>
                Checking...
              </div>
            ) : online ? (
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></span>
                Backend Online
              </div>
            ) : (
              <div className="flex items-center gap-2 text-red-400">
                <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></span>
                Offline
              </div>
            )}
          </div>
        </div>
      </aside>

      <main className="grow overflow-auto bg-slate-950">
        <Outlet />
      </main>

      {jobToast && location.pathname !== "/" && (
        <div className="fixed bottom-6 right-6 z-50 w-[320px] bg-slate-900 border border-slate-700 rounded-xl shadow-xl p-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm text-slate-300 flex items-center gap-2">
                <Film size={14} className="text-cyan-400" />
                Video Analysis
              </div>
              <div className="text-xs text-slate-500 mt-1">
                {jobToast.filename || "Processing video"}
              </div>
            </div>
            <button
              onClick={() => {
                setJobToast(null);
              }}
              className="text-slate-400 hover:text-slate-200"
              title="Dismiss"
            >
              <X size={16} />
            </button>
          </div>
          <div className="mt-3 text-sm text-slate-300">
            Status:{" "}
            <span className="text-cyan-300 font-medium">
              {jobToast.status}
            </span>
          </div>
          {jobToast.status === "completed" && jobToast.resultId && (
            <button
              onClick={() => navigate(`/videos/${jobToast.resultId}`)}
              className="mt-3 w-full px-3 py-2 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white text-sm font-medium"
            >
              View Results
            </button>
          )}
        </div>
      )}
    </div>
  );
};
