import React, { useEffect, useState } from "react";
import { NavLink, Outlet } from "react-router-dom";
import { ScanFace, LayoutDashboard, History } from "lucide-react";
import { checkBackend } from "../services/api";

export const Layout: React.FC = () => {
  const [online, setOnline] = useState<boolean | null>(null);

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
    </div>
  );
};
