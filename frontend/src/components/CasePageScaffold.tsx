import React from "react";
import { Link } from "react-router-dom";
import { ChevronLeft } from "lucide-react";

interface Props {
  backTo: string;
  backLabel: string;
  header: React.ReactNode;
  statusStrip?: React.ReactNode;
  sidebar?: React.ReactNode;
  children: React.ReactNode;
}

export const CasePageScaffold: React.FC<Props> = ({
  backTo,
  backLabel,
  header,
  statusStrip,
  sidebar,
  children,
}) => {
  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center gap-4">
        <Link
          to={backTo}
          className="p-2 bg-slate-900 rounded-full hover:bg-slate-800 text-slate-300 transition-colors border border-slate-800"
        >
          <ChevronLeft size={20} />
        </Link>
        <span className="text-sm text-slate-400">{backLabel}</span>
      </div>

      {header}
      {statusStrip}

      <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1.7fr)_360px] gap-6 items-start">
        <div className="space-y-6 min-w-0">{children}</div>
        {sidebar ? (
          <aside className="space-y-6 xl:sticky xl:top-6">{sidebar}</aside>
        ) : null}
      </div>
    </div>
  );
};
