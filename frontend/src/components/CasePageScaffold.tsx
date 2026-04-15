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
    <div className="mx-auto max-w-7xl px-4 py-8 md:px-6">
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Link
            to={backTo}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-800 bg-slate-900 text-slate-300 transition hover:border-slate-700 hover:bg-slate-800 hover:text-slate-100"
            aria-label={backLabel}
          >
            <ChevronLeft size={18} />
          </Link>
          <span className="text-sm text-slate-400">{backLabel}</span>
        </div>

        {header}
        {statusStrip}

        <div className="grid grid-cols-1 items-start gap-6 xl:grid-cols-[minmax(0,1.7fr)_360px]">
          <main className="min-w-0 space-y-6">{children}</main>
          {sidebar ? (
            <aside className="space-y-6 xl:sticky xl:top-8">{sidebar}</aside>
          ) : null}
        </div>
      </div>
    </div>
  );
};
