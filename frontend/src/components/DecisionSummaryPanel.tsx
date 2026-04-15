import React from "react";

interface Props {
  title?: string;
  verdict: string;
  scoreLabel?: string;
  scoreValue?: string;
  rationale: string[];
  note?: string;
}

export const DecisionSummaryPanel: React.FC<Props> = ({
  title = "Decision Summary",
  verdict,
  scoreLabel,
  scoreValue,
  rationale,
  note,
}) => {
  return (
    <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-5 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <div>
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">
              {title}
            </div>
            <div className="mt-2 text-xl font-semibold text-slate-100">
              {verdict}
            </div>
          </div>

          {scoreLabel && scoreValue ? (
            <div className="min-w-[180px] rounded-2xl border border-slate-800 bg-slate-950/70 px-4 py-3">
              <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
                {scoreLabel}
              </div>
              <div className="mt-1 text-lg font-semibold text-slate-100">
                {scoreValue}
              </div>
            </div>
          ) : null}
        </div>

        <div>
          <div className="mb-2 text-xs uppercase tracking-[0.16em] text-slate-500">
            Basis for Current Decision
          </div>
          <ul className="space-y-2 text-sm text-slate-300">
            {rationale.map((item, index) => (
              <li key={`${item}-${index}`} className="leading-6">
                - {item}
              </li>
            ))}
          </ul>
        </div>

        {note ? <div className="text-sm text-slate-400">{note}</div> : null}
      </div>
    </div>
  );
};
