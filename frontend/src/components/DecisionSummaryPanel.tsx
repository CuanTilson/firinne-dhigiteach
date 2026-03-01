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
    <div className="fd-card p-5 space-y-4">
      <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
        <div>
          <div className="fd-section-title">{title}</div>
          <div className="mt-2 text-xl font-semibold text-slate-100">{verdict}</div>
        </div>
        {scoreLabel && scoreValue ? (
          <div className="fd-panel px-4 py-3 min-w-[180px]">
            <div className="text-xs uppercase tracking-wider text-slate-500">
              {scoreLabel}
            </div>
            <div className="mt-1 text-lg font-semibold text-slate-100">{scoreValue}</div>
          </div>
        ) : null}
      </div>

      <div>
        <div className="text-xs uppercase tracking-wider text-slate-500 mb-2">
          Basis for Current Decision
        </div>
        <ul className="space-y-2 text-sm text-slate-300">
          {rationale.map((item, index) => (
            <li key={`${item}-${index}`}>- {item}</li>
          ))}
        </ul>
      </div>

      {note ? <div className="text-sm text-slate-400">{note}</div> : null}
    </div>
  );
};
