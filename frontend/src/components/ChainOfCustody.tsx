import React from "react";

interface Step {
  label: string;
  timestamp?: string | null;
  status: "complete" | "pending";
}

interface Props {
  steps: Step[];
}

const formatUtc = (value?: string | null) => {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString("en-GB", {
    timeZone: "UTC",
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
  });
};

export const ChainOfCustody: React.FC<Props> = ({ steps }) => {
  return (
    <div className="rounded-3xl border border-slate-800/80 bg-slate-950/55 p-5 shadow-[0_10px_30px_rgba(2,6,23,0.28)]">
      <h3 className="mb-4 text-[11px] uppercase tracking-[0.2em] text-slate-500">
        Chain of Custody
      </h3>

      <div className="space-y-4">
        {steps.map((step, idx) => (
          <div
            key={`${step.label}-${idx}`}
            className="relative flex items-start gap-3"
          >
            {idx < steps.length - 1 ? (
              <div className="absolute left-[5px] top-4 h-[calc(100%+12px)] w-px bg-slate-700" />
            ) : null}

            <div
              className={`relative z-10 mt-1 h-3 w-3 rounded-full border ${
                step.status === "complete"
                  ? "border-emerald-400 bg-emerald-400/80"
                  : "border-slate-600 bg-slate-700"
              }`}
            />

            <div className="flex-1">
              <div className="text-sm text-slate-200">{step.label}</div>
              <div className="font-mono text-xs text-slate-500">
                {formatUtc(step.timestamp)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
