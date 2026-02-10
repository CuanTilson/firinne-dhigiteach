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
    <div className="fd-card p-5">
      <h3 className="fd-section-title mb-4">Chain of Custody</h3>
      <div className="space-y-3">
        {steps.map((step, idx) => (
          <div
            key={`${step.label}-${idx}`}
            className="flex items-start gap-3"
          >
            <div
              className={`mt-1 h-3 w-3 rounded-full border ${
                step.status === "complete"
                  ? "bg-emerald-400/80 border-emerald-400"
                  : "bg-slate-700 border-slate-600"
              }`}
            ></div>
            <div className="flex-1">
              <div className="text-sm text-slate-200">{step.label}</div>
              <div className="text-xs text-slate-500 font-mono">
                {formatUtc(step.timestamp)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
