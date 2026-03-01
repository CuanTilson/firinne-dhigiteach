import React from "react";

interface EvidenceItem {
  label: string;
  status: string;
  tone?: "neutral" | "good" | "warn" | "bad";
  detail?: string;
}

interface Props {
  items: EvidenceItem[];
}

const toneClass: Record<NonNullable<EvidenceItem["tone"]>, string> = {
  neutral: "border-slate-800 bg-slate-950/60 text-slate-200",
  good: "border-emerald-500/30 bg-emerald-500/10 text-emerald-200",
  warn: "border-amber-500/30 bg-amber-500/10 text-amber-200",
  bad: "border-rose-500/30 bg-rose-500/10 text-rose-200",
};

export const EvidenceStatusStrip: React.FC<Props> = ({ items }) => (
  <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-3">
    {items.map((item) => {
      const tone = item.tone ?? "neutral";
      return (
        <div
          key={item.label}
          className={`rounded-lg border p-3 ${toneClass[tone]}`}
        >
          <div className="text-[11px] uppercase tracking-wider opacity-70">
            {item.label}
          </div>
          <div className="mt-1 text-sm font-semibold">{item.status}</div>
          {item.detail ? (
            <div className="mt-1 text-xs opacity-80">{item.detail}</div>
          ) : null}
        </div>
      );
    })}
  </div>
);
