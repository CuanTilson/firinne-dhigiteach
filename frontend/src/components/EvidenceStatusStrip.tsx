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

const dotClass: Record<NonNullable<EvidenceItem["tone"]>, string> = {
  neutral: "bg-slate-400",
  good: "bg-emerald-300",
  warn: "bg-amber-300",
  bad: "bg-rose-300",
};

export const EvidenceStatusStrip: React.FC<Props> = ({ items }) => (
  <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
    {items.map((item) => {
      const tone = item.tone ?? "neutral";

      return (
        <div
          key={item.label}
          className={`rounded-2xl border p-3 ${toneClass[tone]}`}
        >
          <div className="text-[11px] uppercase tracking-[0.16em] opacity-70">
            {item.label}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <span className={`h-2 w-2 rounded-full ${dotClass[tone]}`} />
            <div className="text-sm font-semibold">{item.status}</div>
          </div>
          {item.detail ? (
            <div className="mt-1 text-xs opacity-80">{item.detail}</div>
          ) : null}
        </div>
      );
    })}
  </div>
);
