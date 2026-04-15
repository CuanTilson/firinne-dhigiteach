import React from "react";

interface DetectorInfo {
  name?: string;
  display_name?: string;
  model_version?: string;
  dataset_version?: string;
  weights?: {
    sha256?: string;
    md5?: string;
  };
}

interface Props {
  detector?: DetectorInfo | null;
  fusionMode?: string | null;
  title?: string;
  compact?: boolean;
}

const formatValue = (value: unknown) => {
  if (typeof value === "string" && value.trim()) return value;
  if (typeof value === "number") return String(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  return "Unavailable";
};

export const AnalysisProvenancePanel: React.FC<Props> = ({
  detector,
  fusionMode,
  title = "Analysis Provenance",
  compact = false,
}) => {
  const containerClass = compact
    ? "rounded-2xl border border-slate-200 p-4 space-y-3"
    : "rounded-3xl border border-slate-800/80 bg-slate-950/55 p-5 shadow-[0_10px_30px_rgba(2,6,23,0.28)]";
  const titleClass = compact
    ? "text-xs uppercase tracking-[0.18em] text-slate-500"
    : "text-[11px] uppercase tracking-[0.2em] text-slate-500";
  const valueClass = compact
    ? "mt-1 break-all text-sm text-slate-800"
    : "mt-1 break-all text-sm text-slate-200";
  const labelClass = "text-xs uppercase tracking-[0.16em] text-slate-500";

  return (
    <div className={containerClass}>
      <div className={titleClass}>{title}</div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Field
          label="Detector"
          value={formatValue(detector?.display_name || detector?.name)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Model Version"
          value={formatValue(detector?.model_version)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Dataset Version"
          value={formatValue(detector?.dataset_version)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
        <Field
          label="Weights SHA-256"
          value={formatValue(detector?.weights?.sha256)}
          labelClass={labelClass}
          valueClass={`${valueClass} font-mono text-[13px]`}
        />
        <Field
          label="Fusion Mode"
          value={formatValue(fusionMode)}
          labelClass={labelClass}
          valueClass={valueClass}
        />
      </div>
    </div>
  );
};

const Field = ({
  label,
  value,
  labelClass,
  valueClass,
}: {
  label: string;
  value: string;
  labelClass: string;
  valueClass: string;
}) => (
  <div>
    <div className={labelClass}>{label}</div>
    <div className={valueClass}>{value}</div>
  </div>
);
