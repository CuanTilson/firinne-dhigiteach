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
    ? "border border-slate-200 rounded-lg p-4 space-y-3"
    : "fd-card p-4 space-y-3";
  const titleClass = compact
    ? "text-xs uppercase tracking-wider text-slate-500"
    : "fd-section-title";
  const labelClass = "text-xs uppercase tracking-wider text-slate-500";
  const valueClass = compact ? "text-sm text-slate-800" : "text-sm text-slate-200";

  return (
    <div className={containerClass}>
      <div className={titleClass}>{title}</div>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
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
          valueClass={valueClass}
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
    <div className={`${valueClass} mt-1 break-all`}>{value}</div>
  </div>
);
