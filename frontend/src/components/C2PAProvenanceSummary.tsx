import React from "react";
import type { AnalysisResult } from "../types";

export const C2PAProvenanceSummary: React.FC<{
  c2pa: AnalysisResult["c2pa"];
  compact?: boolean;
}> = ({ c2pa, compact = false }) => {
  const summary = summariseC2PA(c2pa);
  const ingredientCount = Array.isArray(c2pa?.ingredients)
    ? c2pa.ingredients.length
    : 0;

  const softwareAgents = Array.isArray(c2pa?.software_agents)
    ? c2pa.software_agents.slice(0, compact ? 2 : 4)
    : [];

  const sourceTypes = Array.isArray(c2pa?.digital_source_types)
    ? c2pa.digital_source_types.slice(0, compact ? 2 : 4)
    : [];

  return (
    <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900/70 p-4">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-[11px] uppercase tracking-[0.18em] text-slate-500">
            C2PA Provenance Summary
          </div>
          <div className="mt-1 text-sm font-semibold text-slate-100">
            {summary.title}
          </div>
        </div>

        <span
          className={`rounded-full border px-2.5 py-1 text-xs font-semibold ${summary.badgeClass}`}
        >
          {summary.badge}
        </span>
      </div>

      <div
        className={`grid gap-4 text-sm ${
          compact ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2"
        }`}
      >
        <KV label="Manifest Present" value={formatBool(c2pa?.has_c2pa)} />
        <KV label="Signature Valid" value={formatBool(c2pa?.signature_valid)} />
        <KV
          label="AI Assertions"
          value={String(c2pa?.ai_assertions_found?.length ?? 0)}
        />
        <KV label="Ingredients" value={String(ingredientCount)} />
        <KV
          label="Claim Generator"
          value={formatUnknown(c2pa?.claim_generator)}
        />
        <KV label="Signer" value={formatUnknown(c2pa?.signer)} />
      </div>

      {softwareAgents.length > 0 ? (
        <div>
          <div className="mb-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
            Software Agents
          </div>
          <div className="flex flex-wrap gap-2">
            {softwareAgents.map((agent) => (
              <span
                key={String(agent)}
                className="rounded-full border border-slate-700 px-2.5 py-1 text-xs text-slate-300"
              >
                {String(agent)}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {sourceTypes.length > 0 ? (
        <div>
          <div className="mb-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
            Digital Source Types
          </div>
          <div className="space-y-1 text-xs text-slate-300">
            {sourceTypes.map((item) => (
              <div key={String(item)} className="break-all">
                {String(item)}
              </div>
            ))}
          </div>
        </div>
      ) : null}
    </div>
  );
};

function summariseC2PA(c2pa: AnalysisResult["c2pa"]) {
  const hasC2PA = Boolean(c2pa?.has_c2pa);
  const signatureValid = c2pa?.signature_valid === true;
  const aiAssertions = c2pa?.ai_assertions_found?.length ?? 0;

  if (!hasC2PA) {
    return {
      title: "No C2PA manifest detected",
      badge: "No C2PA",
      badgeClass: "border-slate-600 text-slate-400",
    };
  }

  if (!signatureValid) {
    return {
      title: "C2PA manifest present, but signature could not be validated",
      badge: "Present / Invalid",
      badgeClass: "border-amber-500/30 bg-amber-500/10 text-amber-300",
    };
  }

  if (aiAssertions > 0) {
    return {
      title: "Valid C2PA manifest with AI-related assertions",
      badge: "Valid / AI-related",
      badgeClass: "border-rose-500/30 bg-rose-500/10 text-rose-300",
    };
  }

  return {
    title: "Valid C2PA manifest without explicit AI-related assertions",
    badge: "Valid",
    badgeClass: "border-emerald-500/30 bg-emerald-500/10 text-emerald-300",
  };
}

const KV = ({ label, value }: { label: string; value: string }) => (
  <div>
    <div className="text-xs uppercase tracking-[0.16em] text-slate-500">
      {label}
    </div>
    <div className="mt-1 break-all text-slate-200">{value}</div>
  </div>
);

function formatBool(value: unknown) {
  if (value === true) return "Yes";
  if (value === false) return "No";
  return "Unavailable";
}

function formatUnknown(value: unknown) {
  if (typeof value === "string" && value.trim()) return value;
  return "Unavailable";
}
