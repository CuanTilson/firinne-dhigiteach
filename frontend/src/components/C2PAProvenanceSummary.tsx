import React from "react";
import type { AnalysisResult } from "../types";

export const C2PAProvenanceSummary: React.FC<{
  c2pa: AnalysisResult["c2pa"];
  compact?: boolean;
}> = ({ c2pa, compact = false }) => {
  const summary = summarizeC2PA(c2pa);
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
    <div className="rounded-lg border border-slate-700 bg-slate-900 p-4 space-y-3">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-500">
            C2PA Provenance Summary
          </div>
          <div className="mt-1 text-sm font-semibold text-slate-100">
            {summary.title}
          </div>
        </div>
        <span
          className={`px-2 py-1 rounded-full text-xs font-semibold border ${summary.badgeClass}`}
        >
          {summary.badge}
        </span>
      </div>

      <div
        className={`grid gap-3 text-sm ${
          compact ? "grid-cols-1" : "grid-cols-1 md:grid-cols-2"
        }`}
      >
        <KV label="Manifest Present" value={formatBool(c2pa?.has_c2pa)} />
        <KV label="Signature Valid" value={formatBool(c2pa?.signature_valid)} />
        <KV label="AI Assertions" value={String(c2pa?.ai_assertions_found?.length ?? 0)} />
        <KV label="Ingredients" value={String(ingredientCount)} />
        <KV label="Claim Generator" value={formatUnknown(c2pa?.claim_generator)} />
        <KV label="Signer" value={formatUnknown(c2pa?.signer)} />
      </div>

      {softwareAgents.length > 0 ? (
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-1">
            Software Agents
          </div>
          <div className="flex flex-wrap gap-2">
            {softwareAgents.map((agent) => (
              <span
                key={String(agent)}
                className="px-2 py-1 rounded border border-slate-700 text-xs text-slate-300"
              >
                {String(agent)}
              </span>
            ))}
          </div>
        </div>
      ) : null}

      {sourceTypes.length > 0 ? (
        <div>
          <div className="text-xs uppercase tracking-wider text-slate-500 mb-1">
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

function summarizeC2PA(c2pa: AnalysisResult["c2pa"]) {
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
      badgeClass: "border-amber-500/30 text-amber-300 bg-amber-500/10",
    };
  }

  if (aiAssertions > 0) {
    return {
      title: "Valid C2PA manifest with AI-related assertions",
      badge: "Valid / AI-related",
      badgeClass: "border-red-500/30 text-red-300 bg-red-500/10",
    };
  }

  return {
    title: "Valid C2PA manifest without explicit AI-related assertions",
    badge: "Valid",
    badgeClass: "border-emerald-500/30 text-emerald-300 bg-emerald-500/10",
  };
}

const KV = ({ label, value }: { label: string; value: string }) => (
  <div>
    <div className="text-xs uppercase tracking-wider text-slate-500">{label}</div>
    <div className="mt-1 text-slate-200 break-all">{value}</div>
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
