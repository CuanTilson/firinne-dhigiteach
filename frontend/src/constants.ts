import type { ClassificationType } from "./types";

export const API_BASE_URL = "http://localhost:8000";

export const API_ENDPOINTS = {
  DETECT: `${API_BASE_URL}/analysis/image`,
  RECORDS: `${API_BASE_URL}/analysis`,
  STATIC: `${API_BASE_URL}/static`,
};

export const CLASSIFICATION_COLORS: Record<ClassificationType, string> = {
  likely_real: "text-green-400 bg-green-400/10 border-green-400/20",
  likely_ai_generated: "text-red-400 bg-red-400/10 border-red-400/20",
  uncertain: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
};

export const CLASSIFICATION_LABELS: Record<ClassificationType, string> = {
  likely_real: "Likely Authentic",
  likely_ai_generated: "Likely AI Generated",
  uncertain: "Uncertain / Inconclusive",
};

export const fixPath = (p?: string | null) => {
  if (!p || typeof p !== "string") return "";

  let clean = p.replace(/\\/g, "/");
  clean = clean.replace("backend/storage/", "");

  if (clean.startsWith("uploaded/"))
    return `${API_BASE_URL}/uploaded/${clean.replace("uploaded/", "")}`;

  if (clean.startsWith("ela/"))
    return `${API_BASE_URL}/ela/${clean.replace("ela/", "")}`;

  if (clean.startsWith("heatmaps/"))
    return `${API_BASE_URL}/heatmaps/${clean.replace("heatmaps/", "")}`;

  return `${API_BASE_URL}/${clean}`;
};


