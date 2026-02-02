import type { ClassificationType } from "./types";

export const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
export const API_KEY = import.meta.env.VITE_API_KEY ?? "";
export const DEFAULT_ADMIN_KEY = import.meta.env.VITE_ADMIN_KEY ?? "";

export const API_ENDPOINTS = {
  DETECT: `${API_BASE_URL}/analysis/image`,
  DETECT_VIDEO: `${API_BASE_URL}/analysis/video`,
  DETECT_VIDEO_ASYNC: `${API_BASE_URL}/analysis/video/async`,
  RECORDS: `${API_BASE_URL}/analysis`,
  VIDEO_RECORDS: `${API_BASE_URL}/analysis/video`,
  JOBS: `${API_BASE_URL}/jobs`,
  STATIC: `${API_BASE_URL}/static`,
};

export const CLASSIFICATION_COLORS: Record<ClassificationType, string> = {
  likely_real: "text-green-400 bg-green-400/10 border-green-400/20",
  likely_ai_generated: "text-red-400 bg-red-400/10 border-red-400/20",
  uncertain: "text-yellow-400 bg-yellow-400/10 border-yellow-400/20",
  ai_generated_c2pa_flagged: "text-red-400 bg-red-700/20 border-red-500/30",
};

export const CLASSIFICATION_LABELS: Record<ClassificationType, string> = {
  likely_real: "Likely Authentic",
  likely_ai_generated: "Likely AI Generated",
  uncertain: "Uncertain / Inconclusive",
  ai_generated_c2pa_flagged: "AI (C2PA Flagged)",
};

export const fixPath = (p?: string | null): string | undefined => {
  if (!p || typeof p !== "string") return undefined;

  // normalise backslashes
  let clean = p.replace(/\\/g, "/");

  // strip any leading system path up to backend/storage/
  clean = clean.replace(/.*backend\/storage\//, "");

  if (clean.startsWith("uploaded/")) {
    return `${API_BASE_URL}/uploaded/${clean.slice("uploaded/".length)}`;
  }

  if (clean.startsWith("ela/")) {
    return `${API_BASE_URL}/ela/${clean.slice("ela/".length)}`;
  }

  if (clean.startsWith("heatmaps/")) {
    return `${API_BASE_URL}/heatmaps/${clean.slice("heatmaps/".length)}`;
  }

  if (clean.startsWith("video_frames/")) {
    return `${API_BASE_URL}/video_frames/${clean.slice("video_frames/".length)}`;
  }

  if (clean.startsWith("noise/")) {
    return `${API_BASE_URL}/noise/${clean.slice("noise/".length)}`;
  }

  if (clean.startsWith("jpeg_quality/")) {
    return `${API_BASE_URL}/jpeg_quality/${clean.slice("jpeg_quality/".length)}`;
  }

  return `${API_BASE_URL}/${clean}`;
};
