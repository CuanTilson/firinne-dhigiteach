import { API_ENDPOINTS, API_BASE_URL, DEFAULT_ADMIN_KEY } from "../constants";
import type {
  AnalysisRecordSummary,
  AnalysisResult,
  MediaType,
  PaginatedResponse,
  RecordFilters,
  VideoAnalysisDetail,
} from "../types";

/**
 * Uploads an image for forensic analysis
 */
export const detectImage = async (file: File): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('file', file);

  try {
    const response = await fetch(API_ENDPOINTS.DETECT, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Analysis failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Detect API Error:", error);
    throw error;
  }
};

/**
 * Uploads a video for forensic analysis
 */
export const detectVideo = async (file: File): Promise<VideoAnalysisDetail> => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(API_ENDPOINTS.DETECT_VIDEO, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Video analysis failed: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    console.error("Video Detect API Error:", error);
    throw error;
  }
};

/**
 * Fetches the history of records
 */
export const getRecords = async (
  page: number = 1,
  limit: number = 20,
  filters: RecordFilters = {}
): Promise<PaginatedResponse<AnalysisRecordSummary>> => {
  const params = new URLSearchParams();
  params.append('page', page.toString());
  params.append('limit', limit.toString());

  if (filters.filename) params.append('filename', filters.filename);
  if (filters.classification) params.append('classification', filters.classification);
  if (filters.date_from) params.append('date_from', filters.date_from);
  if (filters.date_to) params.append('date_to', filters.date_to);

  try {
    const response = await fetch(`${API_ENDPOINTS.RECORDS}?${params.toString()}`);
    if (!response.ok) throw new Error("Failed to fetch records");
    return await response.json();
  } catch (error) {
    console.error("Get Records Error:", error);
    throw error;
  }
};

/**
 * Fetches a single record detail
 */
export const getRecordById = async (id: number): Promise<AnalysisResult> => {
  try {
    const response = await fetch(`${API_ENDPOINTS.RECORDS}/${id}`);
    if (!response.ok) throw new Error("Failed to fetch record details");
    return await response.json();
  } catch (error) {
    console.error("Get Record Detail Error:", error);
    throw error;
  }
};

/**
 * Fetches a single video record detail
 */
export const getVideoById = async (id: number): Promise<VideoAnalysisDetail> => {
  try {
    const response = await fetch(`${API_ENDPOINTS.VIDEO_RECORDS}/${id}`);
    if (!response.ok) throw new Error("Failed to fetch video details");
    return await response.json();
  } catch (error) {
    console.error("Get Video Detail Error:", error);
    throw error;
  }
};

/**
 * Deletes a record (Admin)
 */
export const deleteRecord = async (
  id: number,
  adminKey?: string,
  mediaType: MediaType = "image"
): Promise<void> => {
  const resolvedKey = adminKey ?? DEFAULT_ADMIN_KEY;
  try {
    const endpoint =
      mediaType === "video" ? API_ENDPOINTS.VIDEO_RECORDS : API_ENDPOINTS.RECORDS;
    const response = await fetch(`${endpoint}/${id}`, {
      method: 'DELETE',
      headers: resolvedKey ? { 'admin-key': resolvedKey } : undefined,
    });
    if (!response.ok) throw new Error("Failed to delete record");
  } catch (error) {
    console.error("Delete Error:", error);
    throw error;
  }
};


/**
 * This returns true if the backend is reachable
 */
export const checkBackend = async (): Promise<boolean> => {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
};
