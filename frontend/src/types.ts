// Data Models matching the Backend API Response

export type ClassificationType =
  | "likely_ai_generated"
  | "likely_real"
  | "uncertain"
  | "ai_generated_c2pa_flagged";

export type MediaType = "image" | "video";

export interface AnalysisResult {
  id?: number;

  filename?: string;
  saved_path: string | null;

  file_integrity: {
    hashes: { sha256: string; md5: string };
    hashes_before?: { sha256: string; md5: string };
    hashes_after?: { sha256: string; md5: string };
    hashes_match?: boolean;
    jpeg_structure: {
      valid_jpeg: boolean;
      missing_soi: boolean;
      missing_eoi: boolean;
      double_compressed: boolean;
      app1_segments: number;
      warnings: string[];
    };
  };

  ml_prediction: {
    probability: number;
    label: string;
  };

  metadata_anomalies: {
    anomaly_score: number;
    findings: string[];
    camera_consistency?: {
      score: number;
      warnings: string[];
      make?: string | null;
      model?: string | null;
    };
  };

  exif_forensics: {
    warnings: string[];
    score: number;
  };

  c2pa: {
    has_c2pa: boolean;
    signature_valid: boolean | null;
    ai_assertions_found: string[];
    tools_detected: string[];
    edit_actions: string[];
    digital_source_types: string[];
    software_agents: string[];
    overall_c2pa_score: number;
    errors: string[];
    claim_generator?: unknown;
    signer?: string | null;
    cert_issuer?: string | null;
    signing_time?: string | null;
    ingredients?: unknown[];
  };

  jpeg_qtables: {
    found: boolean;
    qtables: Record<string, number[]>;
    anomaly_score: number;
    quality_estimate?: number | null;
    double_compression?: {
      mode_quality: number;
      inconsistency_score: number;
      jpeg_quality_heatmap_path?: string | null;
    };
    inconsistency_score?: number | null;
    combined_anomaly_score?: number | null;
  };

  noise_residual: {
    variance: number;
    spectral_flatness: number;
    anomaly_score: number;
    local_variance_min?: number | null;
    local_variance_max?: number | null;
    local_variance_mean?: number | null;
    noise_heatmap_path?: string | null;
    inconsistency_score?: number | null;
    combined_anomaly_score?: number | null;
  };

  ai_watermark: {
    stable_diffusion_detected: boolean;
    confidence: number;
    raw_string: string;
    error: string | null;
  };

  ela_analysis: {
    mean_error: number;
    max_error: number;
    anomaly_score: number;
  };

  forensic_score: number;
  classification: ClassificationType;

  forensic_score_json: {
    final_score: number;
    classification: ClassificationType;
    override: boolean;
  };

  gradcam_heatmap: string | null;
  ela_heatmap: string | null;

  raw_metadata: Record<string, unknown>;
  created_at: string;
}



export interface AnalysisRecordSummary {
  id: number;
  filename: string;
  forensic_score: number;
  classification: ClassificationType;
  thumbnail_url: string;
  created_at: string;
  media_type: MediaType;
}


export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  page: number;
  limit: number;
  total_pages: number;
}

export interface RecordFilters {
  filename?: string;
  classification?: ClassificationType | "";
  date_from?: string;
  date_to?: string;
}

export interface VideoFrameResult extends AnalysisResult {
  frame_index: number;
  timestamp_sec: number;
}

export interface VideoAnalysisDetail {
  id: number;
  filename?: string;
  saved_path?: string;
  thumbnail_path?: string;
  forensic_score: number;
  classification: ClassificationType;
  frame_count: number;
  frames: VideoFrameResult[];
  video_metadata?: Record<string, unknown>;
  created_at: string;
}

export interface VideoJobStatus {
  id: string;
  status: "queued" | "running" | "completed" | "failed";
  created_at?: string;
  started_at?: string;
  finished_at?: string;
  filename?: string;
  result?: VideoAnalysisDetail;
  error?: string;
}
