// Data Models matching the Backend API Response

export type ClassificationType =
  | "likely_ai_generated"
  | "likely_real"
  | "uncertain"
  | "ai_generated_c2pa_flagged";

export interface AnalysisResult {
  id?: number;

  filename?: string;
  saved_path: string | null;

  file_integrity: {
    hashes: { sha256: string; md5: string };
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
  };

  jpeg_qtables: {
    found: boolean;
    qtables: Record<string, number[]>;
    anomaly_score: number;
  };

  noise_residual: {
    variance: number;
    spectral_flatness: number;
    anomaly_score: number;
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