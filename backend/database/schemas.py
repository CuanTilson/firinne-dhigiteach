from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class AnalysisSummary(BaseModel):
    id: int
    filename: str
    forensic_score: float
    classification: str
    created_at: datetime
    thumbnail_url: str
    media_type: str

    model_config = {"from_attributes": True}


class AnalysisDetail(BaseModel):
    id: int
    filename: Optional[str]
    saved_path: Optional[str]

    ml_probability: Optional[float]
    ml_label: Optional[str]

    forensic_score: Optional[float]  # float score
    classification: Optional[str]  # text label

    gradcam_heatmap: Optional[str]
    ela_heatmap: Optional[str]

    forensic_score_json: Optional[Any]
    ml_prediction: Optional[Any]
    metadata_anomalies: Optional[Any]
    file_integrity: Optional[Any]
    ai_watermark: Optional[Any]

    metadata_json: Optional[Any]
    exif_forensics: Optional[Any]
    c2pa: Optional[Any]
    jpeg_qtables: Optional[Any]
    noise_residual: Optional[Any]
    ela_analysis: Optional[Any]

    raw_metadata: Optional[dict]

    created_at: datetime

    model_config = {"from_attributes": True}


class PaginatedAnalysisSummary(BaseModel):
    data: list[AnalysisSummary]
    total: int
    page: int
    limit: int
    total_pages: int


class VideoAnalysisDetail(BaseModel):
    id: int
    filename: Optional[str]
    saved_path: Optional[str]
    thumbnail_path: Optional[str]
    forensic_score: float
    classification: str
    frame_count: int
    frames: list[Any]
    video_metadata: Optional[Any]
    created_at: datetime

    model_config = {"from_attributes": True}
