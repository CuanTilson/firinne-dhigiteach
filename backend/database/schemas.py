from pydantic import BaseModel
from typing import Optional, Any
from datetime import datetime


class AnalysisSummary(BaseModel):
    id: int
    filename: str
    final_score: float
    final_classification: str
    created_at: datetime

    class Config:
        orm_mode = True


class AnalysisDetail(BaseModel):
    id: int
    filename: Optional[str]
    saved_path: Optional[str]

    ml_probability: Optional[float]
    ml_label: Optional[str]

    final_score: Optional[float]
    final_classification: Optional[str]

    gradcam_heatmap: Optional[str]
    ela_heatmap: Optional[str]

    forensic_score: Optional[Any]
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

    class Config:
        orm_mode = True


class PaginatedAnalysisSummary(BaseModel):
    data: list[AnalysisSummary]
    total: int
    page: int
    limit: int
    total_pages: int
