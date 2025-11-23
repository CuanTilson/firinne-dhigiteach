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
    filename: str
    saved_path: str

    ml_probability: float
    ml_label: str

    final_score: float
    final_classification: str

    gradcam_heatmap: Optional[str]
    ela_heatmap: Optional[str]

    metadata_json: Optional[Any]
    exif_forensics: Optional[Any]
    c2pa: Optional[Any]
    jpeg_qtables: Optional[Any]
    noise_residual: Optional[Any]
    ela_analysis: Optional[Any]

    created_at: datetime

    class Config:
        orm_mode = True
