from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from backend.database.db import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String)
    saved_path = Column(String)
    thumbnail_path = Column(String)

    ml_probability = Column(Float)
    ml_label = Column(String)

    forensic_score = Column(Float)
    classification = Column(String)

    gradcam_heatmap = Column(String)
    ela_heatmap = Column(String)

    forensic_score_json = Column(JSON)
    ml_prediction = Column(JSON)
    metadata_anomalies = Column(JSON)
    metadata_json = Column(JSON)
    file_integrity = Column(JSON)
    ai_watermark = Column(JSON)
    exif_forensics = Column(JSON)
    c2pa = Column(JSON)
    jpeg_qtables = Column(JSON)
    noise_residual = Column(JSON)
    ela_analysis = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())


class VideoAnalysisRecord(Base):
    __tablename__ = "video_analysis_records"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String)
    saved_path = Column(String)
    thumbnail_path = Column(String)

    forensic_score = Column(Float)
    classification = Column(String)

    frame_count = Column(Integer)
    frames_json = Column(JSON)
    video_metadata = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
