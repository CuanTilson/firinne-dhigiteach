from sqlalchemy import Column, Integer, String, Float, JSON, DateTime
from sqlalchemy.sql import func
from backend.database.db import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String)
    saved_path = Column(String)

    ml_probability = Column(Float)
    ml_label = Column(String)

    final_score = Column(Float)
    final_classification = Column(String)

    gradcam_heatmap = Column(String)
    ela_heatmap = Column(String)

    # --- Renamed to avoid SQLAlchemy conflict ---
    raw_metadata = Column(JSON)

    exif_forensics = Column(JSON)
    c2pa = Column(JSON)
    jpeg_qtables = Column(JSON)
    noise_residual = Column(JSON)
    ela_analysis = Column(JSON)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
