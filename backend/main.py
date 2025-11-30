from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from PIL import Image

from sqlalchemy.orm import Session
from pathlib import Path
import shutil
import uuid

# ---------------------------
# Local Imports (organised)
# ---------------------------

# Storage + DB
from backend.database.db import Base, engine, SessionLocal
from backend.database.models import AnalysisRecord
from backend.database.schemas import (
    AnalysisSummary,
    AnalysisDetail,
    PaginatedAnalysisSummary,
)

# Upload handling
from backend.analysis.upload import save_uploaded_file, UPLOAD_DIR

# Metadata + forensic tools
from backend.analysis.metadata import (
    extract_image_metadata,
    extract_video_metadata,
    analyse_image_metadata,
    exif_forensics,
    analyse_file_integrity,
)
from backend.analysis.forensics import (
    analyse_noise,
    analyse_qtables,
    perform_ela,
    detect_sd_watermark,
)
from backend.analysis.c2pa_analyser import analyse_c2pa
from backend.analysis.forensic_fusion import fuse_forensic_scores

# ML model + explainability
from backend.models.cnndetect_native import CNNDetectionModel
from backend.explainability.gradcam import GradCAM


# ---------------------------
# Setup
# ---------------------------

WEIGHTS = Path("vendor/CNNDetection/weights/blur_jpg_prob0.5.pth")
Path("backend/storage/ela").mkdir(parents=True, exist_ok=True)
Path("backend/storage/heatmaps").mkdir(parents=True, exist_ok=True)
THUMB_DIR = Path("backend/storage/thumbnails")
THUMB_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_KEY = "secret-admin-key"

app = FastAPI(title="Fírinne Dhigiteach API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static folders
app.mount(
    "/uploaded", StaticFiles(directory="backend/storage/uploaded"), name="uploaded"
)
app.mount("/ela", StaticFiles(directory="backend/storage/ela"), name="ela")
app.mount(
    "/heatmaps", StaticFiles(directory="backend/storage/heatmaps"), name="heatmaps"
)
app.mount(
    "/thumbnails",
    StaticFiles(directory="backend/storage/thumbnails"),
    name="thumbnails",
)


# Create database tables
Base.metadata.create_all(bind=engine)

# Load ML model once
cnndetector = CNNDetectionModel(weights_path=WEIGHTS)


# ---------------------------
# Utilities
# ---------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ---------------------------
# Basic Routes
# ---------------------------


@app.get("/")
def root():
    return {"message": "Fírinne Dhigiteach API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------
# Upload + Metadata Endpoints
# ---------------------------


@app.post("/media/upload")
async def upload_media(file: UploadFile = File(...)):
    saved_path = save_uploaded_file(file)
    return {"status": "success", "filename": file.filename, "saved_to": str(saved_path)}


@app.post("/media/metadata")
async def analyse_media(file: UploadFile = File(...)):
    saved_path = save_uploaded_file(file)
    ext = Path(file.filename).suffix.lower()

    if ext in {".jpg", ".jpeg", ".png"}:
        metadata = extract_image_metadata(saved_path)
    else:
        metadata = extract_video_metadata(saved_path)

    return {
        "filename": file.filename,
        "path": str(saved_path),
        "metadata": metadata,
    }


# ---------------------------
# Analysis Pipeline
# ---------------------------


@app.post("/analysis/image")
async def analyse_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    # -----------------------
    # Save file
    # -----------------------
    suffix = Path(file.filename).suffix or ".png"
    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with filepath.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # Generate 128×128 thumbnail
    thumb_path = THUMB_DIR / f"{uuid.uuid4().hex}_thumb.jpg"

    with Image.open(filepath) as img:
        img.thumbnail((128, 128))
        img.save(thumb_path, "JPEG")

    # -----------------------
    # Run all analysis stages
    # -----------------------

    file_integrity = analyse_file_integrity(filepath)

    ml = cnndetector.predict(filepath)
    ml_prob = ml["probability"]

    metadata = extract_image_metadata(filepath)
    exif_result = exif_forensics(metadata)

    anomaly = analyse_image_metadata(metadata)
    qtinfo = analyse_qtables(filepath)
    noise_info = analyse_noise(filepath)

    watermark_info = detect_sd_watermark(filepath)
    sd_watermark_score = 1.0 if watermark_info["watermark_detected"] else 0.0

    c2pa_info = analyse_c2pa(filepath)
    c2pa_ai_flag = c2pa_info["has_c2pa"] and (
        c2pa_info["overall_c2pa_score"] >= 0.95
        or len(c2pa_info["ai_assertions_found"]) > 0
        or "iptc:compositeWithTrainedAlgorithmicMedia"
        in c2pa_info.get("digital_source_types", [])
        or "photo assist" in c2pa_info.get("software_agents", [])
    )

    # ELA
    ela_path = Path("backend/storage/ela") / f"{uuid.uuid4().hex}_ela.png"
    ela_info = perform_ela(filepath, quality=90, scale_factor=20, save_path=ela_path)

    # Fuse all scores
    fused = fuse_forensic_scores(
        ml_prob,
        anomaly["anomaly_score"],
        c2pa_info["overall_c2pa_score"],
        c2pa_ai_flag,
        ela_info["ela_anomaly_score"],
        noise_info["noise_anomaly_score"],
        qtinfo["qtables_anomaly_score"],
        sd_watermark_score,
    )

    # GradCAM
    heatmap_path = Path("backend/storage/heatmaps") / f"{uuid.uuid4().hex}_gradcam.png"
    cam = GradCAM(cnndetector.get_model(), cnndetector.get_target_layer())
    cam.generate(ml["tensor"], filepath, heatmap_path)

    # -----------------------
    # Save to DB
    # -----------------------
    session = SessionLocal()
    record = AnalysisRecord(
        filename=file.filename,
        saved_path=str(filepath),
        thumbnail_path=str(thumb_path),
        ml_probability=ml_prob,
        ml_label=ml["label"],
        forensic_score=fused["final_score"],  # float score
        classification=fused["classification"],  # text label
        gradcam_heatmap=str(heatmap_path),
        ela_heatmap=ela_info["ela_image_path"],
        forensic_score_json=fused,  # full JSON payload
        ml_prediction={"probability": ml_prob, "label": ml["label"]},
        metadata_anomalies=anomaly,
        file_integrity=file_integrity,
        ai_watermark=watermark_info,
        metadata_json=metadata,
        exif_forensics=exif_result,
        c2pa=c2pa_info,
        jpeg_qtables=qtinfo,
        noise_residual=noise_info,
        ela_analysis=ela_info,
    )

    session.add(record)
    session.commit()
    created_at = record.created_at
    session.close()

    # -----------------------
    # Return API response
    # -----------------------
    return {
        "detector": "CNNDetection + GradCAM + Forensic Fusion + C2PA",
        "input_file": file.filename,
        "saved_path": str(filepath),
        "created_at": created_at.isoformat(),
        "file_integrity": file_integrity,
        "ml_prediction": {"probability": ml_prob, "label": ml["label"]},
        "metadata_anomalies": anomaly,
        "exif_forensics": exif_result,
        "c2pa": {
            "has_c2pa": c2pa_info["has_c2pa"],
            "signature_valid": c2pa_info["signature_valid"],
            "ai_assertions_found": c2pa_info["ai_assertions_found"],
            "tools_detected": c2pa_info["tools_detected"],
            "edit_actions": c2pa_info["edit_actions"],
            "digital_source_types": c2pa_info["digital_source_types"],
            "software_agents": c2pa_info["software_agents"],
            "overall_c2pa_score": c2pa_info["overall_c2pa_score"],
            "errors": c2pa_info["errors"],
        },
        "jpeg_qtables": {
            "found": qtinfo["qtables_found"],
            "qtables": qtinfo["qtables"],
            "anomaly_score": qtinfo["qtables_anomaly_score"],
        },
        "noise_residual": {
            "variance": noise_info["residual_variance"],
            "spectral_flatness": noise_info["spectral_flatness"],
            "anomaly_score": noise_info["noise_anomaly_score"],
        },
        "ai_watermark": {
            "stable_diffusion_detected": watermark_info["watermark_detected"],
            "confidence": watermark_info["confidence"],
            "raw_string": watermark_info.get("raw_watermark_string"),
            "error": watermark_info["error"],
        },
        "ela_analysis": {
            "mean_error": ela_info["mean_error"],
            "max_error": ela_info["max_error"],
            "anomaly_score": ela_info["ela_anomaly_score"],
        },
        "forensic_score": fused["final_score"],
        "classification": fused["classification"],
        "forensic_score_json": fused,
        "ela_heatmap": ela_info["ela_image_path"],
        "gradcam_heatmap": str(heatmap_path),
        "raw_metadata": metadata,
    }


# ---------------------------
# Query + Delete Endpoints
# ---------------------------


@app.get("/analysis", response_model=PaginatedAnalysisSummary)
def list_analysis(
    page: int = 1,
    limit: int = 20,
    classification: str | None = None,
    filename: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    db: Session = Depends(get_db),
):
    offset = (page - 1) * limit
    q = db.query(AnalysisRecord)

    if classification:
        q = q.filter(AnalysisRecord.classification == classification)

    if filename:
        q = q.filter(AnalysisRecord.filename.ilike(f"%{filename}%"))

    if date_from:
        q = q.filter(AnalysisRecord.created_at >= date_from)

    if date_to:
        q = q.filter(AnalysisRecord.created_at <= date_to)

    records = (
        q.order_by(AnalysisRecord.created_at.desc()).offset(offset).limit(limit).all()
    )

    total = q.count()

    return {
        "data": [
            {
                "id": r.id,
                "filename": r.filename,
                "forensic_score": r.forensic_score,
                "classification": r.classification,
                "created_at": r.created_at,
                "thumbnail_url": f"/thumbnails/{Path(r.thumbnail_path).name}",
            }
            for r in records
        ],
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
    }


@app.get("/analysis/{record_id}", response_model=AnalysisDetail)
def get_analysis(record_id: int, db: Session = Depends(get_db)):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()

    if not record:
        raise HTTPException(404, "Record not found")

    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "ml_probability": record.ml_probability,
        "ml_label": record.ml_label,
        "forensic_score": record.forensic_score,  # float
        "classification": record.classification,  # string
        "forensic_score_json": record.forensic_score_json,  # full JSON
        "gradcam_heatmap": record.gradcam_heatmap,
        "ela_heatmap": record.ela_heatmap,
        "ml_prediction": record.ml_prediction,
        "metadata_anomalies": record.metadata_anomalies,
        "file_integrity": record.file_integrity,
        "ai_watermark": record.ai_watermark,
        "metadata_json": record.metadata_json,
        "exif_forensics": record.exif_forensics,
        "c2pa": record.c2pa,
        "jpeg_qtables": record.jpeg_qtables,
        "noise_residual": record.noise_residual,
        "ela_analysis": record.ela_analysis,
        "raw_metadata": record.metadata_json,
        "created_at": record.created_at,
    }


@app.delete("/analysis/{record_id}")
def delete_analysis(
    record_id: int,
    admin_key: str = Header(None),
    db: Session = Depends(get_db),
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")

    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()

    if not record:
        raise HTTPException(404, "Record not found")

    db.delete(record)
    db.commit()

    return {"status": "deleted", "id": record_id}
