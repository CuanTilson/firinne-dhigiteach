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

from PIL import Image, ImageOps

from sqlalchemy.orm import Session
from sqlalchemy import select, literal, union_all, func
from pathlib import Path
import uuid
import os
from datetime import datetime
import asyncio
from dotenv import load_dotenv

# ---------------------------
# Local Imports (organised)
# ---------------------------

# Storage + DB
from backend.database.db import Base, engine, SessionLocal
from backend.database.models import AnalysisRecord, VideoAnalysisRecord
from backend.database.schemas import (
    AnalysisDetail,
    PaginatedAnalysisSummary,
    VideoAnalysisDetail,
)

# Upload handling
from backend.analysis.upload import (
    save_uploaded_file,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

# Metadata + forensic tools
from backend.analysis.metadata import (
    extract_image_metadata,
    extract_video_metadata,
    analyse_image_metadata,
    check_camera_model_consistency,
    exif_forensics,
    analyse_file_integrity,
)
from backend.analysis.forensics import (
    analyse_noise,
    analyse_qtables,
    perform_ela,
    detect_sd_watermark,
    generate_noise_heatmap,
    estimate_jpeg_quality,
    jpeg_double_compression_heatmap,
)
from backend.analysis.c2pa_analyser import analyse_c2pa
from backend.analysis.forensic_fusion import fuse_forensic_scores
from backend.analysis.video import sample_video_frames, get_video_duration_seconds

# ML model + explainability
from backend.models.cnndetect_native import CNNDetectionModel
from backend.explainability.gradcam import GradCAM


# ---------------------------
# Setup
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
STORAGE_DIR = BACKEND_DIR / "storage"

# Load environment variables from backend/.env (if present)
load_dotenv(dotenv_path=BACKEND_DIR / ".env")
# ---------------------------

WEIGHTS = PROJECT_ROOT / "vendor" / "CNNDetection" / "weights" / "blur_jpg_prob0.5.pth"
ELA_DIR = STORAGE_DIR / "ela"
HEATMAPS_DIR = STORAGE_DIR / "heatmaps"
THUMB_DIR = STORAGE_DIR / "thumbnails"
VIDEO_FRAMES_DIR = STORAGE_DIR / "video_frames"
NOISE_DIR = STORAGE_DIR / "noise"
JPEG_QUALITY_DIR = STORAGE_DIR / "jpeg_quality"

ELA_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
NOISE_DIR.mkdir(parents=True, exist_ok=True)
JPEG_QUALITY_DIR.mkdir(parents=True, exist_ok=True)

ADMIN_KEY = os.getenv("FD_ADMIN_KEY")
ALLOW_INSECURE_ADMIN_KEY = os.getenv("FD_ALLOW_INSECURE_ADMIN_KEY") == "1"
if not ADMIN_KEY:
    if ALLOW_INSECURE_ADMIN_KEY:
        ADMIN_KEY = "secret-admin-key"
        print(
            "WARNING: FD_ADMIN_KEY is not set. Using insecure default "
            "because FD_ALLOW_INSECURE_ADMIN_KEY=1."
        )
    else:
        raise RuntimeError("FD_ADMIN_KEY must be set to enable admin actions.")

ALLOWED_ORIGINS = os.getenv("FD_CORS_ORIGINS", "*")
ALLOWED_ORIGINS_LIST = ["*"] if ALLOWED_ORIGINS == "*" else [
    origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()
]

MAX_IMAGE_MB = float(os.getenv("FD_MAX_IMAGE_MB", "20"))
MAX_VIDEO_MB = float(os.getenv("FD_MAX_VIDEO_MB", "200"))
MAX_UPLOAD_MB = float(os.getenv("FD_MAX_UPLOAD_MB", "200"))
app = FastAPI(title="Fírinne Dhigiteach API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS_LIST,
    allow_credentials=ALLOWED_ORIGINS != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static folders
app.mount(
    "/uploaded", StaticFiles(directory=STORAGE_DIR / "uploaded"), name="uploaded"
)
app.mount("/ela", StaticFiles(directory=ELA_DIR), name="ela")
app.mount(
    "/heatmaps", StaticFiles(directory=HEATMAPS_DIR), name="heatmaps"
)
app.mount(
    "/thumbnails",
    StaticFiles(directory=THUMB_DIR),
    name="thumbnails",
)
app.mount(
    "/video_frames",
    StaticFiles(directory=VIDEO_FRAMES_DIR),
    name="video_frames",
)
app.mount("/noise", StaticFiles(directory=NOISE_DIR), name="noise")
app.mount(
    "/jpeg_quality",
    StaticFiles(directory=JPEG_QUALITY_DIR),
    name="jpeg_quality",
)


# Create database tables
Base.metadata.create_all(bind=engine)

# Load ML model once
if not WEIGHTS.is_file():
    raise RuntimeError(
        f"Missing CNNDetection weights at {WEIGHTS}. See README for download steps."
    )
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


def _mb_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024)


def _max_bytes_for_ext(ext: str) -> int:
    ext = ext.lower()
    if ext in IMAGE_EXTENSIONS:
        return _mb_to_bytes(MAX_IMAGE_MB)
    if ext in VIDEO_EXTENSIONS:
        return _mb_to_bytes(MAX_VIDEO_MB)
    return _mb_to_bytes(MAX_UPLOAD_MB)


def _ensure_image_valid(path: Path) -> None:
    try:
        with Image.open(path) as img:
            img.verify()
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(400, "Invalid or corrupted image file.") from exc


def _coerce_storage_path(value: str) -> Path | None:
    try:
        raw_path = Path(value)
    except (TypeError, ValueError):
        return None

    if raw_path.is_absolute():
        try:
            raw_path.relative_to(STORAGE_DIR)
            return raw_path
        except ValueError:
            return None

    clean = value.replace("\\", "/")
    marker = "backend/storage/"
    if marker in clean:
        rel = clean.split(marker, 1)[1]
        return STORAGE_DIR / rel

    for prefix in (
        "uploaded/",
        "ela/",
        "heatmaps/",
        "thumbnails/",
        "video_frames/",
        "noise/",
        "jpeg_quality/",
    ):
        if clean.startswith(prefix):
            return STORAGE_DIR / clean

    return None


def _collect_storage_paths(value, paths: set[Path]) -> None:
    if value is None:
        return
    if isinstance(value, str):
        path = _coerce_storage_path(value)
        if path:
            paths.add(path)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_storage_paths(item, paths)
        return
    if isinstance(value, (list, tuple, set)):
        for item in value:
            _collect_storage_paths(item, paths)


def _cleanup_storage_paths(paths: set[Path]) -> None:
    frame_dirs: set[Path] = set()
    for path in paths:
        if path.is_file():
            if VIDEO_FRAMES_DIR in path.parents:
                frame_dirs.add(path.parent)
            path.unlink(missing_ok=True)

    for folder in frame_dirs:
        try:
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()
        except OSError:
            pass


def run_full_analysis(filepath: Path) -> dict:
    file_integrity = analyse_file_integrity(filepath)

    ml = cnndetector.predict(filepath)
    ml_prob = ml["probability"]

    metadata = extract_image_metadata(filepath)
    exif_result = exif_forensics(metadata)

    anomaly = analyse_image_metadata(metadata)
    camera_consistency = check_camera_model_consistency(metadata)
    anomaly["camera_consistency"] = camera_consistency
    if camera_consistency["warnings"]:
        anomaly["findings"].extend(camera_consistency["warnings"])
    anomaly["anomaly_score"] = min(
        1.0, anomaly["anomaly_score"] + camera_consistency["score"]
    )
    qtinfo = analyse_qtables(filepath)
    jpeg_quality = estimate_jpeg_quality(filepath)
    noise_info = analyse_noise(filepath)
    noise_heatmap_path = NOISE_DIR / f"{uuid.uuid4().hex}_noise.png"
    noise_heatmap = generate_noise_heatmap(filepath, save_path=noise_heatmap_path)

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

    ela_path = ELA_DIR / f"{uuid.uuid4().hex}_ela.png"
    ela_info = perform_ela(filepath, quality=90, scale_factor=20, save_path=ela_path)
    jpeg_quality_path = JPEG_QUALITY_DIR / f"{uuid.uuid4().hex}_jpegq.png"
    jpeg_quality_local = jpeg_double_compression_heatmap(
        filepath, save_path=jpeg_quality_path
    )

    jpeg_inconsistency = float(jpeg_quality_local.get("inconsistency_score") or 0.0)
    qtinfo["quality_estimate"] = jpeg_quality["quality_estimate"]
    qtinfo["double_compression"] = jpeg_quality_local
    qtinfo["inconsistency_score"] = jpeg_inconsistency
    qtinfo["combined_anomaly_score"] = max(
        qtinfo["qtables_anomaly_score"], jpeg_inconsistency
    )

    local_min = float(noise_heatmap["local_variance_min"])
    local_max = float(noise_heatmap["local_variance_max"])
    noise_spread = (local_max - local_min) / (local_max + 1e-8)
    noise_inconsistency = float(min(1.0, max(0.0, noise_spread)))

    noise_info.update(
        {
            "local_variance_min": noise_heatmap["local_variance_min"],
            "local_variance_max": noise_heatmap["local_variance_max"],
            "local_variance_mean": noise_heatmap["local_variance_mean"],
            "noise_heatmap_path": noise_heatmap["noise_heatmap_path"],
            "inconsistency_score": noise_inconsistency,
            "combined_anomaly_score": max(
                noise_info["noise_anomaly_score"], noise_inconsistency
            ),
        }
    )

    fused = fuse_forensic_scores(
        ml_prob,
        anomaly["anomaly_score"],
        c2pa_info["overall_c2pa_score"],
        c2pa_ai_flag,
        ela_info["ela_anomaly_score"],
        noise_info["noise_anomaly_score"],
        qtinfo["combined_anomaly_score"],
        sd_watermark_score,
    )

    heatmap_path = HEATMAPS_DIR / f"{uuid.uuid4().hex}_gradcam.png"
    cam = GradCAM(cnndetector.get_model(), cnndetector.get_target_layer())
    cam.generate(ml["tensor"], filepath, heatmap_path)

    return {
        "file_integrity": file_integrity,
        "ml": ml,
        "ml_prob": ml_prob,
        "metadata": metadata,
        "exif_result": exif_result,
        "anomaly": anomaly,
        "qtinfo": qtinfo,
        "noise_info": noise_info,
        "watermark_info": watermark_info,
        "sd_watermark_score": sd_watermark_score,
        "c2pa_info": c2pa_info,
        "c2pa_ai_flag": c2pa_ai_flag,
        "ela_info": ela_info,
        "ela_path": ela_path,
        "fused": fused,
        "heatmap_path": heatmap_path,
        "camera_consistency": camera_consistency,
    }


def run_video_analysis(video_path: Path, max_frames: int = 16) -> dict:
    frames_dir = VIDEO_FRAMES_DIR / uuid.uuid4().hex
    frames = sample_video_frames(video_path, max_frames=max_frames, output_dir=frames_dir)
    if not frames:
        raise RuntimeError("No frames could be sampled from the video.")

    frame_results = []
    scores = []
    c2pa_flagged = False

    for frame in frames:
        analysis = run_full_analysis(frame["frame_path"])
        frame_result = {
            "frame_index": frame["frame_index"],
            "timestamp_sec": frame["timestamp_sec"],
            "saved_path": str(frame["frame_path"]),
            "created_at": datetime.utcnow().isoformat(),
            "file_integrity": analysis["file_integrity"],
            "ml_prediction": {
                "probability": analysis["ml_prob"],
                "label": analysis["ml"]["label"],
            },
            "metadata_anomalies": analysis["anomaly"],
            "exif_forensics": analysis["exif_result"],
            "c2pa": analysis["c2pa_info"],
            "jpeg_qtables": analysis["qtinfo"],
            "noise_residual": analysis["noise_info"],
            "ai_watermark": {
                "stable_diffusion_detected": analysis["watermark_info"][
                    "watermark_detected"
                ],
                "confidence": analysis["watermark_info"]["confidence"],
                "raw_string": analysis["watermark_info"].get("raw_watermark_string"),
                "error": analysis["watermark_info"]["error"],
            },
            "ela_analysis": {
                "mean_error": analysis["ela_info"]["mean_error"],
                "max_error": analysis["ela_info"]["max_error"],
                "anomaly_score": analysis["ela_info"]["ela_anomaly_score"],
            },
            "forensic_score": analysis["fused"]["final_score"],
            "classification": analysis["fused"]["classification"],
            "forensic_score_json": analysis["fused"],
            "gradcam_heatmap": str(analysis["heatmap_path"]),
            "ela_heatmap": analysis["ela_info"]["ela_image_path"],
            "raw_metadata": analysis["metadata"],
        }
        frame_results.append(frame_result)
        scores.append(analysis["fused"]["final_score"])
        if analysis["fused"]["classification"] == "ai_generated_c2pa_flagged":
            c2pa_flagged = True

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    if c2pa_flagged:
        classification = "ai_generated_c2pa_flagged"
    elif avg_score > 0.7:
        classification = "likely_ai_generated"
    elif avg_score < 0.3:
        classification = "likely_real"
    else:
        classification = "uncertain"

    return {
        "frames": frame_results,
        "frame_count": len(frame_results),
        "forensic_score": avg_score,
        "classification": classification,
    }


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
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(400, "Missing filename.")
    ext = Path(original_name).suffix.lower()
    saved_path = save_uploaded_file(file, max_bytes=_max_bytes_for_ext(ext))
    return {
        "status": "success",
        "filename": original_name,
        "saved_to": str(saved_path),
    }


@app.post("/media/metadata")
async def analyse_media(file: UploadFile = File(...)):
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(400, "Missing filename.")

    ext = Path(original_name).suffix.lower()
    saved_path = save_uploaded_file(file, max_bytes=_max_bytes_for_ext(ext))

    if ext in {".jpg", ".jpeg", ".png"}:
        _ensure_image_valid(saved_path)
        metadata = extract_image_metadata(saved_path)
    else:
        metadata = extract_video_metadata(saved_path)

    return {
        "filename": original_name,
        "path": str(saved_path),
        "metadata": metadata,
    }


# ---------------------------
# Analysis Pipeline
# ---------------------------


@app.post("/analysis/image")
async def analyse_image(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    # -----------------------
    # Save file
    # -----------------------
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(400, "Missing filename.")

    ext = Path(original_name).suffix.lower()
    filepath = save_uploaded_file(file, max_bytes=_max_bytes_for_ext(ext))
    _ensure_image_valid(filepath)

    # Generate 128x128 thumbnail
    thumb_path = THUMB_DIR / f"{uuid.uuid4().hex}_thumb.jpg"

    with Image.open(filepath) as img:
        img = ImageOps.exif_transpose(img)
        img.thumbnail((128, 128))
        img.save(thumb_path, "JPEG")

    # -----------------------
    # Run all analysis stages
    # -----------------------

    analysis = await asyncio.to_thread(run_full_analysis, filepath)
    file_integrity = analysis["file_integrity"]
    ml = analysis["ml"]
    ml_prob = analysis["ml_prob"]
    metadata = analysis["metadata"]
    exif_result = analysis["exif_result"]
    anomaly = analysis["anomaly"]
    qtinfo = analysis["qtinfo"]
    noise_info = analysis["noise_info"]
    watermark_info = analysis["watermark_info"]
    c2pa_info = analysis["c2pa_info"]
    c2pa_ai_flag = analysis["c2pa_ai_flag"]
    ela_info = analysis["ela_info"]
    heatmap_path = analysis["heatmap_path"]
    fused = analysis["fused"]
    camera_consistency = analysis["camera_consistency"]

    # -----------------------
    # Save to DB
    # -----------------------
    record = AnalysisRecord(
        filename=original_name,
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

    db.add(record)
    db.commit()
    db.refresh(record)
    created_at = record.created_at

    # -----------------------
    # Return API response
    # -----------------------
    return {
        "detector": "CNNDetection + GradCAM + Forensic Fusion + C2PA",
        "input_file": original_name,
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
            "claim_generator": c2pa_info.get("claim_generator"),
            "signer": c2pa_info.get("signer"),
            "cert_issuer": c2pa_info.get("cert_issuer"),
            "signing_time": c2pa_info.get("signing_time"),
            "ingredients": c2pa_info.get("ingredients"),
        },
        "jpeg_qtables": {
            "found": qtinfo["qtables_found"],
            "qtables": qtinfo["qtables"],
            "anomaly_score": qtinfo["qtables_anomaly_score"],
            "quality_estimate": qtinfo.get("quality_estimate"),
            "double_compression": qtinfo.get("double_compression"),
            "inconsistency_score": qtinfo.get("inconsistency_score"),
            "combined_anomaly_score": qtinfo.get("combined_anomaly_score"),
        },
        "noise_residual": {
            "variance": noise_info["residual_variance"],
            "spectral_flatness": noise_info["spectral_flatness"],
            "anomaly_score": noise_info["noise_anomaly_score"],
            "local_variance_min": noise_info.get("local_variance_min"),
            "local_variance_max": noise_info.get("local_variance_max"),
            "local_variance_mean": noise_info.get("local_variance_mean"),
            "noise_heatmap_path": noise_info.get("noise_heatmap_path"),
            "inconsistency_score": noise_info.get("inconsistency_score"),
            "combined_anomaly_score": noise_info.get("combined_anomaly_score"),
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
        "camera_consistency": camera_consistency,
        "raw_metadata": metadata,
    }


@app.post("/analysis/video", response_model=VideoAnalysisDetail)
async def analyse_video(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if not (file.content_type or "").startswith("video/"):
        raise HTTPException(400, "Please upload a video file.")

    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(400, "Missing filename.")

    ext = Path(original_name).suffix.lower()
    saved_path = save_uploaded_file(file, max_bytes=_max_bytes_for_ext(ext))

    duration_seconds = get_video_duration_seconds(saved_path)
    if duration_seconds <= 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video appears to be invalid or unreadable.")
    if duration_seconds > 180:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video exceeds 3 minute length limit.")
    video_metadata = extract_video_metadata(saved_path)

    analysis = await asyncio.to_thread(run_video_analysis, saved_path)

    thumbnail_path = THUMB_DIR / f"{uuid.uuid4().hex}_video_thumb.jpg"
    first_frame_path = Path(analysis["frames"][0]["saved_path"])
    with Image.open(first_frame_path) as img:
        img.thumbnail((128, 128))
        img.save(thumbnail_path, "JPEG")

    record = VideoAnalysisRecord(
        filename=original_name,
        saved_path=str(saved_path),
        thumbnail_path=str(thumbnail_path),
        forensic_score=analysis["forensic_score"],
        classification=analysis["classification"],
        frame_count=analysis["frame_count"],
        frames_json=analysis["frames"],
        video_metadata=video_metadata,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "thumbnail_path": record.thumbnail_path,
        "forensic_score": record.forensic_score,
        "classification": record.classification,
        "frame_count": record.frame_count,
        "frames": record.frames_json,
        "video_metadata": record.video_metadata,
        "created_at": record.created_at,
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
    def parse_iso_date(value: str, field: str) -> datetime:
        try:
            return datetime.fromisoformat(value)
        except ValueError as exc:
            raise HTTPException(
                400, f"Invalid {field} date. Use ISO 8601 format."
            ) from exc

    def build_filters(model):
        filters = []
        if classification:
            filters.append(model.classification == classification)
        if filename:
            filters.append(model.filename.ilike(f"%{filename}%"))
        if date_from:
            filters.append(model.created_at >= parse_iso_date(date_from, "date_from"))
        if date_to:
            filters.append(model.created_at <= parse_iso_date(date_to, "date_to"))
        return filters

    image_select = select(
        AnalysisRecord.id.label("id"),
        AnalysisRecord.filename.label("filename"),
        AnalysisRecord.forensic_score.label("forensic_score"),
        AnalysisRecord.classification.label("classification"),
        AnalysisRecord.created_at.label("created_at"),
        AnalysisRecord.thumbnail_path.label("thumbnail_path"),
        literal("image").label("media_type"),
    ).where(*build_filters(AnalysisRecord))

    video_select = select(
        VideoAnalysisRecord.id.label("id"),
        VideoAnalysisRecord.filename.label("filename"),
        VideoAnalysisRecord.forensic_score.label("forensic_score"),
        VideoAnalysisRecord.classification.label("classification"),
        VideoAnalysisRecord.created_at.label("created_at"),
        VideoAnalysisRecord.thumbnail_path.label("thumbnail_path"),
        literal("video").label("media_type"),
    ).where(*build_filters(VideoAnalysisRecord))

    combined_query = union_all(image_select, video_select).subquery()

    total = db.execute(select(func.count()).select_from(combined_query)).scalar_one()
    offset = (page - 1) * limit
    rows = db.execute(
        select(combined_query)
        .order_by(combined_query.c.created_at.desc())
        .offset(offset)
        .limit(limit)
    ).all()

    paged = [
        {
            "id": row._mapping["id"],
            "filename": row._mapping["filename"],
            "forensic_score": row._mapping["forensic_score"],
            "classification": row._mapping["classification"],
            "created_at": row._mapping["created_at"],
            "thumbnail_url": f"/thumbnails/{Path(row._mapping['thumbnail_path']).name}",
            "media_type": row._mapping["media_type"],
        }
        for row in rows
    ]

    return {
        "data": paged,
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


@app.get("/analysis/video/{record_id}", response_model=VideoAnalysisDetail)
def get_video_analysis(record_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(VideoAnalysisRecord).filter(VideoAnalysisRecord.id == record_id).first()
    )

    if not record:
        raise HTTPException(404, "Record not found")

    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "thumbnail_path": record.thumbnail_path,
        "forensic_score": record.forensic_score,
        "classification": record.classification,
        "frame_count": record.frame_count,
        "frames": record.frames_json,
        "video_metadata": record.video_metadata,
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

    paths: set[Path] = set()
    for value in (
        record.saved_path,
        record.thumbnail_path,
        record.gradcam_heatmap,
        record.ela_heatmap,
        record.forensic_score_json,
        record.ml_prediction,
        record.metadata_anomalies,
        record.metadata_json,
        record.file_integrity,
        record.ai_watermark,
        record.exif_forensics,
        record.c2pa,
        record.jpeg_qtables,
        record.noise_residual,
        record.ela_analysis,
    ):
        _collect_storage_paths(value, paths)

    _cleanup_storage_paths(paths)

    db.delete(record)
    db.commit()

    return {"status": "deleted", "id": record_id}


@app.delete("/analysis/video/{record_id}")
def delete_video_analysis(
    record_id: int,
    admin_key: str = Header(None),
    db: Session = Depends(get_db),
):
    if admin_key != ADMIN_KEY:
        raise HTTPException(403, "Invalid admin key")

    record = (
        db.query(VideoAnalysisRecord).filter(VideoAnalysisRecord.id == record_id).first()
    )

    if not record:
        raise HTTPException(404, "Record not found")

    paths: set[Path] = set()
    for value in (
        record.saved_path,
        record.thumbnail_path,
        record.frames_json,
        record.video_metadata,
    ):
        _collect_storage_paths(value, paths)

    _cleanup_storage_paths(paths)

    db.delete(record)
    db.commit()

    return {"status": "deleted", "id": record_id}


