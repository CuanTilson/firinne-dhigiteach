from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Depends,
    Header,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.requests import Request

from PIL import Image, ImageOps

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from sqlalchemy.orm import Session
from sqlalchemy import select, literal, union_all, func, text
from pathlib import Path
import io
import json
import time
from collections import deque
import uuid
import os
import sys
from importlib import metadata as importlib_metadata
from datetime import datetime, timedelta
import asyncio
from dotenv import load_dotenv

# ---------------------------
# Local Imports (organised)
# ---------------------------

# Storage + DB
from backend.database.db import Base, engine, SessionLocal
from backend.database.models import (
    AnalysisRecord,
    VideoAnalysisRecord,
    AudioAnalysisRecord,
    AuditLog,
    AppSetting,
)
from backend.database.schemas import (
    AnalysisDetail,
    PaginatedAnalysisSummary,
    VideoAnalysisDetail,
    AudioAnalysisDetail,
    PaginatedAuditLog,
    SettingsSnapshot,
    SettingsUpdateRequest,
)

# Upload handling
from backend.analysis.upload import (
    save_uploaded_file,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    AUDIO_EXTENSIONS,
)

# Metadata + forensic tools
from backend.analysis.metadata import (
    extract_image_metadata,
    extract_video_metadata,
    analyse_image_metadata,
    check_camera_model_consistency,
    exif_forensics,
    analyse_file_integrity,
    file_hashes,
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
from backend.analysis.audio import (
    analyse_audio_file,
    extract_audio_from_video,
    resolve_ffmpeg_path,
)

# ML model + explainability
from backend.models.cnndetect_native import CNNDetectionModel
from backend.models.model_a_native import ModelADetector
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
AUDIO_PLOTS_DIR = STORAGE_DIR / "audio_plots"
AUDIO_EXTRACT_DIR = STORAGE_DIR / "audio_extracted"

ELA_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
NOISE_DIR.mkdir(parents=True, exist_ok=True)
JPEG_QUALITY_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

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
MAX_AUDIO_MB = float(os.getenv("FD_MAX_AUDIO_MB", "50"))
MAX_UPLOAD_MB = float(os.getenv("FD_MAX_UPLOAD_MB", "200"))
API_KEY = os.getenv("FD_API_KEY")
RATE_LIMIT_PER_MINUTE = int(os.getenv("FD_RATE_LIMIT_PER_MINUTE", "60"))
RETENTION_DAYS = int(os.getenv("FD_RETENTION_DAYS", "0"))
RETENTION_INTERVAL_HOURS = int(os.getenv("FD_RETENTION_INTERVAL_HOURS", "24"))
PIPELINE_VERSION = os.getenv("FD_PIPELINE_VERSION", "dev")
MODEL_VERSION = os.getenv("FD_MODEL_VERSION", "cnn-blur-jpg-0.5")
DATASET_VERSION = os.getenv("FD_DATASET_VERSION", "unknown")
IMAGE_DETECTOR = os.getenv("FD_IMAGE_DETECTOR", "cnndetection")
MODEL_A_WEIGHTS = Path(
    os.getenv(
        "FD_MODEL_A_WEIGHTS",
        str(PROJECT_ROOT / "artifacts" / "model_a_baseline_gpu" / "model_a_best.pt"),
    )
)
MODEL_A_RUN_MANIFEST = Path(
    os.getenv(
        "FD_MODEL_A_RUN_MANIFEST",
        str(PROJECT_ROOT / "artifacts" / "model_a_baseline_gpu" / "run_manifest.json"),
    )
)
VIDEO_MAX_DURATION_SECONDS = int(os.getenv("FD_MAX_VIDEO_SECONDS", "180"))
VIDEO_SAMPLE_FRAMES = int(os.getenv("FD_VIDEO_MAX_FRAMES", "16"))
FFMPEG_PATH = os.getenv("FD_FFMPEG_PATH", "").strip() or None
AUDIO_CLASSIFICATION_BANDS = {
    "ai_likely_min": 0.7,
    "real_likely_max": 0.3,
}
CLASSIFICATION_BANDS = {
    "ai_likely_min": 0.7,
    "real_likely_max": 0.3,
}
FUSION_WEIGHTS = {
    "ml": 0.50,
    "metadata": 0.10,
    "c2pa": 0.10,
    "ela": 0.10,
    "noise": 0.10,
    "jpeg": 0.05,
    "sd_watermark": 0.02,
}
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
app.mount(
    "/audio_plots",
    StaticFiles(directory=AUDIO_PLOTS_DIR),
    name="audio_plots",
)


# Create database tables
Base.metadata.create_all(bind=engine)


def _ensure_sqlite_column(table_name: str, column_name: str, column_sql: str) -> None:
    with engine.begin() as conn:
        columns = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
        existing = {row[1] for row in columns}
        if column_name not in existing:
            conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}"))


_ensure_sqlite_column("video_analysis_records", "audio_analysis", "JSON")
_ensure_sqlite_column("analysis_records", "applied_settings", "JSON")
_ensure_sqlite_column("video_analysis_records", "applied_settings", "JSON")
_ensure_sqlite_column("audio_analysis_records", "applied_settings", "JSON")

# Load ML model once
if not WEIGHTS.is_file():
    raise RuntimeError(
        f"Missing CNNDetection weights at {WEIGHTS}. See README for download steps."
    )
try:
    MODEL_WEIGHTS_HASHES = file_hashes(WEIGHTS)
except Exception:
    MODEL_WEIGHTS_HASHES = {"sha256": "unknown", "md5": "unknown"}
cnndetector = CNNDetectionModel(weights_path=WEIGHTS)
model_a_detector = (
    ModelADetector(MODEL_A_WEIGHTS, MODEL_A_RUN_MANIFEST)
    if MODEL_A_WEIGHTS.is_file()
    else None
)
MODEL_A_METADATA = (
    model_a_detector.get_metadata()
    if model_a_detector is not None
    else {
        "detector": "model_a",
        "display_name": "Model A",
        "weights_sha256": "missing",
    }
)


# ---------------------------
# Utilities
# ---------------------------


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _get_client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _pkg_version(name: str) -> str:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return "unknown"


def _toolchain_snapshot() -> dict:
    resolved_ffmpeg = resolve_ffmpeg_path(FFMPEG_PATH)
    return {
        "python": sys.version.split()[0],
        "fastapi": _pkg_version("fastapi"),
        "uvicorn": _pkg_version("uvicorn"),
        "numpy": _pkg_version("numpy"),
        "opencv": _pkg_version("opencv-python"),
        "pillow": _pkg_version("Pillow"),
        "reportlab": _pkg_version("reportlab"),
        "ffmpeg_available": bool(resolved_ffmpeg),
        "ffmpeg_resolved_path": resolved_ffmpeg or "",
        "ffmpeg_configured_path": FFMPEG_PATH or "",
        "cnndetection_available": True,
        "model_a_available": model_a_detector is not None,
    }


def _get_detector_registry() -> dict[str, dict]:
    registry = {
        "cnndetection": {
            "name": "cnndetection",
            "display_name": "CNNDetection",
            "available": True,
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "weights": MODEL_WEIGHTS_HASHES,
            "detector": cnndetector,
        },
        "model_a": {
            "name": "model_a",
            "display_name": "Model A",
            "available": model_a_detector is not None,
            "model_version": "model-a-resnet18",
            "dataset_version": "genimage-curated-week3",
            "weights": {
                "sha256": MODEL_A_METADATA.get("weights_sha256", "missing"),
                "md5": "n/a",
            },
            "detector": model_a_detector,
        },
    }
    return registry


def _resolve_image_detector(detector_name: str | None = None) -> dict:
    registry = _get_detector_registry()
    selected = (detector_name or IMAGE_DETECTOR or "cnndetection").strip().lower()
    candidate = registry.get(selected)
    if candidate and candidate["available"]:
        return candidate
    return registry["cnndetection"]


def _default_settings_snapshot() -> dict:
    detector_info = _resolve_image_detector()
    registry = _get_detector_registry()
    return {
        "pipeline": {
            "pipeline_version": PIPELINE_VERSION,
            "model_version": detector_info["model_version"],
            "dataset_version": detector_info["dataset_version"],
            "weights": detector_info["weights"],
            "image_detector": detector_info["name"],
            "available_image_detectors": {
                name: {
                    "display_name": item["display_name"],
                    "available": item["available"],
                    "model_version": item["model_version"],
                }
                for name, item in registry.items()
            },
        },
        "limits": {
            "max_image_mb": MAX_IMAGE_MB,
            "max_video_mb": MAX_VIDEO_MB,
            "max_audio_mb": MAX_AUDIO_MB,
            "max_upload_mb": MAX_UPLOAD_MB,
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
            "retention_days": RETENTION_DAYS,
            "retention_interval_hours": RETENTION_INTERVAL_HOURS,
        },
        "thresholds": {
            "classification_bands": dict(CLASSIFICATION_BANDS),
            "fusion_weights": dict(FUSION_WEIGHTS),
            "video_max_duration_seconds": VIDEO_MAX_DURATION_SECONDS,
            "video_sample_frames": VIDEO_SAMPLE_FRAMES,
            "audio_classification_bands": dict(AUDIO_CLASSIFICATION_BANDS),
            "scene_cut_threshold": 0.6,
            "scene_cut_stride": 10,
        },
        "paths": {
            "ffmpeg_path": FFMPEG_PATH or "",
        },
        "toolchain": _toolchain_snapshot(),
    }


def _merge_settings(base: dict, overrides: dict | None) -> dict:
    merged = json.loads(json.dumps(base))
    if not overrides:
        return merged
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _get_runtime_overrides(db: Session | None) -> dict:
    if db is None:
        return {}
    row = db.query(AppSetting).filter(AppSetting.key == "runtime").first()
    if not row or not isinstance(row.value, dict):
        return {}
    return row.value


def _get_settings_snapshot(db: Session | None = None) -> dict:
    snapshot = _merge_settings(_default_settings_snapshot(), _get_runtime_overrides(db))
    configured_ffmpeg = (
        snapshot.get("paths", {}).get("ffmpeg_path")
        if isinstance(snapshot.get("paths"), dict)
        else None
    )
    resolved_ffmpeg = resolve_ffmpeg_path(configured_ffmpeg)
    snapshot["toolchain"] = {
        **snapshot.get("toolchain", {}),
        "ffmpeg_available": bool(resolved_ffmpeg),
        "ffmpeg_resolved_path": resolved_ffmpeg or "",
        "ffmpeg_configured_path": configured_ffmpeg or "",
    }
    return snapshot


def _analysis_runtime_settings(db: Session | None = None) -> dict:
    snapshot = _get_settings_snapshot(db)
    thresholds = snapshot.get("thresholds", {})
    paths = snapshot.get("paths", {})
    return {
        "image_detector": snapshot.get("pipeline", {}).get("image_detector", IMAGE_DETECTOR),
        "classification_bands": thresholds.get("classification_bands", CLASSIFICATION_BANDS),
        "audio_classification_bands": thresholds.get(
            "audio_classification_bands", AUDIO_CLASSIFICATION_BANDS
        ),
        "fusion_weights": thresholds.get("fusion_weights", FUSION_WEIGHTS),
        "video_max_duration_seconds": int(
            thresholds.get("video_max_duration_seconds", VIDEO_MAX_DURATION_SECONDS)
        ),
        "video_sample_frames": int(thresholds.get("video_sample_frames", VIDEO_SAMPLE_FRAMES)),
        "ffmpeg_path": paths.get("ffmpeg_path") or None,
    }


def _validate_settings_update(payload: SettingsUpdateRequest) -> dict:
    cleaned: dict = {}

    if payload.pipeline:
        pipeline: dict = {}
        image_detector = payload.pipeline.get("image_detector")
        if image_detector is not None:
            selected = str(image_detector).strip().lower()
            registry = _get_detector_registry()
            if selected not in registry:
                raise HTTPException(400, "Invalid image detector selection.")
            if not registry[selected]["available"]:
                raise HTTPException(400, "Selected image detector is unavailable.")
            pipeline["image_detector"] = selected
        if pipeline:
            cleaned["pipeline"] = pipeline

    if payload.thresholds:
        thresholds: dict = {}
        bands = payload.thresholds.get("classification_bands")
        if isinstance(bands, dict):
            ai_min = float(bands.get("ai_likely_min", CLASSIFICATION_BANDS["ai_likely_min"]))
            real_max = float(
                bands.get("real_likely_max", CLASSIFICATION_BANDS["real_likely_max"])
            )
            if not (0 <= real_max <= 1 and 0 <= ai_min <= 1 and real_max <= ai_min):
                raise HTTPException(400, "Invalid image classification bands.")
            thresholds["classification_bands"] = {
                "ai_likely_min": ai_min,
                "real_likely_max": real_max,
            }

        audio_bands = payload.thresholds.get("audio_classification_bands")
        if isinstance(audio_bands, dict):
            ai_min = float(
                audio_bands.get("ai_likely_min", AUDIO_CLASSIFICATION_BANDS["ai_likely_min"])
            )
            real_max = float(
                audio_bands.get("real_likely_max", AUDIO_CLASSIFICATION_BANDS["real_likely_max"])
            )
            if not (0 <= real_max <= 1 and 0 <= ai_min <= 1 and real_max <= ai_min):
                raise HTTPException(400, "Invalid audio classification bands.")
            thresholds["audio_classification_bands"] = {
                "ai_likely_min": ai_min,
                "real_likely_max": real_max,
            }

        if "video_max_duration_seconds" in payload.thresholds:
            duration = int(payload.thresholds["video_max_duration_seconds"])
            if duration <= 0:
                raise HTTPException(400, "Video max duration must be positive.")
            thresholds["video_max_duration_seconds"] = duration

        if "video_sample_frames" in payload.thresholds:
            frames = int(payload.thresholds["video_sample_frames"])
            if frames <= 0:
                raise HTTPException(400, "Video sample frames must be positive.")
            thresholds["video_sample_frames"] = frames

        if thresholds:
            cleaned["thresholds"] = thresholds

    if payload.paths:
        paths: dict = {}
        if "ffmpeg_path" in payload.paths:
            ffmpeg_path = str(payload.paths.get("ffmpeg_path") or "").strip()
            paths["ffmpeg_path"] = ffmpeg_path
        if paths:
            cleaned["paths"] = paths

    return cleaned


def _log_audit(
    action: str,
    record_type: str | None = None,
    record_id: int | None = None,
    filename: str | None = None,
    details: dict | None = None,
    request: Request | None = None,
) -> None:
    actor = _get_client_ip(request) if request else "system"
    db = SessionLocal()
    try:
        entry = AuditLog(
            action=action,
            record_type=record_type,
            record_id=record_id,
            filename=filename,
            actor=actor,
            details=details or {},
        )
        db.add(entry)
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


_rate_limit_buckets: dict[str, deque[float]] = {}


def _rate_limit_allowed(client_id: str) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return True
    now = time.time()
    window = 60.0
    bucket = _rate_limit_buckets.get(client_id)
    if bucket is None:
        bucket = deque()
        _rate_limit_buckets[client_id] = bucket
    while bucket and bucket[0] <= now - window:
        bucket.popleft()
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return False
    bucket.append(now)
    return True


def _validate_admin(request: Request, admin_key: str | None) -> None:
    if admin_key != ADMIN_KEY:
        _log_audit(
            action="admin_action_rejected",
            record_type="settings",
            details={"reason": "invalid_admin_key"},
            request=request,
        )
        raise HTTPException(403, "Invalid admin key")


@app.middleware("http")
async def auth_and_rate_limit(request: Request, call_next):
    path = request.url.path
    public_prefixes = (
        "/docs",
        "/redoc",
        "/uploaded/",
        "/ela/",
        "/heatmaps/",
        "/thumbnails/",
        "/video_frames/",
        "/noise/",
        "/jpeg_quality/",
        "/audio_plots/",
    )
    if path in {"/health", "/openapi.json"} or path.startswith(public_prefixes):
        return await call_next(request)

    if API_KEY:
        provided = request.headers.get("x-api-key")
        if provided != API_KEY:
            return Response(status_code=401, content="Invalid API key.")

    client_id = _get_client_ip(request)
    if not _rate_limit_allowed(client_id):
        return Response(status_code=429, content="Rate limit exceeded.")

    return await call_next(request)


def _mb_to_bytes(value: float) -> int:
    return int(value * 1024 * 1024)


def _max_bytes_for_ext(ext: str) -> int:
    ext = ext.lower()
    if ext in IMAGE_EXTENSIONS:
        return _mb_to_bytes(MAX_IMAGE_MB)
    if ext in VIDEO_EXTENSIONS:
        return _mb_to_bytes(MAX_VIDEO_MB)
    if ext in AUDIO_EXTENSIONS:
        return _mb_to_bytes(MAX_AUDIO_MB)
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


def _delete_analysis_record(record: AnalysisRecord, db: Session, commit: bool = True) -> None:
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
    if commit:
        db.commit()


def _delete_video_record(record: VideoAnalysisRecord, db: Session, commit: bool = True) -> None:
    paths: set[Path] = set()
    for value in (
        record.saved_path,
        record.thumbnail_path,
        record.frames_json,
        record.video_metadata,
        record.audio_analysis,
    ):
        _collect_storage_paths(value, paths)
    _cleanup_storage_paths(paths)
    db.delete(record)
    if commit:
        db.commit()


def _delete_audio_record(record: AudioAnalysisRecord, db: Session, commit: bool = True) -> None:
    paths: set[Path] = set()
    for value in (
        record.saved_path,
        record.waveform_path,
        record.audio_metadata,
        record.audio_features,
        record.file_integrity,
    ):
        _collect_storage_paths(value, paths)
    _cleanup_storage_paths(paths)
    db.delete(record)
    if commit:
        db.commit()


def run_full_analysis(filepath: Path, runtime_settings: dict | None = None) -> dict:
    runtime_settings = runtime_settings or _analysis_runtime_settings()
    file_integrity = analyse_file_integrity(filepath)
    hashes_before = file_integrity.get("hashes") or file_hashes(filepath)

    detector_info = _resolve_image_detector(runtime_settings.get("image_detector"))
    detector = detector_info["detector"]
    ml = detector.predict(filepath)
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
        weights=runtime_settings.get("fusion_weights"),
        classification_bands=runtime_settings.get("classification_bands"),
    )
    fused["provenance"] = {
        "image_detector": detector_info["name"],
        "image_detector_display_name": detector_info["display_name"],
        "model_version": detector_info["model_version"],
        "dataset_version": detector_info["dataset_version"],
        "weights_sha256": detector_info["weights"].get("sha256"),
        "fusion_mode": "rule_based_forensic_fusion",
        "fusion_weights": runtime_settings.get("fusion_weights"),
        "classification_bands": runtime_settings.get("classification_bands"),
    }
    fused["decision_path"] = {
        "production_path": True,
        "decision_engine": "rule_based_forensic_fusion",
        "learned_fusion_applied": False,
        "model_b_status": "offline_evaluation_only",
        "image_detector_role": "runtime_selectable_comparison_detector",
    }

    heatmap_path = HEATMAPS_DIR / f"{uuid.uuid4().hex}_gradcam.png"
    cam = GradCAM(detector.get_model(), detector.get_target_layer())
    cam.generate(
        ml["tensor"],
        filepath,
        heatmap_path,
        target_index=ml.get("gradcam_target_index"),
    )

    hashes_after = file_hashes(filepath)
    file_integrity["hashes_before"] = hashes_before
    file_integrity["hashes_after"] = hashes_after
    file_integrity["hashes_match"] = hashes_before == hashes_after
    file_integrity["hashes"] = hashes_before

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
        "detector_info": {
            "name": detector_info["name"],
            "display_name": detector_info["display_name"],
            "model_version": detector_info["model_version"],
            "dataset_version": detector_info["dataset_version"],
            "weights": detector_info["weights"],
        },
    }


def run_video_analysis(
    video_path: Path,
    max_frames: int = 16,
    runtime_settings: dict | None = None,
) -> dict:
    runtime_settings = runtime_settings or _analysis_runtime_settings()
    frames_dir = VIDEO_FRAMES_DIR / uuid.uuid4().hex
    frames = sample_video_frames(video_path, max_frames=max_frames, output_dir=frames_dir)
    if not frames:
        raise RuntimeError("No frames could be sampled from the video.")

    frame_results = []
    scores = []
    c2pa_flagged = False

    for frame in frames:
        analysis = run_full_analysis(frame["frame_path"], runtime_settings=runtime_settings)
        frame_result = {
            "frame_index": frame["frame_index"],
            "timestamp_sec": frame["timestamp_sec"],
            "saved_path": str(frame["frame_path"]),
            "created_at": datetime.utcnow().isoformat(),
            "file_integrity": analysis["file_integrity"],
            "ml_prediction": {
                "probability": analysis["ml_prob"],
                "label": analysis["ml"]["label"],
                "detector": analysis["detector_info"],
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
            "detector_metadata": analysis["detector_info"],
        }
        frame_results.append(frame_result)
        scores.append(analysis["fused"]["final_score"])
        if analysis["fused"]["classification"] == "ai_generated_c2pa_flagged":
            c2pa_flagged = True

    avg_score = float(sum(scores) / len(scores)) if scores else 0.0
    if c2pa_flagged:
        classification = "ai_generated_c2pa_flagged"
    elif avg_score >= float(runtime_settings["classification_bands"]["ai_likely_min"]):
        classification = "likely_ai_generated"
    elif avg_score <= float(runtime_settings["classification_bands"]["real_likely_max"]):
        classification = "likely_real"
    else:
        classification = "uncertain"

    audio_analysis = {
        "available": False,
        "classification": None,
        "forensic_score": None,
        "audio_metadata": None,
        "audio_features": None,
        "file_integrity": None,
        "waveform_path": None,
        "saved_path": None,
        "error": None,
    }
    extracted_audio_path = AUDIO_EXTRACT_DIR / f"{uuid.uuid4().hex}.wav"
    waveform_path = AUDIO_PLOTS_DIR / f"{uuid.uuid4().hex}_video_audio_waveform.png"
    extraction = extract_audio_from_video(
        video_path,
        extracted_audio_path,
        configured_ffmpeg_path=runtime_settings.get("ffmpeg_path"),
    )
    if extraction["ok"]:
        hashes_before = file_hashes(extracted_audio_path)
        audio_result = analyse_audio_file(extracted_audio_path, waveform_path)
        hashes_after = file_hashes(extracted_audio_path)
        audio_bands = runtime_settings.get("audio_classification_bands", AUDIO_CLASSIFICATION_BANDS)
        audio_score = float(audio_result["forensic_score"])
        if audio_score >= float(audio_bands["ai_likely_min"]):
            audio_classification = "likely_ai_generated"
        elif audio_score <= float(audio_bands["real_likely_max"]):
            audio_classification = "likely_real"
        else:
            audio_classification = "uncertain"
        audio_analysis = {
            "available": True,
            "classification": audio_classification,
            "forensic_score": audio_score,
            "audio_metadata": {
                **audio_result["metadata"],
                "hashes_before": hashes_before,
                "hashes_after": hashes_after,
                "hashes_match": hashes_before == hashes_after,
            },
            "audio_features": {
                **audio_result["features"],
                "findings": audio_result["findings"],
            },
            "file_integrity": {
                "hashes_before": hashes_before,
                "hashes_after": hashes_after,
                "hashes_match": hashes_before == hashes_after,
                "hashes": hashes_before,
            },
            "waveform_path": audio_result["features"].get("waveform_path"),
            "saved_path": str(extracted_audio_path),
            "error": None,
        }
    else:
        extracted_audio_path.unlink(missing_ok=True)
        waveform_path.unlink(missing_ok=True)
        audio_analysis["error"] = extraction["error"]

    return {
        "frames": frame_results,
        "frame_count": len(frame_results),
        "forensic_score": avg_score,
        "classification": classification,
        "audio_analysis": audio_analysis,
    }


_jobs: dict[str, dict] = {}
_jobs_lock = asyncio.Lock()


async def _set_job(job_id: str, **updates) -> None:
    async with _jobs_lock:
        job = _jobs.get(job_id, {"id": job_id})
        job.update(updates)
        _jobs[job_id] = job


async def _run_video_job(job_id: str, saved_path: Path, original_name: str) -> None:
    await _set_job(job_id, status="running", started_at=datetime.utcnow().isoformat())
    try:
        hashes_before = file_hashes(saved_path)
        db = SessionLocal()
        try:
            settings_snapshot = _get_settings_snapshot(db)
            runtime_settings = _analysis_runtime_settings(db)
        finally:
            db.close()
        analysis = await asyncio.to_thread(
            run_video_analysis,
            saved_path,
            max_frames=runtime_settings["video_sample_frames"],
            runtime_settings=runtime_settings,
        )
        video_metadata = extract_video_metadata(saved_path)

        thumbnail_path = THUMB_DIR / f"{uuid.uuid4().hex}_video_thumb.jpg"
        first_frame_path = Path(analysis["frames"][0]["saved_path"])
        with Image.open(first_frame_path) as img:
            img.thumbnail((128, 128))
            img.save(thumbnail_path, "JPEG")

        db = SessionLocal()
        try:
            hashes_after = file_hashes(saved_path)
            video_metadata["hashes_before"] = hashes_before
            video_metadata["hashes_after"] = hashes_after
            video_metadata["hashes_match"] = hashes_before == hashes_after
            video_metadata["hashes"] = hashes_before

            record = VideoAnalysisRecord(
                filename=original_name,
                saved_path=str(saved_path),
                thumbnail_path=str(thumbnail_path),
                forensic_score=analysis["forensic_score"],
                classification=analysis["classification"],
                frame_count=analysis["frame_count"],
                frames_json=analysis["frames"],
                video_metadata=video_metadata,
                audio_analysis=analysis.get("audio_analysis"),
                applied_settings=settings_snapshot,
            )
            db.add(record)
            db.commit()
            db.refresh(record)
        finally:
            db.close()

        result = {
            "id": record.id,
            "filename": record.filename,
            "saved_path": record.saved_path,
            "thumbnail_path": record.thumbnail_path,
            "forensic_score": record.forensic_score,
            "classification": record.classification,
            "frame_count": record.frame_count,
            "frames": record.frames_json,
            "video_metadata": record.video_metadata,
            "audio_analysis": record.audio_analysis,
            "applied_settings": record.applied_settings,
            "created_at": record.created_at,
        }

        await _set_job(
            job_id,
            status="completed",
            finished_at=datetime.utcnow().isoformat(),
            result=result,
        )
        _log_audit(
            action="analysis_video_completed",
            record_type="video",
            record_id=record.id,
            filename=original_name,
            details={
                "classification": record.classification,
                "forensic_score": record.forensic_score,
                "job_id": job_id,
                "pipeline_version": PIPELINE_VERSION,
                "model_version": MODEL_VERSION,
                "weights_sha256": MODEL_WEIGHTS_HASHES.get("sha256"),
                "hashes_sha256": video_metadata.get("hashes", {}).get("sha256"),
                "settings_snapshot": settings_snapshot,
                "toolchain": _toolchain_snapshot(),
            },
        )
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        await _set_job(
            job_id,
            status="failed",
            finished_at=datetime.utcnow().isoformat(),
            error=str(exc),
        )
        _log_audit(
            action="analysis_video_failed",
            record_type="video",
            record_id=None,
            filename=original_name,
            details={"job_id": job_id, "error": str(exc)},
        )


def _purge_old_records(days: int) -> int:
    if days <= 0:
        return 0
    cutoff = datetime.utcnow() - timedelta(days=days)
    db = SessionLocal()
    try:
        image_records = (
            db.query(AnalysisRecord).filter(AnalysisRecord.created_at < cutoff).all()
        )
        video_records = (
            db.query(VideoAnalysisRecord)
            .filter(VideoAnalysisRecord.created_at < cutoff)
            .all()
        )
        for record in image_records:
            _delete_analysis_record(record, db, commit=False)
        for record in video_records:
            _delete_video_record(record, db, commit=False)
        db.commit()
        return len(image_records) + len(video_records)
    finally:
        db.close()


async def _retention_loop() -> None:
    if RETENTION_DAYS <= 0:
        return
    while True:
        await asyncio.to_thread(_purge_old_records, RETENTION_DAYS)
        await asyncio.sleep(max(1, RETENTION_INTERVAL_HOURS) * 3600)


def _safe_text(value) -> str:
    if value is None:
        return "-"
    return str(value)


def _draw_wrapped_text(
    pdf: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    max_width: float,
    font_name: str = "Helvetica",
    font_size: int = 10,
    line_height: float = 14,
) -> float:
    words = text.split()
    if not words:
        return y
    pdf.setFont(font_name, font_size)
    line = words[0]
    for word in words[1:]:
        candidate = f"{line} {word}"
        if pdf.stringWidth(candidate, font_name, font_size) <= max_width:
            line = candidate
        else:
            pdf.drawString(x, y, line)
            y -= line_height
            line = word
    pdf.drawString(x, y, line)
    return y - line_height


def _try_add_image(
    pdf: canvas.Canvas,
    path_value: str | None,
    x: float,
    y: float,
    max_w: float,
    max_h: float,
) -> float:
    if not path_value:
        return y
    path = _coerce_storage_path(path_value)
    if not path or not path.exists():
        return y
    try:
        img = ImageReader(str(path))
        pdf.drawImage(img, x, y - max_h, width=max_w, height=max_h, preserveAspectRatio=True, anchor="c")
        return y - max_h - 10
    except Exception:
        return y


def _ensure_page_space(
    pdf: canvas.Canvas,
    y: float,
    min_space: float,
    page_height: float,
    page_top: float = 40,
) -> float:
    if y < min_space:
        pdf.showPage()
        return page_height - page_top
    return y


def _start_report_page(
    pdf: canvas.Canvas,
    width: float,
    height: float,
    title: str,
    subtitle: str,
) -> float:
    pdf.setFillColor(colors.HexColor("#0f172a"))
    pdf.rect(0, height - 82, width, 82, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(40, height - 38, title)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(40, height - 56, subtitle)
    pdf.setFillColor(colors.black)
    return height - 102


def _draw_page_footer(pdf: canvas.Canvas, width: float) -> None:
    y = 24
    pdf.setStrokeColor(colors.HexColor("#cbd5e1"))
    pdf.line(40, y + 8, width - 40, y + 8)
    pdf.setFont("Helvetica", 8)
    pdf.setFillColor(colors.HexColor("#475569"))
    pdf.drawString(40, y, "Firinne Dhigiteach Forensic Report")
    pdf.drawRightString(width - 40, y, f"Page {pdf.getPageNumber()}")
    pdf.setFillColor(colors.black)


def _draw_section_header(pdf: canvas.Canvas, title: str, y: float, page_width: float) -> float:
    bar_h = 16
    pdf.setFillColor(colors.HexColor("#1d4ed8"))
    pdf.roundRect(40, y - bar_h + 2, page_width - 80, bar_h, 4, stroke=0, fill=1)
    pdf.setFillColor(colors.white)
    pdf.setFont("Helvetica-Bold", 10)
    pdf.drawString(48, y - 9, title)
    pdf.setFillColor(colors.black)
    return y - 20


def _truncate_value(value, max_len: int = 240) -> str:
    text = _safe_text(value)
    if len(text) > max_len:
        return f"{text[:max_len]}..."
    return text


def _draw_key_value_section(
    pdf: canvas.Canvas,
    title: str,
    items: list[tuple[str, str]],
    y: float,
    page_width: float,
    page_height: float,
) -> float:
    y = _ensure_page_space(pdf, y, 110, page_height)
    y = _draw_section_header(pdf, title, y, page_width)
    for key, value in items:
        y = _ensure_page_space(pdf, y, 80, page_height)
        pdf.setFont("Helvetica-Bold", 9)
        pdf.setFillColor(colors.HexColor("#0f172a"))
        pdf.drawString(44, y, f"{key}")
        y = _draw_wrapped_text(
            pdf,
            _safe_text(value),
            180,
            y,
            page_width - 224,
            font_name="Helvetica",
            font_size=9,
            line_height=12,
        )
        pdf.setStrokeColor(colors.HexColor("#e2e8f0"))
        pdf.line(44, y + 5, page_width - 44, y + 5)
        y -= 2
    pdf.setFillColor(colors.black)
    return y - 4


def _draw_bullets(
    pdf: canvas.Canvas,
    title: str,
    values: list[str],
    y: float,
    page_width: float,
    page_height: float,
    limit: int = 10,
) -> float:
    if not values:
        return y
    y = _ensure_page_space(pdf, y, 110, page_height)
    y = _draw_section_header(pdf, title, y, page_width)
    pdf.setFont("Helvetica", 9)
    for value in values[:limit]:
        y = _ensure_page_space(pdf, y, 80, page_height)
        y = _draw_wrapped_text(
            pdf,
            f"- {_truncate_value(value, 320)}",
            50,
            y,
            page_width - 100,
            font_name="Helvetica",
            font_size=9,
            line_height=12,
        )
    return y - 4


def _sanitize_metadata_for_report(value):
    if isinstance(value, dict):
        cleaned = {}
        for k, v in value.items():
            key_text = _safe_text(k)
            if key_text.lower() in {"jpegthumbnail", "thumbnail", "icc_profile"}:
                cleaned[key_text] = "[omitted binary-like field]"
                continue
            cleaned[key_text] = _sanitize_metadata_for_report(v)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_metadata_for_report(v) for v in value[:40]]
    if isinstance(value, str) and len(value) > 500:
        return f"{value[:500]}..."
    return value


def _settings_summary_rows(settings_payload) -> list[tuple[str, str]]:
    if not isinstance(settings_payload, dict):
        return [("Settings snapshot", "Unavailable")]
    pipeline = settings_payload.get("pipeline") if isinstance(settings_payload.get("pipeline"), dict) else {}
    thresholds = settings_payload.get("thresholds") if isinstance(settings_payload.get("thresholds"), dict) else {}
    paths = settings_payload.get("paths") if isinstance(settings_payload.get("paths"), dict) else {}
    image_bands = thresholds.get("classification_bands") if isinstance(thresholds.get("classification_bands"), dict) else {}
    audio_bands = thresholds.get("audio_classification_bands") if isinstance(thresholds.get("audio_classification_bands"), dict) else {}
    return [
        ("Pipeline version", _safe_text(pipeline.get("pipeline_version"))),
        ("Image detector", _safe_text(pipeline.get("image_detector"))),
        ("Model version", _safe_text(pipeline.get("model_version"))),
        ("Dataset version", _safe_text(pipeline.get("dataset_version"))),
        ("Image AI threshold", _safe_text(image_bands.get("ai_likely_min"))),
        ("Image real threshold", _safe_text(image_bands.get("real_likely_max"))),
        ("Audio AI threshold", _safe_text(audio_bands.get("ai_likely_min"))),
        ("Audio real threshold", _safe_text(audio_bands.get("real_likely_max"))),
        ("Video sample frames", _safe_text(thresholds.get("video_sample_frames"))),
        ("FFmpeg path", _truncate_value(paths.get("ffmpeg_path"), 120)),
    ]


def _detector_metadata_from_name(detector_name: str | None) -> dict[str, Any]:
    registry_entry = _get_detector_registry().get(detector_name or "")
    if registry_entry:
        return {
            "name": registry_entry.get("name"),
            "display_name": registry_entry.get("display_name"),
            "model_version": registry_entry.get("model_version"),
            "dataset_version": registry_entry.get("dataset_version"),
            "weights": registry_entry.get("weights"),
        }
    if detector_name == "model_a":
        return {
            "name": "model_a",
            "display_name": "Model A",
            "model_version": "model-a-resnet18",
            "dataset_version": "genimage-curated-week3",
            "weights": {
                "sha256": MODEL_A_METADATA.get("weights_sha256", "unknown"),
                "md5": MODEL_A_METADATA.get("weights_md5", "unknown"),
            },
        }
    if detector_name == "cnndetection":
        return {
            "name": "cnndetection",
            "display_name": "CNNDetection",
            "model_version": MODEL_VERSION,
            "dataset_version": DATASET_VERSION,
            "weights": MODEL_WEIGHTS_HASHES,
        }
    return {}


def _normalized_image_ml_prediction(record: AnalysisRecord) -> dict[str, Any]:
    ml_prediction = dict(record.ml_prediction) if isinstance(record.ml_prediction, dict) else {}
    detector = dict(ml_prediction.get("detector")) if isinstance(ml_prediction.get("detector"), dict) else {}
    applied_settings = record.applied_settings if isinstance(record.applied_settings, dict) else {}
    pipeline_settings = (
        applied_settings.get("pipeline") if isinstance(applied_settings.get("pipeline"), dict) else {}
    )
    inferred_detector = _detector_metadata_from_name(pipeline_settings.get("image_detector"))
    if inferred_detector:
        merged_detector = dict(inferred_detector)
        merged_detector.update({k: v for k, v in detector.items() if v not in (None, "", {})})
        ml_prediction["detector"] = merged_detector
    elif detector:
        ml_prediction["detector"] = detector
    return ml_prediction


def _normalized_image_forensic_score_json(record: AnalysisRecord) -> dict[str, Any]:
    fused = dict(record.forensic_score_json) if isinstance(record.forensic_score_json, dict) else {}
    provenance = dict(fused.get("provenance")) if isinstance(fused.get("provenance"), dict) else {}
    decision_path = dict(fused.get("decision_path")) if isinstance(fused.get("decision_path"), dict) else {}
    ml_prediction = _normalized_image_ml_prediction(record)
    detector = ml_prediction.get("detector") if isinstance(ml_prediction.get("detector"), dict) else {}
    if detector:
        provenance.setdefault("image_detector", detector.get("name"))
        provenance.setdefault("image_detector_display_name", detector.get("display_name"))
        provenance.setdefault("model_version", detector.get("model_version"))
        provenance.setdefault("dataset_version", detector.get("dataset_version"))
        weights = detector.get("weights")
        if isinstance(weights, dict):
            provenance.setdefault("weights_sha256", weights.get("sha256"))
    provenance.setdefault("fusion_mode", "rule_based_forensic_fusion")
    provenance.setdefault("fusion_weights", FUSION_WEIGHTS)
    provenance.setdefault("classification_bands", CLASSIFICATION_BANDS)
    decision_path.setdefault("production_path", True)
    decision_path.setdefault("decision_engine", "rule_based_forensic_fusion")
    decision_path.setdefault("learned_fusion_applied", False)
    decision_path.setdefault("model_b_status", "offline_evaluation_only")
    decision_path.setdefault("image_detector_role", "runtime_selectable_comparison_detector")
    fused["provenance"] = provenance
    fused["decision_path"] = decision_path
    return fused


def _normalized_video_metadata(record: VideoAnalysisRecord) -> dict[str, Any]:
    metadata = dict(record.video_metadata) if isinstance(record.video_metadata, dict) else {}
    detector = dict(metadata.get("image_detector")) if isinstance(metadata.get("image_detector"), dict) else {}
    applied_settings = record.applied_settings if isinstance(record.applied_settings, dict) else {}
    pipeline_settings = (
        applied_settings.get("pipeline") if isinstance(applied_settings.get("pipeline"), dict) else {}
    )
    inferred_detector = _detector_metadata_from_name(pipeline_settings.get("image_detector"))
    if inferred_detector:
        merged_detector = dict(inferred_detector)
        merged_detector.update({k: v for k, v in detector.items() if v not in (None, "", {})})
        metadata["image_detector"] = merged_detector
    elif detector:
        metadata["image_detector"] = detector
    return metadata


def _image_provenance_rows(record: AnalysisRecord) -> list[tuple[str, str]]:
    ml_prediction = _normalized_image_ml_prediction(record)
    detector = ml_prediction.get("detector") if isinstance(ml_prediction.get("detector"), dict) else {}
    fused = _normalized_image_forensic_score_json(record)
    provenance = fused.get("provenance") if isinstance(fused.get("provenance"), dict) else {}
    decision_path = fused.get("decision_path") if isinstance(fused.get("decision_path"), dict) else {}
    return [
        ("Image detector", _safe_text(detector.get("display_name") or detector.get("name"))),
        ("Model version", _safe_text(detector.get("model_version") or provenance.get("model_version"))),
        ("Dataset version", _safe_text(detector.get("dataset_version") or provenance.get("dataset_version"))),
        ("Weights SHA-256", _truncate_value(detector.get("weights", {}).get("sha256") if isinstance(detector.get("weights"), dict) else provenance.get("weights_sha256"), 120)),
        ("Fusion mode", _safe_text(provenance.get("fusion_mode"))),
        ("Decision engine", _safe_text(decision_path.get("decision_engine"))),
        ("Learned fusion applied", _safe_text(decision_path.get("learned_fusion_applied"))),
        ("Model B status", _safe_text(decision_path.get("model_b_status"))),
    ]


def _build_image_report_pdf(record: AnalysisRecord) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = _start_report_page(
        pdf,
        width,
        height,
        "Firinne Dhigiteach - Image Analysis Report",
        f"Case #{record.id} | Generated: {_safe_text(record.created_at)}",
    )

    y = _draw_key_value_section(
        pdf,
        "Case Summary",
        [
            ("Record ID", _safe_text(record.id)),
            ("Filename", _truncate_value(record.filename, 180)),
            ("Created", _safe_text(record.created_at)),
            ("Classification", _safe_text(record.classification)),
            ("Forensic score", _safe_text(round(record.forensic_score or 0.0, 4))),
            ("ML label", _safe_text(record.ml_label)),
            (
                "ML probability",
                _safe_text(round(record.ml_probability or 0.0, 4)),
            ),
        ],
        y,
        width,
        height,
    )

    findings: list[str] = []
    anomaly_score = "-"
    if isinstance(record.metadata_anomalies, dict):
        findings = record.metadata_anomalies.get("findings") or []
        anomaly_score = _safe_text(record.metadata_anomalies.get("anomaly_score"))

    file_integrity = record.file_integrity if isinstance(record.file_integrity, dict) else {}
    hashes_before = file_integrity.get("hashes_before") if isinstance(file_integrity.get("hashes_before"), dict) else {}
    hashes_after = file_integrity.get("hashes_after") if isinstance(file_integrity.get("hashes_after"), dict) else {}
    hashes_fallback = file_integrity.get("hashes") if isinstance(file_integrity.get("hashes"), dict) else {}

    y = _draw_key_value_section(
        pdf,
        "Integrity and Traceability",
        [
            ("SHA-256 (before)", _truncate_value(hashes_before.get("sha256") or hashes_fallback.get("sha256"), 120)),
            ("SHA-256 (after)", _truncate_value(hashes_after.get("sha256") or hashes_fallback.get("sha256"), 120)),
            ("MD5 (before)", _truncate_value(hashes_before.get("md5") or hashes_fallback.get("md5"), 80)),
            ("MD5 (after)", _truncate_value(hashes_after.get("md5") or hashes_fallback.get("md5"), 80)),
            ("Hashes match", _safe_text(file_integrity.get("hashes_match"))),
            (
                "JPEG structure valid",
                _safe_text((file_integrity.get("jpeg_structure") or {}).get("valid_jpeg")),
            ),
            ("Metadata anomaly score", anomaly_score),
        ],
        y,
        width,
        height,
    )

    c2pa = record.c2pa if isinstance(record.c2pa, dict) else {}
    watermark = record.ai_watermark if isinstance(record.ai_watermark, dict) else {}
    jpeg = record.jpeg_qtables if isinstance(record.jpeg_qtables, dict) else {}
    noise = record.noise_residual if isinstance(record.noise_residual, dict) else {}

    y = _draw_key_value_section(
        pdf,
        "Forensic Signals",
        [
            ("C2PA present", _safe_text(c2pa.get("has_c2pa"))),
            ("C2PA signature valid", _safe_text(c2pa.get("signature_valid"))),
            ("C2PA score", _safe_text(c2pa.get("overall_c2pa_score"))),
            ("AI watermark signal", _safe_text(watermark.get("stable_diffusion_detected"))),
            ("Watermark confidence", _safe_text(watermark.get("confidence"))),
            ("JPEG quality estimate", _safe_text(jpeg.get("quality_estimate"))),
            ("JPEG inconsistency score", _safe_text(jpeg.get("inconsistency_score"))),
            ("Noise mean residual", _safe_text(noise.get("mean_residual"))),
        ],
        y,
        width,
        height,
    )

    y = _draw_bullets(pdf, "Metadata Findings (sample)", findings, y, width, height, limit=12)
    y = _draw_bullets(
        pdf,
        "C2PA Assertions (sample)",
        [str(x) for x in (c2pa.get("ai_assertions_found") or [])],
        y,
        width,
        height,
        limit=8,
    )

    y = _draw_key_value_section(
        pdf,
        "Analysis Provenance",
        _image_provenance_rows(record),
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Applied Settings Snapshot",
        _settings_summary_rows(record.applied_settings),
        y,
        width,
        height,
    )

    y = _ensure_page_space(pdf, y, 180, height)
    y = _draw_section_header(pdf, "Key Artifacts", y, width)

    pdf.setFont("Helvetica", 10)
    y = _try_add_image(pdf, record.thumbnail_path, 40, y, 160, 120)
    y = _try_add_image(pdf, record.gradcam_heatmap, 220, height - 200, 160, 120)
    y = _try_add_image(pdf, record.ela_heatmap, 400, height - 200, 160, 120)

    noise_heatmap = None
    if isinstance(record.noise_residual, dict):
        noise_heatmap = record.noise_residual.get("noise_heatmap_path")
    y = _try_add_image(pdf, noise_heatmap, 40, y, 160, 120)

    _draw_page_footer(pdf, width)
    pdf.showPage()
    y = _start_report_page(
        pdf,
        width,
        height,
        "Metadata Appendix (Sanitized)",
        f"Case #{record.id}",
    )
    pdf.setFont("Helvetica", 9)
    metadata_payload = _sanitize_metadata_for_report(record.metadata_json or {})
    metadata_text = json.dumps(metadata_payload, indent=2, ensure_ascii=True)
    for line in metadata_text.splitlines():
        y = _ensure_page_space(pdf, y, 60, height)
        y = _draw_wrapped_text(pdf, line, 40, y, width - 80)

    y = _ensure_page_space(pdf, y, 120, height)
    toolchain = _toolchain_snapshot()
    y = _draw_key_value_section(
        pdf,
        "Toolchain Snapshot",
        [(k, _safe_text(v)) for k, v in toolchain.items()],
        y,
        width,
        height,
    )

    _draw_page_footer(pdf, width)
    pdf.save()
    buffer.seek(0)
    return buffer.read()


def _build_video_report_pdf(record: VideoAnalysisRecord) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = _start_report_page(
        pdf,
        width,
        height,
        "Firinne Dhigiteach - Video Analysis Report",
        f"Case #{record.id} | Generated: {_safe_text(record.created_at)}",
    )

    video_metadata = record.video_metadata if isinstance(record.video_metadata, dict) else {}
    hashes_before = video_metadata.get("hashes_before") if isinstance(video_metadata.get("hashes_before"), dict) else {}
    hashes_after = video_metadata.get("hashes_after") if isinstance(video_metadata.get("hashes_after"), dict) else {}
    hashes_fallback = video_metadata.get("hashes") if isinstance(video_metadata.get("hashes"), dict) else {}

    y = _draw_key_value_section(
        pdf,
        "Case Summary",
        [
            ("Record ID", _safe_text(record.id)),
            ("Filename", _truncate_value(record.filename, 180)),
            ("Created", _safe_text(record.created_at)),
            ("Classification", _safe_text(record.classification)),
            ("Forensic score", _safe_text(round(record.forensic_score or 0.0, 4))),
            ("Frame count", _safe_text(record.frame_count)),
        ],
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Integrity and Traceability",
        [
            ("SHA-256 (before)", _truncate_value(hashes_before.get("sha256") or hashes_fallback.get("sha256"), 120)),
            ("SHA-256 (after)", _truncate_value(hashes_after.get("sha256") or hashes_fallback.get("sha256"), 120)),
            ("MD5 (before)", _truncate_value(hashes_before.get("md5") or hashes_fallback.get("md5"), 80)),
            ("MD5 (after)", _truncate_value(hashes_after.get("md5") or hashes_fallback.get("md5"), 80)),
            ("Hashes match", _safe_text(video_metadata.get("hashes_match"))),
            ("Duration (sec)", _safe_text(video_metadata.get("duration_seconds"))),
            ("FPS", _safe_text(video_metadata.get("fps"))),
            ("Resolution", _safe_text(video_metadata.get("resolution"))),
            ("Codec", _safe_text(video_metadata.get("codec"))),
        ],
        y,
        width,
        height,
    )

    audio_analysis = record.audio_analysis if isinstance(record.audio_analysis, dict) else {}
    if audio_analysis:
        audio_integrity = audio_analysis.get("file_integrity") if isinstance(audio_analysis.get("file_integrity"), dict) else {}
        y = _draw_key_value_section(
            pdf,
            "Extracted Audio Evidence",
            [
                ("Audio available", _safe_text(audio_analysis.get("available"))),
                ("Audio classification", _safe_text(audio_analysis.get("classification"))),
                ("Audio forensic score", _safe_text(audio_analysis.get("forensic_score"))),
                ("Audio hashes match", _safe_text(audio_integrity.get("hashes_match"))),
                ("Audio extraction error", _truncate_value(audio_analysis.get("error"), 120)),
            ],
            y,
            width,
            height,
        )

    image_detector = (
        video_metadata.get("image_detector")
        if isinstance(video_metadata.get("image_detector"), dict)
        else {}
    )
    y = _draw_key_value_section(
        pdf,
        "Analysis Provenance",
        [
            ("Frame detector", _safe_text(image_detector.get("display_name") or image_detector.get("name"))),
            ("Model version", _safe_text(image_detector.get("model_version"))),
            ("Dataset version", _safe_text(image_detector.get("dataset_version"))),
            (
                "Weights SHA-256",
                _truncate_value(
                    (image_detector.get("weights") or {}).get("sha256")
                    if isinstance(image_detector.get("weights"), dict)
                    else "",
                    120,
                ),
            ),
            ("Fusion mode", "rule_based_forensic_fusion"),
            ("Decision engine", "rule_based_forensic_fusion"),
            ("Learned fusion applied", "False"),
            ("Model B status", "offline_evaluation_only"),
        ],
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Applied Settings Snapshot",
        _settings_summary_rows(record.applied_settings),
        y,
        width,
        height,
    )

    y = _ensure_page_space(pdf, y, 180, height)
    y = _draw_section_header(pdf, "Sample Frames", y, width)

    frames = []
    if isinstance(record.frames_json, list):
        frames = record.frames_json[:4]

    x = 40
    for frame in frames:
        path_value = None
        if isinstance(frame, dict):
            path_value = frame.get("saved_path")
        y = _try_add_image(pdf, path_value, x, y, 120, 90)
        x += 140
        if x > width - 140:
            x = 40
            y -= 20

    _draw_page_footer(pdf, width)
    pdf.showPage()
    y = _start_report_page(
        pdf,
        width,
        height,
        "Video Metadata Appendix",
        f"Case #{record.id}",
    )
    pdf.setFont("Helvetica", 9)
    sanitized_video_meta = _sanitize_metadata_for_report(video_metadata)
    video_meta_text = json.dumps(sanitized_video_meta, indent=2, ensure_ascii=True)
    for line in video_meta_text.splitlines():
        y = _ensure_page_space(pdf, y, 60, height)
        y = _draw_wrapped_text(pdf, line, 40, y, width - 80)

    y = _ensure_page_space(pdf, y, 120, height)
    toolchain = _toolchain_snapshot()
    y = _draw_key_value_section(
        pdf,
        "Toolchain Snapshot",
        [(k, _safe_text(v)) for k, v in toolchain.items()],
        y,
        width,
        height,
    )

    _draw_page_footer(pdf, width)
    pdf.save()
    buffer.seek(0)
    return buffer.read()


def _build_audio_report_pdf(record: AudioAnalysisRecord) -> bytes:
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = _start_report_page(
        pdf,
        width,
        height,
        "Firinne Dhigiteach - Audio Analysis Report",
        f"Case #{record.id} | Generated: {_safe_text(record.created_at)}",
    )

    audio_metadata = record.audio_metadata if isinstance(record.audio_metadata, dict) else {}
    audio_features = record.audio_features if isinstance(record.audio_features, dict) else {}
    file_integrity = record.file_integrity if isinstance(record.file_integrity, dict) else {}
    hashes_before = file_integrity.get("hashes_before") if isinstance(file_integrity.get("hashes_before"), dict) else {}
    hashes_after = file_integrity.get("hashes_after") if isinstance(file_integrity.get("hashes_after"), dict) else {}
    hashes_fallback = file_integrity.get("hashes") if isinstance(file_integrity.get("hashes"), dict) else {}

    y = _draw_key_value_section(
        pdf,
        "Case Summary",
        [
            ("Record ID", _safe_text(record.id)),
            ("Filename", _truncate_value(record.filename, 180)),
            ("Created", _safe_text(record.created_at)),
            ("Classification", _safe_text(record.classification)),
            ("Forensic score", _safe_text(round(record.forensic_score or 0.0, 4))),
            ("Analysis mode", _safe_text(audio_features.get("analysis_mode"))),
        ],
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Integrity and Traceability",
        [
            ("SHA-256 (before)", _truncate_value(hashes_before.get("sha256") or hashes_fallback.get("sha256"), 120)),
            ("SHA-256 (after)", _truncate_value(hashes_after.get("sha256") or hashes_fallback.get("sha256"), 120)),
            ("Hashes match", _safe_text(file_integrity.get("hashes_match"))),
            ("Duration (sec)", _safe_text(audio_metadata.get("duration_seconds"))),
            ("Sample rate (Hz)", _safe_text(audio_metadata.get("sample_rate_hz"))),
            ("Channels", _safe_text(audio_metadata.get("channels"))),
            ("Container parse", _safe_text(audio_metadata.get("parse_method"))),
        ],
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Audio Signal Findings",
        [
            ("RMS level", _safe_text(audio_features.get("rms_level"))),
            ("Peak level", _safe_text(audio_features.get("peak_level"))),
            ("Clipping ratio", _safe_text(audio_features.get("clipping_ratio"))),
            ("Silence ratio", _safe_text(audio_features.get("silence_ratio"))),
            ("Spectral centroid (Hz)", _safe_text(audio_features.get("spectral_centroid_hz"))),
        ],
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Applied Settings Snapshot",
        _settings_summary_rows(record.applied_settings),
        y,
        width,
        height,
    )

    y = _draw_key_value_section(
        pdf,
        "Analysis Provenance",
        [
            ("Decision engine", "audio_triage_rule_set"),
            ("Learned fusion applied", "False"),
            ("Model B status", "not_applicable"),
        ],
        y,
        width,
        height,
    )

    findings = []
    if isinstance(audio_features.get("findings"), list):
        findings = [str(x) for x in audio_features.get("findings") or []]
    elif isinstance(audio_metadata.get("notes"), list):
        findings = [str(x) for x in audio_metadata.get("notes") or []]
    y = _draw_bullets(pdf, "Audio Findings", findings, y, width, height, limit=12)

    y = _ensure_page_space(pdf, y, 180, height)
    y = _draw_section_header(pdf, "Waveform Preview", y, width)
    y = _try_add_image(pdf, record.waveform_path, 40, y, 500, 150)

    _draw_page_footer(pdf, width)
    pdf.save()
    buffer.seek(0)
    return buffer.read()


# ---------------------------
# Basic Routes
# ---------------------------


@app.get("/")
def root():
    return {"message": "Fírinne Dhigiteach API is running"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
async def start_retention_task():
    if RETENTION_DAYS > 0:
        asyncio.create_task(_retention_loop())


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
    request: Request,
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

    settings_snapshot = _get_settings_snapshot(db)
    runtime_settings = _analysis_runtime_settings(db)
    analysis = await asyncio.to_thread(run_full_analysis, filepath, runtime_settings)
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
    detector_info = analysis["detector_info"]

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
        ml_prediction={
            "probability": ml_prob,
            "label": ml["label"],
            "detector": detector_info,
        },
        metadata_anomalies=anomaly,
        file_integrity=file_integrity,
        ai_watermark=watermark_info,
        metadata_json=metadata,
        exif_forensics=exif_result,
        c2pa=c2pa_info,
        jpeg_qtables=qtinfo,
        noise_residual=noise_info,
        ela_analysis=ela_info,
        applied_settings=settings_snapshot,
    )

    db.add(record)
    db.commit()
    db.refresh(record)
    created_at = record.created_at

    _log_audit(
        action="analysis_image_completed",
        record_type="image",
        record_id=record.id,
        filename=original_name,
        details={
            "classification": fused["classification"],
            "forensic_score": fused["final_score"],
            "pipeline_version": PIPELINE_VERSION,
            "model_version": detector_info["model_version"],
            "weights_sha256": detector_info["weights"].get("sha256"),
            "image_detector": detector_info["name"],
            "hashes_sha256": file_integrity.get("hashes", {}).get("sha256"),
            "settings_snapshot": settings_snapshot,
            "toolchain": _toolchain_snapshot(),
        },
        request=request,
    )

    # -----------------------
    # Return API response
    # -----------------------
    return {
        "id": record.id,
        "detector": detector_info["display_name"],
        "input_file": original_name,
        "saved_path": str(filepath),
        "created_at": created_at.isoformat(),
        "file_integrity": file_integrity,
        "ml_prediction": {
            "probability": ml_prob,
            "label": ml["label"],
            "detector": detector_info,
        },
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
        "applied_settings": settings_snapshot,
        "forensic_score_json": fused,
        "ela_heatmap": ela_info["ela_image_path"],
        "gradcam_heatmap": str(heatmap_path),
        "camera_consistency": camera_consistency,
        "raw_metadata": metadata,
    }


@app.post("/analysis/audio", response_model=AudioAnalysisDetail)
async def analyse_audio(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    content_type = (file.content_type or "").lower()
    original_name = Path(file.filename or "").name
    if not original_name:
        raise HTTPException(400, "Missing filename.")

    ext = Path(original_name).suffix.lower()
    if ext not in AUDIO_EXTENSIONS and not content_type.startswith("audio/"):
        raise HTTPException(400, "Please upload an audio file.")

    saved_path = save_uploaded_file(file, max_bytes=_max_bytes_for_ext(ext))
    hashes_before = file_hashes(saved_path)
    settings_snapshot = _get_settings_snapshot(db)
    runtime_settings = _analysis_runtime_settings(db)

    waveform_path = AUDIO_PLOTS_DIR / f"{uuid.uuid4().hex}_waveform.png"
    analysis = await asyncio.to_thread(analyse_audio_file, saved_path, waveform_path)
    if analysis["features"].get("waveform_path") is None:
        waveform_path = None

    hashes_after = file_hashes(saved_path)
    file_integrity = {
        "hashes_before": hashes_before,
        "hashes_after": hashes_after,
        "hashes_match": hashes_before == hashes_after,
        "hashes": hashes_before,
    }

    audio_metadata = dict(analysis["metadata"])
    audio_metadata["hashes_before"] = hashes_before
    audio_metadata["hashes_after"] = hashes_after
    audio_metadata["hashes_match"] = hashes_before == hashes_after

    audio_features = dict(analysis["features"])
    audio_features["findings"] = analysis["findings"]

    audio_bands = runtime_settings["audio_classification_bands"]
    audio_score = float(analysis["forensic_score"])
    if audio_score >= float(audio_bands["ai_likely_min"]):
        audio_classification = "likely_ai_generated"
    elif audio_score <= float(audio_bands["real_likely_max"]):
        audio_classification = "likely_real"
    else:
        audio_classification = "uncertain"

    record = AudioAnalysisRecord(
        filename=original_name,
        saved_path=str(saved_path),
        waveform_path=str(waveform_path) if waveform_path else None,
        forensic_score=audio_score,
        classification=audio_classification,
        audio_metadata=audio_metadata,
        audio_features=audio_features,
        file_integrity=file_integrity,
        applied_settings=settings_snapshot,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    _log_audit(
        action="analysis_audio_completed",
        record_type="audio",
        record_id=record.id,
        filename=original_name,
        details={
            "classification": record.classification,
            "forensic_score": record.forensic_score,
            "pipeline_version": PIPELINE_VERSION,
            "analysis_mode": audio_features.get("analysis_mode"),
            "hashes_sha256": hashes_before.get("sha256"),
            "settings_snapshot": settings_snapshot,
            "toolchain": _toolchain_snapshot(),
        },
        request=request,
    )

    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "waveform_path": record.waveform_path,
        "forensic_score": record.forensic_score,
        "classification": record.classification,
        "audio_metadata": record.audio_metadata,
        "audio_features": record.audio_features,
        "file_integrity": record.file_integrity,
        "applied_settings": record.applied_settings,
        "created_at": record.created_at,
    }


@app.post("/analysis/video", response_model=VideoAnalysisDetail)
async def analyse_video(
    request: Request,
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
    runtime_settings = _analysis_runtime_settings(db)
    settings_snapshot = _get_settings_snapshot(db)

    duration_seconds = get_video_duration_seconds(saved_path)
    if duration_seconds <= 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video appears to be invalid or unreadable.")
    if duration_seconds > runtime_settings["video_max_duration_seconds"]:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"Video exceeds {runtime_settings['video_max_duration_seconds']} second length limit.",
        )
    hashes_before = file_hashes(saved_path)
    video_metadata = extract_video_metadata(saved_path)

    analysis = await asyncio.to_thread(
        run_video_analysis,
        saved_path,
        max_frames=runtime_settings["video_sample_frames"],
        runtime_settings=runtime_settings,
    )

    thumbnail_path = THUMB_DIR / f"{uuid.uuid4().hex}_video_thumb.jpg"
    first_frame_path = Path(analysis["frames"][0]["saved_path"])
    with Image.open(first_frame_path) as img:
        img.thumbnail((128, 128))
        img.save(thumbnail_path, "JPEG")

    hashes_after = file_hashes(saved_path)
    video_metadata["hashes_before"] = hashes_before
    video_metadata["hashes_after"] = hashes_after
    video_metadata["hashes_match"] = hashes_before == hashes_after
    video_metadata["hashes"] = hashes_before
    video_metadata["image_detector"] = (
        analysis["frames"][0].get("detector_metadata") if analysis.get("frames") else None
    )

    record = VideoAnalysisRecord(
        filename=original_name,
        saved_path=str(saved_path),
        thumbnail_path=str(thumbnail_path),
        forensic_score=analysis["forensic_score"],
        classification=analysis["classification"],
        frame_count=analysis["frame_count"],
        frames_json=analysis["frames"],
        video_metadata=video_metadata,
        audio_analysis=analysis.get("audio_analysis"),
        applied_settings=settings_snapshot,
    )

    db.add(record)
    db.commit()
    db.refresh(record)

    _log_audit(
        action="analysis_video_completed",
        record_type="video",
        record_id=record.id,
        filename=original_name,
        details={
            "classification": record.classification,
            "forensic_score": record.forensic_score,
            "pipeline_version": PIPELINE_VERSION,
            "model_version": (video_metadata.get("image_detector") or {}).get("model_version"),
            "weights_sha256": ((video_metadata.get("image_detector") or {}).get("weights") or {}).get("sha256"),
            "image_detector": (video_metadata.get("image_detector") or {}).get("name"),
            "hashes_sha256": video_metadata.get("hashes", {}).get("sha256"),
            "settings_snapshot": settings_snapshot,
            "toolchain": _toolchain_snapshot(),
        },
        request=request,
    )

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
        "audio_analysis": record.audio_analysis,
        "applied_settings": record.applied_settings,
        "created_at": record.created_at,
    }


@app.post("/analysis/video/async")
async def analyse_video_async(
    request: Request,
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
    runtime_settings = _analysis_runtime_settings(db)

    duration_seconds = get_video_duration_seconds(saved_path)
    if duration_seconds <= 0:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(400, "Video appears to be invalid or unreadable.")
    if duration_seconds > runtime_settings["video_max_duration_seconds"]:
        saved_path.unlink(missing_ok=True)
        raise HTTPException(
            400,
            f"Video exceeds {runtime_settings['video_max_duration_seconds']} second length limit.",
        )

    job_id = uuid.uuid4().hex
    await _set_job(
        job_id,
        status="queued",
        created_at=datetime.utcnow().isoformat(),
        filename=original_name,
    )
    asyncio.create_task(_run_video_job(job_id, saved_path, original_name))

    _log_audit(
        action="analysis_video_queued",
        record_type="video",
        record_id=None,
        filename=original_name,
        details={
            "job_id": job_id,
            "pipeline_version": PIPELINE_VERSION,
        },
        request=request,
    )

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    async with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/audit", response_model=PaginatedAuditLog)
def list_audit_logs(
    page: int = 1,
    limit: int = 50,
    action: str | None = None,
    record_type: str | None = None,
    db: Session = Depends(get_db),
):
    if limit > 200:
        limit = 200
    if page < 1:
        page = 1

    query = db.query(AuditLog)
    if action:
        query = query.filter(AuditLog.action == action)
    if record_type:
        query = query.filter(AuditLog.record_type == record_type)

    total = query.count()
    rows = (
        query.order_by(AuditLog.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    data = [
        {
            "id": row.id,
            "action": row.action,
            "record_type": row.record_type,
            "record_id": row.record_id,
            "filename": row.filename,
            "actor": row.actor,
            "details": row.details,
            "created_at": row.created_at,
        }
        for row in rows
    ]

    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
    }


@app.get("/settings", response_model=SettingsSnapshot)
def get_settings(db: Session = Depends(get_db)):
    return _get_settings_snapshot(db)


@app.put("/settings", response_model=SettingsSnapshot)
def update_settings(
    payload: SettingsUpdateRequest,
    request: Request,
    admin_key: str = Header(None),
    db: Session = Depends(get_db),
):
    _validate_admin(request, admin_key)
    cleaned = _validate_settings_update(payload)
    current = _get_runtime_overrides(db)
    merged = _merge_settings(current, cleaned)

    row = db.query(AppSetting).filter(AppSetting.key == "runtime").first()
    if row:
        row.value = merged
    else:
        row = AppSetting(key="runtime", value=merged)
        db.add(row)
    db.commit()

    snapshot = _get_settings_snapshot(db)
    _log_audit(
        action="settings_updated",
        record_type="settings",
        record_id=None,
        filename=None,
        details={
            "updated_fields": cleaned,
            "snapshot": snapshot,
        },
        request=request,
    )
    return snapshot


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
    media_type: str | None = None,
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

    selects = []
    if media_type in (None, "", "image"):
        selects.append(
            select(
                AnalysisRecord.id.label("id"),
                AnalysisRecord.filename.label("filename"),
                AnalysisRecord.forensic_score.label("forensic_score"),
                AnalysisRecord.classification.label("classification"),
                AnalysisRecord.created_at.label("created_at"),
                AnalysisRecord.thumbnail_path.label("thumbnail_path"),
                literal("image").label("media_type"),
            ).where(*build_filters(AnalysisRecord))
        )

    if media_type in (None, "", "video"):
        selects.append(
            select(
                VideoAnalysisRecord.id.label("id"),
                VideoAnalysisRecord.filename.label("filename"),
                VideoAnalysisRecord.forensic_score.label("forensic_score"),
                VideoAnalysisRecord.classification.label("classification"),
                VideoAnalysisRecord.created_at.label("created_at"),
                VideoAnalysisRecord.thumbnail_path.label("thumbnail_path"),
                literal("video").label("media_type"),
            ).where(*build_filters(VideoAnalysisRecord))
        )

    if media_type in (None, "", "audio"):
        selects.append(
            select(
                AudioAnalysisRecord.id.label("id"),
                AudioAnalysisRecord.filename.label("filename"),
                AudioAnalysisRecord.forensic_score.label("forensic_score"),
                AudioAnalysisRecord.classification.label("classification"),
                AudioAnalysisRecord.created_at.label("created_at"),
                AudioAnalysisRecord.waveform_path.label("thumbnail_path"),
                literal("audio").label("media_type"),
            ).where(*build_filters(AudioAnalysisRecord))
        )

    if not selects:
        return {
            "data": [],
            "total": 0,
            "page": page,
            "limit": limit,
            "total_pages": 0,
        }

    combined_query = (
        selects[0].subquery() if len(selects) == 1 else union_all(*selects).subquery()
    )

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
            "thumbnail_url": (
                f"/audio_plots/{Path(row._mapping['thumbnail_path']).name}"
                if row._mapping["media_type"] == "audio" and row._mapping["thumbnail_path"]
                else f"/thumbnails/{Path(row._mapping['thumbnail_path']).name}"
                if row._mapping["thumbnail_path"]
                else ""
            ),
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

    normalized_ml_prediction = _normalized_image_ml_prediction(record)
    normalized_fused = _normalized_image_forensic_score_json(record)
    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "ml_probability": record.ml_probability,
        "ml_label": record.ml_label,
        "forensic_score": record.forensic_score,  # float
        "classification": record.classification,  # string
        "forensic_score_json": normalized_fused,  # full JSON
        "gradcam_heatmap": record.gradcam_heatmap,
        "ela_heatmap": record.ela_heatmap,
        "ml_prediction": normalized_ml_prediction,
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
        "applied_settings": record.applied_settings,
        "created_at": record.created_at,
    }


@app.get("/analysis/video/{record_id}", response_model=VideoAnalysisDetail)
def get_video_analysis(record_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(VideoAnalysisRecord).filter(VideoAnalysisRecord.id == record_id).first()
    )

    if not record:
        raise HTTPException(404, "Record not found")

    normalized_video_metadata = _normalized_video_metadata(record)
    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "thumbnail_path": record.thumbnail_path,
        "forensic_score": record.forensic_score,
        "classification": record.classification,
        "frame_count": record.frame_count,
        "frames": record.frames_json,
        "video_metadata": normalized_video_metadata,
        "audio_analysis": record.audio_analysis,
        "applied_settings": record.applied_settings,
        "created_at": record.created_at,
    }


@app.get("/analysis/audio", response_model=PaginatedAnalysisSummary)
def list_audio_analysis(
    page: int = 1,
    limit: int = 20,
    classification: str | None = None,
    filename: str | None = None,
    db: Session = Depends(get_db),
):
    if limit > 200:
        limit = 200
    if page < 1:
        page = 1

    query = db.query(AudioAnalysisRecord)
    if classification:
        query = query.filter(AudioAnalysisRecord.classification == classification)
    if filename:
        query = query.filter(AudioAnalysisRecord.filename.ilike(f"%{filename}%"))

    total = query.count()
    rows = (
        query.order_by(AudioAnalysisRecord.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    data = [
        {
            "id": row.id,
            "filename": row.filename,
            "forensic_score": row.forensic_score,
            "classification": row.classification,
            "created_at": row.created_at,
            "thumbnail_url": f"/audio_plots/{Path(row.waveform_path).name}"
            if row.waveform_path
            else "",
            "media_type": "audio",
        }
        for row in rows
    ]

    return {
        "data": data,
        "total": total,
        "page": page,
        "limit": limit,
        "total_pages": (total + limit - 1) // limit,
    }


@app.get("/analysis/audio/{record_id}", response_model=AudioAnalysisDetail)
def get_audio_analysis(record_id: int, db: Session = Depends(get_db)):
    record = (
        db.query(AudioAnalysisRecord).filter(AudioAnalysisRecord.id == record_id).first()
    )

    if not record:
        raise HTTPException(404, "Record not found")

    return {
        "id": record.id,
        "filename": record.filename,
        "saved_path": record.saved_path,
        "waveform_path": record.waveform_path,
        "forensic_score": record.forensic_score,
        "classification": record.classification,
        "audio_metadata": record.audio_metadata,
        "audio_features": record.audio_features,
        "file_integrity": record.file_integrity,
        "applied_settings": record.applied_settings,
        "created_at": record.created_at,
    }


@app.get("/analysis/{record_id}/report.pdf")
def get_image_report(
    record_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()
    if not record:
        raise HTTPException(404, "Record not found")
    pdf_bytes = _build_image_report_pdf(record)
    filename = f"analysis_{record_id}.pdf"
    _log_audit(
        action="report_generated",
        record_type="image",
        record_id=record.id,
        filename=record.filename,
        details={"format": "pdf"},
        request=request,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/analysis/audio/{record_id}/report.pdf")
def get_audio_report(
    record_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    record = (
        db.query(AudioAnalysisRecord).filter(AudioAnalysisRecord.id == record_id).first()
    )
    if not record:
        raise HTTPException(404, "Record not found")
    pdf_bytes = _build_audio_report_pdf(record)
    filename = f"audio_analysis_{record_id}.pdf"
    _log_audit(
        action="report_generated",
        record_type="audio",
        record_id=record.id,
        filename=record.filename,
        details={"format": "pdf"},
        request=request,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/analysis/video/{record_id}/report.pdf")
def get_video_report(
    record_id: int,
    request: Request,
    db: Session = Depends(get_db),
):
    record = (
        db.query(VideoAnalysisRecord).filter(VideoAnalysisRecord.id == record_id).first()
    )
    if not record:
        raise HTTPException(404, "Record not found")
    pdf_bytes = _build_video_report_pdf(record)
    filename = f"video_analysis_{record_id}.pdf"
    _log_audit(
        action="report_generated",
        record_type="video",
        record_id=record.id,
        filename=record.filename,
        details={"format": "pdf"},
        request=request,
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.delete("/analysis/{record_id}")
def delete_analysis(
    record_id: int,
    request: Request,
    admin_key: str = Header(None),
    db: Session = Depends(get_db),
):
    _validate_admin(request, admin_key)

    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == record_id).first()

    if not record:
        raise HTTPException(404, "Record not found")

    _delete_analysis_record(record, db)

    _log_audit(
        action="record_deleted",
        record_type="image",
        record_id=record_id,
        filename=record.filename,
        details={"admin": True},
        request=request,
    )
    return {"status": "deleted", "id": record_id}


@app.delete("/analysis/audio/{record_id}")
def delete_audio_analysis(
    record_id: int,
    request: Request,
    admin_key: str = Header(None),
    db: Session = Depends(get_db),
):
    _validate_admin(request, admin_key)

    record = (
        db.query(AudioAnalysisRecord).filter(AudioAnalysisRecord.id == record_id).first()
    )

    if not record:
        raise HTTPException(404, "Record not found")

    _delete_audio_record(record, db)

    _log_audit(
        action="record_deleted",
        record_type="audio",
        record_id=record_id,
        filename=record.filename,
        details={"admin": True},
        request=request,
    )
    return {"status": "deleted", "id": record_id}


@app.delete("/analysis/video/{record_id}")
def delete_video_analysis(
    record_id: int,
    request: Request,
    admin_key: str = Header(None),
    db: Session = Depends(get_db),
):
    _validate_admin(request, admin_key)

    record = (
        db.query(VideoAnalysisRecord).filter(VideoAnalysisRecord.id == record_id).first()
    )

    if not record:
        raise HTTPException(404, "Record not found")

    _delete_video_record(record, db)

    _log_audit(
        action="record_deleted",
        record_type="video",
        record_id=record_id,
        filename=record.filename,
        details={"admin": True},
        request=request,
    )
    return {"status": "deleted", "id": record_id}


