from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.analysis.upload import save_uploaded_file
from backend.analysis.metadata_extractor import (
    extract_image_metadata,
    extract_video_metadata,
)
from backend.analysis.c2pa_analyser import analyse_c2pa
from backend.analysis.metadata_analyser import analyse_image_metadata
from backend.analysis.forensic_fusion import fuse_forensic_scores
from backend.analysis.exif_forensics import exif_forensics
from backend.analysis.ela import perform_ela
from backend.analysis.jpeg_qtable import analyse_qtables
from backend.analysis.noise_analysis import analyse_noise
from backend.analysis.watermark_sd import detect_sd_watermark
from backend.analysis.file_integrity import analyse_file_integrity
from backend.database.db import Base, engine, SessionLocal
from backend.database.models import AnalysisRecord
from backend.inference.cnndetect_native import CNNDetectionModel
from backend.explainability.gradcam import GradCAM
from pathlib import Path
import shutil, uuid

UPLOAD_DIR = Path("backend/uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = Path("vendor/CNNDetection/weights/blur_jpg_prob0.5.pth")
Path("backend/generated/ela").mkdir(parents=True, exist_ok=True)
Path("backend/generated/heatmaps").mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Fírinne Dhigiteach API")

# Auto-creates the SQLite tables
Base.metadata.create_all(bind=engine)

# load CNNDetection model once
cnndetector = CNNDetectionModel(weights_path=WEIGHTS)


@app.get("/")
def read_root():
    return {"message": "Fírinne Dhigiteach API is running"}


@app.post("/upload")
async def upload_media(file: UploadFile = File(...)):
    saved_path = save_uploaded_file(file)
    return {"status": "success", "filename": file.filename, "saved_to": str(saved_path)}


@app.post("/analyse")
async def analyse_media(file: UploadFile = File(...)):
    saved_path = save_uploaded_file(file)
    ext = Path(file.filename).suffix.lower()

    if ext in [".jpg", ".jpeg", ".png"]:
        metadata = extract_image_metadata(saved_path)
    else:
        metadata = extract_video_metadata(saved_path)

    return {
        "filename": file.filename,
        "path": str(saved_path),
        "metadata": metadata,
    }


@app.post("/detect/image/cnndetection")
async def detect_image_cnndetection(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file.")

    suffix = Path(file.filename).suffix or ".png"
    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with filepath.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    file_integrity = analyse_file_integrity(filepath)

    # 1. ML prediction
    result = cnndetector.predict(filepath)
    ml_prob = result["probability"]

    # 2. Metadata extraction
    metadata = extract_image_metadata(filepath)

    exif_result = exif_forensics(metadata)

    qtinfo = analyse_qtables(filepath)

    noise_info = analyse_noise(filepath)

    watermark_info = detect_sd_watermark(filepath)
    sd_watermark_score = 1.0 if watermark_info["watermark_detected"] else 0.0

    # 3. Metadata anomaly scoring
    anomaly = analyse_image_metadata(metadata)

    # 4. C2PA / Content Credentials analysis
    c2pa_info = analyse_c2pa(filepath)

    # Decide if C2PA should override (Option 1)
    c2pa_ai_flag = c2pa_info["has_c2pa"] and (
        # High confidence from our heuristic
        c2pa_info["overall_c2pa_score"] >= 0.95
        # Explicit AI assertions
        or len(c2pa_info["ai_assertions_found"]) > 0
        # IPTC AI source type (this is the strongest possible AI signal)
        or "iptc:compositeWithTrainedAlgorithmicMedia"
        in c2pa_info.get("digital_source_types", [])
        # Samsung Photo Assist
        or "photo assist" in c2pa_info.get("software_agents", [])
    )

    # ELA analysis
    ela_output_path = Path("backend/generated/ela") / f"{uuid.uuid4().hex}_ela.png"
    ela_info = perform_ela(
        filepath, quality=90, scale_factor=20, save_path=ela_output_path
    )

    # 5. Fuse into forensic score
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

    # 6. GradCAM heatmap
    heatmap_path = (
        Path("backend/generated/heatmaps") / f"{uuid.uuid4().hex}_gradcam.png"
    )

    cam = GradCAM(cnndetector.get_model(), cnndetector.get_target_layer())
    cam.generate(result["tensor"], filepath, heatmap_path)

    # 7. Save to database
    session = SessionLocal()
    record = AnalysisRecord(
        filename=file.filename,
        saved_path=str(filepath),
        ml_probability=ml_prob,
        ml_label=result["label"],
        final_score=fused["final_score"],
        final_classification=fused["classification"],
        gradcam_heatmap=str(heatmap_path),
        ela_heatmap=ela_info["ela_image_path"],
        raw_metadata=metadata,
        exif_forensics=exif_result,
        c2pa=c2pa_info,
        jpeg_qtables=qtinfo,
        noise_residual=noise_info,
        ela_analysis=ela_info,
    )
    session.add(record)
    session.commit()
    session.close()

    return {
        "detector": "CNNDetection + GradCAM + Forensic Fusion + C2PA",
        "input_file": file.filename,
        "saved_path": str(filepath),
        "file_integrity": file_integrity,
        "ml_prediction": {
            "probability": ml_prob,
            "label": result["label"],
        },
        "metadata_anomalies": anomaly,
        "exif_forensics": exif_result,
        # --- UPDATED C2PA OUTPUT ---
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
            # "raw_manifest": c2pa_info["raw_manifest"],  # optional
        },
        "jpeg_qtables": {
            "found": qtinfo["qtables_found"],
            "qtables": qtinfo["qtables"],  # optional
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
            "ela_heatmap": ela_info["ela_image_path"],
        },
        "forensic_score": fused,
        "gradcam_heatmap": str(heatmap_path),
        "raw_metadata": metadata,
    }
