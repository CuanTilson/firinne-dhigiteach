from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.analysis.upload import save_uploaded_file
from backend.analysis.metadata_extractor import (
    extract_image_metadata,
    extract_video_metadata,
)
from backend.analysis.metadata_analyser import analyse_image_metadata
from backend.analysis.forensic_fusion import fuse_forensic_scores
from backend.inference.cnndetect_native import CNNDetectionModel
from backend.explainability.gradcam import GradCAM
from pathlib import Path
import shutil, uuid

UPLOAD_DIR = Path("backend/uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

WEIGHTS = Path("vendor/CNNDetection/weights/blur_jpg_prob0.5.pth")

app = FastAPI(title="Fírinne Dhigiteach API")

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

    # 1. ML prediction
    result = cnndetector.predict(filepath)
    ml_prob = result["probability"]

    # 2. Metadata extraction
    metadata = extract_image_metadata(filepath)

    # 3. Metadata anomaly scoring
    anomaly = analyse_image_metadata(metadata)

    # 4. Fuse into forensic score
    fused = fuse_forensic_scores(ml_prob, anomaly["anomaly_score"])

    # 5. GradCAM
    heatmap_path = (
        Path("backend/generated/heatmaps") / f"{uuid.uuid4().hex}_gradcam.png"
    )

    cam = GradCAM(cnndetector.get_model(), cnndetector.get_target_layer())
    cam.generate(result["tensor"], filepath, heatmap_path)

    return {
        "detector": "CNNDetection + GradCAM + Forensic Fusion",
        "input_file": file.filename,
        "saved_path": str(filepath),
        "ml_prediction": {
            "probability": ml_prob,
            "label": result["label"],
        },
        "metadata_anomalies": anomaly,
        "forensic_score": fused,
        "gradcam_heatmap": str(heatmap_path),
        "raw_metadata": metadata,  # optional
    }
