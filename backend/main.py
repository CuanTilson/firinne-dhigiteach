from fastapi import FastAPI, UploadFile, File, HTTPException
from backend.analysis.upload import save_uploaded_file
from backend.analysis.metadata_extractor import (
    extract_image_metadata,
    extract_video_metadata,
)
from pathlib import Path

from backend.inference.cnndetect_cli import run_cnndetection
import shutil, uuid

UPLOAD_DIR = Path("backend/uploaded_files")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


app = FastAPI(title="Fírinne Dhigiteach API")


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
    # basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    # save uploaded file
    suffix = Path(file.filename).suffix or ".png"
    filepath = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"

    with filepath.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    # run detector
    result = run_cnndetection(filepath)

    return {
        "detector": "CNNDetection",
        "input_file": file.filename,
        "saved_path": str(filepath),
        "result": result,
    }
